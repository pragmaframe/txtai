"""
Lightweight MCP OAuth 2.1 proxy with Google IdP.

Implements exactly what Claude.ai's MCP client needs:
  GET  /.well-known/oauth-protected-resource   RFC 9728
  GET  /.well-known/oauth-authorization-server RFC 8414
  POST /register                               RFC 7591 (dynamic client registration)
  GET  /authorize                              auth code + PKCE → Google OIDC redirect
  GET  /callback                               Google callback → issue our own JWT
  POST /token                                  auth code → JWT
  GET  /jwks                                   public key for JWT verification
  ALL  /mcp{path}                              validate JWT → proxy to txtai:8000

Google is used only during the browser login leg.
This proxy issues its own RS256 JWTs — txtai never sees credentials.
"""

import base64
import hashlib
import logging
import os
import secrets
import time
import uuid
from urllib.parse import urlencode

import httpx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
import jwt
from jwt.exceptions import InvalidTokenError
from starlette.background import BackgroundTask

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

GOOGLE_CLIENT_ID = os.environ["GOOGLE_CLIENT_ID"]
GOOGLE_CLIENT_SECRET = os.environ["GOOGLE_CLIENT_SECRET"]
MCP_DOMAIN = os.environ["MCP_DOMAIN"]
# ALLOWED_EMAIL: exact address ("user@example.com") or domain only ("example.com")
# Leave empty to allow any authenticated Google account.
ALLOWED_EMAIL = os.environ.get("ALLOWED_EMAIL", "")

BASE_URL = f"https://{MCP_DOMAIN}"
TXTAI_URL = os.environ.get("TXTAI_URL", "http://txtai:8000")
TOKEN_TTL = int(os.environ.get("TOKEN_TTL", "3600"))  # seconds

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"

# ── RSA keypair — generated fresh each startup (ephemeral is fine: tokens are
#    short-lived and Claude re-registers on each session per the MCP spec) ─────

_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_public_key = _private_key.public_key()

_private_pem = _private_key.private_bytes(
    serialization.Encoding.PEM,
    serialization.PrivateFormat.PKCS8,
    serialization.NoEncryption(),
).decode()

_public_pem = _public_key.public_bytes(
    serialization.Encoding.PEM,
    serialization.PublicFormat.SubjectPublicKeyInfo,
).decode()

KEY_ID = secrets.token_hex(8)

# Build JWKS from public key numbers
_pub_numbers = _public_key.public_numbers()


def _b64url_uint(n: int) -> str:
    length = (n.bit_length() + 7) // 8
    return base64.urlsafe_b64encode(n.to_bytes(length, "big")).rstrip(b"=").decode()


JWKS = {
    "keys": [
        {
            "kty": "RSA",
            "use": "sig",
            "alg": "RS256",
            "kid": KEY_ID,
            "n": _b64url_uint(_pub_numbers.n),
            "e": _b64url_uint(_pub_numbers.e),
        }
    ]
}

# ── In-memory stores ──────────────────────────────────────────────────────────
# Client registrations: ephemeral, Claude re-registers on each session.
clients: dict[str, dict] = {}

# Auth codes: short-lived (5 min), consumed once at /token.
auth_codes: dict[str, dict] = {}

# PKCE state: maps our random state → context for the Google roundtrip.
pkce_state: dict[str, dict] = {}

# ── Persistent HTTP client for proxying ───────────────────────────────────────
_proxy_client = httpx.AsyncClient(timeout=None)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(docs_url=None, redoc_url=None, title="MCP OAuth Proxy")


def _oauth_error(error: str, description: str, status: int = 400) -> JSONResponse:
    return JSONResponse(
        {"error": error, "error_description": description}, status_code=status
    )


# ── RFC 9728 — OAuth Protected Resource metadata ─────────────────────────────

@app.get("/.well-known/oauth-protected-resource")
async def protected_resource():
    """Tell Claude where to find our Authorization Server."""
    return {
        "resource": BASE_URL,
        "authorization_servers": [BASE_URL],
    }


# ── RFC 8414 — Authorization Server metadata ─────────────────────────────────

@app.get("/.well-known/oauth-authorization-server")
async def as_metadata():
    """Authorization Server discovery document."""
    return {
        "issuer": BASE_URL,
        "authorization_endpoint": f"{BASE_URL}/authorize",
        "token_endpoint": f"{BASE_URL}/token",
        "jwks_uri": f"{BASE_URL}/jwks",
        "registration_endpoint": f"{BASE_URL}/register",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": [
            "client_secret_post",
            "client_secret_basic",
            "none",
        ],
        "scopes_supported": ["openid", "email", "profile"],
    }


# ── JWKS ──────────────────────────────────────────────────────────────────────

@app.get("/jwks")
async def jwks():
    """Public key set for verifying our JWTs."""
    return JWKS


# ── RFC 7591 — Dynamic Client Registration ───────────────────────────────────

@app.post("/register")
async def register(request: Request):
    """
    Accept any client registration.
    MCP clients re-register on every session — no persistence needed.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    client_id = str(uuid.uuid4())
    client_secret = secrets.token_urlsafe(32)
    redirect_uris = body.get("redirect_uris", [])

    clients[client_id] = {
        "secret": client_secret,
        "redirect_uris": redirect_uris,
    }

    log.info("Registered client %s redirect_uris=%s", client_id, redirect_uris)

    return JSONResponse(
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "client_id_issued_at": int(time.time()),
            "client_secret_expires_at": 0,  # never
            "redirect_uris": redirect_uris,
            "grant_types": ["authorization_code"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "client_secret_post",
        },
        status_code=201,
    )


# ── Authorization endpoint ────────────────────────────────────────────────────

@app.get("/authorize")
async def authorize(request: Request):
    """Validate the auth request, then redirect to Google OIDC."""
    params = dict(request.query_params)

    client_id = params.get("client_id", "")
    redirect_uri = params.get("redirect_uri", "")
    response_type = params.get("response_type", "")
    state = params.get("state", "")
    code_challenge = params.get("code_challenge", "")
    code_challenge_method = params.get("code_challenge_method", "S256")

    if client_id not in clients:
        return _oauth_error("invalid_client", "Unknown client_id")
    if response_type != "code":
        return _oauth_error("unsupported_response_type", "Only code supported")

    # Map our random state → the client's original PKCE + redirect context
    our_state = secrets.token_urlsafe(32)
    pkce_state[our_state] = {
        "code_challenge": code_challenge,
        "code_challenge_method": code_challenge_method,
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "original_state": state,
    }

    google_params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": f"{BASE_URL}/callback",
        "response_type": "code",
        "scope": "openid email profile",
        "state": our_state,
        "access_type": "online",
        "prompt": "select_account",
    }

    google_url = f"{GOOGLE_AUTH_URL}?{urlencode(google_params)}"
    log.info("Redirecting client %s to Google", client_id)
    return RedirectResponse(google_url, status_code=302)


# ── Google callback ───────────────────────────────────────────────────────────

@app.get("/callback")
async def callback(request: Request):
    """
    Google redirects here after login.
    Validates the Google ID token, checks allowed email, issues our auth code.
    """
    params = dict(request.query_params)
    error = params.get("error", "")
    code = params.get("code", "")
    state = params.get("state", "")

    if error:
        raise HTTPException(400, f"Google returned error: {error}")
    if not code:
        raise HTTPException(400, "Missing code from Google")
    if state not in pkce_state:
        raise HTTPException(400, "Invalid or expired state")

    ctx = pkce_state.pop(state)

    # Exchange Google auth code for tokens
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": f"{BASE_URL}/callback",
                "grant_type": "authorization_code",
            },
        )

    if resp.status_code != 200:
        log.error("Google token exchange failed: %s", resp.text)
        raise HTTPException(502, "Google token exchange failed")

    token_data = resp.json()
    id_token = token_data.get("id_token")
    if not id_token:
        raise HTTPException(502, "No id_token in Google response")

    # Extract claims from the Google ID token without signature verification.
    # We trust this token arrived via our own redirect (state-bound), so skipping
    # re-verification against Google's JWKS is acceptable here.
    try:
        claims = jwt.decode(id_token, options={"verify_signature": False})
    except InvalidTokenError as exc:
        raise HTTPException(502, f"Failed to decode Google ID token: {exc}") from exc

    email = claims.get("email", "")
    if not email:
        raise HTTPException(502, "No email in Google ID token")

    # Enforce email allowlist
    if ALLOWED_EMAIL:
        if "@" in ALLOWED_EMAIL:
            # Exact address match
            if email.lower() != ALLOWED_EMAIL.lower():
                log.warning("Rejected login from %s (not in allowlist)", email)
                raise HTTPException(403, f"Email {email!r} is not allowed")
        else:
            # Domain-only match (ALLOWED_EMAIL = "example.com")
            if not email.lower().endswith(f"@{ALLOWED_EMAIL.lower()}"):
                log.warning("Rejected login from %s (domain not allowed)", email)
                raise HTTPException(403, f"Email domain not allowed")

    # Issue our auth code (5-minute TTL, one-time use)
    our_code = secrets.token_urlsafe(32)
    auth_codes[our_code] = {
        "client_id": ctx["client_id"],
        "redirect_uri": ctx["redirect_uri"],
        "email": email,
        "code_challenge": ctx["code_challenge"],
        "code_challenge_method": ctx["code_challenge_method"],
        "expiry": time.time() + 300,
    }

    log.info("Issued auth code for %s → client %s", email, ctx["client_id"])

    redirect_params: dict[str, str] = {"code": our_code}
    if ctx["original_state"]:
        redirect_params["state"] = ctx["original_state"]

    redirect_url = f"{ctx['redirect_uri']}?{urlencode(redirect_params)}"
    return RedirectResponse(redirect_url, status_code=302)


# ── Token endpoint ────────────────────────────────────────────────────────────

@app.post("/token")
async def token(request: Request):
    """Exchange auth code for our signed JWT."""
    form = await request.form()
    grant_type = form.get("grant_type", "")
    code = str(form.get("code", ""))
    redirect_uri = str(form.get("redirect_uri", ""))
    code_verifier = str(form.get("code_verifier", ""))

    # Client authentication: form fields take priority, then Basic auth header
    client_id = str(form.get("client_id", ""))
    client_secret = str(form.get("client_secret", ""))

    if not client_id:
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("basic "):
            try:
                decoded = base64.b64decode(auth_header[6:]).decode()
                client_id, _, client_secret = decoded.partition(":")
            except Exception:
                return _oauth_error("invalid_client", "Malformed Basic auth")

    if grant_type != "authorization_code":
        return _oauth_error("unsupported_grant_type", "Only authorization_code supported")

    if code not in auth_codes:
        return _oauth_error("invalid_grant", "Invalid or expired code")

    ctx = auth_codes.pop(code)  # consume immediately

    if time.time() > ctx["expiry"]:
        return _oauth_error("invalid_grant", "Code has expired")

    if ctx["client_id"] != client_id:
        return _oauth_error("invalid_client", "client_id mismatch")

    # PKCE verification (required when code_challenge was provided)
    if ctx["code_challenge"]:
        if not code_verifier:
            return _oauth_error("invalid_grant", "code_verifier required")
        digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
        computed = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        if computed != ctx["code_challenge"]:
            return _oauth_error("invalid_grant", "PKCE verification failed")

    # Issue our RS256 JWT
    now = int(time.time())
    payload = {
        "iss": BASE_URL,
        "sub": ctx["email"],
        "email": ctx["email"],
        "iat": now,
        "exp": now + TOKEN_TTL,
        "client_id": client_id,
        "jti": secrets.token_hex(16),
    }

    access_token = jwt.encode(
        payload, _private_pem, algorithm="RS256", headers={"kid": KEY_ID, "alg": "RS256"}
    )

    log.info("Issued JWT for %s (client %s)", ctx["email"], client_id)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": TOKEN_TTL,
    }


# ── MCP proxy — JWT validation + transparent forwarding ──────────────────────

_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)

_SKIP_REQUEST_HEADERS = _HOP_BY_HOP | {"host", "authorization", "content-length"}
_SKIP_RESPONSE_HEADERS = _HOP_BY_HOP | {"content-length"}


def _validate_jwt(authorization: str) -> dict:
    """Raise HTTPException if the Bearer JWT is missing or invalid."""
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(
            401,
            "Missing Bearer token",
            headers={"WWW-Authenticate": f'Bearer realm="{BASE_URL}"'},
        )
    token_str = authorization[7:]
    try:
        return jwt.decode(
            token_str,
            _public_pem,
            algorithms=["RS256"],
            options={"verify_aud": False},
        )
    except InvalidTokenError as exc:
        raise HTTPException(
            401,
            f"Invalid token: {exc}",
            headers={"WWW-Authenticate": f'Bearer realm="{BASE_URL}", error="invalid_token"'},
        ) from exc


@app.api_route(
    "/mcp{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def mcp_proxy(request: Request, path: str = ""):
    """Validate JWT then stream the request/response to/from txtai."""
    _validate_jwt(request.headers.get("authorization", ""))

    # Build upstream URL
    upstream_url = f"{TXTAI_URL}/mcp{path}"
    if request.url.query:
        upstream_url += f"?{request.url.query}"

    # Forward headers (strip auth + hop-by-hop)
    fwd_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in _SKIP_REQUEST_HEADERS
    }

    body = await request.body()

    upstream_req = _proxy_client.build_request(
        request.method,
        upstream_url,
        content=body,
        headers=fwd_headers,
    )
    upstream_resp = await _proxy_client.send(upstream_req, stream=True)

    resp_headers = {
        k: v
        for k, v in upstream_resp.headers.items()
        if k.lower() not in _SKIP_RESPONSE_HEADERS
    }

    return StreamingResponse(
        upstream_resp.aiter_bytes(),
        status_code=upstream_resp.status_code,
        headers=resp_headers,
        background=BackgroundTask(upstream_resp.aclose),
    )


# ── Startup / shutdown ────────────────────────────────────────────────────────

@app.on_event("shutdown")
async def _shutdown():
    await _proxy_client.aclose()
