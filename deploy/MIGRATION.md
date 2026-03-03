# Migration: kb-mcp-server → pure txtai 9.7

## Problem

The Claude agent's MCP knowledge-base server runs **kb-mcp-server**, a community wrapper around txtai that exposes semantic search and graph RAG as MCP tools.

**Why this is a problem:**
- kb-mcp-server pins txtai at **8.3.x**; txtai is currently at **9.7.0** — more than a full major version behind.
- The project appears **unmaintained**.
- This means missing security patches, performance improvements, new features, and growing incompatibility with newer Python/ML dependencies.

---

## Plan

Replace kb-mcp-server with a direct deployment of **txtai 9.7**, which natively provides every feature needed and more.

### What txtai 9.7 provides natively

| Capability | How |
|---|---|
| MCP server | `mcp: true` in config.yml — all endpoints exposed at `/mcp` via `fastapi-mcp>=0.2.0` |
| Semantic search | `/search`, `/batchsearch` endpoints; hybrid vector + BM25 keyword scoring |
| Graph RAG | Built-in graph module (NetworkX); topic clustering, path traversal, centrality scoring |
| Full RAG pipeline | `/rag` endpoint (LLM answer generation) — bonus, not available in kb-mcp-server |

### The only gap: OAuth

txtai has built-in SHA-256 bearer token auth but no OAuth/OIDC.
Two options were considered:

**Option A — oauth2-proxy sidecar** *(chosen)*
Add an oauth2-proxy Docker container between nginx and txtai. txtai stays unauthenticated on the internal Docker network; oauth2-proxy handles all OAuth2 token validation.

**Option B — port kb-mcp-server's Google JWT validation in-process**
Implement a txtai custom dependency class (`DEPENDENCIES` env var) that validates Google JWTs. No reverse proxy needed, but couples auth logic to the app layer.

---

## Decisions

### OAuth: Option A (oauth2-proxy sidecar)

**Rationale:**
- The infrastructure already uses OPNSense with the nginx plugin as a TLS-terminating reverse proxy — adding one more proxy hop is trivial.
- txtai stays completely auth-unaware; the auth layer can be changed (or upgraded) independently.
- `--skip-jwt-bearer-tokens=true` makes oauth2-proxy accept MCP bearer tokens from Claude.ai without requiring a browser redirect on every API call.
- No Python code to maintain for auth.

**Request flow:**
```
Claude.ai
  → HTTPS 443 → OPNSense → nginx (TLS termination)
  → HTTP      → oauth2-proxy:4180 (OAuth2 / OIDC validation)
  → HTTP      → txtai:8000 (MCP + search + graph)
```

### Embedding model: `sentence-transformers/all-MiniLM-L6-v2`

384-dimensional, fast, good quality for general-purpose semantic search. The default txtai model. Must be the same on both the build machine and the serving machine — dimension mismatch will prevent loading the archive.

### Hybrid scoring: BM25 + dense vectors

Enabled `scoring: {method: bm25, terms: true}` alongside the vector index. This improves recall for exact-match queries (proper nouns, code identifiers) without hurting semantic results.

### Graph: NetworkX backend with Louvain topic detection

`minscore: 0.15` creates edges between semantically similar nodes. Louvain community detection (`algorithm: louvain`) with 4-term BM25 topic labels gives meaningful cluster names without tuning.

### Build / serve separation

The full index (FAISS vectors + SQLite content + NetworkX graph + BM25 scoring) is serialised to a single `.tar.gz` archive via `embeddings.save()`. The archive is built on a separate machine and transferred via `scp`. txtai auto-extracts it on startup when `path` points to a `.tar.gz` file.

The model weights are **not** bundled — only the model name is stored. The serving machine downloads the model on first start and caches it under `HF_HOME`.

### txtai is not exposed to the host

In `docker-compose.yml`, the `txtai` service has no `ports` binding. Only `oauth2-proxy` binds to `127.0.0.1:4180`. This ensures all traffic passes through the auth layer regardless of nginx configuration.

---

## Files created

| File | Purpose |
|---|---|
| `deploy/config.yml` | txtai API configuration (MCP, graph, BM25, read-only serving mode) |
| `deploy/docker-compose.yml` | txtai + oauth2-proxy services |
| `deploy/.env.example` | Google OAuth2 credentials template |
| `deploy/build.py` | Index builder — text files or CSV → portable `.tar.gz` archive |
| `deploy/nginx-location.conf` | OPNSense nginx location block with SSE/MCP settings |

---

## Verification checklist

1. `docker compose up -d` — both services start cleanly
2. `curl http://localhost:8000/docs` from inside the Docker network — txtai Swagger UI loads
3. `npx @modelcontextprotocol/inspector` → connect to `https://ragmcp.yourdomain.com/mcp` — search, graph, and RAG tools are listed
4. Run a semantic search query via MCP tool call — relevant results returned
5. Run a graph search query — graph context (topic clusters, related nodes) returned
6. Confirm Claude agent can call tools and receive relevant context via the public HTTPS endpoint

---

## Features not ported from kb-mcp-server

| Feature | Decision |
|---|---|
| Causal boost (1.1–1.3× relevance for "why/how" queries) | Not ported — nice-to-have, not core |
| `kb-build` CLI / config templates | Replaced by `build.py` + standard txtai YAML |
| Portable `.tar.gz` archives | Native txtai `embeddings.save()` / `embeddings.load()` |
| Multi-language query patterns | Not ported |
