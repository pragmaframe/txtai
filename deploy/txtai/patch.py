"""Patch txtai's application.py to pass base_url to FastApiMCP's HTTP client."""
import importlib.util

# Locate the file without importing it (importing triggers onnxruntime/NLTK init)
spec = importlib.util.find_spec("txtai.api.application")
assert spec and spec.origin, "txtai.api.application not found"
path = spec.origin
with open(path) as f:
    src = f.read()

patched = src.replace(
    "FastApiMCP(application, http_client=AsyncClient(timeout=100))",
    'FastApiMCP(application, http_client=AsyncClient(base_url="http://localhost:8000", timeout=100))',
)
assert patched != src, f"Pattern not found in {path} — check txtai version"

with open(path, "w") as f:
    f.write(patched)

print(f"Patched {path}")
