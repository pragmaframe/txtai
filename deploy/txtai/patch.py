"""Patch txtai's application.py to pass server_url to FastApiMCP."""
import inspect
import txtai.api.application as m

path = inspect.getfile(m)
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
