FROM neuml/txtai-cpu:latest

# fastapi_mcp needs server_url so it knows where to call back when executing
# tools — without it, it builds URLs from the bare path (e.g. "/search") which
# httpx rejects with "missing protocol".
COPY patch.py /tmp/patch.py
RUN python3 /tmp/patch.py
