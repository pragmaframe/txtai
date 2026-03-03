FROM neuml/txtai-cpu:latest

# onnxruntime 1.17+ requires AVX. This machine has SSE4.2 but no AVX/AVX2
# (Intel Atom/Tremont). Downgrade to the last AVX-free release.
RUN pip install --force-reinstall --no-cache-dir "onnxruntime==1.15.1"

# fastapi_mcp needs base_url on its HTTP client so tool calls resolve correctly.
# Use grep to find the file and sed to patch it — no Python import needed.
RUN file=$(grep -rl \
      'FastApiMCP(application, http_client=AsyncClient(timeout=100))' \
      /usr/local/lib/) \
    && sed -i \
      's|FastApiMCP(application, http_client=AsyncClient(timeout=100))|FastApiMCP(application, http_client=AsyncClient(base_url="http://localhost:8000", timeout=100))|' \
      "$file" \
    && echo "Patched $file"
