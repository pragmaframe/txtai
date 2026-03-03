FROM neuml/txtai-cpu:latest

# fastapi_mcp needs base_url on its HTTP client so tool calls resolve correctly.
# Use grep to find the file and sed to patch it — no Python import needed.
RUN file=$(grep -rl \
      'FastApiMCP(application, http_client=AsyncClient(timeout=100))' \
      /usr/local/lib/) \
    && sed -i \
      's|FastApiMCP(application, http_client=AsyncClient(timeout=100))|FastApiMCP(application, http_client=AsyncClient(base_url="http://localhost:8000", timeout=100))|' \
      "$file" \
    && echo "Patched $file"
