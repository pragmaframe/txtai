FROM neuml/txtai-cpu:latest

# onnxruntime's .so has PT_GNU_STACK set to executable (RWE), which newer
# kernels reject with EINVAL.  Clear the execstack bit with patchelf.
RUN apt-get update && \
    apt-get install -y --no-install-recommends patchelf && \
    find /usr/local/lib/python3.10/site-packages/onnxruntime -name "*.so" \
        -exec patchelf --clear-execstack {} \; && \
    apt-get remove -y patchelf && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# fastapi_mcp needs base_url on its HTTP client so tool calls resolve correctly.
# Use grep to find the file and sed to patch it — no Python import needed.
RUN file=$(grep -rl \
      'FastApiMCP(application, http_client=AsyncClient(timeout=100))' \
      /usr/local/lib/) \
    && sed -i \
      's|FastApiMCP(application, http_client=AsyncClient(timeout=100))|FastApiMCP(application, http_client=AsyncClient(base_url="http://localhost:8000", timeout=100))|' \
      "$file" \
    && echo "Patched $file"
