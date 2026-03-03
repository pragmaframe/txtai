FROM neuml/txtai-cpu:latest

# This machine is Intel Atom/Tremont (SSE4.2 only, no AVX/AVX2).
# Several compiled extensions require AVX2 and must be addressed:
#
#  - onnxruntime 1.17+  → downgrade to 1.15.1 (last SSE4.2-compatible release)
#  - numpy 2.x          → onnxruntime 1.15.1 ABI requires numpy<2
#  - hnswlib            → pre-built wheel uses AVX2; rebuild with NO_MANUAL_VECTORIZATION=1
#
# onnxruntime's .so also has PT_GNU_STACK=RWE which newer kernels reject;
# patchelf --clear-execstack removes that flag.

RUN pip install --force-reinstall --no-cache-dir "onnxruntime==1.15.1" "numpy<2"

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential patchelf && \
    NO_MANUAL_VECTORIZATION=1 pip install --force-reinstall --no-cache-dir --no-binary hnswlib hnswlib && \
    find /usr/local/lib/python3.10/site-packages/onnxruntime -name "*.so" \
         -exec patchelf --clear-execstack {} \; && \
    apt-get remove -y build-essential patchelf && \
    apt-get autoremove -y && \
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
