FROM neuml/txtai-cpu:latest

# This machine is Intel Atom/Tremont (SSE4.2 only, no AVX/AVX2).
# Two onnxruntime issues need fixing:
#
#  1. AVX2: onnxruntime 1.17+ emits AVX2 instructions at runtime → SIGILL.
#     Downgrade to 1.15.1, the last release that runs on SSE4.2-only CPUs.
#
#  2. execstack: onnxruntime's .so has PT_GNU_STACK=RWE; newer kernels reject
#     loading it with EINVAL.  Clear the bit with patchelf.
RUN pip install --force-reinstall --no-cache-dir "onnxruntime==1.15.1" "numpy<2"

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
