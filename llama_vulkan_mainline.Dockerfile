FROM ubuntu:24.04 AS builder
RUN apt-get update && apt-get install -y \
    git cmake build-essential pkg-config python3 \
    libvulkan-dev glslang-tools glslc \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
WORKDIR /build/llama.cpp

# Patch: accept ssm_dt without .bias suffix (Ollama GGUF compat)
COPY patch_upstream_ssm_dt.py /tmp/patch_upstream_ssm_dt.py
RUN python3 /tmp/patch_upstream_ssm_dt.py

RUN cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=ON \
    -DGGML_NATIVE=ON -DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON \
    -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=ON -DGGML_OPENMP=ON \
    -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=ON \
    && cmake --build build --config Release -j$(nproc)

FROM ubuntu:24.04
RUN apt-get update && apt-get install -y \
    libgomp1 libvulkan1 mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/llama.cpp/build/bin/llama-server /usr/local/bin/
COPY --from=builder /build/llama.cpp/build/bin/llama-bench /usr/local/bin/
COPY --from=builder /build/llama.cpp/build/bin/libllama.so /usr/local/lib/
COPY --from=builder /build/llama.cpp/build/bin/libggml.so /usr/local/lib/
COPY --from=builder /build/llama.cpp/build/bin/libggml-base.so /usr/local/lib/
COPY --from=builder /build/llama.cpp/build/bin/libggml-vulkan.so /usr/local/lib/
COPY --from=builder /build/llama.cpp/build/bin/libggml-cpu.so /usr/local/lib/
COPY --from=builder /build/llama.cpp/build/bin/libmtmd.so /usr/local/lib/
RUN ldconfig
EXPOSE 8080
ENTRYPOINT ["llama-server"]
