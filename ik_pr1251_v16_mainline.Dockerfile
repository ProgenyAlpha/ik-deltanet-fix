FROM ubuntu:24.04 AS builder
RUN apt-get update && apt-get install -y git cmake build-essential pkg-config python3 && rm -rf /var/lib/apt/lists/*
WORKDIR /build
RUN git clone --depth 1 https://github.com/YurkoHoshko/ik_llama.cpp.git
WORKDIR /build/ik_llama.cpp

# Patch 1: accept ssm_dt without .bias suffix (Ollama GGUF compat)
RUN sed -i 's|tn(LLM_TENSOR_SSM_DT,         "bias",   i), {hparams.ssm_dt_rank}|tn(LLM_TENSOR_SSM_DT,                    i), {hparams.ssm_dt_rank}|' src/llama-load-tensors.cpp

# Patch 2: per-layer n_embd_k/v_gqa in tensor loader
RUN sed -i '/\/\/ Full-attention layer/a\            const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(i);\n            const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa(i);' src/llama-load-tensors.cpp

# Patch 3: per-layer n_head_kv in build_layer_attn lambda
RUN sed -i '/build_layer_attn = \[&\]/a\        const int64_t n_head_kv = hparams.n_head_kv(il);' src/llama-build-context.cpp

# Patch 4: Fix g permutation
RUN sed -i 's|ggml_permute(ctx0, g, 2, 0, 3, 1)|ggml_permute(ctx0, g, 1, 0, 2, 3)|' src/llama-build-context.cpp

# Patch 5: MAINLINE TRANSPLANT — replace Codex's delta-net with upstream llama.cpp implementation
COPY patch_mainline_transplant.py /tmp/patch_mainline_transplant.py
RUN python3 /tmp/patch_mainline_transplant.py

RUN cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=ON -DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON \
    -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=ON -DGGML_OPENMP=ON \
    -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=ON \
    && cmake --build build --config Release -j$(nproc)

FROM ubuntu:24.04
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/ik_llama.cpp/build/bin/llama-server /usr/local/bin/
COPY --from=builder /build/ik_llama.cpp/build/bin/llama-bench /usr/local/bin/
COPY --from=builder /build/ik_llama.cpp/build/src/libllama.so /usr/local/lib/
COPY --from=builder /build/ik_llama.cpp/build/ggml/src/libggml.so /usr/local/lib/
COPY --from=builder /build/ik_llama.cpp/build/examples/mtmd/libmtmd.so /usr/local/lib/
RUN ldconfig
EXPOSE 8080
ENTRYPOINT ["llama-server"]
