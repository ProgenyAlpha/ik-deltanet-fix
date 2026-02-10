import sys

with open('ggml/src/ggml.c', 'r') as f:
    content = f.read()

# Remove L2 normalization from the fused delta_net kernel.
# The chunked path doesn't normalize q/k, and it produces correct output.
# The kernel should match: use raw q, k values without normalization.
# Also remove the scale factor — chunked path doesn't scale either.

# Replace the normalization + per-element usage with raw values
old = r"""            const float g_val    = g_data[g_head_offset + t];
            const float beta_raw = beta_data[g_head_offset + t];

            float q_norm_sq = 0.0f;
            float k_norm_sq = 0.0f;
            for (int64_t i = 0; i < head_dim; ++i) {
                q_norm_sq += q_t[i] * q_t[i];
                k_norm_sq += k_t[i] * k_t[i];
            }
            const float q_norm_inv = 1.0f / sqrtf(q_norm_sq + eps);
            const float k_norm_inv = 1.0f / sqrtf(k_norm_sq + eps);

            const float beta_val = 1.0f / (1.0f + expf(-beta_raw));
            const float decay    = expf(fminf(g_val, 50.0f));

            float attn_score = 0.0f;
            for (int64_t i = 0; i < head_dim; ++i) {
                attn_score += (k_t[i] * k_norm_inv) * (q_t[i] * q_norm_inv * scale);
            }

            float * out_t = out_data + out_head_offset + t * out_token_stride;

            for (int64_t row = 0; row < head_dim; ++row) {
                float v_prime = 0.0f;
                float out_val = 0.0f;

                for (int64_t col = 0; col < head_dim; ++col) {
                    const float k_col = k_t[col] * k_norm_inv;
                    const float q_col = q_t[col] * q_norm_inv * scale;
                    const float s = state[row + col * head_dim];

                    v_prime += s * k_col * beta_val * decay;
                    out_val += s * q_col * decay;
                }

                const float v_new = v_t[row] * beta_val - v_prime;
                v_new_buf[row] = v_new;
                out_t[row] = out_val + v_new * attn_score;
            }

            for (int64_t col = 0; col < head_dim; ++col) {
                const float k_col = k_t[col] * k_norm_inv;
                for (int64_t row = 0; row < head_dim; ++row) {
                    float s = state[row + col * head_dim];
                    s = decay * s + v_new_buf[row] * k_col;
                    state[row + col * head_dim] = fminf(fmaxf(s, -1e6f), 1e6f);
                }
            }"""

new = r"""            const float g_val    = g_data[g_head_offset + t];
            const float beta_raw = beta_data[g_head_offset + t];

            const float beta_val = 1.0f / (1.0f + expf(-beta_raw));
            const float decay    = expf(fminf(g_val, 50.0f));

            // attn_score = k^T @ q (raw, no normalization, no scale)
            float attn_score = 0.0f;
            for (int64_t i = 0; i < head_dim; ++i) {
                attn_score += k_t[i] * q_t[i];
            }

            float * out_t = out_data + out_head_offset + t * out_token_stride;

            for (int64_t row = 0; row < head_dim; ++row) {
                float v_prime = 0.0f;
                float out_val = 0.0f;

                for (int64_t col = 0; col < head_dim; ++col) {
                    const float k_col = k_t[col];
                    const float q_col = q_t[col];
                    const float s = state[row + col * head_dim];

                    v_prime += s * k_col * beta_val * decay;
                    out_val += s * q_col * decay;
                }

                const float v_new = v_t[row] * beta_val - v_prime;
                v_new_buf[row] = v_new;
                out_t[row] = out_val + v_new * attn_score;
            }

            for (int64_t col = 0; col < head_dim; ++col) {
                const float k_col = k_t[col];
                for (int64_t row = 0; row < head_dim; ++row) {
                    float s = state[row + col * head_dim];
                    s = decay * s + v_new_buf[row] * k_col;
                    state[row + col * head_dim] = fminf(fmaxf(s, -1e6f), 1e6f);
                }
            }"""

if old in content:
    content = content.replace(old, new, 1)
    print("KERNEL NO-NORM PATCH OK")
else:
    print("KERNEL NO-NORM PATCH FAIL")
    sys.exit(1)

with open('ggml/src/ggml.c', 'w') as f:
    f.write(content)
