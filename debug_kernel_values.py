import sys

with open('ggml/src/ggml.c', 'r') as f:
    content = f.read()

# Add detailed value dumps for head 0, first token, first call only.
# This prints actual values so we can manually verify the kernel's computation.

old = """            const float g_val    = g_data[g_head_offset + t];
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
            const float decay    = expf(fminf(g_val, 50.0f));"""

new = """            const float g_val    = g_data[g_head_offset + t];
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

            // === DEBUG: dump values for head 0, token 0, first 2 calls ===
            static int _dbg_val = 0;
            if (ith == 0 && head_idx == 0 && t == 0 && _dbg_val < 2) {
                fprintf(stderr, "DN_VAL[c=%d] T=%lld S=%lld H=%lld B=%lld\\n", _dbg_val, (long long)n_tokens, (long long)head_dim, (long long)n_heads, (long long)n_seqs);
                fprintf(stderr, "  g=%.6f beta_raw=%.6f -> beta=%.6f decay=%.6f\\n", g_val, beta_raw, beta_val, decay);
                fprintf(stderr, "  |q|=%.6f |k|=%.6f q_norm_inv=%.6e k_norm_inv=%.6e scale=%.6f\\n",
                    sqrtf(q_norm_sq), sqrtf(k_norm_sq), q_norm_inv, k_norm_inv, scale);
                fprintf(stderr, "  q[0..3]=%.6f %.6f %.6f %.6f\\n", q_t[0], q_t[1], q_t[2], q_t[3]);
                fprintf(stderr, "  k[0..3]=%.6f %.6f %.6f %.6f\\n", k_t[0], k_t[1], k_t[2], k_t[3]);
                fprintf(stderr, "  v[0..3]=%.6f %.6f %.6f %.6f\\n", v_t[0], v_t[1], v_t[2], v_t[3]);
                float sn = 0.0f; for (int64_t i = 0; i < 4; i++) sn += state[i];
                fprintf(stderr, "  state[0..3]=%.6f %.6f %.6f %.6f sum4=%.6f\\n", state[0], state[1], state[2], state[3], sn);
                fflush(stderr);
            }"""

if old in content:
    content = content.replace(old, new, 1)
    print("DEBUG VALUES PATCH A OK")
else:
    print("DEBUG VALUES PATCH A FAIL")
    sys.exit(1)

# Patch B: print output values after computation
old2 = """            for (int64_t col = 0; col < head_dim; ++col) {
                const float k_col = k_t[col] * k_norm_inv;
                for (int64_t row = 0; row < head_dim; ++row) {
                    float s = state[row + col * head_dim];
                    s = decay * s + v_new_buf[row] * k_col;
                    state[row + col * head_dim] = fminf(fmaxf(s, -1e6f), 1e6f);
                }
            }
        }"""

new2 = """            for (int64_t col = 0; col < head_dim; ++col) {
                const float k_col = k_t[col] * k_norm_inv;
                for (int64_t row = 0; row < head_dim; ++row) {
                    float s = state[row + col * head_dim];
                    s = decay * s + v_new_buf[row] * k_col;
                    state[row + col * head_dim] = fminf(fmaxf(s, -1e6f), 1e6f);
                }
            }

            // === DEBUG: dump output for head 0, token 0, first 2 calls ===
            if (ith == 0 && head_idx == 0 && t == 0 && _dbg_val < 2) {
                float * ot = out_data + out_head_offset + t * out_token_stride;
                fprintf(stderr, "  attn_score=%.6e\\n", attn_score);
                fprintf(stderr, "  out[0..3]=%.6e %.6e %.6e %.6e\\n", ot[0], ot[1], ot[2], ot[3]);
                fprintf(stderr, "  v_new[0..3]=%.6e %.6e %.6e %.6e\\n", v_new_buf[0], v_new_buf[1], v_new_buf[2], v_new_buf[3]);
                fprintf(stderr, "  new_state[0..3]=%.6f %.6f %.6f %.6f\\n", state[0], state[1], state[2], state[3]);
                _dbg_val++;
                fflush(stderr);
            }
        }"""

if old2 in content:
    content = content.replace(old2, new2, 1)
    print("DEBUG VALUES PATCH B OK")
else:
    print("DEBUG VALUES PATCH B FAIL")
    sys.exit(1)

with open('ggml/src/ggml.c', 'w') as f:
    f.write(content)

print("All debug value patches applied")
