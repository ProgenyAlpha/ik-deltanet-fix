import sys

with open('ggml/src/ggml.c', 'r') as f:
    content = f.read()

# Patch: Instrument the row loop to trace EVERY intermediate value for head 0, t=0, row=0.
# This will tell us exactly where the computation diverges.

old = """            float * out_t = out_data + out_head_offset + t * out_token_stride;

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
            }
        }"""

new = r"""            float * out_t = out_data + out_head_offset + t * out_token_stride;

            // === DEBUG: trace for head 0, token 0 ===
            static int _dbg_deep = 0;
            const int _do_dbg = (head_idx == 0 && t == 0 && _dbg_deep < 2);

            if (_do_dbg) {
                fprintf(stderr, "DN_DEEP[c=%d] T=%lld S=%lld H=%lld B=%lld ith=%d nth=%d\n",
                    _dbg_deep, (long long)n_tokens, (long long)head_dim, (long long)n_heads, (long long)n_seqs, ith, nth);
                fprintf(stderr, "  g=%.8f beta_raw=%.8f -> beta=%.8f decay=%.8f scale=%.8f\n",
                    (double)g_val, (double)beta_raw, (double)beta_val, (double)decay, (double)scale);
                fprintf(stderr, "  |q|=%.8f |k|=%.8f q_norm_inv=%.8e k_norm_inv=%.8e\n",
                    (double)sqrtf(q_norm_sq), (double)sqrtf(k_norm_sq), (double)q_norm_inv, (double)k_norm_inv);
                fprintf(stderr, "  attn_score=%.8e\n", (double)attn_score);
                fprintf(stderr, "  q[0..3]=%.8f %.8f %.8f %.8f\n", (double)q_t[0], (double)q_t[1], (double)q_t[2], (double)q_t[3]);
                fprintf(stderr, "  k[0..3]=%.8f %.8f %.8f %.8f\n", (double)k_t[0], (double)k_t[1], (double)k_t[2], (double)k_t[3]);
                fprintf(stderr, "  v[0..3]=%.8f %.8f %.8f %.8f\n", (double)v_t[0], (double)v_t[1], (double)v_t[2], (double)v_t[3]);
                fprintf(stderr, "  state[0..3]=%.8e %.8e %.8e %.8e\n",
                    (double)state[0], (double)state[1], (double)state[2], (double)state[3]);
                // Check: is out_data pre-zeroed?
                fprintf(stderr, "  out_t[0..3] BEFORE=%.8e %.8e %.8e %.8e\n",
                    (double)out_t[0], (double)out_t[1], (double)out_t[2], (double)out_t[3]);
                // Pointer addresses for aliasing check
                fprintf(stderr, "  PTRS: out_data=%p state_out=%p state=%p out_t=%p\n",
                    (void*)out_data, (void*)(out_data + head_dim * n_tokens * n_heads * n_seqs),
                    (void*)state, (void*)out_t);
                fflush(stderr);
            }

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

                // === DEBUG: trace row 0 computation ===
                if (_do_dbg && row == 0) {
                    fprintf(stderr, "  ROW0: v_prime=%.8e out_val=%.8e v_new=%.8e\n",
                        (double)v_prime, (double)out_val, (double)v_new);
                    fprintf(stderr, "  ROW0: out_t[0] = out_val + v_new*attn = %.8e + %.8e*%.8e = %.8e\n",
                        (double)out_val, (double)v_new, (double)attn_score, (double)out_t[0]);
                    fprintf(stderr, "  ROW0: v_t[0]*beta = %.8e * %.8f = %.8e\n",
                        (double)v_t[0], (double)beta_val, (double)(v_t[0]*beta_val));
                    fflush(stderr);
                }
            }

            if (_do_dbg) {
                fprintf(stderr, "  out_t[0..3] AFTER_ROW=%.8e %.8e %.8e %.8e\n",
                    (double)out_t[0], (double)out_t[1], (double)out_t[2], (double)out_t[3]);
                fprintf(stderr, "  v_new_buf[0..3]=%.8e %.8e %.8e %.8e\n",
                    (double)v_new_buf[0], (double)v_new_buf[1], (double)v_new_buf[2], (double)v_new_buf[3]);
                fflush(stderr);
            }

            for (int64_t col = 0; col < head_dim; ++col) {
                const float k_col = k_t[col] * k_norm_inv;
                for (int64_t row = 0; row < head_dim; ++row) {
                    float s = state[row + col * head_dim];
                    s = decay * s + v_new_buf[row] * k_col;
                    state[row + col * head_dim] = fminf(fmaxf(s, -1e6f), 1e6f);
                }
            }

            if (_do_dbg) {
                fprintf(stderr, "  out_t[0..3] AFTER_STATE=%.8e %.8e %.8e %.8e\n",
                    (double)out_t[0], (double)out_t[1], (double)out_t[2], (double)out_t[3]);
                fprintf(stderr, "  new_state[0..3]=%.8e %.8e %.8e %.8e\n",
                    (double)state[0], (double)state[1], (double)state[2], (double)state[3]);
                _dbg_deep++;
                fflush(stderr);
            }
        }"""

if old in content:
    content = content.replace(old, new, 1)
    print("DEBUG V12 PATCH OK")
else:
    print("DEBUG V12 PATCH FAIL - trying to find the code block")
    # Print surrounding context to help debug
    idx = content.find('float * out_t = out_data + out_head_offset')
    if idx >= 0:
        print(f"Found 'float * out_t' at position {idx}")
        print(repr(content[idx:idx+200]))
    else:
        print("Could not find 'float * out_t' at all!")
    sys.exit(1)

with open('ggml/src/ggml.c', 'w') as f:
    f.write(content)

print("All debug v12 patches applied")
