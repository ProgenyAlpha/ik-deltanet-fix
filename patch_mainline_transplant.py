"""
Transplant upstream llama.cpp's DeltaNet implementation into ik_llama.cpp PR #1251.

Replaces Codex's build_delta_net_chunking and build_delta_net_autoregressive with
upstream's implementations. Removes the fused kernel dispatch path.

Key changes from Codex's version:
1. k_beta: direct ggml_mul(k, beta) instead of explicit repeat_4d
2. g_diff: direct sub without repeat_4d of g_last
3. key_gdiff, q_g_exp: direct mul instead of repeat_4d wrapping
4. attn_kq operand order: mul(attn_kq, decay_mask) not mul(decay_mask, attn_kq)
5. identity: ggml_repeat instead of ggml_repeat_4d
6. State reshape: reshape from [S_v, S_v*H_v, 1, n_seqs] to [S_v, S_v, H_v, n_seqs] inside functions
7. Dispatch: always chunking for T>1, autoregressive for T=1, no fused path
"""

import sys

with open('src/llama-build-context.cpp', 'r') as f:
    content = f.read()

patches_applied = 0

# ============================================================
# PATCH 1: Replace build_delta_net_chunking
# ============================================================
# Find the start and end of the chunking lambda
chunk_start_marker = '    auto build_delta_net_chunking = [&](ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,'
chunk_end_marker = '    auto build_delta_net_autoregressive = [&](ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,'

chunk_start = content.find(chunk_start_marker)
chunk_end = content.find(chunk_end_marker)

if chunk_start < 0:
    print("FAIL: Could not find build_delta_net_chunking start")
    sys.exit(1)
if chunk_end < 0:
    print("FAIL: Could not find build_delta_net_autoregressive start")
    sys.exit(1)

# The upstream chunking implementation, adapted as a lambda
upstream_chunking = '''    auto build_delta_net_chunking = [&](ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                        ggml_tensor * g, ggml_tensor * beta, ggml_tensor * state,
                                        ggml_tensor * causal_mask, ggml_tensor * identity,
                                        ggml_tensor * diag_mask, int il) -> std::pair<ggml_tensor *, ggml_tensor *> {
        // UPSTREAM TRANSPLANT from llama.cpp src/models/qwen3next.cpp
        // See: https://github.com/ProgenyAlpha/ik-deltanet-fix
        const int64_t S_k      = q->ne[0];
        const int64_t H_k      = q->ne[1];
        const int64_t n_tokens = q->ne[2];
        const int64_t n_seqs   = q->ne[3];

        const int64_t S_v = v->ne[0];
        const int64_t H_v = v->ne[1];

        GGML_ASSERT(v->ne[2] == n_tokens);
        GGML_ASSERT(k->ne[2] == n_tokens);
        GGML_ASSERT(g->ne[0] == H_v && g->ne[1] == n_tokens && g->ne[2] == n_seqs);
        GGML_ASSERT(beta->ne[0] == H_v && beta->ne[2] == n_tokens && beta->ne[3] == n_seqs);
        GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
        GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);
        GGML_ASSERT(H_k == H_v);

        const float eps_norm = hparams.f_norm_rms_eps;
        q = ggml_l2_norm(ctx0, q, eps_norm);
        k = ggml_l2_norm(ctx0, k, eps_norm);

        const float scale = 1.0f / sqrtf(S_v);
        q = ggml_scale(ctx0, q, scale);
        beta = ggml_sigmoid(ctx0, beta);

        cb(q, "q_in", il);
        cb(k, "k_in", il);
        cb(v, "v_in", il);
        cb(beta, "beta_in", il);
        cb(g, "g_in", il);

        q = ggml_cont_4d(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3), S_v, n_tokens, H_v, n_seqs);
        k = ggml_cont_4d(ctx0, ggml_permute(ctx0, k, 0, 2, 1, 3), S_v, n_tokens, H_v, n_seqs);
        v = ggml_cont_4d(ctx0, ggml_permute(ctx0, v, 0, 2, 1, 3), S_v, n_tokens, H_v, n_seqs);
        g = ggml_cont_4d(ctx0, ggml_permute(ctx0, g, 2, 0, 3, 1), n_tokens, 1, H_k, n_seqs);

        beta  = ggml_cont(ctx0, ggml_permute(ctx0, beta, 2, 0, 1, 3));
        state = ggml_reshape_4d(ctx0, state, S_v, S_v, H_v, n_seqs);

        cb(q, "q_perm", il);
        cb(k, "k_perm", il);
        cb(v, "v_perm", il);
        cb(beta, "beta_perm", il);
        cb(g, "g_perm", il);
        cb(state, "state_in", il);

        const int64_t chunk_size = QWEN3NEXT_CHUNK_SIZE;
        const int64_t pad = (chunk_size - n_tokens % chunk_size) % chunk_size;
        const int64_t n_chunks = (n_tokens + pad) / chunk_size;

        q = ggml_pad(ctx0, q, 0, pad, 0, 0);
        k = ggml_pad(ctx0, k, 0, pad, 0, 0);
        v = ggml_pad(ctx0, v, 0, pad, 0, 0);
        g = ggml_pad(ctx0, g, pad, 0, 0, 0);
        beta = ggml_pad(ctx0, beta, 0, pad, 0, 0);

        ggml_tensor * v_beta = ggml_mul(ctx0, v, beta);
        ggml_tensor * k_beta = ggml_mul(ctx0, k, beta);

        q      = ggml_reshape_4d(ctx0, q,      S_k, chunk_size, n_chunks, H_k * n_seqs);
        k      = ggml_reshape_4d(ctx0, k,      S_k, chunk_size, n_chunks, H_k * n_seqs);
        k_beta = ggml_reshape_4d(ctx0, k_beta, S_k, chunk_size, n_chunks, H_k * n_seqs);
        v      = ggml_reshape_4d(ctx0, v,      S_v, chunk_size, n_chunks, H_v * n_seqs);
        v_beta = ggml_reshape_4d(ctx0, v_beta, S_v, chunk_size, n_chunks, H_v * n_seqs);

        g    = ggml_reshape_4d(ctx0, g, chunk_size, 1, n_chunks, H_k * n_seqs);
        beta = ggml_reshape_4d(ctx0, beta, 1, chunk_size, n_chunks, H_k * n_seqs);

        ggml_tensor * g_cumsum = ggml_cumsum(ctx0, g);
        cb(g_cumsum, "g_cumsum", il);

        ggml_tensor * gcs_i = ggml_repeat_4d(ctx0, g_cumsum, chunk_size, chunk_size, n_chunks, H_v * n_seqs);
        ggml_tensor * gcs_j = ggml_reshape_4d(ctx0, g_cumsum, 1, chunk_size, n_chunks, H_v * n_seqs);

        ggml_tensor * gcs_j_broadcast =
            ggml_repeat_4d(ctx0, gcs_j, chunk_size, chunk_size, n_chunks, H_v * n_seqs);

        ggml_tensor * decay_mask = ggml_sub(ctx0, gcs_j_broadcast, gcs_i);
        cb(decay_mask, "decay_mask", il);

        decay_mask = ggml_mul(ctx0, decay_mask, diag_mask);
        decay_mask = ggml_exp(ctx0, decay_mask);
        decay_mask = ggml_mul(ctx0, decay_mask, diag_mask);

        ggml_tensor * kmulkbeta = ggml_mul_mat(ctx0, k, k_beta);

        ggml_tensor * k_decay = ggml_mul(ctx0, kmulkbeta, decay_mask);
        ggml_tensor * attn    = ggml_neg(ctx0, ggml_mul(ctx0, k_decay, causal_mask));
        cb(attn, "attn_pre_solve", il);

        ggml_tensor * attn_lower = ggml_mul(ctx0, attn, causal_mask);
        ggml_tensor * lhs        = ggml_sub(ctx0, ggml_repeat(ctx0, identity, attn_lower), attn_lower);

        ggml_tensor * lin_solve  = ggml_solve_tri(ctx0, lhs, attn, true, true, false);
        attn                     = ggml_mul(ctx0, lin_solve, causal_mask);
        attn                     = ggml_add(ctx0, attn, identity);
        cb(attn, "attn_solved", il);

        v = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, v_beta)), attn);

        ggml_tensor * g_cumsum_t = ggml_cont(ctx0, ggml_transpose(ctx0, g_cumsum));
        ggml_tensor * gexp       = ggml_exp(ctx0, g_cumsum_t);

        ggml_tensor * kbeta_gexp = ggml_mul(ctx0, k_beta, gexp);
        cb(kbeta_gexp, "kbeta_gexp", il);

        ggml_tensor * k_cumdecay =
            ggml_cont(ctx0, ggml_transpose(ctx0, ggml_mul_mat(ctx0, attn, ggml_cont(ctx0, ggml_transpose(ctx0, kbeta_gexp)))));
        cb(k_cumdecay, "k_cumdecay", il);

        ggml_tensor * attn_kq = ggml_mul_mat(ctx0, k, q);
        attn_kq = ggml_mul(ctx0, attn_kq, decay_mask);
        attn_kq = ggml_mul(ctx0, attn_kq, diag_mask);
        cb(attn_kq, "attn_kq", il);

        ggml_tensor * g_last = ggml_view_4d(ctx0, g_cumsum, 1, 1, g_cumsum->ne[2], g_cumsum->ne[3],
                                            g_cumsum->nb[1], g_cumsum->nb[2], g_cumsum->nb[3],
                                            (g_cumsum->ne[0] - 1) * ggml_element_size(g_cumsum));
        g_last = ggml_cont(ctx0, g_last);
        cb(g_last, "g_last", il);

        ggml_tensor * g_last_exp = ggml_exp(ctx0, g_last);
        cb(g_last_exp, "g_last_exp", il);

        ggml_tensor * g_last_repeat = ggml_repeat_4d(ctx0, g_last, chunk_size, 1, n_chunks, H_v * n_seqs);
        ggml_tensor * g_diff = ggml_neg(ctx0, ggml_sub(ctx0, g_cumsum, g_last_repeat));
        cb(g_diff, "g_diff", il);

        ggml_tensor * g_diff_exp = ggml_exp(ctx0, g_diff);
        ggml_tensor * g_diff_exp_t = ggml_reshape_4d(ctx0, g_diff_exp,
                                                     1, chunk_size, n_chunks, g_diff_exp->ne[3]);

        ggml_tensor * key_gdiff = ggml_mul(ctx0, k, g_diff_exp_t);
        cb(key_gdiff, "key_gdiff", il);

        ggml_tensor * key_gdiff_t = ggml_cont(ctx0, ggml_transpose(ctx0, key_gdiff));
        cb(key_gdiff_t, "key_gdiff_t", il);

        ggml_tensor * new_state = state;
        cb(new_state, "new_state", il);

        ggml_tensor * core_attn_out = nullptr;

        for (int64_t chunk = 0; chunk < n_chunks; chunk++) {
            ggml_tensor * q_chunk          = get_slice_2d(q, chunk);
            ggml_tensor * v_chunk          = get_slice_2d(v, chunk);
            ggml_tensor * gexp_chunk       = get_slice_2d(gexp, chunk);
            ggml_tensor * k_cumdecay_chunk = get_slice_2d(k_cumdecay, chunk);
            ggml_tensor * attn_chunk       = get_slice_2d(attn_kq, chunk);
            cb(attn_chunk, "attn_chunk", il);

            ggml_tensor * state_t = ggml_cont_4d(ctx0, ggml_permute(ctx0, new_state, 1, 0, 2, 3), S_v, S_v, 1, H_v * n_seqs);

            ggml_tensor * v_prime = ggml_mul_mat(ctx0, state_t, k_cumdecay_chunk);
            cb(v_prime, "v_prime_chunk", il);

            ggml_tensor * v_new   = ggml_sub(ctx0, ggml_repeat(ctx0, v_chunk, v_prime), v_prime);
            ggml_tensor * v_new_t = ggml_cont(ctx0, ggml_transpose(ctx0, v_new));
            cb(v_new, "v_new_chunk", il);

            ggml_tensor * q_g_exp    = ggml_mul(ctx0, q_chunk, gexp_chunk);
            ggml_tensor * attn_inter = ggml_mul_mat(ctx0, state_t, q_g_exp);
            cb(attn_inter, "attn_inter_chunk", il);

            ggml_tensor * v_attn = ggml_mul_mat(ctx0, v_new_t, attn_chunk);
            cb(v_attn, "v_attn_chunk", il);

            ggml_tensor * core_attn_out_chunk = ggml_add(ctx0, attn_inter, v_attn);
            cb(core_attn_out_chunk, "core_attn_out_chunk", il);

            core_attn_out = core_attn_out == nullptr
                ? core_attn_out_chunk
                : ggml_concat(ctx0, core_attn_out, core_attn_out_chunk, 2);

            ggml_tensor * k_gdiff_t = get_slice_2d(key_gdiff_t, chunk);
            ggml_tensor * kgdmulvnew = ggml_mul_mat(ctx0, v_new_t, k_gdiff_t);

            ggml_tensor * gexp_last_chunk = ggml_cont(ctx0, get_slice_2d(g_last_exp, chunk));
            new_state = ggml_add(ctx0,
                ggml_mul(ctx0, new_state, ggml_reshape_4d(ctx0, gexp_last_chunk, gexp_last_chunk->ne[0], gexp_last_chunk->ne[1], H_v, n_seqs)),
                ggml_reshape_4d(ctx0, kgdmulvnew, kgdmulvnew->ne[0], kgdmulvnew->ne[1], H_v, n_seqs));
        }

        ggml_tensor * output_tokens = ggml_view_4d(ctx0, core_attn_out,
                S_v, n_tokens, H_v, n_seqs,
                ggml_row_size(core_attn_out->type, S_v),
                ggml_row_size(core_attn_out->type, S_v * chunk_size * n_chunks),
                ggml_row_size(core_attn_out->type, S_v * chunk_size * n_chunks * H_v), 0);
        output_tokens = ggml_cont(ctx0, output_tokens);
        cb(output_tokens, "output_tokens", il);

        output_tokens = ggml_permute(ctx0, output_tokens, 0, 2, 1, 3);
        output_tokens = ggml_cont(ctx0, output_tokens);

        return {output_tokens, new_state};
    };

'''

# ============================================================
# PATCH 2: Replace build_delta_net_autoregressive
# ============================================================
auto_end_marker = '    // Fused DeltaNet path.'
auto_end = content.find(auto_end_marker)
if auto_end < 0:
    # Try alternative marker
    auto_end_marker = '    auto build_delta_net_fused = [&]('
    auto_end = content.find(auto_end_marker)

if auto_end < 0:
    print("FAIL: Could not find end of autoregressive (fused path start)")
    sys.exit(1)

upstream_autoregressive = '''    auto build_delta_net_autoregressive = [&](ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                              ggml_tensor * g, ggml_tensor * beta, ggml_tensor * state,
                                              int il) -> std::pair<ggml_tensor *, ggml_tensor *> {
        // UPSTREAM TRANSPLANT from llama.cpp src/models/qwen3next.cpp
        const int64_t S_k      = q->ne[0];
        const int64_t H_k      = q->ne[1];
        const int64_t n_tokens = q->ne[2];
        const int64_t n_seqs   = q->ne[3];

        const int64_t S_v = v->ne[0];
        const int64_t H_v = v->ne[1];

        GGML_ASSERT(n_tokens == 1);
        GGML_ASSERT(H_k == H_v);

        const float eps_norm = hparams.f_norm_rms_eps;
        q = ggml_l2_norm(ctx0, q, eps_norm);
        k = ggml_l2_norm(ctx0, k, eps_norm);

        const float scale = 1.0f / sqrtf(S_v);
        q    = ggml_scale(ctx0, q, scale);
        beta = ggml_sigmoid(ctx0, beta);

        cb(q, "q_in", il);
        cb(k, "k_in", il);
        cb(v, "v_in", il);
        cb(beta, "beta_in", il);
        cb(g, "g_in", il);

        state = ggml_reshape_4d(ctx0, state, S_v, S_v, H_v, n_seqs);

        ggml_tensor * g_t    = ggml_reshape_4d(ctx0, ggml_transpose(ctx0, g), 1, 1, H_k, n_seqs);
        ggml_tensor * beta_t = ggml_reshape_4d(ctx0, ggml_transpose(ctx0, beta), 1, 1, H_k, n_seqs);

        g_t = ggml_exp(ctx0, g_t);
        state = ggml_mul(ctx0, state, g_t);

        ggml_tensor * k_t_unsqueezed = ggml_reshape_4d(ctx0, k, 1, S_v, H_v, n_seqs);
        ggml_tensor * kv_mem         = ggml_mul(ctx0, state, k_t_unsqueezed);
        kv_mem = ggml_transpose(ctx0, ggml_sum_rows(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, kv_mem))));

        ggml_tensor * v_t    = ggml_reshape_4d(ctx0, v, S_v, 1, H_v, n_seqs);
        ggml_tensor * v_diff = ggml_sub(ctx0, v_t, kv_mem);
        ggml_tensor * delta  = ggml_mul(ctx0, v_diff, beta_t);

        ggml_tensor * k_t_delta = ggml_mul(ctx0, ggml_repeat_4d(ctx0, k_t_unsqueezed, S_v, S_v, H_v, n_seqs), delta);
        state                   = ggml_add(ctx0, state, k_t_delta);

        ggml_tensor * q_t_unsqueezed = ggml_reshape_4d(ctx0, q, 1, S_v, H_v, n_seqs);
        ggml_tensor * state_q        = ggml_mul(ctx0, state, q_t_unsqueezed);
        ggml_tensor * core_attn_out =
            ggml_transpose(ctx0, ggml_sum_rows(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, state_q))));

        cb(core_attn_out, "output_tokens", il);
        cb(state, "new_state", il);

        return {core_attn_out, state};
    };

'''

# Apply patches
# Replace chunking: from chunk_start to chunk_end (exclusive)
content = content[:chunk_start] + upstream_chunking + content[chunk_end:]
patches_applied += 1
print(f"PATCH 1: Replaced build_delta_net_chunking with upstream version")

# Re-find autoregressive markers after chunking replacement shifted positions
auto_start_marker = '    auto build_delta_net_autoregressive = [&](ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,'
auto_start = content.find(auto_start_marker)
# Find the fused path start (which marks end of autoregressive)
fused_marker = '    // Fused DeltaNet path.'
fused_start = content.find(fused_marker)
if fused_start < 0:
    fused_marker = '    auto build_delta_net_fused = [&]('
    fused_start = content.find(fused_marker)

if auto_start < 0 or fused_start < 0:
    print(f"FAIL: Could not find autoregressive bounds (auto_start={auto_start}, fused_start={fused_start})")
    sys.exit(1)

# Replace autoregressive + fused (remove fused entirely, keep just autoregressive)
# The fused lambda ends right before build_qkvz — that's the safe boundary
qkvz_marker = '    auto build_qkvz = [&]('
qkvz_start = content.find(qkvz_marker)

if qkvz_start < 0:
    print("FAIL: Could not find build_qkvz (sibling lambda after fused)")
    sys.exit(1)

print(f"  autoregressive starts at: {auto_start}")
print(f"  build_qkvz starts at: {qkvz_start}")
print(f"  replacing {qkvz_start - auto_start} chars (autoregressive + fused lambdas)")

# Replace from autoregressive start to build_qkvz with just the upstream autoregressive
content = content[:auto_start] + upstream_autoregressive + content[qkvz_start:]
patches_applied += 1
print(f"PATCH 2: Replaced autoregressive + removed fused path")

# ============================================================
# PATCH 3: Replace dispatch logic — always chunking/autoregressive, no fused
# ============================================================
old_dispatch_block = """        const bool use_fused_delta_net = use_fused_delta_mode;

        if (use_fused_delta_net) {
            attn_out = build_delta_net_fused(q_conv, k_conv, v_conv, gate, beta, state, il);
        } else {
            GGML_ASSERT(causal_mask != nullptr);
            GGML_ASSERT(identity    != nullptr);
            GGML_ASSERT(diag_mask   != nullptr);

            attn_out = n_tok == 1
                ? build_delta_net_autoregressive(q_conv, k_conv, v_conv, gate, beta, state, il)
                : build_delta_net_chunking(q_conv, k_conv, v_conv, gate, beta, state, causal_mask, identity, diag_mask, il);
        }"""

new_dispatch_block = """        // MAINLINE TRANSPLANT: always use upstream chunking + autoregressive, no fused kernel
        // See: https://github.com/ProgenyAlpha/ik-deltanet-fix
        GGML_ASSERT(causal_mask != nullptr);
        GGML_ASSERT(identity    != nullptr);
        GGML_ASSERT(diag_mask   != nullptr);

        attn_out = n_tok == 1
            ? build_delta_net_autoregressive(q_conv, k_conv, v_conv, gate, beta, state, il)
            : build_delta_net_chunking(q_conv, k_conv, v_conv, gate, beta, state, causal_mask, identity, diag_mask, il);"""

if old_dispatch_block in content:
    content = content.replace(old_dispatch_block, new_dispatch_block, 1)
    patches_applied += 1
    print(f"PATCH 3: Replaced dispatch to use upstream paths only")
else:
    print("WARN: Could not find exact dispatch block, trying without use_fused_delta_net line")
    # Try finding just the dispatch part
    alt_dispatch = content.find('use_fused_delta_net = use_fused_delta_mode')
    if alt_dispatch >= 0:
        print(f"  Found use_fused_delta_mode at position {alt_dispatch}")
        print(f"  Context: {repr(content[alt_dispatch-50:alt_dispatch+200])}")
    else:
        print("  Could not find use_fused_delta_mode either")

with open('src/llama-build-context.cpp', 'w') as f:
    f.write(content)

print(f"\nAll {patches_applied} patches applied successfully")
if patches_applied < 3:
    print("WARNING: Not all patches applied!")
    sys.exit(1)
