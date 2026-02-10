import sys

with open('src/llama-build-context.cpp', 'r') as f:
    content = f.read()

# Change the dispatch logic so fused mode only handles T=1 (generation),
# while T>1 (prompt eval) always uses chunked path.
# This tests whether the fused kernel's T>1 path is the issue.

old = """        if (use_fused_delta_net) {
            attn_out = build_delta_net_fused(q_conv, k_conv, v_conv, gate, beta, state, il);
        } else {
            GGML_ASSERT(causal_mask != nullptr);
            GGML_ASSERT(identity    != nullptr);
            GGML_ASSERT(diag_mask   != nullptr);

            attn_out = n_tok == 1
                ? build_delta_net_autoregressive(q_conv, k_conv, v_conv, gate, beta, state, il)
                : build_delta_net_chunking(q_conv, k_conv, v_conv, gate, beta, state, causal_mask, identity, diag_mask, il);
        }"""

new = """        if (use_fused_delta_net && n_tok == 1) {
            // Fused kernel for single-token generation only
            attn_out = build_delta_net_fused(q_conv, k_conv, v_conv, gate, beta, state, il);
        } else {
            GGML_ASSERT(causal_mask != nullptr);
            GGML_ASSERT(identity    != nullptr);
            GGML_ASSERT(diag_mask   != nullptr);

            attn_out = n_tok == 1
                ? build_delta_net_autoregressive(q_conv, k_conv, v_conv, gate, beta, state, il)
                : build_delta_net_chunking(q_conv, k_conv, v_conv, gate, beta, state, causal_mask, identity, diag_mask, il);
        }"""

if old in content:
    content = content.replace(old, new, 1)
    print("HYBRID DISPATCH PATCH OK")
else:
    print("HYBRID DISPATCH PATCH FAIL")
    sys.exit(1)

with open('src/llama-build-context.cpp', 'w') as f:
    f.write(content)
