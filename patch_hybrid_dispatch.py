import sys

with open('src/llama-build-context.cpp', 'r') as f:
    content = f.read()

# Hybrid dispatch: fused for prefill (T>1, correct PPL), autoregressive for generation (T=1, correct output)
# The fused kernel has a threading bug in cont+permute ops that only manifests at T=1.
# The autoregressive path handles T=1 correctly with standard GGML ops.
# See: https://github.com/ProgenyAlpha/ik-deltanet-fix

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

new = """        // HYBRID DISPATCH: fused for prefill (T>1), autoregressive for generation (T=1)
        // Fused kernel has correct PPL but threading bug in cont+permute at T=1
        // See: https://github.com/ProgenyAlpha/ik-deltanet-fix
        if (use_fused_delta_net && n_tok > 1) {
            attn_out = build_delta_net_fused(q_conv, k_conv, v_conv, gate, beta, state, il);
        } else if (n_tok == 1) {
            attn_out = build_delta_net_autoregressive(q_conv, k_conv, v_conv, gate, beta, state, il);
        } else {
            GGML_ASSERT(causal_mask != nullptr);
            GGML_ASSERT(identity    != nullptr);
            GGML_ASSERT(diag_mask   != nullptr);

            attn_out = build_delta_net_chunking(q_conv, k_conv, v_conv, gate, beta, state, causal_mask, identity, diag_mask, il);
        }"""

if old in content:
    content = content.replace(old, new, 1)
    print("PATCH hybrid_dispatch OK")
else:
    print("PATCH hybrid_dispatch FAIL - searching for context...")
    idx = content.find('use_fused_delta_net')
    if idx >= 0:
        block_start = content.rfind('\n', 0, idx)
        print(f"Found use_fused_delta_net near position {idx}")
        print(repr(content[block_start:block_start+500]))
    else:
        print("Could not find 'use_fused_delta_net' in file")
    sys.exit(1)

with open('src/llama-build-context.cpp', 'w') as f:
    f.write(content)

print("Hybrid dispatch patch applied successfully")
