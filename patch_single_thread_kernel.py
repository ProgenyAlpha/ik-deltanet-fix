import sys

with open('ggml/src/ggml.c', 'r') as f:
    content = f.read()

# Fix: Force single-threaded execution of delta_net kernel.
# The kernel has a multi-threading race condition that produces garbage output.
# With this fix, only thread 0 executes (processes all 32 heads).
# Threads 1-N return immediately and wait at the GGML barrier.
# All other ops (matmul, etc.) still use all threads.

old = """    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t total_heads = n_heads * n_seqs;
    const int64_t heads_per_thread = (total_heads + nth - 1) / nth;
    const int64_t h_start = ith * heads_per_thread;
    const int64_t h_end = (h_start + heads_per_thread < total_heads) ? h_start + heads_per_thread : total_heads;"""

new = """    // WORKAROUND: force single-threaded execution due to race condition
    // See: https://github.com/YurkoHoshko/ik_llama.cpp/pull/1251
    if (params->ith != 0) {
        return;
    }
    const int ith = 0;
    const int nth = 1;

    const int64_t total_heads = n_heads * n_seqs;
    const int64_t heads_per_thread = (total_heads + nth - 1) / nth;
    const int64_t h_start = ith * heads_per_thread;
    const int64_t h_end = (h_start + heads_per_thread < total_heads) ? h_start + heads_per_thread : total_heads;"""

if old in content:
    content = content.replace(old, new, 1)
    print("PATCH single_thread_kernel OK")
else:
    print("PATCH single_thread_kernel FAIL - searching for context...")
    idx = content.find('const int ith = params->ith;')
    if idx >= 0:
        print(f"Found at position {idx}")
        print(repr(content[idx:idx+300]))
    else:
        print("Could not find 'const int ith = params->ith;'")
    sys.exit(1)

with open('ggml/src/ggml.c', 'w') as f:
    f.write(content)

print("All patches applied")
