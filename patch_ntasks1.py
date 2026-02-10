import sys

with open('ggml/src/ggml.c', 'r') as f:
    content = f.read()

# Patch: Force n_tasks=1 for DELTA_NET to work around multi-threading race condition.
# This means only thread 0 executes the delta_net kernel, while other ops still use all threads.

old = """        case GGML_OP_DELTA_NET:
            {
                n_tasks = n_threads;
            } break;"""

new = """        case GGML_OP_DELTA_NET:
            {
                n_tasks = 1;  // WORKAROUND: fused kernel has multi-threading race condition
            } break;"""

if old in content:
    content = content.replace(old, new, 1)
    print("PATCH n_tasks=1 OK")
else:
    print("PATCH n_tasks=1 FAIL - searching for context...")
    idx = content.find('GGML_OP_DELTA_NET')
    while idx >= 0:
        print(f"  Found at position {idx}: {repr(content[idx:idx+80])}")
        idx = content.find('GGML_OP_DELTA_NET', idx+1)
    sys.exit(1)

# Also add a one-time diagnostic to print state_in vs dst pointer addresses
# to verify no memory aliasing (prints once then never again)
old2 = """    float * v_new_buf = (float *) malloc(head_dim * sizeof(float));
    if (!v_new_buf) {
        return;
    }"""

new2 = r"""    float * v_new_buf = (float *) malloc(head_dim * sizeof(float));
    if (!v_new_buf) {
        return;
    }

    // Diagnostic: verify no memory aliasing between state_in and dst
    static int _alias_check = 0;
    if (ith == 0 && _alias_check < 1) {
        const float * si = (const float *) dst->src[5]->data;
        float * od = (float *) dst->data;
        int64_t os = head_dim * n_tokens * n_heads * n_seqs;
        float * so = od + os;
        fprintf(stderr, "DN_ALIAS_CHECK: state_in=%p dst_data=%p state_out=%p (offset=%lld floats)\n",
            (void*)si, (void*)od, (void*)so, (long long)os);
        fprintf(stderr, "DN_ALIAS_CHECK: state_in_end=%p state_out_end=%p\n",
            (void*)(si + head_dim*head_dim*n_heads*n_seqs),
            (void*)(so + head_dim*head_dim*n_heads*n_seqs));
        int aliased = ((char*)si >= (char*)od && (char*)si < (char*)(so + head_dim*head_dim*n_heads*n_seqs)) ||
                      ((char*)od >= (char*)si && (char*)od < (char*)(si + head_dim*head_dim*n_heads*n_seqs));
        fprintf(stderr, "DN_ALIAS_CHECK: ALIASED=%s\n", aliased ? "YES !!!" : "no");
        fprintf(stderr, "DN_ALIAS_CHECK: nth=%d n_tasks=1 (workaround active)\n", nth);
        _alias_check++;
        fflush(stderr);
    }"""

if old2 in content:
    content = content.replace(old2, new2, 1)
    print("PATCH alias_check OK")
else:
    print("PATCH alias_check FAIL")
    sys.exit(1)

with open('ggml/src/ggml.c', 'w') as f:
    f.write(content)

print("All v13 patches applied")
