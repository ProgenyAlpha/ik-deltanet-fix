import sys

with open('src/llama-model.cpp', 'r') as f:
    content = f.read()

import re
# Make ALL ssm_dt.bias create_tensor calls optional
pattern = r'(create_tensor\(tn\(LLM_TENSOR_SSM_DT,\s*"bias".*?),\s*0\)'
replacement = r'\1, llama_model_loader::TENSOR_NOT_REQUIRED)'
content_new, count = re.subn(pattern, replacement, content)
if count > 0:
    content = content_new
    print(f"PATCH ssm_dt.bias optional: {count} replacements")
else:
    print("PATCH ssm_dt.bias FAIL - pattern not found")
    # Try to find nearby code
    idx = content.find('ssm_dt_b')
    if idx >= 0:
        print(f"Found ssm_dt_b at pos {idx}: {repr(content[idx:idx+120])}")
    sys.exit(1)

with open('src/llama-model.cpp', 'w') as f:
    f.write(content)
