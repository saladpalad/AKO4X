# C++

Two files: `solution/kernel.cpp` and `solution/binding.py` (TVM-FFI Python
binding). `binding.py` exposes the entry point via `@register_func`:

```python
import torch
from tvm.ffi import register_func

@register_func("kernel")
def kernel(input, weight):
    output = torch.empty_like(input)
    # call into the compiled kernel.cpp
    return output
```

The TVM-FFI builder compiles `.cpp` automatically but can't link external
shared libraries at compile time. That doesn't limit a valid solution:
cuBLAS / cuDNN are disallowed anyway (write your own kernel — the
`benchmark` skill's "Valid solution" rule), and CUTLASS is header-only
(`#include` it, no linking). Single-source detail: the `benchmark` skill,
"TVM FFI builder behavior".

`config.toml` schema (which `language` value to set, `entry_point` format) is centralized in the `benchmark` skill.
