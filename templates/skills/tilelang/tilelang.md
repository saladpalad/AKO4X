# TileLang

Single file `solution/kernel.py`. Requires `tilelang` package — without it, all workloads report COMPILE_ERROR.

TileLang uses a factory pattern: an outer function decorated with `@tilelang.jit` returns a `@T.prim_func` kernel. The JIT compiler infers the target (CUDA/HIP) from input tensors at first call.

```python
import torch
import tilelang
import tilelang.language as T

@tilelang.jit
def my_kernel_factory(M: int, N: int, block_size: int = 256, dtype: str = "float16"):
    @T.prim_func
    def my_kernel(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M * N, block_size), threads=block_size) as bx:
            for i in T.Parallel(block_size):
                idx = bx * block_size + i
                if idx < M * N:
                    C[idx] = A[idx] + B[idx]
    return my_kernel

@torch.no_grad()
def run(input, weight):
    M, N = input.shape
    output = torch.empty_like(input)
    # First call compiles; subsequent calls reuse the compiled kernel
    my_kernel_factory(M, N, dtype=str(input.dtype).split(".")[-1])(input, weight, output)
    return output
```

Key points: `@tilelang.jit` wraps a factory that captures compile-time
params (shapes / tile sizes / dtypes); high-level tile ops `T.gemm()` /
`T.copy()` / `T.reduce()` and `T.alloc_shared()` / `T.alloc_fragment()`
cover most fused patterns.

## Program-Dependent Launch (PDL) — TileLang bindings

Generic PDL theory (when it helps vs regresses, the Waves-Per-SM
decision table) is in the `cuda` skill ("Program-Dependent Launch (PDL)
for kernel→kernel overlap"); this section is only the TileLang binding.
TileLang exposes native PDL primitives in `tilelang/language/pdl.py`:

- `T.pdl_trigger()` — lowers to `cudaTriggerProgrammaticLaunchCompletion`
  (PTX `griddepcontrol.launch_dependents`).
- `T.pdl_sync()` — lowers to `cudaGridDependencySynchronize`
  (PTX `griddepcontrol.wait`).

**Key difference from Triton**: TileLang's JIT automatically sets
`CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION` when these
primitives appear in a kernel — **no host-side `launch_pdl=True` flag
needed** on the TileLang side. (On Triton you must pass the flag at the
call site; without it, `gdc_launch_dependents()` is a no-op.)

**Mixed-language pattern** (TileLang producer + Triton consumer):

```python
@T.prim_func
def producer(...):
    with T.Kernel(grid, threads=threads) as bx:
        # ... compute + T.copy writes ...
        T.pdl_trigger()                   # LAST stmt; JIT sets launch attr

@triton.jit
def consumer(...):
    # ... address / constant setup ...
    gdc_wait()                            # from triton.language.extra.cuda
    # ... load + compute ...

consumer[grid](..., launch_pdl=True)      # Triton requires explicit flag
```

## Gotchas / known-broken patterns

API and behavior surprises that have silently cost multiple iterations.
Re-verify if the toolchain changes.

- **`@tilelang.jit` MUST decorate the factory, not the inner `@T.prim_func`.**
  Applying `@tilelang.jit` directly on top of `@T.prim_func` inside a wrapper
  function (e.g. `def build(): @tilelang.jit \n @T.prim_func \n def main(...):
  ...; return main`) compiles AND runs without error but produces a kernel
  with no pipelining — observed a 450× slowdown (53.5 ms/call vs 0.12 ms for
  the same shape in Triton) on an MLA prefill kernel at M=64, BLOCK_N=64.
  **Fix**: put `@tilelang.jit(...)` on the *factory* function and `@T.prim_func`
  on the inner kernel function the factory returns — exactly as the example
  at the top of this skill shows. The "wrong" pattern is a silent perf trap.

- **Python `and` / `or` short-circuits on TileLang bool tile expressions.**
  `causal and valid_kv` where both are TileLang `<`/`<=` results evaluates
  `bool(causal_expr)` at Python tracing time — TileLang expression objects
  are typically truthy, so the `and` returns just `valid_kv` and the LHS
  branch is silently dropped. Symptom: kernel compiles, runs, PASSES
  correctness via the 0.9 matched-ratio (i.e. ~10% of output is wrong but
  not enough to fail), with a large rel_err but borderline abs_err. **Fix**:
  use bitwise `&` / `|` to combine tile booleans:
  `keep = (causal_expr) & (valid_expr)`.

- **`T.copy(src, dst[i*H:(i+1)*H, :])` with runtime-indexed slice destination
  is broken.** With `i` a runtime variable (e.g. a `T.serial` loop index),
  TileLang does not lower the slice-destination indexing correctly — the
  destination tile receives garbage / wrong-stride data. Symptom:
  INCORRECT_NUMERICAL with abs_err in the 10s (vs typical bf16 attention
  error in the 1e-2 range). **Workaround**: use a per-element
  `for m, dd in T.Parallel(M, D): dst[m, dd] = src[m // H + i_base, ...]`
  loop instead. Hardware coalescing makes this fine — the rewrite is
  semantic-only for the failing pattern.

`config.toml` schema (which `language` value to set, `entry_point` format) is centralized in the `benchmark` skill.
