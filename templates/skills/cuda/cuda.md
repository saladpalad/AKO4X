# CUDA

Two `[build]` entry-point patterns; the TVM-FFI builder compiles `.cu`
automatically in both. (`config.toml` field placement — `[build].language`
/ `entry_point` / `destination_passing_style` — is in the `benchmark`
skill.)

**Direct symbol export (recommended).** Export the wrapper from `.cu` with
`TVM_FFI_DLL_EXPORT_TYPED_FUNC`; it receives `tvm::ffi::TensorView` args
(`.data_ptr()`, `.size(dim)`). Set `entry_point = "kernel.cu::my_kernel"`,
`destination_passing_style = true` (outputs are trailing args). The launch
**must** target PyTorch's current stream — a bare `<<<grid, block>>>` uses
the null stream, which `torch.cuda.graph()` does not capture (see
"Kernel-launch stream is NOT optional" below):

```cuda
#include <tvm/ffi/container/tensor.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void my_cuda_kernel(const half* in, half* out, int n) { /* ... */ }

void my_kernel(tvm::ffi::TensorView input, tvm::ffi::TensorView output) {
    int n = input.size(0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    my_cuda_kernel<<<(n+255)/256, 256, /*shmem=*/0, stream>>>(
        static_cast<const half*>(input.data_ptr()),
        static_cast<half*>(output.data_ptr()), n);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(my_kernel, my_kernel);
```

**Python binding.** Use `binding.py` for full Python control (torch custom ops,
launching the kernel yourself). Set
`entry_point = "binding.py::kernel"`, `destination_passing_style = false`:

```python
from tvm.ffi import register_func

@register_func("kernel")
def kernel(A, B):
    output = torch.empty(...)
    # launch your compiled kernel (ctypes / torch custom op)
    return output
```

The TVM-FFI builder can't link external shared libraries at compile time,
but that doesn't limit a valid solution: cuBLAS / cuDNN are disallowed
(write your own kernel — the `benchmark` skill's "Valid solution" rule),
and CUTLASS is header-only (`#include`, no linking). Single-source detail:
the `benchmark` skill, "TVM FFI builder behavior".

## Tips

### Kernel-launch stream is NOT optional under CUDA graph capture

`my_kernel<<<grid, block>>>(...)` without a stream argument targets the
legacy/null stream (id 0), which `torch.cuda.graph()` does **not** put
into capture mode. The launch runs immediately during capture but is
**not recorded into the graph** — subsequent `g.replay()`s skip it,
leaving stale output. Correctness often passes by coincidence (eager
warmup + per-workload fixed inputs in `use_isolated_runner = true` mode
leave the right bytes in the destination buffer). When the missing
kernel is alone in the graph, PyTorch warns `"The CUDA Graph is empty"`
on `capture_end`; paired with any kernel that captures correctly
(e.g. a Triton kernel) the warning is suppressed → silent failure.

Always route chevron launches through PyTorch's current stream:

```cuda
#include <ATen/cuda/CUDAContext.h>

void my_kernel_wrapper(tvm::ffi::TensorView x, tvm::ffi::TensorView out) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    my_cuda_kernel<<<grid, block, /*shmem_bytes=*/0, stream>>>(...);
}
```

Per-kernel: 4th chevron arg mandatory in any `.cu` that might run under
`torch.cuda.graph` capture. One concrete instance of silent kernel
skipping under graph capture; per-operator evidence lives in the
relevant `docs/prior/TRAPS.md` (when your operator has a prior archive).

### `cvt.rn.bf16x2.e4m3x2` is not supported on sm_100

If you write a score/conversion kernel that packs fp8 → bf16 pairwise,
the hardware `cvt.rn.bf16x2.e4m3x2` PTX instruction is **sm_89 / sm_90a
only** (Ada, Hopper). On sm_100 (Blackwell) `ptxas` rejects it with
`Feature 'cvt.bf16x2.e4m3x2' not supported on .target 'sm_100'`. Fall
back to per-element conversion:

```cuda
__nv_fp8_e4m3 fp8_val;  // raw bit pattern in .__x
fp8_val.__x = ...;
__nv_bfloat16 bf = __float2bfloat16(static_cast<float>(fp8_val));
```

The compiler fuses this chain into native Blackwell cvt ops; measurable
overhead vs a hypothetical hardware pairwise cvt but not catastrophic.
The PTX gap is worth knowing on its own when porting fp8↔bf16
score/conversion paths from earlier-arch kernels.

### Configuration & language binding

- **bfloat16 intrinsics**: The TVM FFI builder does not add `-D__CUDA_NO_BFLOAT16_CONVERSIONS__`, so `__nv_bfloat162float` and related intrinsics work out of the box. If you hit bfloat16 conversion errors, make sure you're using `language = "cuda"`, not `language = "python"` with `torch.utils.cpp_extension` (which adds that define by default).
- **Mixed Triton + CUDA**: Use `language = "python"` with `kernel.py` that imports both Triton and loads CUDA via `cpp_extension`. Place `cpp_extension.load_inline()` at module level (outside `run()`) so the `.so` is compiled once and available across benchmark subprocess invocations.
- **`load_inline(name=...)` caches by name, NOT by source**: `torch.utils.cpp_extension.load_inline` writes the compiled `.so` to `~/.cache/torch_extensions/<name>/` and on subsequent imports re-uses whatever is there if the **name** matches. If you edit the CUDA source but keep the same `name=`, the cache may still serve the *old* binary. Two ways this bites:
  - **When preserving multiple variants side-by-side** (e.g. comparing your current `solution/kernel.py` against a prior variant in `docs/prior/variants/`), give each variant a unique `name=` so they don't alias each other (e.g., `name="radix_topk_graph"` vs `name="radix_topk_no_graph"`).
  - **When iterating in a single Python process** (unusual inside the isolated-runner bench, but common in ad-hoc debugging), bump the `name=` after each meaningful source change (or `rm -rf ~/.cache/torch_extensions/<name>/`) to guarantee a fresh compile.
  Benchmarks run in fresh subprocesses (see `use_isolated_runner`), so this doesn't affect `bash scripts/bench.sh`, but it does affect interactive REPL / notebook use.
- **Which path to use**: Pure CUDA → `language = "cuda"` (TVM FFI, recommended). Need PyTorch ops or mixed Triton + CUDA → `language = "python"`.

### `__launch_bounds__` to lift the Triton register-spill ceiling

For register-resident reduction kernels with bounded per-thread state, an
explicit `__launch_bounds__(threads_per_block, min_blocks_per_sm)` pin lets
NVCC commit to your occupancy target. This is a CUDA-only lever: Triton's
internal allocator may over-commit and spill before the same shape spills
under NVCC + an explicit cap.

Example: a Triton kernel at a given tile may spill (Triton's allocator
predicts spill and shrinks); the same shape under NVCC with
`__launch_bounds__(128, 4)` — meaning "at least 4 blocks/SM, so cap at
65536/(128*4) = 128 regs/thread" — can fit naturally well under that
ceiling without spilling.

When this matters: shapes where Triton declines to grow tile width because
its allocator predicts spill, but the actual register count under NVCC fits
comfortably under a chosen `min_blocks_per_sm`. The pin is a *minimum*
occupancy hint, not a register cap — NVCC will use fewer regs than the
ceiling if the kernel naturally fits, raising occupancy further. Pairs
naturally with a CTA-count target documented in the relevant per-family
`TRAPS.md`.

## Per-call overhead: audit GPU↔CPU syncs

When the bench reports a per-call overhead floor (the gap between Python
entering `run()` and the first GPU instruction), auditing pre-kernel
GPU↔CPU syncs is often the highest-leverage fix. Each of the following
inserts an implicit `cudaStreamSynchronize` when called on a CUDA tensor:
`.item()`, `.cpu()`, `.tolist()`, `bool(t)`, `int(t)`, `float(t)`, any
indexing that would read a value into Python (`t[0]` on a 0-d, `len(t)`
on certain types, etc.). Each sync drains the launch queue and stalls the
next kernel until **all prior work** on the stream finishes. Audit any of
these in the per-call hot path before shipping a solution that's
small-shape sensitive.

Common pre-kernel sync hotspots:

- `total = chunk_offsets[-1].item()` to know how to size a tensor →
  use a CPU-computed upper bound (e.g. `NT_max = T // BT + N`),
  sentinel-pad the tensor, early-return on the sentinel inside kernels.
- `cu_seqlens.cpu().tolist()` for grid construction → compute
  per-program via a tiny GPU helper kernel that reads `cu_seqlens`.
- `if some_tensor.item() > 0:` → guard with a sentinel-friendly kernel
  branch instead.

## Capture stable shapes into a CUDA graph

The benchmark calls `run()` many times per workload with **stable
shapes and stable cu_seqlens** (exact iteration count comes from
`warmup_runs` + `solution_iterations` × `num_trials` in `[benchmark]`;
see the `benchmark` skill). If your solution issues ≥2
sequential CUDA / Triton launches per call, you can capture the launch
sequence into one graph per shape tuple and replay for the remaining
calls — collapsing 2-3+ launch round-trips into one.

Pattern (existing variants under `docs/prior/variants/`, when your
operator has a prior archive, may demonstrate this):

```python
_GRAPH_CACHE = {}

def run(*inputs):
    key = (T, num_seqs, has_state, scale)  # shape signature
    if key in _GRAPH_CACHE:
        g = _GRAPH_CACHE[key]
        for name, t in zip(("q", "k", ...), inputs):
            g[name].copy_(t, non_blocking=True)
        g['graph'].replay()
        return g['output'].clone(), g['new_state'].clone()
    # First call: allocate static buffers + capture
    ...
```

**Anti-pattern**: do NOT capture if shapes vary per call — capture cost
(~few ms) exceeds replay savings, and `_GRAPH_CACHE` would grow
unboundedly. Add an `os.environ.get('NO_GRAPH')` gate so NCU profile
runs can bypass the graph (graph replay blinds NCU to per-kernel
attribution).

**Note on `.clone()` and `.copy_()`.** CUPTI span counts every memcpy
as a GPU activity. The two `.clone()` calls in the snippet above are
2 memcpys per iter that the bench harness's per-iter timer discards
each loop (see the `benchmark` skill's workload model). When
the harness's output-consumption contract permits it, returning
static-buffer refs without `.clone()` is safe and drops them from
span. Likewise, when input tensor identities are stable across iters
within a trial, comparing `(t.data_ptr(), …)` to the last-call tuple
lets you skip the per-iter input copies. See the per-op TRAPS in
`docs/prior/<family>/TRAPS.md` for the safety argument.

## Program-Dependent Launch (PDL) for kernel→kernel overlap

Hopper sm_90+ exposes PDL — a hardware scheduling attribute that lets
a consumer kernel's blocks start executing before its producer has fully
finished, provided the consumer block's prefix only touches memory the
producer hasn't written yet. Producer emits a trigger at its tail;
consumer waits at its first load of producer-written data. Works under
`torch.cuda.graph` capture. Useful for reducing the ~0.5 µs-per-kernel
launch overhead on graph replay when multiple kernels chain in the
per-call hot path.

Per-DSL device + host bindings differ — see the Triton / TileLang /
CuTe-DSL skills for the language-specific intrinsics and host-flag
semantics; the generic decision logic below applies to all.

**When PDL helps:** producer grid ≪ SM_count — idle SMs absorb consumer
blocks during producer tail, hiding launch overhead. Per-operator
evidence (variant + measured win) lives in the relevant
`docs/prior/` (when your operator has a prior archive).

**When PDL regresses:** producer saturates SMs at 1 block/SM (shared-mem
or register limited). Consumer blocks dispatched by PDL land on SMs where
producer blocks haven't released, contending for L1/shared/registers
instead of overlapping.

**Diagnostic before adding PDL:** run NCU, check `Waves Per SM` on the
producer.

| Waves Per SM | Action |
|---|---|
| < 0.5 | PDL trigger is pure win; enable both sides. |
| ~ 1.0 w/ producer at 1 block/SM | **Do NOT trigger** — consumer contends. Host-side launch attribute alone on consumer is neutral-to-small-positive. |
| > 1.5 | PDL trigger safe; later waves of producer overlap with consumer. |

**Overlap window:** place the consumer's wait as late as possible —
after address / constant arithmetic but before the first load of
producer-written data. The interval between consumer kernel start and
the wait is where the consumer prefetches its own constants while
producer tail drains.
