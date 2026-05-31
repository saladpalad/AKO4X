# CuTe DSL (NVIDIA CUTLASS Python DSL)

Python-defined kernels that compile to PTX at runtime. Requires `nvidia-cutlass-dsl` package. Install in the Modal image with `--no-deps` (see the build-image commit history) to avoid downgrading `apache-tvm-ffi` and breaking TileLang.

Single file `solution/kernel.py` with the standard `run()` interface:

```python
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def _my_kernel(X: cute.Tensor, Y: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    # ... per-thread logic ...

@cute.jit
def _my_jit(X: cute.Tensor, Y: cute.Tensor, n: cutlass.Constexpr):
    _my_kernel(X, Y).launch(grid=((n + 127) // 128, 1, 1), block=(128, 1, 1))

@torch.no_grad()
def run(input, weight):
    output = torch.empty_like(input)
    _my_jit(from_dlpack(input), from_dlpack(output), input.shape[0])
    return output
```

## Notes

- **Tensor conversion**: `from_dlpack(torch_tensor, assumed_align=N)` — default alignment is element-size (2 bytes for bf16). Required `assumed_align=16` for `cp.async` 128-bit copy atoms, otherwise ncu fails with `src ptr alignment (16 bits) does not meet requirement (128 bits)`.
- **Probing that the runtime works**: an `import cutlass.cute` failure surfaces opaquely through the bench harness (see the `benchmark` skill's status-enum section for why import errors become `COMPILE_ERROR`). Fail-loud inside the kernel (e.g., `assert _CUTE_OK` after an `import … except` guard) to surface the real error message instead.
- **Decorator APIs actually used** (`nvidia-cutlass-dsl >= 4.3.4`):
  - `@cute.kernel` — the device-side function
  - `@cute.jit` — the host-side JIT launcher that calls `.launch(grid=..., block=...)`
  - `cute.arch.thread_idx()`, `cute.arch.block_idx()`, `cute.arch.sync_threads()`, `cute.arch.cp_async_commit_group()`, `cute.arch.cp_async_wait_group(N)` — low-level device intrinsics
  - `cute.math.exp`, `cute.math.log`, `cute.math.exp2` — **note**: `cute.arch` does **not** have `log`

## Probing available API (don't trust old docs)

The CUTLASS DSL `docs/` lags the actual installed wheel, and variant
headers from earlier sessions can be out of date. **Before concluding a
primitive is missing, grep the installed package source**:

```bash
# List all launch-config kwargs available in your installed version
grep -A 20 "class LaunchConfig" $(python3 -c "import cutlass, os; print(os.path.dirname(cutlass.__file__))")/base_dsl/dsl.py

# Confirm specific primitive availability
python3 -c "import cutlass.cute as cute; print([x for x in dir(cute) if 'make' in x])"
python3 -c "from cutlass.cute.nvgpu import warp, cpasync; print([x for x in dir(warp) if 'Mma' in x])"
```

**Available `kernel.launch(**kwargs)` options on `nvidia-cutlass-dsl ≥ 4.3.4`**:

| kwarg | Effect |
|-------|--------|
| `grid=(x,y,z)`, `block=(x,y,z)` | standard CUDA launch |
| `cooperative=True` | all blocks resident simultaneously, enables grid-wide sync |
| `cluster=[x,y,z]` | Hopper/Blackwell block cluster |
| `use_pdl=True` | Program Dependent Launch — blocks start before prior kernel finishes; races data-dependent reads, test carefully |
| `min_blocks_per_mp=N` | launch_bounds hint for compiler |
| `dynamic_shared_memory_size=N` | explicit dynamic SMEM size |

`use_pdl=True` is the CuTe PDL binding; generic PDL theory + the
Waves-Per-SM decision table are in the `cuda` skill ("Program-Dependent
Launch (PDL) for kernel→kernel overlap").

**Launch-kwarg gotchas:**
- `cluster_size_x=N` kwarg does NOT work; use `cluster=[x,y,z]` as a list.
- `cooperative=True` was marked as missing/needing GlobalBarrier in pre-v6 LESSONS —
  **that was wrong**. The kwarg alone works for single-kernel coop launches.
  Grid-wide sync between waves within a cooperative kernel still needs a barrier
  (custom atomics-based `GlobalBarrier` from the Ampere example).
- Max cooperative grid = per-SM limit × num SMs. B200: 148 SMs means ≤148 blocks
  for true-coop residency (larger grids fall back to normal launch).

## Blackwell donor starting points (tcgen05)

For any Blackwell tcgen05 kernel (GEMM, attention, mamba2-SSD), do not
write tcgen05.mma + TMA + cluster pipelining from scratch — vendor and
customize CUTLASS's reference kernels. They live in **two** mirrored
locations on the Modal CI image:

- `thirdparty/cutlass/examples/python/CuTeDSL/blackwell/` (cutlass git
  checkout — the source paths the upstream docs link to)
- `<flashinfer-install>/data/cutlass/examples/python/CuTeDSL/blackwell/`
  (bundled inside the `flashinfer` wheel — accessible at runtime via
  `os.path.dirname(flashinfer.__file__) + '/data/cutlass/...'`. Use this
  path when probing donor APIs from inside `solution/kernel.py`, since
  `thirdparty/` is not on the kernel-image search path.)

| File | Use when |
|---|---|
| `dense_gemm.py` (~1900 lines) | Pure GEMM, correctness-first. TMA-pipelined load + tcgen05 MMA + 2cta optional. Simplest pipeline. |
| `dense_gemm_persistent.py` (~2200 lines) | Pure GEMM, higher perf. Adds persistent tile scheduling + warp specialization. Typically +3-5% absolute score over non-persistent on huge-M. |
| `fmha.py` (~3100 lines) | Fused multi-head attention forward. Capped at D_head ∈ {32, 64, 128}; rejects D_head=512 (MLA). Use for standard MHA / GQA, not MLA. |
| `fmha_bwd.py` | FMHA backward; pair with `fmha.py` for training kernels. |
| **`mla.py` (~5200 lines)** | **Multi-Head Latent Attention — the right starting point for any MLA-family operator on Blackwell.** Implements `cluster_shape_mnk=(2,1,1)` + `use_2cta_instrs=True` + `latent_dim=512` + `rope_dim=64` + `mma_qk_tiler_mn[0]=128` (per-CTA M=64). TMA + warp-specialized + persistent + split-KV + page-table + variable-seq, all wired. The class is `BlackwellMultiHeadLatentAttentionForward`; its `can_implement(B, K, H, L, R, in_dtype, out_dtype, acc_dtype, lse_dtype, mma_qk_tiler_mn, mma_pv_tiler_mn, split_kv, is_persistent, is_cpasync, is_var_seq, is_var_split_kv, use_page_table, page_size) -> bool` staticmethod is a cheap pure-Python probe — call it before committing to a port. **Caveats**: decode-only signature (q_len=1 per batch via num_head packing); dtype allowlist accepts only `Float8E4M3FN` / `Float16` (no bf16); `H<128` ⇒ `split_kv` must be 1; the cpasync path forces H=128 so for TP-split MLA (H<128) the TMA-without-page-table path is the clean fit. |
| `mamba2_ssd/` | Mamba2 state-space model fused-attention donor. |
| `mixed_input_fmha/`, `mixed_input_gemm/`, `blockwise_gemm/` | Specialized variants — only relevant if your operator already matches their shape. |

Strip the host harness (everything from `def run(` / `def run_dense_gemm(` onward) and import the kernel class from your `solution/kernel.py`. GEMM donors write to `(M, K, L), (N, K, L), (M, N, L)` layouts with L=1 — use `t.unsqueeze(-1)` on 2D torch tensors before `from_dlpack`. With `use_tma_store=True`, OOB tiles are allowed (input dimensions don't need to be multiples of the cta tile). MLA donor writes to `[num_head, latent_dim, batch_size]` (decode-style); for prefill, pack `BLOCK_Q * num_head` into the donor's `num_head` dim.

**`PersistentDenseGemmKernel.__call__` signature gotcha**: takes `(a, b, c, max_active_clusters, stream)` — five args at *compile* time via `cute.compile(...)` — but the *runtime* call from the compiled object drops `max_active_clusters` since it's `cutlass.Constexpr`. Compute it as:

```python
hw = cutlass.utils.HardwareInfo()
max_active_clusters = hw.get_max_active_clusters(
    cluster_shape_m * cluster_shape_n
)
compiled = cute.compile(gemm, a, b, c, max_active_clusters, stream)
compiled(a, b, c, stream)  # NB: 4 args at runtime, not 5
```

**Tile-shape prior for fp16 GEMM huge-M (K ≈ 4096, N ≈ 2048 on B200)**:
`mma_tiler=(256, 256)`, `cluster=(2, 1)`, `use_2cta_instrs=True` beats
`(128, 256)` by ~2% and beats `(256, 128)` by ~6% on huge-M average.
Smaller `cta_tile_N` starves MMA throughput; max-legal `(256, 256)`
amortizes per-cta sync best. Cluster `(2, 2)` (A multicast on N) is wash
overall — only helps M > ~14000 where A exceeds the B200 ~120 MB L2 cap.
Use this as a starting prior on similar fp16 GEMM shapes; ablate around
it if N or K differs materially.

**MLA donor empirical findings** (B200, `nvidia-cutlass-dsl ≥ 4.3.4`):
the `can_implement(...)` staticmethod accepts axes `latent=512`,
`rope=64`, `cluster=(2,1,1)`, `use_2cta_instrs=True`,
`mma_qk_tiler_mn[0]=128` (→ per-CTA M=64) for `Float16` /
`Float8E4M3FN` only — bf16 is allowlisted out (one-line patch site is
in `mla.py` around line 4279). For TP-split MLA (H<128 query heads) the
cpasync path is unavailable (`H<128 ⇒ is_cpasync must be False`), so
the TMA-without-page-table path is the clean fit; `split_kv` must be 1
when H<128. A pure-Python `_cute_donor_probe()` (call `can_implement`
with your axes, no GPU touch) is a cheap import-time check before
committing iters; for the BUILD-layer probe call `cute.compile(donor,
...)` against your shape with a bf16-allowlist monkey-patch — both
patterns have shipped successfully on 38/38 Modal workers at our axes
with bf16 inputs.

**Two BUILD-layer gaps not gated by `can_implement`**:
(a) `can_implement` accepts `mma_qk_tiler_mn=(128,64) + mma_pv_tiler_mn=(128,64)` but the constructor then derives `iterations_pv_k = mma_qk_tiler[1] // mma_pv_tiler[2] = 64 // 128 = 0`, and `cute.compile(...)` fails at trace with `ValueError: Expected size in shape to be strictly positive, but got 0`. Use the donor's own CLI defaults — `mma_qk_tiler_mn=(128,128)`, `mma_pv_tiler_mn=(128,256)` (mla.py:5034/5040) — as the BUILD-safe starting config. The gate is incomplete; a `can_implement = True` does not guarantee a `cute.compile` will succeed.
(b) The `_cute_donor_probe()` template prints to stderr via `print(..., file=_sys.stderr)`. The isolated-runner discards each worker's stderr for PASSED workloads by default — meaning the probe runs silently and the trajectory has no `[probe]` markers. Add `--capture-logs` to the bench command (`bash scripts/bench.sh --label <iter-N> --capture-logs`) for any iter that adds or modifies a forensic probe; the markers then land under `results.json[results][<def>][<uuid>][log]`.

**Architectural caveats for adapting the MLA donor to prefill**: the donor is a **decode** kernel with `is_causal=False` baked into both its reference invocation and its softmax warp structure — no causal-mask path exists anywhere in its ~5200 lines. Injecting causal-prefill semantics requires modifying the warp-specialized softmax interlude (compute warps 0-3 in the `@cute.kernel` body around mla.py:836); this is kernel surgery, not adapter work. Additionally, the donor's scheduler dispatches per-batch (not per-(batch, q_block)) so prefill's per-q-block kv-end variation (`prefix_len + q_block_start + BLOCK_Q`) isn't expressible without rewriting the persistent + tile-scheduler. If you're porting for **decode**, the donor is adapter-ready; for **prefill**, budget ~10-20+ iters for the softmax-warp surgery.

## Gotchas / known-broken patterns

API and behavior surprises that have cost multiple iterations. Re-verify
if the toolchain changes.

- **Standalone `@cute.kernel` doesn't `torch.cuda.graph`-capture.** Symptom:
  `UserWarning: The CUDA Graph is empty. ... captured on wrong device or
  stream` + CUPTI fails with `No kernel activities recorded`. The TVM-FFI
  environment-stream CuTe DSL uses for stream binding doesn't pick up the
  graph-capture stream. **Workaround:** launch a Triton or CUDA kernel as
  **anchor** first inside the same `torch.cuda.graph(g)` block; subsequent
  CuTe DSL launches inherit the capture stream with ~0 µs overhead.
  ⚠️ **Second-order gotcha:** the
  anchor workaround **lets the CuTe kernel execute during the `with
  torch.cuda.graph(g):` block**, but the `.launch()` call **does NOT get
  recorded into the graph**. On subsequent `g.replay()`, the CuTe kernel
  **does not run** — only the anchor does. (One concrete instance of the
  general silent-kernel-skipping failure mode; the detection recipe in
  the `bench` skill applies to any suspected case.)
  Symptom cascade:
    1. Replay writes nothing to the output tensor that CuTe was supposed
       to populate; the buffer keeps the value from the one-time CuTe
       launch at capture time.
    2. The full silent-skip cascade (why this passes correctness silently
       and inflates the headline) lives in the `benchmark` skill
       under "Silent-skip cascade".
  **Detection:** confirm with the zero-output / poison-cell /
  varying-inputs tests in the `bench` skill ("Silent kernel skipping
  under graph capture") — the single-source detection recipe.
  **Until `nvidia-cutlass-dsl` upstream fixes `@cute.kernel.launch()` to
  respect capture mode, do not combine `@cute.kernel` with
  `torch.cuda.graph` for any kernel that is the output writer.** If you
  must, always validate via NCU (`scripts/profile.sh`) AND run at least
  one of those detection tests before trusting the speedup.
  *(filed as flashinfer-bench issue #414.)*
- **SMEM alignment for MMA is NOT satisfied by `alloc_smem(alignment=...)`
  alone.** `cute.arch.alloc_smem(BFloat16, n_elems, alignment=128)` aligns
  the base pointer, but `cute.make_tensor(ptr, cute.make_layout((H, D)))`
  gives a plain row-major view where per-element access stride × width
  is 16 bits (bf16). The MMA's `ldsm` atom requires 128-bit alignment
  per-load-unit, which plain row-major does not provide. **Symptom:**
  `'cute.copy' op src ptr alignment (16 bits) does not meet requirement
  (128 bits)` on the first `cute.copy(LdMatrix8x8x16bOp, ...)`. `Copy
  UniversalOp` fallback hits the same root cause at llvm level.
  **Fix:** use a **swizzled composed layout** for any SMEM tensor fed to
  MMA:
  ```python
  atom = cute.make_composed_layout(
      cute.make_swizzle(3, 3, 3),
      0,
      cute.make_layout((8, 64), stride=(64, 1)),
  )
  sQ_layout = cute.tile_to_shape(atom, (H, DKV), (0, 1))
  ```
  See `thirdparty/cutlass/examples/python/CuTeDSL/ampere/flash_attention_v2.py`
  lines 225–241 for the full pattern. Swizzle is **non-optional** for
  bf16 MMA with D_head ≥ 64.
- **`tensor[i].llvm_ptr` fails on SMEM atomics.** `cute.make_tensor(...)[i]`
  returns a value, not a `Pointer`. `cute.arch.atomic_add(ptr=x.llvm_ptr, ...)`
  needs a real `Pointer`. **Workaround:** keep the raw return of
  `cute.arch.alloc_smem(...)` (which IS a `Pointer`) and use pointer
  arithmetic: `(alloc_smem_ptr + i).llvm_ptr`.
- **`while` loops blocked by the DSL preprocessor.** Runtime loops must use
  `for it in cutlass.range(0, n_iters)` with an explicit `if i < N:` guard
  on OOB lanes. Compile-time loops use `cutlass.range_constexpr(N)`. Plain
  Python `range(N)` inside `@cute.kernel` / `@cute.jit` works but is
  treated as `range_dynamic`.
- **`cute.arch.load(ptr)` requires `dtype` positional arg.**
  `TypeError: load() missing 1 required positional argument: 'dtype'`.
  Call `cute.arch.load(ptr, cutlass.Int32)` (or the relevant Numeric type).
- **Int16 → Int32 conversion sign-extends.** `cutlass.Int32(int16_val)`
  preserves the sign bit; for data you've reinterpreted as `int16` but
  that's logically unsigned (e.g. bf16 bits via `.view(torch.int16)`),
  mask with `& cutlass.Int32(0xFFFF)` before bucket-shifting or comparison.
  Without the mask, "negative" int16 values give huge bucket indices →
  SMEM OOB writes → next batch's output slice gets corrupted.
- **Hillis-Steele warp scan via `shuffle_sync_up` + `if lane >= offset: x = x + y`
  produces INCORRECT prefix sum.** Verified on two unroll styles
  (`cutlass.range_constexpr(5)` inner loop and manually-unrolled 5 levels).
  Root cause not fully isolated — likely a CuTe DSL SSA-rebinding issue
  with conditional accumulation across shuffles. **Workaround:** fall back
  to single-thread serial scan (256 iters on `tid == 0`); the per-call
  overhead on a 256-bucket histogram is in the sub-µs range and
  correctness is reliable.
- **No `__match_any_sync` equivalent in `cute.arch`.** Blocks porting
  warp-cooperative histograms that rely on grouping same-value lanes.
  Per-thread atomic is the working fallback; an N-way `vote_ballot_sync`
  per known bucket value works only when the value domain is narrow AND
  statically known.
- **No early `return` inside `@cute.kernel`.** `DSLAstPreprocessorError`.
  Wrap the whole body in an `if <in_range>: ...` guard, or use
  `if sl <= TOPK: <short-path> else: <main>`.

`config.toml` schema (which `language` value to set, `entry_point` format) is centralized in the `benchmark` skill.
