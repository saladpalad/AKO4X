# Triton

Single-file `solution/kernel.py`: `@triton.jit` kernels launched from a
`@torch.no_grad()` `run(*inputs)` that returns the output (or writes the
trailing arg under `destination_passing_style`). Grid is the usual
callable/tuple: `_kernel[(grid,)](…, BLOCK=…)`.

## Autotune Tips

- **See selected configs**: A host-shell `TRITON_PRINT_AUTOTUNING=1` does **not** reach a Modal worker — `scripts/run_modal.py` dispatches into a remote container and only sets that env var when `capture_autotune=True` is passed (which only `--variance-check` does). Two working paths on the modal backend:
    1. `bash scripts/bench.sh --variance-check 2 --first 1` — auto-enables autotune capture and prints the picked-config digest line in the variance summary.
    2. Set `os.environ["TRITON_PRINT_AUTOTUNING"] = "1"` at the top of `solution/kernel.py` (it runs *inside* the worker, so the env var IS in the right process), then bench with `--capture-logs` — the autotune trial lines + "best config" land in `trajectory/*/results.json[…]["log"]` (≤20 KB).

  On the local backend the original `TRITON_PRINT_AUTOTUNING=1 bash scripts/bench.sh` still works (no cross-process boundary).
- **Variance from re-selection**: If benchmark scores fluctuate (>5%), check whether autotune is picking different configs across runs. If so, hardcode the best config and remove `@triton.autotune`.
- **In-place / DPS kernels**: Autotune trials reuse the same input tensors. If the kernel writes in-place, earlier trials corrupt inputs for later ones — use `restore_value` on the affected pointer arguments.
- **Stale compilation cache**: Benchmark and profiler use a project-local Triton cache (`.triton_cache/`). If you suspect stale compiled kernels, `rm -rf .triton_cache/` to force recompilation.

## `num_warps` can degrade MMA throughput at small-N fp8 tiles

Lowering register pressure by raising `num_warps` is the natural move
when NCU says a kernel is register-limited to 12.5% theoretical
occupancy. It can backfire silently on `tl.dot`s with fp8 operands at
small-N tiles: Triton's MMA tile-picker selects a *different* primitive
at different warp counts, and the primitive picked at `num_warps=4`
can run at lower per-cycle throughput than the one at `num_warps=2`.

Example pattern (fp8 `tl.dot` at tile M=N=64, K=128):

| num_warps | regs/thread | theoretical occ | NCU duration |
|-----------|-------------|-----------------|--------------|
| 2         | high        | low (~12%)      | shorter      |
| 4         | low         | high (~44%)     | **longer**   |

Despite the theoretical occupancy headroom at `num_warps=4`, achieved
occupancy barely moves and NCU duration goes UP. Dynamic smem / block
doubles because Triton's pipeliner allocates a larger double-buffer
for the different MMA shape.

**Why:** Triton's MMA picker maximizes per-warp throughput given
`(M, N, K, warp_count)`. For M=N=64 fp8 at `num_warps=4`, each warp
gets a 16×16 sub-tile instead of 32×32; the per-cycle throughput of
the smaller primitive is lower on B200 even though more warps are in
flight. Achieved occupancy tracks the slower primitive's stalls, not
the occupancy headroom.

**How to verify before trusting a `num_warps` prediction:**
1. Capture the autotune-picked config — `bash scripts/bench.sh --variance-check 2 --first 1` on modal (or the host-env-var form on local); see "See selected configs" in `## Autotune Tips` above for both backends.
2. Dump the compiled PTX from `.triton_cache/*/..._kernel.ptx` and
   grep the `mma.sync` mnemonic — a change in the `.m*n*k*` suffix
   between num_warps values is the fingerprint that the tile-picker
   moved.
3. NCU `SchedulerStats` + `WarpStateStats`: if stall reasons shift
   toward `stall_mma_throttle` / `stall_mio_throttle` rather than
   dropping, the MMA primitive regressed and the occupancy gain is
   net-negative.

## Split-K reduce kernels: prefer the 2D-tile merged form over scalar `tl.static_range`

When writing the *second* kernel in a split-K → combine pipeline (the "reduce" that
merges per-split `(m, l, O)` into the final output), the two common shapes are:

```python
# Split form: per-iter scalar exp + scalar PM reload
for si in tl.static_range(0, NS):
    w_s = tl.exp(tl.load(pm_ptr + ... + si * stride_pm) - m_g)
    o_s = tl.load(po_ptr + ... + si * stride_po + d).to(tl.float32)
    acc += w_s * o_s

# Merged 2D-tile form: one load, one reduction
s = tl.arange(0, NS)
po_tile = tl.load(po_ptr + ... + s[:, None] * stride_po + d[None, :]).to(tl.float32)
acc = tl.sum(w[:, None] * po_tile, axis=0)  # w = tl.exp(m_vals - m_g) pre-computed as vector
```

**The 2D-tile form generally wins on Triton** — Triton's IR exposes enough dataflow
that the compiler schedules cp.async-streamed loads with per-thread partial
accumulation; there's no need to hand-split into separate passes the way CuTe DSL
sometimes rewards.

**Register-budget ceiling**: the tile must fit per-block registers. Rough guide on
CUDA 13.2 + Triton 3.6.0 (B200):

- `NS × D_CHUNK × sizeof(fp32) ≤ 8 KB` — safe.
- `≈ 8–16 KB` — verify via `bash scripts/bench.sh --extremes` (modal-only; on local backend use `--group` with the min/max axis values).
- `> ~16 KB` — assume spill; shrink `D_CHUNK` (increase grid) or keep the scalar
  `tl.static_range` form.

**Store-coalescing floor (output dtype matters)**: the register-budget ceiling is
an upper bound on `D_CHUNK`; there's also a lower bound when the reduce output is
narrow (bf16 / fp16) and stored with row-stride equal to a wide head dim
(`D_CKV`-style). Each per-CTA store is `H × D_CHUNK × sizeof(output)` bytes laid
out as `H` rows of `D_CHUNK × sizeof(output)` bytes each. When the per-row size
drops below ~64 B the stores fragment into sub-128 B partial-cache-line writes
and large-batch reduce throughput collapses (observed >20% regression at
borderline ~32 B vs ~64 B per row in bf16 reduce kernels on B200, even when the
tile cleared the register-budget ceiling). Aim for `D_CHUNK × sizeof(output) ≥
64 B` per row, and prefer 128 B (one full cache line) when register budget
allows. The per-CTA-store size is what matters — making the **grid larger**
while keeping the same total bytes does NOT recover coalescing.

**Warp count**: for tiny tiles prefer `num_warps=1` with `D_CHUNK` equal to the warp
size (32), giving exactly one output element per thread. This maximizes block count
on small-grid reduces where `grid ≪ SM_count` — useful when the producer can't
fill the SMs and you want PDL-style overlap with the consumer.

## Program-Dependent Launch (PDL) — Triton bindings

Generic PDL theory — when it helps vs regresses, the Waves-Per-SM
decision table — is in the `cuda` skill ("Program-Dependent Launch
(PDL) for kernel→kernel overlap"). This section is only the Triton
binding. Triton 3.6+ exposes PDL intrinsics in
`triton.language.extra.cuda` (the project pins only `triton>=3.5.0`,
so confirm the import resolves before relying on it):

```python
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

@triton.jit
def producer(...):
    # ... compute + all tl.store ops ...
    gdc_launch_dependents()   # LAST statement of kernel body

@triton.jit
def consumer(...):
    # ... address/constant setup (pre-loaded work) ...
    gdc_wait()                # BEFORE first load of producer's output
    # ... load producer outputs + compute ...
```

Host-side: pass `launch_pdl=True` at the call site. Without it the
device-side intrinsics are no-ops (the kwarg sets
`CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION` via
`cuLaunchKernelEx`):

```python
producer[grid](..., launch_pdl=True)
consumer[grid](..., launch_pdl=True)
```

Works under `torch.cuda.graph` capture.

**Overlap window:** place `gdc_wait()` as late as possible in the
consumer — after any constant/address arithmetic but before the first
load of producer-written data. The interval is where the consumer
prefetches its own constants while producer tail drains.

## Triton 3.6 advanced-load levers (warp_specialize, tensor_descriptor)

`tl.range(..., warp_specialize=True)` and `tl.make_tensor_descriptor(...)`
(the TMA path) look like obvious levers for memory-latency-bound kernels.
In Triton 3.6 they are restricted in ways the surface API doesn't
advertise; smoke them before structuring a round around either.

### `warp_specialize=True` — restricted to pure mma-accumulator pipelines

In Triton 3.6 on B200, `tl.range(..., warp_specialize=True)` aborts in
the ttgir pass for any loop body that contains **non-mma operations
between mma calls within the same iteration** — including the natural
flash-attention shape (load → mma(QK) → softmax → mma(PV) → acc), where
the softmax interlude between the two mmas breaks it even though each
mma feeds an accumulator. Observed failure: `RuntimeError: PassManager::run failed`
in `make_ttgir`, with no further diagnostic. It reproduces both with
regular `tl.load` and with TMA `desc.load(...)`. Confirmed failure modes:

  - Reduce-shaped kernels (load → reduce → store) — no mma at all.
  - Flash-attention-shaped (load → mma(QK) → softmax → mma(PV) → acc) —
    the softmax interlude breaks `warp_specialize=True` on Triton 3.6 /
    CUDA 13.2 / B200, even though each mma feeds an accumulator.

Confirmed working: pure mma accumulator loops (`acc += tl.dot(A, B)`
with no intervening non-mma ops). Treat `warp_specialize=True` as
applicable **only** to that pattern — flash-attention, attention-sink,
and any softmax-interrupted matmul accumulator will fail.

### `tl.make_tensor_descriptor` — needs persistent + a fat tile

TMA via `tl.make_tensor_descriptor` issues `cp.async.bulk.tensor` to
SMEM and bypasses L1. It has two costs the docs don't surface:
1. **Per-CTA descriptor setup**. With a non-persistent grid (one CTA
   per tile), creating the descriptor in the kernel body adds enough
   per-CTA overhead to lose against `ld.global.b128` even at HBM-bound
   B. Always amortize via a persistent grid-stride loop.
2. **Small-tile penalty**. The mbarrier-wait on the bulk DMA is a
   serial CTA-level dependency. For tiles ≲ ~16 KB, the in-flight
   issue volume from many concurrent CTAs each running multiple b128
   loads (a row-per-program shape) beats one bulk-DMA-then-wait per
   CTA — even with several `num_stages` of pipelining over the
   descriptor.

Empirical sketch on a streaming-reduction shape (~8 KB tile, large B,
B200 / CUDA 13.2 / Triton 3.6, `bash scripts/bench.sh --extremes`):

- `ld.global.b128` with hw-scheduled CTAs (the default anchor) is the
  fastest variant on small streaming tiles.
- TMA with a non-persistent grid regresses below the non-TMA anchor —
  the per-CTA descriptor setup dominates at this tile size.
- TMA with a persistent grid + `num_stages` pipelining closes most of
  the gap but does not beat a non-TMA persistent grid at the same tile.

TMA's bulk-DMA wins only when the tile is big enough that one DMA-then-
wait per CTA is faster than the many b128 issues the row-per-program
shape needs — i.e., it pays for matmul-shaped tiles (≥ ~64 KB), not for
small (h ≤ 256) streaming reductions. Run the four-variant smoke
yourself if you suspect your shape sits near the boundary.

### Smoke checks before committing a round to either lever

- `python3 -c "import triton.language as tl; print(hasattr(tl, 'make_tensor_descriptor'))"`
  confirms the TMA API is exposed (it is in 3.6, no-op in earlier).
- For `warp_specialize=True`: build the kernel once locally; a
  `PassManager::run failed` at compile time means this Triton version
  doesn't accept your kernel pattern. Don't iterate against it.
- For TMA: a `--extremes` smoke at the largest B is enough to see the
  per-CTA descriptor cliff. If the smoke is ≪ anchor at the largest
  B, don't escalate to a labeled bench.

## Blackwell M=64 fp32-acc — tcgen05/tmem path + count-permanent allocator

Triton 3.6 + CUDA 13.2 on B200 (sm_100) emits two different acc placements
depending on the M dim of `tl.dot`:

| acc shape | placement | budget |
|-----------|-----------|--------|
| M=32 fp32 | registers (+ spill to local mem if D large) | reg budget |
| M≥64 fp32 | tcgen05 **tensor memory** (tmem) | 512 columns ≈ 64 KB |

The tmem path is "the Blackwell native" — what TileLang's default schedule
fails to reach (see `dsa-sparse-attention` / `mla-paged-prefill-causal-h16-ckv512-kpe64-ps1`
archives) and what the Triton M=32 spill path costs you. Reaching it from
Triton requires two things working together:

### 1. D-tile the QK to bypass the big-K wgmma codegen bug

A single `tl.dot(qn, tl.trans(kc))` at shape `(M=64, K=D_LARGE, N=BLOCK_N)`
(D_LARGE ≥ ~256) emits an mma chain whose SMEM descriptor is misaligned —
failure mode: kernel launches, then **CUDA Misaligned Address** at runtime
(BLOCK_N=32 num_warps∈{4,8} is the known repro shape on B200 / CUDA 13.2 /
Triton 3.6 for flash-attention-shaped kernels with D_CKV ≥ ~256). Bypass:
split K into NUM_D_CHUNKS chunks of D_TILE ≤ ~256 and chain `tl.dot(... acc=s)`:

```python
s = tl.dot(qn_c0, tl.trans(kc_c0))
s = tl.dot(qn_c1, tl.trans(kc_c1), acc=s)
# ... NUM_D_CHUNKS times
```

This emits NUM_D_CHUNKS smaller mma's that compile cleanly. The misalignment
is in the SHAPE of the single big mma, not Triton itself.

### 2. Triton's tmem allocator is tensor-count-permanent (no slot multiplexing)

Once the kernel compiles past the D-tile QK, the next gate is tmem capacity.
Triton 3.6's tmem allocator gives **each declared accumulator tensor its own
permanent slot for the kernel's lifetime** — there is no liveness-based slot
multiplexing, even when SSA reads/writes would allow it (e.g. 4 acc fragments
in a D-tiled OV loop that update sequentially per kv iter).

Total tmem usage = sum of declared acc tensor sizes + small overhead. On
B200 the budget is **512 columns** (~ 64 KB).

**Knobs that DON'T multiplex tmem at M=64 fp32 acc** (empirically tested on
a D=512 acc[M=64, D=512] flash-attention shape):

| Knob varied | Tmem cells | Effect |
|---|---|---|
| `BLOCK_N` 32 → 16 | 528 → 520 | -8 (`s` tile only) |
| `D_TILE` 128 → 256 (4 chunks → 2) | 520 → 520 | none — total bytes unchanged |
| `num_warps` 8 → 4 | 520 → 520 | none |
| `num_stages` 2 → 1 | n/a (SMEM lever) | doesn't touch tmem |

Implication: any M=64 fp32-acc attention shape with `M * D_acc * 4B` close
to or exceeding 64 KB will hit `OutOfResources: tensor memory` in Triton 3.6,
no matter how the kernel body is chunked. For a typical MLA-style attention
shape (M=64, D=512), acc alone = 64 KB ≈ full budget → +8 cells of `s`
overhead → 520 > 512 always.

**Workarounds when you need M=64 + D ≥ 256**:
- Drop to bf16 acc via `tl.dot(... out_dtype=tl.bfloat16)` — halves tmem
  but tanks online-softmax precision.
- Drop to M=32 — loses head-grouping wins; revert the D-tile.
- Move to **CuTe-DSL** or **hand-written tcgen05 PTX** — both expose direct
  tmem allocator handles (`cute.arch.alloc_tmem`, `tcgen05.alloc.b32`),
  letting you allocate a smaller tmem region and have multiple acc
  fragments cycle through it as they're produced + consumed by the
  `acc * alpha + new` update. See `cute-dsl` skill "Blackwell GEMM
  starting points (tcgen05)" for the CuTe-DSL recipe.

### Smoke check before committing to a Triton M=64 round

Compile-only probe (no bench needed):

```python
# in kernel body, declare exactly the acc footprint you'd need:
acc = tl.zeros([64, D_ACC_PER_FRAGMENT], dtype=tl.float32)
# ... at least one tl.dot(... acc=acc) to force tmem allocation ...
```

Then `bash scripts/bench.sh --first <wl-that-triggers-this-path>` and check
for `OutOfResources: tensor memory`. If the required cells > 512, the path
is structurally closed in Triton 3.6 — pivot to CuTe-DSL early rather than
spending iters on chunking knobs.

## Grid-axis dispatch order: `tl.program_id(0)` varies fastest

Triton launches the user's `grid = (G0, G1, G2)` as CUDA gridDim
`(G0, G1, G2)`, and the HW scheduler dispatches CTAs in blockIdx.x-fastest
order. That means **the logical dim you bind to `tl.program_id(0)` is the
dim that varies fastest in HW dispatch order** — adjacent CTAs in pid-linear
order share `pid(1)` and `pid(2)`, and run concurrently on adjacent SMs.

Implication for L2-locality reorders: just swapping the grid tuple at the
call site is a no-op if you don't also change which axis `pid(0)` binds
inside the kernel. The CTAs that run concurrently (and therefore benefit
from L2 promotion) are the ones varying `pid(0)` — not the ones varying
the later dims.

Example — flash-decode split kernel, ~4096 CTAs at (B=64, H_KV=8, SPLIT=8):

| `grid` | `pid(0)` binds | Adjacent CTAs share | L2 fanout |
|---|---|---|---|
| `(B, H_KV, S)` | `b` | `(h_kv, s)` — different b → different pages | poor |
| `(B, S, H_KV)` | `b` | `(s, h_kv)` — STILL different b | poor (same as above; no-op) |
| `(H_KV, S, B)` | `h_kv` | `(s, b)` — same kv_indices slice → same pages | **good** |

The third row measures meaningful L2-fanout gain at flash-decode shapes
(`pid(0)` binding to the page-sharing axis lets adjacent CTAs reuse the
same KV pages); the second row is a true no-op (just swapping the grid
tuple without re-binding `pid(0)`).

## Persistent kernels + work-stealing — Triton 3.6 gotchas

Three Triton-3.6-specific behaviors that change the persistent-kernel cost
model from the textbook description. Smoke each before committing a round
to a persistent design.

1. **`return` is rejected inside a `while` or `for` body.** Triton 3.6 emits
   `Cannot have return statements inside while or for statements in triton`.
   The standard persistent skeleton (`while True: tid = atomic_add(...); if
   tid >= N: return`) fails to compile. Use a sentinel flag instead:

   ```python
   active = True
   while active:
       tid = tl.atomic_add(COUNTER, 1, sem='relaxed', scope='gpu')
       if tid >= N:
           active = False
       elif tid < N_PROD:
           ...  # producer branch
       else:
           ...  # reducer branch
   ```

2. **The pipeliner does not carry across outer-`while` iterations.** Each
   task in the persistent loop pays the full `num_stages` cp.async pipeline
   prologue + epilogue. For tasks shorter than ~2× `num_stages` BLOCK_N
   iters, the prologue is a large fraction of per-task wall. Measured
   fingerprint: the same `@triton.jit` kernel body wrapped in
   `grid=(N_TASKS,)` + `program_id`-dispatch is faster than
   `grid=(NUM_SMS,)` + atomic-counter dispatch by ~1.2-1.3× on
   attention-shaped workloads when N_TASKS ≤ a few NUM_SMS. Implication:
   persistent kernels only pay off when each task is BIG enough that
   prologue ≪ useful work, OR when N_TASKS ≫ NUM_SMS and HW dispatch
   headroom is genuinely insufficient.

3. **Spin-wait at `grid = num_SMs` stalls the SM.** A persistent CTA waiting
   on a global counter (`while done < n: done = tl.atomic_add(ptr, 0,
   sem='acquire')`) blocks all warps in that CTA in lockstep on the spin
   branch. With grid=num_SMs (1 CTA/SM) every spinner idles its entire SM —
   measured at -14× on a B=16 decode subset when ~128 reducer CTAs spin
   while ~20 SMs drain the producer queue alone. If you need in-kernel
   producer/consumer sync, **oversubscribe the grid** (e.g. `grid = 4 *
   num_SMs`) so producer warps can co-run on the SM of a spinner; or use
   `cache_modifier='.cv'` on `tl.load` instead of `atomic_add(ptr, 0)` to
   reduce atomic-port pressure (supported in 3.6 but does not solve the
   SM-stall on its own).

See also: `atomic_add(ptr, val, sem=…, scope=…)` is documented in 3.6 with
`sem in {'acquire','release','acq_rel','relaxed'}` and `scope in {'cta',
'gpu','sys'}`.

## Tensor layout: deterministic hcat / vcat via `tl.join` + permute + reshape

Triton's `tl.cat(a, b)` only supports `can_reorder=True` (per its docstring:
"Current implementation of cat supports only can_reorder=True"), meaning
output element order is unspecified. **Do not use `tl.cat` when positional
correctness matters** — it's correct only for later reductions that are
invariant to element order. A small-tile smoke test can pass while a
larger-tile real bench silently misorders.

For deterministic column (hcat) or row (vcat) concatenation of 2D tensors
use `tl.join` (stacks along a new minor dim) + `tl.permute` + `tl.reshape`:

```python
# hcat (column concat): [M, N] + [M, N] → [M, 2N]
joined = tl.join(a, b)                      # [M, N, 2]
perm = tl.permute(joined, (0, 2, 1))        # [M, 2, N]
ab_concat = tl.reshape(perm, (M, 2 * N))    # [M, 2N]  rows preserved, cols [a|b]

# vcat (row concat): [M, N] + [M, N] → [2M, N]
joined = tl.join(a, b)                      # [M, N, 2]
perm = tl.permute(joined, (2, 0, 1))        # [2, M, N]
ab_concat = tl.reshape(perm, (2 * M, N))    # [2M, N]  cols preserved, rows [a;b]
```

Chain these to build larger layouts — e.g. constructing a
block-lower-triangular `A_inv` from per-block tiles to replace many
small matmuls with one large one, lifting m-utilization toward 100%
of the native MMA tile width.

**Register-budget warning**: a built tile of `[64, 64]` bf16 ≈ 8 KB
in flight per CTA. Check occupancy impact if the surrounding kernel
is already register-limited; for kernels running at 4+ CTAs/SM,
verify the build cost doesn't dominate.

**When not to use**: wastes compute on zero-padded blocks when valid
blocks ≪ total. Break-even is roughly
`(valid / total)^2 > issue_ratio`; below that, keep the per-sub-block
form with runtime guards.

`config.toml` schema (which `language` value to set, `entry_point` format) is centralized in the `benchmark` skill.
