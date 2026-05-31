# Cross-variant TRAPS — dsa-sparse-attention

Toolchain / measurement-methodology facts that apply to **every** variant
in this archive, regardless of which one is anchor. Created 2026-04-22
(v10 session); each entry has a **Why** so future sessions can judge
whether their context flips the fact.

---

## Measurement methodology

### AB-compare deltas do NOT compose cumulatively

**Fact:** Two independent AB measurements showing `B − A = +xₐ` and
`C − B = +x_b` do **not** give `C − A = xₐ + x_b` when benched in a
third session. You must run a direct cumulative AB `C − A` in one
Modal container to claim the cumulative delta.

**Why:** Each AB cancels its own session's drift, but carries ~±0.05x
residual noise per run from workload-level variance. Summing N
independent AB deltas accumulates N×0.05x ≈ ±0.15x noise that can
reverse sign or double count. Additionally, optimizations can interact:
a change measured +0.22x on top of A may be only +0.05x (or negative)
on top of B if B already captured part of the win through a different
mechanism.

**Seen in v10 session:** iter-5 AB vs iter-0 = +0.22x (real), iter-10
AB vs iter-5 = +0.23x (real), but direct iter-10 vs iter-0 in a single
AB session = -0.18x. Diagnosed: the iter-10 "merged reduce" win was
measured once under favorable drift; on re-test stacked cumulatively
it actually regressed the Triton-path T=1/2 by ~1x. Corrected in
iter-21 to `+0.27x` (2-run repeat).

**How to apply:** Before claiming a variant wins by "sum of prior
AB deltas," run one direct AB against the earliest reference state
(v5/v6 anchor) in a single container. If the direct sum doesn't match,
trust the direct measurement.

---

### Modal session drift can be ~±1x on this operator

**Fact:** The same unchanged kernel variant benchmarked in two different
Modal containers (fresh allocations, same GPU type) can report absolute
speedup numbers differing by up to ~1x. An unchanged cute_reduce_v6 has
been measured at 75.53x, 75.60x, 75.71x, 75.82x, 76.44x, 76.95x,
108.81x (T=1 single-workload noise) across the v8–v10 sessions.

**Why:** Modal B200 tenancy, driver version rollouts, thermal state,
and CUDA runtime lazy-init timing all drift between container
allocations. The ±1x is an across-container floor; within-container
variance (3-run variance-check) is typically CV ≤ 0.1%.

**How to apply:** Do not compare absolute scores across different
`ako4fib-run-*` sub-envs — ever. Use `--ab-compare` against a saved
trajectory in your own container to reason about deltas. Treat any
standalone single-session delta of `< 1x` as noise; require either
multi-run variance-check agreement or in-session AB.

---

### Chevron CUDA launches need an explicit stream under `torch.cuda.graph` (2026-04-24)

See `../dsa-topk-indexer/TRAPS.md` section "`my_kernel<<<grid, block>>>`
without an explicit stream is silently skipped by `torch.cuda.graph`
capture" for the full mechanism + fix + detection tests.

Orthogonal to the CuTe DSL `.launch()` trap documented below — the
CuTe trap is "TVM-FFI stream binding doesn't pick up capture mode",
the chevron trap is "legacy stream 0 isn't in capture mode." Both
produce the same silent failure mode (kernel runs during capture,
not recorded into graph, output stale on replay, correctness masks
it via fixed-input replay). No current dsa-sparse-attention variant
uses raw chevron launches from `load_inline` code, but if a future
variant introduces one alongside the existing Triton / CuTe DSL
kernels under graph capture, apply the `at::cuda::getCurrentCUDAStream()`
fix pre-emptively.

---

## Grid-size-dependent optimum within the same operator

### Reduce loop-merge sign flips between Triton-path and TileLang-path

**Fact (revised 2026-04-24, v11 session):** The merge-form vs split-form
question is resolved differently for each implementation language:

- **Pure-Triton reduce kernels**: the *merged 2D-tile form* —
  `po_tile = tl.load(...[NS, D_CHUNK])`, `acc = tl.sum(w[:,None] * po_tile, axis=0)` —
  beats the scalar `tl.static_range` split form by +3.37x AB (drift-cancelled)
  on the TRI-path at T=1/2. See `variants/hybrid_2d_reduce/kernel.py` for the
  reference implementation. The merge form **regresses** when applied to the
  TL-path reduce, but for a different reason (register spill; see next entry).
- **CuTe DSL reduce kernels**: the older observation (split on small-grid TRI,
  merge on large-grid TL) is the best documented data we have, but is
  **unverifiable on current stack** pending flashinfer-bench #414 — both sides
  of any v5/v6/v7 AB were equally bugged. Do not treat the CuTe merge/split
  question as settled until upstream lands.

**Why the Triton-path flipped:** The scalar static_range form recomputes
`tl.exp(PM[si] - m_g)` once per iteration (NS scalar loads + NS scalar `exp`
calls). The 2D-tile form loads PM once as a vector, computes `w` once, and
folds the whole accumulation into a single `tl.sum(w[:, None] * po_tile, axis=0)`
that Triton lowers to cp.async-streamed loads with per-thread partial
accumulation in registers. The v5-era "split for nvcc scheduler interleaving"
observation was a **CuTe-DSL codegen artifact** — Triton's IR exposes enough
dataflow that the merged form schedules at least as well as the split form
when the tile fits registers.

**Grid analysis (still accurate):**
- TileLang-path reduce: grid = T × H × d_splits ≈ 6k–8k blocks at T=6–8.
  Each SM holds many blocks; scheduler budget isn't the constraint.
- Triton-path reduce: grid = T × H × d_splits ≈ 64–128 blocks (pre-v11,
  D_CHUNK=128 split). Each SM holds ≤1 block; the 2D-tile merge win at v11
  came from combining `D_CHUNK=32 + num_warps=1` (pushing grid to 256 blocks,
  ≈100% SM occupation) with the per-block latency cut from the merge.

**Seen in:** v5 (split introduced for CuTe), v6/v7 (merge claims — both sides
of CuTe AB unverifiable per #414), v10 (Triton-path merge tried as scalar
`static_range` sharing, regressed; v10 extrapolated the "stays split" rule),
**v11 (2D-tile merge on TRI-path wins +3.37x AB, 55.31x variance-verified)**.

**How to apply:** Test both forms per language and per grid-size bucket.
Source-level rewrites that look equivalent may generate very different SASS
across Triton / CuTe / TileLang; trust only measured AB deltas on the current
toolchain. The 2D-tile merge form is currently the **preferred pattern for
Triton split-K reduce kernels** where the NS × D_CHUNK fp32 tile fits
per-block registers (see the next entry for the budget ceiling).

---

### Triton `tl.load` 2D-tile register budget caps the merge-form ceiling

**Fact (2026-04-24, v11 session):** A `tl.load` of shape `[NS, D_CHUNK]`
followed by `tl.sum(w[:, None] * tile, axis=0)` works well when the tile fits
per-block registers, but spills catastrophically past that ceiling. Empirical
boundary on CUDA 13.2 + Triton 3.6.0 (B200, sm_100):

| Tile size (fp32) | Outcome |
|------------------|---------|
| `[32, 32]` = 4 KB | optimum, +3.37x AB vs scalar split form |
| `[32, 64]` = 8 KB | +2.14x AB (still fits, slightly worse than `[32, 32]`) |
| `[32, 128]` = 16 KB | +1.23x AB (marginal, closer to spill boundary) |
| `[16, 512]` = 32 KB | **register spill** — T=8 collapses 49.5 → 35x |
| `[16, 256]` = 16 KB | **still spills on TL-path** — T=8 → 41x |

**Why:** Triton lowers `tl.sum(w[:,None] * tile, axis=0)` to per-thread
partial accumulation held in registers; an over-budget tile spills to local
memory (thread-private L1), losing far more than the merged form saves on
`exp` calls. The exact threshold depends on block register pressure from
other live tensors, not just tile size — the TL-path observation shows that
16 KB can still spill when the kernel has other fp32 fragments alive
(`w`, `l_vals`, etc.), while a cleanly-written TRI reduce comfortably holds
16 KB.

**How to apply:** Before converting a scalar `tl.static_range` reduce into
the 2D-tile merged form, pre-estimate `NS × D_CHUNK × sizeof(fp32)`:
- ≤ ~8 KB: safe, try it.
- ~8–16 KB: likely works on simple reducers but verify via AB.
- > ~16 KB: assume spill; shrink `D_CHUNK` (increase grid) or fall back to the
  scalar static_range form. If the grid is already saturated (T×H ≥ SM count),
  more grid via smaller D_CHUNK won't help either; keep scalar.

**How to verify cheaply:** `bash scripts/bench.sh --extremes` (T=1 + max T)
usually surfaces a spill within ~1 min — the affected T group's speedup
drops by >5x, unmistakable against drift noise.

---

## Toolchain constraints on this stack (CUDA 13.2, nvidia-cutlass-dsl ≥4.3.4, tilelang)

### TileLang `T.Kernel` does not support cluster launch

**Fact:** `T.Kernel(num_tokens, NS, threads=threads, cluster=(1, 2, 1))`
raises `TypeError: Kernel() got an unexpected keyword argument 'cluster'`.

**Why:** TileLang's DSL does not expose Hopper/Blackwell thread-block-
cluster semantics. Cluster-launched kernels with DSMEM cross-block
communication must be written in pure CuTe DSL.

**How to apply:** For any architectural direction that needs cluster
launch (e.g., the fused fwd+reduce in
`variants/cute_reduce_v7/FUSED_KERNEL_DESIGN.md`), don't try to extend
the existing TileLang fwd. Plan on rewriting the affected kernel in
CuTe DSL from scratch. Budget this as a ~3000-line MMA port (reference:
Blackwell `fmha.py` ≈ 3100 lines).

---

### CuTe DSL `alloc_smem` pointer rejects runtime-tid Python subscript

**Fact:** Attempting to index a pointer returned by
`cute.arch.alloc_smem(cutlass.Float32, N)` with a runtime thread index,
e.g.,

```python
pm_smem = cute.arch.alloc_smem(cutlass.Float32, NS)
if tid < NS:
    pm_smem[tid] = cutlass.Float32(PM[tok, tid, hd])  # COMPILE_ERROR
```

raises `DSLRuntimeError: '<class 'cutlass.base_dsl._mlir_helpers.arith.ArithValue'>' object cannot be interpreted as an integer`.

**Why:** `alloc_smem` returns a low-level `Pointer`, not a `cute.Tensor`.
Python subscript on a `Pointer` requires a compile-time integer; runtime
integer values (`ArithValue`) aren't auto-converted.

**Workaround:** Wrap the raw pointer in a `cute.Tensor` via
`cute.make_tensor(ptr, layout)` with an appropriate layout. Then
subscripting the tensor with runtime indices works as expected.

**How to apply:** Any SMEM-caching optimization in CuTe DSL reduce
kernels (e.g., caching PM/PL before the main loop to share across
warps) needs the `make_tensor` wrapper. Don't use the raw pointer
path — it silently passes through `@cute.jit` compilation up until
the runtime lowering where the error surfaces.

---

### `cooperative=True` and `cluster=(…)` don't compose on kernel launches

**Fact:** Passing both `cooperative=True` and `cluster=(…)` to a
`.launch(…)` call regresses performance by ~0.5x on the reduce kernel
in this operator.

**Why:** Cooperative-launch (grid-wide sync via cooperative groups)
and cluster-launch (GPC-localized DSMEM + cluster_arrive/wait sync) use
different launch mechanisms with different guarantees. Combining them
forces both subsystems to coordinate, and at least in CuTe DSL 4.3.4
on CUDA 13.2 this adds real overhead without adding any capability.

**How to apply:** Pick one. For kernels that don't use either
cooperative grid_sync or cluster DSMEM, any of `cooperative=True` /
`cluster=(…)` / neither (vanilla) are all within ±0.05x noise of each
other on this operator — keep whichever the variant's header
documents.

---

### Prior "cooperative = +0.55x" measurement was session drift

**Fact:** v6 session recorded "+0.55x cooperative reduce launch
(iter 5, 3-run variance-confirmed)". v10 session's direct AB tests
found vanilla ≈ cluster ≈ cooperative, all within ±0.05x of each
other.

**Why:** v6's 3-run variance-check ran in a single Modal container;
within-container CV is ~0.1%. But the "+0.55x" was the delta from
toggling cooperative on/off across two different container sessions,
which have ~±1x drift. The real delta is ≤ 0.05x.

**How to apply:** When re-evaluating a prior session's "+0.Nx from
launch-mode knob", require an in-container AB of the current kernel
with the knob toggled both ways. Don't carry forward drift-masked
wins as kernel requirements.

---

### `@cute.kernel` is not captured into `torch.cuda.graph` — CuTe reduce variants have inflated headlines (2026-04-23)

**Framework-level mechanism** (bench fixed-inputs × silent-kernel-skip ×
CUPTI blindness = inflated headline with passing correctness): see
`templates/benchmark.md` "Silent kernel skipping under graph capture"
for the generalized detection recipe. Entry below is the operator-
specific `@cute.kernel` cause + affected-variants evidence for this
archive.

**Fact:** `@cute.kernel.launch()` does **not** participate in CUDA
graph capture. Calling it inside `with torch.cuda.graph(g):` makes
the kernel **execute immediately** during the capture block, but the
launch is **not recorded into the graph**. Subsequent `g.replay()`
does not run the CuTe kernel. If the CuTe kernel is the final output
writer, `output` retains the stale value from the one-time launch at
capture time.

In `flashinfer-bench`, each workload has fixed inputs across all
measured iterations. Stale output therefore coincidentally matches
the reference, correctness passes, and CUPTI reports latency only
for whatever IS in the graph (the Triton/TileLang anchor) — the CuTe
reduce's GPU time is invisible to `bench_gpu_time_with_cupti`.

**Why:** TVM-FFI's environment-stream mechanism that CuTe DSL uses
for stream binding doesn't pick up `torch.cuda.graph()`'s capture
stream. The empty-graph warning only surfaces when the CuTe kernel
is alone; with a Triton anchor, the graph is non-empty → PyTorch
suppresses the warning → silent failure.

**Affected variants in this archive:**
- `cute_reduce` (unmeasured) — CuTe radix as the final reducer.
- `cute_reduce_v5` (71.05×, 2026-04-18) — same bug pattern.
- `cute_reduce_v6` (75.60×, 2026-04-20) — anchor; true per-call
  latency likely ~50× based on the `hybrid_dual_ns` (52×, Triton
  reduce, no CuTe) honest alternative measured on the same workloads.
- `cute_reduce_v7` (75.61×, 2026-04-22) — candidate; +0.27× AB over
  v6 is within drift and any delta will be re-measurable only after
  the upstream fix.

**Honest baselines** (no CuTe, full pipeline captured):
- `pure_triton` (45×) — minimum-dep reference.
- `hybrid_dual_ns` (52×) — Triton fwd + Triton reduce, both in graph.

**Evidence / cross-validation:**
- Side-by-side in the same Modal container, h16 ckv512 kpe64 topk2048
  workload (T=1): Triton fwd + CuTe reduce reports 4.51 µs CUPTI /
  6.16 µs Event; Triton fwd + Triton reduce reports 9.47 µs CUPTI /
  10.87 µs Event. The ~5 µs gap is the reduce's GPU time the CuTe
  solution silently skips. Event measurement confirms CuTe kernel
  doesn't run during replay (otherwise Event would also see the
  extra work).
- Three independent tests trigger on v6-pattern solutions: (1)
  zero-output sanity check — output stays zero after replay; (2)
  poison-cell test — vandalized cells survive across replays; (3)
  varying-inputs test — outputs byte-identical despite different q
  contents per call.

**How to apply:**
- Don't treat `cute_reduce_v{5,6,7}` headline speedups as the real
  performance; the true per-call work is roughly the `hybrid_dual_ns`
  level (~50×).
- When promoting a new CuTe-reduce variant as anchor, always
  cross-check against `hybrid_dual_ns` or `pure_triton` in the same
  container; if the CuTe variant reports <1.2× of the Triton reference
  delta plausibly, treat it as session-drift-inflated.
- After `nvidia-cutlass-dsl` upstream fixes `.launch()` to respect
  capture mode, re-measure all four CuTe variants to get the real
  cumulative delta; current v6→v7 "+0.27×" claim is unverifiable.
- Filed upstream as flashinfer-bench issue #414 (detection gate +
  upstream fix requested). Full repro: sibling
  `flashinfer-bench-cute-repro/final_repro.py`.

---

## Triton 3.6 Program-Dependent Launch (PDL) on Blackwell

### PDL overlap helps when producer has SM slack; regresses when producer saturates

**Fact:** Adding PDL to a Triton→Triton kernel pair via
`from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait`
+ `launch_pdl=True` on both launches is a **+1.39x drift-cancelled
AB** win (+2.52% overall score; +3.12x at T=1, +2.45x at T=2) **when
the producer kernel has idle SMs.** The same technique applied to a
TileLang fwd that is 1 block/SM smem-limited (via `T.pdl_trigger()` in
TL fwd body) **regresses T=8 by 3-4x** (50 → 46-47x).

**Why (helps):** `gdc_launch_dependents()` in the producer signals
"dependents may dispatch now" at the producer's tail. With
`launch_pdl=True` set on the consumer (sets
`CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`), the
consumer's blocks dispatch into idle SMs during the producer's
final wave, then block at `gdc_wait()` until the trigger fires. On
this operator TRI fwd at T=1 has grid (1, 32) = 32 blocks on 148 SMs
(0.22 waves per NCU); many SMs absorb reduce blocks for free during
TRI fwd's tail, hiding ~0.5 µs of per-kernel launch overhead.

**Why (regresses):** When the producer is 1 block/SM (smem or
register limited) and the grid ≥ SM_count, EVERY SM has a producer
block that hasn't released yet. PDL-dispatched consumer blocks land
on the same SMs and contend for L1/shared/registers rather than
overlap. On this operator, TL fwd at T=8 has grid (8, 16) = 128
blocks each 166 KB smem = 1 block/SM on 148 SMs; 20 SMs are idle,
128 are busy. Triggered reduce regresses T=8 from 50 → 46-47x
because the 128 contested SMs dominate.

**How to apply:**
- **Check NCU "Waves Per SM" on the producer.** If >1.0, PDL trigger
  is probably safe (wave 1 drains first, wave 2+ can overlap with
  consumer setup). If ~1.0 with producer at 1 block/SM, PDL trigger
  almost certainly regresses. If <0.5, PDL trigger is pure win.
- **Producer-side trigger in Triton:** `gdc_launch_dependents()` as
  the last statement of the kernel body (after all `tl.store`s).
- **Producer-side trigger in TileLang:** `T.pdl_trigger()` as the
  last statement inside `with T.Kernel(...) as ...:` block.
  TileLang's JIT auto-sets the launch attribute when this is
  present.
- **Consumer-side wait:** `gdc_wait()` (Triton) or `T.pdl_sync()`
  (TileLang) AFTER address/constant setup but BEFORE the first load
  of the producer's output. Placing it later expands the overlap
  window.
- **Consumer-side launch flag:** `launch_pdl=True` on the consumer's
  kernel launch (Triton). TileLang's JIT auto-sets it.
- **Testing safely:** always compare via `--ab-compare <prior-label>`
  in the same Modal container — cross-session drift on this operator
  is ±1x, so a naive single-session compare cannot resolve the
  ±0.5 µs savings PDL buys.

**Seen in v12 (2026-04-24):** TRI fwd+reduce with full PDL triad →
`hybrid_pdl` variant 57.55 ± 0.02x vs `hybrid_2d_reduce` 55.31 ± 0.07x
(3-run variance-check, same container AB +1.39x).

---

### Fused fwd+reduce via atomic-counter last-block is slower than separate kernels on small grids

**Fact:** Replacing the two-kernel (fwd → reduce) pipeline with a
single Triton kernel that does fwd + `tl.atomic_add(counter, 1,
sem='acq_rel', scope='gpu')` + `if old == NS-1: <reduce body inline>`
regresses T=1 from 76 → 9x (with reduce body) or 76 → 61x
(atomic-only, separate reduce kernel still invoked). Counter
initialized via `counter[:Tv].zero_()` pre-launch.

**Why:** Three compounding overheads:
1. The `.zero_()` memset captured as a graph node adds ~0.5 µs per
   replay AND disrupts the `_last_si_ptr` early-exit fast-path.
2. `scope='gpu'` forces the atomic through L2's coherence domain —
   ~500 ns per block even with minimal contention.
3. The conditional `if old == NS-1: <reduce...>` branch inflates
   register pressure even for blocks that skip the branch. The
   reduce body's `[NS=32, D=512]` fp32 tile = 64 KB per block of
   fragment is at the register-spill threshold; Triton can't
   eliminate dead-for-most-blocks registers at compile time.

**How to apply:** On Triton 3.6 for this grid shape, prefer separate
kernels + PDL (see prior TRAPS entry) over atomic-counter fusion.
The ~0.5 µs reduce launch saved by fusion is dwarfed by the ~2 µs
atomic/memset/register costs. True grid-wide sync for fusion would
need CUDA cooperative-launch or cluster launch, neither of which
Triton 3.6 exposes cleanly.

**Seen in v12 iter:** dsa-sparse-attention `_fused_triton_fwd_reduce`
attempt (2026-04-24).

---

### Pre-sorting `sparse_indices` for KV gather locality is too expensive

**Fact:** Inserting `torch.sort(sparse_indices, dim=-1,
out=(static_buf, perm_buf))` before fwd (to cluster KV gathers by
page address → L1/L2 locality gain) regresses T=1 from 76 → 20x
and T=2 from 65 → 19x.

**Why:** `torch.sort` on `[T, 2048]` int32 takes ~20 µs per call
even inside `torch.cuda.graph` replay. NCU on TL fwd at T=8 reports
DRAM throughput 2.75% with L1 hit rate 87% and L2 hit rate 42% —
the KV gather is NOT bandwidth-bound, so there's no headroom for
the sort cost to amortize against. The L2 is already cache-warm
from TMA / cp.async streaming into shared memory.

**How to apply:** Before proposing any "pre-process inputs to improve
cache locality" optimization, check NCU DRAM throughput. If <10%,
the kernel is stall-bound or launch-overhead-bound rather than
memory-bound, and locality optimizations cannot amortize their
overhead. Locality optimizations are only profitable when DRAM is
≥50% utilized.

**Seen in v12 iter:** dsa-sparse-attention sort-before-fwd experiment
(2026-04-24).

---

### Structural-change invalidates parameter sweeps (Triton MMA tile-picker) (2026-04-25)

**Fact:** The Triton MMA tile-picker is sensitive to the 4-tuple
`(num_warps, M, N, K)` of each `tl.dot` — not just to `M/N/K` at a
fixed warp count. After any structural change that shifts `M`, `N`,
`K`, register pressure, or smem budget (tile-shape, fragment-split,
H-split, K-unification, new buffer layout), the entire previous
dead-end list for `num_warps` / `num_stages` on that kernel becomes
**unverified**. Two confirmed dead-end-to-winner flips on this
operator in v13 after H-split on the Triton fwd:

| Knob | hybrid_pdl (v12) status | hybrid_pdl_v2 (v13, post-H-split) |
|------|-------------------------|-----------------------------------|
| `num_stages=2` on TRI fwd | dead-end (carried forward) | **+0.496x AB winner** (iter-12) |
| `num_warps=8` on TRI fwd  | dead-end (implicit — prior sweep at nw=4) | **+1.55x AB winner** (iter-21) |

**Why:** Two independent mechanisms at play.

1. **Smem budget shift.** Pre-H-split, the Triton fwd held
   `Qn_s [H=16, D=512] bf16 = 16 KB` and `Qp_s [H=16, D=64] bf16 =
   2 KB` plus the KV working set. Adding `num_stages=2` doubled the
   cp.async buffer footprint and overflowed the per-block smem
   headroom. Post-H-split, `H=8` halves `Qn_s` to 8 KB and `Qp_s` to
   1 KB, freeing room for the double-buffer. The dead-end was a
   smem-pressure failure, not a fundamental latency issue.
2. **MMA primitive selection.** Triton's codegen picks a different
   PTX MMA instruction at different `(num_warps, M, N, K)` tuples;
   the picker has cliffs, not a smooth curve. At `M=16 (pre-split)`,
   `nw=4` was optimal and `nw=8/16` regressed. At `M=8 (post-split)`,
   `nw=8` selected a measurably faster primitive than `nw=4`. The
   prior sweep's rejects reflected the primitive cliff at `M=16`, not
   at `M=8`.

**Practical rule:** After ANY structural change to a Triton kernel,
re-run the parameter sweep from scratch. A 2×2 sweep (two warp
counts × two stage counts) costs ~4 bench iterations — cheap
insurance against the trap below.

**Cost of ignoring this:** In the v13 session, iter-9 committed
H_SPLIT=2 without re-sweeping `num_warps`. The session proceeded
through iter-10 to iter-20 (11 labeled iterations) before iter-21
re-tested `num_warps=8` on a nudge from the user. The recovered
delta was +1.55x AB — **the largest single win of the session**,
gated behind 11 iterations of wasted effort exploring orthogonal
directions. Two `num_warps=8` sweeps at iter-10 would have surfaced
it immediately.

**WHEN narrow:** Triton kernels where structural knobs (H-split,
tile shape, fragment split, thread count, smem allocation) have
been changed since the last parameter sweep on that kernel.

**WHEN broad:** Any code-generator-based MMA path (Triton, TileLang
tcgen05, CuTe DSL) where the compiler picks primitives based on the
full launch-geometry tuple. The parameter search space is
non-convex and non-monotone; structural changes can unlock
previously-dominated regions.

**Anti-pattern:** Treating the prior variant's Dead-ends section as
immutable after a structural derivative. Carry-forward dead-ends
without re-verification are only safe when the surrounding kernel
state is unchanged.

**Seen in v13 iter-9 → iter-21:** dsa-sparse-attention
`hybrid_pdl_v2` recovered +0.9x after 11 iterations of delay.

---

## `_last_si_ptr` int fast-path — FIB-contract-validated optimization, not a trap (2026-05-23 audit, revised)

**Status (TL;DR):** Contest-legal per official confirmation. Listed
here so future readers porting the kernel to a non-isolated-runner
runtime (e.g. raw SOL-ExecBench, production serving with buffer pool)
know to drop the fast-path. **Under FIB / contest contract, this is
not a bug — keep the anchor as-is.**

**Official confirmation (yongwww, 2026-04-19, contest organizer
response to a direct question):**
> "Reusing a captured CUDA Graph when shapes and captured tensor
> addresses remain stable within the same isolated subprocess. … In
> another implementation, there is also an address-stability-based
> hot path that replays the previously captured graph when a stable
> sparse-index pointer is observed. … i think the techs you mentioned
> above are all valid."

This is the exact pattern at L561-564. Confirmed legal.

---

**Structural fact (for hypothetical non-FIB callers):**
`variants/hybrid_pdl_v2/kernel.py` L561-564 is a single-pointer
fast-path that bypasses the full-key check:

```python
si_ptr = sparse_indices.data_ptr()
if si_ptr == _last_si_ptr and _last_graph is not None:
    _last_graph.replay()
    return _last_out, _last_lse
```

The full 6-tuple key check at L567 is only reached when the fast-path
misses. Under FIB's per-trial fresh-tensor model with
`use_isolated_runner=true`, the fast-path is correct (per-trial all 5
input tensor pointers change together, so the chance of `si_ptr`
matching while other pointers differ is ~0). Under buffer-pool reuse
(SOL-ExecBench cross-workload, production serving with persistent
activation buffers), the int32 `sparse_indices` slot is often the
last to be re-allocated (small, frequently-reused) — so `si_ptr` can
match while `q_nope` / `q_pe` / `ckv_cache` / `kpe_cache` have moved.

**Why:** The cached graph baked in addresses for `q_nope`, `q_pe`,
`ckv_cache`, `kpe_cache` at capture time. On replay those addresses
are read — but the caller's current `q_nope` is at a NEW address.
Result: graph reads from the OLD address, which now contains the
previous workload's data (or freed memory) → INCORRECT_NUMERICAL or
segfault.

**FIB-contract gating:** `config.toml` sets
`use_isolated_runner = true` (`spawn.py` default since 2026-05). Each
workload runs in a fresh subprocess, so module-level state
(`_last_si_ptr`, `_graph_cache`) is clean each workload. Within a
workload, all 5 input tensors come from the same fresh allocation →
addresses change together at trial boundary → fast-path correctly
invalidates.

**Detection:** `cheat_check_modal.py` does NOT detect this — it
mutates inputs in-place keeping pointers stable; the fast-path
correctly hits and replay reads mutated memory → PASS. Forensic-
identified via static audit (10-kernel sweep on 2026-05-23), not yet
measured-failing.

**Same trap family across this archive:**
- `dsa-topk-indexer/v8_radix_bt256` — SAFE (full 7-tuple key check,
  no fast-path bypass)
- `gdn-decode/cuda_bv32_register_resident` — SAFE (full 9-tuple
  `_last_key == key` check, no single-ptr shortcut)
- `gdn-prefill/cuda_graph_v5` — SAFE (shape-only key + always-copy
  into static buffers; reference pattern for SOL-compat)

**Cross-campaign trap family (different operators, same structural
shape):**
- `reference/mla-paged-prefill-causal-h16-ckv512-kpe64-ps1/.../iter5-...`
  — Form A `_max_q_cache` host scalar cache + Form C 7-tuple graph
  cache. See sibling family `TRAPS.md`.
- `reference/moe-fp8-block-scale-ds-routing-.../iter4-...` — Form B
  `_GRAPH_BY_KEY` keyed on `(T, dev)` with only `hs_ptr`
  revalidation. See sibling family `TRAPS.md` +
  `variants/iter4-.../submission_no_cache.py`.

**Pre-existing acknowledgment:** the variant's header lessons section
(line 107-108) labels the fast-path "safe under
use_isolated_runner=true" — the kernel author already documented
the contract dependency. This audit confirms the dependency was
correctly identified.

**Non-FIB porting note:** if reusing this kernel in a runtime
**without** per-workload subprocess isolation (raw SOL-ExecBench,
production serving with persistent activation buffers), drop the
`_last_si_ptr` fast-path and rely on the full-key check at L567+.
Cost: one dict lookup per call instead of one int compare;
negligible on this operator's microsecond-scale latency. A drop-in
non-FIB derivative is provided at
`variants/hybrid_pdl_v2/submission_no_cache.py` (importlib-based,
overrides only `run()`). This is a **portability variant**, NOT a
correctness fix for the FIB/contest anchor — the canonical
`kernel.py` remains the contest-correct implementation.

**How to apply:** treat the fast-path as **FIB/contest-specific**.
Keep it in `kernel.py` for the contest contract. Use
`submission_no_cache.py` only when porting to a runtime where
subprocess isolation is NOT provided.

---

**Audit context:** the 2026-05-23 ten-kernel sweep initially
classified this fast-path as a "trap" via structural analysis alone
(without consulting contest rules). The user surfaced yongwww's
explicit 2026-04-19 confirmation, which is dispositive: this is contest-validated
optimization, not a structural problem. The entry above is preserved
in its "structural-fact" form so future audits see both the
mechanism and the contract that gates it.
