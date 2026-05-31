# DSA Sparse Attention — Archive

Working kernel variants preserved from prior optimization sessions on
the `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64` operator family.
Last updated 2026-04-25 (v13 session).

## ⚠️ CuTe-reduce variants are measurement-inflated (2026-04-23)

A newly identified bug — `@cute.kernel.launch()` does not participate
in `torch.cuda.graph` capture — makes the CuTe-reduce variants
(`cute_reduce`, `cute_reduce_v5`, `cute_reduce_v6`, `cute_reduce_v7`)
report speedups inflated by the reduce's GPU time (the reduce runs
during capture, then `g.replay()` skips it; output stays stale from
capture but matches the reference because benchmark inputs are fixed
per workload). See `TRAPS.md` section "`@cute.kernel` is not captured
into `torch.cuda.graph`" and flashinfer-bench issue #414 for evidence.

**Anchor has been switched to `hybrid_pdl_v2` (Triton + TileLang, no
CuTe dep)** until the upstream fix in `nvidia-cutlass-dsl` lands; at
that point the CuTe variants should be re-measured and promoted back
if the delta survives.

## Anchor

**`variants/hybrid_pdl_v2/`** — 58.34 ± 0.04x variance-verified
(CV 0.06%, 3-run, Modal B200, CUDA 13.2, 2026-04-25); 58.89x labeled
single-run. Variance-floor AB vs `hybrid_pdl`: **+0.79x (+1.37%)**.
Same-container in-session AB (drift-cancelled): **+1.55x (+2.72%)**.
Four compounding changes on top of `hybrid_pdl`:

1. **Triton fwd H_SPLIT=2** — 8 heads per block, grid `(T, 32, 2)`
   doubles block count at T=1,2 (closes the 0.22-wave gap at T=1).
2. **Triton fwd num_stages=2** — only fits post-H-split; halved
   `Qn_s`/`Qp_s` opens smem room for cp.async double-buffer. Was a
   dead-end in `hybrid_pdl`.
3. **Triton fwd num_warps=8** — MMA tile-picker re-selects a faster
   primitive at `(nw=8, M=8, N=64, K=512)`. Biggest single AB
   contributor.
4. **TileLang fwd D-chunked acc_o** into `2×[H, D/2]` fragments —
   GEMM scheduling slack in tcgen05; registers unchanged.

Inherits all seven `hybrid_pdl` carry-forward wins (PDL overlap,
2D-tile merged TRI reduce, D_CHUNK=32×num_warps=1, T=1 USE_MASK, TL PO
transpose, TL NI=1 fastpath, `_last_si_ptr` int fast-path). Fully
captured into `torch.cuda.graph`; no `cutlass-dsl` dependency.
Unaffected by flashinfer-bench issue #414.

Per-T (3-run mean ± std): T=1 78.94 ± 0.29, T=2 69.12 ± 0.11,
T=6 51.35 ± 0.00, T=7 50.56 ± 0.05, T=8 50.53 ± 0.01.

Keep `hybrid_pdl` + `hybrid_2d_reduce` as preserved lineage
fallbacks if a future Triton toolchain regresses tile-picker
behavior at nw=8 post-H-split, or if `launch_pdl` is removed.

## Honest alternatives

- **`variants/hybrid_pdl/`** — 57.55 ± 0.02x; v12 predecessor to
  `hybrid_pdl_v2`. Same architecture minus the Triton-fwd H-split
  + num_stages=2 + num_warps=8 compound and minus the TL-fwd
  D-chunked `acc_o`. Fallback if a Triton 3.6 regression breaks
  the tile-picker primitive selection at `(nw=8, M=8, N=64, K=512)`.
- **`variants/hybrid_2d_reduce/`** — 55.31 ± 0.07x; v11
  pre-PDL predecessor. Fallback if a toolchain regression disables
  PDL entirely.
- **`variants/hybrid_dual_ns/`** — 47.43x; v10 pre-breakthrough
  baseline. Same Triton fwd + TileLang fwd, but scalar static_range
  form on both reduces (no 2D-tile merge, no PDL). Kept as lineage +
  cheap fallback.
- **`variants/pure_triton/`** — 45x; minimum-dep Triton-only.
- **`variants/pure_tilelang/`** — TileLang-only.

## Quarantined (re-measure after upstream fix)

Listed newest-first; all four share the same `flashinfer-bench #414`
bug (`@cute.kernel.launch()` bypasses `torch.cuda.graph` capture).

- **`variants/cute_reduce_v7/`** — previously listed as candidate;
  claimed "+0.27× (0.36%) AB vs v6" in v10 session (2026-04-22).
  Both sides of the AB were equally bugged; the delta is
  unverifiable until the upstream fix. `FUSED_KERNEL_DESIGN.md`
  fused-fwd+reduce lever still valid in principle but defer any new
  ~3000-line MMA port until the measurement path is trustworthy.
- **`variants/cute_reduce_v6/`** — previously listed as anchor at
  75.60 ± 0.08× (5-run variance-check, Modal B200, CUDA 13.2,
  2026-04-20). Headline inflated; honest per-call latency likely ~50×.
  Requires `[benchmark] use_isolated_runner = true` on persistent
  runners. See `kernel.py` header for architecture + the WARNING
  block added in 2026-04-23.
- **`variants/cute_reduce_v5/`** — 71.05× (legacy); immediate
  predecessor to v6. Same bug; retained for delta-decomposition
  comparisons post-fix.
- **`variants/cute_reduce/`** — earliest CuTe-DSL breakthrough
  sample from v4 session; unmeasured speedup on current stack.

## Canonical baseline

`baseline.json` — 23 workloads, 546.1 µs aggregate mean, sourced from
`flashinfer_wrapper_5af199` (FlashInfer Python wrapper — the expert
baseline for this operator), measured on Modal B200 / CUDA 12.8 at
2026-04-15. Frozen denominator for all speedup numbers; do not replace
without re-measuring every variant. Full provenance in `baseline.json`
under `provenance`.

## Cross-variant traps

See `TRAPS.md` for toolchain / methodology facts that apply regardless
of which variant is anchor. Added in v10 session (2026-04-22): AB deltas
don't compose cumulatively; Modal session drift ~±1x; TileLang
`T.Kernel` rejects `cluster` kwarg; CuTe DSL `alloc_smem` pointer needs
`make_tensor` wrapper for runtime-tid subscript. **Updated 2026-04-24
(v11):** the v10 "reduce loop-merge sign flips between Triton-path and
TileLang-path" entry was rewritten — the sign flip was a CuTe-DSL
codegen artifact. On pure Triton, a *merged 2D-tile load* (not the
v10 scalar merge) beats the split form on TRI-path (+3.37x AB) and
register-spills on TL-path. New entry added covering the 2D-tile
register-budget ceiling. **Updated 2026-04-24 (v12):** three new
entries on Triton 3.6 Program-Dependent Launch: PDL overlap helps
when producer has SM slack, regresses when producer saturates
(TL fwd at 1 block/SM); fused-fwd+reduce via atomic-counter
last-block pattern is slower than separate kernels on small grids;
pre-sorting `sparse_indices` for KV gather locality is too expensive
relative to L2 already being cache-warm. **Updated 2026-04-25 (v13):**
one new entry on structural-change-invalidates-parameter-sweeps —
after H-split, two prior dead-ends in the `hybrid_pdl` header
(`num_stages=2` and `num_warps=8` on Triton fwd) became the winning
config, leaving +0.9x on the table for 9 iters before being
recovered at iter-21.
