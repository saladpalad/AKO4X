# GDN Prefill — Archive

Working kernel variants preserved from prior optimization sessions on
the `gdn_prefill_*` operator family (Qwen3-Next linear-attention layers,
prefill stage). Last updated 2026-04-25.

## Anchor

**`variants/cuda_graph_v5/`** — current best, **4.70x** (iter-2 labeled
run, Modal B200 CUDA 13.2, 2026-04-25, 100/100 PASS, range
1.95x–10.45x; see `variants/cuda_graph_v5/result.json`). Variance
unmeasured. Trajectory-direct Δ over v4 iter-0 carryover = **+0.027x**
(4.676 → 4.703), of which iter-2 contributes +0.024x and an iter-1
PDL gating refinement contributes +0.003x drift. Inherits v4's eight
Triton kernels, NT_est-gated BV_o dispatch, and PDL chain unchanged;
adds ONE structural win:

1. **FP8 e4m3 for `_fwd_o_kernel`'s three MMAs** (iter-2, +0.024x).
   Inline `.to(tl.float8e4nv)` on q, k, h, v_tile operands before each
   `tl.dot` (q@hᵀ, q@k, A@v_new). No scaling — operands are raw bf16
   model activations in roughly [-10, 10], deep inside e4m3 ±448 range.
   Gain is well below the 2× tcgen05 throughput ceiling because fwd_o
   was 21% compute TP (occupancy/latency-bound). The session also
   tried FP8 in `_kkt_solve_kernel` phases 5/6 but it failed correctness
   (matrix-inverse intermediates exceed e4m3 range — see TRAPS.md
   entry "FP8 e4m3 casting: safe for bounded inputs, corrupts derived
   intermediates"). The PDL gating refinement (`USE_PDL_WAIT` constexpr
   for fwd_o on grid ∈ [148, 350]) is in v5 but is bench-mean drift
   level — useful only for closing v4's open direction with empirical
   ceiling = drift, not the +0.05x v4 had estimated.

See `kernel.py` header for full Lessons / Dead-end / Open directions.
The biggest unrealized opportunity from this session is **iter-4 h_buf
elimination** (fold q@hᵀ into state_rec; ceiling +0.05–0.1x); blocked
by Modal infrastructure timeout, code structure reviewed and preserved
as v5's HIGHEST-priority open direction. Variance-check is still an
open direction; `variants/cuda_graph_v3/` remains the variance-checked
fallback (CV 0.15%) — cite v3 for any sub-1% future comparison until
the latest anchor's noise floor is measured.

## Prior anchors

- **`variants/cuda_graph_v4/`** — superseded by v5 on 2026-04-25,
  preserved as the FP8-free fallback. Was the fourth archived variant
  at 4.67x (Modal B200 CUDA 13.2, 100/100 PASS, range 1.95x–10.60x,
  variance unmeasured). Forked from v3 (4.56x); two structural wins
  lifted to 4.67x (+2.4% over v3 via AB-compare): BV_o adaptive
  dispatch in `_fwd_o_kernel` (+0.046x, gated on NT_est≥19 for grid
  saturation on 148-SM B200), and PDL chain across every producer→
  consumer pair (+0.06x). Both wins inherited unchanged by v5. v4's
  session also surfaced two cross-variant traps now in TRAPS.md:
  num_warps=8 on `_fwd_o_kernel` is a correctness (not perf) regression,
  and PDL borderline-wave producers (1.0–1.5 waves) cause consumer
  contention.

- **`variants/cuda_graph_v3/`** — superseded by v4 on 2026-04-25,
  preserved as the variance-checked baseline. Was the third archived
  variant at 4.56 ± 0.007x (3-run CV 0.15%, Modal B200 CUDA 13.2,
  2026-04-24, 100/100 PASS, range 1.92x–10.08x). Forked from v2 (4.25x);
  three MMA-scheduling wins lifted to 4.56x (+7.3% over v2):
  unified K=128 in `_state_recurrence_kernel` (+0.16x), same
  unification in `_fused_single_chunk_kernel` (+0.10x), phase 5/6
  big-matmul in `_kkt_solve_kernel` via `tl.join + permute + reshape`
  (+0.06x). All three wins inherited by v4. Patterns portable beyond
  gdn-prefill — see `TRAPS.md` entries #7 (unified K), #8 (tl.join
  idiom), #9 (padding anti-pattern).

- **`variants/cuda_graph_v2/`** — superseded by v3 on 2026-04-24,
  preserved for TRAPS.md provenance. Was the second archived variant
  at 4.25x (Modal B200 CUDA 13.2, 2026-04-24, 100/100 PASS, variance
  unmeasured). Inherited v1's three-Triton-kernel CUDA-Graph
  foundation; added four additional kernels (three `_kkt_solve_tiny*`
  specializations + `_fused_single_chunk_kernel`), CUPTI-aware
  `data_ptr()` skip-copy, and output-clone removal. The three
  independent-≥4%-on-mean wins of the v2 session were: skip input
  copies via data_ptr() comparison (+58%), fused state_rec+fwd_o for
  NT=1 (+6%), specialized kkt_solve per T-bucket (+8%). All still
  inherited by v3 and v4.

- **`variants/cuda_graph_v1/`** — superseded by v2 on 2026-04-24,
  preserved for TRAPS.md provenance. Was the first archived variant
  for the gdn-prefill family at 1.83 ± 0.003x (3-run CV 0.14%, Modal
  B200 CUDA 13.2, 2026-04-23, 100/100 PASS). FLA-port chunked
  delta-rule in Triton: three Triton kernels wrapped in a per-shape
  `torch.cuda.CUDAGraph`. The two structural breakthroughs of that
  session (each independently worth +40-50%): chunk-meta `.item()`
  GPU-sync removal (v1 iter-8) and per-shape CUDA Graph capture
  (v1 iter-11). Both inherited by v2, v3, and v4 as the foundation.

## A/B baseline

`variants/cuda_graph_v4/` is the explicit A/B reference for the current
anchor. v5's iter-0 (baseline carryover) re-measured v4 on the same
Modal container at 4.676x — within-drift of v4's archived 4.67x. The
load-bearing number for v5's claim is the **trajectory-direct Δ within
the same session container**: iter-0 4.676 → iter-2 4.703 = +0.027x.
This is not a formal `bench.sh --ab-compare` result (the session relied
on direct same-container trajectory comparison rather than the per-iter
ab-compare protocol used in v4's session), but it controls for
container-level drift since both numbers came from the same Modal
allocation. v4's per-iter AB-compare deltas remain the gold-standard
record for that variant. Deeper provenance: `variants/cuda_graph_v3/`
holds the only variance-checked baseline in the chain (CV 0.15%); cite
v3 for any sub-1% claim until a tighter v4/v5 variance-check lands.

## Canonical baseline

`baseline.json` — 100 workloads of `gdn_prefill_qk4_v8_d128_k_last`
(Qwen3-Next TP=4 prefill: q=[T,4,128], k=[T,4,128], v=[T,8,128] bf16,
state=[N,8,128,128] fp32). Frozen flashinfer expert latencies (Modal
B200, CUDA 13.2). Denominator for all variant `result.json` speedups
in this archive — do not replace without re-measuring every variant.

## Configuration

All archived variants require `[benchmark] use_isolated_runner = true`.
Module-level `_GRAPH_CACHE` would alias across workloads in a
persistent runner. Default in `scripts/bench_utils.py:64` is now
`True` since 2026-04-23 (commit `838eead`); children spawned before
that date carry the old `False` default until re-spawned.
