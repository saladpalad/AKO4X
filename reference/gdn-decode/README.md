# GDN Decode — Archive

Working kernel variants for `gdn_decode_qk4_v8_d128_k_last` (Gated
Delta Net decode, group-value-attention, k-last state). First snapshot
2026-04-23 from `ako4fib-run-gdn_decode_v0`; anchor updated 2026-04-25
from `ako4fib-run-gdn_decode_v3-2` (CUDA branch).

## Anchor

**`variants/cuda_bv32_register_resident/`** — 1.18× ± 0.00174×
(3-run variance-check, Modal B200 CUDA 13.2, 2026-04-25, 54/54 PASS).
Pure CUDA via `torch.utils.cpp_extension.load_inline`. Same single-pass
register-resident dataflow as the prior Triton anchor, but with
**BV=32 dispatch at B≥32** halving the CTA grid (2048→1024 at B=32).
Possible because `__launch_bounds__(128, 4)` pins a register budget
Triton's BV=32 spilled past. Drift-free A/B vs prior Triton anchor:
mean +5.0%, large-B (B=32/48/64) +5–8%. Per-B numbers and lessons
live in `kernel.py` header + `config.toml` + `variance.json`.

## Fallbacks

**`variants/triton_swap_grid_evict_lsr/`** — 1.13× ± 0.00166×
(3-run variance-check, 2026-04-24). Pure Triton ceiling for this op;
the anchor cleared it +5% via the larger-tile dispatch. Worth keeping
as the cross-language reference: it documents the SWAP_GRID +
eviction-hint stack that the CUDA anchor inherited.

**`variants/triton_bv_dispatch_graph/`** — 1.126× mean (5-trial,
2026-04-23). Pure Triton BV dispatch (BV=8 for B≤8, else BV=16) +
input-pointer-keyed CUDA graph cache. Predecessor of the Triton
fallback above; kept for the graph-cache + `_last_key` hot-path
pattern (also reused by the CUDA anchor verbatim).

## Canonical baseline

`baseline.json` — 54 workloads, expert reference (CUDA 13.2,
2026-04-22). Denominator for all speedups; do not replace without
re-measuring every variant. Workload axis: `batch_size` ∈
{1(×10), 4(×8), 8(×7), 16(×7), 32(×7), 48(×7), 64(×8)}. Fresh inputs
regenerated per call.

## Cross-variant traps

See [`TRAPS.md`](TRAPS.md) for: eviction+grid combinatorics,
`num_stages≥2` regression on short-trip register-resident loops,
graph-cache pointer aliasing (`use_isolated_runner` correctness
requirement), Modal session drift (±5–15% cross-container vs ±1%
within), persistent-outer-loop regression against SWAP_GRID, NCU's
"Est. Local Speedup" overestimate for SMEM-staging fixes,
BV / CTA-count sweet spot at ~1024 CTAs/wave on B200, and TileLang's
mismatch with matvec/decode reductions.
