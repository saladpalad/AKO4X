# DSA TopK Indexer — Archive

Working kernel variants preserved from prior optimization sessions on
the `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64` operator family.
Last updated 2026-04-25 (v0 + v0-1 sessions — 2 new dead-ends recorded
[multi-CTA 2-kernel pipeline; K-split + num_stages with explicit loop],
v8 silent-skip TRAP verified NOT applicable, anchor kernel.py cleaned
of one unused param; headline 45.55x unchanged).

## Anchor

**`variants/v8_radix_bt256/`** — current best, 45.55 ± 0.05x (3-run
variance-check CV 0.10%, Modal B200, CUDA 13.2, 2026-04-22). Sole
perf-relevant change from fast_split_v6: CUDA radix `BT=64 → BT=256`.
+0.73x over v6 standalone. See `kernel.py` header for full
Lessons / Dead-ends / Open directions.

## Quarantined (re-measure after upstream fix)

**`variants/evict_last_v4/`** — prior production; bundles CuTe DSL radix
as correctness-check fallback. 41.6× on CUDA 13.2. ⚠️ **The headline
is inflated**: the CuTe DSL radix inside the `torch.cuda.graph(g)` block
is not captured (see `TRAPS.md` and flashinfer-bench issue #414) — it
runs only during capture, then `g.replay()` skips it, and the topk
output stays stale from capture time. Current anchor
(`v8_radix_bt256`, 45.55×, pure CUDA) and `fast_split_v6` (44.82×,
pure Triton + CUDA) are honest alternatives with all kernels captured;
both outperform `evict_last_v4` once the measurement artifact is
accounted for. Kernel preserved as-is; re-measure after upstream fix.

## Fallbacks

- **`variants/fast_split_v6/`** — prior anchor (2026-04-20), 44.82 ± 0.05x
  variance-check. Superseded by v8_radix_bt256 via single-line radix BT change.
- **`variants/warp_coop_v3/`** — CUDA-only radix, pre-`evict_last`,
  lean deps.

## Legacy lineage (not re-measured on CUDA 13.2)

These two preserve the pre-warp-coop kernel (CUB `BlockScan<int,1024>`
radix, BT=1024) as a milestone. `result.json` shows
`status: "not_measured_on_this_env"` with only `legacy_speedup_headline`
fields — the CUDA 12.8 numbers are not reproducible on the current
image.

- **`variants/graph_cached/`** — 41.35 ± 0.16x legacy (CUDA 12.8,
  3-run variance-check 2026-04-17).
- **`variants/no_graph/`** — 32.25 ± 0.03x legacy. Used to be the
  graph-capture toggle-proof, but `v8_radix_bt256` now ships an
  in-place `DSA_NO_GRAPH=1` env gate that does the same A/B on the
  current kernel — prefer that for fresh measurements.

## Canonical baseline

`baseline.json` — 128 workloads, 119 µs aggregate mean (legacy CUDA 12.8).
Denominator for all speedup numbers; do not replace without re-measuring
every variant.

> On the CUDA 13.2 CI image (Apr 24 eval target) the canonical expert
> baseline solution crashes (`deep_gemm.get_paged_mqa_logits_metadata`
> asserts `context_lens.dim() == 2`, baseline passes 1-D); this
> `baseline.json` is the only reliable denominator until the contest
> ships an update.

## Cross-variant traps

See [`TRAPS.md`](TRAPS.md) for toolchain / methodology facts that apply
to every variant regardless of which is anchor. Two entries so far:
(1) `@cute.kernel.launch()` is silently skipped by `torch.cuda.graph`
capture — the evict_last_v4 headline is inflated by this; (2) raw
`<<<grid, block>>>` without `at::cuda::getCurrentCUDAStream()` is also
silently skipped under capture (added 2026-04-24 from v9 session;
**verified NOT applicable to v8 anchor on 2026-04-25** — `DSA_NO_GRAPH=1`
A/B confirms the radix is honestly captured & replayed, and the
proposed `getCurrentCUDAStream()` patch causes a -4× regression on slow
workloads due to per-call lookup overhead; trap remains valid for any
NEW load_inline kernels added to a captured graph).
