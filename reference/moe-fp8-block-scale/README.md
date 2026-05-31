# MoE FP8 Block-Scale — Archive

Working kernel variants preserved from prior optimization sessions on the
`moe_fp8_block_scale_*` operator family (DeepSeek-V3/R1 no-aux routing +
FP8 block-scale grouped GEMMs). The specific operator optimized in this
archive is
`moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`.

Last updated 2026-04-25 (anchor switched to `fused_routing_v2`;
`fused_graph_all_t` and `fused_graph_swiglu_tuned` quarantined for
hoisting routing out of the captured graph + pointer-keyed reuse —
same cheat family as flashinfer-bench #414, see `TRAPS.md` "Cheat
patterns surfaced in this archive").

## ⚠️ Two graph-captured variants quarantined (2026-04-25)

`fused_graph_all_t` (1.380×) and `fused_graph_swiglu_tuned` (1.407×)
both **hoist the routing kernel out of the captured `torch.cuda.graph`**
and short-circuit replay when `routing_logits.data_ptr()` matches the
prior call's pointer (counts / sorted_tokens / weight_vec are reused
as inputs to the captured GEMM1+SwiGLU+GEMM2). This violates the
official rule against "skipping a kernel because inputs look the same"
and against "previous outputs reused as results" — same cheat family
as the CuTe-DSL graph-capture skip (flashinfer-bench #414). The 2.4%
headline gain at T=14107 the anchor claimed is exactly the routing
kernel's per-iteration GPU time the rule forbids hiding from the
timer.

**Anchor has been switched to `fused_routing_v2`** (1.204× ± 0.004×,
no graph capture, runs the full pipeline every call). See
`TRAPS.md` "Hoisting routing out of the captured graph + pointer-keyed
reuse is the same cheat family as flashinfer-bench #414" for the full
analysis + detection methodology.

## Scoreboard (canonical baseline, 3-run variance-check)

| Role | Variant | Headline | CV | Notes |
|---|---|---|---|---|
| **Anchor** | `variants/fused_routing_v2/` | **1.204x ± 0.004x** | 0.30% | fused-routing single-kernel, no graph capture |
| Earlier anchor (moe1) | `variants/fused_indirect_v1/` | 1.000x ± 0.001x | 0.10% | scatter architecture baseline |
| A/B baseline | `variants/presync_v1/` | 0.842x ± 0.001x | 0.10% | moe1 iter-30 |
| Quarantined (moe_v0) | `variants/fused_graph_swiglu_tuned/` | 1.407x ± 0.001x ⚠️ | 0.07% | inflated — routing hoisted out of graph |
| Quarantined (moe4) | `variants/fused_graph_all_t/` | 1.380x ± 0.003x ⚠️ | 0.20% | inflated — routing hoisted out of graph |

All measured on Modal B200 sm_100 under the CUDA 13.2 / Triton 3.6 /
`flashinfer-ci-cu132:20260401-2c675fb` image, against the canonical
`baseline.json` (MD5 `a1d2be64…`). The 2026-04-23 sweep covered the four
non-moe_v0 rows; moe_v0's row was variance-checked 2026-04-25 in its own
session container.

## Anchor

**`variants/fused_routing_v2/`** — 1.204x ± 0.004x, session ako4fib-run-
moe2 (2026-04-23). The honest pre-graph-capture architecture: fuses
routing + scatter + weight-write into one kernel, drops the
Python-level fancy-index weight_vec pre-gather, co-locates weight_vec in
per-expert layout with sorted_tokens, skips CPU sync for 256 < T ≤ 2048,
side-stream memset for output. **No CUDA Graph capture** — the entire
pipeline runs every call, so per-iteration GPU work is what the timer
measures. All 19 workloads positive (T=1 1.397×, T=15 1.446×,
T=11948 1.252×, T=14107 1.175×). Requires
`[benchmark] use_isolated_runner = true`.

## Earlier anchor (moe1)

**`variants/fused_indirect_v1/`** — 1.000x ± 0.001x, session ako4fib-run-
moe1 (2026-04-22; re-measured under canonical 2026-04-23). First variant
to beat the expert baseline on mean speedup. Introduces the core scatter
architecture that all later variants inherit: bucket-scatter dispatch
replacing `torch.argsort`, inline exclusive cumsum in every consumer,
count-atomic merged into routing kernel at T≤256. Kept in archive for
cross-reference; superseded by v2.

Under the canonical baseline v1 dropped from a prior-header 1.022x to
1.000x (−0.02x). The shift comes from the baseline replacement — the
pre-canonical global cache (MD5 `836840d3…`) had 20–25% slower small-T
expert latencies than the canonical reference, which inflated all
variants' headline speedups. Canonical is now authoritative; do not
replace without re-measuring every variant (see `baseline.json`
provenance block).

## Quarantined (re-measure as honest after routing is moved INTO the graph)

The quarantined variants are kept in the archive for two reasons:
(a) the per-call architectural wins on top of `fused_routing_v2`
(`tl.atomic_add(..., sem="relaxed")`, multi-stream `output.zero_()`,
SwiGLU multi-row tiling, eviction hints, the moe_v0-era PDL +
M_pad-conditional `num_warps`) are themselves valid optimizations and
worth porting onto a graph-captured architecture that **includes
routing**; (b) preserving them documents the cheat pattern for future
sessions.

**`variants/fused_graph_swiglu_tuned/`** — 1.407× ± 0.001× headline
(quarantined 2026-04-25). Inherits every architectural win from
`fused_graph_all_t` (CUDA Graph capture for all T, **routing hoisted
OUT of the graph**, `sem="relaxed"` atomics, multi-stream memset,
eviction hints, persistent output buffer) and layers three small
tunings on top: PDL on the 3-kernel compute chain, SwiGLU
`num_warps = 4 if M_pad < 2048 else 2`, SwiGLU `ROWS = 4 → 8`. The
three tunings are honest; the headline is inflated by the
routing-hoist (~2.4% at T=14107 per the kernel's own comment, less at
small T). Honest per-call latency would be roughly the
`fused_routing_v2` level minus the per-call architectural wins — let's
call it ~1.34–1.38×. Re-measure after routing is moved back into the
captured graph.

**`variants/fused_graph_all_t/`** — 1.380× ± 0.003× headline
(quarantined 2026-04-25). Same hoist-out-of-graph pattern as
`swiglu_tuned`; immediate predecessor. Honest per-call latency would
be roughly the `fused_routing_v2` level plus the orthogonal
per-call-architecture wins.

## A/B baseline

**`variants/presync_v1/`** — 0.842x ± 0.001x (3-run, CV 0.10%). Immediate
predecessor to v1 (iter-30 snapshot). Retained to A/B-isolate the three
launch-overhead wins v1 layered on top (iter-31 skip-sync T≤256, iter-34
inline cumsum, iter-37 merge count atomic into routing).

## Earliest anchor (moe1)

**`variants/fused_indirect_v1/`** — 1.000x ± 0.001x, session ako4fib-run-
moe1 (2026-04-22; re-measured under canonical 2026-04-23). First variant
to beat the expert baseline on mean speedup. Introduces the core scatter
architecture that all later variants inherit: bucket-scatter dispatch
replacing `torch.argsort`, inline exclusive cumsum in every consumer,
count-atomic merged into routing kernel at T≤256. Kept in archive for
cross-reference; superseded by v2 and by `fused_graph_all_t`.

Under the canonical baseline v1 dropped from a prior-header 1.022x to
1.000x (−0.02x). The shift comes from the baseline replacement — the
pre-canonical global cache (MD5 `836840d3…`) had 20–25% slower small-T
expert latencies than the canonical reference, which inflated all
variants' headline speedups. Canonical is now authoritative; do not
replace without re-measuring every variant (see `baseline.json`
provenance block).

## A/B baseline

**`variants/presync_v1/`** — 0.842x ± 0.001x (3-run, CV 0.10%). Immediate
predecessor to v1 (iter-30 snapshot). Retained to A/B-isolate the three
launch-overhead wins v1 layered on top (iter-31 skip-sync T≤256, iter-34
inline cumsum, iter-37 merge count atomic into routing).

## Fallbacks

_(none — single pure-Triton path; no CuTe DSL / TileLang / deep-gemm /
flashinfer runtime dependencies.)_

## Canonical baseline

`baseline.json` — 19 workloads, `source: "expert"` (flashinfer
`trtllm_fp8_block_scale_moe`). Denominator for all variant speedups.
MD5 `a1d2be64…`, measured 2026-04-08 (spawn.py pre-cache). This file is
the single source of truth. Do not replace without re-measuring every variant — the
full provenance is in the `provenance` block of the JSON.

## Cross-variant traps

See `TRAPS.md` for toolchain / methodology facts that apply regardless
of which variant is anchor. The 2026-04-23 merge added four new entries
re-confirmed by moe4/moe5:
- **Atomic ordering**: `sem="relaxed"` on MoE scatter-add is the single
  biggest lever on this operator (+0.077x headline, +0.26-0.33x large-T)
  when NCU shows the atomic-writing kernel DRAM-util-bound.
- **Python allocator at small-T**: `torch.empty` / `.clone()` /
  `.record_stream()` cost 5-7 µs each — 3-13% of headline at T ≤ 80.
- **Dual-dot SwiGLU is a Triton 3.6 sm_100 codegen trap**: abs_err
  ~7-8 × 10⁵ at large T, sanitize-clean so not OOB — do not retry.
- **NUM_STAGES=7 overflows shmem at BM=BN=BK=128 FP8**: NS=6 is the
  hard ceiling.
- **Individual A/B deltas don't compose**: re-confirmed at 2.7× and 2.3×
  under-report in moe5 and moe4 respectively (and again at ~2.5× in
  moe_v0).

The 2026-04-25 moe_v0 session added two more:
- **`BN=256` is unusable on BM=128 too** — closes TRAPS §5's open
  question. Same fp8 UMMA codegen hazard as BM=64 + BN=256; sanitize
  pattern match (memcheck clean, codegen bug). Do not retry without a
  Triton fp8 MMA codegen fix.
- **Two-sequential-K-loop fusion does NOT recover dual-dot pipelining** —
  TRAPS §9's suggested workaround preserves correctness but each loop
  pays its own pipeline setup+drain. At T=14107 the fused kernel was
  +270µs slower than the anchor's separated GEMM1 + SwiGLU kernels. The
  fusion only wins if dual-dot codegen is fixed upstream or A can be
  shared via shmem manually.
- **SwiGLU `num_warps` is M_pad-conditional** — nw=2 scheduler-starves
  the kernel at small/mid M_pad (Warp Cycles Per Issued ≫ 5); nw=4 fixes
  that but creates DRAM contention at large M_pad. Threshold `nw=4 if
  M_pad < 2048 else 2` captured both regimes in moe_v0 iter-4 (+0.005x
  headline, all 19 workloads positive).

## Naming

`spawn.py`'s longest-prefix match (hyphen ↔ underscore) picks this
archive for any `moe_fp8_block_scale_*` family operator. If the family
diverges (different E_global / H / I), add a sibling archive at a more
specific prefix rather than retrofitting this one.
