# rmsnorm-h128 — Archive

Working kernel variants preserved from optimization sessions on the
`rmsnorm_h128` operator (RMSNorm with hidden_size=128, bf16 IO,
captured from Qwen3-30B-A3B). Variable axis is `batch_size`; the
workload set spans **B = 4 → 520128** (14 workloads).

Campaign started 2026-05-20 on Modal B200, CUDA 13.0. Last updated
2026-05-27 (round 4 closed; campaign ended — see "Campaign closure"
below).

## Anchor

**`variants/triton_tiered_rows/`** — ~1.22-1.23x drift-corrected (Modal
B200, CUDA 13.0, Triton 3.6.0, 2026-05-20). 14/14 workloads pass.
Single Triton kernel with the (BLOCK_ROWS, num_warps) pair picked from B
in a `_pick` table; H=128 baked constexpr so the row reduction fully
unrolls and ld.global.b128 vector loads emit. Two refinements on top of
the base shape: `num_stages=2` (vs Triton's default 3 — no inner K-loop
to amortize the deeper pipeline) and a constexpr `ALIGNED` fast-path
that skips the row mask when `B % BLOCK_ROWS == 0` (12/14 workloads
qualify under the chosen tiers).

Three regime-distinct wins compound to the headline:

1. **Tiny-B (≤256, 8 workloads): 1.0-1.17x** — `BLOCK_ROWS=1` puts
   one row per SM on the B200's ~148 SMs, parallelising the per-row
   load latency that bounds aggregated-block alternatives.
2. **Mid-B (316-2528, 4 workloads): ~1.0x** — flat against the
   hand-tuned FlashInfer kernel; not closable by simple BLOCK_ROWS /
   num_warps sweeps. This is the remaining structural gap (see "Open
   directions" in the variant header).
3. **Huge-B (49532-520128, 4 workloads): 1.43-1.69x** — HBM-bound;
   `BLOCK_ROWS=32, num_warps=4` delivers ~6.5 of B200's ~8 TB/s peak.
   NCU's "raise occupancy" advice misled iter-7 here (-4.87%);
   per-thread ILP wins over concurrent-blocks occupancy for streaming
   register-resident reduces.

Variance check (iter-2 base config): 1.2046 ± 0.0031 (CV 0.3%, n=3),
with iter-11 (num_stages=2, +0.6% A/B) and iter-12 (ALIGNED fast-path,
+2.3% A/B) stacked on top.

## Canonical baseline

`baseline.json` — captured during round-1 iter-1 on Modal B200 / CUDA
13.0 (auto-promoted from the Round-0 environment-only shell). 14
workloads, `source: "expert"`.

## Campaign closure

Four rounds (2026-05-20 → 2026-05-27). The round-1 `triton_tiered_rows`
kernel remains the anchor; rounds 2, 3, and 4 produced no kernel
improvement and each closed a separate structural lever:

- **Round 2** probed *regime specialization* — separate kernels for
  tiny / mid / huge B, dispatched at launch time. Closed: persistent
  kernel for mid-B regressed −15%, 2-kernel ALIGNED-main + masked-
  cleanup split for unaligned huge-B regressed −5.4%, hand-written
  CUDA mirror of the huge-B path reached only Triton parity (−2.2%
  best). Forensic value: corrected the anchor's lesson-2 calibration
  (added NCU readings: 40 regs / 12 CTAs/SM / 75% theoretical / 66%
  achieved occupancy / 59% L1TEX-scoreboard stall) so future rounds
  can't invert "ILP > occupancy" into "Triton runs at low occupancy."
- **Round 3** probed *reduction shape* — persistent grid-stride with
  inner `tl.range` num_stages pipelining, TMA via
  `tl.make_tensor_descriptor`, and `tl.range(warp_specialize=True)`.
  Closed: persistent regressed −10% headline (HW concurrent-CTA
  scheduling delivers more in-flight memory than compiler
  num_stages can replicate at this 8 KB tile); TMA needed a fat
  tile and never beat the anchor at h=128; `warp_specialize=True`
  aborted in Triton 3.6's ttgir pass on streaming-reduce kernels.
  The Triton 3.6 advanced-load lever restrictions were promoted to
  the global Triton skill doc via the round-3 phase-2 proposal.
- **Round 4** (mode-2) probed *load-issue substrate* — same row-per-
  program reduction shape, three alternative CUDA / inline-PTX load
  paths against Triton's emitted `ld.global.cs.v4.b32`. Substrate
  ordering established empirically (vs anchor at huge-B):
  `ld.global.cs` (anchor) > `cp.async.cg.shared.global` (iter-2:
  −4-14% headline, B=49532 worst at 1.33 vs anchor 1.475) >
  `ld.global.nc` / `__ldg` (iter-3: 1.09x headline, B=49532 1.11).
  iter-4's direct inline-PTX `ld.global.cs.v4.b32` + predicated
  unaligned trailing-block reached **drift-cancelled parity-minus
  2.27% A/B vs anchor** — aligned huge-B within 2% of anchor
  (1.63/1.64 vs 1.66/1.69), unaligned huge-B (B=49532, 65016) still
  trails the anchor by Δ +0.20 / +0.12 because Triton's vectorized
  mask-load emission for the trailing CTA is structurally better-
  scheduled than the per-row inline-PTX equivalent under NVCC.
  Forensic value: DRAM-bandwidth analysis pinned the headline ceiling
  — anchor at B=520128 moves X+Y = 0.266 GB in 41.5μs = **6.4 TB/s ≈
  80% of B200's 8 TB/s peak HBM**, so the 59% L1TEX-scoreboard stall
  is the NCU signature of an *already DRAM-saturated* kernel; load-
  variant choice (`.cs` / `.cg` / `.nc` / cp.async) varies how the
  scoreboard accounts in-flight memory but not the throughput ceiling.
  Also closed: the mid-B (B ∈ {316, 1088, 2528}) ~1.0x ceiling vs
  FlashInfer is the per-call launch-overhead floor (~2.3μs ref
  latency both kernels hit), not closable from any load axis. No
  phase-2 proposals (mode-2); the load-substrate-ordering finding
  lives in this README only — substrate ordering is operator-
  specific (already-HBM-bound streaming reduce), not a general
  enough mechanism for the Triton skill doc.

Conclusion: the load-substrate, reduction-shape, and tile-knob axes
are all closed within ≲2-3% of the anchor. The headline ceiling is
the operator itself at 80% HBM-bandwidth efficiency on B200, not
choices the round-1 kernel made. Reopening this campaign genuinely
requires either (a) a different Triton version that accepts
`warp_specialize` on streaming-reduce shapes; (b) a multi-kernel
pipeline with PDL that pipelines partials across launches; (c)
matching Triton's vectorized-mask emission for unaligned huge-B in
CUDA (would recover ~2-3% on B=49532/65016 — bounded upside); or (d)
something genuinely new at the algorithm-form level. Re-running any
lever closed in r2/r3/r4 won't move the headline.
