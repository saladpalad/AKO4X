# Iteration Log

Append a Summary row for every labeled bench. That's the only required action.

When a change is worth pre-committing a hypothesis to (architectural shift, sweep you want to track, dead-end probe), write `Expected: ...` in `## Notes` BEFORE running the bench. The point is to catch retrofitted explanations afterward — no required format. Good `Expected` lines name a mechanism, an affected dimension, and a predicted delta (e.g. "Fusing X saves 1 launch × 19 workloads — expect +2-3% on small seq only").

At session end, write a brief synthesis in `## Notes`: kernel state, remaining bottlenecks, dead ends to skip ("needs tuning" ≠ dead end), what's worth trying next session.

## Summary

| Iter | Title | Score | Passed | Notes |
|------|-------|-------|--------|-------|
| 1 | initial-triton split-KV flash-decode | 0.717x | 47/47 | b=1→0.373x b=16→0.954x b=64→0.804x; cliff at b=1 kv∈[608,2708]: 0.04-0.09x |
| 2 | no-static-range reduce + skip-reduce-when-1 | 0.686x | 47/47 | b=1→0.529x (+42%, cliff fixed: kv=2708 0.056→0.284x) but b=16→0.762x (-20%) regression — runtime range slower at small NUM_SPLITS than iter-1's unrolled static_range |
| 3 | cap-at-16 splits + revert reduce to unrolled+two-pass | 0.971x | 47/47 | b=1→1.04x b=16→1.07x b=64→0.813x; cap-16 keeps reduce-loop unrolled cheap, fixes both prior issues. b=64 large-kv now slowest (kv=75145 0.533x; kv=22745 0.387x) |
| 4 | buffer cache (o_part/lse_part) | 0.905x | 47/47 | A/B vs iter-3: Δ=-0.20% (noise). Confirmed-neutral. Standalone -0.066 score is pure session drift. Keeping the cache (no harm, marginal future amortization). |
| 5 | num_warps=8 + num_stages=3 | 0.611x | 47/47 | REGRESSION across all groups: b=1→0.661x b=16→0.665x b=64→0.535x. Latency ~50% worse on b=64 kv=75145 (138→209µs). Reverted. num_warps=8 hurts at M=16 matmul (warps idle). num_stages=3 may have evicted occupancy via smem pressure. |
| 6 | BLOCK_N=32 | 0.877x | 47/47 | REGRESSION b=64→0.644x. Doesn't eliminate spill (acc is the bulk) and adds 2x iter overhead. Reverted. NCU on iter-3 w/ index 41 confirmed split kernel @ 254 regs/thread (max) + 34048 spill reqs / 100% overhead → 9% mem throughput, 12% theoretical occupancy. Next: D-tile across CTAs to drop persistent acc (32KB) to half. |
| 7 | D-tile 2-way across CTAs | 0.677x | 47/47 | REGRESSION b=64→0.449x (138→249µs on kv=75145). Doubling K_NOPE load (full for logits + chunk for value) doubles BW; the spill reduction doesn't pay back. Reverted. Lesson: D-tile needs K-reuse which Triton doesn't expose cleanly. |
| 8 | num_splits target=512 (splits=8 for b=64) | 0.931x | 47/47 | A/B vs iter-3: **+3.90% confirmed win**. b=64 +0.111x (0.81→0.90x); top movers 5bef8d88 +0.26x, 1c3743b9 +0.21x, 939f995a +0.21x. b=1/b=16 unchanged code (cap=16 hits before target/batch). True score ~1.01x. Worst b=64 latency on kv=75145 dropped 138→94µs. |
| 9 | num_splits target=1024 (splits=16 for b=64) | A/B Δ=-1.93% | 47/47 | b=64 -9.91% via A/B (regression). Reduce kernel unrolling 16 iters of 32KB fp32 reads grew faster than per-CTA work savings. splits=8 is sweet spot for b=64. Reverted. |
| 10 | cache output/lse buffers | A/B Δ=-0.07% | 47/47 | Neutral. torch.empty cost on output/lse already negligible. Reverted to keep code clean. |
| 11 | num_warps=8 for b=64 split kernel | A/B Δ=-3.47% | 47/47 | b=64 -9.25% via A/B (regression). M=16 matmul doesn't scale with more warps even when CTA count is large. Reverted. |
| 12 | bf16 partial output (was fp32) | A/B Δ=**+10.68%** | 47/47 | **Major win.** b=1 +10.2%, b=16 +11.0%, b=64 +10.9% (b=64 broke 1.0x: 0.932→1.04x). Halves partial alloc + reduce-kernel BW. Precision OK (output still bf16 anyway, partial is acc/l_i softmax-weighted average bounded by v values). Effective score ~1.11x. |

## Notes

### iter-1 (initial-triton split-KV flash-decode, 0.717x)
Baseline is `flashinfer_wrapper_03f7b0` (BatchMLAPagedAttentionWrapper), so reference latency 11–73 µs.

Per-group: b=1→0.373x (avg 0.195ms), b=16→0.954x (0.026ms), b=64→0.804x (0.067ms).

**Cliff observed at b=1**: kv∈[8..508] runs in 12–20 µs (~0.7–0.95x), then JUMPS to 200 µs at kv=608, 400 µs at kv=2708. The cliff coincides with `_choose_num_splits` returning ≥32 (kv=608 → 32 splits; kv=2708 → 64 splits). Strongly suspect reduce kernel pathology from `tl.static_range(NUM_SPLITS)` unrolling 32–64 iterations of 32 KB fp32 loads — likely register spilling / I-cache thrash.

Expected (iter-2): switch reduce loop to runtime `range(NUM_SPLITS)` (constexpr NUM_SPLITS but compiler chooses not to unroll). Should remove the cliff and push b=1 large-kv from 0.04–0.09x toward >0.5x. Marginal effect on b=16/b=64 since those use NUM_SPLITS ≤ 8.

Actual (iter-2): cliff partially fixed (kv=2708 0.056→0.284x, not the >0.5x I hoped). Surprise: b=16 *regressed* from 0.954x→0.762x — the runtime loop is slower than the unrolled static_range at small NUM_SPLITS. Lesson: unroll-vs-loop is a regime tradeoff at NUM_SPLITS≈16.

Iter-3 lesson: capping NUM_SPLITS at 16 + keeping constexpr/unrolled reduce gets the best of both worlds (kv=2708 hit 0.938x). Cliff is fully gone.

### iter-3 (cap-16, 0.971x)
b=64 now the bottleneck. Worst workloads:
- 5bef8d88 (kv=75145, ~1174/seq): 138µs / 0.533x
- 939f995a (kv=22745, ~355/seq): 139µs / 0.387x
- 1c3743b9 (kv=68745, ~1074/seq): 143µs / 0.416x

For b=64 the num_splits cap gives splits=4. Total 256 CTAs. Possible levers: BLOCK_N=128 (fewer iters / chunk), num_warps=8 (more memory parallelism per SM), buffer cache for the 8MB partial allocation.

Expected (iter-4): buffer cache on o_part/lse_part. Saves ~10-20µs per call (esp. b=64 with 8MB partial). Broad +1-3% on score.

Actual (iter-4): A/B vs iter-3 shows Δ=-0.20% (noise). Not the win I expected. Hypothesis on why: torch.empty for fp32 partials is cheaper than I estimated, or the cost is hidden by other overheads. Kept the cache (harmless).

### iter-5/6 dead ends
- num_warps=8 + num_stages=3 (iter-5): -0.36 score. M=16 too small for many warps; smem pressure hurt occupancy. Lesson: increase parallelism via grid not warps for this shape.
- BLOCK_N=32 (iter-6): -0.10 score on b=64. acc is the spill culprit, not k_nope (per NCU). Reducing BLOCK_N doesn't help acc — only adds iter overhead.

### iter-7 (D-tile, 2-way)
Expected: split CTAs across pid_d ∈ {0,1}, each owning half of acc (16, 256). Halves persistent register footprint. NCU baseline @ iter-3 showed 254 regs/thread + 34048 spills/100% overhead. If D-tile drops to ~220 regs/thread we may avoid the worst spills, lift b=64 from 0.81x toward 1.0x. Costs: full K loaded twice (once full for logits, once D-chunk for value); 2x CTA count; full Q load redundant across pid_d. Net expected +0.05 to +0.10 if spill reduction lands.

Actual (iter-7): 0.677x — major regression across all groups, b=64 → 0.449x (was 0.813). Latency on b=64 kv=75145 went 138 → 249 µs. The double-K-load cost (full + chunk) ~doubles memory traffic and dwarfs any spill reduction. Q still 16KB in regs, so spill likely not even meaningfully reduced. Reverted.

Lesson: D-tile across CTAs only pays off if K can be reused (smem-resident) between the logits matmul and the value matmul. With Triton's current API, no clean way to do this without re-loading. Dead end unless we materialize K to smem via a single load pattern that Triton recognizes.

### Plateau analysis
iter-3 (0.971x) remains best. NCU shows the bottleneck is register spilling in the split kernel (254 regs/thread, 34048 spill reqs). Spill-reduction strategies tried so far:
- BLOCK_N=32: -0.09 (k_nope shrank but acc is the real culprit, plus more loop overhead)
- num_warps=8: -0.36 (M=16 matmul wastes extra warps)
- D-tile across CTAs (iter-7): -0.29 (doubles K load BW)

Other angles to explore (in priority order):
1. **Atomic-add reduce** (skip the separate reduce kernel — save 1 launch + 1 fp32 R/W pass).
2. **Larger num_splits for b=64** (the cap=16 may be too restrictive; try splits=8 for b=64).
3. **Different op-ordering** to let Triton stage K in smem and reuse for value matmul.
4. **bf16 partial output** (halve reduce-kernel BW; acc stays fp32 internally).
