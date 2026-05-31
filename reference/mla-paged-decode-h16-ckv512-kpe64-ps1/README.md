# MLA Paged Decode — Archive

Working kernel variants preserved from optimization sessions on the
`mla_paged_decode_h16_ckv512_kpe64_ps1` operator (Multi-head Latent
Attention paged decode, DeepSeek-V3 TP=8 → 16 query heads, ckv=512 /
kpe=64, page_size=1).

Campaign started 2026-05-19 on Modal B200, CUDA 13.0. Last updated
2026-05-20 (round 4 closed).

## Anchor

**`variants/triton_split_kv_2d_reduce_pdl_rw2/`** — 1.46x labeled-
single-run (iter-9, 47/47 passed, Modal B200, CUDA 13.0, Triton 3.6.0,
2026-05-20). Same-session A/B vs the prior anchor: **+17% cumulative**
(PDL +11.41% then reduce-num_warps=2 +5.91%, both confirmed in-
container). Group breakdown after r3: b=1 ~1.55x, b=16 ~1.50x,
b=64 ~1.18x. No multi-session variance verification yet.

Two compounding wins stacked on `triton_split_kv_2d_reduce`:

1. **PDL between split → reduce kernels** — `gdc_launch_dependents()`
   at split-end + `gdc_wait()` at reduce-start + `launch_pdl=True` at
   both call sites. The overlap window comes from address/constant
   prep in reduce + the split kernel's tail wave on adjacent SMs.
   Biggest on small batches where reduce is a non-trivial fraction
   (b=1 +15%, b=16 +13%, b=64 +4.5%).
2. **Reduce kernel `num_warps=2`** (was 4) — for the tiny reduce tile
   (H=16, D_CHUNK=32, NS=8 or 16) the original 4 warps × 128 threads
   was over-paralleled and underused the load pipe; 2 × 64 matches
   the load pipe. `num_warps=1` also helps but less (+4.71%);
   `num_warps=8` is -22%. Biggest on b=64 (+12%).

## Candidate sibling (marginal — variance-check pending)

**`variants/triton_split_kv_2d_reduce_pdl_rw2_brdispatch/`** — 1.48x
labeled-single-run (r4 iter-8, 47/47 passed). Anchor + one conditional
reduce-tile dispatch: `D_CHUNK=64 nw=4` for `b≥16`, `D_CHUNK=32 nw=2`
(parent setting) for `b=1`. **Headline Δ +1.4% over anchor is within
Modal session noise (~5-15%)**; per-batch signal IS real (b=16 +8-11pp
1.50 → 1.61), b=64 +1-2pp, b=1 noise-bound. **Not yet variance-verified
across sessions** — promote to anchor only after a 5-run variance check
shows the headline is real. Until then, keep the parent
`triton_split_kv_2d_reduce_pdl_rw2` as the canonical anchor.

## Fallback

**`variants/triton_split_kv_2d_reduce/`** — 1.21x. Prior anchor; same
underlying split-K + 2D-tile-reduce structure, without PDL or the
reduce-warp tuning. Keep as the algorithmically equivalent
non-PDL fallback (in case a future PTX/Driver issue regresses PDL).

## History

- **round 1 (2026-05-19)** — Brand-new family. Phase-1 timed out at
  10800s. Sub built a Triton split-KV flash-decode from scratch through
  12 committed iterations, peaking at iter-12 (bf16 partial output) for
  AB+10.68% over its own iter-3 baseline. No archive_variant per
  strict step-4 protocol; iter-12 kernel preserved as the round-2 seed.
  Failure bundle in `_failed/mla-paged-decode-r1/`.
- **round 2 (2026-05-20)** — Seeded from r1 iter-12 (1.09x first labeled
  bench). 8 iterations, 2 wins (iter-3 +11%, iter-5 +1% headline /
  +17–23% on b=64 large), 5 dead ends. Final 1.22x effective / 1.21x
  bench iter-8. `triton_split_kv_2d_reduce` archived. Phase-2 yielded
  one accepted harness edit (triton skill: store-coalescing floor).
- **round 3 (2026-05-20)** — Seeded from the r2 anchor. 9 iterations,
  2 confirmed A/B wins (iter-5 PDL +11.41%, iter-9 reduce num_warps=2
  +5.91%), 6 dead ends. Final **1.46x** (iter-9). Notably refuted
  anchor open-direction #1 (persistent / chunked-split kernel: -33% on
  b=64; this kernel is latency-bound and needs MORE in-flight CTAs,
  not fewer). `triton_split_kv_2d_reduce_pdl_rw2` archived and rotated
  to anchor. Phase-2 yielded one accepted harness edit (profiler-ncu
  skill: "No kernels were profiled" has TWO common causes, NVTX scope
  vs CUDA graph capture).
- **round 4 (2026-05-20)** — Seeded from the r3 anchor. 8 iterations,
  1 marginal headline win (iter-5 conditional reduce-tile dispatch
  D_CHUNK=64 nw=4 for b≥16: +0.97% A/B headline, +8-11pp b=16, within
  drift on b=64). 5 lanes closed (per-call-site BLOCK_N=32 dispatch is
  neutral; split num_stages=3 -10% occupancy crush; reduce D_CHUNK=128
  nw=8 regresses past D_CHUNK=64 ceiling; reduce nw=8 at D_CHUNK=64
  over-parallels). Final **1.48x** archived as marginal sibling
  `triton_split_kv_2d_reduce_pdl_rw2_brdispatch` — NOT rotated to anchor
  pending variance verification. Sub explicitly identified the
  Triton-side structural plateau (split kernel occupancy-pinned at
  12.5% theoretical, reduce kernel at tile-warp sweet spot). Phase-2
  yielded one accepted harness edit (bench-utils: `--ab-compare`
  with `--label` now saves the B-side run as a trajectory snapshot,
  enabling chained AB-compares).

## Open directions (carry-forward)

The campaign has likely reached the Triton-side structural plateau —
split kernel occupancy-pinned at 12.5% theoretical, reduce kernel at
its tile-warp sweet spot. The two remaining bad b=64 workloads
(`939f995a` at 0.84x avg=974, `1c3743b9` at 0.92x avg=1074) appear
algorithm-class-limited; the expert reference is structurally
different on those shapes (likely a flash-decode that fuses
logits/value into a single pass). r4 closed the cheap lanes
(BLOCK_N=32 dispatch, deeper pipeline, larger reduce tile).

Remaining lanes (engineering-heavy, uncertain payoff):

1. **TileLang or CuTe DSL single-pass flash-decode** — fuse split→reduce
   into one kernel that keeps K smem-resident across logits and value
   matmuls. Likely what the expert reference does on the worst b=64
   shapes. ~2h to first benchable state; **beware `@cute.kernel.launch`
   + `torch.cuda.graph` capture** (the known sibling-family TRAPS in
   this repo).
2. **Persistent reduce with batch-tile (B_TILE=2)** — for b=64
   specifically, each reduce CTA processes 2 batches' worth of one
   D-chunk. Halves reduce CTA count to 256 → eliminates the tail wave.
   Smaller scope (~30 min). Predicted +2-3% on b=64.
3. **FP8 K cache** — halves K-gather BW. But K-gather is L1TEX-latency-
   bound, not BW-bound, so unlikely to move scatter-gather perf;
   correctness gymnastics are real. Lower-priority lane.
4. **Variance-check (5-run) of the brdispatch sibling vs the anchor**
   — decisive on whether the +1.4% headline is real. Cheap, ~30 min
   Modal time, gates promotion to anchor.

Refuted (do not retry without new framing):
- Persistent kernel that REDUCES in-flight CTA count (r3 iter-1: -33%
  on b=64; latency-bound kernel needs MORE CTAs, not fewer).
- Per-call-site BLOCK_N=32 dispatch (r4 iter-0/1: within drift, no
  signal in either direction).
- Reduce D_CHUNK shrinkage to clear register budget (r2 iter-6: -22%
  on b=64, hit the store-coalescing floor — see triton skill).
- Reduce / split kernel `num_warps=8` (multi-round: -11 to -22%, MMA
  tile picker degrades; see anchor + sibling headers).

See each variant's `kernel.py` header for variant-specific lessons and
dead-ends (per `templates/agent/lessons-convention.md`).
