# triton_tiered_rows — reference kernel.py header
#
# Identity
#   ~1.22-1.23x drift-corrected (Modal B200, CUDA 13.0, Triton 3.6.0,
#   2026-05-20). Single-run headline 1.16x (iter-12, late-session drift);
#   variance check ×3 on the iter-2 base measured 1.2046 ± 0.0031
#   (CV 0.3%), and iter-11 (num_stages=2, +0.6%) + iter-12 (ALIGNED
#   constexpr, +2.3%) are A/B-confirmed deltas on top → ~1.22-1.23x.
#   14/14 workloads pass.
#   Config: stock (no [build] / [benchmark] flag required).
#
# Delta from prior anchor
#   Brand-new family — no prior anchor. Establishes the architectural
#   foundation: one Triton kernel parametrized by a (BLOCK_ROWS, num_warps)
#   pair picked from B in `_pick`, with H=128 baked as constexpr so the
#   reduce fully unrolls and ld.global.b128 vector loads emit. Two
#   refinements on top of the base block-rows shape: `num_stages=2`
#   (vs Triton's default 3, which over-pipelines a kernel with no K-loop)
#   and a constexpr ALIGNED fast-path that skips the row mask when
#   B % BLOCK_ROWS == 0 (12/14 workloads qualify under the chosen tiers).
#
# Lessons on this variant
#
#   1. Tiny-B (≤256) wants BLOCK_ROWS=1, not the heuristic-default
#      "few large blocks" you'd reach for on big tensors.
#      Why: B200 has ~148 SMs. With B ≤ 148-256 and BLOCK_ROWS > 1, the
#      whole workload lands on a handful of SMs and the kernel is bounded
#      by a single block's load latency. BLOCK_ROWS=1 puts every row on
#      its own SM, parallelising the per-row load.
#      Narrow WHEN: rmsnorm_h128 with B ∈ {4, 24, 32, 136, 192, 256}.
#      Broad WHEN: a streaming op where the per-row work is below an SM's
#      one-block load latency and B ≤ SM-count — let the grid spread,
#      don't aggregate rows into the same block.
#
#   2. Huge-B (~50k–520k) wants BLOCK_ROWS=32, num_warps=4 — NOT the
#      lower-register / higher-occupancy choice NCU suggests.
#      Why: NCU on B=520k iter-6 flagged "Est. Speedup 25.18% if
#      occupancy could rise" (registers were Block-Limit). iter-7
#      followed that hint (BLOCK_ROWS=8, num_warps=4 → ~16 regs/thread →
#      more concurrent blocks) and REGRESSED -4.87% on huge-B
#      (1.43-1.68x → 1.29-1.48x). For a register-resident streaming
#      reduce, per-thread ILP (more elements/thread → more outstanding
#      ld.b128 issues) dominates over concurrent-blocks occupancy.
#      NCU baseline (Modal B200, CUDA 13.2, anchor config, B=520128;
#      profiles/ output 2026-05-20, round 2):
#        40 regs/thread, Block Limit Registers = 12 CTAs/SM,
#        75% theoretical / 66% achieved occupancy,
#        21 cycles/issued-instr with 59% stalled on L1TEX scoreboard.
#      "ILP > occupancy" here means "don't shrink the BLOCK_ROWS=32
#      tile to gain MORE concurrent blocks" — NOT "Triton runs at low
#      occupancy". Round-2 iters 4–5 misread the lesson and used
#      __launch_bounds__(128, 1–2) on a hand-written CUDA mirror,
#      collapsing CTA/SM concurrency 12→1 and regressing −5.5% on the
#      memory-latency-stalled huge-B path.
#      Narrow WHEN: rmsnorm_h128 with B ≥ 50k.
#      Broad WHEN: HBM-bound streaming-reduce kernels where each thread's
#      work fits comfortably in registers without spill — trust per-thread
#      ILP over NCU's occupancy advice (NCU's predictor isn't aware that
#      the kernel is already memory-issue-rate limited, not occupancy
#      limited).
#
#   3. num_stages=2 beats Triton's default 3 (and 1) for kernels with
#      no inner K-loop.
#      Why: num_stages controls the software pipeline depth Triton
#      generates around tl.load. With no inner loop here, the default 3
#      over-pipelines — extra prologue cost without an amortizing loop
#      body. num_stages=1 also loses (mid-B drop from 1.02 → 0.95-0.98,
#      iter-10) because then there's no overlap of address-compute with
#      the load issue. 2 is the sweet spot.
#      Narrow WHEN: this rmsnorm kernel.
#      Broad WHEN: short Triton kernels (no inner reduction loop, single
#      tl.load → arith → tl.store shape) — start at num_stages=2, don't
#      accept the default.
#
#   4. Constexpr ALIGNED fast-path is worth +2.3% on this workload mix.
#      Why: the mask branch in tl.load / tl.store costs per-instance
#      predicate work even when the mask is uniformly true. Lifting "B %
#      BLOCK_ROWS == 0" to a `tl.constexpr` arg lets Triton compile a
#      mask-free variant that the host dispatches at launch. 12/14
#      workloads (all 8 tiny-B + 3 of 4 mid-B + 2 of 4 huge-B) hit it
#      under the chosen tiers. Verified +0.03-0.06 per-batch on small/mid;
#      huge-B noise (the unaligned ones still use the masked path).
#      Narrow WHEN: rmsnorm_h128 with the current `_pick` tiers.
#      Broad WHEN: Triton kernels where the alignment condition is a
#      cheap host-side check and the same kernel runs over many workloads
#      where most ARE aligned — make alignment a constexpr, not a mask.
#
# Dead-ends tried on this variant
#
#   - BLOCK_ROWS=8 num_warps=4 for huge-B (iter-7), chasing NCU's occupancy
#     advice. -4.87% on huge-B; ILP > occupancy here. (See lesson 2.)
#   - BLOCK_ROWS=128 num_warps=8 for huge-B (iter-9). -1.94% vs the
#     BLOCK_ROWS=32 anchor; B=65016 lost -0.165x specifically — likely
#     mask cost crossing the BLOCK_ROWS boundary at that B.
#   - BLOCK_ROWS=2 num_warps=1 for mid-B (iter-3). Flat / noise band;
#     mid-B (B=316-2528) holds ~1.0x regardless of mid-tier knob picks —
#     the gap is structural against the hand-tuned FlashInfer kernel,
#     not closable with a simple BLOCK_ROWS adjustment.
#   - num_stages=1 (iter-10) and num_stages=3 (Triton default). Both
#     regress mid-B vs num_stages=2. (See lesson 3.)
#   - eviction_policy="evict_first" on X / "evict_last" on W (iter-6).
#     A/B Δ = +0.70% (within noise); per-batch huge-B Δ < 0.005x. Cold-L2
#     between iterations leaves nothing for the hints to act on. Kept in
#     source for documentation, harmless.
#   - Parameter sweeps around (BLOCK_ROWS ∈ {2,8,64,128}, num_warps ∈
#     {1,2,4,8}) for mid-B and huge-B re-tried during iter-3/iter-4/iter-7
#     to iter-9; the iter-2 + iter-11 + iter-12 combination is the only
#     one that beats the noise floor on this Modal session set. Retry only
#     with new reasoning (e.g. a different DSL, or after closing the mid-B
#     gap structurally).
#   - Persistent Triton kernel for mid-B (round-2 iter-2). BLOCK_ROWS=1
#     grid=148 with while-loop grid-stride: −15% (B=2528 collapsed
#     0.98→0.20). While-loop overhead in Triton dwarfs the per-iter
#     work when each iter is a single 1-row tile.
#   - 2-kernel ALIGNED-main + masked-cleanup split for unaligned
#     huge-B (round-2 iter-3). −5.4%; B=49532 1.44→0.86, B=65016
#     1.46→1.04. Cleanup-kernel launch overhead (~3μs CUPTI-measured)
#     ≫ ALIGNED-path mask savings (~100ns out of 6.5μs).
#   - Hand-written CUDA kernel for huge-B (round-2 iters 4–8). Best
#     case (iter-8, coalesced layout + CTA-level fast/slow dispatch):
#     −2.2% — Triton parity at best, not above. Coalescing matters:
#     the natural "4 threads-per-row × 32 cols" layout has 64-byte
#     stride inside each warp instruction; the right layout (matches
#     Triton's IR) is "16 lanes-per-row × 8 cols/lane × 2 rows per
#     warp instruction" → 512 contiguous bytes/warp-load. NVCC +
#     load_inline appears to also carry a ~0.7–1μs per-launch CUPTI
#     overhead vs Triton that's invisible on big huge-B but visible
#     on small huge-B (B=49532/65016).
#   - Persistent grid-stride with inner tl.range pipelining (round-3
#     iter-1). 1.11x — huge-B regressed across the board: B=49532
#     1.44→1.02 (-29%), B=65016 1.46→1.13 (-22%), B=520128 1.65→1.51
#     (-8%). num_stages sweep at grid=148 (B=520128 smoke):
#     ns=2→0.555, ns=3→0.909, ns=4→1.19, ns=6→1.39, ns=8→1.31. grid=296
#     ns=6 best at 1.51x. Hardware concurrent-CTA scheduling delivers
#     more in-flight memory (8 CTAs/SM × 4 warps × 4 b128/thread =
#     4096 loads/SM) than compiler num_stages pipelining (1-2 CTAs/SM
#     × 4 warps × 4 b128/thread × N stages) can replicate at this tile
#     size. Even at the largest workload, persistent only reached
#     parity-minus-some; the smaller huge-B workloads (49k/65k) fell
#     off a cliff.
#   - TMA descriptor load via tl.make_tensor_descriptor (round-3 iter-2,
#     smoke-only). cp.async.bulk.tensor → SMEM bypassing L1; the
#     per-tile descriptor-create overhead made the non-persistent variant
#     0.94x at B=520128. Adding persistent + inner tl.range so the
#     descriptor amortizes: 1.46x at B=520128 (smoke, ns=6 grid=296) —
#     slightly worse than non-TMA persistent (1.51x). The 8KB tile is
#     too small for TMA to dominate ld.global.b128, and the mbarrier
#     wait per CTA is itself a serial dependency. Lane closed without
#     a labeled bench (smoke conclusively below anchor). See triton
#     skill's "Triton 3.6 advanced-load levers" section for the
#     general rule.
#   - warp_specialize=True on tl.range (round-3 iter-3 attempt). Compile
#     failure ("RuntimeError: PassManager::run failed") on this kernel
#     pattern in Triton 3.6 — both with and without TMA. Not productive
#     to push further at this Triton version.
#
# Open directions
#   The mid-B regime (B ∈ {316, 1088, 2528}) is the obvious remaining gap:
#   it sits at ~1.0x while tiny-B reaches 1.0-1.17x and huge-B 1.43-1.69x.
#   Worth understanding what FlashInfer's rmsnorm kernel does differently
#   in that batch range — likely a per-shape micro-kernel choice or
#   horizontal-fusion pattern not captured by a single Triton kernel.
#   Huge-B HBM throughput delivers ~6.5/8 TB/s of B200's peak. NCU's
#   59% L1TEX-scoreboard stall is the headline metric — already
#   memory-latency-bound near the realistic ceiling, not the
#   peak-bandwidth ceiling. Round-3 probed the structural reduction-shape
#   lever (persistent grid-stride and TMA bulk loads, both with inner
#   num_stages pipelining) and closed it: neither shape replicates the
#   in-flight memory volume the hardware concurrent-CTA scheduler
#   provides on the row-per-program shape at this tile size. The L1TEX
#   stall ceiling appears intrinsic to the operator at H=128, not to the
#   reduction shape per se.

"""rmsnorm_h128 — Triton kernel.

Architecture
- One program per BLOCK_ROWS rows (each row = 128 bf16 elements).
- Each program loads a (BLOCK_ROWS, 128) tile, computes per-row mean(x^2) in
  fp32, applies rsqrt and weight, writes back in bf16.
- BLOCK_ROWS picked by heuristic on B (B200 has ~148 SMs):
    tiny B (..256)    -> 1  rows, 1 warp     (one row per SM → max parallelism)
    small B (..1024)  -> 4  rows, 2 warps
    mid   B (..16k)   -> 16 rows, 4 warps
    huge  B (16k+)    -> 32 rows, 4 warps    (HBM-bound)
- num_stages=2 explicit (default 3 over-stages this no-K-loop kernel).
- ALIGNED constexpr fast-path skips the row mask when B % BLOCK_ROWS == 0.
  12/14 of the workload set is aligned for these picks → real +2.3% win on
  the small/mid range over the masked-always version.

Notes
- CUPTI times GPU kernels only (cold L2 between iters), so Python / launch
  overhead is excluded. The score is HBM-bandwidth limited at large B and
  SM-spread / per-thread-ILP limited at small/mid B.
- H=128 is constexpr so the row reduction fully unrolls and ld.global.b128
  vector loads are emitted.
- See `docs/prior/variants/v1/kernel.py` for the campaign retrospective —
  what worked, what didn't, NCU gotchas, recommended next probes.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    B,
    BLOCK_ROWS: tl.constexpr,
    H: tl.constexpr,
    INV_H: tl.constexpr,
    EPS: tl.constexpr,
    ALIGNED: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_ROWS
    rows = row_start + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, H)
    offs = rows[:, None] * H + cols[None, :]

    if ALIGNED:
        x = tl.load(X_ptr + offs, eviction_policy="evict_first")
    else:
        mask = rows < B
        x = tl.load(X_ptr + offs, mask=mask[:, None], other=0.0,
                    eviction_policy="evict_first")
    x_f = x.to(tl.float32)

    w = tl.load(W_ptr + cols, eviction_policy="evict_last").to(tl.float32)

    ss = tl.sum(x_f * x_f, axis=1) * INV_H
    rrms = tl.rsqrt(ss + EPS)

    y = x_f * rrms[:, None] * w[None, :]

    if ALIGNED:
        tl.store(Y_ptr + offs, y.to(Y_ptr.dtype.element_ty))
    else:
        tl.store(Y_ptr + offs, y.to(Y_ptr.dtype.element_ty),
                 mask=(rows < B)[:, None])


def _pick(B):
    # B200 has ~148 SMs. For B < ~148 we want 1 row per block so every row
    # gets its own SM in parallel; per-kernel time is then bounded by a single
    # row's load latency rather than serialised through few SMs.
    #
    # Huge-B note: iter-7 dropping to BLOCK_ROWS=8 (lower registers, higher
    # occupancy per NCU) regressed -12%. NCU's "Est. Speedup" overestimate
    # warning applies — per-thread ILP (more elts/thread → more outstanding
    # ld.b128) matters more than occupancy for this streaming reduce kernel.
    if B <= 256:
        return 1, 1
    if B <= 1024:
        return 4, 2
    if B <= 16384:
        return 16, 4
    return 32, 4


@torch.no_grad()
def run(hidden_states, weight):
    B, H = hidden_states.shape
    assert H == 128
    out = torch.empty_like(hidden_states)
    BLOCK_ROWS, num_warps = _pick(B)
    grid = ((B + BLOCK_ROWS - 1) // BLOCK_ROWS,)
    _rmsnorm_kernel[grid](
        hidden_states, weight, out,
        B,
        BLOCK_ROWS=BLOCK_ROWS,
        H=H,
        INV_H=1.0 / H,
        EPS=1e-6,
        ALIGNED=(B % BLOCK_ROWS == 0),
        num_warps=num_warps,
        num_stages=2,
    )
    return out
