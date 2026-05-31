# Variant: v8_radix_bt256
# Source: ako4fib-run-indexer_v8/solution/kernel.py (iter-4 final, session 2026-04-22)
#
# Identity
#   45.55 ± 0.05x (3-run variance-check CV 0.10%, Modal B200, CUDA 13.2, 2026-04-22).
#   +0.73x over fast_split_v6 anchor (44.82x); A/B vs v6 deferred (both ran in
#   their own sessions — compare the per-workload tables in result.json).
#   Sole perf-relevant change vs fast_split_v6: radix kernel
#   `constexpr int BT = 64` → `BT = 256`. Ships an optional `DSA_NO_GRAPH=1`
#   env-var gate that skips CUDA graph capture for NCU-visible profiling.
#   Config requires [benchmark] use_isolated_runner = true on persistent-runner
#   environments (shared-denominator pointer-aliasing risk).
#
# Delta from fast_split_v6
#   One CUDA line moved: in `radix_select_topk()`, `constexpr int BT = 64` →
#   `BT = 256`. That single change widens each resident radix CTA from 2 warps
#   to 8, so the 30 SMs that host the (B=30-sized) grid get enough in-flight
#   warps to hide atomic / L1-TEX latency. NCU kernel duration on the worst
#   slow-path workload dropped 75us → 27us; end-to-end bench score moved
#   45.01x → 46.13x, variance-checked. No algorithmic or architectural change
#   beyond that one token. Header comments were rewritten to this convention;
#   the v5/v6 lean-cleanup log paragraph was dropped as stale. Added a
#   top-level `_DISABLE_GRAPH = bool(os.environ.get("DSA_NO_GRAPH"))` with
#   guarded graph-capture branches so profile runs can skip the capture
#   without removing it from the production path — see Lessons below.
#
# Lessons on this variant
#
#   +1.1x radix BT=64 → BT=256 (single-CTA per row)
#     How:           `constexpr int BT = 256` in `radix_select_topk()`
#     Why:           the radix grid is B=30 on 148 SMs, so only 30 SMs ever
#                    hold a block; each SM's 1 resident block needs enough
#                    warps to hide atomic / L1-TEX latency (NCU showed 92%
#                    of cycles had no eligible warp at BT=64). 8 warps/SM
#                    covers the stall budget; smaller regressed, larger
#                    (512, 1024) was within noise.
#     WHEN narrow:   radix-top-K kernels with single-CTA-per-row, B<=64 rows,
#                    row length N≈5824, 16-bit ordinalized keys.
#     WHEN broad:    any latency-bound, low-grid-count kernel where per-SM
#                    warp occupancy is the dominant stall lever AND register
#                    / shared-memory pressure permits a 4×/8× larger block
#                    without spilling. Doesn't help if the grid already fills
#                    the GPU — the benefit is per-active-SM, not per-grid.
#
#   DSA_NO_GRAPH env-var gate for NCU visibility
#     How:           `_DISABLE_GRAPH = bool(os.environ.get("DSA_NO_GRAPH"))`
#                    at module scope; guards the graph-capture branches
#                    in `run()`. Profile runs set the var → eager launches,
#                    all kernels visible to NCU. Labeled benches leave it
#                    unset → captured hot path.
#     Why:           NCU's `cuGraphLaunch` mode does not surface per-kernel
#                    attribution uniformly — only the first kernel of a
#                    replay reliably appears under per-kernel filtering
#                    (see `docs/profiler/ncu.md`). Env gate forces eager
#                    launches so every kernel gets its own NCU pass.
#                    v0 session 2026-04-25 verified the v8 radix is NOT
#                    silently skipped (DSA_NO_GRAPH=1 vs default mode
#                    produce identical per-workload timing on the slowest
#                    workloads). The gate is profile-only ergonomics, not
#                    a workaround for broken capture. The earlier reading
#                    of TRAPS.md entry #2 as "this kernel is broken under
#                    capture" was wrong; see TRAPS.md for the closed
#                    verification path.
#     WHEN narrow:   solutions wrapping multi-kernel pipelines in
#                    `torch.cuda.CUDAGraph()` on this harness.
#     WHEN broad:    any workflow needing per-kernel NCU attribution
#                    against a captured-graph production path.
#     Anti-pattern:  DO NOT leave `_DISABLE_GRAPH=True` hard-coded — graph
#                    capture is a wall-time win worth ~30% on end-to-end;
#                    the env gate is profile-only.
#
#   See ../../TRAPS.md "Gate sub-1× --ab-compare wins on --variance-check
#   3+" for the v8-session episode (parallel session reported +0.84×
#   num_stages=2 win; variance-check said −0.74×) — cross-variant
#   methodology, applies regardless of which variant is anchor.
#
# Dead-ends tried on this variant
#   Each is an expectation prior — retry only if your context flips the Why.
#
#   - Score kernel token-chunk (BLOCK_T=16 or 32, GPT-Pro-recommended).
#     Regs dropped 206→128 (theoretical occ 12.5%→25%); NCU duration grew
#     7.7us→8.9us and bench regressed 2-4%. Why: per-iter fixed overhead
#     (extra loads / reduce / store / loop control) > the occupancy gain at
#     PS=64 with num_warps=2. Broader WHEN: small inner tiles inside
#     register-limited kernels often don't amortize the chunk overhead.
#
#   - Radix 10+6 / 11+5 wider bit split. Regressed 45.48x / 44.30x vs 46.13x.
#     Why: 1024- / 2048-bucket hist init + 32- / 64-iter find_threshold scan
#     cost > phase-2 atomic savings. Finding-threshold scan grows linearly
#     with bucket count, and phase-2 wasn't the dominant cost to begin with.
#     Broader WHEN: wider radix only wins when phase-2 atomic contention
#     dominates; for 16-bit ordinalized keys with ~6 unique hi-bytes, 8+8
#     is the sweet spot.
#
#   - Multi-CTA radix with per-iter global-atomic gather counter. Bench 46x
#     → 4.5x. Why: with NC=5 CTAs per batch × 4 warps × 9 iters, the
#     `atomicAdd(&out_cnt[bid])` hotspot serialized ~360 gmem atomics per
#     batch. Broader WHEN: any multi-CTA top-K design needs a count-then-
#     reserve-then-write pattern, not per-iter atomic-append.
#
#   - Multi-CTA radix with CTA-reservation + smem buffer (GPT-Pro's
#     recommended pattern). Neutral at best. Why: with NC=5 and B=30 →
#     150 blocks on 148 SMs = ~1 block per SM = 4 warps per active SM,
#     which is *worse* than single-CTA BT=256's 8 warps per active SM.
#     Higher NC (up to 32) pushed per-CTA work too small and launch
#     overhead dominant. Broader WHEN: for small-B / small-N top-K on B200,
#     single-CTA-per-row is the right operating point — FlashInfer
#     independently chose `num_clusters=1` for `max_model_len < 8192`.
#
#   - Score tensor SMEM caching in radix (load N scores once, 3 passes on
#     smem). Within noise. Why: NCU shows 92% of stalls on L1TEX scoreboard
#     (atomic deps inside the CTA), so the kernel is stall-bound, not
#     L2-read-bound; eliminating 2/3 of gmem reads doesn't move the
#     dominant cost.
#
#   - Privatized per-warp histograms in radix. Within noise. Why: match_any
#     already reduces phase-1 atomics to ~6 per warp (≈ unique hi-byte
#     values), so cross-warp atomic contention wasn't the dominant stall.
#     The reduction step added its own cost.
#
#   - Score kernel num_stages=2 (one-char port from a parallel session).
#     -0.74x variance-check regression here (3-run mean 45.00x ± 0.16% CV
#     vs baseline 45.74 ± 0.03% CV), despite the parallel session measuring
#     +0.84x on their setup. Why: different surrounding BT (parallel
#     session used BT=64 radix) and possibly different scores dtype
#     change the end-to-end bottleneck mix. Not universally portable.
#
#   - `evict_first` on K in score kernel, `BLOCK_K=128` on fast-path
#     `_all_tokens_kernel`. Single-run regressions. Fold into a
#     parameter-sweep long-tail: retry only with new reasoning.
#
#   - Score kernel `num_warps=2→4` alone (v9 iter-1, 2026-04-24). -7x
#     full-bench (46.08 → 39.09x); slow-path groups each -7-15x.
#     NCU: regs 206→72, theoretical occ 12.5%→43.75%, achieved occ
#     barely moved (6.47→6.99%), NCU duration went UP 7.23→19.07us,
#     dynamic smem/block doubled 8→16KB. Why: Triton picks a less
#     efficient MMA primitive at (M=64, N=64, K=128) fp8 when
#     num_warps=4; the pipeliner also allocates a larger smem
#     double-buffer for the new MMA shape. Broader WHEN: register-
#     pressure reductions from raising num_warps don't translate if
#     the MMA tile-picker degrades at the new warp count. Verify the
#     emitted mma.sync m/n/k suffix before trusting occupancy-based
#     predictions.
#
#   - Score kernel multi-page merge `MERGE=2` with `if pid < n_pages`
#     inside `tl.static_range(0, MERGE)` (v9 iter-2, 2026-04-24). -3.4x
#     full-bench (42.68x); slow-path groups -3-10x each; fast-path
#     unchanged. Why: the scalar branch inside `tl.static_range`
#     defeats Triton's hoisting / CSE — per-iter overhead (even with Q
#     and w loads hoisted to the outer scope) dominates the Q/w load
#     amortization savings. Broader WHEN: if you must bounds-check
#     across static_range iters, do it via `tl.load(..., mask=...)` or
#     split the loop around the branch — never a scalar `if` inside
#     `tl.static_range`.
#
#   - Score kernel `num_stages=1→2` on the straight-line (no-loop)
#     form (v9 iter-3/4, 2026-04-24). Drift-cancelled AB Δ = +0.015x
#     (+0.03%, pure noise). Refines the 2026-04-22 result above (-0.74x
#     on a BT=64 radix base): on BT=256 the change is neutral, not
#     negative. Why: no `tl.range` loop in this kernel → nothing for
#     num_stages to pipeline. Both measurements are correct for their
#     surrounding state. Broader WHEN: `num_stages>1` needs an explicit
#     loop (HD-dim chunking, multi-page merge without the scalar branch
#     issue above) to have any effect; don't read the sign of a prior
#     session's result across different radix / smem / reg state.
#
#   - Score kernel K-split `HD=128 → 2×K=64` via `tl.range(0, 2,
#     num_stages=2)`, the explicit-loop variant of the prior num_stages
#     dead-end (v0-1 session, 2026-04-25). AB Δ = -0.00x (drift). Why:
#     even WITH a real `tl.range` loop for num_stages to pipeline, the
#     (M=64, N=64, K=128) fp8 MMA at this tile is bandwidth-bound on
#     K reads, not latency-bound on the MMA pipeline; pipelining hides
#     latency the kernel doesn't have. Refines prior entry: don't try
#     `num_stages>1` again on this score kernel regardless of loop
#     structure unless you can also reduce HBM K volume per call.
#
#   - Multi-CTA radix as a 2-kernel pipeline (v0 session, 2026-04-25):
#     `radix_threshold_kernel` (single-CTA per row, BT=256, drops gather,
#     writes threshold16 + count_lt to globals + zeros gather counters)
#     + `multi_cta_gather_kernel` (grid (B, NC=4..8), BT=256, two-pass:
#     warp-shuffle reduce in pass1 → atomic-reserve global LT/TIE
#     ranges → warp-coop ballot write in pass2). AB Δ = -0.04x (within
#     drift). NCU confirmed parallelism: gather Waves/SM 0.03 → 0.14,
#     L1/TEX hit 17% → 77%; total NCU duration 24.6us vs single-radix
#     27.7us (~0.3-0.6us actual savings). Net 0% at bench because the
#     extra kernel-boundary launch + global atomicAdd for output range
#     reservation eat the parallelism gain. Why: radix is L1-TEX-stall-
#     bound (per the existing "Score tensor SMEM caching" lesson) — more
#     SMs ≠ less wall time when each SM stalls on the same dependency
#     chain. This addresses BOTH prior multi-CTA failure modes (no
#     per-iter atomic; per-CTA = 8 warps not 4) and STILL loses,
#     confirming single-CTA-per-row is structural for this shape, not
#     a tuning artifact. Broader WHEN: don't expect SM-count parallelism
#     to win on stall-bound kernels regardless of how clean the cross-
#     CTA coordination is.
#
#   - CUDA score-kernel rewrite via `nvcuda::wmma` m16n16k16 bf16 MMA
#     (fp8→float→bf16 conversion during gmem→smem copy; PTX
#     `cvt.rn.bf16x2.e4m3x2` is sm_89/sm_90a only — ptxas rejects on
#     sm_100 — so the conversion runs per-element through fp32). v9
#     iter-5, 2026-04-24: 34.05x full-bench (-12x vs Triton anchor).
#     NCU: 14.78us (2× Triton 7.23us), 241 regs/thread, same 12.5%
#     theoretical / 6.41% achieved occupancy. Serialised single-c_frag
#     variant regressed further to 2.95x on wl 113 (A-reload overhead
#     dominated the register savings). Why: bf16 MMA is half the
#     throughput of fp8 MMA on Blackwell at (M=64,N=64,K=128); the
#     kernel is MMA-bandwidth-bound, not register-bound — lowering
#     regs didn't help, and the nvcuda::wmma API doesn't expose fp8
#     operands in CUDA 13.2. Broader WHEN: don't rewrite Triton fp8
#     MMA kernels into CUDA bf16 wmma hoping to offset via smem/reg
#     tuning — the 2× throughput gap is binding on this tile. The
#     only remaining CUDA replacement route is inline-PTX fp8
#     `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` +
#     `ldmatrix.x4.m8n8.b16` for A.
#
# Open directions
#   For B=30 / N≈5824 on a 148-SM B200, two independent GPT Pro
#   consultations and FlashInfer's own `num_clusters=1` threshold (when
#   `max_model_len < 8192`) converge on "single-CTA per row is the right
#   operating point; 46x is practical ceiling for this shape." Realistic
#   remaining headroom is +0.5–1.5x.
#
#   The bounded spike worth trying next (if any) is a CuTe / CUTLASS C++
#   rewrite of the score kernel using CUDA 13's `enable_smem_spilling` to
#   attack the 206 regs/thread cap (which limits score-kernel occupancy
#   to 12.5%), plus TCGen05 MMA + tmem-resident accumulators. Budget 2-3
#   hours; exit if regs don't fall into ~128-144 or score-kernel wall time
#   doesn't drop ≥0.4us. Cluster launch was tried in prior sessions and
#   hits "unspecified launch failure" under graph capture — not the path.
#
#   Narrowed 2026-04-24 (v9 session): `nvcuda::wmma` m16n16k16 bf16 is
#   ruled out (regressed 12x full-bench; bf16 MMA ≈ 0.5× fp8 MMA
#   throughput on Blackwell at this tile — see dead-end above). The
#   specific CUDA primitive that might still match or beat Triton is
#   inline-PTX fp8
#   `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` paired with
#   `ldmatrix.x4.m8n8.b16` for the A operand (and a transposed variant
#   for B). TCGen05 + tmem is strictly optional at that point; getting
#   the MMA intrinsic right is the first-order problem, reg/smem tuning
#   is second-order. Exit criteria unchanged.

import os

import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline


_cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Scores are pre-ordinalized by the Triton score kernel — the radix reads
// uint16 ordinals directly (smaller ordinal = larger real value).

// Warp-only inclusive scan over an `NumBuckets`-bin histogram. Only warp 0
// participates; other warps wait at the next __syncthreads(). Replaces CUB
// BlockScan<int, BT> (which forced BT=1024 to align with the scan size).
//
// `clear_after_read`: when true, warp 0 zeroes each hist slot after reading
// it so the histogram is ready for the next phase WITHOUT a separate clear
// loop + __syncthreads(). Saves one barrier between phase-1 find_threshold
// and phase-2 hist init.
template <int NumBuckets = 256, bool ClearAfterRead = false>
__device__ __forceinline__ void find_threshold_warp(
    int* hist, int base_count, int TOPK, int* out_thr, int* out_count_lt
) {
    if (threadIdx.x < 32) {
        int lane = threadIdx.x;
        int prefix = 0;
        int need = TOPK - base_count;
        constexpr int N_ITER = NumBuckets / 32;
        #pragma unroll
        for (int iter = 0; iter < N_ITER; iter++) {
            int idx = iter * 32 + lane;
            int val = hist[idx];
            if (ClearAfterRead) hist[idx] = 0;
            int incl = val;
            #pragma unroll
            for (int offset = 1; offset < 32; offset *= 2) {
                int tmp = __shfl_up_sync(0xFFFFFFFFu, incl, offset);
                if (lane >= offset) incl += tmp;
            }
            int my_incl = incl + prefix;
            int my_excl = my_incl - val;
            if (my_incl >= need && my_excl < need) {
                *out_thr = idx;
                *out_count_lt = base_count + my_excl;
            }
            prefix += __shfl_sync(0xFFFFFFFFu, incl, 31);
        }
    }
}

template<int BT>
__global__ __launch_bounds__(BT, 1)
void radix_select_topk_kernel(
    const __nv_bfloat16* __restrict__ scores,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    int* __restrict__ output,
    int MT, int stride_s, int stride_bt, int stride_out, int TOPK, int PS
) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int sl = seq_lens[bid];

    if (sl <= TOPK) {
        for (int k = tid; k < TOPK; k += BT) {
            int tok = -1;
            if (k < sl) {
                int pg = k / PS;
                tok = __ldg(&block_table[bid * stride_bt + pg]) * PS + (k & (PS - 1));
            }
            output[bid * stride_out + k] = tok;
        }
        return;
    }

    const unsigned short* __restrict__ raw =
        reinterpret_cast<const unsigned short*>(scores + bid * stride_s);
    const unsigned long long* __restrict__ raw4 =
        reinterpret_cast<const unsigned long long*>(raw);
    const int N = sl;
    const int N4 = N >> 2;
    const int N4_tail = N4 << 2;

    __shared__ int hist[256];
    __shared__ int s_thr_hi, s_count_lt, s_out_cnt, s_cnt_tie;

    // Vectorized init: 64 int4 stores instead of 256 int stores.
    if (tid < 64) {
        reinterpret_cast<int4*>(hist)[tid] = make_int4(0, 0, 0, 0);
    }
    if (tid == 0) { s_thr_hi = 0; s_count_lt = 0; }
    __syncthreads();

    // Phase 1: hi-byte hist with warp-coop match_any. Each warp issues ONE
    // atomic per unique bucket value (vs 1 per element). Score data clusters
    // in ~6 hi-byte buckets (post-ReLU positive, narrow exp range).
    {
        const unsigned FULL_MASK = 0xFFFFFFFFu;
        const int lane = tid & 31;
        const int n_iters4 = (N4 + BT - 1) / BT;
        for (int iter = 0; iter < n_iters4; iter++) {
            int i = iter * BT + tid;
            bool in_range = (i < N4);
            unsigned long long v = in_range ? raw4[i] : 0ULL;
            #pragma unroll
            for (int sub = 0; sub < 4; sub++) {
                unsigned short ord = (unsigned short)(v >> (16 * sub));
                int bucket = in_range ? (int)(ord >> 8) : -1;
                unsigned peers = __match_any_sync(FULL_MASK, bucket);
                int leader = __ffs(peers) - 1;
                int count = __popc(peers);
                if (bucket >= 0 && lane == leader) {
                    atomicAdd(&hist[bucket], count);
                }
            }
        }
        for (int i = N4_tail + tid; i < N; i += BT)
            atomicAdd(&hist[(int)(raw[i] >> 8)], 1);
    }
    __syncthreads();

    // Phase-1 scan with concurrent hist clear: warp 0 zeroes each slot after
    // reading it so hist is ready for phase 2 with NO separate clear+sync.
    find_threshold_warp<256, true>(hist, 0, TOPK, &s_thr_hi, &s_count_lt);
    __syncthreads();

    int thr_hi = s_thr_hi;
    int count_lt_stage1 = s_count_lt;

    // Phase 2: lo-byte hist filtered by hi==thr_hi. Per-thread atomic — the
    // ~95% sentinel rate makes match_any overhead exceed savings here.
    for (int i = tid; i < N4; i += BT) {
        unsigned long long v = raw4[i];
        unsigned short o0 = (unsigned short)(v), o1 = (unsigned short)(v >> 16);
        unsigned short o2 = (unsigned short)(v >> 32), o3 = (unsigned short)(v >> 48);
        if ((int)(o0 >> 8) == thr_hi) atomicAdd(&hist[(int)(o0 & 0xFFu)], 1);
        if ((int)(o1 >> 8) == thr_hi) atomicAdd(&hist[(int)(o1 & 0xFFu)], 1);
        if ((int)(o2 >> 8) == thr_hi) atomicAdd(&hist[(int)(o2 & 0xFFu)], 1);
        if ((int)(o3 >> 8) == thr_hi) atomicAdd(&hist[(int)(o3 & 0xFFu)], 1);
    }
    for (int i = N4_tail + tid; i < N; i += BT) {
        unsigned short o = raw[i];
        if ((int)(o >> 8) == thr_hi) atomicAdd(&hist[(int)(o & 0xFFu)], 1);
    }
    // tid==0 writes counters in parallel with other threads' atomicAdds
    // (different SMEM addresses, no conflict). Single sync afterward.
    if (tid == 0) {
        s_thr_hi = 0;
        s_count_lt = count_lt_stage1;
        s_out_cnt = 0;
        s_cnt_tie = 0;
    }
    __syncthreads();
    find_threshold_warp<256, false>(hist, count_lt_stage1, TOPK, &s_thr_hi, &s_count_lt);
    __syncthreads();

    int threshold16 = (thr_hi << 8) | s_thr_hi;
    int count_lt = s_count_lt;

    // Warp-coop gather: 32× fewer atomics via ballot+popc + leader-only
    // atomicAdd of n_lt/n_tie. Warp-uniform iteration count keeps
    // __ballot_sync seeing all 32 lanes even when N mod BT lands mid-warp.
    const unsigned FULL_MASK = 0xFFFFFFFFu;
    const int lane = tid & 31;
    const unsigned lane_bit_below = (1u << lane) - 1u;
    const int n_iters = (N + BT - 1) / BT;
    for (int iter = 0; iter < n_iters; iter++) {
        int i = iter * BT + tid;
        bool in_range = (i < N);
        int key = in_range ? (int)raw[i] : (threshold16 + 1);
        bool surv_lt  = in_range && (key <  threshold16);
        bool surv_tie = in_range && (key == threshold16);

        unsigned mask_lt  = __ballot_sync(FULL_MASK, surv_lt);
        unsigned mask_tie = __ballot_sync(FULL_MASK, surv_tie);
        int n_lt  = __popc(mask_lt);
        int n_tie = __popc(mask_tie);
        int warp_rank_lt  = __popc(mask_lt  & lane_bit_below);
        int warp_rank_tie = __popc(mask_tie & lane_bit_below);

        int base_lt  = 0;
        int base_tie = 0;
        if (lane == 0) {
            if (n_lt)  base_lt  = atomicAdd(&s_out_cnt, n_lt);
            if (n_tie) base_tie = atomicAdd(&s_cnt_tie, n_tie);
        }
        base_lt  = __shfl_sync(FULL_MASK, base_lt,  0);
        base_tie = __shfl_sync(FULL_MASK, base_tie, 0);

        if (surv_lt) {
            int pg = i / PS;
            int tok = __ldg(&block_table[bid * stride_bt + pg]) * PS + (i & (PS - 1));
            output[bid * stride_out + base_lt + warp_rank_lt] = tok;
        }
        if (surv_tie) {
            int pos = count_lt + base_tie + warp_rank_tie;
            if (pos < TOPK) {
                int pg = i / PS;
                int tok = __ldg(&block_table[bid * stride_bt + pg]) * PS + (i & (PS - 1));
                output[bid * stride_out + pos] = tok;
            }
        }
    }
}

torch::Tensor radix_select_topk(
    torch::Tensor scores, torch::Tensor block_table, torch::Tensor seq_lens,
    torch::Tensor output, int TOPK, int PS
) {
    int B = scores.size(0);
    int MT = scores.size(1);
    constexpr int BT = 256;
    radix_select_topk_kernel<BT><<<B, BT>>>(
        reinterpret_cast<const __nv_bfloat16*>(scores.data_ptr()),
        block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
        output.data_ptr<int>(),
        MT, scores.stride(0), block_table.stride(0), output.stride(0), TOPK, PS);
    return output;
}
"""

_cuda_module = load_inline(
    name="radix_topk_v8_bt256",
    cpp_sources=[
        "torch::Tensor radix_select_topk(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int);",
    ],
    cuda_sources=_cuda_src,
    functions=["radix_select_topk"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_100a"],
    verbose=False,
)


# ── Triton kernels ──

@triton.jit
def _all_tokens_kernel(
    bt_ptr, sl_ptr, out_ptr,
    stride_bt_b, stride_out_b,
    BLOCK_K: tl.constexpr, TOPK: tl.constexpr, PS: tl.constexpr,
):
    # Split TOPK across (TOPK/BLOCK_K) blocks per batch — for B=1..3 fast-path
    # workloads the 1-block-per-batch grid uses only 1..3 of 148 SMs.
    # Splitting raises SM utilization at the cost of more graph-launch overhead.
    bid = tl.program_id(0)
    blk = tl.program_id(1)
    seq_len = tl.load(sl_ptr + bid)
    k = blk * BLOCK_K + tl.arange(0, BLOCK_K)
    valid = k < seq_len
    pg = k // PS
    off = (k % PS).to(tl.int32)
    phys = tl.load(bt_ptr + bid * stride_bt_b + pg, mask=valid, other=0)
    tok = phys.to(tl.int32) * PS + off
    tok = tl.where(valid, tok, -1)
    tl.store(out_ptr + bid * stride_out_b + k, tok)


@triton.jit
def _dsa_score_kernel(
    q_ptr, k_fp8_ptr, k_f32_ptr, w_ptr, sl_ptr, bt_ptr, out_ptr,
    stride_q_b, stride_bt_b, stride_s_b,
    KS: tl.constexpr, KSF: tl.constexpr, SOF: tl.constexpr,
    NH: tl.constexpr, PS: tl.constexpr, HD: tl.constexpr,
    TOPK: tl.constexpr,
):
    bid = tl.program_id(0)
    pid = tl.program_id(1)
    ti = tl.arange(0, PS)
    out_base = bid * stride_s_b + pid * PS
    seq_len = tl.load(sl_ptr + bid)
    # Skip batches the radix fast branch handles directly.
    if seq_len <= TOPK:
        return
    n_pages = tl.cdiv(seq_len, PS)
    if pid >= n_pages:
        return
    phys_page = tl.load(bt_ptr + bid * stride_bt_b + pid)
    hi = tl.arange(0, NH)
    di = tl.arange(0, HD)
    q = tl.load(q_ptr + bid * stride_q_b + hi[:, None] * HD + di[None, :],
                eviction_policy="evict_last")
    # evict_last on K + scale: the 22MB K footprint fits B200's 120MB L2
    # across the 100-iter steady state; default evict_first churns L2.
    k = tl.load(k_fp8_ptr + phys_page * KS + ti[:, None] * HD + di[None, :],
                eviction_policy="evict_last")
    S = tl.dot(q, tl.trans(k))
    S = tl.maximum(S, 0.0)
    w = tl.load(w_ptr + bid * NH + hi, eviction_policy="evict_last")
    S *= w[:, None]
    scores = tl.sum(S, axis=0)
    scale = tl.load(k_f32_ptr + phys_page * KSF + SOF + ti,
                    eviction_policy="evict_last")
    scores *= scale
    # Pre-ordinalize bf16 so the radix kernel can read uint16 ordinals directly
    # (skipping per-element bf16_ord recomputation in three passes).
    # Sign-aware: post-weighted-sum scores can be negative when w is mixed-sign.
    bits = scores.to(tl.bfloat16).to(tl.uint16, bitcast=True)
    ord_v = tl.where(bits >= 0x8000, bits, bits ^ 0x7FFF)
    tl.store(out_ptr + out_base + ti, ord_v.to(tl.int16, bitcast=True))


# ── Buffer + per-shape graph cache ──
_output_buf = None
_scores_cache = {}   # (B, MT) → reusable int16 scores tensor (stable stride for graph)
_graph_cache = {}    # (B, M, all_data_ptrs) → (graph, output_view, scores_ref)
_graph_pool = None   # shared pool across all captured graphs

_DISABLE_GRAPH = bool(os.environ.get("DSA_NO_GRAPH"))


def _run_kernels(q_index_fp8, k_fp8, k_f32, weights, seq_lens, block_table, scores, output, B, M, MT):
    _dsa_score_kernel[(B, M)](
        q_index_fp8, k_fp8, k_f32, weights, seq_lens, block_table, scores,
        q_index_fp8.stride(0), block_table.stride(0), MT,
        KS=8448, KSF=2112, SOF=2048, NH=64, PS=64, HD=128, TOPK=2048,
        num_warps=2, num_stages=1,
    )
    _cuda_module.radix_select_topk(scores, block_table, seq_lens, output, 2048, 64)


@torch.no_grad()
def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
    global _output_buf, _graph_cache, _graph_pool

    B, M = block_table.shape
    device = block_table.device

    if _output_buf is None or _output_buf.shape[0] < B:
        _output_buf = torch.empty(max(B, 32), 2048, dtype=torch.int32, device=device)
    output = _output_buf[:B]

    cache_key = (
        B, M,
        q_index_fp8.data_ptr(), k_index_cache_fp8.data_ptr(),
        weights.data_ptr(), seq_lens.data_ptr(), block_table.data_ptr(),
    )
    if not _DISABLE_GRAPH:
        cached = _graph_cache.get(cache_key)
        if cached is not None:
            graph, cached_output = cached[0], cached[1]
            graph.replay()
            return (cached_output,)

    if M <= 32:
        # Fast path: seq_len ≤ TOPK, just emit all valid token indices.
        # BLOCK_K=256 → 8 blocks per batch (grid (B, 8)) so B=1..3 workloads
        # fill 8..24 SMs instead of 1..3.
        BLOCK_K = 256
        N_K_BLOCKS = 2048 // BLOCK_K
        def run_fast():
            _all_tokens_kernel[(B, N_K_BLOCKS)](
                block_table, seq_lens, output,
                block_table.stride(0), output.stride(0),
                BLOCK_K=BLOCK_K, TOPK=2048, PS=64,
            )
        run_fast()  # eager warmup
        if not _DISABLE_GRAPH:
            try:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    run_fast()
                torch.cuda.current_stream().wait_stream(stream)
                if _graph_pool is None:
                    _graph_pool = torch.cuda.graph_pool_handle()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream, pool=_graph_pool):
                    run_fast()
                _graph_cache[cache_key] = (graph, output, None)
            except Exception:
                pass
        return (output,)

    NP = k_index_cache_fp8.shape[0]
    MT = M * 64
    k_raw = k_index_cache_fp8.view(torch.uint8).view(NP, 8448)
    k_fp8 = k_raw.view(torch.float8_e4m3fn)
    k_f32 = k_raw.view(torch.float32)

    shape_key = (B, MT)
    scores = _scores_cache.get(shape_key)
    if scores is None:
        # int16 storage for pre-ordinalized scores (the CUDA radix reinterprets
        # as uint16 regardless). Skipping the bf16 output bitcast trims one step
        # from the score kernel's epilogue.
        scores = torch.empty((B, MT), dtype=torch.int16, device=device)
        _scores_cache[shape_key] = scores

    _run_kernels(q_index_fp8, k_fp8, k_f32, weights, seq_lens, block_table, scores, output, B, M, MT)

    if not _DISABLE_GRAPH:
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                _run_kernels(q_index_fp8, k_fp8, k_f32, weights, seq_lens, block_table, scores, output, B, M, MT)
            torch.cuda.current_stream().wait_stream(stream)

            if _graph_pool is None:
                _graph_pool = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream, pool=_graph_pool):
                _run_kernels(q_index_fp8, k_fp8, k_f32, weights, seq_lens, block_table, scores, output, B, M, MT)
            _graph_cache[cache_key] = (graph, output, scores)
        except Exception:
            pass

    return (output,)
