# Variant: fast_split_v6
# Source: ako4fib-run-indexer_v6/solution/kernel.py (iter-20 final, 2026-04-20)
# Superseded 2026-04-22 by ../v8_radix_bt256/ (single-line radix BT=64→256).
#
# Identity
#   44.82 ± 0.05× (3-run variance-check CV 0.10%, Modal B200, CUDA 13.2).
#   warp_coop_v3's CUDA radix + evict_last_v4's score-kernel K+scale
#   eviction + bf16_ord pre-ordinalization, plus the two v6 levers below.
#   No nvidia-cutlass-dsl dep; v6 iter-1 smoke observed CuTe DSL radix
#   regressing 50% vs CUDA warp-coop on CUDA 13.2 / Triton 3.6 (contradicts
#   v4's parity observation), so that path was dropped.
#
# Delta vs v4 reference (41.64× inflated)
#   Drift-free A/B vs iter-3 baseline same container: +3.24× cumulative for
#   the two v6 levers stacked on the v3+v4 skeleton.
#
# Lessons on this variant
#
#   Score-kernel early-return for sl ≤ TOPK batches (+0.94× A/B, iter-10)
#     How:           `if seq_len <= TOPK: return` at the top of the score
#                    kernel, before any loads.
#     Why:           the radix fast branch handles those batches without
#                    reading scores, so computing them is pure waste —
#                    score kernel's Q / K / w loads + MMA + w·S reduce +
#                    scale-mul all unused.
#     WHEN narrow:   two-kernel pipelines (score-then-select) where a fast
#                    branch in the selector already short-circuits the
#                    trivial case. Short-circuit upstream too.
#     WHEN broad:    any kernel whose output is known (by the downstream
#                    kernel's branch) to be unused for some workloads —
#                    early-return in the upstream kernel saves the whole
#                    dataflow, not just the tail.
#
#   Fast-path grid fanout `(B, 8)` with BLOCK_K=256 (+3.19× A/B, iter-14)
#     How:           `_all_tokens_kernel[(B, N_K_BLOCKS)](...)` with
#                    `BLOCK_K = 256`, `N_K_BLOCKS = 2048 // BLOCK_K = 8`,
#                    splitting the fixed TOPK=2048 across 8 blocks per
#                    batch.
#     Why:           B=1..3 fast-path workloads with a flat (B) grid use
#                    only 1..3 of 148 SMs; splitting along TOPK fills
#                    8..24 SMs and amortizes launch over more in-flight
#                    work. Sweep iter-14/15/15b/17 chose BLOCK_K=256; 512
#                    was equivalent, 128 regressed small-B (per-block
#                    work too small to cover launch overhead).
#     WHEN narrow:   fast-path / trivial-output kernels with tiny batch
#                    dimension on high-SM-count GPUs (≥100 SMs), fixed
#                    per-batch output size that can split along a second
#                    dim.
#     WHEN broad:    any kernel where the natural launch dim doesn't fill
#                    the GPU — pick a second dim to split along; size the
#                    per-block work so it still amortizes the extra
#                    per-block launch overhead (empirically ~128 elements
#                    is the floor at B200's launch-cost scale).

import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline


# ── CUDA radix top-K (warp-coop hist + warp-only scan + warp-coop gather) ──
_cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Scores in the bf16 buffer ARE pre-ordinalized by the score kernel
// (smaller ordinal = larger float). No runtime bf16_ord needed here.

// Warp-only inclusive scan over a 256-bin histogram. Only warp 0 participates;
// other warps wait at the next __syncthreads(). Replaces CUB BlockScan<int, BT>
// (which forced BT=1024 to align with the scan size).
//
// `clear_after_read`: when true, warp 0 zeroes each hist slot after reading
// it so the histogram is ready for the next phase WITHOUT a separate clear
// loop + __syncthreads(). Saves one barrier between phase-1 find_threshold
// and phase-2 hist init.
template <bool ClearAfterRead = false>
__device__ __forceinline__ void find_threshold_warp(
    int* hist, int base_count, int TOPK, int* out_thr, int* out_count_lt
) {
    if (threadIdx.x < 32) {
        int lane = threadIdx.x;
        int prefix = 0;
        int need = TOPK - base_count;
        #pragma unroll
        for (int iter = 0; iter < 8; iter++) {
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
    // in ~6 hi-byte buckets (post-ReLU positive, narrow exp range), so the
    // collision rate is high and this saves ~32× atomic traffic.
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
    // reading it, so hist is ready for phase 2 with NO separate clear loop +
    // syncthreads (saves one barrier vs the warp_coop_v3 layout).
    find_threshold_warp<true>(hist, 0, TOPK, &s_thr_hi, &s_count_lt);
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
    // (different SMEM addresses, no conflict). Single sync afterward waits
    // for both, saving one barrier vs the v3 layout's separate sync after the
    // hist loop.
    if (tid == 0) {
        s_thr_hi = 0;
        s_count_lt = count_lt_stage1;
        s_out_cnt = 0;
        s_cnt_tie = 0;
    }
    __syncthreads();
    find_threshold_warp(hist, count_lt_stage1, TOPK, &s_thr_hi, &s_count_lt);
    __syncthreads();

    int threshold16 = (thr_hi << 8) | s_thr_hi;
    int count_lt = s_count_lt;

    // Warp-coop gather: 32× fewer atomics via ballot+popc + leader-only
    // atomicAdd of n_lt/n_tie. Warp-uniform iteration count keeps
    // __ballot_sync(FULL_MASK, ...) seeing all 32 lanes even when N mod BT
    // lands mid-warp.
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
    constexpr int BT = 64;
    radix_select_topk_kernel<BT><<<B, BT>>>(
        reinterpret_cast<const __nv_bfloat16*>(scores.data_ptr()),
        block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
        output.data_ptr<int>(),
        MT, scores.stride(0), block_table.stride(0), output.stride(0), TOPK, PS);
    return output;
}
"""

_cuda_module = load_inline(
    name="radix_topk_v6_iter6_bt64",
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
    # workloads, the original 1-block-per-batch grid uses only 1..3 of 148 SMs.
    # Splitting raises SM utilization at the cost of more graph-launch overhead;
    # within the captured CUDA graph each kernel boundary is ~100 ns so the
    # tradeoff favors more parallelism for tiny B.
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
    M,
    KS: tl.constexpr, KSF: tl.constexpr, SOF: tl.constexpr,
    NH: tl.constexpr, PS: tl.constexpr, HD: tl.constexpr,
    TOPK: tl.constexpr,
):
    bid = tl.program_id(0)
    pid = tl.program_id(1)
    ti = tl.arange(0, PS)
    out_base = bid * stride_s_b + pid * PS
    seq_len = tl.load(sl_ptr + bid)
    # Skip score for batches the radix's `sl <= TOPK` fast branch handles
    # directly (it gathers all valid token indices without reading scores).
    # In mixed-seq_lens slow-path workloads (M>32 with some short batches),
    # this avoids the wasted MMA + write for those short batches.
    if seq_len <= TOPK:
        return
    n_pages = tl.cdiv(seq_len, PS)
    if pid >= n_pages:
        return
    phys_page = tl.load(bt_ptr + bid * stride_bt_b + pid)
    hi = tl.arange(0, NH)
    di = tl.arange(0, HD)
    # evict_last on Q (reused across M tiles per batch — keeps it in L2).
    q = tl.load(q_ptr + bid * stride_q_b + hi[:, None] * HD + di[None, :],
                eviction_policy="evict_last")
    # evict_last on K + scale (v4 iter-2 lever): the 22MB K footprint fits
    # B200's 120MB L2 across the 100-iter/trial steady state; default
    # evict_first churns K through L2 unnecessarily.
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
    # Pre-ordinalize bf16 score so the radix kernel can read uint16 ordinals
    # directly (skipping per-element bf16_ord recomputation in three passes).
    # Sign-aware: post-weighted-sum scores can be negative when w is mixed-sign.
    bits = scores.to(tl.bfloat16).to(tl.uint16, bitcast=True)
    ord_v = tl.where(bits >= 0x8000, bits, bits ^ 0x7FFF)
    tl.store(out_ptr + out_base + ti, ord_v.to(tl.int16, bitcast=True))


# ── Buffer + per-shape graph cache ──
_output_buf = None
_scores_cache = {}  # (B, MT) → reusable bf16 scores tensor (stable stride for graph)
_graph_cache = {}   # (B, M, all_data_ptrs) → (graph, output_view, scores_ref)
_graph_pool = None  # shared pool across all captured graphs


def _run_kernels(q_index_fp8, k_fp8, k_f32, weights, seq_lens, block_table, scores, output, B, M, MT):
    _dsa_score_kernel[(B, M)](
        q_index_fp8, k_fp8, k_f32, weights, seq_lens, block_table, scores,
        q_index_fp8.stride(0), block_table.stride(0), MT,
        M,
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
    cached = _graph_cache.get(cache_key)
    if cached is not None:
        graph, cached_output = cached[0], cached[1]
        graph.replay()
        return (cached_output,)

    if M <= 32:
        # Fast path: seq_len ≤ TOPK, just emit all valid token indices.
        # BLOCK_K=256 → 8 blocks per batch (2D grid (B, 8)) so B=1..3
        # workloads fill 8..24 SMs instead of 1..3. Sweep (iter-14/15/15b/17)
        # showed 256/512 equivalent; 128 regressed B≥4 (launch overhead vs
        # tiny per-block work). Adaptive (128 for B≤3, 256 else) was neutral
        # vs fixed 256 — not worth the branch.
        BLOCK_K = 256
        N_K_BLOCKS = 2048 // BLOCK_K
        def run_fast():
            _all_tokens_kernel[(B, N_K_BLOCKS)](
                block_table, seq_lens, output,
                block_table.stride(0), output.stride(0),
                BLOCK_K=BLOCK_K, TOPK=2048, PS=64,
            )
        run_fast()  # eager warmup
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
        # int16 storage for pre-ordinalized scores (CUDA radix reinterprets as
        # uint16 regardless). Skipping the bf16 output bitcast trims one step
        # from the score kernel's epilogue.
        scores = torch.empty((B, MT), dtype=torch.int16, device=device)
        _scores_cache[shape_key] = scores

    _run_kernels(q_index_fp8, k_fp8, k_f32, weights, seq_lens, block_table, scores, output, B, M, MT)

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
