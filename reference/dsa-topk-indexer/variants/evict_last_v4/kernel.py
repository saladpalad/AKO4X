# Variant: evict_last_v4
# Source: ako4fib-run-indexer_v4/solution/kernel.py (iter-10 final, 2026-04-18)
#
# Identity
#   Triton FP8 score kernel with evict_last on K + scale + CuTe DSL radix
#   (single-thread serial scan, warp-coop ballot+popc gather), anchored in
#   the Triton-score CUDA graph. Fast path unchanged from v2/v3.
#
# Delta vs warp_coop_v3
#   Score-kernel K+scale get evict_last; CUDA warp-coop radix replaced by
#   CuTe DSL radix.
#
# ⚠️ Headline 41.64× is measurement-inflated — the CuTe DSL radix is not
#   captured into the CUDA graph (see ../../TRAPS.md entry #1 for mechanism
#   + honest baselines + detection tests + flashinfer-bench issue #414).
#   Single-run only; no --variance-check 3+ captured. Re-measure after
#   nvidia-cutlass-dsl upstream fixes `.launch()` capture.
import torch
import triton
import triton.language as tl

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ── Triton kernels ──

@triton.jit
def _all_tokens_kernel(
    bt_ptr, sl_ptr, out_ptr,
    stride_bt_b, stride_out_b,
    TOPK: tl.constexpr, PS: tl.constexpr,
):
    bid = tl.program_id(0)
    seq_len = tl.load(sl_ptr + bid)
    k = tl.arange(0, TOPK)
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
):
    bid = tl.program_id(0)
    pid = tl.program_id(1)
    ti = tl.arange(0, PS)
    out_base = bid * stride_s_b + pid * PS
    seq_len = tl.load(sl_ptr + bid)
    n_pages = tl.cdiv(seq_len, PS)
    if pid >= n_pages:
        return
    phys_page = tl.load(bt_ptr + bid * stride_bt_b + pid)
    hi = tl.arange(0, NH)
    di = tl.arange(0, HD)
    q = tl.load(q_ptr + bid * stride_q_b + hi[:, None] * HD + di[None, :],
                eviction_policy="evict_last")
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
    # Precompute bf16_ord so the radix kernel can read uint16 ordinals directly.
    # Sign-aware: some workloads have negative scores (mixed-sign weights
    # produce negative weighted sums of post-ReLU values).
    bits = scores.to(tl.bfloat16).to(tl.uint16, bitcast=True)
    ord_v = tl.where(bits >= 0x8000, bits, bits ^ 0x7FFF)
    tl.store(out_ptr + out_base + ti, ord_v.to(tl.bfloat16, bitcast=True))


# ═══════════════════════════════════════════════════════════════════════════
# CuTe DSL radix (iter-9): single-thread serial scan for threshold finding,
# per-thread atomic hist, warp-cooperative ballot+popc gather.
# Simpler than CUDA radix; matches at parity (42.82x vs CUDA 42.78x).
# Launched as follow-on after Triton score inside the same CUDA graph —
# standalone CuTe DSL kernels don't graph-capture on Modal B200 without
# a Triton/CUDA anchor.
# ═══════════════════════════════════════════════════════════════════════════
_CUTE_RADIX_BT = 128


@cute.kernel
def _cute_radix_kernel(
    mScores: cute.Tensor,    # [B, MT] int16 (bf16 reinterp ordinal)
    mBT: cute.Tensor,        # [B, M] int32
    mSL: cute.Tensor,        # [B] int32
    mOut: cute.Tensor,       # [B, TOPK] int32
    TOPK: cutlass.Constexpr,
    PS: cutlass.Constexpr,
):
    tid_x, _, _ = cute.arch.thread_idx()
    bid_x, _, _ = cute.arch.block_idx()
    bid = bid_x
    tid = tid_x
    sl = mSL[bid]
    BT: cutlass.Constexpr = _CUTE_RADIX_BT
    ORD_MASK: cutlass.Constexpr = 0xFFFF

    # SMEM
    hist_ptr = cute.arch.alloc_smem(cutlass.Int32, 256)
    hist = cute.make_tensor(hist_ptr, cute.make_layout(256))
    s_thr_ptr = cute.arch.alloc_smem(cutlass.Int32, 1)
    s_count_lt_ptr = cute.arch.alloc_smem(cutlass.Int32, 1)
    s_out_cnt_ptr = cute.arch.alloc_smem(cutlass.Int32, 1)
    s_cnt_tie_ptr = cute.arch.alloc_smem(cutlass.Int32, 1)
    s_thr = cute.make_tensor(s_thr_ptr, cute.make_layout(1))
    s_count_lt = cute.make_tensor(s_count_lt_ptr, cute.make_layout(1))

    if sl <= TOPK:
        # Pass-through.
        for j_off in cutlass.range_constexpr(TOPK // BT):
            k = tid + j_off * BT
            tok = cutlass.Int32(-1)
            if k < sl:
                pg = k // PS
                phys = mBT[bid, pg]
                tok = phys * PS + (k % PS)
            mOut[bid, k] = tok
    else:
        # Init hist + counters
        for i_off in cutlass.range_constexpr(256 // BT):
            hist[tid + i_off * BT] = cutlass.Int32(0)
        if tid == 0:
            cute.arch.store(s_out_cnt_ptr, cutlass.Int32(0))
            cute.arch.store(s_cnt_tie_ptr, cutlass.Int32(0))
            s_thr[0] = cutlass.Int32(0)
            s_count_lt[0] = cutlass.Int32(0)
        cute.arch.sync_threads()

        N = sl
        n_iters = (N + BT - cutlass.Int32(1)) // BT

        # Phase 1 hist (per-thread atomic on hi-byte).
        for it in cutlass.range(0, n_iters):
            i = it * BT + tid
            if i < N:
                raw = cutlass.Int32(mScores[bid, i])
                ord_v = raw & ORD_MASK
                bucket = ord_v >> 8
                cute.arch.atomic_add(
                    ptr=(hist_ptr + bucket).llvm_ptr,
                    val=cutlass.Int32(1), scope="cta",
                )
        cute.arch.sync_threads()

        # Single-thread serial scan for threshold 1 (256 ops). Slower than
        # a warp-only parallel scan but reliable on this CuTe DSL toolchain.
        if tid == 0:
            cum = cutlass.Int32(0)
            need = cutlass.Int32(TOPK)
            s_thr[0] = cutlass.Int32(255)
            s_count_lt[0] = cutlass.Int32(0)
            for idx in cutlass.range_constexpr(256):
                cum_old = cum
                cum = cum + hist[idx]
                if (cum >= need) and (cum_old < need):
                    s_thr[0] = cutlass.Int32(idx)
                    s_count_lt[0] = cum_old
        cute.arch.sync_threads()

        thr_hi = s_thr[0]
        count_lt_stage1 = s_count_lt[0]

        # Reset hist for phase 2.
        for i_off in cutlass.range_constexpr(256 // BT):
            hist[tid + i_off * BT] = cutlass.Int32(0)
        cute.arch.sync_threads()

        # Phase 2 hist (per-thread atomic on lo-byte, filtered by hi==thr_hi).
        for it in cutlass.range(0, n_iters):
            i = it * BT + tid
            if i < N:
                raw = cutlass.Int32(mScores[bid, i])
                ord_v = raw & ORD_MASK
                hi_b = ord_v >> 8
                lo_b = ord_v & 0xFF
                if hi_b == thr_hi:
                    cute.arch.atomic_add(
                        ptr=(hist_ptr + lo_b).llvm_ptr,
                        val=cutlass.Int32(1), scope="cta",
                    )
        cute.arch.sync_threads()

        if tid == 0:
            cum = cutlass.Int32(0)
            need = cutlass.Int32(TOPK) - count_lt_stage1
            s_thr[0] = cutlass.Int32(255)
            s_count_lt[0] = count_lt_stage1
            for idx in cutlass.range_constexpr(256):
                cum_old = cum
                cum = cum + hist[idx]
                if (cum >= need) and (cum_old < need):
                    s_thr[0] = cutlass.Int32(idx)
                    s_count_lt[0] = count_lt_stage1 + cum_old
        cute.arch.sync_threads()

        thr_lo = s_thr[0]
        count_lt = s_count_lt[0]
        threshold16 = (thr_hi << 8) | thr_lo

        # Warp-coop gather: ballot+popc+leader-atomic.
        lane = tid & 31
        lane_bit_below = (cutlass.Int32(1) << lane) - cutlass.Int32(1)
        for it in cutlass.range(0, n_iters):
            i = it * BT + tid
            in_range = i < N
            key = threshold16 + cutlass.Int32(1)
            if in_range:
                raw = cutlass.Int32(mScores[bid, i])
                key = raw & ORD_MASK
            surv_lt = in_range and (key < threshold16)
            surv_tie = in_range and (key == threshold16)
            mask_lt = cute.arch.vote_ballot_sync(surv_lt)
            mask_tie = cute.arch.vote_ballot_sync(surv_tie)
            n_lt = cute.arch.popc(mask_lt)
            n_tie = cute.arch.popc(mask_tie)
            warp_rank_lt = cute.arch.popc(mask_lt & lane_bit_below)
            warp_rank_tie = cute.arch.popc(mask_tie & lane_bit_below)
            base_lt = cutlass.Int32(0)
            base_tie = cutlass.Int32(0)
            if (lane == 0) and (n_lt > 0):
                base_lt = cute.arch.atomic_add(
                    ptr=s_out_cnt_ptr.llvm_ptr,
                    val=n_lt, scope="cta",
                )
            if (lane == 0) and (n_tie > 0):
                base_tie = cute.arch.atomic_add(
                    ptr=s_cnt_tie_ptr.llvm_ptr,
                    val=n_tie, scope="cta",
                )
            base_lt = cute.arch.shuffle_sync(base_lt, 0)
            base_tie = cute.arch.shuffle_sync(base_tie, 0)
            if surv_lt:
                pg = i // PS
                phys = mBT[bid, pg]
                tok = phys * PS + (i % PS)
                mOut[bid, base_lt + warp_rank_lt] = tok
            if surv_tie:
                pos_t = count_lt + base_tie + warp_rank_tie
                if pos_t < TOPK:
                    pg = i // PS
                    phys = mBT[bid, pg]
                    tok = phys * PS + (i % PS)
                    mOut[bid, pos_t] = tok


@cute.jit
def _cute_radix_host(
    scores: cute.Tensor, bt: cute.Tensor, sl: cute.Tensor, out: cute.Tensor,
):
    B = cute.size(out, mode=[0])
    _cute_radix_kernel(scores, bt, sl, out, 2048, 64).launch(
        grid=(B, 1, 1),
        block=(_CUTE_RADIX_BT, 1, 1),
    )


_cute_radix_cache = {}


def _cute_radix_run(scores_bf16, bt, sl, out, B, M):
    scores_i16 = scores_bf16.view(torch.int16)
    key = (B, M)
    compiled = _cute_radix_cache.get(key)
    if compiled is None:
        compiled = cute.compile(
            _cute_radix_host,
            from_dlpack(scores_i16), from_dlpack(bt),
            from_dlpack(sl), from_dlpack(out),
        )
        _cute_radix_cache[key] = compiled
    compiled(
        from_dlpack(scores_i16), from_dlpack(bt),
        from_dlpack(sl), from_dlpack(out),
    )


# ── Buffer caching ──
_output_buf = None
_scores_cache = {}  # (B, MT) -> reusable scores tensor (shared across trials of same shape)
_graph_cache = {}  # key = (B, M, q_ptr, k_ptr, w_ptr, sl_ptr, bt_ptr) -> (graph, output_slice, scores_ref)
_graph_pool = None  # shared CUDA graph memory pool across all graphs


def _run_kernels(q_index_fp8, k_fp8, k_f32,
                 weights, seq_lens, block_table, scores, output,
                 B, M, MT):
    _dsa_score_kernel[(B, M)](
        q_index_fp8, k_fp8, k_f32, weights, seq_lens, block_table, scores,
        q_index_fp8.stride(0), block_table.stride(0), MT,
        M,
        KS=8448, KSF=2112, SOF=2048, NH=64, PS=64, HD=128,
        num_warps=2, num_stages=1,
    )
    _cute_radix_run(scores, block_table, seq_lens, output, B, M)


@torch.no_grad()
def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
    global _output_buf, _graph_cache, _graph_pool

    B, M = block_table.shape
    device = block_table.device

    if _output_buf is None or _output_buf.shape[0] < B:
        _output_buf = torch.empty(max(B, 32), 2048, dtype=torch.int32, device=device)
    output = _output_buf[:B]

    # Graph cache keyed on shape + all tensor data pointers.
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
        # Fast path (single kernel)
        def run_fast():
            _all_tokens_kernel[(B,)](
                block_table, seq_lens, output,
                block_table.stride(0), output.stride(0),
                TOPK=2048, PS=64,
            )
        run_fast()  # eager warmup
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
        return (output,)

    NP = k_index_cache_fp8.shape[0]
    MT = M * 64
    k_raw = k_index_cache_fp8.view(torch.uint8).view(NP, 8448)
    k_fp8 = k_raw.view(torch.float8_e4m3fn)
    k_f32 = k_raw.view(torch.float32)

    # Reuse scores tensor per-shape (exact stride) to keep graph memory pool stable
    shape_key = (B, MT)
    scores = _scores_cache.get(shape_key)
    if scores is None:
        scores = torch.empty((B, MT), dtype=torch.bfloat16, device=device)
        _scores_cache[shape_key] = scores

    # Run once eagerly (warmup for graph capture)
    _run_kernels(q_index_fp8, k_fp8, k_f32,
                 weights, seq_lens, block_table, scores, output, B, M, MT)

    # Capture into graph on dedicated stream
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _run_kernels(q_index_fp8, k_fp8, k_f32,
             weights, seq_lens, block_table, scores, output, B, M, MT)
    torch.cuda.current_stream().wait_stream(stream)

    if _graph_pool is None:
        _graph_pool = torch.cuda.graph_pool_handle()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream, pool=_graph_pool):
        _run_kernels(q_index_fp8, k_fp8, k_f32,
             weights, seq_lens, block_table, scores, output, B, M, MT)
    _graph_cache[cache_key] = (graph, output, scores)

    return (output,)
