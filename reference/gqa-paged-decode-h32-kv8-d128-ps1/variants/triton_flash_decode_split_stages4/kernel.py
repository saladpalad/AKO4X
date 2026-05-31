"""
iter14-triton-flash-decode-split-stages4 — round-1 anchor for
gqa_paged_decode_h32_kv8_d128_ps1.

============================================================================
1. Identity
============================================================================
Operator: gqa_paged_decode_h32_kv8_d128_ps1
DSL: Triton (3-kernel: fused-when-SPLIT=1 / split-K / reduce).
Score: **1.17x mean speedup vs FlashInfer expert** (48/48 PASS). B200, modal.
Per-batch: B=1 0.78x (10us), B=16 1.10x (16us), B=64 1.62x (65us).
Round-1 anchor (no prior variants existed before this run).

============================================================================
2. Delta from prior anchor
============================================================================
No prior anchor — this kernel was spawned from the pure-Python reference. The
reference looped `for b in range(batch_size)` doing eager bmm + softmax per
batch element; the scoring baseline is FlashInfer's
`BatchDecodeWithPagedKVCacheWrapper`. This variant is the first GPU-native
implementation in this campaign and beats the scoring baseline on average.

============================================================================
3. Lessons on this variant (load-bearing for round-2)
============================================================================
- **Host-side `.item()` sync was the single biggest cost on small batches.**
  iter-1 (with `int(kv_lens.max().item())` on the host) ran B=1 at 67us flat;
  iter-2 (batch_size-keyed SPLIT, no GPU sync) dropped it to 8us. 8x on B=1,
  4x on B=16. The lesson: never derive SPLIT/BLOCK from per-input tensor data
  on the host for a decode kernel; the d2h roundtrip serializes everything.
- **B=64 needs deep splits for variable-KV-length load balance.** SPLIT=1
  sees tail latency dominated by the longest batch's KV; SPLIT=8 (4096 CTAs
  total) cuts B=64 from 230us → 67us. Past SPLIT=8 the reduce-relaunch cost
  dominates and over-split regresses.
- **Conditional BLOCK_N matters.** BLOCK_N=64 uniformly hurts B=1 (8→10us)
  but helps B=64 (79→71us). The wider MMA tile costs extra mask work on
  mostly-empty chunks at small batch sizes.
- **num_stages tunes async-prefetch of K/V vs compute.** 2→3→4 each gave
  ~3% on B=64 (memory-bound, DRAM% climbs as more loads are in flight).
- **exp2/log2 with qk_scale = sm_scale * log2(e) baked on host** is the
  standard FlashAttention trick; saves one mul per softmax step AND makes
  the base-2 LSE output `m + log2(l)` direct (no post-divide by ln(2)).
- **page_size=1 + scattered KV gather is L1/L2-unfriendly.** NCU shows
  L1 hit rate 2.94%, L2 hit rate 1.75% — pure HBM streaming. The only
  reuse comes from GQA: one K/V tile fed to 4 Q heads in the same CTA.

Final tuning matrix (iter-12 = iter-14 = anchor):
  B=1:  SPLIT=32, BLOCK_N=32, num_stages=4 (split), num_stages=2 (reduce)
  B=16: SPLIT=4,  BLOCK_N=64, num_stages=4 (split), num_stages=2 (reduce)
  B=64: SPLIT=8,  BLOCK_N=64, num_stages=4 (split), num_stages=2 (reduce)
  All paths: BLOCK_H=16 (tl.dot min M), num_warps=4 split / 2 reduce.

Correctness invariants (don't break in round-2):
- LSE is **base-2** (`logsumexp / ln(2)`). The harness's correctness check
  catches this. We compute in base-2 internally (exp2/log2 + qk_scale*log2(e))
  so the LSE is just `m + log2(l)`.
- Empty batches (kv_indptr[b] == kv_indptr[b+1]) → output zeros, lse = -inf.
- BLOCK_H=16 has only 4 useful Q rows; mask stores to `h_mask = offs_h < GQA`.

NCU on B=64 split kernel (workload 32):
- DRAM throughput 51.8% (3.97 / ~8 TB/s); memory-bound.
- L1 hit 2.94%, L2 hit 1.75% — pure scatter-gather streaming.
- Achieved occupancy 27.9% / theoretical 31.25%; 80 reg/thread cap.
- SM compute throughput 26.85% — NOT compute-bound. Memory is the lever.

============================================================================
4. Dead-ends tried (don't re-try without changing surrounding context)
============================================================================
- **BLOCK_N=128 uniformly** (iter-4) and **only for B=64** (iter-9): B=64
  67→72us, B=1/B=16 regress. Bigger MMA tile costs SMEM and drops occupancy.
- **num_warps=8 split kernel** (iter-10, NCU-motivated): B=64 67→80us, B=16
  1.07→0.97x. Extra warps don't amortize the 16×64×128 MMA shape and worsen
  reg pressure.
- **SPLIT=8 for B=16** (iter-7): B=16 1.07→0.98x. Reduce-relaunch overhead
  exceeds the parallelism gain at B=16's smaller per-CTA work.
- **SPLIT=16 for B=64** (iter-13): B=64 65→76us. Over-split — per-CTA work
  too tiny to amortize the dispatch overhead.

============================================================================
5. Open directions (round-2 fork levers — structural, NOT knob-tuning)
============================================================================
1. **One-launch decode via cluster-cooperative or atomic-based cross-CTA
   reduce.** Currently the second launch (reduce kernel) costs ~10-12us per
   call — visible on every workload. A B200 cluster of up to 16 CTAs can
   exchange via DSMEM; SPLIT∈{4,8,16} all fit in a cluster. Would eliminate
   `partial_o/m/l` HBM round-trip and one launch tax. Biggest expected
   gain on B=1/B=16 where the 10us reduce launch is a large fraction of
   total latency.

2. **Pack 4 head groups per CTA so BLOCK_H=16 has all useful rows.** Today
   only 4 of 16 MMA rows do work. Each CTA processes 4 KV heads simultaneously
   (one batch, all 4 head groups → 16 useful Q rows). Same total memory
   traffic but 4× better MMA utilization; CTA count drops 4×. Needs SMEM
   accounting — at BLOCK_N=64 the K tile is 64KB; would likely need
   num_stages=1 or BLOCK_N=32 to fit. NCU showed compute throughput at 27%,
   so MMA underutilization is real headroom.

3. **Persistent kernel.** One CTA per SM walks a work-queue of (batch, h_kv,
   split) tuples. Amortizes per-launch latency across many work units. Pairs
   naturally with (1) — last worker on each SM does the local reduce in SMEM.
   Most beneficial for B=1 (where launch overhead dominates).

4. **Dual-path B=1 fused vs split selection.** Today B=1 always pays
   split+reduce overhead. A fused-at-SPLIT=1 path for short KV would save
   ~5us per call — but needs the KV-length predicate without a `.item()`
   sync. Possible via: (a) two-kernel "probe then dispatch" pattern, or
   (b) always-launch-fused plus always-launch-split where the loser
   early-exits (cheap but doubles SMEM/reg allocation requests on host).

5. **TMA-based bulk-gather of K/V into shared memory.** B200's TMA does
   2D async copies but NOT scatter-gather. Workaround: first TMA-fetch
   `kv_indices[chunk_start:end]` (contiguous in HBM) into SMEM, then use
   indices to compute per-CTA descriptors and re-issue TMA copies for K/V.
   Net win uncertain — the per-row K/V load is already coalesced; TMA's
   main advantage (async + multicast) is already available via Triton's
   cp.async pipeline driven by num_stages.

6. **Smaller dtype accumulation.** acc is f32 (16 rows × 128 = 8KB). Could
   try f16/bf16 acc on the V dot, but precision-sensitive for long sequences;
   defer until baseline acc-error budget is characterized.

"""

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_decode_fused_kernel(
    Q,                  # [B, H_Q, D]  bf16
    K_CACHE,            # [N_PAGES, page=1, H_KV, D]  bf16
    V_CACHE,            # [N_PAGES, page=1, H_KV, D]  bf16
    KV_INDPTR,          # [B + 1]      i32
    KV_INDICES,         # [num_kv_indices]  i32
    OUTPUT,             # [B, H_Q, D]  bf16
    LSE,                # [B, H_Q]     f32 (base-2)
    qk_scale,  # sm_scale * log2(e); base-2 softmax via tl.exp2
    stride_qb, stride_qh,
    stride_kp, stride_kh,
    stride_vp, stride_vh,
    stride_ob, stride_oh,
    stride_lb,
    GQA: tl.constexpr,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    h_kv = tl.program_id(1)

    kv_start = tl.load(KV_INDPTR + b)
    kv_end = tl.load(KV_INDPTR + b + 1)
    kv_len = kv_end - kv_start

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    h_mask = offs_h < GQA

    q = tl.load(
        Q + b * stride_qb
        + (h_kv * GQA + offs_h)[:, None] * stride_qh
        + offs_d[None, :],
        mask=h_mask[:, None],
        other=0.0,
    )

    m_i = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, D), dtype=tl.float32)

    for n_start in range(0, kv_len, BLOCK_N):
        n_idx = n_start + offs_n
        n_mask = n_idx < kv_len

        page_idx = tl.load(KV_INDICES + kv_start + n_idx, mask=n_mask, other=0)
        k = tl.load(
            K_CACHE + page_idx[:, None] * stride_kp + h_kv * stride_kh
            + offs_d[None, :],
            mask=n_mask[:, None], other=0.0,
        )
        s_qk = tl.dot(q, tl.trans(k)) * qk_scale
        s_qk = tl.where(n_mask[None, :], s_qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(s_qk, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s_qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v = tl.load(
            V_CACHE + page_idx[:, None] * stride_vp + h_kv * stride_vh
            + offs_d[None, :],
            mask=n_mask[:, None], other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    is_empty = m_i == -float("inf")
    safe_l = tl.maximum(l_i, 1.0)
    out = acc / safe_l[:, None]

    tl.store(
        OUTPUT + b * stride_ob
        + (h_kv * GQA + offs_h)[:, None] * stride_oh
        + offs_d[None, :],
        out.to(OUTPUT.dtype.element_ty),
        mask=h_mask[:, None],
    )

    # m_i and l_i are already in base 2 (we used exp2 internally), so
    # LSE_base2 = m + log2(l) directly — no division by ln(2).
    lse_val = tl.where(
        is_empty,
        -float("inf"),
        m_i + tl.log2(l_i),
    )
    tl.store(
        LSE + b * stride_lb + (h_kv * GQA + offs_h),
        lse_val,
        mask=h_mask,
    )


@triton.jit
def _flash_decode_split_kernel(
    Q, K_CACHE, V_CACHE, KV_INDPTR, KV_INDICES,
    PARTIAL_O,          # [B, H_KV, S, GQA, D]  f32
    PARTIAL_M,          # [B, H_KV, S, GQA]     f32
    PARTIAL_L,          # [B, H_KV, S, GQA]     f32
    qk_scale,  # sm_scale * log2(e)
    stride_qb, stride_qh,
    stride_kp, stride_kh,
    stride_vp, stride_vh,
    stride_pob, stride_poh, stride_pos, stride_pog,
    stride_pmb, stride_pmh, stride_pms,
    stride_plb, stride_plh, stride_pls,
    GQA: tl.constexpr,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT: tl.constexpr,
):
    b = tl.program_id(0)
    h_kv = tl.program_id(1)
    s = tl.program_id(2)

    kv_start = tl.load(KV_INDPTR + b)
    kv_end = tl.load(KV_INDPTR + b + 1)
    kv_len = kv_end - kv_start

    split_size = tl.cdiv(kv_len, SPLIT)
    chunk_start = s * split_size
    chunk_end = tl.minimum(chunk_start + split_size, kv_len)

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    h_mask = offs_h < GQA

    q = tl.load(
        Q + b * stride_qb
        + (h_kv * GQA + offs_h)[:, None] * stride_qh
        + offs_d[None, :],
        mask=h_mask[:, None], other=0.0,
    )

    m_i = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, D), dtype=tl.float32)

    for n_start in range(chunk_start, chunk_end, BLOCK_N):
        n_idx = n_start + offs_n
        n_mask = n_idx < chunk_end
        page_idx = tl.load(KV_INDICES + kv_start + n_idx, mask=n_mask, other=0)
        k = tl.load(
            K_CACHE + page_idx[:, None] * stride_kp + h_kv * stride_kh
            + offs_d[None, :],
            mask=n_mask[:, None], other=0.0,
        )
        s_qk = tl.dot(q, tl.trans(k)) * qk_scale
        s_qk = tl.where(n_mask[None, :], s_qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s_qk, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s_qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(
            V_CACHE + page_idx[:, None] * stride_vp + h_kv * stride_vh
            + offs_d[None, :],
            mask=n_mask[:, None], other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    tl.store(
        PARTIAL_O + b * stride_pob + h_kv * stride_poh + s * stride_pos
        + offs_h[:, None] * stride_pog + offs_d[None, :],
        acc,
        mask=h_mask[:, None],
    )
    tl.store(
        PARTIAL_M + b * stride_pmb + h_kv * stride_pmh + s * stride_pms + offs_h,
        m_i,
        mask=h_mask,
    )
    tl.store(
        PARTIAL_L + b * stride_plb + h_kv * stride_plh + s * stride_pls + offs_h,
        l_i,
        mask=h_mask,
    )


@triton.jit
def _flash_decode_reduce_kernel(
    PARTIAL_O, PARTIAL_M, PARTIAL_L,
    OUTPUT, LSE,
    stride_pob, stride_poh, stride_pos, stride_pog,
    stride_pmb, stride_pmh, stride_pms,
    stride_plb, stride_plh, stride_pls,
    stride_ob, stride_oh,
    stride_lb,
    GQA: tl.constexpr,
    D: tl.constexpr,
    SPLIT: tl.constexpr,
):
    b = tl.program_id(0)
    h_q = tl.program_id(1)
    h_kv = h_q // GQA
    g = h_q % GQA

    offs_s = tl.arange(0, SPLIT)
    offs_d = tl.arange(0, D)

    m = tl.load(
        PARTIAL_M + b * stride_pmb + h_kv * stride_pmh
        + offs_s * stride_pms + g
    )
    l = tl.load(
        PARTIAL_L + b * stride_plb + h_kv * stride_plh
        + offs_s * stride_pls + g
    )

    m_global = tl.max(m, axis=0)
    is_empty = m_global == -float("inf")
    m_safe = tl.where(is_empty, 0.0, m_global)
    alpha = tl.exp2(m - m_safe)
    l_global = tl.sum(l * alpha, axis=0)

    po = tl.load(
        PARTIAL_O + b * stride_pob + h_kv * stride_poh
        + offs_s[:, None] * stride_pos + g * stride_pog + offs_d[None, :]
    )
    out_unnorm = tl.sum(po * alpha[:, None], axis=0)
    safe_l = tl.maximum(l_global, 1.0)
    out = out_unnorm / safe_l

    tl.store(
        OUTPUT + b * stride_ob + h_q * stride_oh + offs_d,
        out.to(OUTPUT.dtype.element_ty),
    )
    # Partial m/l are base-2 (split kernel used exp2), so LSE is directly m + log2(l).
    lse_val = tl.where(
        is_empty,
        -float("inf"),
        m_global + tl.log2(l_global),
    )
    tl.store(LSE + b * stride_lb + h_q, lse_val)


def _select_splits(batch_size: int) -> int:
    """Host-side SPLIT policy keyed only on batch_size — no device sync.

    Goal: enough total CTAs (B * H_KV * SPLIT) to saturate B200's 148 SMs.
    Empty chunks are cheap (early-exit), so over-splitting short-KV workloads
    is OK.
    """
    if batch_size >= 64:
        return 8   # 64 * 8 * 8 = 4096 — sweet spot (iter-13 SPLIT=16 regressed)
    if batch_size >= 16:
        return 4   # 16 * 8 * 4 = 512 — SPLIT=8 measurably worse here
    if batch_size >= 4:
        return 8
    if batch_size >= 2:
        return 16
    return 32      # B=1: 1 * 8 * 32 = 256


@torch.no_grad()
def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, num_qo_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape

    assert num_qo_heads == 32
    assert num_kv_heads == 8
    assert head_dim == 128
    assert page_size == 1
    assert kv_indptr.shape[0] == batch_size + 1

    GQA = num_qo_heads // num_kv_heads  # 4
    BLOCK_H = 16
    # Conditional BLOCK_N: small batches are launch/overhead-dominated so
    # wider MMA tiles hurt (extra mask work on mostly-empty chunks); larger
    # batches are bandwidth/compute-bound and amortize bigger tiles cleanly.
    # Iter-9 tried BLOCK_N=128 for B=64 — measured regression 67→72us; the
    # bigger MMA tile cost more registers/SMEM and occupancy dropped. Stay at 64.
    BLOCK_N = 64 if batch_size >= 16 else 32
    SPLIT = _select_splits(batch_size)
    # Bake log2(e) into the QK-scale so the softmax can use base-2 exp/log
    # natively — matches the contract's base-2 LSE output and is cheaper on
    # B200's SFU than the natural-log path.
    qk_scale = sm_scale * 1.4426950408889634

    device = q.device
    output = torch.empty(
        (batch_size, num_qo_heads, head_dim),
        dtype=torch.bfloat16, device=device,
    )
    lse = torch.empty(
        (batch_size, num_qo_heads), dtype=torch.float32, device=device,
    )

    if SPLIT == 1:
        grid = (batch_size, num_kv_heads)
        _flash_decode_fused_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices,
            output, lse,
            qk_scale,
            q.stride(0), q.stride(1),
            k_cache.stride(0), k_cache.stride(2),
            v_cache.stride(0), v_cache.stride(2),
            output.stride(0), output.stride(1),
            lse.stride(0),
            GQA=GQA,
            D=head_dim,
            BLOCK_H=BLOCK_H,
            BLOCK_N=BLOCK_N,
            num_warps=4,
            num_stages=2,
        )
        return output, lse

    partial_o = torch.empty(
        (batch_size, num_kv_heads, SPLIT, GQA, head_dim),
        dtype=torch.float32, device=device,
    )
    partial_m = torch.empty(
        (batch_size, num_kv_heads, SPLIT, GQA),
        dtype=torch.float32, device=device,
    )
    partial_l = torch.empty(
        (batch_size, num_kv_heads, SPLIT, GQA),
        dtype=torch.float32, device=device,
    )

    grid_split = (batch_size, num_kv_heads, SPLIT)
    _flash_decode_split_kernel[grid_split](
        q, k_cache, v_cache, kv_indptr, kv_indices,
        partial_o, partial_m, partial_l,
        qk_scale,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(2),
        partial_o.stride(0), partial_o.stride(1),
        partial_o.stride(2), partial_o.stride(3),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
        GQA=GQA,
        D=head_dim,
        BLOCK_H=BLOCK_H,
        BLOCK_N=BLOCK_N,
        SPLIT=SPLIT,
        # NCU on B=64: DRAM 52%, occupancy 28% (80 reg/thread). num_warps=8
        # measured a regression (B=64 67→80us) — extra warps don't amortize
        # against the 16×64×128 MMA tile shape. num_stages=3 went 67→65us;
        # try 4 to push prefetch further.
        num_warps=4,
        num_stages=4,
    )

    grid_red = (batch_size, num_qo_heads)
    _flash_decode_reduce_kernel[grid_red](
        partial_o, partial_m, partial_l,
        output, lse,
        partial_o.stride(0), partial_o.stride(1),
        partial_o.stride(2), partial_o.stride(3),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
        output.stride(0), output.stride(1),
        lse.stride(0),
        GQA=GQA,
        D=head_dim,
        SPLIT=SPLIT,
        num_warps=2,
        num_stages=2,
    )

    return output, lse
