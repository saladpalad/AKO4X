# Variant: presync_v1
# Source: ako4fib-run-moe1/trajectory/20260422_193944_iter-30_indirect-gemm1/kernel.py
# Operator: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
#
# Identity
#   0.842x ± 0.001x (3-run variance-check, CV 0.10%, Modal B200 sm_100, CUDA 13.2,
#   flashinfer-ci-cu132:20260401-2c675fb image, 2026-04-23T18:41, canonical
#   baseline). Baseline = flashinfer `trtllm_fp8_block_scale_moe`
#   (`../../baseline.json`, MD5 `a1d2be64…`). Prior header (2026-04-22) reported
#   0.841x single-run under the pre-canonical cache baseline — essentially
#   unchanged under canonical, which is coincidental (small-T gains/losses
#   cancel here). Retained as the A/B baseline for isolating the three launch-
#   overhead wins layered into the fused_indirect_v1 anchor.
#
# Delta from pre-iter-30 state (0.783x at iter-29)
#   iter-30 moved the A-gather out of a dedicated _gather_a_and_scale kernel and into
#   GEMM1's K-loop via a sorted_tokens[] indirection, removing one kernel launch and
#   the M_pad×H A_perm intermediate. Bucket-scatter dispatch (iter-18) and fused
#   GEMM2+scatter-add (iter-28/29) were already in place.
#
# What differs from fused_indirect_v1 (anchor, 1.022x)
#   (1) `counts.to('cpu')` sync runs for every T — no T≤256 skip-sync fast path.
#   (2) Dedicated _exclusive_cumsum_kernel fills an offs_ext buffer; consumers read
#       offs_ext rather than computing the prefix inline from counts[].
#   (3) Routing kernel does not atomic-add counts; a separate _dispatch_count_kernel
#       always runs.
#   (4) No module-level buffer cache — torch.empty / torch.zeros are called each run.
#   (5) sorted_tokens requires zero-init (dummy slots point to token 0).
#
# Lessons on this variant
#
#   +0.10x fused Triton routing kernel (iter-15, 0.368x → 0.467x)
#     How:           one Triton kernel does sigmoid → per-group top-2 sum → top-4
#                    groups via tl.sort → top-8 experts via tl.sort → weights
#                    normalization and topk_idx write (exclusive-cumsum-of-mask for
#                    stable positions).
#     Why:           the original routing pipeline was ~10 small torch ops
#                    contributing ~100 µs of cumulative CPU and launch overhead;
#                    small-T total latency is 120-250 µs so this was the single
#                    largest small-T lever.
#     WHEN narrow:   MoE no-aux (DeepSeek-V3) routing with bias-added gating needing
#                    per-group TopK then expert TopK; E_GLOBAL ≤ 256 so tl.sort is
#                    reasonable.
#     WHEN broad:    fuse a multi-step torch pre-processing chain into one kernel
#                    when each step is small and cumulative launch / dispatch
#                    overhead dominates wall time.
#
#   +0.21x bucket-scatter dispatch replaces torch.argsort (iter-18, 0.467x → 0.673x)
#     How:           _dispatch_count_kernel histograms topk_idx into counts[E_LOCAL];
#                    _dispatch_scatter_kernel uses `exclusive_cumsum(counts)[bucket]
#                    + atomic_add(counter[bucket], 1)` for the destination slot.
#     Why:           argsort over T*TOP_K ints is sort-bound; atomic-based histogram
#                    + scatter stays cheap because contention is bounded by E_LOCAL=32
#                    buckets and sm_100 L1 atomics absorb it well.
#     WHEN narrow:   top-K expert routing with E_LOCAL ≤ 64 and T up to ~16K,
#                    downstream needs grouped-by-expert layout only.
#     WHEN broad:    replace a sort with atomic-bucket-scatter when #buckets <<
#                    #elements AND the consumer does not need total order within a
#                    group.
#
#   fused GEMM2 + weighted scatter-add (iter-28/29)
#     How:           GEMM2 epilogue loads t = sorted_tokens[m_abs] and
#                    w = weight_vec[m_abs], then
#                      atomic_add(output[t, :], (acc * w).to(bf16))
#                    directly into the [T, H] output. No O[M_pad, H] intermediate
#                    buffer; no final `output.to(bfloat16)` cast.
#     Why:           removes a scatter-add kernel launch and an M_pad×H bf16
#                    allocation + write + read. The atomic_add cost is bounded
#                    because each token is written by at most TOP_K=8 experts.
#     WHEN narrow:   MoE GEMM2 where each token's multiple expert outputs need a
#                    weighted sum and the indirection (sorted_tokens[m_abs]) is
#                    lane-local.
#     WHEN broad:    fuse a scatter-add into the producer epilogue when the final
#                    output footprint is smaller than the staged intermediate AND
#                    the scatter address is available as a cheap indirection.
#
#   +0.06x inline A-gather into GEMM1 via sorted_tokens indirection (iter-30)
#     How:           in GEMM1, each lane loads tok = sorted_tokens[m_abs] and then
#                    reads hidden_states[tok, k_offs] directly inside the K-loop;
#                    A-scale uses HS_scale[kb, tok]. No _gather_a_and_scale launch;
#                    no A_perm [M_pad, H] intermediate.
#     Why:           removes one Triton kernel and the M_pad×H FP8 A_perm buffer
#                    (multi-MB at large T). The in-kernel indirect load preserves
#                    within-lane K-contiguity; the cross-lane locality was already
#                    lost in the original gather.
#     WHEN narrow:   grouped FP8 GEMM where A needs a gather-permutation (not
#                    expansion) and K is contiguous.
#     WHEN broad:    fuse an upstream gather into the consumer's GMEM load pattern
#                    whenever the gather is a pure permute AND the consumer already
#                    needs multi-block access to the source tensor.
#
# Dead-ends tried on this variant
#   The dead-end catalog in variants/fused_indirect_v1/kernel.py header applies to
#   presync_v1 too — those entries were verified in iters 2-29, all on code closer to
#   this snapshot than to the anchor. Read that catalog rather than duplicating here.
#
# Open directions
#   The three launch-overhead wins that move this 0.841x snapshot to the 1.022x
#   anchor are each independently A/B-able against this variant; see the anchor
#   header's Delta section. Beyond those, the per-T profile shape (weakest at T=901,
#   strongest at T=11948) is already present at this snapshot and unchanged by the
#   launch-overhead wins — the next structural lever (persistent GEMM for the
#   latency-bound mid-T regime) should be investigated on top of the anchor, not
#   here.

import torch
import triton
import triton.language as tl


FP8_MAX = 448.0


# ─────────────────────────── Triton kernels ───────────────────────────

@triton.jit
def _grouped_fp8_gemm1_indirect_kernel(
    HS_ptr,              # fp8 [T, K]
    HS_scale_ptr,        # fp32 [K//BK, T]
    sorted_tokens_ptr,   # int64 [M_pad]
    B_ptr,               # fp8 [G, N, K]
    B_scale_ptr,         # fp32 [G, N//BN, K//BK]
    C_ptr,               # bf16 [M_pad, N]
    offs_ptr,            # int32 [G+1]
    T,
    K: tl.constexpr, N: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    stride_bg: tl.constexpr,
    stride_bsg: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    group = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    g_start = tl.load(offs_ptr + group)
    g_end = tl.load(offs_ptr + group + 1)
    m_count = g_end - g_start

    m_tile_start = pid_m * BLOCK_M
    if m_tile_start >= m_count:
        return

    m_in_group = m_tile_start + tl.arange(0, BLOCK_M)
    m_mask = m_in_group < m_count
    m_abs = g_start + m_in_group

    # Per-lane token indices (indirection into hidden_states)
    tok = tl.load(sorted_tokens_ptr + m_abs, mask=m_mask, other=0)  # [BLOCK_M] int64

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offs_tile = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    B_group_ptr = B_ptr + group * stride_bg
    B_sc_base = B_scale_ptr + group * stride_bsg + pid_n * NUM_K_BLOCKS

    for kb in tl.range(0, NUM_K_BLOCKS, num_stages=NUM_STAGES):
        k_offs = kb * BLOCK_K + k_offs_tile

        a_fp8 = tl.load(HS_ptr + tok[:, None] * K + k_offs[None, :],
                        mask=m_mask[:, None], other=0.0)
        b_fp8 = tl.load(B_group_ptr + n_offs[:, None] * K + k_offs[None, :])

        a_sc = tl.load(HS_scale_ptr + kb * T + tok, mask=m_mask, other=0.0)
        b_sc = tl.load(B_sc_base + kb)

        partial = tl.dot(a_fp8, tl.trans(b_fp8))
        acc += partial * (a_sc[:, None] * b_sc)

    c_bf16 = acc.to(tl.bfloat16)
    tl.store(C_ptr + m_abs[:, None] * N + n_offs[None, :], c_bf16, mask=m_mask[:, None])


@triton.jit
def _grouped_fp8_fp8bs_gemm_kernel(
    A_ptr,           # fp8 [M_total, K]
    A_scale_ptr,     # fp32 [K//BK, M_total]  (per-row per-K-block)
    B_ptr,           # fp8 [G, N, K]
    B_scale_ptr,     # fp32 [G, N//BN, K//BK]
    C_ptr,           # bf16 [M_total, N]
    offs_ptr,        # int32 [G+1]
    M_total: tl.constexpr,
    K: tl.constexpr, N: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    stride_bg: tl.constexpr,
    stride_bsg: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    group = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    g_start = tl.load(offs_ptr + group)
    g_end = tl.load(offs_ptr + group + 1)
    m_count = g_end - g_start

    m_tile_start = pid_m * BLOCK_M
    if m_tile_start >= m_count:
        return

    m_in_group = m_tile_start + tl.arange(0, BLOCK_M)
    m_mask = m_in_group < m_count
    m_abs = g_start + m_in_group

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offs_tile = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    A_row_ptr = A_ptr + m_abs[:, None] * K
    B_group_ptr = B_ptr + group * stride_bg
    B_sc_base = B_scale_ptr + group * stride_bsg + pid_n * NUM_K_BLOCKS

    for kb in tl.range(0, NUM_K_BLOCKS, num_stages=NUM_STAGES):
        k_offs = kb * BLOCK_K + k_offs_tile

        a_fp8 = tl.load(A_row_ptr + k_offs[None, :], mask=m_mask[:, None], other=0.0)
        b_fp8 = tl.load(B_group_ptr + n_offs[:, None] * K + k_offs[None, :])

        a_sc = tl.load(A_scale_ptr + kb * M_total + m_abs, mask=m_mask, other=0.0)
        b_sc = tl.load(B_sc_base + kb)

        partial = tl.dot(a_fp8, tl.trans(b_fp8))
        acc += partial * (a_sc[:, None] * b_sc)

    c_bf16 = acc.to(tl.bfloat16)
    tl.store(C_ptr + m_abs[:, None] * N + n_offs[None, :], c_bf16, mask=m_mask[:, None])


@triton.jit
def _gather_a_and_scale_kernel(
    HS_ptr,              # fp8 [T, H]
    HS_scale_ptr,        # fp32 [nHB, T]
    sorted_tokens_ptr,   # int64 [M_pad]
    A_perm_ptr,          # fp8 [M_pad, H]
    A_scale_perm_ptr,    # fp32 [nHB, M_pad]
    T, M_pad,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,  # 128
    NUM_H_BLOCKS: tl.constexpr,
):
    m = tl.program_id(0)
    hb = tl.program_id(1)

    tok = tl.load(sorted_tokens_ptr + m)

    # Gather A tile (BLOCK_H=128 fp8 bytes)
    h_offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    hs_fp8 = tl.load(HS_ptr + tok * H + h_offs)
    tl.store(A_perm_ptr + m * H + h_offs, hs_fp8)

    # Gather scale (one fp32 per tile)
    sc = tl.load(HS_scale_ptr + hb * T + tok)
    tl.store(A_scale_perm_ptr + hb * M_pad + m, sc)


@triton.jit
def _fused_routing_kernel(
    logits_ptr,        # [T, E_GLOBAL] fp32
    bias_ptr,          # [E_GLOBAL] bf16
    topk_idx_out_ptr,  # [T, TOP_K] int32
    weights_out_ptr,   # [T, E_GLOBAL] fp32
    routed_scaling,    # fp32 scalar
    E_GLOBAL: tl.constexpr,    # 256
    N_GROUP: tl.constexpr,     # 8
    GROUP_SIZE: tl.constexpr,  # 32
    TOPK_GROUP: tl.constexpr,  # 4
    TOP_K: tl.constexpr,       # 8
):
    t = tl.program_id(0)

    e_offs = tl.arange(0, E_GLOBAL)
    logit = tl.load(logits_ptr + t * E_GLOBAL + e_offs)
    b = tl.load(bias_ptr + e_offs).to(tl.float32)

    s = tl.sigmoid(logit)
    s_wb = s + b  # [E_GLOBAL]

    # Reshape to [N_GROUP, GROUP_SIZE] and compute top-2 sum per group
    s_wb_2d = tl.reshape(s_wb, (N_GROUP, GROUP_SIZE))  # [8, 32]
    row_max = tl.max(s_wb_2d, axis=1, keep_dims=True)                      # [8, 1]
    is_max = s_wb_2d >= row_max
    s_wb_masked = tl.where(is_max, tl.full(s_wb_2d.shape, -3.4e38, tl.float32), s_wb_2d)
    row_max2 = tl.max(s_wb_masked, axis=1, keep_dims=True)                 # [8, 1]
    gs = tl.reshape(row_max + row_max2, (N_GROUP,))                        # [8]

    # Find threshold for top-4 groups (4th largest)
    gs_sorted = tl.sort(gs, descending=True)                               # [8]
    # Take element at index TOPK_GROUP-1 using masked sum
    idx_group = tl.arange(0, N_GROUP)
    thresh_group = tl.sum(tl.where(idx_group == (TOPK_GROUP - 1), gs_sorted, 0.0))

    group_mask = gs >= thresh_group                                        # [8] bool

    # Broadcast group_mask to expert-level: [8, 32] then flatten to [256]
    group_mask_2d = tl.broadcast_to(group_mask[:, None], (N_GROUP, GROUP_SIZE))
    emask = tl.reshape(group_mask_2d, (E_GLOBAL,))                         # [256] bool

    scores_pruned = tl.where(emask, s_wb, -3.4e38)                         # [256]

    # Top-K experts: find threshold (8th largest)
    sp_sorted = tl.sort(scores_pruned, descending=True)                    # [256]
    idx_e = tl.arange(0, E_GLOBAL)
    thresh_topk = tl.sum(tl.where(idx_e == (TOP_K - 1), sp_sorted, 0.0))

    topk_mask = scores_pruned >= thresh_topk                               # [256] bool

    # Weights: s * topk_mask, then normalize by sum, multiply by scaling.
    topk_mask_f = topk_mask.to(tl.float32)
    w_raw = s * topk_mask_f                                                # [256]
    w_sum = tl.sum(w_raw) + 1e-20                                          # scalar
    weights_out = (w_raw / w_sum) * routed_scaling                         # [256]
    tl.store(weights_out_ptr + t * E_GLOBAL + e_offs, weights_out)

    # topk_idx: write the indices of selected experts (any order; scatter-add doesn't care).
    # Compute exclusive cumsum of topk_mask → write position; store at that position.
    cumsum_mask = tl.cumsum(topk_mask.to(tl.int32), axis=0)                # inclusive
    write_pos = cumsum_mask - 1                                            # exclusive → index of this match
    # Only store where mask==1 and write_pos < TOP_K (safety).
    store_mask = topk_mask & (write_pos < TOP_K)
    # Compute address: topk_idx_out_ptr[t, write_pos] = idx_e
    tl.store(topk_idx_out_ptr + t * TOP_K + write_pos, idx_e, mask=store_mask)


@triton.jit
def _dispatch_count_kernel(
    topk_idx_ptr,    # [T, TOP_K] int32
    counts_out,      # [E_LOCAL] int32 (zero-init)
    local_start,
    T,
    E_LOCAL: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    t_offs = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offs < T

    for k in tl.static_range(TOP_K):
        idx = tl.load(topk_idx_ptr + t_offs * TOP_K + k, mask=t_mask, other=0)
        shifted = idx - local_start
        valid = t_mask & (shifted >= 0) & (shifted < E_LOCAL)
        bucket = tl.where(valid, shifted, 0)
        tl.atomic_add(counts_out + bucket, 1, mask=valid)


@triton.jit
def _exclusive_cumsum_kernel(
    counts_ptr,    # [E_LOCAL] int32
    offs_ext_ptr,  # [E_LOCAL + 1] int32 (output)
    E_LOCAL: tl.constexpr,
):
    # Single-block; E_LOCAL is small (32).
    offs = tl.arange(0, E_LOCAL)
    c = tl.load(counts_ptr + offs)
    cs = tl.cumsum(c, axis=0)     # inclusive
    # Exclusive: out[0]=0, out[i]=cs[i-1] for i>=1.
    # Equivalently store inclusive cs at offset+1 and 0 at offset 0.
    tl.store(offs_ext_ptr + 0, 0)
    tl.store(offs_ext_ptr + 1 + offs, cs)


@triton.jit
def _dispatch_scatter_kernel(
    topk_idx_ptr,       # [T, TOP_K] int32
    offs_ext_ptr,       # [E_LOCAL+1] int32 (cumulative starts)
    counter_ptr,        # [E_LOCAL] int32 (zero-init, per-bucket counter)
    sorted_tokens_out,  # [M_pad] int64
    sorted_experts_out, # [M_pad] int32
    T, local_start,
    E_LOCAL: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    t_offs = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offs < T

    for k in tl.static_range(TOP_K):
        idx = tl.load(topk_idx_ptr + t_offs * TOP_K + k, mask=t_mask, other=0)
        shifted = idx - local_start
        valid = t_mask & (shifted >= 0) & (shifted < E_LOCAL)
        bucket = tl.where(valid, shifted, 0)

        pos = tl.atomic_add(counter_ptr + bucket, 1, mask=valid)
        base = tl.load(offs_ext_ptr + bucket, mask=valid, other=0)
        slot = base + pos

        tl.store(sorted_tokens_out + slot, t_offs.to(tl.int64), mask=valid)
        tl.store(sorted_experts_out + slot, shifted, mask=valid)


@triton.jit
def _grouped_fp8_gemm2_fused_scatter_kernel(
    A_ptr,              # fp8 [M_pad, K]
    A_scale_ptr,        # fp32 [K//BK, M_pad]
    B_ptr,              # fp8 [G, N, K]
    B_scale_ptr,        # fp32 [G, N//BN, K//BK]
    offs_ptr,           # int32 [G+1]
    weight_vec_ptr,     # fp32 [M_pad]  (first N_total valid)
    sorted_tokens_ptr,  # int64 [M_pad]
    output_ptr,         # bf16 [T, N]
    M_total: tl.constexpr,
    K: tl.constexpr, N: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    stride_bg: tl.constexpr,
    stride_bsg: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    group = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    g_start = tl.load(offs_ptr + group)
    g_end = tl.load(offs_ptr + group + 1)
    m_count = g_end - g_start

    m_tile_start = pid_m * BLOCK_M
    if m_tile_start >= m_count:
        return

    m_in_group = m_tile_start + tl.arange(0, BLOCK_M)
    m_mask = m_in_group < m_count
    m_abs = g_start + m_in_group

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offs_tile = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    A_row_ptr = A_ptr + m_abs[:, None] * K
    B_group_ptr = B_ptr + group * stride_bg
    B_sc_base = B_scale_ptr + group * stride_bsg + pid_n * NUM_K_BLOCKS

    for kb in tl.range(0, NUM_K_BLOCKS, num_stages=NUM_STAGES):
        k_offs = kb * BLOCK_K + k_offs_tile

        a_fp8 = tl.load(A_row_ptr + k_offs[None, :], mask=m_mask[:, None], other=0.0)
        b_fp8 = tl.load(B_group_ptr + n_offs[:, None] * K + k_offs[None, :])

        a_sc = tl.load(A_scale_ptr + kb * M_total + m_abs, mask=m_mask, other=0.0)
        b_sc = tl.load(B_sc_base + kb)

        partial = tl.dot(a_fp8, tl.trans(b_fp8))
        acc += partial * (a_sc[:, None] * b_sc)

    # Epilogue: scale by routing weight; atomic-add to output[sorted_tokens[m], :].
    t = tl.load(sorted_tokens_ptr + m_abs, mask=m_mask, other=0)   # [BM] int64
    w = tl.load(weight_vec_ptr + m_abs, mask=m_mask, other=0.0)    # [BM] fp32

    scaled = (acc * w[:, None]).to(tl.bfloat16)

    # Scatter-atomic-add to output[t, n_offs]
    out_ptrs = output_ptr + t[:, None] * N + n_offs[None, :]
    tl.atomic_add(out_ptrs, scaled, mask=m_mask[:, None])


@triton.jit
def _scaled_scatter_add_kernel(
    O_ptr,             # bf16 [M_pad, H]
    weight_vec_ptr,    # fp32 [M_pad]
    sorted_tokens_ptr, # int64 [M_pad]
    output_ptr,        # bf16 [T, H]
    N_total,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    m = tl.program_id(0)
    hb = tl.program_id(1)

    if m >= N_total:
        return

    t = tl.load(sorted_tokens_ptr + m)
    w = tl.load(weight_vec_ptr + m)

    h_offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    o = tl.load(O_ptr + m * H + h_offs).to(tl.float32)
    scaled = (o * w).to(tl.bfloat16)
    tl.atomic_add(output_ptr + t * H + h_offs, scaled)


@triton.jit
def _swiglu_quant_kernel(
    G1_ptr,       # [N_total, 2I] bf16
    C_ptr,        # [N_total, I] fp8
    Cscale_ptr,   # [nIB, N_total] fp32
    N_total,
    I: tl.constexpr,
    BLOCK_I: tl.constexpr,      # = 128
):
    m = tl.program_id(0)
    ib = tl.program_id(1)

    i_offs = ib * BLOCK_I + tl.arange(0, BLOCK_I)

    x1 = tl.load(G1_ptr + m * (2 * I) + i_offs).to(tl.float32)
    x2 = tl.load(G1_ptr + m * (2 * I) + I + i_offs).to(tl.float32)

    silu_x2 = x2 * tl.sigmoid(x2)
    val = silu_x2 * x1

    amax = tl.max(tl.abs(val), axis=0)
    scale = tl.where(amax > 1e-10, amax / 448.0, 1.0)

    val_q = (val / scale).to(tl.float8e4nv)

    tl.store(C_ptr + m * I + i_offs, val_q)
    tl.store(Cscale_ptr + ib * N_total + m, scale)


# ─────────────────────────── Python wrapper ───────────────────────────

@torch.no_grad()
def run(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
):
    H = 7168
    I = 2048
    BLOCK = 128
    E_local = gemm1_weights.shape[0]
    E_global = 256
    T = routing_logits.shape[0]
    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4
    group_size = E_global // N_GROUP
    nHB = H // BLOCK
    nIB = I // BLOCK
    n1B = (2 * I) // BLOCK

    device = hidden_states.device

    # ─────── Routing (fused Triton kernel) ───────
    topk_idx = torch.empty((T, TOP_K), dtype=torch.int32, device=device)
    weights = torch.empty((T, E_global), dtype=torch.float32, device=device)
    _fused_routing_kernel[(T,)](
        routing_logits, routing_bias,
        topk_idx, weights,
        float(routed_scaling_factor),
        E_global, N_GROUP, group_size, TOPK_GROUP, TOP_K,
        num_warps=2, num_stages=1,
    )

    local_start = int(local_expert_offset)

    # ─────── Dispatch (bucket scatter via Triton) ───────
    counts = torch.zeros(E_local, dtype=torch.int32, device=device)
    DISP_BT = 32 if T < 256 else 64
    _dispatch_count_kernel[(triton.cdiv(T, DISP_BT),)](
        topk_idx, counts, local_start, T,
        E_local, TOP_K, DISP_BT,
        num_warps=1, num_stages=1,
    )

    offs_ext = torch.empty(E_local + 1, dtype=torch.int32, device=device)
    _exclusive_cumsum_kernel[(1,)](
        counts, offs_ext, E_local,
        num_warps=1, num_stages=1,
    )

    counts_cpu = counts.to('cpu', non_blocking=False)
    N_total = int(counts_cpu.sum().item())
    max_count = int(counts_cpu.max().item())
    if N_total == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    # Pad buffer size to stabilize M_total constexpr across calls.
    M_pad = max(((N_total + 127) // 128) * 128, 128)

    sorted_tokens = torch.empty(M_pad, dtype=torch.int64, device=device)
    sorted_experts_buf = torch.empty(M_pad, dtype=torch.int32, device=device)
    counter_buf = torch.zeros(E_local, dtype=torch.int32, device=device)
    _dispatch_scatter_kernel[(triton.cdiv(T, DISP_BT),)](
        topk_idx, offs_ext, counter_buf,
        sorted_tokens, sorted_experts_buf,
        T, local_start,
        E_local, TOP_K, DISP_BT,
        num_warps=1, num_stages=1,
    )
    sorted_tokens_valid = sorted_tokens[:N_total]
    sorted_experts = sorted_experts_buf[:N_total]

    # ─────── Grouped GEMM 1 (indirect — reads hidden_states via sorted_tokens) ───────
    G1 = torch.empty((M_pad, 2 * I), dtype=torch.bfloat16, device=device)
    if max_count >= 256:
        BLOCK_M_1 = 128
        NUM_STAGES_1 = 6
        NUM_WARPS_1 = 8
    else:
        BLOCK_M_1 = 64
        NUM_STAGES_1 = 4
        NUM_WARPS_1 = 4
    BLOCK_N_1 = 128
    BLOCK_K_1 = 128
    max_m_tiles = triton.cdiv(max_count, BLOCK_M_1)
    grid_1 = (E_local, max_m_tiles, triton.cdiv(2 * I, BLOCK_N_1))
    _grouped_fp8_gemm1_indirect_kernel[grid_1](
        hidden_states,
        hidden_states_scale,
        sorted_tokens,          # full padded buffer; GEMM reads via offs_ext
        gemm1_weights,
        gemm1_weights_scale,
        G1,
        offs_ext,
        T,
        H, 2 * I,
        H // BLOCK_K_1,
        2 * I * H,
        n1B * nHB,
        BLOCK_M_1, BLOCK_N_1, BLOCK_K_1, NUM_STAGES_1,
        num_warps=NUM_WARPS_1,
    )

    # ─────── Fused SwiGLU + quantize to FP8 ───────
    C_fp8 = torch.empty((M_pad, I), dtype=torch.float8_e4m3fn, device=device)
    C_scale = torch.empty((nIB, M_pad), dtype=torch.float32, device=device)
    _swiglu_quant_kernel[(N_total, nIB)](
        G1, C_fp8, C_scale,
        M_pad, I, BLOCK,
        num_warps=1, num_stages=1,
    )

    # ─────── Compute routing weight lookup for fused epilogue ───────
    sorted_global_experts = sorted_experts + local_start              # [N_total]
    weight_vec = weights[sorted_tokens_valid, sorted_global_experts]  # [N_total] fp32

    # ─────── Grouped GEMM 2 + scatter-add (fused) ───────
    output = torch.zeros((T, H), dtype=torch.bfloat16, device=device)
    if max_count >= 256:
        BLOCK_M_2 = 128
        NUM_STAGES_2 = 6
        NUM_WARPS_2 = 8
    else:
        BLOCK_M_2 = 64
        NUM_STAGES_2 = 4
        NUM_WARPS_2 = 4
    BLOCK_N_2 = 128
    BLOCK_K_2 = 128
    max_m_tiles_2 = triton.cdiv(max_count, BLOCK_M_2)
    grid_2 = (E_local, max_m_tiles_2, triton.cdiv(H, BLOCK_N_2))
    _grouped_fp8_gemm2_fused_scatter_kernel[grid_2](
        C_fp8, C_scale,
        gemm2_weights, gemm2_weights_scale,
        offs_ext,
        weight_vec, sorted_tokens_valid,
        output,
        M_pad,
        I, H,
        I // BLOCK_K_2,
        H * I,
        nHB * nIB,
        BLOCK_M_2, BLOCK_N_2, BLOCK_K_2, NUM_STAGES_2,
        num_warps=NUM_WARPS_2,
    )
    return output
