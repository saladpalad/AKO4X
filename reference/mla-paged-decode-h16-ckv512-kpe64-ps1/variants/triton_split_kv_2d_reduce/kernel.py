# triton_split_kv_2d_reduce — reference kernel.py header
#
# Identity
#   1.21x (bench iter-8 revert-confirm, 47/47 passed, Modal B200, CUDA 13.2,
#   Triton 3.6.0, 2026-05-20). Group breakdown: b=1 1.34x, b=16 1.32x, b=64 0.99x.
#   No special config required.
#
# Delta from prior (operator reference impl — pure Python)
#   First archived variant of the family. Three architectural pieces compounded
#   to ~1.22x: (1) Triton split-KV flash-decode with bf16 partial output for
#   halved reduce-kernel BW (seed inherited from a prior unarchived session);
#   (2) 2D-tile merged reduce kernel — load [NS, H, D_CHUNK] and reduce over
#   NS axis instead of scalar-loop unroll over splits; reduce grid lifted
#   (B,) → (B, D_CKV/D_CHUNK = 16), 16x reduce CTAs for b=1 (the dominant win);
#   (3) surgical splits=16 for b≥32 ∧ avg≥600 — doubles split kernel grid for
#   b=64 large-kv (waves/SM 1.73 → 3.46, tail effect 50% → 25%) while leaving
#   b=64 medium at splits=8 (avoids over-split mask waste).
#
# Lessons on this variant
#
#   +11% 2D-tile reduce (load NS×H×D_CHUNK, reduce over NS)
#     How:           reduce grid (B,) → (B, D_CKV/D_CHUNK); D_CHUNK=32; replaces
#                    tl.static_range(NUM_SPLITS) scalar-loop reduce.
#     Why:           single-CTA-per-batch reduce was 1-CTA underutilized for
#                    small-batch; 16-CTAs/batch saturates SMs.
#     WHEN narrow:   b=1 (1 → 16 CTAs); also helps b=16 +14%.
#     WHEN broad:    split-K reduce where reduce-grid << SM count; 2D-tile form
#                    on Triton lets the compiler stream-load with per-thread
#                    partial accumulation.
#
#   +17-23% on b=64 large-kv: surgical splits=16 for b≥32 ∧ avg≥600
#     How:           _choose_num_splits dispatches splits=16 only when batch
#                    AND avg-per-batch are both large; remaining cases stay at
#                    target=512 / cap=16.
#     Why:           NCU showed split kernel at 1.73 waves/SM (50% tail effect);
#                    doubling splits → 3.46 waves, 25% tail. Confirmed on
#                    kv=62345 (0.65→0.76), kv=68745 (0.68→0.84), kv=75145 (0.82→1.01).
#     WHEN narrow:   b=64 with avg/seq ≥ 600; b=64 medium (kv=22745/27545)
#                    regressed at splits=16 (over-split mask waste).
#     WHEN broad:    waves/SM near 1 → splits-doubling is monotone; near 0.5 or
#                    when avg/split < ~30 it over-splits and reduce cost climbs.
#
# Dead-ends tried on this variant
#   Each is an expectation prior. Re-verify cheaply if your toolchain shifted.
#
#   - BLOCK_N=128 in split kernel: -22% across-the-board. Smem doubled 72KB → 144KB
#     → occupancy crushed (3 CTAs/SM → 1). Reduced-iter win was dwarfed by occupancy loss.
#   - num_stages=1 in split kernel: -34% on b=64. cp.async pipelining hides
#     more latency than the smem headroom from single-staging can buy. Confirms
#     kernel is latency-bound, not occupancy-bound, despite 10.7% achieved.
#   - num_stages=3 (prior session): smem pressure dropped occupancy.
#   - BLOCK_N=32 (prior session): acc, not k_nope, is the spill culprit; reducing
#     BLOCK_N adds iter overhead without addressing acc.
#   - D-tile across CTAs in split kernel (prior session): doubles K BW because
#     Triton can't reuse K between logits and value matmuls.
#   - num_warps=8 on either kernel: MMA tile-picker selects sub-optimal layouts.
#     Triton skill warns about this at H=16 / M=16.
#   - reduce D_CHUNK=16: -7% headline, -22% on b=64. Hit the store-coalescing
#     floor — 32 B/row stores fragmented into sub-cache-line writes. See triton
#     skill "Store-coalescing floor (output dtype matters)" (added 2026-05-20 r2).
#   - Output/lse buffer cache (prior session): torch.empty already negligible;
#     A/B noise.
#
# Open directions
#   Split kernel is occupancy-pinned at 12.5% theoretical (Block Limit Regs=2
#   AND Block Limit Smem=2 on B200), 0 spills, achieved 10.7% with 30.8%
#   L1TEX scoreboard stalls — gather-bound on tok[:, None] * stride_ckv_n page
#   indirection. Three lanes to push past:
#   (1) Persistent split kernel — launch ~296 CTAs (= 2 CTAs/SM × 148 SMs) and
#       loop over multiple (batch, split) work units per CTA; saves Q-load dup
#       (~5-10% HBM for Q at splits=16) and the 22% NCU-flagged tail imbalance.
#       Estimated +3-7% on b=64 large.
#   (2) TileLang or CuTe DSL for the split kernel — finer reg/smem layout
#       control to break the 2-CTAs/SM ceiling. The acc [16, 512] fp32 (8KB) is
#       the binding reg constraint; manual layout could spill acc to smem to
#       free regs. Potentially +5-15% on b=64 large.
#   (3) Per-call-site BLOCK_N variant — emit BLOCK_N=32 split kernel for b=64
#       small-kv (avg/split < 30) to reduce mask waste. Requires separate
#       Triton compile + smem-pressure analysis.

import math
import torch
import triton
import triton.language as tl


@triton.jit
def _mla_split_kernel(
    Q_NOPE, Q_PE,
    CKV, KPE,
    KV_INDPTR, KV_INDICES,
    O_PART, LSE_PART,
    sm_scale_log2,
    stride_qn_b, stride_qn_h,
    stride_qp_b, stride_qp_h,
    stride_ckv_n, stride_kpe_n,
    stride_op_s, stride_op_b, stride_op_h,
    stride_lp_s, stride_lp_b,
    NUM_SPLITS,
    H: tl.constexpr,
    D_CKV: tl.constexpr,
    D_KPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)

    kv_beg = tl.load(KV_INDPTR + pid_b)
    kv_end = tl.load(KV_INDPTR + pid_b + 1)
    seq_len = kv_end - kv_beg

    work_per_split = tl.cdiv(seq_len, NUM_SPLITS)
    s_beg = pid_s * work_per_split
    s_end = tl.minimum(s_beg + work_per_split, seq_len)

    off_h = tl.arange(0, H)
    off_d = tl.arange(0, D_CKV)
    off_dp = tl.arange(0, D_KPE)

    lse_out_ptr = LSE_PART + pid_s * stride_lp_s + pid_b * stride_lp_b + off_h
    o_out_ptr = (
        O_PART
        + pid_s * stride_op_s
        + pid_b * stride_op_b
        + off_h[:, None] * stride_op_h
        + off_d[None, :]
    )

    if s_beg >= s_end:
        tl.store(lse_out_ptr, tl.full([H], -float("inf"), dtype=tl.float32))
        tl.store(o_out_ptr, tl.zeros([H, D_CKV], dtype=O_PART.dtype.element_ty))
        return

    q_nope = tl.load(
        Q_NOPE + pid_b * stride_qn_b + off_h[:, None] * stride_qn_h + off_d[None, :]
    )
    q_pe = tl.load(
        Q_PE + pid_b * stride_qp_b + off_h[:, None] * stride_qp_h + off_dp[None, :]
    )

    m_i = tl.full([H], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([H], dtype=tl.float32)
    acc = tl.zeros([H, D_CKV], dtype=tl.float32)

    for n_off in range(s_beg, s_end, BLOCK_N):
        off_n = n_off + tl.arange(0, BLOCK_N)
        mask_n = off_n < s_end

        tok = tl.load(KV_INDICES + kv_beg + off_n, mask=mask_n, other=0).to(tl.int64)

        k_nope = tl.load(
            CKV + tok[:, None] * stride_ckv_n + off_d[None, :],
            mask=mask_n[:, None],
            other=0.0,
        )
        k_pe = tl.load(
            KPE + tok[:, None] * stride_kpe_n + off_dp[None, :],
            mask=mask_n[:, None],
            other=0.0,
        )

        s = tl.dot(q_nope, tl.trans(k_nope))
        s += tl.dot(q_pe, tl.trans(k_pe))
        s = tl.where(mask_n[None, :], s * sm_scale_log2, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp2(s - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(k_nope.dtype), k_nope)
        m_i = m_new

    acc_norm = acc / l_i[:, None]
    lse_i = m_i + tl.log2(l_i)

    tl.store(o_out_ptr, acc_norm.to(O_PART.dtype.element_ty))
    tl.store(lse_out_ptr, lse_i)


@triton.jit
def _mla_reduce_kernel(
    O_PART, LSE_PART,
    O_FINAL, LSE_FINAL,
    stride_op_s, stride_op_b, stride_op_h,
    stride_lp_s, stride_lp_b,
    stride_of_b, stride_of_h,
    stride_lf_b,
    NUM_SPLITS: tl.constexpr,
    H: tl.constexpr,
    D_CKV: tl.constexpr,
    D_CHUNK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    off_h = tl.arange(0, H)
    off_d = pid_d * D_CHUNK + tl.arange(0, D_CHUNK)
    off_s = tl.arange(0, NUM_SPLITS)

    lse_all = tl.load(
        LSE_PART
        + off_s[:, None] * stride_lp_s
        + pid_b * stride_lp_b
        + off_h[None, :]
    )
    m_final = tl.max(lse_all, axis=0)
    w_all = tl.where(lse_all == -float("inf"), 0.0, tl.exp2(lse_all - m_final[None, :]))
    sum_w = tl.sum(w_all, axis=0)

    o_all = tl.load(
        O_PART
        + off_s[:, None, None] * stride_op_s
        + pid_b * stride_op_b
        + off_h[None, :, None] * stride_op_h
        + off_d[None, None, :]
    ).to(tl.float32)

    acc = tl.sum(w_all[:, :, None] * o_all, axis=0)
    out = acc / sum_w[:, None]

    tl.store(
        O_FINAL + pid_b * stride_of_b + off_h[:, None] * stride_of_h + off_d[None, :],
        out.to(tl.bfloat16),
    )

    if pid_d == 0:
        lse_final = m_final + tl.log2(sum_w)
        tl.store(LSE_FINAL + pid_b * stride_lf_b + off_h, lse_final)


@triton.jit
def _mla_single_kernel(
    Q_NOPE, Q_PE,
    CKV, KPE,
    KV_INDPTR, KV_INDICES,
    O_FINAL, LSE_FINAL,
    sm_scale_log2,
    stride_qn_b, stride_qn_h,
    stride_qp_b, stride_qp_h,
    stride_ckv_n, stride_kpe_n,
    stride_of_b, stride_of_h,
    stride_lf_b,
    H: tl.constexpr,
    D_CKV: tl.constexpr,
    D_KPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)

    kv_beg = tl.load(KV_INDPTR + pid_b)
    kv_end = tl.load(KV_INDPTR + pid_b + 1)
    seq_len = kv_end - kv_beg

    off_h = tl.arange(0, H)
    off_d = tl.arange(0, D_CKV)
    off_dp = tl.arange(0, D_KPE)

    q_nope = tl.load(
        Q_NOPE + pid_b * stride_qn_b + off_h[:, None] * stride_qn_h + off_d[None, :]
    )
    q_pe = tl.load(
        Q_PE + pid_b * stride_qp_b + off_h[:, None] * stride_qp_h + off_dp[None, :]
    )

    m_i = tl.full([H], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([H], dtype=tl.float32)
    acc = tl.zeros([H, D_CKV], dtype=tl.float32)

    for n_off in range(0, seq_len, BLOCK_N):
        off_n = n_off + tl.arange(0, BLOCK_N)
        mask_n = off_n < seq_len

        tok = tl.load(KV_INDICES + kv_beg + off_n, mask=mask_n, other=0).to(tl.int64)

        k_nope = tl.load(
            CKV + tok[:, None] * stride_ckv_n + off_d[None, :],
            mask=mask_n[:, None],
            other=0.0,
        )
        k_pe = tl.load(
            KPE + tok[:, None] * stride_kpe_n + off_dp[None, :],
            mask=mask_n[:, None],
            other=0.0,
        )

        s = tl.dot(q_nope, tl.trans(k_nope))
        s += tl.dot(q_pe, tl.trans(k_pe))
        s = tl.where(mask_n[None, :], s * sm_scale_log2, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp2(s - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(k_nope.dtype), k_nope)
        m_i = m_new

    out = acc / l_i[:, None]
    lse_final = m_i + tl.log2(l_i)

    tl.store(
        O_FINAL + pid_b * stride_of_b + off_h[:, None] * stride_of_h + off_d[None, :],
        out.to(tl.bfloat16),
    )
    tl.store(LSE_FINAL + pid_b * stride_lf_b + off_h, lse_final)


_LOG2E = 1.4426950408889634

_buffer_cache: dict = {}


def _get_buf(shape, dtype, device):
    key = (shape, dtype, device.index if device.type == "cuda" else -1)
    buf = _buffer_cache.get(key)
    if buf is None:
        buf = torch.empty(shape, dtype=dtype, device=device)
        _buffer_cache[key] = buf
    return buf


def _choose_num_splits(batch_size: int, num_kv: int) -> int:
    avg = num_kv / max(1, batch_size)
    # For b≥32 with substantial per-batch work, push to splits=16. NCU on
    # b=64 large-kv showed waves/SM = 1.73 → 50% tail-effect; doubling
    # splits doubles waves and shrinks the tail-effect fraction. Medium
    # kv b=64 (avg < 600) regresses when over-split — keep splits=8 there.
    if batch_size >= 32 and avg >= 600:
        return 16
    target = 512
    splits = max(1, min(16, target // batch_size))
    while splits > 1 and avg < splits * 16:
        splits //= 2
    return max(1, splits)


def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, H, D_CKV = q_nope.shape
    D_KPE = q_pe.shape[-1]
    device = q_nope.device

    num_kv = kv_indices.shape[0]

    ckv = ckv_cache.view(-1, D_CKV)
    kpe = kpe_cache.view(-1, D_KPE)

    num_splits = _choose_num_splits(batch_size, num_kv)

    output = torch.empty((batch_size, H, D_CKV), dtype=torch.bfloat16, device=device)
    lse = torch.empty((batch_size, H), dtype=torch.float32, device=device)

    sm_scale_log2 = float(sm_scale) * _LOG2E
    BLOCK_N = 64

    if num_splits == 1:
        _mla_single_kernel[(batch_size,)](
            q_nope, q_pe, ckv, kpe,
            kv_indptr, kv_indices,
            output, lse,
            sm_scale_log2,
            q_nope.stride(0), q_nope.stride(1),
            q_pe.stride(0), q_pe.stride(1),
            ckv.stride(0), kpe.stride(0),
            output.stride(0), output.stride(1),
            lse.stride(0),
            H=H, D_CKV=D_CKV, D_KPE=D_KPE, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )
    else:
        o_part = _get_buf((num_splits, batch_size, H, D_CKV), torch.bfloat16, device)
        lse_part = _get_buf((num_splits, batch_size, H), torch.float32, device)

        _mla_split_kernel[(batch_size, num_splits)](
            q_nope, q_pe, ckv, kpe,
            kv_indptr, kv_indices,
            o_part, lse_part,
            sm_scale_log2,
            q_nope.stride(0), q_nope.stride(1),
            q_pe.stride(0), q_pe.stride(1),
            ckv.stride(0), kpe.stride(0),
            o_part.stride(0), o_part.stride(1), o_part.stride(2),
            lse_part.stride(0), lse_part.stride(1),
            num_splits,
            H=H, D_CKV=D_CKV, D_KPE=D_KPE, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )

        D_CHUNK = 32
        _mla_reduce_kernel[(batch_size, D_CKV // D_CHUNK)](
            o_part, lse_part,
            output, lse,
            o_part.stride(0), o_part.stride(1), o_part.stride(2),
            lse_part.stride(0), lse_part.stride(1),
            output.stride(0), output.stride(1),
            lse.stride(0),
            NUM_SPLITS=num_splits,
            H=H, D_CKV=D_CKV, D_CHUNK=D_CHUNK,
            num_warps=4,
        )

    return output, lse
