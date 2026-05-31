# triton_split_kv_2d_reduce_pdl_rw2_brdispatch — reference kernel.py header
#
# Identity
#   1.48x labeled-single-run (iter-8, 47/47 passed, Modal B200, CUDA 13.0,
#   Triton 3.6.0, 2026-05-20). MARGINAL sibling of the parent (1.46x).
#   Headline Δ +1.4% is within Modal session noise (~5-15% drift); the
#   per-batch signal IS real — same-session A/B vs the parent shows
#   b=16 +8-11pp (1.50→1.61), b=64 +1-2%, b=1 noise-bound (-6% same-code
#   noise floor). NOT variance-verified across sessions yet; treat
#   headline as "≥ parent" and per-batch b=16 as the actual lever.
#   No special config required.
#
# Delta from triton_split_kv_2d_reduce_pdl_rw2
#   One tuning change in `_choose_reduce_dispatch` (or equivalent
#   call-site logic): for b≥16 use D_CHUNK=64 / num_warps=4; for b=1
#   keep D_CHUNK=32 / num_warps=2 (the parent's setting). Bigger
#   reduce-tile amortizes the per-CTA work over more elements and
#   recovers store-coalescing (D_CHUNK=64 × bf16 = 128 B/row, full
#   cacheline); the doubled per-CTA work matches num_warps=4. At b=1
#   the grid-fill loss from halving reduce CTAs (16 → 8) dominates the
#   per-CTA win, so the conditional preserves the parent's b=1 path.
#
# Lessons on this variant
#
#   +8-11pp on b=16, +1-2pp on b=64: D_CHUNK=64 nw=4 reduce-tile for b≥16
#     How:           dispatch reduce kernel with D_CHUNK=64 / num_warps=4
#                    when batch_size ≥ 16; keep D_CHUNK=32 / num_warps=2
#                    for b=1.
#     Why:           At b≥16 the reduce CTA is fully load-pipe-bound; the
#                    doubled per-CTA work (D_CHUNK=32→64) doubles store
#                    width (64 B→128 B = full cacheline) and reduces
#                    launch overhead per batch. num_warps=4 matches the
#                    doubled work; num_warps=2 left half the load pipe
#                    idle (verified iter-4 -2.64%).
#     WHEN narrow:   batch_size ≥ 16; b=1 grid-fill-loss dominates the
#                    per-CTA win (16 reduce CTAs → 8 halves the SM
#                    saturation; verified by iter-3's b=1 -7.9%).
#     WHEN broad:    Tiny-tile reduce kernels where increasing D_CHUNK
#                    doubles store width AND register pressure; the
#                    num_warps optimum shifts up with per-CTA work — at
#                    D_CHUNK=64 nw=4 is optimal, at D_CHUNK=32 nw=2 was
#                    optimal (parent variant). Sweep both jointly, not
#                    independently.
#
# Dead-ends tried on this variant
#   Each is an expectation prior. Re-verify cheaply if your toolchain shifted.
#
#   - Per-call-site BLOCK_N=32 dispatch in SPLIT kernel for b=64 small-kv
#     (iter-0/1): Δ +0.29% A/B, per-batch ±6% noise. Triton's
#     mask-predicated K-loads on B200 either DO skip the BW (and the
#     smaller MMA tile cost cancels) OR don't skip it (no win anywhere).
#     **Refutes the prior "per-call-site BLOCK_N variant" open-direction.**
#   - Split kernel num_stages=3 (iter-2): -10% across batches. +50% smem
#     per CTA (72KB → 108KB) cuts split-kernel occupancy from 2 → 1
#     CTAs/SM, halving the in-flight latency-hiding pool. Confirms the
#     kernel is latency-bound on K-gather: more in-flight CTAs >
#     deeper per-CTA pipeline.
#   - Reduce D_CHUNK=128 nw=8 for b=64 (iter-6): -1.5pp vs iter-5's
#     D_CHUNK=64 nw=4. Past D_CHUNK=64 the per-CTA register pressure
#     rises faster than the store-coalescing benefit. D_CHUNK=64 is
#     the reduce-tile ceiling for this shape.
#   - Reduce nw=8 for D_CHUNK=64 b=64 (iter-7): -11.5% on b=64. Same
#     over-paralleling failure mode as the parent's nw=8 dead-end at
#     D_CHUNK=32. Reduce kernel's optimal nw scales with per-CTA work
#     (D_CHUNK=32→nw=2, D_CHUNK=64→nw=4, D_CHUNK=128→nw≤4); going past
#     the optimum fragments the load pipe.
#
# Open directions
#   The Triton-side structural ceiling appears reached: split kernel
#   occupancy-pinned at 12.5% theoretical, reduce kernel at its
#   per-batch-tile sweet spot. The two algorithm-class-limited b=64
#   workloads (939f995a 0.84x avg=974, 1c3743b9 0.92x avg=1074) remain
#   stuck. Three lanes from the parent's carry-forward remain in play
#   but are engineering-heavy:
#   (1) TileLang or CuTe DSL single-pass flash-decode that fuses
#       logits/value into one kernel — likely what the expert reference
#       does on the worst b=64 shapes. ~2h to first benchable state;
#       beware @cute.kernel.launch + torch.cuda.graph capture (see
#       sibling-family TRAPS in this repo).
#   (2) Persistent reduce with batch-tile (B_TILE=2) for b=64: halves
#       reduce CTA count to 256 → eliminates the tail wave. Smaller
#       scope (~30 min). Predicted +2-3% on b=64.
#   (3) Variance-check (5-run) of this variant vs the parent to
#       confirm the +1.4% headline is real (currently at noise floor).
#       Cheap, decisive on whether to keep the brdispatch lever.

import math
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait


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
    gdc_launch_dependents()


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

    gdc_wait()
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
    # Large b=64 with very large kv: push to splits=16 (validated at avg≥974
    # in anchor). Threshold widened to 900 in iter-3.
    if batch_size >= 32 and avg >= 900:
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
            H=H, D_CKV=D_CKV, D_KPE=D_KPE, BLOCK_N=64,
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
            H=H, D_CKV=D_CKV, D_KPE=D_KPE, BLOCK_N=64,
            num_warps=4, num_stages=2,
            launch_pdl=True,
        )

        # Conditional reduce-tile dispatch (iter-5 anchor):
        #  - b=1: small reduce grid (1 × D_CKV/D_CHUNK = 16 CTAs); keep
        #    D_CHUNK=32 num_warps=2 (anchor) — more CTAs help fill the SMs.
        #  - b≥16: D_CHUNK=64 num_warps=4 wins via 128B coalesced store +
        #    halved CTA count. Pushed further (D_CHUNK=128 nw=8) regressed
        #    in iter-6 (-1.5pp); nw=8 with D_CHUNK=64 regressed in iter-7
        #    (-11pp b=64). D_CHUNK=64 nw=4 is the reduce-tile sweet spot.
        if batch_size >= 16:
            D_CHUNK = 64
            reduce_warps = 4
        else:
            D_CHUNK = 32
            reduce_warps = 2
        _mla_reduce_kernel[(batch_size, D_CKV // D_CHUNK)](
            o_part, lse_part,
            output, lse,
            o_part.stride(0), o_part.stride(1), o_part.stride(2),
            lse_part.stride(0), lse_part.stride(1),
            output.stride(0), output.stride(1),
            lse.stride(0),
            NUM_SPLITS=num_splits,
            H=H, D_CKV=D_CKV, D_CHUNK=D_CHUNK,
            num_warps=reduce_warps,
            launch_pdl=True,
        )

    return output, lse
