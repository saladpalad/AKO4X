# triton_split_kv_2d_reduce_pdl_rw2 — reference kernel.py header
#
# Identity
#   1.46x labeled-single-run (iter-9, 47/47 passed, Modal B200, CUDA 13.0,
#   Triton 3.6.0, 2026-05-20). Same-session A/B vs the parent variant:
#   +17% cumulative (PDL +11.41% then reduce-num_warps=2 +5.91%, both
#   confirmed in-container). No special config required.
#
# Delta from triton_split_kv_2d_reduce
#   Two compounding wins stacked on the parent, no architectural change to
#   the split-K + 2D-tile-reduce structure:
#   1. **PDL (Program-Dependent Launch) between split → reduce kernels**:
#      `gdc_launch_dependents()` at the end of the split kernel +
#      `gdc_wait()` at the start of the reduce kernel, plus
#      `launch_pdl=True` at both call sites. The overlap window comes from
#      address/constant prep in reduce + the split kernel's tail wave.
#      +11.41% A/B; biggest on small batches where reduce is a non-trivial
#      fraction (b=1 +15%, b=16 +13%, b=64 +4.5%).
#   2. **Reduce kernel num_warps=2 (was 4)**: +5.91% A/B, b=64 +12%.
#      For the tiny reduce tile (H=16, D_CHUNK=32, NS=8 or 16) the original
#      4 warps × 128 threads was over-paralleled and underused the load
#      pipe; 2 warps × 64 threads matches the load-pipe. num_warps=1 also
#      helps but +4.71% (loses on small batches); num_warps=8 was -22%.
#
# Lessons on this variant
#
#   +11.41% PDL between split → reduce kernels
#     How:           In split kernel, call `gdc_launch_dependents()` after
#                    the final store. In reduce kernel, call `gdc_wait()`
#                    before the first load. Both call sites use
#                    `launch_pdl=True`.
#     Why:           The reduce kernel's address/constant prep + early
#                    register setup overlaps with the split kernel's tail
#                    wave on adjacent SMs. Largest win when reduce is a
#                    large fraction of total time (small batches).
#     WHEN narrow:   This kernel's split→reduce pair; b=1 +15%, b=16 +13%,
#                    b=64 +4.5%.
#     WHEN broad:    Any two-kernel pipeline where the consumer has
#                    nontrivial pre-amble work that can run concurrent
#                    with the producer's tail wave; PDL is essentially
#                    free when both kernels are persistent-grid or have
#                    waves > 1.
#
#   +5.91% Reduce kernel num_warps=2 (tile-matched warp count)
#     How:           `num_warps=2` on the 2D-tile reduce kernel; tile shape
#                    [NS=8or16, H=16, D_CHUNK=32].
#     Why:           For tiny per-CTA tiles the load-pipe is the bottleneck,
#                    not the compute parallelism. Over-paralleling threads
#                    fragments the load and the warp scheduler can't hide
#                    the gather latency.
#     WHEN narrow:   This reduce tile shape; b=64 +12% (largest), b=16 +9%,
#                    b=1 +3%. num_warps=1 helps less (+4.71%), num_warps=4
#                    was previous setting, num_warps=8 is -22%.
#     WHEN broad:    Tiny-tile reduce kernels with H × D_CHUNK ≤ ~1KB:
#                    sweep {1, 2, 4} warps; the matching skill heuristic
#                    "tiny-tile prefer num_warps=1" is almost right —
#                    num_warps=2 hits the load-pipe sweet spot when each
#                    thread still has ≥32 elements of work.
#
# Dead-ends tried on this variant
#   Each is an expectation prior. Re-verify cheaply if your toolchain shifted.
#
#   - Chunked-split / "persistent-style" split kernel (iter-1): -33% on
#     b=64 (939f995a 0.763 → 0.368). Reducing in-flight CTA count by 4× kills
#     latency hiding for L1TEX-bound K-gather. The split kernel is latency-
#     bound, not occupancy-bound — more CTAs is the right direction, NOT
#     fewer. **This refutes anchor open-direction (1)** as written: a
#     persistent kernel that consolidates work into fewer CTAs is the wrong
#     lane for this kernel. (A persistent kernel that PRESERVES the CTA
#     count but eliminates Q-load duplication + tail imbalance is still
#     untried.)
#   - splits=32 for b=64 avg≥950 (iter-2): -10% on the targeted workloads.
#     Past ~1024 CTAs the latency-hiding benefit saturates; the additional
#     BW overhead dominates.
#   - splits=16 threshold avg≥1050 (was 900) (iter-6): regression on
#     workloads at avg≈974. Confirms splits=16 IS still needed at avg≈974;
#     r2's threshold ≥600 was set with margin, r3 iter-3 narrowed to ≥900
#     which is the practical edge.
#   - splits=4 for b=64 small-kv avg<200 (iter-4): -9% targeted. Latency-
#     hiding loss > K-BW saving even at 70% mask waste.
#   - num_warps=2 on SPLIT kernel (iter-8): -33%. MMA tile picker degrades
#     at non-4 warps for H=16 matmul; same fail mode as num_warps=8.
#     Split kernel is **locked at num_warps=4**.
#   - tl.dot `acc=` kwarg refactor (iter-7): neutral. Triton already fuses
#     `+= tl.dot()` equivalently.
#   - `eviction_policy="evict_first"` on K_NOPE / K_PE (iter-6b): neutral.
#   - bf16 pre-scaled Q (folding sm_scale into Q): correctness fail.
#     bf16 rounding of `Q * 0.144` loses precision past the per-workload
#     tolerance.
#
# Open directions
#   The remaining b=64 worst workloads (939f995a at 0.78x avg=974 and
#   1c3743b9 at 0.86x avg=1074) appear algorithm-class-limited at this
#   level of effort — the expert reference is structurally different on
#   those shapes (likely a flash-decode that fuses logits/value into one
#   pass). Three lanes from the prior session's carry-forward remain
#   in play:
#   (1) TileLang or CuTe DSL split kernel — finer reg/smem layout control
#       to break the 2-CTAs/SM theoretical ceiling. The acc [16, 512] fp32
#       (8KB) is the binding reg constraint; manual layout could spill
#       acc to smem to free regs. Potentially +5-15% on b=64 large.
#   (2) Per-call-site BLOCK_N variant — BLOCK_N=32 split kernel for b=64
#       small-kv (avg/split < 30) to cut mask waste.
#   (3) FP8 K cache — halves K-gather BW. Requires correctness gymnastics
#       for the per-page scale factor; untried in this campaign.
#   (4) Anchor open-direction "persistent kernel" RE-FRAMED: NOT to reduce
#       CTA count, but to PRESERVE the CTA count while eliminating Q-load
#       duplication across (batch × split) work units. Each persistent CTA
#       processes one (batch, split) but reuses the producer-side Q stage
#       across batches. Engineering cost ≥ TileLang rewrite; deferred.

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
            launch_pdl=True,
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
            num_warps=2,
            launch_pdl=True,
        )

    return output, lse
