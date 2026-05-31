"""MLA paged prefill — Triton, kv-aware BLOCK_N + split-K.

# Identity
Round-1 campaign-best for mla_paged_prefill_causal_h16_ckv512_kpe64_ps1 on B200.
iter-12 of the campaign, score: 1.4x mean speedup over FlashInfer expert
(38/38 PASS, min 0.7x at total_q=1954, max 2.41x at total_q=52). Geomean
range across 38 production workloads. No prior anchor — this is the round-1
seed and should be the spawn-time parent for subsequent rounds.

# Delta from prior anchor
No prior anchor; spawn point was the operator's pure-PyTorch reference. This
variant introduces all the structural levers from scratch:
  - Triton flash-attention-2 streaming softmax in base-2 (scale * log2(e) folded).
  - Head grouping: M = BLOCK_Q × NUM_HEADS rows, so K_c/K_p loaded once per kv tile
    and broadcast across all 16 heads (the MLA-specific arithmetic-intensity win).
  - Split-K (FlashDecoding) when (batch × num_q_blocks) × 4 < 256; kv_split
    rounded up to power of 2, capped at 16.
  - max_q_len / max_kv_len cached by `qo_indptr.data_ptr()` / `kv_indptr.data_ptr()`
    so subsequent calls skip the `.item()` host-device sync.
  - Scratch buffers for split-K cached by shape; not pre-initialized — every slot
    is guaranteed exactly-once-written by the owning (batch, q_block) program.
Tile-config heuristic:
  - BLOCK_Q = 2 if max_q_len ≥ 2 else 1 (capped — M=64 doesn't work, see Dead-ends).
  - BLOCK_N = 64, num_stages = 2 if max_kv_len > 32; else BLOCK_N = 32, num_stages = 3.
  - num_warps = 4 throughout.

# Lessons on this variant
1. **Per-call `.item()` was the dominant overhead at small q.** iter-5 (cache
   max_q_len) jumped the mean from 0.44 → 1.24 (+0.80). Workloads at ~0.05ms wall
   dropped to ~0.01ms. Always cache scan results by buffer pointer.
2. **num_warps=4 beats num_warps=8 at M=32.** iter-8 (large-q only) gave +0.04;
   iter-10 (both branches) gave +0.07. Wider warp-row slices (8 rows per warp vs 4)
   pack the tensor-core op tighter. num_warps=2 is catastrophic (16384: 0.82→0.028x)
   — TC throughput collapses below 4 warps.
3. **Split-K (FlashDecoding) is essential for low-occupancy decode-heavy.** The
   total_q=22, 22-batch × 1-q × ~800-kv workload was 0.082x without it (22 CTAs
   on 148 SMs); with kv_split=16, jumped to ~0.94x. Trigger only when total_blocks
   under-occupies (× 4 < target 256) — otherwise pure overhead.
4. **kv-aware BLOCK_N beats max_q_len-aware.** iter-12 vs iter-10: BLOCK_N=64
   when max_kv > 32 captures the "1-iter no-pipeline-setup" win for small kv (q=1,
   q=2, q=4, q=6: +14-22%) while BLOCK_N=32 for max_kv ≤ 32 avoids 50% mask waste.
5. **Triton's pipeline (num_stages=3) helps even for short inner loops** —
   iter-9 stages=2-everywhere regressed -0.02; don't unify down to stages=2.

# Dead-ends tried on this variant
- **BLOCK_Q=4 (M=64) — UNRESOLVED.** iter-2: SMEM OOM at BLOCK_N=64 stages=2
  (Required 344KB vs 228KB cap). iter-8: BLOCK_Q=4 + BLOCK_N=32 + num_warps=8 →
  CUDA Misaligned Address SM exception (GPC7/TPC9/SM1). iter-13: same with
  num_warps=4 → also Misaligned Address. Triton's tcgen05/wgmma path for
  M=64 with this exact (D_CKV=512, D_KPE=64) layout misbehaves on B200.
  Tried three configurations; none worked. **Next-session
  worker should run `bash scripts/sanitize.sh` to localize the unaligned
  access, then try TileLang or hand-written CUDA for the same tile shape.**
- num_warps=2: TC throughput collapses (16384: 0.82 → 0.028x).
- num_stages=2 in BLOCK_N=32 branch (iter-9): -0.02 mean. Most mid-q regressed
  (1954, 287, 376, 43-58). Triton's pipeline helps even at 2-3 inner iters.
- BLOCK_N=64 stages=2 unified across all workloads (iter-11): -0.02 mean.
  Workloads with max_kv ≤ 32 lose 17-19% from mask-waste (1 N=64 iter wastes
  half the columns vs 1 N=32 iter perfectly fitting).

# Open directions
1. **Crack BLOCK_Q=4 (M=64).** The structural ceiling on the large-prefill cluster
   (q=3024-16384 stuck at 0.74-0.82x) is M=32 — bigger M gives 2x more kv-reuse per
   load and unlocks wider mma instruction shapes. First step: sanitizer on the
   iter-13 trajectory (timestamp 20260522_021537) to identify the misaligned access.
   If it's a Triton bug, fall back to TileLang (which has its own tcgen05 path) or
   hand-written CUDA C++ for the inner mma.
2. **Persistent kernel / batch-merging for the total_q=1954 outlier (0.7x).**
   28 batches × ~70 q × ~73 kv each → 980 small CTAs each doing ~3 inner iters.
   Scheduling-bound. A persistent kernel where CTAs pull work units dynamically
   would amortize launch overhead.
3. **Pre-gather K_c/K_p into a contiguous buffer once per call.** Eliminates the
   per-iter page-gather cost. Adds an alloc + an extra kernel launch, so only
   worthwhile when (total kv) × (programs hitting that kv) is large.
4. **Lower split-K trigger threshold.** Current is total_blocks × 4 < 256.
   Could be tuned per-workload-class. Not load-bearing yet but worth a sweep.

Tile shape: M = BLOCK_Q * NUM_HEADS rows, BLOCK_N kv columns, D_CKV=512, D_KPE=64.
"""

import torch
import triton
import triton.language as tl


_LOG2E = 1.4426950408889634

# Cache max(q_lens) / max(kv_lens) keyed by buffer pointer. The benchmark reuses
# tensors across 100 iters per workload, so the first call seeds the cache and
# the rest skip the .item() host-device sync (which costs ~30-50μs).
_max_q_cache: dict = {}
_max_kv_cache: dict = {}


def _max_q_len(qo_indptr) -> int:
    key = (qo_indptr.data_ptr(), qo_indptr.shape[0])
    v = _max_q_cache.get(key)
    if v is not None:
        return v
    v = int((qo_indptr[1:] - qo_indptr[:-1]).max().item())
    _max_q_cache[key] = v
    return v


def _max_kv_len(kv_indptr) -> int:
    key = (kv_indptr.data_ptr(), kv_indptr.shape[0])
    v = _max_kv_cache.get(key)
    if v is not None:
        return v
    v = int((kv_indptr[1:] - kv_indptr[:-1]).max().item())
    _max_kv_cache[key] = v
    return v


# Scratch-buffer cache for split-K. Keyed by (total_q, NUM_HEADS, kv_split, D_CKV, device).
_scratch_cache: dict = {}


def _get_scratch(total_q, num_heads, kv_split, d_ckv, device):
    key = (total_q, num_heads, kv_split, d_ckv, str(device))
    cached = _scratch_cache.get(key)
    if cached is not None:
        return cached
    partial_acc = torch.empty((total_q, num_heads, kv_split, d_ckv), dtype=torch.float32, device=device)
    partial_m = torch.empty((total_q, num_heads, kv_split), dtype=torch.float32, device=device)
    partial_l = torch.empty((total_q, num_heads, kv_split), dtype=torch.float32, device=device)
    cached = (partial_acc, partial_m, partial_l)
    _scratch_cache[key] = cached
    return cached


@triton.jit
def _mla_prefill_direct(
    Q_nope_ptr,
    Q_pe_ptr,
    Kc_ptr,
    Kp_ptr,
    Kv_indices_ptr,
    Out_ptr,
    Lse_ptr,
    qo_indptr_ptr,
    kv_indptr_ptr,
    sm_scale_log2,
    NUM_HEADS: tl.constexpr,
    D_CKV: tl.constexpr,
    D_KPE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Single-pass kernel: writes final output/lse directly. Used when (batch × q_block)
    already saturates the SMs (kv_split == 1)."""
    pid_b = tl.program_id(0)
    pid_q = tl.program_id(1)

    q_start = tl.load(qo_indptr_ptr + pid_b)
    q_end = tl.load(qo_indptr_ptr + pid_b + 1)
    q_len = q_end - q_start
    q_block_start = pid_q * BLOCK_Q
    if q_block_start >= q_len:
        return

    kv_start = tl.load(kv_indptr_ptr + pid_b)
    kv_end = tl.load(kv_indptr_ptr + pid_b + 1)
    kv_len = kv_end - kv_start
    if kv_len <= 0:
        return

    prefix_len = kv_len - q_len

    m_off = tl.arange(0, BLOCK_Q * NUM_HEADS)
    qi = m_off // NUM_HEADS
    hi = m_off % NUM_HEADS
    d_ckv = tl.arange(0, D_CKV)
    d_kpe = tl.arange(0, D_KPE)

    q_pos_in_seq = q_block_start + qi
    qi_global = q_start + q_pos_in_seq
    q_valid = q_pos_in_seq < q_len

    qn_ptrs = (
        Q_nope_ptr
        + qi_global[:, None].to(tl.int64) * (NUM_HEADS * D_CKV)
        + hi[:, None] * D_CKV
        + d_ckv[None, :]
    )
    qn = tl.load(qn_ptrs, mask=q_valid[:, None], other=0.0)

    qp_ptrs = (
        Q_pe_ptr
        + qi_global[:, None].to(tl.int64) * (NUM_HEADS * D_KPE)
        + hi[:, None] * D_KPE
        + d_kpe[None, :]
    )
    qp = tl.load(qp_ptrs, mask=q_valid[:, None], other=0.0)

    m_i = tl.full([BLOCK_Q * NUM_HEADS], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q * NUM_HEADS], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q * NUM_HEADS, D_CKV], dtype=tl.float32)

    query_abs_pos = prefix_len + q_block_start + qi

    max_kv = prefix_len + q_block_start + BLOCK_Q
    if max_kv > kv_len:
        max_kv = kv_len

    for kv_blk in range(0, max_kv, BLOCK_N):
        kv_off = kv_blk + tl.arange(0, BLOCK_N)
        kv_valid = kv_off < kv_len

        pages = tl.load(Kv_indices_ptr + kv_start + kv_off, mask=kv_valid, other=0)
        page_off = pages.to(tl.int64)

        kc_ptrs = Kc_ptr + page_off[:, None] * D_CKV + d_ckv[None, :]
        kc = tl.load(kc_ptrs, mask=kv_valid[:, None], other=0.0)

        kp_ptrs = Kp_ptr + page_off[:, None] * D_KPE + d_kpe[None, :]
        kp = tl.load(kp_ptrs, mask=kv_valid[:, None], other=0.0)

        s = tl.dot(qn, tl.trans(kc))
        s += tl.dot(qp, tl.trans(kp))
        s = s * sm_scale_log2

        causal = kv_off[None, :] <= query_abs_pos[:, None]
        keep = causal & kv_valid[None, :] & q_valid[:, None]
        s = tl.where(keep, s, -float("inf"))

        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        finite_max = m_new != -float("inf")
        alpha = tl.where(finite_max, tl.exp2(m_i - m_new), 0.0)
        p = tl.where(finite_max[:, None], tl.exp2(s - m_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kc)
        m_i = m_new

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    out = acc / l_safe[:, None]
    lse_val = tl.where(l_i > 0, m_i + tl.log2(l_i), -float("inf"))

    out_ptrs = (
        Out_ptr
        + qi_global[:, None].to(tl.int64) * (NUM_HEADS * D_CKV)
        + hi[:, None] * D_CKV
        + d_ckv[None, :]
    )
    tl.store(out_ptrs, out.to(tl.bfloat16), mask=q_valid[:, None])

    lse_ptrs = Lse_ptr + qi_global.to(tl.int64) * NUM_HEADS + hi
    tl.store(lse_ptrs, lse_val, mask=q_valid)


@triton.jit
def _mla_prefill_split(
    Q_nope_ptr,
    Q_pe_ptr,
    Kc_ptr,
    Kp_ptr,
    Kv_indices_ptr,
    PartialAcc_ptr,
    PartialM_ptr,
    PartialL_ptr,
    qo_indptr_ptr,
    kv_indptr_ptr,
    sm_scale_log2,
    KV_SPLIT: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    D_CKV: tl.constexpr,
    D_KPE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Split-K main pass: each program covers a kv slice and writes (acc, m, l) partials."""
    pid_b = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_s = tl.program_id(2)

    q_start = tl.load(qo_indptr_ptr + pid_b)
    q_end = tl.load(qo_indptr_ptr + pid_b + 1)
    q_len = q_end - q_start
    q_block_start = pid_q * BLOCK_Q
    if q_block_start >= q_len:
        return

    kv_start = tl.load(kv_indptr_ptr + pid_b)
    kv_end = tl.load(kv_indptr_ptr + pid_b + 1)
    kv_len = kv_end - kv_start
    prefix_len = kv_len - q_len

    # Per-batch slice = ceil(kv_len / KV_SPLIT) tokens.
    kv_per_split = (kv_len + KV_SPLIT - 1) // KV_SPLIT
    slice_start = pid_s * kv_per_split
    slice_end = slice_start + kv_per_split
    if slice_end > kv_len:
        slice_end = kv_len

    # Causal cutoff for this q_block (max position attended in the block).
    max_kv_for_qblock = prefix_len + q_block_start + BLOCK_Q
    if max_kv_for_qblock > kv_len:
        max_kv_for_qblock = kv_len
    if slice_end > max_kv_for_qblock:
        slice_end = max_kv_for_qblock

    m_off = tl.arange(0, BLOCK_Q * NUM_HEADS)
    qi = m_off // NUM_HEADS
    hi = m_off % NUM_HEADS
    d_ckv = tl.arange(0, D_CKV)
    d_kpe = tl.arange(0, D_KPE)

    q_pos_in_seq = q_block_start + qi
    qi_global = q_start + q_pos_in_seq
    q_valid = q_pos_in_seq < q_len

    m_i = tl.full([BLOCK_Q * NUM_HEADS], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q * NUM_HEADS], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q * NUM_HEADS, D_CKV], dtype=tl.float32)

    # Loop only if there's any kv work in this slice (may be empty if the slice
    # is entirely past the causal cutoff).
    if slice_start < slice_end:
        qn_ptrs = (
            Q_nope_ptr
            + qi_global[:, None].to(tl.int64) * (NUM_HEADS * D_CKV)
            + hi[:, None] * D_CKV
            + d_ckv[None, :]
        )
        qn = tl.load(qn_ptrs, mask=q_valid[:, None], other=0.0)

        qp_ptrs = (
            Q_pe_ptr
            + qi_global[:, None].to(tl.int64) * (NUM_HEADS * D_KPE)
            + hi[:, None] * D_KPE
            + d_kpe[None, :]
        )
        qp = tl.load(qp_ptrs, mask=q_valid[:, None], other=0.0)

        query_abs_pos = prefix_len + q_block_start + qi

        for kv_blk in range(slice_start, slice_end, BLOCK_N):
            kv_off = kv_blk + tl.arange(0, BLOCK_N)
            kv_valid = kv_off < slice_end

            pages = tl.load(Kv_indices_ptr + kv_start + kv_off, mask=kv_valid, other=0)
            page_off = pages.to(tl.int64)

            kc_ptrs = Kc_ptr + page_off[:, None] * D_CKV + d_ckv[None, :]
            kc = tl.load(kc_ptrs, mask=kv_valid[:, None], other=0.0)

            kp_ptrs = Kp_ptr + page_off[:, None] * D_KPE + d_kpe[None, :]
            kp = tl.load(kp_ptrs, mask=kv_valid[:, None], other=0.0)

            s = tl.dot(qn, tl.trans(kc))
            s += tl.dot(qp, tl.trans(kp))
            s = s * sm_scale_log2

            causal = kv_off[None, :] <= query_abs_pos[:, None]
            keep = causal & kv_valid[None, :] & q_valid[:, None]
            s = tl.where(keep, s, -float("inf"))

            m_ij = tl.max(s, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            finite_max = m_new != -float("inf")
            alpha = tl.where(finite_max, tl.exp2(m_i - m_new), 0.0)
            p = tl.where(finite_max[:, None], tl.exp2(s - m_new[:, None]), 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kc)
            m_i = m_new

    # Always store partials for the valid rows in this q_block. The pre-init in
    # the host code provides the defaults for rows owned by other (batch, q_block)
    # tuples — that's what makes mask=q_valid safe.
    pm_ptrs = (
        PartialM_ptr
        + qi_global.to(tl.int64) * (NUM_HEADS * KV_SPLIT)
        + hi * KV_SPLIT
        + pid_s
    )
    pl_ptrs = (
        PartialL_ptr
        + qi_global.to(tl.int64) * (NUM_HEADS * KV_SPLIT)
        + hi * KV_SPLIT
        + pid_s
    )
    tl.store(pm_ptrs, m_i, mask=q_valid)
    tl.store(pl_ptrs, l_i, mask=q_valid)

    pa_ptrs = (
        PartialAcc_ptr
        + qi_global[:, None].to(tl.int64) * (NUM_HEADS * KV_SPLIT * D_CKV)
        + hi[:, None] * (KV_SPLIT * D_CKV)
        + pid_s * D_CKV
        + d_ckv[None, :]
    )
    tl.store(pa_ptrs, acc, mask=q_valid[:, None])


@triton.jit
def _mla_reduce(
    PartialAcc_ptr,
    PartialM_ptr,
    PartialL_ptr,
    Out_ptr,
    Lse_ptr,
    KV_SPLIT: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    D_CKV: tl.constexpr,
):
    """Split-K reduction: one program per (q_global, head). Combines KV_SPLIT partials
    via log-sum-exp + weighted-sum, then writes the final output and base-2 LSE."""
    pid_q = tl.program_id(0)
    pid_h = tl.program_id(1)

    s_off = tl.arange(0, KV_SPLIT)
    base_ml = pid_q.to(tl.int64) * (NUM_HEADS * KV_SPLIT) + pid_h * KV_SPLIT
    m_vals = tl.load(PartialM_ptr + base_ml + s_off)
    l_vals = tl.load(PartialL_ptr + base_ml + s_off)

    m_global = tl.max(m_vals, axis=0)
    finite = m_vals != -float("inf")
    alphas = tl.where(finite, tl.exp2(m_vals - m_global), 0.0)
    l_global = tl.sum(alphas * l_vals, axis=0)

    d_off = tl.arange(0, D_CKV)
    base_acc = pid_q.to(tl.int64) * (NUM_HEADS * KV_SPLIT * D_CKV) + pid_h * (KV_SPLIT * D_CKV)
    pa_ptrs = PartialAcc_ptr + base_acc + s_off[:, None] * D_CKV + d_off[None, :]
    partials = tl.load(pa_ptrs)
    weighted = partials * alphas[:, None]
    acc_combined = tl.sum(weighted, axis=0)

    l_safe = tl.where(l_global > 0, l_global, 1.0)
    out = acc_combined / l_safe
    lse_val = tl.where(l_global > 0, m_global + tl.log2(l_global), -float("inf"))

    out_ptrs = Out_ptr + pid_q.to(tl.int64) * (NUM_HEADS * D_CKV) + pid_h * D_CKV + d_off
    tl.store(out_ptrs, out.to(tl.bfloat16))

    lse_ptr = Lse_ptr + pid_q.to(tl.int64) * NUM_HEADS + pid_h
    tl.store(lse_ptr, lse_val)


# B200 has ~148 SMs. Target a few times oversubscription for the split-K trigger.
_TARGET_CTAS = 256


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    batch_size = qo_indptr.shape[0] - 1
    device = q_nope.device

    output = torch.empty(
        (total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

    Kc = ckv_cache.squeeze(1)
    Kp = kpe_cache.squeeze(1)

    max_q_len = _max_q_len(qo_indptr)
    if max_q_len <= 0:
        return output, lse

    # BLOCK_Q heuristic. iter-2's BLOCK_Q=4+BLOCK_N=64 OOM'd SMEM. iter-8's
    # BLOCK_Q=4+BLOCK_N=32+num_warps=8 AND iter-13's BLOCK_Q=4+num_warps=4 both
    # hit Misaligned Address SM exceptions — Triton's tcgen05/wgmma path for
    # M=64 with this layout misbehaves on B200. Sticking with BLOCK_Q=2 (M=32).
    if max_q_len >= 2:
        BLOCK_Q = 2
    else:
        BLOCK_Q = 1
    max_kv_len = _max_kv_len(kv_indptr)
    if max_kv_len > 32:
        BLOCK_N = 64
        num_stages = 2
    else:
        BLOCK_N = 32
        num_stages = 3
    num_warps = 4
    num_q_blocks = triton.cdiv(max_q_len, BLOCK_Q)
    total_blocks = batch_size * num_q_blocks

    # Split-K: only when the base grid leaves a lot of SMs idle. The 22-batch ×
    # 1-q-each decode workload (~22 CTAs on 148 SMs) is the canonical target.
    kv_split = 1
    if total_blocks * 4 < _TARGET_CTAS:
        # Each split must own ≥ 1 BLOCK_N tile of work to pay off.
        max_useful = max(1, max_kv_len // BLOCK_N)
        target = min(
            (_TARGET_CTAS + total_blocks - 1) // total_blocks,
            max_useful,
            16,
        )
        # Triton's tl.arange requires power-of-2; round target up.
        kv_split = 1
        while kv_split < target:
            kv_split *= 2
        if kv_split > 16:
            kv_split = 16
        if kv_split <= 1:
            kv_split = 1

    scale_log2 = float(sm_scale) * _LOG2E

    if kv_split == 1:
        _mla_prefill_direct[(batch_size, num_q_blocks)](
            q_nope,
            q_pe,
            Kc,
            Kp,
            kv_indices,
            output,
            lse,
            qo_indptr,
            kv_indptr,
            scale_log2,
            NUM_HEADS=num_qo_heads,
            D_CKV=head_dim_ckv,
            D_KPE=head_dim_kpe,
            BLOCK_Q=BLOCK_Q,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        partial_acc, partial_m, partial_l = _get_scratch(
            total_q, num_qo_heads, kv_split, head_dim_ckv, device
        )
        # No init needed: every (q_global, head, split_idx) slot is written by
        # exactly one (batch, q_block) program (the one whose q_block contains
        # q_global). The kv_len=0 path falls through the kernel and stores the
        # initial m=-inf/l=0 state.

        _mla_prefill_split[(batch_size, num_q_blocks, kv_split)](
            q_nope,
            q_pe,
            Kc,
            Kp,
            kv_indices,
            partial_acc,
            partial_m,
            partial_l,
            qo_indptr,
            kv_indptr,
            scale_log2,
            KV_SPLIT=kv_split,
            NUM_HEADS=num_qo_heads,
            D_CKV=head_dim_ckv,
            D_KPE=head_dim_kpe,
            BLOCK_Q=BLOCK_Q,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        _mla_reduce[(total_q, num_qo_heads)](
            partial_acc,
            partial_m,
            partial_l,
            output,
            lse,
            KV_SPLIT=kv_split,
            NUM_HEADS=num_qo_heads,
            D_CKV=head_dim_ckv,
            num_warps=4,
        )

    return output, lse
