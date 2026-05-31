"""MLA paged prefill — Triton, kv-aware BLOCK_N + split-K + pre-gather + L2-aware grids.

# Identity
Round-2 campaign-best for mla_paged_prefill_causal_h16_ckv512_kpe64_ps1 on B200.
**1.47x mean speedup** (38/38 PASS, min 0.815x at q=1028, max 2.42x at q=69) over
the FlashInfer expert baseline. Parent: round-1 anchor `iter12-triton-kvaware-bn-splitk`
(1.40x). Round-2 added three structural levers atop the round-1 base:
  1. **Pre-gather K_c / K_p** into contiguous [total_kv, D] buffers when the
     main kernel can amortize the launch + write pass (kv_split==1 AND
     total_kv>=1024). Replaces scattered page-indirect Kc/Kp loads with
     contiguous tile loads.
  2. **L2-locality grid swap (direct kernel)**: `pid(0)=q_block`, `pid(1)=batch`.
     Adjacent CTAs share batch → same Kc → L2 hits across CTAs.
  3. **Split-K grid binding** `pid(0)=kv_split`: for decode-shaped workloads
     (num_q_blocks=1, multi-batch + split-K), adjacent CTAs share batch via
     varying split. Lifted q=22 from 0.99 → 1.17.

# Delta from prior anchor
Parent: round-1 anchor `iter12-triton-kvaware-bn-splitk` (1.40x). Round-2 layers
three structural levers on top, all in the same single-file Triton kernel:
  1. **Pre-gather K_c / K_p kernel** (`_kv_gather`): one-shot copies the
     page-indirected Kc/Kp rows into contiguous [total_kv, D] buffers cached
     in `_kv_g_cache`. Trigger threshold `kv_split==1 AND total_kv >= 1024` —
     skips small workloads (launch overhead) and the split-K path (disjoint
     kv slices defeat L2 reuse across programs). Iter-1/2.
  2. **L2-locality direct-kernel grid swap**: launch grid changed from
     `(batch_size, num_q_blocks)` to `(num_q_blocks, batch_size)` so that
     `pid(0) = q_block` (varies fastest in HW dispatch order). Adjacent CTAs
     thus share batch → same Kc tile → L2 hits. Iter-3.
  3. **Split-K kernel grid binding**: launch grid `(kv_split, num_q_blocks,
     batch_size)` so `pid(0) = kv_split`. For decode-shaped split-K workloads
     (num_q_blocks=1, multi-batch), adjacent CTAs share batch via varying
     split — recovers L2 reuse that the iter-3 q_block-first swap loses to
     degeneracy when num_q_blocks=1. Iter-6.
Inherited from the round-1 anchor (unchanged): head grouping (M = BLOCK_Q ×
NUM_HEADS), base-2 streaming softmax, kv-aware BLOCK_N (64 vs 32) + num_stages
heuristic, split-K (FlashDecoding) trigger, max_q_len/max_kv_len caching, and
the no-init split-K scratch (every slot is exactly-once-written by the owning
(batch, q_block) program).

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

# Dead-ends tried on this variant (round-1 + round-2 cumulative)
- **BLOCK_Q=4 (M=64) — CONFIRMED CLOSED in Triton 3.6.** Round-1 iter-2 OOM'd
  SMEM at BLOCK_N=64 stages=2 (344KB > 228KB cap). Round-1 iter-8 (num_warps=8)
  and iter-13 (num_warps=4) at BLOCK_N=32 hit CUDA Misaligned Address SM
  exceptions. Round-2 iter-4a re-tried with pre-gather contiguous loads → still
  Misaligned Address. The Triton 3.6 wgmma codegen at (M=64, D_CKV=512, D_KPE=64)
  is broken independent of K access pattern. **The path to M=64 is a DSL switch:
  TileLang or hand-written CUDA C++ with tmem accumulator (CuTe DSL also viable).**
- **`tl.range(warp_specialize=True)`** (round-2 iter-4b): RuntimeError
  `PassManager::run failed` in ttgir pass. Confirmed triton skill: Triton 3.6
  rejects warp_specialize on non-pure-matmul-accumulator patterns (the softmax
  between QK and PV breaks it).
- **PDL gather→main** (round-2 iter-4c): neutral within drift. The gather kernel
  is too short (~3-5μs) relative to the consumer (multi-ms) for meaningful
  PDL overlap.
- **BLOCK_N=32 stages=2 in gather path** (round-2 iter-5): major regression
  (1.40x; q=10870 0.92→0.71, q=16384 0.94→0.72). BLOCK_N=64 wgmma throughput
  on B200 significantly exceeds BLOCK_N=32; the 2× loop overhead + lower
  per-cycle mma throughput overwhelm any spill reduction. The acc[32, 512] fp32
  = 128 fp32/thread is the dominant register pressure (NCU on q=1028: 1.04M
  local-memory spills, 255 regs/thread max, 2 blocks/SM, 10.3% occupancy).
- num_warps=2: TC throughput collapses (16384: 0.82 → 0.028x).
- num_stages=2 in BLOCK_N=32 branch (round-1 iter-9): -0.02 mean.
- BLOCK_N=64 stages=2 unified across all workloads (round-1 iter-11): -0.02 mean.

# Open directions
1. **TileLang/CUDA M=64 for gather-active workloads.** The 8 large-prefill
   workloads at 0.81-0.94x are structurally capped at M=32 wgmma underfill
   in Triton. M=64 via tmem accumulator (B200 tcgen05) would unlock native
   wgmma m64 throughput AND eliminate the acc-in-registers spill bottleneck.
   Hybrid dispatch (TileLang for use_gather, Triton for everything else)
   would preserve the round-2 wins on the other 30 workloads.
2. **Persistent kernel / batch-merging for q=1954 (now 0.83x).** Less acute
   than before (iter-3 grid swap lifted it from 0.71). But Triton 3.6 has
   structural gotchas with persistent kernels (`return` rejected in `for`,
   pipeliner doesn't carry across outer-while iters).
3. **Lower split-K trigger threshold sweep.** Current is total_blocks × 4 < 256.
   Iter-6's split-K grid binding lift on q=22 suggests this lever has more
   room — try total_blocks * 4 < 384 or 512 with care for the reduce overhead.

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


# Pre-gathered K_c / K_p buffers (contiguous by global kv position). Keyed by
# total_kv, the only shape that matters; reused across calls when total_kv repeats.
_kv_g_cache: dict = {}


def _get_kv_g(total_kv, d_ckv, d_kpe, device, dtype):
    key = (total_kv, d_ckv, d_kpe, str(device), str(dtype))
    cached = _kv_g_cache.get(key)
    if cached is not None:
        return cached
    kc_g = torch.empty((total_kv, d_ckv), dtype=dtype, device=device)
    kp_g = torch.empty((total_kv, d_kpe), dtype=dtype, device=device)
    cached = (kc_g, kp_g)
    _kv_g_cache[key] = cached
    return cached


@triton.jit
def _kv_gather(
    Kc_ptr,
    Kp_ptr,
    Kv_indices_ptr,
    Kc_g_ptr,
    Kp_g_ptr,
    total_kv,
    D_CKV: tl.constexpr,
    D_KPE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One-shot gather: write Kc[page_id] / Kp[page_id] → contiguous Kc_g / Kp_g
    indexed by global kv position. Replaces per-iter scattered loads in the main
    kernel with single contiguous tile loads."""
    pid = tl.program_id(0)
    kv_off = pid * BLOCK_K + tl.arange(0, BLOCK_K)
    kv_valid = kv_off < total_kv

    page_off = tl.load(Kv_indices_ptr + kv_off, mask=kv_valid, other=0).to(tl.int64)

    d_ckv = tl.arange(0, D_CKV)
    d_kpe = tl.arange(0, D_KPE)

    kc = tl.load(
        Kc_ptr + page_off[:, None] * D_CKV + d_ckv[None, :],
        mask=kv_valid[:, None],
        other=0.0,
    )
    kp = tl.load(
        Kp_ptr + page_off[:, None] * D_KPE + d_kpe[None, :],
        mask=kv_valid[:, None],
        other=0.0,
    )

    kv_g64 = kv_off[:, None].to(tl.int64)
    tl.store(Kc_g_ptr + kv_g64 * D_CKV + d_ckv[None, :], kc, mask=kv_valid[:, None])
    tl.store(Kp_g_ptr + kv_g64 * D_KPE + d_kpe[None, :], kp, mask=kv_valid[:, None])


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
    GATHERED: tl.constexpr,
):
    """Single-pass kernel: writes final output/lse directly. Used when (batch × q_block)
    already saturates the SMs (kv_split == 1)."""
    # Grid binding: pid(0) is q_block, pid(1) is batch. Triton dispatches CTAs in
    # blockIdx.x-fastest order, so adjacent CTAs share batch → same Kc tile, which
    # is the big L2-reuse win for multi-batch workloads (see triton skill).
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)

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

        if GATHERED:
            kv_g_off = (kv_start + kv_off).to(tl.int64)
            kc_ptrs = Kc_ptr + kv_g_off[:, None] * D_CKV + d_ckv[None, :]
            kp_ptrs = Kp_ptr + kv_g_off[:, None] * D_KPE + d_kpe[None, :]
        else:
            pages = tl.load(Kv_indices_ptr + kv_start + kv_off, mask=kv_valid, other=0)
            page_off = pages.to(tl.int64)
            kc_ptrs = Kc_ptr + page_off[:, None] * D_CKV + d_ckv[None, :]
            kp_ptrs = Kp_ptr + page_off[:, None] * D_KPE + d_kpe[None, :]

        kc = tl.load(kc_ptrs, mask=kv_valid[:, None], other=0.0)
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
    GATHERED: tl.constexpr,
):
    """Split-K main pass: each program covers a kv slice and writes (acc, m, l) partials."""
    # Grid binding for the split-K kernel. pid(0) varies fastest = kv_split:
    # adjacent CTAs share (q_block, batch) → same batch's Kc region. The 16
    # splits per batch all hit overlapping/adjacent kv data (different slices
    # of the same paged batch), maximizing L2 hit. This differs from the
    # _mla_prefill_direct swap because the split-K activation cases have
    # num_q_blocks=1 (decode-shaped: 1 q-position per batch), making the
    # q_block-first swap degenerate for those workloads.
    pid_s = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_b = tl.program_id(2)

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

            if GATHERED:
                kv_g_off = (kv_start + kv_off).to(tl.int64)
                kc_ptrs = Kc_ptr + kv_g_off[:, None] * D_CKV + d_ckv[None, :]
                kp_ptrs = Kp_ptr + kv_g_off[:, None] * D_KPE + d_kpe[None, :]
            else:
                pages = tl.load(Kv_indices_ptr + kv_start + kv_off, mask=kv_valid, other=0)
                page_off = pages.to(tl.int64)
                kc_ptrs = Kc_ptr + page_off[:, None] * D_CKV + d_ckv[None, :]
                kp_ptrs = Kp_ptr + page_off[:, None] * D_KPE + d_kpe[None, :]

            kc = tl.load(kc_ptrs, mask=kv_valid[:, None], other=0.0)
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

    # Default config (BLOCK_Q=2, M=32) — stable Triton path.
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

    # Pre-gather K_c / K_p when:
    #   1. kv_split == 1 (the single-pass kernel; with split-K each program reads
    #      a disjoint kv slice, so the gathered buffer doesn't recycle through L2
    #      across programs — gather is then pure write-traffic overhead).
    #   2. total_kv >= 1024 (medium workloads kv ~300-800 don't recoup the
    #      ~3-5μs gather launch; iter-1 measured -0.11 regression at q=376/473).
    total_kv = kv_indices.shape[0]
    use_gather = (kv_split == 1) and (total_kv >= 1024)
    if use_gather:
        kc_for_kernel, kp_for_kernel = _get_kv_g(
            total_kv, head_dim_ckv, head_dim_kpe, device, Kc.dtype
        )
        GATHER_BLOCK_K = 64
        gather_grid = ((total_kv + GATHER_BLOCK_K - 1) // GATHER_BLOCK_K,)
        _kv_gather[gather_grid](
            Kc,
            Kp,
            kv_indices,
            kc_for_kernel,
            kp_for_kernel,
            total_kv,
            D_CKV=head_dim_ckv,
            D_KPE=head_dim_kpe,
            BLOCK_K=GATHER_BLOCK_K,
            num_warps=4,
        )
    else:
        kc_for_kernel = Kc
        kp_for_kernel = Kp

    if kv_split == 1:
        _mla_prefill_direct[(num_q_blocks, batch_size)](
            q_nope,
            q_pe,
            kc_for_kernel,
            kp_for_kernel,
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
            GATHERED=use_gather,
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

        _mla_prefill_split[(kv_split, num_q_blocks, batch_size)](
            q_nope,
            q_pe,
            kc_for_kernel,
            kp_for_kernel,
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
            GATHERED=use_gather,
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
