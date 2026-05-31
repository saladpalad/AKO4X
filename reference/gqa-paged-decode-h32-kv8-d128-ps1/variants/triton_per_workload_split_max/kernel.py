"""
iter3-per-workload-split-routing — round-4 fork of `triton_pdl_grid_dispatch` (1.44x).

============================================================================
1. Identity
============================================================================
Operator: gqa_paged_decode_h32_kv8_d128_ps1
DSL: Triton (3-kernel: fused / split / reduce; PDL-overlapped).
Parent: `triton_pdl_grid_dispatch` (round-2 anchor, 1.44x variance-checked).
Target GPU: B200, modal.
Round-4 score: 1.42x single-run; drift-cancelled A/B vs iter-2 (parent-like):
  +0.27% headline, +1.14% B=16. Real per-workload gains on long-KV B=16:
  1e15ed03 (avg_kv_len 808): +27%; ccdc67b6 (avg 1306): +20%.

============================================================================
2. Delta from parent
============================================================================
- **Per-workload SPLIT_MAX routing on host-available `avg_kv_len`.** Parent
  used a constexpr `SPLIT` keyed only on batch_size. Iter-1 tried global
  SPLIT_MAX=8 for B=16: regressed -3.5% because the reduce kernel's
  per-CTA partial-buffer load doubled (4 slots → 8) for the 14 short-KV
  workloads where no boost was needed. Iter-3 fixes this by routing
  per-workload on `avg_kv_len = kv_indices.shape[0] // batch_size` — a
  host-only shape query, no GPU sync (round-1's closed dead-end on
  `.item()` peek does not apply).

  - B=16, avg_kv_len < 400 (14 of 16 workloads): SPLIT_MAX=4 = BASE_SPLIT,
    exactly equivalent to parent's constexpr SPLIT=4.
  - B=16, avg_kv_len >= 400 (2 of 16: 1e15ed03 avg=808, ccdc67b6 avg=1306):
    SPLIT_MAX=8 with dynamic boost. Each CTA computes
    `dynamic_split = min(SPLIT_MAX, max(BASE_SPLIT, cdiv(kv_len, MIN_TOKENS_PER_CTA)))`
    so per-batch short-KV stays at 4 splits, per-batch long-KV pushes
    toward 8.
  - B=64: SPLIT_MAX=BASE_SPLIT=8. All 16 workloads uniform kv_len 795-921 →
    dynamic_split always 8, matches parent's constexpr.
  - B=1: untouched fused path.

- **CTAs with `s >= dynamic_split` write `m=-inf, l=0` markers and return.**
  Reduce kernel filters on `m == -inf` slots: PARTIAL_O load is masked
  (`mask=(~is_slot_empty)[:, None], other=0.0`) to handle uninit memory
  safely.

============================================================================
3. Closed dead-ends from this round (round-4)
============================================================================
- **iter-1: SPLIT_MAX=8 globally for B=16** (no routing). Headline -3.5%,
  B=16 -18%. Reduce kernel per-CTA partial load doubles (parent SPLIT=4 →
  iter-1 SPLIT_MAX=8). On 14/16 B=16 workloads that don't need boost, this
  inflation eats more than the long-KV gain on the 2 that do. **CLOSED**:
  any future SPLIT_MAX > 4 for B=16 must be gated per-workload.

- **iter-4: route B=16 short-KV (avg_kv_len<400) through fused kernel**.
  Headline -5.55%, B=16 -24%. Per-batch kv_len variance within "short-KV"
  workloads (e.g. 0ea2f83b has kv_len 129..466) is enough that the longest
  fused CTA dominates the wall time. The avg_kv_len trigger doesn't bound
  max-CTA work. Round-3 already closed uniform fused-B16 (-6%); iter-4
  confirms the closure extends to partial-routing-by-avg.

- **iter-5: reduce kernel num_warps 2→4**. Headline -3.22%, B=64 -9%. The
  reduce kernel's per-CTA work (2-4KB partial_o load) is too small to
  amortize the extra warp overhead. Stay at num_warps=2.

============================================================================
4. Inherited closed dead-ends (rounds 1-3) — do not re-try
============================================================================
- One-launch atomic-last-writer reduce: -10-15% (compiler keeps reduce
  locals in live-set; release-add inserts membar.gl).
- B=16 constexpr SPLIT 4→8 (round-1 iter-7, round-2 iter-8, round-3): chunks
  pipeline-amortize differently across workloads — iter-3 of round-4
  partially solves with per-workload routing.
- split num_stages=5: regresses on B=64; stay at 4.
- num_warps=2 uniform on split: -2.35%.
- Grid swap keeping b at pid(0): no-op (Triton dispatches pid(0)-fastest).
- h_kv-fastest grid applied to B=16: -7%.
- fused-for-B16 (round-3, round-4 iter-4): -6% / -5.5% — load-imbalance.
- BLOCK_N=32-for-B16-split (round-3): -4%.

============================================================================
5. Open directions for round-5
============================================================================
- **Persistent kernel + work-stealing** (round-2/3/4 brief). Dynamically
  reassigns work across CTAs to handle per-batch kv_len variance —
  precisely the failure mode of fused-B=16. Big refactor; substantial
  risk. Best lever given current constraints.
- **Packed reduce kernel** (NCU 97% local speedup). Pack GQA=4 q-heads per
  reduce CTA → grid (B, H_KV) instead of (B, H_QO). Modest expected gain.
- **Cluster-cooperative reduce via DSMEM (num_ctas)**. Eliminates the
  PARTIAL_O HBM round-trip. Triton 3.6 cluster on Blackwell reportedly
  immature — smoke probe first or switch DSL.
- **f16 accumulator / partial_o in bf16** — halves HBM traffic on
  partials. Risky for tolerance (current abs_err already 1e-2 to 1e-3).
- **MIN_TOKENS_PER_CTA / threshold tuning** — current 128 (B=16) and 400
  (avg threshold). Sweep both to see if more workloads can flip to the
  boost path without inflation cost.
"""

import torch
import triton
import triton.language as tl

# PDL intrinsics: Triton 3.6+ exposes these in triton.language.extra.cuda.
# Probe import on module load; if unavailable, fall back to a no-op so the
# kernel still compiles on older Triton.
try:
    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
    _HAS_PDL = True
except (ImportError, AttributeError):  # pragma: no cover
    _HAS_PDL = False

    @triton.jit
    def gdc_launch_dependents():  # type: ignore[no-redef]
        pass

    @triton.jit
    def gdc_wait():  # type: ignore[no-redef]
        pass


@triton.jit
def _flash_decode_fused_kernel(
    Q, K_CACHE, V_CACHE, KV_INDPTR, KV_INDICES,
    OUTPUT, LSE,
    qk_scale,
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
    PARTIAL_O, PARTIAL_M, PARTIAL_L,
    qk_scale,
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
    SPLIT_MAX: tl.constexpr,
    BASE_SPLIT: tl.constexpr,
    MIN_TOKENS_PER_CTA: tl.constexpr,
    GRID_MODE: tl.constexpr,
):
    if GRID_MODE == 0:
        # Mode 0: grid = (B, H_KV, SPLIT_MAX). b varies fastest. Used for
        # B≤16 where L2 already fits the working set; b-fastest dispatch
        # spreads each h_kv across more concurrent batches (round-1 default).
        b = tl.program_id(0)
        h_kv = tl.program_id(1)
        s = tl.program_id(2)
    else:
        # Mode 1: grid = (H_KV, SPLIT_MAX, B). h_kv varies fastest. Used for
        # B=64 where KV working set is ~64MB; 8 consecutive CTAs share (b, s)
        # → same kv_indices slice → L2 hits across the h_kv-fanout.
        h_kv = tl.program_id(0)
        s = tl.program_id(1)
        b = tl.program_id(2)

    kv_start = tl.load(KV_INDPTR + b)
    kv_end = tl.load(KV_INDPTR + b + 1)
    kv_len = kv_end - kv_start

    # Dynamic split per (b, h_kv), floored at BASE_SPLIT to preserve the
    # parent's parallelism on short/medium KV. The boost only kicks in for
    # long KV where parent's constexpr SPLIT was HBM-undersaturated.
    #
    # B=64 (BASE=8, MIN=64): all bench batches have kv_len ≥ 795 →
    #   cdiv ≥ 13 → capped at SPLIT_MAX=8. Equivalent to parent's SPLIT=8.
    # B=16 (BASE=4, MIN=128, SPLIT_MAX=8):
    #   short-KV (kv_len 65..200): cdiv≤2 → floor 4 (matches parent's 4).
    #   medium-KV (kv_len 269..547): cdiv 3..5 → split 4..5.
    #   long-KV (kv_len 809..1307): cdiv 7..11 → split 7..8 (BOOST from
    #   parent's 4 → parent was HBM-undersaturated here).
    dynamic_split = tl.minimum(
        SPLIT_MAX,
        tl.maximum(BASE_SPLIT, tl.cdiv(kv_len, MIN_TOKENS_PER_CTA)),
    )

    offs_h = tl.arange(0, BLOCK_H)
    h_mask = offs_h < GQA

    if s >= dynamic_split:
        # Mark this slot empty so the reduce kernel ignores it. PARTIAL_O is
        # left untouched (uninit) — reduce masks the load on `m == -inf`.
        # Tiny store (BLOCK_H float32 each); ~tens of cycles total.
        tl.store(
            PARTIAL_M + b * stride_pmb + h_kv * stride_pmh + s * stride_pms + offs_h,
            tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32),
            mask=h_mask,
        )
        tl.store(
            PARTIAL_L + b * stride_plb + h_kv * stride_plh + s * stride_pls + offs_h,
            tl.zeros((BLOCK_H,), dtype=tl.float32),
            mask=h_mask,
        )
        gdc_launch_dependents()
        return

    split_size = tl.cdiv(kv_len, dynamic_split)
    chunk_start = s * split_size
    chunk_end = tl.minimum(chunk_start + split_size, kv_len)

    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)

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
    # Releases the dependent (reduce) kernel's grid as soon as this producer's
    # stores commit. Reduce CTAs then run their address-arithmetic / m,l-load
    # in parallel with our tail drain.
    gdc_launch_dependents()


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
    # Grid (H_QO, B): h_q varies fastest. 4 consecutive CTAs share (b, h_kv)
    # and hit the same PARTIAL_O[b, h_kv, :, :, :] slice at different g rows.
    h_q = tl.program_id(0)
    b = tl.program_id(1)
    h_kv = h_q // GQA
    g = h_q % GQA

    offs_s = tl.arange(0, SPLIT)
    offs_d = tl.arange(0, D)
    # All address arithmetic above is producer-independent — runs during the
    # PDL overlap window. Wait for partial-O/M/L visibility only at the first
    # producer-data load (just below).
    gdc_wait()

    m = tl.load(
        PARTIAL_M + b * stride_pmb + h_kv * stride_pmh
        + offs_s * stride_pms + g
    )
    l = tl.load(
        PARTIAL_L + b * stride_plb + h_kv * stride_plh
        + offs_s * stride_pls + g
    )

    # Slot-level empty mask: dynamic-split producer writes m=-inf for unused
    # slots and may leave PARTIAL_O uninitialized (possibly NaN). Mask the
    # PARTIAL_O load so empty slots contribute exactly 0.
    is_slot_empty = m == -float("inf")

    m_global = tl.max(m, axis=0)
    is_empty = m_global == -float("inf")
    m_safe = tl.where(is_empty, 0.0, m_global)
    alpha = tl.exp2(m - m_safe)
    l_global = tl.sum(l * alpha, axis=0)

    po = tl.load(
        PARTIAL_O + b * stride_pob + h_kv * stride_poh
        + offs_s[:, None] * stride_pos + g * stride_pog + offs_d[None, :],
        mask=(~is_slot_empty)[:, None],
        other=0.0,
    )
    out_unnorm = tl.sum(po * alpha[:, None], axis=0)
    safe_l = tl.maximum(l_global, 1.0)
    out = out_unnorm / safe_l

    tl.store(
        OUTPUT + b * stride_ob + h_q * stride_oh + offs_d,
        out.to(OUTPUT.dtype.element_ty),
    )
    lse_val = tl.where(
        is_empty,
        -float("inf"),
        m_global + tl.log2(l_global),
    )
    tl.store(LSE + b * stride_lb + h_q, lse_val)


def _select_split_config(batch_size: int, avg_kv_len: int):
    """Return (SPLIT_MAX, BASE_SPLIT, MIN_TOKENS_PER_CTA) for the split
    kernel.

    Per-workload routing on `avg_kv_len = kv_indices.numel() / batch_size`
    (host-available shape, no GPU sync). The SPLIT_MAX=8 path costs ~2x
    reduce-kernel partial-buffer bandwidth vs SPLIT_MAX=4; only worth it on
    workloads where at least some batches have kv_len >> BLOCK_N (so the
    split-K boost from 4→8 CTAs lands).

    BASE_SPLIT floors dynamic_split at parent's parallelism so short-KV
    batches in a long-KV workload still get parent's behavior. The dynamic
    formula `min(SPLIT_MAX, max(BASE_SPLIT, cdiv(kv_len, MIN_TOKENS)))` is
    applied per (b, h_kv) inside the kernel.

    B=64: avg_kv_len ~800-900 uniform across all bench workloads. SPLIT_MAX=8
      with BASE=8 → dynamic_split always 8 (parent's constexpr).
    B=16, avg_kv_len < 400 (14 of 16 workloads): SPLIT_MAX=4 = BASE → parent's
      constexpr 4, no dynamic, no reduce inflation.
    B=16, avg_kv_len >= 400 (2 of 16 workloads, mean 808 and 1306):
      SPLIT_MAX=8 → boost kicks in for long batches via dynamic_split.
      Per round-4 iter-1 A/B data: this workload-class gained +20-26%
      from the boost.
    """
    if batch_size >= 64:
        return 8, 8, 64
    if batch_size >= 16:
        if avg_kv_len >= 400:
            return 8, 4, 128
        return 4, 4, 128
    if batch_size >= 4:
        return 8, 4, 96
    if batch_size >= 2:
        return 16, 8, 64
    # B=1 takes the fused path; this branch is unreachable.
    return 1, 1, 1


@torch.no_grad()
def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, num_qo_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape

    assert num_qo_heads == 32
    assert num_kv_heads == 8
    assert head_dim == 128
    assert page_size == 1
    assert kv_indptr.shape[0] == batch_size + 1

    GQA = num_qo_heads // num_kv_heads
    BLOCK_H = 16
    BLOCK_N = 64 if batch_size >= 16 else 32
    qk_scale = sm_scale * 1.4426950408889634

    device = q.device
    output = torch.empty(
        (batch_size, num_qo_heads, head_dim),
        dtype=torch.bfloat16, device=device,
    )
    lse = torch.empty(
        (batch_size, num_qo_heads), dtype=torch.float32, device=device,
    )

    # avg_kv_len from host-available shape — no GPU sync.
    avg_kv_len = kv_indices.shape[0] // batch_size

    # B=1 → fused single-launch path (round-2 anchor lever, +20% on B=1).
    # iter-4 closed: short-KV B=16 fused regresses -24% (-43% on worst
    # workloads) — even within "short-KV" B=16 workloads, per-batch kv_len
    # variance (e.g. 0ea2f83b: 129..466) makes the longest-CTA dominate.
    if batch_size == 1:
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
            num_stages=6,
        )
        return output, lse

    SPLIT_MAX, BASE_SPLIT, MIN_TOKENS_PER_CTA = _select_split_config(
        batch_size, avg_kv_len
    )

    partial_o = torch.empty(
        (batch_size, num_kv_heads, SPLIT_MAX, GQA, head_dim),
        dtype=torch.float32, device=device,
    )
    partial_m = torch.empty(
        (batch_size, num_kv_heads, SPLIT_MAX, GQA),
        dtype=torch.float32, device=device,
    )
    partial_l = torch.empty(
        (batch_size, num_kv_heads, SPLIT_MAX, GQA),
        dtype=torch.float32, device=device,
    )

    # Per-batch grid-order dispatch (round-2 iter-12/13). Mode-0 (b-fastest)
    # for B≤16 where L2 fits the working set; mode-1 (h_kv-fastest) for B=64
    # where K/V pages benefit from h_kv-fanout L2 hits.
    if batch_size >= 64:
        grid_split = (num_kv_heads, SPLIT_MAX, batch_size)
        grid_mode = 1
    else:
        grid_split = (batch_size, num_kv_heads, SPLIT_MAX)
        grid_mode = 0
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
        SPLIT_MAX=SPLIT_MAX,
        BASE_SPLIT=BASE_SPLIT,
        MIN_TOKENS_PER_CTA=MIN_TOKENS_PER_CTA,
        GRID_MODE=grid_mode,
        num_warps=4,
        num_stages=4,
        launch_pdl=_HAS_PDL,
    )

    grid_red = (num_qo_heads, batch_size)
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
        SPLIT=SPLIT_MAX,
        num_warps=2,
        num_stages=2,
        launch_pdl=_HAS_PDL,
    )

    return output, lse
