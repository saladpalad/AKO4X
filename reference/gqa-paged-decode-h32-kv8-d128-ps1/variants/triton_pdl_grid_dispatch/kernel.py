"""
iter14-triton-pdl-grid-dispatch — round-2 campaign-best for
gqa_paged_decode_h32_kv8_d128_ps1.

============================================================================
1. Identity
============================================================================
Operator: gqa_paged_decode_h32_kv8_d128_ps1
DSL: Triton (3-kernel: fused / split / reduce; PDL-overlapped).
Score: **1.44x mean speedup** vs FlashInfer expert (variance-check 3
runs, CV 0.3%; 48/48 PASS). B200, modal. Per-batch:
  B=1  1.29x ± 0.002 (CV 0.1%)  — was 0.78x in round-1 anchor (+65%)
  B=16 1.30x ± 0.012 (CV 0.9%)  — was 1.10x (+18%)
  B=64 1.75x ± 0.000 (CV 0.0%)  — was 1.62x (+8%)
Parent (round-1 anchor): triton_flash_decode_split_stages4 (1.17x).
14 iters in this round.

============================================================================
2. Delta from prior anchor
============================================================================
- **PDL between split and reduce** (iter2; +10% headline). End the split
  kernel with `gdc_launch_dependents()`; in the reduce kernel, put
  `gdc_wait()` immediately before the first PARTIAL load (after all
  address arithmetic + constant materialisation). Launch BOTH kernels
  with `launch_pdl=_HAS_PDL` so cuLaunchKernelEx sets
  PROGRAMMATIC_STREAM_SERIALIZATION. Hides ~2us of the reduce launch
  tail in the split-tail drain window.
- **B=1 → SPLIT=1 fused-only path with deep pipe** (iter3/4/6; +20% on
  B=1). B=1 was launch-tax-bound; routing it through the fused kernel
  eliminates the reduce launch entirely. The fused path runs at 1 CTA/SM
  on B=1, so `num_stages=6` is free (96KB SMEM ≪ 228KB/SM headroom) and
  fully hides the 600-800ns HBM hops over 18-iter long-KV workloads.
- **Per-batch grid-order dispatch on the split kernel** (iter12/13;
  +4% B=64). For B=64, grid `(H_KV, SPLIT, B)` (h_kv at pid(0), varies
  fastest) puts 8 consecutive CTAs on the same (b, s) → identical
  kv_indices slice → 8-CTA fanout improves DRAM request coalescing.
  NCU confirmed DRAM throughput 51.8 → 56.85%. For B≤16, keep the
  round-1 `(B, H_KV, SPLIT)` order (b-fastest) — at B=16 the working
  set already fits L2 comfortably and the new layout regressed -7%.
- **Reduce-kernel grid swap** `(B, H_QO)` → `(H_QO, B)` (iter14; +1.3%).
  4 consecutive reduce CTAs share (b, h_kv) → hit the same 16KB
  PARTIAL_O block at different g rows. Helps B=16 most (reduce ~5us is
  NOT fully PDL-hidden behind the ~10us B=16 split tail).

============================================================================
3. Lessons on this variant (load-bearing for round-3)
============================================================================
- **PDL effective gain is ~2us per overlap window**, not the full launch
  tax. The reduce kernel's HOST overhead (driver dispatch + grid setup)
  amortises in the producer-tail drain, but the kernel's intrinsic
  runtime is not hidden. Implication: for B=16 the reduce's ~5us
  intrinsic runtime is partially exposed — that's why a reduce-kernel
  speedup (iter14) helps B=16 specifically.
- **Triton's `tl.atomic_add(release)` + merged "last CTA does reduce"
  causes register-bloat regression** (iter1). The dead-branch reduce
  locals stay in the live-set across the MMA loop; the per-CTA
  release-add inserts a membar.gl flush. NOT retried.
- **HW dispatch is pid(0)-fastest in Triton.** iter11 swapped grid dims
  but kept `b = tl.program_id(0)` — net no-op. To change dispatch
  order you MUST change which pid axis binds to which logical dim.
- **Pipeline depth saturates around num_stages=4 for the split kernel,
  num_stages=6 for the fused-B=1 kernel.** num_stages=5 on split
  regressed -2.86% (round-1's +3% per-stage trend stops at 4);
  num_stages=8 on fused was noise vs =6.
- **Variance-check overall score (1.44x, CV 0.3%) is rock-solid** but
  individual workload speedups across B=16 oscillate ±5% per session
  even on identical code. Don't trust per-batch deltas under ±5% as a
  signal — always `--ab-compare`.
- NCU on iter13 B=64 split: DRAM 56.85%, achieved occupancy 29.5%
  (theoretical 31.25%), Block-Limit-Regs=5 AND Block-Limit-SharedMem=5.
  92 reg/thread, 39KB dynamic SMEM/block. Memory-throughput bound
  (60.13% Mem Throughput, 33% SM Compute) — compute-side levers have
  diminishing returns until reg/SMEM pressure drops structurally.

============================================================================
4. Dead-ends tried (don't re-try without changing surrounding context)
============================================================================
- **One-launch via atomic last-writer reduce** (iter1, -10-15% on every
  batch). Compiler keeps reduce locals in live-set across split loop;
  per-CTA atomic_add release inserts membar.gl. Would need a separate
  persistent reducer CTA pool with NO shared code with the producer.
- **split num_stages=5** (iter5, B=64 1.69→1.57). Either crosses an
  SMEM/SASS schedule edge or saturates HBM queue depth differently.
- **B=16 SPLIT 4 → 8** (iter8, -3.20%). Chunks become ~250 tokens =
  4 BLOCK_N=64 iters per CTA = pipeline doesn't amortize. Confirmed
  round-1's iter-7 even with PDL hiding the reduce-relaunch cost.
- **Split kernel num_warps=2 uniformly** (iter9, -2.35%; B=16 +0.05
  but B=64 -0.15). Mixed-sign — can't be a single uniform value.
- **Dispatched num_warps (B=16→2, B=64→4)** (iter10, +0.19% noise).
  Iter9's B=16 +0.05 was within session drift.
- **Grid swap `(B, H_KV, S) → (B, S, H_KV)` keeping b at pid(0)**
  (iter11, no-op). Triton HW dispatch is pid(0)-fastest; later-dim
  reordering changes nothing.
- **h_kv-fastest grid applied uniformly** (iter12, B=16 -7%). At
  B=16 working set fits L2 → no upside, layout's wave composition
  hurts somewhere else.

============================================================================
5. Open directions (round-3 fork levers)
============================================================================
1. **Persistent kernel** (round-2 brief's lever 2). Each CTA owns a
   full (b, h_kv) and streams all KV — no inter-CTA reduce, no SPLIT
   parallelism overhead. Loses on B=1 (only 8 CTAs); needs intra-CTA
   work-stealing across (b, h_kv). Substantial refactor; expected
   modest gain over 1.44x current.
2. **Cluster-cooperative reduce via num_ctas + DSMEM.** B200 cluster
   exposes DSMEM for inter-CTA exchange. SPLIT∈{4, 8, 16} all fit in
   a cluster — would eliminate the PARTIAL_O HBM round-trip entirely.
   Triton 3.6 cluster on Blackwell is reportedly immature; smoke-test
   with `num_ctas=` before structuring a round around it.
3. **Register-pressure reduction.** NCU shows split kernel at 92
   reg/thread, occupancy 29.5% (cap 31.25%), Block-Limit-Registers=5.
   "Est. Local Speedup 68.75% if occupancy maxed". Hard to action
   without structural changes (smaller acc tile, f16 acc, pack-4-GQA
   reduces useful MMA rows from 4/16 → 16/16).
4. **GQA packing** (round-2 brief's lever 2 alt). MMA at 33% SM
   compute → 4× useful rows per MMA. But DRAM still 57% binding,
   so brief's "lower leverage" call remains right.
5. **TMA bulk-gather of K/V**. Per Triton skill, tiles need ≥64KB to
   beat ld.global.b128 + cp.async pipeline. Our K/V tile at BLOCK_N=64
   is 16KB. Lower priority.
6. **Smaller-dtype accumulation** (f16/bf16 V·acc). Brief's lever 6;
   precision-sensitive; defer until acc-error budget characterised.
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
    SPLIT: tl.constexpr,
    GRID_MODE: tl.constexpr,
):
    if GRID_MODE == 0:
        # Mode 0: grid = (B, H_KV, SPLIT). b varies fastest. Used for B≤16
        # where L2 already fits the working set comfortably; this layout
        # spreads each h_kv across more concurrent batches and was the
        # round-1 default.
        b = tl.program_id(0)
        h_kv = tl.program_id(1)
        s = tl.program_id(2)
    else:
        # Mode 1: grid = (H_KV, SPLIT, B). h_kv varies fastest. Used for
        # B=64 where the KV working set is ~64MB and L2 capacity matters;
        # 8 consecutive CTAs share (b, s) → same kv_indices slice → same
        # K/V pages → L2 hits across the h_kv-fanout.
        h_kv = tl.program_id(0)
        s = tl.program_id(1)
        b = tl.program_id(2)

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
    # and hit the same PARTIAL_O[b, h_kv, :, :, :] slice (16KB for SPLIT=8,
    # GQA=4, D=128 f32) at different g rows. Likely L1/L2 hits on the
    # ~16KB partial block.
    h_q = tl.program_id(0)
    b = tl.program_id(1)
    h_kv = h_q // GQA
    g = h_q % GQA

    offs_s = tl.arange(0, SPLIT)
    offs_d = tl.arange(0, D)
    # All address arithmetic above is producer-independent — runs during the
    # PDL overlap window. Wait for partial-O/M/L visibility only at the
    # FIRST producer-data load (just below).
    gdc_wait()

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
    lse_val = tl.where(
        is_empty,
        -float("inf"),
        m_global + tl.log2(l_global),
    )
    tl.store(LSE + b * stride_lb + h_q, lse_val)


def _select_splits(batch_size: int) -> int:
    if batch_size >= 64:
        return 8
    if batch_size >= 16:
        return 4
    if batch_size >= 4:
        return 8
    if batch_size >= 2:
        return 16
    # B=1: all bench workloads have KV ≤ 547 tokens (mean ~105), so the
    # fused single-launch path beats split+reduce — the work per CTA is
    # ~3us at the worst case and there's no second kernel launch.
    return 1


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
    SPLIT = _select_splits(batch_size)
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
        # Fused path is hit only at B=1 (1 CTA/SM, 228KB SMEM headroom).
        # iter4 measured num_stages=2→4 gave +14% on B=1; deeper pipe is
        # cheap here because we are nowhere near the occupancy floor.
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

    # iter12 measured the h_kv-fastest grid layout: +6% on B=64 but -7%
    # on B=16. Dispatch per batch: mode-0 (b-fastest, round-1 default)
    # for B≤16 where L2 already fits comfortably; mode-1 (h_kv-fastest)
    # for B=64 where the working set is large enough that promoting K/V
    # pages into L2 via the h_kv-fanout pays off.
    if batch_size >= 64:
        grid_split = (num_kv_heads, SPLIT, batch_size)
        grid_mode = 1
    else:
        grid_split = (batch_size, num_kv_heads, SPLIT)
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
        SPLIT=SPLIT,
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
        SPLIT=SPLIT,
        num_warps=2,
        num_stages=2,
        launch_pdl=_HAS_PDL,
    )

    return output, lse
