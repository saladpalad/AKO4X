# Variant: fused_indirect_v1
# Source: ako4fib-run-moe1/solution/kernel.py (iter-40 final, session 2026-04-22)
# Operator: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
#
# Identity
#   1.000x ± 0.001x (3-run variance-check, CV 0.10%, Modal B200 sm_100, CUDA 13.2,
#   flashinfer-ci-cu132:20260401-2c675fb image, 2026-04-23T18:35, canonical
#   baseline). Baseline = flashinfer `trtllm_fp8_block_scale_moe`
#   (`../../baseline.json`, MD5 `a1d2be64…`).
#   Per-T mean speedup from the same 3-run:
#     T=1    0.985x    T=7     1.032x    T=14   1.032x    T=15   1.051x    T=16   0.973x
#     T=32   0.990x    T=52    0.980x    T=53   0.958x    T=54   0.997x    T=55   0.946x
#     T=56   0.923x    T=57    0.943x    T=58   0.956x    T=59   0.916x    T=62   1.114x
#     T=80   0.960x    T=901   0.906x    T=11948 1.195x   T=14107 1.135x
#   Per-T CV ≤ 0.9% (small-T T=1/14/15/16/32 in the 0.3–0.9% range, large-T ≤ 0.2%).
#   Prior header (2026-04-22) reported 1.022x under the pre-canonical global cache
#   baseline (MD5 `836840d3…`); under the canonical this variant drops ~-0.02x,
#   reflecting that the cache baseline had 20-25% slower small-T expert latencies
#   than the canonical reference. Large-T per-T speedups (T=11948, T=14107) are
#   within noise of prior values — canonical baseline is closer to the reference
#   at large T, tighter at small T.
#   Build deps: torch ≥ 2.9 (FP8 dtype), triton ≥ 3.6 (sm_100 UMMA FP8 MMA). No
#   flashinfer / deep-gemm / CUTLASS DSL / TileLang runtime dependency.
#   Config requires `[benchmark] use_isolated_runner = true` on persistent-runner
#   environments.
#
# Delta from presync_v1 (iter-30 snapshot, 0.841x)
#   presync_v1 already carries the indirect GEMM1 gather and fused GEMM2+scatter-add
#   architecture. fused_indirect_v1 layers three launch-overhead wins on top:
#   iter-31 skip `counts.to('cpu')` on the small-T path (T≤256 uses T as an upper bound
#   on max_count); iter-34 inline the exclusive-cumsum inside every consumer kernel,
#   dropping the dedicated prefix kernel; iter-37 merge the counts[] atomic into the
#   routing kernel for T≤256, dropping a separate _dispatch_count_kernel launch.
#   iter-33 added module-level caching of the reusable scratch buffers (counts,
#   counter, sorted_tokens, sorted_experts_buf) to keep torch.empty/.zero_ off the
#   measured path. Cumulative ≈ +0.18x (0.841x → 1.022x); session drift accounts for
#   the rest of the mismatch against the iter-by-iter deltas below.
#
# Lessons on this variant
#
#   +0.21x bucket-scatter dispatch replaces torch.argsort (iter-18)
#     How:           _dispatch_count_kernel histograms topk_idx into counts[E_LOCAL]
#                    via tl.atomic_add(counts, 1);
#                    _dispatch_scatter_kernel emits
#                      slot = exclusive_cumsum(counts)[bucket] + atomic_add(counter[bucket], 1)
#                    to place (token, expert) pairs into the sorted layout.
#     Why:           argsort over T*TOP_K ints is sort-bound; atomic-based histogram +
#                    scatter is two lean kernels with contention bounded by E_LOCAL=32
#                    buckets — on sm_100 L1 atomics this stays cheap through T=14107.
#     WHEN narrow:   top-K expert routing with E_LOCAL ≤ 64 and T up to ~16K; the
#                    downstream consumers need grouped-by-expert layout but no total
#                    order within a group.
#     WHEN broad:    replace a sort with atomic-bucket-scatter when #buckets << #elements
#                    AND the consumer only needs grouped (not totally ordered) output.
#
#   +0.10x skip counts.to('cpu') sync for T ≤ 256 (iter-31)
#     How:           small-T path sets `max_count = T` and
#                    `M_pad = max(ceil(T*TOP_K/128)*128, 128)`; each kernel masks per
#                    group via the inline cumsum over counts[], so overallocated
#                    slots are inert.
#     Why:           one device-to-host sync is ~20-30 µs on Modal B200 and total
#                    small-T latency is ~120-250 µs — the sync is >10% of wall time.
#                    The upper bound is safe because per-group count ≤ T and M_pad
#                    over-allocation at T=256 is ≤ 2048 rows (negligible vs M_pad on
#                    large-T paths).
#     WHEN narrow:   T ≤ 256 on this operator. iter-31's original threshold was T≤128;
#                    the final code widened to T≤256 alongside iter-37 once the
#                    routing kernel absorbed the count-atomic.
#     WHEN broad:    any pipeline where a counts/length scalar gates a host-visible
#                    branch; if a cheap upper bound exists and over-allocation is
#                    small, skip the sync and let kernels mask the slack.
#
#   +0.05x inline exclusive-cumsum in every consumer kernel (iter-34)
#     How:           each consumer (_dispatch_scatter, GEMM1, GEMM2+scatter) loads
#                    `all_counts = counts[0:E_LOCAL]` once per block then computes its
#                    group offset as `tl.sum(tl.where(g_idx < group, all_counts, 0))`.
#     Why:           E_LOCAL = 32 fits in a single warp's register tile, so the
#                    inline prefix is constant-cost. The standalone cumsum kernel was
#                    ~5-10 µs of pure launch overhead contributing no work the
#                    consumers weren't already doing (they all load counts anyway).
#     WHEN narrow:   E_LOCAL ≤ ~64 (the array fits in one warp-tile register file).
#     WHEN broad:    a dedicated prefix kernel is wasted launch overhead when the
#                    prefix base is short AND every downstream kernel already loads
#                    the underlying array.
#
#   +0.03x merge counts atomic into routing kernel for T ≤ 256 (iter-37)
#     How:           `MERGE_COUNT` constexpr flag in _fused_routing_kernel; when set,
#                    after computing topk_mask the kernel also does
#                      tl.atomic_add(counts_ptr + (idx - local_start), 1, mask=local_mask)
#                    for the local-expert subset. _dispatch_count_kernel then only
#                    runs for T > 256.
#     Why:           saves one kernel launch (~5 µs). Atomic contention scales with
#                    T*TOP_K/E_LOCAL lanes/bucket: at T=256 that is ≈ 64 lanes per
#                    bucket, which sm_100 L1 atomics absorb cheaply. At large T the
#                    per-bucket contention outgrows the launch saved, so the merge
#                    is gated on T ≤ 256.
#     WHEN narrow:   T ≤ 256 on this operator (TOP_K=8, E_LOCAL=32 → T/4 lanes/bucket).
#     WHEN broad:    fold a lightweight atomic count into its producer when the
#                    producer's grid already visits every element AND per-bucket
#                    contention stays within the same order as a wave of lanes.
#     Follow-up (v2): the T>256 gate here turned out to be over-conservative on
#                    sm_100 B200. `fused_routing_v2`'s iter-2 merged counts AND the
#                    slot scatter into the routing kernel at all T through T=14107
#                    (3528 atomics/bucket) with +0.093x in-session A/B and no
#                    measurable contention regression.
#
# Dead-ends tried on this variant
#   Each is an expectation prior — retry only if your toolchain or surrounding code
#   flips the Why. Scope is this variant; do not propagate forward without rethinking.
#
#   - Upfront BF16 weight dequant + torch._grouped_mm (iter-2). Expect ~10× regression.
#     Why: the dequant materializes a 3.76 GB transient per call plus a
#     .transpose(1,2).contiguous() copy; the weight traffic alone exceeds the compute
#     budget. Worth a retry only if a zero-copy dequant path appears OR weights can
#     be pre-staged outside the timed region.
#
#   - tl.static_range over the 56-iter K-loop (iter-3). Expect register explosion and
#     zero software-pipelining. Why: a fully-unrolled loop forbids Triton from
#     overlapping loads with math. Use `tl.range(..., num_stages=N)` with dynamic bounds.
#
#   - Grid sized by `ceil(T*TOP_K / BM)` as max_m_tiles (iter-3/4). Expect launch
#     overhead to dominate on large T: at T=14107 this yields ~1.8M blocks, most
#     early-exiting. Scope the grid to `ceil(max_count / BM)` per group instead.
#
#   - M_total passed as a runtime (non-constexpr) int (iter-10). Expect ptr-arithmetic
#     codegen to pessimize. Why: Triton caches by constexpr tuple, so the recompile
#     cost across the handful of distinct T values seen in a session is usually
#     cheaper than leaving strides runtime-variable.
#
#   - Fused GEMM1 + SwiGLU + FP8-quant in one kernel with dual [BM, BN] fp32 accumulators
#     for gate and up (iter-22; iter-23 reverted). Expect TMEM overflow on sm_100.
#     Why: two 64×128×fp32 accumulators = 64 KB ≈ 512 TMEM columns per CTA, hitting
#     the Blackwell per-CTA TMEM cap; any num_stages>1 spills. Keep SwiGLU+quant as a
#     follow-up kernel.
#
#   - BLOCK_M = 32 on small-max_count branches. Expect fallback to SIMT FP8 dot. Why:
#     sm_100 UMMA FP8 requires minimum tile M = 64. Floor BM at 64.
#
#   - BLOCK_M = 128 when max_count < 256. Expect >50% of M rows masked-inactive per
#     tile, wasting MMA compute. Why: at max_count≈50 with BM=128 only ~40% of the
#     tile's M-rows carry work. Current code's BM=128 threshold is max_count ≥ 256.
#
#   - num_stages = 6 on the BM=64 GEMM branch. Expect shmem-footprint spill. Why: 6
#     pipeline copies of (BM=64 × BK=128 FP8 A + BN=128 × BK=128 FP8 B) exceeds the
#     comfortable per-SM shmem budget. num_stages = 4 holds on BM=64; num_stages = 6
#     is only used on BM=128 where B-tile reuse dominates.
#
#   - Inline weight gather in GEMM2 epilogue (iter-38, smoke-level evidence). Expect
#     scattered GMEM loads with poor coalescing vs. the current pre-gathered
#     weight_vec[M_pad]. Full-bench delta not recorded in this session; retry only if
#     pre-gather surfaces as a hotspot in profiles.
#     Follow-up (v2): iter-38 compared `pre-gather-via-fancy-index` vs `inline
#     scatter-gather` — the fancy-index itself was firing 2 small CUDA kernels
#     (add + gather) from Python. `fused_routing_v2`'s iter-1 removes that whole
#     Python path; iter-3 co-locates weight_vec with sorted_tokens so the epilogue
#     load is sequential not scattered. Cumulatively +0.087x in-session A/B. The
#     "scattered gather is bad" lesson is correct; what was missing was that the
#     Python-level fancy-index pre-gather was ALSO bad.
#
#   - End-to-end CUDA Graph capture. Expect graph replay to fail because
#     gemm1_weights and gemm2_weights are fresh tensors with different device pointers
#     each call. Rules out graph-level launch-overhead reductions unless the harness
#     changes upstream.
#
#   - Prefix-sum-based dispatch (cumsum over T*TOP_K for per-element destination slot,
#     iter-18 alt). Expect slower than the atomic bucket-scatter. Why: cumsum scans
#     T*TOP_K int32s (up to ≈112K at T=14107) while atomic scatter touches only the
#     E_LOCAL=32 buckets.
#
#   - Serial per-token SwiGLU (grid = (M_pad,), inner Python loop over nIB, iter-16).
#     Expect small-T regression. Why: the 1D grid leaves SMs idle when M_pad is
#     small; 2D grid (M_pad, nIB) fills the GPU across all T.
#
# Open directions
#   T=901 is the weakest per-T bucket (0.906x under canonical). Its max_count lands
#   on the BM=64 path, leaving the GEMMs latency-bound rather than compute-bound —
#   a persistent GEMM kernel (grid = num_SMs × k with an atomic tile counter) is
#   the shape-appropriate lever for this regime.
#
#   Already-explored downstream: `fused_routing_v2` (1.204x) and
#   `fused_graph_all_t` (1.380x) both dominate this anchor across all T. If
#   starting a fresh optimization pass, begin from fused_graph_all_t instead of
#   v1 unless specifically isolating a pre-graph-capture regression.
#
#   Other plausible levers: BLOCK_N = 256 with two B-scales loaded per tile on the
#   large-T paths (⚠ numerical hazard: `fused_routing_v2` iter-8 hit
#   `max_abs_err ≈ 8.6e5` at T=901 on BM=64 × BN=256 × BK=128 FP8; sanitize memcheck
#   clean, so the bug is in `tl.dot` codegen or scale-broadcast logic at that tile
#   shape, not OOB. Not a drop-in — needs NCU PTX inspection before retry.);
#   native UMMA block-scale via `tl.dot_scaled` if Triton ever exposes
#   128-block scaling (MX format is 32-block and does not match the DeepSeek
#   block-scale layout) — would remove the per-K-iter
#   `acc += partial * (a_sc[:, None] * b_sc)` fp32 multiply. An iterative top-K in the
#   routing kernel replacing tl.sort(256) is only a routing-share win at T=1 where
#   routing is a larger fraction of total latency.

import torch
import triton
import triton.language as tl


# Module-level cache of small reusable buffers (keyed by device).
_BUF_CACHE: dict = {}


def _get_cached(key, shape, dtype, device):
    full_key = (key, shape, dtype, str(device))
    buf = _BUF_CACHE.get(full_key)
    if buf is None:
        buf = torch.empty(shape, dtype=dtype, device=device)
        _BUF_CACHE[full_key] = buf
    return buf


# ─────────────────────────── Triton kernels ───────────────────────────

@triton.jit
def _grouped_fp8_gemm1_indirect_kernel(
    HS_ptr,              # fp8 [T, K]
    HS_scale_ptr,        # fp32 [K//BK, T]
    sorted_tokens_ptr,   # int64 [M_pad]
    B_ptr,               # fp8 [G, N, K]
    B_scale_ptr,         # fp32 [G, N//BN, K//BK]
    C_ptr,               # bf16 [M_pad, N]
    counts_ptr,          # int32 [E_LOCAL]
    T,
    K: tl.constexpr, N: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    stride_bg: tl.constexpr,
    stride_bsg: tl.constexpr,
    E_LOCAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    group = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Inline exclusive-cumsum from counts[] to compute group bounds.
    all_counts = tl.load(counts_ptr + tl.arange(0, E_LOCAL))
    g_idx = tl.arange(0, E_LOCAL)
    g_start = tl.sum(tl.where(g_idx < group, all_counts, 0))
    m_count = tl.sum(tl.where(g_idx == group, all_counts, 0))

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
def _fused_routing_kernel(
    logits_ptr,        # [T, E_GLOBAL] fp32
    bias_ptr,          # [E_GLOBAL] bf16
    topk_idx_out_ptr,  # [T, TOP_K] int32
    weights_out_ptr,   # [T, E_GLOBAL] fp32
    counts_ptr,        # [E_LOCAL] int32 (maybe unused if MERGE_COUNT=False)
    local_start,
    routed_scaling,    # fp32 scalar
    E_GLOBAL: tl.constexpr,    # 256
    N_GROUP: tl.constexpr,     # 8
    GROUP_SIZE: tl.constexpr,  # 32
    TOPK_GROUP: tl.constexpr,  # 4
    TOP_K: tl.constexpr,       # 8
    E_LOCAL: tl.constexpr,     # 32
    MERGE_COUNT: tl.constexpr, # whether to atomic-add to counts[]
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

    # Optional: simultaneously count selected LOCAL experts into counts[] (avoids a dedicated count kernel).
    if MERGE_COUNT:
        shifted = idx_e - local_start                                       # [E_GLOBAL]
        local_mask = topk_mask & (shifted >= 0) & (shifted < E_LOCAL)
        # Atomic-add 1 to counts[shifted] for each lane with local_mask set.
        bucket = tl.where(local_mask, shifted, 0)
        tl.atomic_add(counts_ptr + bucket, 1, mask=local_mask)


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
def _dispatch_scatter_kernel(
    topk_idx_ptr,       # [T, TOP_K] int32
    counts_ptr,         # [E_LOCAL] int32
    counter_ptr,        # [E_LOCAL] int32 (zero-init)
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

    all_counts = tl.load(counts_ptr + tl.arange(0, E_LOCAL))  # [E_LOCAL]
    ec = tl.arange(0, E_LOCAL)

    for k in tl.static_range(TOP_K):
        idx = tl.load(topk_idx_ptr + t_offs * TOP_K + k, mask=t_mask, other=0)
        shifted = idx - local_start
        valid = t_mask & (shifted >= 0) & (shifted < E_LOCAL)
        bucket = tl.where(valid, shifted, 0)

        pos = tl.atomic_add(counter_ptr + bucket, 1, mask=valid)
        # Per-lane exclusive cumsum base: sum all_counts[j] for j < bucket[lane]
        base = tl.sum(tl.where(ec[None, :] < bucket[:, None], all_counts[None, :], 0), axis=1)

        slot = base + pos

        tl.store(sorted_tokens_out + slot, t_offs.to(tl.int64), mask=valid)
        tl.store(sorted_experts_out + slot, shifted, mask=valid)


@triton.jit
def _grouped_fp8_gemm2_fused_scatter_kernel(
    A_ptr,              # fp8 [M_pad, K]
    A_scale_ptr,        # fp32 [K//BK, M_pad]
    B_ptr,              # fp8 [G, N, K]
    B_scale_ptr,        # fp32 [G, N//BN, K//BK]
    counts_ptr,         # int32 [E_LOCAL]
    weight_vec_ptr,     # fp32 [M_pad]
    sorted_tokens_ptr,  # int64 [M_pad]
    output_ptr,         # bf16 [T, N]
    M_total: tl.constexpr,
    K: tl.constexpr, N: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    stride_bg: tl.constexpr,
    stride_bsg: tl.constexpr,
    E_LOCAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    group = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    all_counts = tl.load(counts_ptr + tl.arange(0, E_LOCAL))
    g_idx = tl.arange(0, E_LOCAL)
    g_start = tl.sum(tl.where(g_idx < group, all_counts, 0))
    m_count = tl.sum(tl.where(g_idx == group, all_counts, 0))

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

    out_ptrs = output_ptr + t[:, None] * N + n_offs[None, :]
    tl.atomic_add(out_ptrs, scaled, mask=m_mask[:, None])


@triton.jit
def _swiglu_quant_kernel(
    G1_ptr,       # [M_pad, 2I] bf16
    C_ptr,        # [M_pad, I] fp8
    Cscale_ptr,   # [nIB, M_pad] fp32
    M_pad_stride,
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
    tl.store(Cscale_ptr + ib * M_pad_stride + m, scale)


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

    local_start = int(local_expert_offset)

    # ─────── Routing (fused Triton kernel; optionally atomic-adds counts in-kernel) ───────
    topk_idx = torch.empty((T, TOP_K), dtype=torch.int32, device=device)
    weights = torch.empty((T, E_global), dtype=torch.float32, device=device)
    counts = _get_cached('counts', (E_local,), torch.int32, device)
    counts.zero_()

    MERGE_COUNT = T <= 256  # for small T, atomic contention is manageable
    _fused_routing_kernel[(T,)](
        routing_logits, routing_bias,
        topk_idx, weights,
        counts, local_start,
        float(routed_scaling_factor),
        E_global, N_GROUP, group_size, TOPK_GROUP, TOP_K,
        E_local, MERGE_COUNT,
        num_warps=2, num_stages=1,
    )

    DISP_BT = 32 if T < 256 else 64
    if not MERGE_COUNT:
        # Large-T path: use separate count kernel to avoid heavy atomic contention in routing.
        _dispatch_count_kernel[(triton.cdiv(T, DISP_BT),)](
            topk_idx, counts, local_start, T,
            E_local, TOP_K, DISP_BT,
            num_warps=1, num_stages=1,
        )
    # Exclusive cumsum is computed inline inside each consumer kernel from `counts`.

    if T <= 256:
        # Skip CPU sync: use upper-bound estimates (consumers mask via the
        # inline cumsum of counts[] so overallocated slots are inert).
        max_count = T  # worst case: all tokens route to one expert
        M_pad = max(((T * TOP_K + 127) // 128) * 128, 128)
    else:
        counts_cpu = counts.to('cpu', non_blocking=False)
        N_total = int(counts_cpu.sum().item())
        max_count = int(counts_cpu.max().item())
        if N_total == 0:
            return torch.zeros((T, H), dtype=torch.bfloat16, device=device)
        M_pad = max(((N_total + 127) // 128) * 128, 128)

    sorted_tokens = _get_cached('sorted_tokens', (M_pad,), torch.int64, device)
    sorted_experts_buf = _get_cached('sorted_experts', (M_pad,), torch.int32, device)
    counter_buf = _get_cached('counter', (E_local,), torch.int32, device)
    sorted_tokens.zero_()
    counter_buf.zero_()
    _dispatch_scatter_kernel[(triton.cdiv(T, DISP_BT),)](
        topk_idx, counts, counter_buf,
        sorted_tokens, sorted_experts_buf,
        T, local_start,
        E_local, TOP_K, DISP_BT,
        num_warps=1, num_stages=1,
    )

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
        sorted_tokens,
        gemm1_weights,
        gemm1_weights_scale,
        G1,
        counts,
        T,
        H, 2 * I,
        H // BLOCK_K_1,
        2 * I * H,
        n1B * nHB,
        E_local,
        BLOCK_M_1, BLOCK_N_1, BLOCK_K_1, NUM_STAGES_1,
        num_warps=NUM_WARPS_1,
    )

    # ─────── Fused SwiGLU + quantize to FP8 ───────
    C_fp8 = torch.empty((M_pad, I), dtype=torch.float8_e4m3fn, device=device)
    C_scale = torch.empty((nIB, M_pad), dtype=torch.float32, device=device)
    _swiglu_quant_kernel[(M_pad, nIB)](
        G1, C_fp8, C_scale,
        M_pad, I, BLOCK,
        num_warps=1, num_stages=1,
    )

    # ─────── Compute routing weight lookup for fused epilogue ───────
    # Use full [M_pad] buffers; dummy slots safely index weights[0, local_start] (unused downstream).
    sorted_global_experts_full = sorted_experts_buf + local_start           # [M_pad] int32
    weight_vec = weights[sorted_tokens, sorted_global_experts_full]         # [M_pad] fp32

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
        counts,
        weight_vec, sorted_tokens,
        output,
        M_pad,
        I, H,
        I // BLOCK_K_2,
        H * I,
        nHB * nIB,
        E_local,
        BLOCK_M_2, BLOCK_N_2, BLOCK_K_2, NUM_STAGES_2,
        num_warps=NUM_WARPS_2,
    )
    return output
