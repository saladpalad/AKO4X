# Variant: triton_swap_grid_evict_lsr
# Source: ako4fib-run-gdn_decode_v1-2/solution/kernel.py (iter-9 final;
#         merged output of concurrent sessions ako4fib-run-gdn_decode_v1
#         and ako4fib-run-gdn_decode_v1-2 which independently converged).
#
# ─── Identity ─────────────────────────────────────────────────────────
# Drift-free A/B vs v0 iter-27 anchor (triton_bv_dispatch_graph):
# **+0.58% mean** (Modal B200, CUDA 13.2, 2026-04-24, 54/54 PASS). Per-B
# A/B Δ:
#   B=1  −1.08%   B=4  −1.67%   B=8  −0.13%   B=16 −0.56%     (noise)
#   B=32 +2.09%   B=48 +3.62%   B=64 +3.05%                   (real gain)
# B=48 crosses 1.0× (0.981 → 1.02) — first large-B bucket beating expert.
# Pure Triton + torch; no TileLang, no CuTe DSL, no manual CUDA.
# Required flags:
#   [benchmark] use_isolated_runner = true   (inherited from v0 anchor —
#     input-ptr-keyed graph cache aliases across workloads in persistent
#     runners when PyTorch recycles addresses).
#   DISABLE_LLVM_OPT=disable-lsr   env var, set via os.environ.setdefault
#     at module-import time inside kernel.py itself.
#
# ─── Delta from prior anchor (triton_bv_dispatch_graph) ──────────────
# Three stacked changes, all discovered in concurrent v1/v1-2 sessions;
# each passed an independent drift-free A/B-compare before landing:
#   - Conditional SWAP_GRID at B≥32. Swap program_id order so pid_v runs
#     outer: 8 CTAs that share the same (b, h) are now issued
#     back-to-back instead of interleaved across (b, h) pairs. Guarded
#     by B≥32 because the unconditional form regresses small B (grid
#     too small to tolerate reordering; v1 iter-6 saw B=1 −8.4%).
#   - Eviction-policy hints stacked on top of SWAP_GRID. q/k tagged
#     `evict_last` (reused across the 8 sibling CTAs); state and
#     new_state tagged `evict_first` (64 KB streaming bytes shouldn't
#     pollute the working set). Hints are a combinatorial optimization:
#     without SWAP_GRID, adjacent CTAs have different (b, h) so nothing
#     is reusable — v1-2 iter-5 tried them alone and saw Δ = −0.52%.
#   - DISABLE_LLVM_OPT=disable-lsr set at module import (measured
#     neutral; kept as zero-cost hedge — see inline comment at the
#     setdefault call for rationale).
# No algorithmic change to the forward math — same register-resident
# single-pass fused kernel as the prior anchor.
#
# ─── Lessons on this variant ─────────────────────────────────────────
# 1. **Conditional grid-dim swap for DRAM row-buffer coherence.**
#    Narrow WHEN: gdn_decode_qk4_v8_d128_k_last at B≥32 only. Grid is
#    (B*HV, V/BV). Swapping makes pid_v the outer (fast-varying) axis,
#    so the 8 CTAs sharing same (b, h) are scheduled back-to-back.
#    Below B=32 the grid is too small to tolerate the swap (v1 iter-6
#    measured mean −1.81% drift-free, driven by B=1 −8.4%).
#    Broad WHEN: any kernel where N independent CTAs load the same
#    small footprint (here: q/k ~1 KB per (b, h_qk)) per group, and
#    the default pid order interleaves groups. Reordering to keep
#    intra-group CTAs adjacent benefits DRAM row-buffer opens. Skip
#    if total grid < few × num_SMs (single-wave regime disrupts more
#    than it helps).
#    WHY: NCU delta post-swap on B≥32: WCPI 14.08 → 13.49 (6% IPC lift),
#    state-load stall cycles 4.3 → 4.1 (−9.3%). L2 hit rate actually
#    *dropped* slightly (2.24% → 1.99%), so the mechanism is DRAM
#    row-buffer coherence across intra-group CTAs, NOT q/k L2 reuse.
#    The initial hypothesis (L2 reuse) was wrong; NCU pinned the real
#    mechanism.
#
# 2. **Eviction-policy hints as finishing touch, only after SWAP_GRID.**
#    Narrow WHEN: on top of #1. Mark q/k as `evict_last` (they ARE
#    reusable across the 8 sibling CTAs under SWAP_GRID); mark state
#    and new_state as `evict_first` (64 KB read + 64 KB write per
#    (b, h) is pure streaming). Unconditional; negligible code cost.
#    Broad WHEN: any kernel where the dominant stream bytes push the
#    cache-relevant payload out of L2, PROVIDED the grid layout
#    actually creates reuse for the "persist" tensor. Without that
#    precondition the hints are a no-op or net-negative.
#    WHY: v1 iter-10 drift-free A/B on top of iter-7 (SWAP_GRID):
#    per-B Δ B=32 +1.0%, B=48 +0.6%, B=64 +0.5% — small but real at
#    large B where state streaming dominates. Mean is noise-neutral
#    (−0.16%) because small-B buckets are graph-cache-dominated and
#    don't benefit. See TRAPS.md for the combinatorial framing.
#
# ─── Dead-ends tried on this variant (expectation priors) ────────────
# Each is re-verifiable; don't trust blindly if your toolchain shifted.
#
# - **V-loop + num_stages=2 to pipeline state load.** v1 iter-1/2,
#   v1-2 iter-2 independently tried this; regressed 10-15% at B≥16.
#   See TRAPS.md "Triton's `num_stages ≥ 2` regresses short-trip
#   register-resident loops" for mechanism (SMEM double-buffer +
#   conservative alias analysis on new_state RMW).
# - **BV=8 unconditional for all B.** v1-2 iter-1, v1 iter-4, v0
#   iter-17 — same regression 8-12% at B≥16. Doubles CTA grid without
#   enough occupancy lift to offset the fixed per-CTA overhead
#   (gates + q/k/v load run twice as often per output element). The
#   `BV=8 at B≤8 / BV=16 otherwise` dispatch is Pareto-optimal.
# - **tl.make_block_ptr for state I/O.** v1 iter-5, neutral. Triton
#   already generates equivalent code from the manual pointer-arith
#   pattern `state_ptr + offs_v[:, None] * K + offs_k[None, :]` when
#   the shape is constexpr. Block_ptr adds 7-8 lines without measurable
#   benefit.
# - **Full-grid SWAP_GRID unconditional (not guarded by B).** v1
#   iter-6, drift-free −1.81%. Large-B wins (+2-3%) destroyed by
#   small-B losses (B=1 −8.4%, B=8 −5.7%). Conditional dispatch in
#   iter-7 kept both halves.
# - **Manual source-reorder: hoist state_tile load before gates.**
#   v1 iter-8 (−3% mean), v1-2 iter-7 (−3-7% per B). Triton's compiler
#   was already scheduling state load optimally (WCPI 13.08); the
#   explicit reorder disrupted register-liveness downstream. Don't
#   manually out-schedule the Triton scheduler.
# - **Pure-swap TMA via triton.tools.tensor_descriptor.**
#   v1-2 iter-6, −4 to −14% across B. Triton 3.6 exposes TMA but
#   lowers HBM→SMEM→REG; direct LDG HBM→REG beats it on the 8 KB
#   state tile. TMA wins only paired with num_stages+loop for
#   pipelined bulk transfer, or with warp specialization.
# - **Parameter sweeps** around {num_warps=2 at B=1, num_warps=8 at
#   B≥32 with BV=32, BV=32 with num_warps=4, tl.max_contiguous /
#   tl.multiple_of hints, object-identity graph-cache fast path, cache
#   modifier `.cs` / `.cg` on state load} — all neutral or regressed.
#   Retry only with new reasoning.
# - **Persistent kernel outer loop.** v2 iter-1 (2026-04-24),
#   regressed 24-37% at B=48 for both N=1 and N=2 (two independent
#   failure modes). See TRAPS.md "Persistent outer-loop breaks
#   SWAP_GRID's DRAM row-buffer coherence" for mechanism. Don't
#   retry while SWAP_GRID is the active grid shape.
# - **`cache_modifier=".cs"` on `tl.store` for new_state write.**
#   v2 iter-5 (2026-04-24). Triton 3.6 rejects at runtime
#   (RUNTIME_ERROR on all 7 B=48 smoke workloads). Triton's tl.store
#   effectively accepts only default (`.wb`) — .cs and other
#   streaming-store modifiers documented elsewhere aren't exposed.
#   Store-side finding complements the prior load-side `.cs`/`.cg`
#   neutral result (already in the Parameter sweeps bullet above).
# - **`num_warps=2` at B=32 with `BV=16`.** v2 iter-3 (2026-04-24),
#   0.924× (vs anchor 0.95× at B=32). Register spill: 51 regs × 64
#   threads forces 32 fp32 state slots per thread, pushing per-thread
#   register count above the 64-reg budget Triton can use without
#   LMEM spill. Prior only tested num_warps=2 at B=1 (neutral); B≥16
#   specifically fails for the same spill reason num_warps=8 failed
#   (upper side of same register-rounding cliff).
#
# ─── Open directions ─────────────────────────────────────────────────
# - **Warp specialization.** Dedicate 1 warp-group to async-copy state
#   into SMEM while the other warp-group reduces + writes. Would amortize
#   the SMEM roundtrip that killed the pure-swap TMA and potentially
#   close the WCPI gap vs CuTe-style kernels. Triton 3.6 has
#   experimental `tl.range(..., num_stages=N)` with warp-specialize
#   knobs in the matmul tutorial — worth adapting here.
# - **TMA + K-loop + num_stages=3.** Split K=128 contraction into two
#   or four chunks, each loaded via `TensorDescriptor.load`, pipelined
#   with num_stages. Amortizes the SMEM roundtrip under compute. Risk:
#   we need state twice (kdot_v/qs reduce AND new_state write), so the
#   loop needs to either hold state in SMEM between phases or accept
#   one extra state read.
# - **CUDA + TMA rewrite.** v0 anchor's open-directions list flagged
#   TMA as the missing piece that would let a CUDA rewrite match
#   Triton's WCPI. TMA availability is no longer a blocker (CUDA 13.2
#   has it); v0 CUDA variants regressed 15% without TMA, a rewrite
#   with TMA + warp-spec producer-consumer + explicit cp.async.bulk
#   is now the remaining path to push past the ~1.13× ceiling Triton
#   seems to impose on this shape.

import math
import os

# DISABLE_LLVM_OPT=disable-lsr: register-bound kernel (51 regs/thread,
# 43% occupancy) with heavy affine state-tile indexing — measured
# neutral here but kept as zero-cost insurance. setdefault lets a
# user-supplied env var override.
os.environ.setdefault("DISABLE_LLVM_OPT", "disable-lsr")

# NO_GRAPH gate: skip CUDA graph capture so NCU sees individual
# cuLaunchKernels (graph replay hides them behind one cuGraphLaunch).
# Leave unset in production; use `run_modal_profile.py --env NO_GRAPH=1`
# when profiling.
_NO_GRAPH = bool(os.environ.get("NO_GRAPH"))

import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_decode_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr,
    A_log_ptr, a_ptr, dt_bias_ptr, b_in_ptr,
    output_ptr, new_state_ptr,
    scale,
    HV: tl.constexpr,
    HQ: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    BV: tl.constexpr,
    HQV_RATIO: tl.constexpr,
    HAS_STATE: tl.constexpr,
    SWAP_GRID: tl.constexpr,
):
    if SWAP_GRID:
        pid_v = tl.program_id(0)
        pid_bh = tl.program_id(1)
    else:
        pid_bh = tl.program_id(0)
        pid_v = tl.program_id(1)
    b = pid_bh // HV
    h = pid_bh % HV
    h_qk = h // HQV_RATIO

    offs_v = pid_v * BV + tl.arange(0, BV)
    offs_k = tl.arange(0, K)

    A_log_h = tl.load(A_log_ptr + h)
    a_h = tl.load(a_ptr + b * HV + h).to(tl.float32)
    dt_bias_h = tl.load(dt_bias_ptr + h)
    b_h = tl.load(b_in_ptr + b * HV + h).to(tl.float32)
    x = a_h + dt_bias_h
    softplus_x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    g = tl.exp(-tl.exp(A_log_h) * softplus_x)
    beta = tl.sigmoid(b_h)

    qk_base = b * (HQ * K) + h_qk * K
    v_base = b * (HV * V) + h * V
    # q/k are reusable across the 8 CTAs that share (b,h) under SWAP_GRID;
    # evict_last hints Triton to keep them L2-resident. The headline win
    # from SWAP_GRID itself is DRAM row-buffer coherence, not L2 reuse
    # (NCU showed L2 hit rate dropped 2.24% → 1.99% post-swap). These
    # eviction hints are a finishing touch, worth +0.5-1% per-B at B≥32.
    q_vec = tl.load(
        q_ptr + qk_base + offs_k, eviction_policy="evict_last",
    ).to(tl.float32)
    k_vec = tl.load(
        k_ptr + qk_base + offs_k, eviction_policy="evict_last",
    ).to(tl.float32)
    v_vec = tl.load(v_ptr + v_base + offs_v).to(tl.float32)

    state_base = b * (HV * V * K) + h * (V * K)
    if HAS_STATE:
        # 64 KB per (b,h) streaming read — evict_first so state bytes
        # don't displace the L2-resident q/k vectors.
        state_tile = tl.load(
            state_ptr + state_base + offs_v[:, None] * K + offs_k[None, :],
            eviction_policy="evict_first",
        )
    else:
        state_tile = tl.zeros([BV, K], dtype=tl.float32)

    kdot_v = tl.sum(state_tile * k_vec[None, :], axis=1)
    qs = tl.sum(state_tile * q_vec[None, :], axis=1)

    old_v = g * kdot_v
    delta_v = beta * (v_vec - old_v)
    qk_dot = tl.sum(q_vec * k_vec)

    output_tile = scale * (g * qs + qk_dot * delta_v)
    new_state_tile = g * state_tile + delta_v[:, None] * k_vec[None, :]

    tl.store(
        new_state_ptr + state_base + offs_v[:, None] * K + offs_k[None, :],
        new_state_tile,
        eviction_policy="evict_first",
    )
    tl.store(
        output_ptr + v_base + offs_v,
        output_tile.to(tl.bfloat16),
    )


# ═══════════════════════════════════════════════════════════════
# CUDA graph cache. The benchmark calls run() many times per
# workload with stable tensor addresses (use_isolated_runner=true);
# PyTorch launch overhead is significant for small-B kernels
# (B=1 kernel body ~3 µs, Python dispatch ~5 µs). Capture once,
# replay on subsequent calls to eliminate host-side overhead.
# Pattern cribbed from reference/dsa-sparse-attention/cute_reduce_v6.
# ═══════════════════════════════════════════════════════════════
_graph_cache = {}
_graph_cnt = {}
_static_out = {}
_static_new_state = {}

_last_key = None
_last_graph = None
_last_out = None
_last_new_state = None


def _launch(q, k, v, state_arg, A_log, a, dt_bias, b_in, output, new_state,
            scale_f, B, BV, has_state):
    # SWAP_GRID at B≥32: pid_v outer puts CTAs sharing same (b,h) back-to-back
    # so DRAM row-buffer stays warm across the 8 intra-group CTAs (NCU
    # measured WCPI 14.08 → 13.49, state-load stall 4.3 → 4.1 cycles).
    # Guarded at B≥32 because the unconditional form regresses small B
    # (v1 iter-6: mean −1.81% driven by B=1 −8.4%); conditional form
    # (iter-7 in v1) gives B=32 +1.9%, B=48 +3.2%, B=64 +2.4% drift-free.
    swap_grid = B >= 32
    grid = (128 // BV, B * 8) if swap_grid else (B * 8, 128 // BV)
    _gdn_decode_kernel[grid](
        q, k, v, state_arg, A_log, a, dt_bias, b_in,
        output, new_state,
        scale_f,
        HV=8, HQ=4, V=128, K=128,
        BV=BV, HQV_RATIO=2,
        HAS_STATE=has_state,
        SWAP_GRID=swap_grid,
        num_warps=4, num_stages=1,
    )


@torch.no_grad()
def run(q, k, v, state, A_log, a, dt_bias, b, scale):
    global _last_key, _last_graph, _last_out, _last_new_state

    B = q.shape[0]
    has_state = state is not None

    # Fast hot-path: same input pointers → replay cached graph.
    state_ptr = state.data_ptr() if has_state else 0
    key = (q.data_ptr(), k.data_ptr(), v.data_ptr(), state_ptr,
           a.data_ptr(), b.data_ptr(), A_log.data_ptr(), dt_bias.data_ptr(), B)

    if key == _last_key and _last_graph is not None:
        _last_graph.replay()
        return _last_out, _last_new_state

    g = _graph_cache.get(key)
    if g is not None:
        _last_key = key
        _last_graph = g
        _last_out = _static_out[B]
        _last_new_state = _static_new_state[B]
        g.replay()
        return _last_out, _last_new_state

    # Allocate static output buffers per B (shape-keyed, reused across calls).
    if B not in _static_out:
        device = q.device
        _static_out[B] = torch.empty(B, 1, 8, 128, dtype=torch.bfloat16, device=device)
        _static_new_state[B] = torch.empty(B, 8, 128, 128, dtype=torch.float32, device=device)
    output = _static_out[B]
    new_state = _static_new_state[B]

    if scale is None or (isinstance(scale, float) and scale == 0.0):
        scale_f = 1.0 / math.sqrt(128)
    else:
        scale_f = float(scale)

    # BV dispatch by batch size (anchor config, no changes this variant):
    #   B≤8 → BV=8 to double the CTA grid for SM fill (at B=1, BV=16 gives
    #         only 64 blocks, under-filling B200's 148 SMs).
    #   B≥16 → BV=16 for stable large-tile throughput (register pressure
    #         manageable at 51 regs/thread, num_warps=4).
    # Always single-tile, single-pass, register-resident — no inner V-loop,
    # no num_stages pipelining (both tried in v1/v1-2 iter-1/2 and
    # regressed 10-15%; see kernel header Dead-ends).
    state_arg = state if has_state else new_state
    BV = 8 if B <= 8 else 16

    cnt = _graph_cnt.get(key, 0) + 1
    _graph_cnt[key] = cnt

    def do_launch():
        _launch(q, k, v, state_arg, A_log, a, dt_bias, b,
                output, new_state, scale_f, B, BV, has_state)

    # Always launch once directly (needed for warmup and Triton JIT).
    do_launch()

    # After warmup, capture a graph so subsequent replays skip Python overhead.
    # NO_GRAPH escape hatch keeps each iter as an eager cuLaunchKernel for NCU.
    if cnt >= 2 and not _NO_GRAPH:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            do_launch()
        _graph_cache[key] = graph
        _last_key = key
        _last_graph = graph
        _last_out = output
        _last_new_state = new_state

    return output, new_state
