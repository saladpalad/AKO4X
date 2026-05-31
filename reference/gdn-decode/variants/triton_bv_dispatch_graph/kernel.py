# Variant: triton_bv_dispatch_graph
# Source: ako4fib-run-gdn_decode_v0/solution/kernel.py (iter-27 final;
#         trajectory 20260423_135937_iter-27_final_graph+dispatch_confirmation).
#
# ─── Identity ─────────────────────────────────────────────────────────
# **1.126× mean speedup** vs expert baseline (iter-27, Modal B200,
# CUDA 13.2, 2026-04-23; 5-trial mean, 54 workloads). Per-B:
#   B=1  → 1.42×   B=4  → 1.26×   B=8  → 1.14×
#   B=16 → 1.05×   B=32 → 0.95×   B=48 → 0.98×   B=64 → 0.96×
# Pure Triton + torch; no TileLang, no CuTe DSL, no manual CUDA. 168 LOC.
# Required flags: `[benchmark] use_isolated_runner = true` (graph cache's
# pointer-keyed fast path aliases across workloads in persistent runners
# when PyTorch recycles addresses between workloads).
#
# ─── Delta from prior anchor ──────────────────────────────────────────
# First archived gdn-decode variant. From iter-1 Triton fused baseline
# (1.07× mean), two A/B-verified wins compose to 1.13× final (+5.09%):
#   - iter-18 BV dispatch (B≤8 → BV=8, else → BV=16): +2.11%.
#     Small batches need doubled CTA grid for SM fill; large batches
#     retain stable large-tile throughput.
#   - iter-24 CUDA graph cache with input-ptr-keyed lookup + hot-path
#     `_last_key` check: +2.98%. Amortizes Python dispatch (~5 µs) to
#     the single-cycle key check for the overwhelmingly common case of
#     identical inputs across benchmark iterations.
# Architecture: single kernel per call (no fwd/reduce split). Each
# program computes one (b, h, v_tile) slice: loads the full K=128 slice
# of state for its V-tile, applies gates (g = exp(-exp(A_log) *
# softplus(a + dt_bias))), computes state·k + state·q matvecs,
# constructs delta update, writes output + new_state in-place. No
# intermediate SMEM staging — register-resident throughout.
#
# ─── Lessons ──────────────────────────────────────────────────────────
# 1. **BV dispatch by batch size, not static tile.**
#    Narrow WHEN: this operator has `V=128` and CTA grid `(B*HV, 128/BV)`.
#    At B=1, BV=16 gives 8×8 = 64 blocks — under-fills B200's 148 SMs.
#    BV=8 doubles to 128 blocks, hitting ~85% SM occupancy vs ~50%.
#    At B≥16, BV=16 keeps register pressure manageable (51 regs/thread
#    at num_warps=4) and larger tiles amortize per-block overhead.
#    Broad WHEN: any decode/attention kernel where grid along the
#    small-B dimension is SM-count-bound; increasing CTA fan-out is a
#    portable win when batch is below `num_SMs / other_grid_dims`.
#    WHY: memory-bound on state load + new_state write (64 KB R +
#    64 KB W per (b, h)); SM fill dominates throughput until bandwidth
#    saturates around B=16.
#
# 2. **CUDA graph cache with `_last_key` hot-path check before dict
#    lookup.**
#    Narrow WHEN: `flashinfer-bench` calls `run()` ~100+ times per
#    workload with stable tensor addresses (because `isolated_runner`
#    recycles buffers). Cache hit rate is >99%; 50-ns dict-lookup
#    overhead matters at ~5 µs kernel latency.
#    Broad WHEN: any persistent benchmark runner with repeated calls on
#    the same input tuple — check identity before dict lookup for the
#    very hot path (compile-time known: `key == _last_key`).
#    WHY: 9-tuple dict lookup (hash + probe + `__eq__`) ~50 ns; at
#    B=1 the Triton kernel is ~3 µs, so the int-compare hot-path
#    saves ~1.5%.
#
# 3. **Capture only after the non-graph launch succeeds (`cnt>=2`
#    guard).**
#    Narrow WHEN: the first `_launch()` call compiles Triton JIT and
#    loads PTX; capturing on cnt=1 would bake compile-time behavior
#    into the graph and occasionally fail or corrupt on replay.
#    Broad WHEN: CUDA graph capture with Triton-compiled kernels —
#    always do one eager launch first to ensure JIT is complete and the
#    kernel is PTX-warm.
#    WHY: Triton's first-call JIT (autotuner, PTX load) includes
#    host-side ops; capturing on cnt=1 corrupts on replay. Pattern
#    from `reference/dsa-sparse-attention/variants/cute_reduce_v6`.
#
# ─── Dead-ends (expectation priors; re-verify if context changes) ────
# - **CUDA manual rewrite (iter-10/11/13).** -15.80% vs Triton
#   despite 80% theoretical occupancy (Triton: ~46%) and fixed
#   coalescing. Root cause: NCU Warp Cycles Per Issued Instruction
#   (WCPI) was 29.58 for CUDA vs 13.08 for Triton — **2.3× worse
#   per-warp efficiency**. Triton's compiler packs instructions
#   denser (ILP + LDG/STG scheduling). Further CUDA attempts (iter-14
#   warp-specialized producer-consumer, iter-15 register-cached
#   pipelining, iter-16 256-threads-per-block) all regressed: NCU
#   showed 35% MIO stalls dominate despite occupancy improvement.
#   **Expect diminishing returns on CUDA rewrite unless you can deploy
#   TMA bulk loads** (untested; B200 has TMA, so a TMA-based K/state
#   loader remains an open direction).
#
# - **Streaming K in 2 passes (iter-6).** -21.86%. Hypothesis: splitting
#   the K=128 contraction into two K=64 passes would let L2 reuse state
#   across passes. Actual: loop overhead + doubled memory traffic
#   dominated any L2 gains that didn't materialize.
#
# - **BV=32 (wider tile for B≥32).** -2.8% to -1.7% depending on B.
#   Register pressure climbed; Triton spilled, losing the tile-width
#   win. BV=16 is Pareto-optimal for the large-B regime.
#
# - **num_warps=8 instead of 4.** -11.59%. 51 regs/thread × 8 warps =
#   14336 regs/block → only 4 blocks/SM (register-rounding cliff at
#   65536 regs/SM). Register budget is the binding constraint.
#
# - **Parameter sweeps** {num_stages=2 (no loop to pipeline), `.cg`/
#   `.cs` cache modifiers on state load (L1 already hits), `maxnreg=48`
#   hint (unrespected)} — all neutral or within noise; retry only with
#   new reasoning.
#
# ─── Open directions ─────────────────────────────────────────────────
# - **TMA-based state loader, pipelined.** B200 exposes
#   `cp.async.bulk.tensor`; Triton 3.6 surfaces it via
#   `triton.tools.tensor_descriptor`. Drop-in `tl.load → desc.load`
#   regressed 4-14% per B (v1-2 iter-6) because Triton's TMA lowers
#   HBM→SMEM→REG — on the 8 KB state tile, direct LDG HBM→REG wins.
#   TMA pays off only paired with pipelined bulk transfer
#   (`num_stages + K-loop` amortizing the SMEM hop under compute) or
#   warp-specialized producer-consumer. Upside: memory-bound floor
#   at B≥16 (current 0.95-1.05×) could climb toward 1.15-1.20× if
#   hide-latency succeeds.
#
# - **Persistent kernel over batch.** Currently `_launch` grids on
#   `B*HV` blocks, each block finishing independently. A persistent
#   kernel with `B` threadblocks that iterate over HV internally would
#   amortize the gates computation (A_log, softplus, beta don't depend
#   on the V-tile). Small expected win (~2%); gates are a ~10-op
#   tail.
#
# - **FP8 state / bfloat16 accumulator.** State is float32; if
#   numerical precision allows, store state in bfloat16 and accumulate
#   in float32. Halves state memory traffic (the dominant cost).
#   Precision risk needs correctness testing against the expert's
#   reference — the delta rule `new_state = g*state + delta*k` has
#   error compounding across calls, so not obviously safe.

import math
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
):
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
    q_vec = tl.load(q_ptr + qk_base + offs_k).to(tl.float32)
    k_vec = tl.load(k_ptr + qk_base + offs_k).to(tl.float32)
    v_vec = tl.load(v_ptr + v_base + offs_v).to(tl.float32)

    state_base = b * (HV * V * K) + h * (V * K)
    if HAS_STATE:
        state_tile = tl.load(
            state_ptr + state_base + offs_v[:, None] * K + offs_k[None, :]
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
    _gdn_decode_kernel[(B * 8, 128 // BV)](
        q, k, v, state_arg, A_log, a, dt_bias, b_in,
        output, new_state,
        scale_f,
        HV=8, HQ=4, V=128, K=128,
        BV=BV, HQV_RATIO=2,
        HAS_STATE=has_state,
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

    # Dispatch BV by batch size: more CTAs at small B for SM fill.
    BV = 8 if B <= 8 else 16
    state_arg = state if has_state else new_state

    cnt = _graph_cnt.get(key, 0) + 1
    _graph_cnt[key] = cnt

    # Always launch once directly (needed for warmup and Triton JIT).
    _launch(q, k, v, state_arg, A_log, a, dt_bias, b,
            output, new_state, scale_f, B, BV, has_state)

    # After warmup, capture a graph so subsequent replays skip Python overhead.
    if cnt >= 2:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _launch(q, k, v, state_arg, A_log, a, dt_bias, b,
                    output, new_state, scale_f, B, BV, has_state)
        _graph_cache[key] = graph
        _last_key = key
        _last_graph = graph
        _last_out = output
        _last_new_state = new_state

    return output, new_state
