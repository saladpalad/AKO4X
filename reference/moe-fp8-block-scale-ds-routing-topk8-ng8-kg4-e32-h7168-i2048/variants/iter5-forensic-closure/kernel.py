"""
iter5-forensic-closure — round-5 forensic-closure variant for
moe-fp8-block-scale-ds-routing-topk8-ng8-kg4-e32-h7168-i2048 (B200 / Modal).

═══════════════════════════════════════════════════════════════════════════
1) Identity
─────────────────────────────────────────────────────────────────────────
Score: 1.33x mean (functionally identical runtime to parent `iter4-
autotune-tactic-sweep-gated`; expected within bench noise on re-run).
Closing evidence: **two within-trtllm levers (PDL=False, shuffled-weight)
empirically refuted with concrete failure modes** (see sections 3 + 4).
Lever class for within-trtllm space: closed at the vendor-cubin ceiling.
The natural successor campaign is a hand-rolled GEMM1+SwiGLU+GEMM2 chain
(CUTLASS / CuTe-DSL / DeepGEMM) — explicitly out of round-5 scope.

Runtime: unchanged from iter4 anchor. Captured-graph wrap of the trtllm
6-kernel pipeline (`torch.cuda.graph()` of `moe_op.trtllm_fp8_block_scale_
moe`); `_GRAPH_MIN_T=2` T=1 eager fallback (preserves matched_ratio gate);
per-shape `[tile_N, config]` tactic sweep gated to T≥1000 via
`_AUTOTUNE_MIN_T=1000`; `data_ptr()` trial-boundary detection → recapture
+ cached-tactic reuse. The contribution of this variant is **forensic
closure documentation in sections 3-5**, not a perf lift.

═══════════════════════════════════════════════════════════════════════════
2) Delta from prior anchor
─────────────────────────────────────────────────────────────────────────
Parent: `iter4-autotune-tactic-sweep-gated` (1.33x mean, 19/19 PASS,
range 1.070x-1.750x).

Code delta: header only. All function bodies and constants
(`_USE_SHUFFLED_WEIGHT=False`, `_GRAPH_MIN_T=2`, `_AUTOTUNE_MIN_T=1000`,
PDL via `device_support_pdl`) preserved verbatim from iter4. Same
correctness contract: `do_finalize=True`, in-place write to pre-
allocated `(T, H, device)` output buffer cache.

═══════════════════════════════════════════════════════════════════════════
3) Lessons on this variant
─────────────────────────────────────────────────────────────────────────
Inherits iter4's lessons (load-bearing autotune gate, tactic-schema
correction, `_FALLBACK_TACTIC` inclusion, captured-graph composition).
New round-5 lessons:

- **PDL=True is correctness-load-bearing at T=1, not just a perf
  optimization.** iter5a probe set `device_support_pdl()→False` and
  re-ran the full bench A/B vs iter4 (same Modal container). Workload
  e05c6c03 (T=1, eager-fallback path — no graph capture involved)
  returned `INCORRECT_NUMERICAL` while A-side iter4 ran at 1.2625 mean.
  All other binding params identical between A and B sides; only
  `enable_pdl` differed. Mechanism (most likely): the cubin's 6-kernel
  pipeline (routing → permute → GEMM1 → SwiGLU → GEMM2 → combine) has
  PDL-aware barriers; with PDL=False the serialized fallback schedule
  shifts FP8 rounding boundaries at T=1, pushing the borderline
  matched_ratio (already 0.92 at T=1 per iter3c capture-mode note)
  below the 0.9 gate. T=1 is uniquely sensitive because per-token
  output concentration is highest (1 token × 8 routes / 32
  local-experts) — every output element is the single hot output for
  some routing branch, no smoothing buffer. **Implication:** future
  iters must NOT default PDL off as a "safe" choice; PDL is wired into
  matched_ratio survivability.

- **`use_shuffled_weight=True` is incompatible with DeepSeekFp8
  block-scale via the public binding.** iter5b probes attempted the
  shuffled-weight fast path two ways, both refuted:
    * **v1 (weights-only shuffle):** applied
      `reorder_rows_for_gated_act_gemm` per-expert on `gemm1_weights`
      ([E, M, K] → interleave gate/up rows: out[e,2i,:]=in[e,i,:],
      out[e,2i+1,:]=in[e,M/2+i,:]). Scales unchanged. Smoke test on T=7
      returned `INCORRECT_NUMERICAL abs_err=1.02e+06, rel_err=5.44e+05`
      — garbage output, not borderline. The cubin's `Bmm_E4m3_E4m3E4m3_
      Fp32_t128x8x128u2_s8_et64x8...rM_TN_transOut_dsFp8_schPd4x2x2x3...`
      shuffled-fast-path variant reads weights at shuffled row positions
      but is reading per-block scales at the ORIGINAL (non-shuffled)
      block layout, so the MMA pairs (shuffled-weight-block,
      original-scale-block) are mismatched.
    * **v2 (weights + block-level scale shuffle):** in addition to v1,
      applied the same interleave at block granularity on
      `gemm1_weights_scale` ([E, M/128, K/128] → out_s[e,2j,:]=in_s[e,j,
      :], out_s[e,2j+1,:]=in_s[e,M_blk/2+j,:]). Smoke test on T=7
      returned `TIMEOUT` (900s — kernel hung). The block-level scale
      shuffle is mathematically WRONG anyway (within an output block of
      128 rows after shuffle, half the rows come from input top-half
      and half from input bottom-half; per-block scale storage can't
      distinguish them), but the timeout suggests it also corrupts the
      cubin's tile-scheduling state.
  **Why this lever is closed for our exact config:** FlashInfer's own
  reference test for `ng8_kg4_e32_h7168_i2048` uses
  `use_shuffled_weight=False` (per round-5 sub-agent's reading of the
  test). The vendor wires the shuffled fast path only for `MajorMn` /
  `BlockMajorK` weight layouts (line 1444 in
  `trtllm_fused_moe_kernel_launcher.cu` pairs `useShuffledMatrix=true`
  with `MatrixLayout::BlockMajorK`). With `MajorK` (our layout) +
  block-scale, the Python binding's `use_shuffled_weight=True` exposes
  an undocumented / unsupported combination. Pursuing it further
  requires either reformatting weights to `BlockMajorK` (a much larger
  data layout change than a row shuffle) or kernel-level scale-handling
  reverse engineering — both out of scope.

═══════════════════════════════════════════════════════════════════════════
4) Dead-ends tried (this iter)
─────────────────────────────────────────────────────────────────────────
- **iter5a — `enable_pdl=False` on top of iter4.** Forensically refuted
  by T=1 `INCORRECT_NUMERICAL` (see section 3). Lever closed: PDL is
  correctness-load-bearing at extreme small-T, not just a perf knob.
  Upper bound on lift across the 19-workload mean is bounded ≤0
  because the T=1 gate cannot be bypassed.

- **iter5b-v1 — `use_shuffled_weight=True` with per-expert
  `reorder_rows_for_gated_act_gemm` on gemm1 weights only.** Refuted by
  `abs_err=1.02e+06` on T=7 smoke (see section 3). The DeepSeekFp8
  block-scale shuffled cubin does NOT auto-handle scale-to-shuffled-row
  alignment internally.

- **iter5b-v2 — same + block-level interleave of `gemm1_weights_scale`
  at M/128 granularity.** Refuted by `TIMEOUT` on T=7 smoke (see
  section 3). Block-level scale shuffle is mathematically wrong for
  this row-level weight shuffle (each post-shuffle 128-row block mixes
  scales from two non-adjacent input blocks); also produced kernel hang
  rather than just incorrect output.

- **PDL=True + T=1 graph-capture revisit** — NOT attempted in round-5.
  Already refuted in iter3c (capture-mode FP8 perturbation pushes the
  borderline T=1 matched_ratio below the 0.9 gate; eager fallback is
  the load-bearing fix). The headroom even if survivable is ≤0.5x ×
  1/19 ≈ +0.026x on mean — below the round-4 variance floor of 1.5%.

Inherits iter4's dead-ends (none on the autotune lever itself beyond
the T<1000 gate rationale). Inherits iter3c's dead-end on T=1 graph
capture.

═══════════════════════════════════════════════════════════════════════════
5) Open directions (forensic-closure outcome)
─────────────────────────────────────────────────────────────────────────
**Round-5 outcome: within-trtllm lever space is empirically closed.**
The 3 within-trtllm levers identified at round-4 close as follows:
  * Shuffled-weight: refuted (iter5b v1+v2 — see section 3-4).
  * PDL=False: refuted (iter5a — T=1 correctness).
  * T=1 graph-capture revisit: already refuted in iter3c; sub-baseline
    even if survivable.

The only remaining direction with potential to exceed trtllm's tactic-
catalog ceiling is the **whole-DSL alternative** (CUTLASS /
CuTe-DSL grouped-GEMM / DeepGEMM hand-rolled chain for the (E=32,
tile_N=4096, H=7168, I=2048) shape). Sub-4 explicitly identified this
as a many-iter commitment, NOT a round-5 lever. Natural framing for the
next master:
  * Replace `moe_op.trtllm_fp8_block_scale_moe` with a custom GEMM1 +
    SwiGLU + GEMM2 chain, keeping the trtllm routing kernel.
  * Target the two largest workloads (T=11948, T=14107) where vendor
    heuristic clamps to tile_N=4096 and round-4's tactic sweep already
    extracts ~+0.05-0.07x — implies a non-trivial cubin-catalog gap.
  * Use captured-graph as proven framework; keep eager T=1 fallback.

Pathological hazards documented for the successor (so they don't burn
iter budget rediscovering):
  * **PDL has correctness implications at T=1** — keep `enable_pdl=True`
    in any new kernel chain (not just for perf).
  * **`use_shuffled_weight=True` with public block-scale binding is
    unsupported** — if the custom GEMM1 internally uses a CuTe shuffled
    layout, the layout transformation must be done by the new code, not
    delegated to the trtllm binding.
  * **Trial-boundary `data_ptr()` detection** is necessary because
    bench-harness inputs change between trials but not within a trial.
    Any cached re-formatting of weights (shuffle, transpose, etc.) must
    invalidate on data_ptr mismatch and re-do in the untimed prelude.

═══════════════════════════════════════════════════════════════════════════
"""

import os

import torch

from flashinfer.fused_moe.core import (
    DtypeTrtllmGen,
    get_trtllm_moe_sm100_module,
)
from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module
from flashinfer.tllm_enums import (
    ActivationType,
    Fp8QuantizationType,
    RoutingMethodType,
    WeightLayout,
)
from flashinfer.utils import device_support_pdl


# Operator-fixed constants (per docs/definition.json / operator name).
_TOP_K = 8
_N_GROUP = 8
_TOPK_GROUP = 4
_NUM_EXPERTS = 256
_INTERMEDIATE_SIZE = 2048
_HIDDEN_SIZE = 7168

# Vendor-default fallback: tile_N=-1 → selectDefaultTileN; config=-1 → default cfg.
_FALLBACK_TACTIC = [-1, -1]

# Hard-coded enum / flag values (per parent anchor's config).
_ROUTING_METHOD = RoutingMethodType.DeepSeekV3.value
_WEIGHT_LAYOUT = WeightLayout.MajorK.value
_FP8_QUANT = Fp8QuantizationType.DeepSeekFp8
_ACTIVATION = ActivationType.Swiglu.value
_USE_SHUFFLED_WEIGHT = False
_DO_FINALIZE = True
_NORM_TOPK_PROB = True

# Dtype enums (FP8 E4m3 for activation + weights, matching the operator spec).
_DTYPE_ACT = DtypeTrtllmGen.E4m3
_DTYPE_WEIGHTS = DtypeTrtllmGen.E4m3

# Env-gated escape hatches.
_NO_GRAPH = bool(int(os.environ.get("NO_GRAPH", "0")))
_VALIDATE_CAPTURE = bool(int(os.environ.get("VALIDATE_CAPTURE", "0")))
_NO_AUTOTUNE = bool(int(os.environ.get("NO_AUTOTUNE", "0")))
# If T < _GRAPH_MIN_T, skip capture and run eager. Default 2 (skip T=1 only).
# See parent kernel's section-3 rationale for the matched_ratio tolerance
# interaction with capture-mode FP8 perturbations at T=1.
_GRAPH_MIN_T = int(os.environ.get("GRAPH_MIN_T", "2"))

# Per-tactic timing budget. 3 iters × N tactics is the autotune cost per
# shape (subprocess). Lower = noisier; higher = more prelude time but
# untimed so it doesn't affect CUPTI latency.
_AUTOTUNE_ITERS = int(os.environ.get("AUTOTUNE_ITERS", "3"))
_AUTOTUNE_WARMUP = int(os.environ.get("AUTOTUNE_WARMUP", "1"))

# Autotune only runs for T >= _AUTOTUNE_MIN_T. Below that, the vendor
# heuristic [-1, -1] is the proven winner. Rationale (round-4 ab-compare
# evidence): drift-cancelled per-workload delta vs iter3c showed
# +0.047x at T=11948, +0.068x at T=14107, ≈0 for T=1..901, and a severe
# -0.265x regression at T=1 (timing-noise floor at sub-100µs per call
# breaks the median-of-3 tactic ranking — fallback gets de-ranked). Gate
# at T>=1000 captures the two real wins and preserves vendor-default
# behavior everywhere else.
_AUTOTUNE_MIN_T = int(os.environ.get("AUTOTUNE_MIN_T", "1000"))

# Shared graph mempool — all captures within a process share the same
# private pool so re-captures don't fragment the allocator.
_GRAPH_POOL = None

# Lazy-initialized module + per-process caches.
_MOE_OP = None
_PDL_BY_DEVICE: dict = {}
_EMPTY_BY_DTYPE_DEVICE: dict = {}
_EMPTY_INT32_BY_DEVICE: dict = {}
_OUTPUT_BY_SHAPE_DEVICE: dict = {}

# Graph state — keyed by (T, device.index).
_GRAPH_BY_KEY: dict = {}

# Tactic cache — keyed by (T, hidden_size, intermediate_size,
# local_num_experts, device.index). Value is a 2-list [tile_N, config].
_TACTIC_BY_KEY: dict = {}


def _init_moe_op():
    """Force module build, cubin-loader setup, and grab the raw C++ binding."""
    global _MOE_OP
    if _MOE_OP is not None:
        return _MOE_OP
    _ = get_trtllm_moe_sm100_module()
    module = gen_trtllm_gen_fused_moe_sm100_module()
    _MOE_OP = module.build_and_load()
    return _MOE_OP


def _get_output_buffer(num_tokens: int, hidden_size: int, device: torch.device) -> torch.Tensor:
    key = (num_tokens, hidden_size, device.index)
    buf = _OUTPUT_BY_SHAPE_DEVICE.get(key)
    if buf is None:
        buf = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
        _OUTPUT_BY_SHAPE_DEVICE[key] = buf
    return buf


def _get_empty(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    key = (dtype, device.index)
    t = _EMPTY_BY_DTYPE_DEVICE.get(key)
    if t is None:
        t = torch.empty(0, dtype=dtype, device=device)
        _EMPTY_BY_DTYPE_DEVICE[key] = t
    return t


def _get_empty_int32(device: torch.device) -> torch.Tensor:
    key = device.index
    t = _EMPTY_INT32_BY_DEVICE.get(key)
    if t is None:
        t = torch.empty(0, dtype=torch.int32, device=device)
        _EMPTY_INT32_BY_DEVICE[key] = t
    return t


def _get_pdl(device: torch.device) -> bool:
    key = device.index
    p = _PDL_BY_DEVICE.get(key)
    if p is None:
        p = device_support_pdl(device)
        _PDL_BY_DEVICE[key] = p
    return p


def _eager_call(
    moe_op,
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    output,
    topk_ids,
    expert_weights,
    enable_pdl,
    local_expert_offset,
    local_num_experts,
    routed_scaling_factor,
    tactic,
):
    """The single binding invocation — used in eager, capture, and autotune paths."""
    moe_op.trtllm_fp8_block_scale_moe(
        routing_logits,
        topk_ids,
        expert_weights,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        output,
        _NUM_EXPERTS,
        _TOP_K,
        _N_GROUP,
        _TOPK_GROUP,
        _INTERMEDIATE_SIZE,
        int(local_expert_offset),
        local_num_experts,
        float(routed_scaling_factor),
        _ROUTING_METHOD,
        _USE_SHUFFLED_WEIGHT,
        _WEIGHT_LAYOUT,
        _DO_FINALIZE,
        enable_pdl,
        tactic,
        _FP8_QUANT,
        _ACTIVATION,
        _NORM_TOPK_PROB,
        None,  # routing_replay_out
    )


def _autotune_tactic(
    moe_op,
    num_tokens,
    local_num_experts,
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    output,
    topk_ids,
    expert_weights,
    enable_pdl,
    local_expert_offset,
    routed_scaling_factor,
):
    """Sweep all valid [tile_N, config] tactics; return the fastest.

    Returns _FALLBACK_TACTIC on any enumeration/timing failure.
    """
    # Enumerate candidates. Signature matches MoERunner.get_valid_tactics
    # at flashinfer/fused_moe/core.py:1005-1027.
    try:
        candidates = moe_op.trtllm_get_valid_moe_configs(
            _DTYPE_ACT,
            _DTYPE_WEIGHTS,
            _FP8_QUANT,
            _TOP_K,
            _HIDDEN_SIZE,
            _INTERMEDIATE_SIZE,
            local_num_experts,
            _ACTIVATION,
            _USE_SHUFFLED_WEIGHT,
            _WEIGHT_LAYOUT,
            num_tokens,
        )
    except Exception:
        return _FALLBACK_TACTIC

    # Normalize cubin's Array<Array<int64_t>> to Python list-of-2-lists.
    explicit = []
    for c in candidates:
        try:
            tac = [int(c[0]), int(c[1])]
            explicit.append(tac)
        except Exception:
            continue

    # Always include the vendor fallback in the sweep.
    sweep = explicit + [list(_FALLBACK_TACTIC)]

    # Initial warmup so first-tactic measurement is not contaminated by
    # cubin cold-load / cudaMalloc-for-workspace.
    for _ in range(_AUTOTUNE_WARMUP):
        try:
            _eager_call(
                moe_op, routing_logits, routing_bias, hidden_states,
                hidden_states_scale, gemm1_weights, gemm1_weights_scale,
                gemm2_weights, gemm2_weights_scale, output, topk_ids,
                expert_weights, enable_pdl, local_expert_offset,
                local_num_experts, routed_scaling_factor,
                list(_FALLBACK_TACTIC),
            )
        except Exception:
            pass
    torch.cuda.synchronize()

    best_tactic = list(_FALLBACK_TACTIC)
    best_ms = float("inf")

    start_evt = [torch.cuda.Event(enable_timing=True) for _ in range(_AUTOTUNE_ITERS)]
    end_evt = [torch.cuda.Event(enable_timing=True) for _ in range(_AUTOTUNE_ITERS)]

    for tactic in sweep:
        try:
            # Per-tactic warmup (1 call) — handles tile-switch workspace
            # alloc & cubin variant load.
            _eager_call(
                moe_op, routing_logits, routing_bias, hidden_states,
                hidden_states_scale, gemm1_weights, gemm1_weights_scale,
                gemm2_weights, gemm2_weights_scale, output, topk_ids,
                expert_weights, enable_pdl, local_expert_offset,
                local_num_experts, routed_scaling_factor, tactic,
            )
            for i in range(_AUTOTUNE_ITERS):
                start_evt[i].record()
                _eager_call(
                    moe_op, routing_logits, routing_bias, hidden_states,
                    hidden_states_scale, gemm1_weights, gemm1_weights_scale,
                    gemm2_weights, gemm2_weights_scale, output, topk_ids,
                    expert_weights, enable_pdl, local_expert_offset,
                    local_num_experts, routed_scaling_factor, tactic,
                )
                end_evt[i].record()
            torch.cuda.synchronize()
            elapsed = sorted(start_evt[i].elapsed_time(end_evt[i]) for i in range(_AUTOTUNE_ITERS))
            median = elapsed[_AUTOTUNE_ITERS // 2]
            if median < best_ms:
                best_ms = median
                best_tactic = tactic
        except Exception:
            # Tactic invalid for this shape / cubin path; skip.
            continue

    return best_tactic


def _capture_graph(
    moe_op,
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    output,
    topk_ids,
    expert_weights,
    enable_pdl,
    local_expert_offset,
    local_num_experts,
    routed_scaling_factor,
    tactic,
):
    """Capture the binding call into a CUDA graph and return it.

    Three warmup calls on a side stream precede capture (the canonical
    torch.cuda.graph pattern); these settle the binding's one-time state
    and let PyTorch's allocator's first-time cudaMalloc paths fire
    outside the capture window.
    """
    global _GRAPH_POOL

    torch.cuda.synchronize()
    s = torch.cuda.Stream(device=output.device)
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _eager_call(
                moe_op, routing_logits, routing_bias, hidden_states,
                hidden_states_scale, gemm1_weights, gemm1_weights_scale,
                gemm2_weights, gemm2_weights_scale, output, topk_ids,
                expert_weights, enable_pdl, local_expert_offset,
                local_num_experts, routed_scaling_factor, tactic,
            )
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    if _GRAPH_POOL is None:
        _GRAPH_POOL = torch.cuda.graph_pool_handle()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=_GRAPH_POOL):
        _eager_call(
            moe_op, routing_logits, routing_bias, hidden_states,
            hidden_states_scale, gemm1_weights, gemm1_weights_scale,
            gemm2_weights, gemm2_weights_scale, output, topk_ids,
            expert_weights, enable_pdl, local_expert_offset,
            local_num_experts, routed_scaling_factor, tactic,
        )
    return g


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
    moe_op = _MOE_OP if _MOE_OP is not None else _init_moe_op()

    device = hidden_states.device
    num_tokens = hidden_states.shape[0]
    output = _get_output_buffer(num_tokens, _HIDDEN_SIZE, device)
    topk_ids = _get_empty_int32(device)
    expert_weights = _get_empty(routing_logits.dtype, device)
    enable_pdl = _get_pdl(device)
    local_num_experts = gemm1_weights.shape[0]

    # Resolve per-shape tactic (autotune once per subprocess; cache hit
    # on subsequent calls within the same shape).
    tactic_key = (num_tokens, _HIDDEN_SIZE, _INTERMEDIATE_SIZE, local_num_experts, device.index)
    tactic = _TACTIC_BY_KEY.get(tactic_key)
    if tactic is None:
        if _NO_AUTOTUNE or num_tokens < _AUTOTUNE_MIN_T:
            tactic = list(_FALLBACK_TACTIC)
        else:
            tactic = _autotune_tactic(
                moe_op, num_tokens, local_num_experts,
                routing_logits, routing_bias, hidden_states,
                hidden_states_scale, gemm1_weights, gemm1_weights_scale,
                gemm2_weights, gemm2_weights_scale, output, topk_ids,
                expert_weights, enable_pdl, local_expert_offset,
                routed_scaling_factor,
            )
        _TACTIC_BY_KEY[tactic_key] = tactic

    if _NO_GRAPH or num_tokens < _GRAPH_MIN_T:
        _eager_call(
            moe_op, routing_logits, routing_bias, hidden_states,
            hidden_states_scale, gemm1_weights, gemm1_weights_scale,
            gemm2_weights, gemm2_weights_scale, output, topk_ids,
            expert_weights, enable_pdl, local_expert_offset,
            local_num_experts, routed_scaling_factor, tactic,
        )
        return output

    key = (num_tokens, device.index)
    entry = _GRAPH_BY_KEY.get(key)
    hs_ptr = hidden_states.data_ptr()

    if entry is None or entry["hs_ptr"] != hs_ptr:
        if entry is not None:
            del _GRAPH_BY_KEY[key]
            entry = None
        g = _capture_graph(
            moe_op, routing_logits, routing_bias, hidden_states,
            hidden_states_scale, gemm1_weights, gemm1_weights_scale,
            gemm2_weights, gemm2_weights_scale, output, topk_ids,
            expert_weights, enable_pdl, local_expert_offset,
            local_num_experts, routed_scaling_factor, tactic,
        )
        entry = {"graph": g, "hs_ptr": hs_ptr, "output": output}
        _GRAPH_BY_KEY[key] = entry

    if _VALIDATE_CAPTURE:
        output.zero_()

    entry["graph"].replay()

    if _VALIDATE_CAPTURE:
        torch.cuda.synchronize()
        assert output.abs().sum().item() > 0, (
            f"VALIDATE_CAPTURE: output is all-zero after replay at T={num_tokens}"
        )

    return output
