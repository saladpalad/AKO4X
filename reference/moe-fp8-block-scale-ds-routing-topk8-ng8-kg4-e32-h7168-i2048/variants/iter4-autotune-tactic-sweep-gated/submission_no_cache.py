"""
submission_no_cache — SOL-compat derivative of iter4-autotune-tactic-sweep-gated.

═══════════════════════════════════════════════════════════════════════════
0) SOL-compat preface (this file's reason for existing)
─────────────────────────────────────────────────────────────────────────
This file is a stripped-down sibling of `kernel.py` (iter4 anchor) intended
for callers OTHER than the FIB benchmark — production LLM serving, the
SOL-ExecBench online judge, persistent-buffer cheat-checks.

The iter4 anchor relies on the FIB harness's allocation contract:
"each workload runs in its own subprocess, all 8 input tensors are
fresh-allocated per workload, 100 iters share input data_ptrs within
a trial". Under that contract, `_GRAPH_BY_KEY[(num_tokens, dev)]`
keyed with `hidden_states.data_ptr()` re-validation correctly detects
trial-boundary tensor re-allocation. The +0.11x (graph capture) and
+0.85% drift-cancelled (tactic autotune) lifts are real WITHIN that
contract.

Under any OTHER caller — buffer-pool reuse, persistent activation
caches, indptr re-use across workloads with different content — the
captured graph embeds pointers for `routing_logits`, `routing_bias`,
`gemm1_weights*`, `gemm2_weights*` from the prior workload, only the
primary `hidden_states.data_ptr()` is re-checked. Result: stale-pointer
replay → INCORRECT_NUMERICAL or segfault. See
`reference/.../TRAPS.md` "`data_ptr()`-keyed CUDA graph cache is
FIB-contract-specific" for full forensic.

**What this file changes vs iter4 anchor:**
- `_GRAPH_BY_KEY`, `_GRAPH_POOL`, `_GRAPH_MIN_T`, `_NO_GRAPH`,
  `_VALIDATE_CAPTURE`, `_capture_graph()` — REMOVED. Every call goes
  through `_eager_call`. The +0.11x graph win is sacrificed for
  SOL/prod correctness.
- `_TACTIC_BY_KEY`, `_autotune_tactic`, `_AUTOTUNE_MIN_T`,
  `_AUTOTUNE_ITERS`, `_NO_AUTOTUNE` — KEPT. Tactic is keyed on shape
  (T, hidden_size, intermediate_size, local_num_experts, device) and
  the cubin's tactic selection is content-independent — same shape →
  same optimal tactic regardless of content distribution. Safe under
  cross-workload reuse.
- `_OUTPUT_BY_SHAPE_DEVICE` / `_EMPTY_*` / `_PDL_BY_DEVICE` — KEPT.
  Pre-allocated output buffers / empty placeholders / static device
  capability flags. Content-independent.

**Expected score vs anchor:** anchor was 1.33x mean (1.07-1.75x range)
under FIB. Without graph capture, T=1 unchanged (anchor already eager-
fallbacked T<2), T≥2 loses the +0.11x graph lift → expected ~1.20-
1.25x mean. The autotune win at T=11948/14107 is preserved.

═══════════════════════════════════════════════════════════════════════════
Below: original iter4 identity / delta / lessons header preserved as
history. The kernel body inherits iter4's correctness contract minus
the cached-graph fast path.
═══════════════════════════════════════════════════════════════════════════

iter4-autotune-tactic-sweep — round-4 anchor candidate for
moe-fp8-block-scale-ds-routing-topk8-ng8-kg4-e32-h7168-i2048 (B200 / Modal).

1) Identity
─────────────────────────────────────────────────────────────────────────
Round-4 lever: **per-shape autotune sweep of trtllm's `[tile_N, config]`
tactic space**, layered on top of iter3c's CUDA-graph capture+replay.
(In THIS derivative: capture/replay layer removed; autotune retained.)

For full identity / delta / lessons / dead-ends / open directions, see
the sibling `kernel.py` in this same variant directory.
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


_TOP_K = 8
_N_GROUP = 8
_TOPK_GROUP = 4
_NUM_EXPERTS = 256
_INTERMEDIATE_SIZE = 2048
_HIDDEN_SIZE = 7168

_FALLBACK_TACTIC = [-1, -1]

_ROUTING_METHOD = RoutingMethodType.DeepSeekV3.value
_WEIGHT_LAYOUT = WeightLayout.MajorK.value
_FP8_QUANT = Fp8QuantizationType.DeepSeekFp8
_ACTIVATION = ActivationType.Swiglu.value
_USE_SHUFFLED_WEIGHT = False
_DO_FINALIZE = True
_NORM_TOPK_PROB = True

_DTYPE_ACT = DtypeTrtllmGen.E4m3
_DTYPE_WEIGHTS = DtypeTrtllmGen.E4m3

_NO_AUTOTUNE = bool(int(os.environ.get("NO_AUTOTUNE", "0")))
_AUTOTUNE_ITERS = int(os.environ.get("AUTOTUNE_ITERS", "3"))
_AUTOTUNE_WARMUP = int(os.environ.get("AUTOTUNE_WARMUP", "1"))
_AUTOTUNE_MIN_T = int(os.environ.get("AUTOTUNE_MIN_T", "1000"))

_MOE_OP = None
_PDL_BY_DEVICE: dict = {}
_EMPTY_BY_DTYPE_DEVICE: dict = {}
_EMPTY_INT32_BY_DEVICE: dict = {}
_OUTPUT_BY_SHAPE_DEVICE: dict = {}

# Tactic cache — keyed on SHAPE only (no data_ptr). Tactic selection at
# the cubin level depends on (T, hidden_size, intermediate_size,
# local_num_experts), not on input content. Safe under cross-workload
# data_ptr reuse: two workloads with the same shape get the same tactic,
# which is the correct optimal choice for that shape regardless of which
# workload is calling.
_TACTIC_BY_KEY: dict = {}


def _init_moe_op():
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
        None,
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

    explicit = []
    for c in candidates:
        try:
            tac = [int(c[0]), int(c[1])]
            explicit.append(tac)
        except Exception:
            continue

    sweep = explicit + [list(_FALLBACK_TACTIC)]

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
            continue

    return best_tactic


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

    _eager_call(
        moe_op, routing_logits, routing_bias, hidden_states,
        hidden_states_scale, gemm1_weights, gemm1_weights_scale,
        gemm2_weights, gemm2_weights_scale, output, topk_ids,
        expert_weights, enable_pdl, local_expert_offset,
        local_num_experts, routed_scaling_factor, tactic,
    )
    return output
