"""
iter2-direct-cpp-bypass — round-2 anchor candidate for
moe-fp8-block-scale-ds-routing-topk8-ng8-kg4-e32-h7168-i2048 (B200 / Modal).

═══════════════════════════════════════════════════════════════════════════
1) Identity
─────────────────────────────────────────────────────────────────────────
Round-2 lever: **strip the FlashInfer Python wrapper** down to a single
direct call into the TVM-FFI-loaded C++ binding
`moe_op.trtllm_fp8_block_scale_moe`. Same cubin, same routing → GEMM1 →
SwiGLU → GEMM2 → combine kernel as the parent
(`iter1-trtllm-turnkey-seed`); the work being amortized is *non-kernel
call-site overhead* concentrated on the small-T fast path.

Round-1 left ~50–80µs of Python wrapper work on every `run()` call (T=1
total ≈ 100µs → roughly half is non-kernel). This iter removes:
  • the `register_custom_op` Python-decorated op frame (no-op in
    flashinfer-utils, but still a Python function frame),
  • per-call `MoERunner(...)` construction (12 attribute assignments),
  • per-call `MoEInputs(...)` dataclass construction (6 fields),
  • per-call `_make_tuning_config(...)` (builds
    `get_last_power_of_2_num_tokens_buckets(8192,1)` list of 14 buckets
    + DynamicTensorSpec — non-trivial Python work),
  • per-call `AutoTuner.choose_one(...)` (singleton lock, kwarg dict
    assembly, `search_cache` hash; **even on a cache hit, ~10µs**),
  • per-call `torch.empty(0, …)` for `topk_ids` / `expert_weights`
    placeholders (we cache the singletons),
  • per-call `torch.empty(T,H, …)` for output (we cache by `(T,H)` —
    bench loops over fixed shapes; cache is bounded at 19 entries),
  • per-call `device_support_pdl(...)` (cached),
  • per-call `logger.warning_once(...)` from outer wrapper (cached but
    still a Python frame).

The bypass is **eager**, not graph-captured. Each `run()` still issues a
real C++ kernel launch — no silent-skip-cascade risk.

═══════════════════════════════════════════════════════════════════════════
2) Delta from prior anchor
─────────────────────────────────────────────────────────────────────────
Parent: `iter1-trtllm-turnkey-seed` (1.027x, 19/19 PASS). The parent
called `flashinfer.fused_moe.trtllm_fp8_block_scale_moe(...)` — the outer
@flashinfer_api wrapper, which delegates to
`get_trtllm_moe_sm100_module().trtllm_fp8_block_scale_moe(...)` (the
inner `register_custom_op`-decorated `trtllm_fp8_block_scale_moe_op`),
which then calls `moe_op.trtllm_fp8_block_scale_moe(...)` (the C++
binding).

This iter skips both Python layers and calls the C++ binding directly
with hard-coded `tactic=[-1, -1]` (the same fallback the parent picks
because `tune_mode=False`).

Same correctness contract: `do_finalize=True`, kernel writes in-place to
the pre-allocated output buffer.

═══════════════════════════════════════════════════════════════════════════
3) Lessons on this variant
─────────────────────────────────────────────────────────────────────────
- **`register_custom_op` is a no-op `lambda x: x` in flashinfer** for
  Torch >= 2.4 (see `flashinfer/utils.py:344-363`). So the inner op is
  a plain Python function — but it still has all the construction work
  inside. We get our win by not entering its frame at all.
- **`get_trtllm_moe_sm100_module()` is `@functools.cache`d**, so the
  underlying `gen_trtllm_gen_fused_moe_sm100_module()`,
  `module.build_and_load()`, and `setup_cubin_loader(...)` all happen
  exactly once across the process. First call kicks them off; we cache
  the returned `moe_op` (the tvm-ffi-loaded module) for re-use.
- **C++ binding signature** at `core.py:1786-1816` is the contract: 29
  positional args, order matters. `tactic` is a `List[int]` of length 2
  (`[-1, -1]` = "no tuning, use default config"). `routing_replay_out`
  is `Optional[torch.Tensor]` — we pass `None`.
- **Output is written in-place**: when `do_finalize=True`, the C++
  binding's return value is unused by the parent's Python wrapper
  (`return [output]` at `core.py:1819`). We get the same semantic by
  pre-allocating `output` once per `(T, H)` and returning it from
  `run()`. Eager mode — no replay aliasing.
- **Fresh-inputs contract still holds**: hidden_states, routing_logits,
  scales are FRESH each call (harness contract). Only the OUTPUT buffer
  is reused — the kernel overwrites it every call; the harness reads it
  after the call returns. No staleness, no silent-skip.
- **Cached buffer keys**:
  - `topk_ids` / `expert_weights` placeholders → keyed by
    `(device, routing_logits.dtype)`. The dtype is bf16 per the
    workloads; one slot.
  - `output` → keyed by `(T, H, device.index)`. H=7168 is fixed.
    Bounded by the 19 unique workload T values. Memory: at most
    `2 × 14107 × 7168 = ~200 MB` total across all cached buffers — well
    within Modal B200 80 GB.
- **`use_isolated_runner = true` in config.toml**, so each workload
  runs in its own subprocess; the cache is per-subprocess and starts
  fresh on each workload. No cross-workload contamination, no growth
  unbounded.

═══════════════════════════════════════════════════════════════════════════
4) Dead-ends tried (this iter)
─────────────────────────────────────────────────────────────────────────
None ruled out yet on this variant. (Inherits the parent's dead-end set:
`tune_max_num_tokens` past 8192, pre-dequant to bf16, 300s timeout.)

Open question for next iter (NOT a dead-end yet): the cached `output`
buffer assumes the harness doesn't read it after our `run()` returned
and before the next call's kernel completes. flashinfer-bench's
isolated-runner model does a `torch.cuda.synchronize()` between
measurement iterations, so this is safe. If a future bench mode runs
iterations without sync, the cached-output strategy would need a
double-buffer.

═══════════════════════════════════════════════════════════════════════════
5) Open directions (priority-ordered, surviving from parent)
─────────────────────────────────────────────────────────────────────────
1. **CUDA-graph capture per `(seq_len, local_expert_offset)`** for
   `seq_len <= 32`. After this iter's Python-overhead removal, the
   remaining small-T floor is whatever `cudaLaunchKernel` + cubin
   driver dispatch costs (~5-20µs). Graph replay can take that to ~1µs.
   The risk class is the silent-skip cascade (see `flashinfer-bench`
   SKILL); validate with zero-output / poison-cell / varying-inputs
   tests before declaring an anchor.
2. **Persistent autotune cache** in the Modal volume. Same as parent
   #2 — would benefit larger-T workloads where the GEMM dominates. Pay
   one ~5-10 min `with autotune(True, ...): warm_up()` once across
   containers; save ~10-30% on T>=512 workloads.
3. **`use_shuffled_weight=True` + `reorder_rows_for_gated_act_gemm`**:
   needs the harness's fresh-input model — only viable if reorder runs
   INSIDE a captured graph (coupled to #1).
4. **Workload-aware enable_pdl tuning**: B200 supports PDL; we cache
   `enable_pdl=True` at startup. The C++ kernel may have shape-specific
   thresholds where PDL hurts; worth a sweep but micro-optimization.

═══════════════════════════════════════════════════════════════════════════
"""

import torch

from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module
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

# Hard-coded "fallback tactic" — same as what AutoTuner returns when
# `tune_mode=False` and no cached tactic exists.
_FALLBACK_TACTIC = [-1, -1]

# Hard-coded enum / flag values (per parent anchor's config).
_ROUTING_METHOD = RoutingMethodType.DeepSeekV3.value
_WEIGHT_LAYOUT = WeightLayout.MajorK.value
_FP8_QUANT = Fp8QuantizationType.DeepSeekFp8
_ACTIVATION = ActivationType.Swiglu.value
_USE_SHUFFLED_WEIGHT = False
_DO_FINALIZE = True
_NORM_TOPK_PROB = True

# Lazy-initialized module + per-process caches. `use_isolated_runner=True`
# means these are scoped to one workload's subprocess.
_MOE_OP = None
_PDL_BY_DEVICE: dict = {}
_EMPTY_BY_DTYPE_DEVICE: dict = {}  # (dtype, device.index) -> 0-element tensor
_EMPTY_INT32_BY_DEVICE: dict = {}  # device.index -> 0-element int32 tensor
_OUTPUT_BY_SHAPE_DEVICE: dict = {}  # (T, H, device.index) -> bf16 output buffer


def _init_moe_op():
    """Force module build, cubin-loader setup, and grab the raw C++ binding.

    `get_trtllm_moe_sm100_module()` is `@functools.cache`d, so the heavy
    lifting (JIT compile / .so load / cubin callback registration) only
    happens on the first call across the process. We also fetch the
    underlying `moe_op` from `gen_trtllm_gen_fused_moe_sm100_module()`
    (also `@functools.cache`d) so subsequent calls don't need to traverse
    the SimpleNamespace.
    """
    global _MOE_OP
    if _MOE_OP is not None:
        return _MOE_OP

    # Side-effect: invokes setup_cubin_loader(...) and builds the
    # SimpleNamespace + MoERunner class (cached). We don't use the
    # SimpleNamespace; we go straight to the underlying tvm-ffi module
    # below. But this call is still needed so the cubin loader is wired.
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
        _FALLBACK_TACTIC,
        _FP8_QUANT,
        _ACTIVATION,
        _NORM_TOPK_PROB,
        None,  # routing_replay_out
    )

    return output
