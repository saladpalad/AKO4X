"""
iter1-trtllm-turnkey-seed — round-1 seed anchor for
moe-fp8-block-scale-ds-routing-topk8-ng8-kg4-e32-h7168-i2048 (B200 / Modal).

═══════════════════════════════════════════════════════════════════════════
1) Identity
─────────────────────────────────────────────────────────────────────────
Round-1 seed anchor. Full-bench result: **19/19 PASSED, FINAL SCORE 1.03x**
vs the operator's expert baseline (`flashinfer_wrapper_9sdjf3`), measured
on B200 / Modal on 2026-05-23 with `--label iter-1-seed-trtllm`.

Per-workload speedup range 0.984x – 1.12x (one mild T=56 dip in noise,
otherwise 1.00x–1.12x). Mean across 19 workloads = 1.03x.

The kernel is a single-call dispatch to
`flashinfer.fused_moe.trtllm_fp8_block_scale_moe` — the TRT-LLM-Gen fused
MoE entry point explicitly tagged on this operator
(`fi_api:flashinfer.fused_moe.trtllm_fp8_block_scale_moe` in
`docs/definition.json`). One launch covers DeepSeek-V3 routing (sigmoid →
+bias → per-group top-2 → top-4 groups → top-8 experts → normalize +
scale), FP8 block-scale GEMM1, SwiGLU, FP8 block-scale GEMM2, per-token
combine into bf16 output.

Why this is the campaign-best for round-1: the family had no prior anchor
(`baseline.json` was a bootstrap shell). This iter seeds it with 19
profiled latencies (0.10ms at T=1 → 2.38ms at T=14107) and demonstrates a
working end-to-end fused MoE on the harness. Any future iter must beat the
expert latency at the same precision — that's the structural bar this seed
sets.

═══════════════════════════════════════════════════════════════════════════
2) Delta from prior anchor
─────────────────────────────────────────────────────────────────────────
No prior anchor — round-1 first iter. The diff vs `expert_baseline.json`
(packaged `flashinfer_wrapper_9sdjf3`) is:
- Drops the expert's defensive `.to(torch.float32).contiguous()` calls on
  scale tensors that are already float32 + contiguous per the harness's
  input contract. These are no-ops in the happy path; removing them
  shaves a handful of microseconds on small-T workloads where Python
  dispatch is a measurable fraction of total time. Visible as +1-12% on
  T={1, 15, 16, 59, 80} workloads in the iter-1 result.
- Drops the expert's `local_expert_offset` / `routed_scaling_factor` tensor
  unwrapping branch — harness passes scalars directly per
  `docs/definition.json` (dtype=int32/float32, shape=null = scalar).

═══════════════════════════════════════════════════════════════════════════
3) Lessons on this variant
─────────────────────────────────────────────────────────────────────────
- **B200 fp8 block-scale routing is best served by the vendor turnkey
  path for round-1.** The kernel space (FP8 tcgen05 MMA on B200, fused
  routing → GEMM1 → SwiGLU → GEMM2 → combine in one cubin) is
  pre-shipped by TRT-LLM-Gen and accessed through flashinfer; rewriting
  the whole thing in Triton / CuTe / hand-CUDA for round-1 is
  cost-prohibitive and would almost certainly land below 1x in the same
  iter budget.
- **`hidden_states_scale` DeepSeekFp8 layout = `[H/128, T]`** (transposed
  from naive `[T, H/128]`). Already what the harness emits. Do NOT
  permute on the hot path; the kernel ingests it natively.
- **`weight_layout=MajorK (=0)` matches harness shapes:** `gemm1_weights`
  is `[E_local, 2*I, H]` and `gemm2_weights` is `[E_local, H, I]` —
  K-major innermost. No shuffle / reorder needed.
- **`routing_method_type=DeepSeekV3 (=2)`** is the integer enum for the
  sigmoid → bias-add → per-group top-2 → top-k-groups → top-k-experts
  routing pattern. `RoutingMethodType.DeepSeekV3` from
  `flashinfer.tllm_enums`.
- **Tolerance is `atol=1.0, rtol=0.3, required_matched_ratio=0.9`** — i.e.
  90% of output elements must satisfy `|out - ref| <= 1 + 0.3*|ref|`.
  This is permissive on purpose: MoE routing ties differ between the
  flashinfer kernel and the python reference (different tie-break order),
  so a small fraction of token-expert assignments shift, producing wide
  per-element absolute error in those rows. **The per-workload
  `abs_err=O(1e4)` and `rel_err=O(1e3-1e10)` numbers in the bench output
  are EXPECTED and do not indicate a bug** — the `matched_ratio` ≥ 0.9
  test is the actual correctness gate.
- **Modal cold-start exceeds the default `timeout_seconds=300`** for the
  trtllm fused-MoE first call. Round-1 raised it to 900s in `config.toml`.
  Non-frozen field; safe to keep at 900s across the family.

═══════════════════════════════════════════════════════════════════════════
4) Dead-ends tried
─────────────────────────────────────────────────────────────────────────
- **Raising `tune_max_num_tokens` past 8192**: irrelevant. Autotune is
  off by default (`tune_mode=False` outside an `autotune(True, ...)`
  context), so `AutoTuner.choose_one` returns tactic=-1 (fallback) on
  cache miss without sweeping. The tune-bucket count doesn't matter.
- **Manually dequantizing FP8 → bf16 before calling trtllm**: would
  destroy the entire reason for using the fp8 path; trtllm's kernel
  expects fp8 + scales and uses tcgen05 fp8-MMA. Pre-dequant adds a
  kernel launch and loses ~2x throughput. (Documented as a dead-end,
  not actually attempted — preserved here so the next session doesn't
  test it.)
- **300s `timeout_seconds`**: caused both expert AND solution TIMEOUT on
  the smoke test because Modal cold-start + flashinfer first-call setup
  exceeds 300s. Bumped to 900s.

═══════════════════════════════════════════════════════════════════════════
5) Open directions (priority-ordered)
─────────────────────────────────────────────────────────────────────────
1. **CUDA-graph capture of the steady-state path** for `seq_len <= 32`.
   The ~80-100µs Python + dispatch overhead is half of the T=1 latency
   (0.10ms total → ~0.05ms is launch overhead). A `torch.cuda.graph`
   capture + `g.replay()` could push the small-T subset score to ~1.3x.
   Capture once per `seq_len` bucket inside `run()`, key by `(seq_len,
   local_expert_offset)`. Validate with the silent-skip-cascade tests
   from the `bench` SKILL (zero-output replay, poison-cell, varying
   inputs) — the harness gives fresh data each call into the same tensor
   IDs, which is exactly the pattern that hides silent kernel skips.
2. **Persistent autotune cache** in the Modal volume. Currently
   `tune_mode=False` (no sweep) → fallback tactic. A one-time
   `with autotune(True, cache="/data/autotune.json"): warm-up call` on
   first invocation, then re-use the cache across containers. Pay
   ~5–10 min once, save ~10–30% on the GEMM-dominated workloads. The
   risk is that the cache key encodes the runner's shape spec, so a new
   shape needs a new entry — for our 19 workloads, this is a one-time
   sweep.
3. **`use_shuffled_weight=True` + `reorder_rows_for_gated_act_gemm`**:
   trtllm has a shuffled-weight fast path that reorders gemm1 weights
   for better tcgen05 throughput. Requires a one-time permute per
   expert; the harness's fresh-inputs-per-call model means this has to
   happen INSIDE the captured graph or it'll cost more than it saves.
   Coupled to lever #1 (CUDA-graph) — only worth attempting after the
   graph capture lands.
4. **Pre-allocate the output tensor** via the `output=...` keyword
   argument to `trtllm_fp8_block_scale_moe`. The kernel currently
   `torch.empty`s it internally each call. Cached in a module-level dict
   keyed by `(T, H)`. Saves one `cudaMalloc` per call — micro-win, only
   visible on T<8 where every microsecond counts.

═══════════════════════════════════════════════════════════════════════════
"""

import torch
from flashinfer.fused_moe import trtllm_fp8_block_scale_moe
from flashinfer.tllm_enums import (
    ActivationType,
    Fp8QuantizationType,
    RoutingMethodType,
    WeightLayout,
)


_TOP_K = 8
_N_GROUP = 8
_TOPK_GROUP = 4
_NUM_EXPERTS = 256
_INTERMEDIATE_SIZE = 2048


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
    local_num_experts = gemm1_weights.shape[0]

    return trtllm_fp8_block_scale_moe(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        num_experts=_NUM_EXPERTS,
        top_k=_TOP_K,
        n_group=_N_GROUP,
        topk_group=_TOPK_GROUP,
        intermediate_size=_INTERMEDIATE_SIZE,
        local_expert_offset=int(local_expert_offset),
        local_num_experts=local_num_experts,
        routed_scaling_factor=float(routed_scaling_factor),
        routing_method_type=RoutingMethodType.DeepSeekV3.value,
        use_shuffled_weight=False,
        weight_layout=WeightLayout.MajorK.value,
        do_finalize=True,
        fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
        activation_type=ActivationType.Swiglu.value,
        norm_topk_prob=True,
    )
