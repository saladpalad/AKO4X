# moe-fp8-block-scale-ds-routing-topk8-ng8-kg4-e32-h7168-i2048

Closed-loop campaign on operator `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`
(DeepSeek-V3 MoE, FP8 block-scale, fully fused: routing → GEMM1 → SwiGLU →
GEMM2 → per-token combine). Target: B200 / Modal. Baseline measured
2026-05-23 from FlashInfer's `trtllm_fp8_block_scale_moe` expert (cuda 13.0,
19 workloads, T=1..14107).

## Anchor

**iter4-autotune-tactic-sweep-gated** — 1.33x mean (19/19 PASS, range
1.070x–1.750x). Builds on iter3c's capture-and-replay framework by adding
a per-shape, per-subprocess sweep of `moe_op.trtllm_get_valid_moe_configs`
to pick the best `[tile_N, config]` pair (timed via `cudaEvent`), then
bakes the chosen tactic into the captured graph. Sweep gated to T≥1000
(`_AUTOTUNE_MIN_T=1000`); below the gate, the kernel reverts to
`tactic=[-1,-1]` vendor fallback because sub-100µs timing noise was
de-ranking the fallback when included as one of N candidates. The lift
target is the vendor heuristic's tile_N=4096 clamp at the two largest
workloads — T=11948 1.016x→1.07x (+0.047x), T=14107 1.018x→1.09x
(+0.068x), reproducible across two drift-cancelled ab-compare runs.
+0.85% drift-cancelled vs iter3c overall (small mean delta because 2/19
workloads carry the lift; min floor moved up materially).

## History

- **round-1** (2026-05-23) — seed anchor `iter1-trtllm-turnkey-seed` at 1.027x.
  Auto-promoted baseline.json from bootstrap shell to real (expert-profiled,
  19 workloads). 1 accepted harness edit: flashinfer-bench SKILL now warns
  that `timeout_seconds=300` default is too tight for vendor-cubin-backed
  kernels (trtllm_*) on Modal cold-start (bump to 900).
- **round-2** (2026-05-23) — anchor rotates to `iter2-direct-cpp-bypass` at
  1.21x (+0.18x via direct C++ binding call, bypassing FlashInfer's
  Python wrapper layers `flashinfer.fused_moe.trtllm_fp8_block_scale_moe`
  → `register_custom_op`-decorated op → `moe_op.trtllm_fp8_block_scale_moe`).
  Mechanism stripped ~30µs Python frame work per call on T=1 (0.10ms →
  0.070ms); large-T essentially unchanged (vendor GEMM ceiling). 1 accepted
  harness edit: flashinfer-bench SKILL "Input freshness contract" section
  rewritten to clarify inputs are STABLE within a trial (~103 iterations)
  and only change BETWEEN trials (5 by default) — corrects the prior
  per-call wording that would have misframed any CUDA-graph capture/replay
  evaluation. Verified against `flashinfer/testing/utils.py::bench_gpu_time_with_cupti`
  and `flashinfer_bench/bench/evaluators/default.py`.
- **round-3** (2026-05-23) — anchor rotates to
  `iter3c-cuda-graph-eager-T1-fallback` at 1.32x (+0.11x via
  `torch.cuda.graph()` capture+replay of the trtllm 6-kernel pipeline,
  with `_GRAPH_MIN_T=2` eager-fallback at T=1 to preserve the
  matched_ratio gate that iter2's borderline T=1 abs_err already grazed).
  Trial-boundary detection: `hidden_states.data_ptr()` probe →
  re-capture on miss. Architecture-(a) static-grid verified against
  trtllm cubin source (`trtllm_fused_moe_runner.cu`,
  `trtllm_fused_moe_routing_deepseek.cu` — grid dims derived from host
  scalars, no host reads of device routing data). Silent-skip canary:
  variance-check 1/3 completed at 1.28x (within 3% of labeled 1.32x);
  runs 2/3 hit Modal session-reuse anomaly (HARNESS ANOMALY now surfaced,
  see round-3 ledger). Cross-iter triangulation: 3 fresh Modal containers
  (v1, 3b, 3c) hit same per-workload latencies within 1-3% spread.
  1 accepted harness edit: bench_utils.py `run_variance_check` print branch
  distinguishes empty-traces "HARNESS ANOMALY" from real "all failed",
  paired with bench SKILL benchmark.md callout (master-extended, paired
  COUPLED-references update).
  Open levers for next round: persistent autotune cache in Modal volume
  (T≥901 floor at vendor's `tactic=[-1,-1]` fallback); PDL tuning sweep
  under graph replay; shuffled-weight `use_shuffled_weight=True` +
  `reorder_rows_for_gated_act_gemm` (amortizes over 100 captured replays
  per trial); T=1 graph-capture revisit with stabilization (low priority,
  small headroom).
- **round-4** (2026-05-23) — anchor rotates to
  `iter4-autotune-tactic-sweep-gated` at 1.33x (+0.85% drift-cancelled
  over iter3c via per-subprocess sweep of vendor-validated
  `[tile_N, config]` tactic pairs, gated to T≥1000). Lift concentrates at
  T=11948 (+0.047x) and T=14107 (+0.068x) where vendor heuristic clamps
  at tile_N=4096 and the cubin catalog has non-default `config` variants
  that outperform the default; reproducible across two ab-compare runs.
  Forensic insight (corrects misleading parent open-direction note):
  tactic schema is `[tile_N, config]`, NOT `[gemm1_tactic, gemm2_tactic]`
  — sub verified against `csrc/trtllm_fused_moe_kernel_launcher.cu:143-148`.
  Cross-env portability sidestepped: per-subprocess in-memory cache, no
  Modal volume needed. 2 accepted harness edits (master-extended): (a)
  bench_utils.py `print_results` adds HARNESS ANOMALY branch for the
  single-labeled-bench Modal session failure shape (cudaErrorDevicesUnavailable
  on the reference baseline) — mirrors round-3's variance-check edit; paired
  bench SKILL benchmark.md callout. (b) bench_utils.py
  `_find_trajectory_snapshot` now falls back to `docs/prior/variants/<label>/`
  when `trajectory/` is empty (fresh round-N spawn), enabling
  `--ab-compare <parent-anchor>` immediately on spawn without a manual
  `mkdir + cp` shim; paired bench SKILL benchmark.md update.
  Open directions for round-5 (sub-identified, structural framing for next
  master): **whole-DSL alternative** (CUTLASS / CuTe-DSL grouped GEMM /
  DeepGEMM targeting the (E=32, tile_N=4096, H=7168, I=2048) shape) is
  the only remaining direction that could exceed trtllm's tactic-catalog
  ceiling — many-iter commitment, natural for a fresh campaign with the
  spec frozen here. Within-trtllm levers (shuffled-weight, T=1
  graph-capture revisit, PDL sweep) all share the same vendor-cubin
  ceiling; worth a forensic-closure round but unlikely to substantially
  shift the mean.

- **round-5** (2026-05-23) — **forensic-closure round**; anchor STAYS at
  iter4 (1.33x). Archive `iter5-forensic-closure` at 1.31x (Δ=+0.23% vs
  iter4 within drift; variance-check CV 0.054% — tightest of the
  campaign). Kernel body functionally identical to iter4; contribution
  is the 5-section header documenting closures of three within-trtllm
  levers:
  - **PDL=False (iter5a)**: REFUTED — PDL is correctness-load-bearing
    at T=1 (not just perf). Without PDL, the cubin's serialized
    schedule shifts FP8 rounding boundaries at the borderline T=1
    matched_ratio, dropping below 0.9 gate → INCORRECT_NUMERICAL.
    Upper bound on lift is zero at T=1 (correctness gate) and
    ≤vendor-default elsewhere.
  - **shuffled-weight (iter5b v1+v2)**: REFUTED across two failure
    modes. v1 (weights-only shuffle) → INCORRECT_NUMERICAL
    `abs_err=1.02e+06` because the cubin's shuffled fast path reads
    weights at shuffled positions but per-block scales at original
    layout. v2 (+ block-level scale shuffle) → kernel TIMEOUT (block
    shuffle is mathematically wrong: post-shuffle row blocks mix
    input-top and input-bottom halves, single per-block scale can't
    represent both). trtllm's `use_shuffled_weight=True` is internally
    wired ONLY paired with `MatrixLayout::BlockMajorK` (verified at
    `csrc/trtllm_fused_moe_kernel_launcher.cu:1444`); our MajorK +
    block-scale + shuffled is undocumented/unsupported via the public
    binding. Future path requires either weight layout transform or
    kernel-level scale-handling reverse-engineering — both
    multi-iter, out of round scope.
  - **T=1 graph-capture revisit**: deferred. iter3c's eager-fallback
    gate already captures the closure forensically (FP8 perturbation
    pushes borderline matched_ratio past tolerance); no new evidence
    needed.
  1 accepted harness edit (sub-evidenced, master-translated to source):
  spawn.py modal-backend bench/profile/sanitize wrapper generation
  prepends venv-discovery prelude — `if ! command -v modal; then walk
  up .venv .. ../.venv .. ../../.venv … and PATH-prepend the first
  match`. Evidence: r5 first labeled bench failed `modal: command not
  found` (sub had to prefix every call with explicit PATH). Source-tree
  patch is spawn.py:611-628 (sub's child-env-path scope translates to
  spawn.py's generator strings).

## Campaign convergence note

Within-trtllm lever space is **empirically closed** after round 5:
- Python wrapper bypass: +0.18x (round 2)
- CUDA graph capture (with T=1 eager fallback): +0.11x (round 3)
- Per-shape tactic sweep gated to T≥1000: +0.0058x absolute / +0.85%
  drift-cancelled (round 4)
- PDL toggle: refuted (correctness-load-bearing)
- Shuffled-weight: refuted (unsupported with block-scale + MajorK)
- T=1 graph-capture revisit: closed via iter3c eager-fallback

Final anchor: **iter4-autotune-tactic-sweep-gated at 1.33x mean (19/19
PASS, range 1.070x–1.750x, variance CV 0.054% on the sub-baseline
iter5 with same body)**.

Remaining direction with potential lift over trtllm's tactic-catalog
ceiling on T≥11948 (vendor heuristic clamps at tile_N=4096): a
hand-rolled grouped FP8 block-scale GEMM (CUTLASS / CuTe-DSL /
DeepGEMM) targeted at our shape mix — many-iter commitment, natural
for a fresh campaign with the spec frozen here. Within-trtllm levers
are exhausted in this codebase.
