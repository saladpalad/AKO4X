# Benchmark reference

## Workload model

A workload is one (operator, input-shape) pair identified by a stable UUID listed in `docs/workloads.jsonl`. Per workload, the harness:

1. Generates fresh random input tensors at the shape declared by the workload.
2. Calls your `run(...)` function once per warmup, then `solution_iterations` times for measurement.
3. Compares the last call's output against the reference implementation under per-operator tolerance.

**Input freshness contract** (load-bearing, agents have lost iterations to violating this):

- **Inputs are STABLE across iterations within a trial**: a single trial bundles `warmup_runs + solution_iterations` calls (~103 by default) into one `time_runnable(...)` invocation that reuses the SAME `input_args` tuple — same tensor objects, same GPU addresses, same contents (`flashinfer/testing/utils.py::bench_gpu_time_with_cupti` closes `call_fn` over `input_args` and runs it in a tight loop). CUDA-graph capture + replay is viable within a trial.
- **Inputs change BETWEEN trials**: `num_trials` (default 5) outer-loop iterations each generate a fresh input set with new tensor objects and new memory addresses (`flashinfer_bench/bench/evaluators/default.py` `for inp in inputs:`). A kernel that caches anything keyed by `data_ptr()` or that captures a CUDA graph must detect the trial boundary (e.g. compare cached `data_ptr()` against the new call's pointer) and re-cache / re-capture.
- **Don't cache anything that depends on input *values***: pre-computed indices, transposed weights, routing tables — anything derived from tensor contents. The *contents* of any tensor in a future trial are unpredictable even though they're stable within a single trial. Caching shape-derived structures (grid dims, output buffers keyed by `(shape, dtype)`) is fine.
- **Fixed inputs across replays of the same captured graph**: when you `torch.cuda.graph` capture and `g.replay()`, the same captured input tensors are re-used. This means stale output from an earlier successful invocation can silently match the reference if the kernel responsible for writing that output isn't actually replayed (see "Silent-skip cascade" below).

`docs/workloads.jsonl` is informational only — the runner loads workloads from the dataset path / Modal volume, so editing or trimming `docs/workloads.jsonl` does not change what runs (it only breaks per-group scoring by orphaning UUIDs).

## Status enum

Returned per workload by the harness; surfaces in `bash scripts/bench.sh` output and in `results.json`.

- **`PASSED`** — output within tolerance, latency recorded.
- **`COMPILE_ERROR`** — kernel failed to compile / load before any execution. **Includes import-time failures**: e.g. `flashinfer-bench` loads the `cutlass` module at kernel-build time, so a failed `import cutlass.cute` surfaces as `COMPILE_ERROR` for every workload, not as a Python traceback. To get the real error, fail-loud inside the kernel (e.g. `assert _CUTE_OK` after an `import … except` guard).
- **`INCORRECT_NUMERICAL`** — output exceeded tolerance vs reference. Run `bash scripts/sanitize.sh --index <N>` first; races / NaN / OOB writes often surface as INCORRECT_NUMERICAL with a wide blast radius across workloads.
- **`RUNTIME_ERROR`** — kernel raised at execution time (CUDA error, Python exception during `run()`).
- **`TIMEOUT`** — wall-time exceeded `timeout_seconds` (per-workload). Common on the first run profiling the pure-Python reference at large shapes.

## Reference & cached baseline

The reference implementation is **pure Python / PyTorch** — slow but easy to validate. It's defined inside the operator's `definition.json` (`"reference"` key) and used for two things:

1. **Correctness oracle**: each measurement run compares your output against a fresh reference run.
2. **Latency denominator**: speedup = reference_latency / your_latency.

The reference latency is profiled once and cached as `baseline.json` per environment, so subsequent runs only profile your solution.

**Baseline freshness rule**: cached baseline is invalidated when:
- The workload-UUID set in `docs/workloads.jsonl` differs from the cached set, OR
- The baseline source path / signature changes (e.g., expert-baseline file replaced, reference Python implementation revised in `definition.json`).

Implementation: `scripts/bench_utils.py::load_baseline` (freshness checks live inside it); written by `scripts/bench_utils.py::save_baseline`.

To force re-profiling: `bash scripts/bench.sh --force-baseline`.

## Scoring formula

**Score = arithmetic mean of speedup factors across all workloads.**

```
speedup_w = reference_latency_w / your_latency_w
score = mean(speedup_w over all workloads)
```

The score is a **tracking metric**, not an optimization target. Because the reference is pure Python/PyTorch, absolute speedup values can range from 300x to 9000x+ depending on workload size — this is expected and does not indicate a problem. What matters is that your kernel's **latency** decreases across iterations.

Implementation: `scripts/bench_utils.py::compute_score`.

## `config.toml` schema

Each child env's `config.toml` has three tables — `[solution]`, `[build]`, `[benchmark]` (benchmark defaults + per-op_type overrides are merged into `[benchmark]` at spawn time). The kernel-facing keys you change — `language` / `entry_point` / `destination_passing_style` — live under **`[build]`**, not at top level:

```toml
[solution]
name = "<operator>-solution"
definition = "<operator>"
author = "user"

[build]
gpu = "<gpu-name>"
dataset_path = "/path/to/dataset"  # local backend only
language = "triton"                # python | cpp | cuda | triton | tilelang  (CuTe DSL uses python)
entry_point = "kernel.py::run"     # <file>::<function> path inside solution/
destination_passing_style = false

[benchmark]
baseline_iterations = 5        # reference profiling iterations (keep low, default 5)
solution_iterations = 100      # solution profiling iterations
num_trials = 5                 # trials per workload
warmup_runs = 3                # warmup before measurement
timeout_seconds = 300          # per-workload evaluation timeout
use_isolated_runner = true     # run each workload in an isolated process
atol = 0.01                    # absolute tolerance per element
rtol = 0.01                    # relative tolerance per element
required_matched_ratio = 0.9   # fraction of elements that must match
```

### Field semantics

- **`[build].language`**: which compile path the harness uses — one of `python` / `triton` / `cpp` / `cuda` / `tilelang` (the values flashinfer-bench accepts; there is no `cute-dsl` value). `python` → no compilation, imported as a module (plain PyTorch — **also the value for CuTe DSL kernels**, which self-JIT at call time via the `@cute.kernel` decorator); `cpp`/`cuda` → TVM FFI builder; `triton`/`tilelang` → JIT compilation by the respective DSL.
- **`[build].entry_point`**: `<file>::<function>` relative to `solution/`. Per-language conventions:
  - `python` (pure PyTorch): `kernel.py::run`. No JIT compilation, no extra packages — wrap `run()` body in `@torch.no_grad()` to drop autograd overhead during measurement.
  - `triton` / `tilelang`: `kernel.py::run` (DSL-specific JIT compilation). **CuTe DSL** also uses `kernel.py::run` with `language = python` (it self-JITs via `@cute.kernel`).
  - `cpp`: `binding.py::kernel` (Python binding wraps the C++).
  - `cuda`: `kernel.cu::<kernel_name>` (with `destination_passing_style = true`) OR `binding.py::kernel` (host wrapper, `destination_passing_style = false`).
- **`[build].destination_passing_style`**: when `true`, the harness pre-allocates the output tensor and passes it as the last positional arg; your kernel writes into it in-place. When `false`, your function returns the output. Convenience for low-overhead kernels that don't need to allocate.
- **`[benchmark].baseline_iterations`** is intentionally low (default 5). The cached baseline only needs to be approximately correct. Baseline profiling automatically uses a lightweight configuration (warmup=1, trials=1) to minimize overhead.
- **`[benchmark].solution_iterations`** can stay at 100. Optimized kernels launch few CUDA ops, so profiling is fast.
- **`[benchmark].use_isolated_runner = true`** runs each workload in a fresh subprocess — guards against cross-workload state aliasing (e.g. module-level caches that persist between workloads and serve stale data on the next workload). The default-off bug history caused silent INCORRECT_NUMERICAL; default is now `true`.
- **`[benchmark].timeout_seconds`** is the per-workload evaluation timeout (subprocess lifetime: import + correctness + warmup + measurement). The default `300` is fine for hand-written kernels (Triton / TileLang / .cu) whose first-call setup is cheap. It is **commonly too tight for vendor-cubin-backed kernels on Modal cold-start** — `trtllm_*_moe` / `trtllm_*_attention` paths bundle a cubin loader (`setup_cubin_loader`) that does the first-time module fetch + setup inside the timed window. Symptom: both the expert baseline AND your solution emit `TIMEOUT` on the same workload (smoke `--first 1` is enough to surface this). Fix: bump to `900` in `config.toml`. The field is **not** in the frozen-for-comparability list — only scoring formula / baseline freshness rule / tolerance keys are.
- **Vendor-cubin NCU profileability** — `trtllm_*` paths bundle vendor cubins (`setup_cubin_loader`) but the loaded modules emit standard SASS + SM metrics under `ncu`. Profile them like any other kernel; no opacity caveat applies. (Common misconception worth flagging because the cubin packaging suggests otherwise.)

### Tolerance keys

`atol` / `rtol` / `required_matched_ratio` are populated into `[benchmark]` at spawn time from per-operator overrides. `required_matched_ratio` is the fraction of output elements that must satisfy `|out - ref| <= atol + rtol * |ref|` for the workload to be considered PASSED. Some operators (e.g. routing-style ops with deliberate ties) use ratios below 1.0.

## Frozen for bench comparability

When running under a multi-run campaign, edits to **fib-specific behavior** that anchor cross-run comparability are rejected across runs:

- **Scoring formula** — `compute_score` arithmetic mean of speedups (above).
- **Baseline freshness rule** — what invalidates `baseline.json` (above), implemented in `bench_utils.py::load_baseline`.
- **`save_baseline` write path** — `bench_utils.py::save_baseline`.
- **Per-operator tolerance keys** — `atol` / `rtol` / `required_matched_ratio` in `[benchmark]`.

To change any of these, start fresh with re-measured baselines. (The general scope policy lives in your `closed-loop-scope.md`; the items above are the fib-specific instances of it.)

## TVM FFI builder behavior

flashinfer-bench compiles `[build].language = cpp` / `cuda` kernels via TVM FFI. The one builder constraint worth single-sourcing here (the `cpp` and `cuda` skills point at this section):

- **No `extra_ldflags` pass-through to the linker** — external shared libraries can't be link-bound at compile time. In practice this costs a valid solution nothing: the precompiled vendor libraries you might think to link, **cuBLAS / cuDNN**, are disallowed regardless (see "Valid solution: write your own kernel" below — they're someone else's kernel), and **CUTLASS**, the allowed building block, is header-only (`#include` it; the headers ship with the flashinfer install, no linking needed). Don't `dlopen` cuBLAS / cuDNN to route around either the linker limit or the policy.

## Valid solution: write your own kernel (no operator-library delegation)

The benchmark scores *your* kernel. A solution must **implement the operator itself** — it may **not** hand the operator's core computation to a precompiled vendor kernel. The test: *is the heavy compute done by code you wrote (a DSL / `.cu` / CUTLASS-instantiated kernel), or by a black-box routine someone else compiled?* The latter is delegation — you'd be measuring their kernel, not yours.

- **Banned** — precompiled vendor *operator* kernels, reached any way:
  - operator-kernel libraries: `import flashinfer; flashinfer.<op>(...)`, `torch.ops.flashinfer.*`, `deepgemm.*`.
  - the closed vendor math libraries **cuBLAS / cuDNN**: linking them, `dlopen`/`dlsym`-ing `libcublas` / `libcudnn` from a `.cu` / `.cpp`, or letting a thin `torch` wrapper stand in *as* the whole operator (e.g. a GEMM solution that just returns `torch.matmul(...)`, an attention solution that just returns `F.scaled_dot_product_attention(...)`). Calling one of these IS the shortcut this rule exists to stop.
- **Allowed** — source-level building blocks you compile into your own kernel:
  - **CUTLASS / CuTe** — a header-only template library you instantiate in your `.cu` (and the basis of the CuTe DSL); you write and configure the kernel, so it is yours.
  - the DSLs: **Triton / TileLang / CuTe DSL**.
  - plain **`torch` tensor ops as glue** — reshape, gather, small elementwise, output setup *around* a kernel you wrote. The line is core-compute vs. glue: torch orchestrating your kernel is fine; a torch op standing in *as* the operator is the banned case above.

CUTLASS is treated differently from cuBLAS / cuDNN on purpose: CUTLASS is source you compose and compile (like a DSL); cuBLAS / cuDNN are closed precompiled kernels you can only call. The first is the job; the second is delegation.

This is a benchmark-specific validity rule: it travels with this SKILL if the benchmark is swapped, and a different benchmark may relax or redefine it.

## Silent-skip cascade (the keystone)

A silent kernel skip — kernel doesn't actually execute on graph replay — is **silent under flashinfer-bench specifically because of fib's input-replay model**:

1. Some kernel doesn't run on `g.replay()` (DSL-specific cause — see the corresponding DSL skill: e.g. `@cute.kernel.launch()` not entering capture mode, CUDA chevron without explicit stream).
2. The output buffer keeps stale value from a prior successful execution.
3. **flashinfer-bench replays each workload with the same input tensor addresses (graph capture's contract) AND the captured tensors hold the same data as the warmup call** — so the stale output happens to match the freshly-computed reference → correctness `PASSED` silently.
4. CUPTI faithfully reports the latency of whatever IS in the graph (the surviving kernels) → headline speedup is inflated by whatever fraction of the real per-call work was in the skipped kernel.

Detection methodology (zero-output / poison-cell / varying-inputs tests) lives in the `bench` skill; per-DSL specific causes live in the relevant DSL skill; the cascade itself — why fib makes this silent — lives here.

## NCU profiling NVTX range

The benchmark's NCU agent wraps the profiled call in `torch.cuda.nvtx.range("flashinfer_bench_ncu_profile")` and passes `--nvtx --nvtx-include "flashinfer_bench_ncu_profile]"` to ncu (see `flashinfer_bench/agents/ncu.py`). The `profiler-ncu` skill points here for this range name: if a profiled kernel doesn't land inside this range, ncu reports "No kernels were profiled".

## COUPLED references

- `scripts/benchmark_adapter.py` — the sole `flashinfer_bench`-importing module (the swap seam); the runtime core and runners call into its plain-data functions (`str`/`list`/`dict` in, normalized dict out — no benchmark types cross).
- `scripts/bench_utils.py` — runtime core; the frozen-behavior code paths above are inside this file.
- `scripts/run_local.py` / `scripts/run_modal.py` — backend runners that call into `bench_utils` / `benchmark_adapter`.
- `config.toml`'s `[benchmark]` table — benchmark defaults + per-op_type tolerance overrides, populated at spawn time.
- `docs/workloads.jsonl` (in child env) — UUID + axis listing for the operator (informational; runner reads from dataset path).
