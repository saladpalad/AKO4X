"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA GPUs via Modal, with trajectory tracking.
GPU type is read from config.toml (set at environment creation time).
Caches reference baseline on first run for stable, efficient subsequent runs.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

# Read config locally; on Modal remote container config.toml doesn't exist
# (image & GPU are already determined at deploy time, so defaults are fine)
_config_path = PROJECT_ROOT / "config.toml"
if _config_path.exists():
    with open(_config_path, "rb") as _f:
        _config = tomllib.load(_f)
    _GPU_TYPE = _config.get("build", {}).get("gpu", "b200").upper()
else:
    _GPU_TYPE = "B200"

import modal

# The adapter is the sole flashinfer_bench importer. This module-level import
# runs in BOTH contexts: locally at `modal run` discovery (PROJECT_ROOT — the
# child dir — is on sys.path above, so the `scripts.` package import resolves),
# AND inside the container, because Modal re-imports this module to locate the
# remote function. In the container __file__ is /root/run_modal.py so
# PROJECT_ROOT is "/" and `scripts` isn't importable — the adapter is in the
# image at /root/project/scripts (add_local_file below), so fall back to the
# bare name with that dir on sys.path. Re-raise anything that isn't the missing
# `scripts` package (e.g. flashinfer_bench genuinely absent locally). Only plain
# image-pin CONSTANTS cross here — no benchmark types — so the remote function
# signature is plain data and Modal cloudpickles only str/list/dict.
try:
    from scripts.benchmark_adapter import (
        MODAL_PACKAGE_PIN,
        MODAL_EXTRA_PIN,
        MODAL_IMAGE_REGISTRY,
        MODAL_PYTHON,
    )
except ModuleNotFoundError as exc:
    if exc.name not in ("scripts", "scripts.benchmark_adapter"):
        raise
    sys.path.insert(0, "/root/project/scripts")
    from benchmark_adapter import (
        MODAL_PACKAGE_PIN,
        MODAL_EXTRA_PIN,
        MODAL_IMAGE_REGISTRY,
        MODAL_PYTHON,
    )

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

# Base image: official flashinfer-bench CI image (matches bare-metal eval env).
#
# Why switched from nvidia/cuda:12.8.0-devel (2026-04-19): the bare-metal eval
# runs `flashinfer/flashinfer-ci-cu132:20260401-2c675fb` with CUDA 13.2 + PyTorch
# 2.12.0+cu132 + Triton 3.6.0 + FlashInfer (main, from source) + FlashInfer-Bench
# (main, from source) + cupti-python + deep-gemm + helion + mlc-ai-tirx-cu130
# (TVM) + nvidia-cutlass-dsl (CuTe DSL). Aligning our dev env means compile +
# runtime behaviour we measure on Modal is much closer to what the eval sees.
# Residual mismatch: Modal B200 is sm_100, eval is sm_100a — hardware, not image.
#
# Rollback (if the official image breaks something): swap `_base_image` back to
#   modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
# and re-add the old pip_install chain (see git history of this file).
# Official CI image gives us CUDA 13.2 + PyTorch 2.12 + Triton 3.6 matching eval.
# We still need add_python=3.12 because modal's function runner expects Python at
# /usr/local/bin/python3.12 which the bare CI image doesn't expose — so we layer
# modal's Python on top and pip-install our Python deps into it. (Flashinfer-bench
# and flashinfer-python end up duplicated vs the image's native Python, but only
# modal's Python path matters for the runner.)
_base_image = (
    modal.Image.from_registry(
        MODAL_IMAGE_REGISTRY,
        add_python=MODAL_PYTHON,
    )
    .env({"CUDA_HOME": "/usr/local/cuda"})
)

image = (
    _base_image
    .pip_install("torch", "numpy", "cupti-python", "triton", "tilelang",
                 "apache-tvm-ffi")
    .pip_install(MODAL_PACKAGE_PIN)
    .pip_install(MODAL_EXTRA_PIN)
    .pip_install("torch-c-dlpack-ext")
    .run_commands("apt-get update && apt-get install -y git clang && pip install wheel && pip install git+https://github.com/deepseek-ai/DeepGEMM.git@main --no-build-isolation")
    # --no-deps: direct install pulls a newer apache-tvm-ffi (>=0.2) that is incompatible with TileLang.
    .run_commands("pip install --no-deps nvidia-cutlass-dsl>=4.3.4")
    # Self-disabling CUTLASS-headers patch.
    .run_commands(
        "FI_DATA=$(python3 -c 'import flashinfer, os; print(os.path.dirname(flashinfer.__file__))')/data && "
        "if [ -f $FI_DATA/cutlass/include/cutlass/cutlass.h ]; then "
        "echo 'CUTLASS headers already present in flashinfer install, skipping patch'; exit 0; fi && "
        "mkdir -p $FI_DATA/cutlass && cd $FI_DATA/cutlass && "
        "git init -q && git remote add origin https://github.com/NVIDIA/cutlass.git && "
        "git fetch --depth 1 -q origin da5e086dab31d63815acafdac9a9c5893b1c69e2 && "
        "git checkout -q FETCH_HEAD && rm -rf $FI_DATA/cutlass/.git"
    )
    # bench_utils is no longer imported in-container (the bench loop now lives in
    # adapter.run); only the adapter file is needed remotely. config.toml stays for
    # any in-container diagnostics that key off the build section.
    .add_local_file(str(PROJECT_ROOT / "scripts" / "benchmark_adapter.py"), "/root/project/scripts/benchmark_adapter.py")
    .add_local_file(str(PROJECT_ROOT / "config.toml"), "/root/project/config.toml")
)


@app.function(image=image, gpu=f"{_GPU_TYPE}:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(
    blob: str,
    uuids: list,
    params: dict,
    capture_logs: bool = False,
    capture_autotune: bool = False,
):
    """Run benchmark on Modal GPU and return the normalized results dict.

    Trace-set assembly, the engine call, and result extraction all live in
    adapter.run; the workloads are the pre-resolved `uuids` and `params` is the
    BenchmarkConfig kwargs dict. When capture_autotune=True, returns
    {"results": <normal shape>, "autotune_log": <str>}. Only str/list/dict cross
    the .remote() boundary — no pydantic types are cloudpickled.
    """
    import sys as _sys
    _sys.path.insert(0, "/root/project/scripts")
    from benchmark_adapter import run

    return run(blob, uuids, params, dataset_path=TRACE_SET_PATH,
               capture_logs=capture_logs, capture_autotune=capture_autotune)


@app.local_entrypoint()
def main(label: str = None, force_baseline: bool = False, quiet: bool = False,
         first: int = 0, group: str = None, exclude_group: str = None, index: str = None,
         variance_check: int = 0, smoke: bool = False, extremes: bool = False,
         ab_compare: str = None, capture_logs: bool = False):
    """Pack solution and run benchmark on Modal GPU, with trajectory tracking.

    --variance-check N: run the same unchanged solution N times to measure
    across-run noise (~N times bench cost). Result saved to trajectory/noise-floor.json.

    --ab-compare <label>: run current solution + a trajectory snapshot
    back-to-back in the same Modal container. Drift cancels, delta is tight.
    Use when --variance-check's cross-session drift (±1x) would swamp signal.

    --extremes: run only the smallest + largest group-axis values (e.g. seq_len
    min and max). Input-spectrum correctness probe — NOT a performance verdict.
    Mutually exclusive with --first, --index, --group, --smoke.

    --capture-logs: also capture stdout/stderr for PASSED workloads (default:
    only non-PASSED). Use when diagnosing silent performance regressions where
    kernel.py print(...) output would otherwise be discarded by the
    isolated-runner redirect.
    """
    # Line-buffer stdout so progress prints stream to caller when piped.
    sys.stdout.reconfigure(line_buffering=True)

    from functools import partial

    import scripts.benchmark_adapter as adapter
    from scripts.bench_utils import (find_group_axis, parse_int_filter,
                                     run_and_report, run_variance_check,
                                     run_ab_compare)
    from scripts.pack_solution import pack_solution

    # Mutual exclusion: --extremes must not combine with other subset filters.
    if extremes and (first or smoke or group or index):
        import sys as _sys
        print("ERROR: --extremes cannot be combined with --first/--index/--group/--smoke "
              "(all are subset filters).", file=_sys.stderr)
        _sys.exit(1)

    # Parse group/exclude/index filters
    group_axis = ""
    group_values = None
    exclude_group_values = None
    workload_indices = None

    if group or exclude_group or smoke or extremes:
        group_axis = find_group_axis()
        if not group_axis and (group or exclude_group):
            import sys as _sys
            print("ERROR: --group/--exclude-group requires a variable axis in definition.json, but none found.",
                  file=_sys.stderr)
            _sys.exit(1)
        if not group_axis and extremes:
            import sys as _sys
            print("ERROR: --extremes requires a variable axis in definition.json, but none found. "
                  "Use --first 1 or --index for single-workload probes.", file=_sys.stderr)
            _sys.exit(1)
        # --smoke with no group_axis falls back silently to first workload (select_workload_uuids handles it).
    if group:
        group_values = parse_int_filter(group)
    if exclude_group:
        exclude_group_values = parse_int_filter(exclude_group)
    if index:
        workload_indices = parse_int_filter(index)

    if not quiet:
        print("Packing solution from source files...")
    solution_path = pack_solution(quiet=quiet)
    solution_blob = solution_path.read_text()

    if not quiet:
        meta = adapter.solution_meta(solution_blob)
        print(f"\nLoaded: {meta['name']} ({meta['definition']})")

    run_fn = partial(run_benchmark.remote, capture_logs=capture_logs)

    # Filters are resolved to uuids inside the orchestrators (select_workload_uuids).
    filters = dict(
        max_workloads=first, group_values=group_values,
        exclude_group_values=exclude_group_values, workload_indices=workload_indices,
        smoke=smoke, extremes=extremes,
    )

    if ab_compare:
        run_ab_compare(
            solution_blob, run_fn,
            label=ab_compare,
            backend=f"modal-{_GPU_TYPE.lower()}",
            quiet=quiet,
            current_label=label,
            **filters,
        )
        return

    if variance_check > 0:
        # Variance check opts into autotune capture so the summary table can
        # show per-run selected-config and flag autotune drift.
        run_fn_autotune = partial(run_benchmark.remote, capture_logs=capture_logs,
                                  capture_autotune=True)
        run_variance_check(
            solution_blob, run_fn_autotune,
            n_runs=variance_check,
            backend=f"modal-{_GPU_TYPE.lower()}",
            quiet=quiet,
            label=label,
            **filters,
        )
        return

    run_and_report(
        solution_blob, run_fn,
        force_baseline=force_baseline,
        label=label,
        backend=f"modal-{_GPU_TYPE.lower()}",
        quiet=quiet,
        **filters,
    )
