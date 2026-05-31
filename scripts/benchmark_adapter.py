"""Benchmark adapter — the single seam between AKO4X and the active benchmark.

This is the **only** module in the repo that imports ``flashinfer_bench``.
Everything else (the runners, ``bench_utils``, ``pack_solution``, the
cheat-check) reaches the benchmark through the **plain-data functions** below —
no benchmark types ever cross this boundary. The default benchmark is
**flashinfer-bench**.

Porting AKO4X to a different benchmark
--------------------------------------
Reimplement the public functions below so they resolve to your benchmark's
runtime, then rewrite the ``benchmark`` SKILL (``templates/skills/benchmark/``)
and ``templates/benchmark/evaluation.toml``. Nothing else under ``scripts/``
should need to change — ``bench_utils`` and the runners only ever pass/receive
``str`` / ``list[str]`` / ``dict``. The full procedure is in ``docs/porting.md``.

Public surface (plain data in, plain data out — no benchmark types escape here)
-------------------------------------------------------------------------------
Discovery   : ``list_workloads(dataset_path, definition) -> [{"uuid","axes"}, ...]``
Packing     : ``pack(source_dir, build_cfg, *, name, definition, author) -> blob:str``
              ``solution_meta(blob) -> {"name","definition","author"}``
Execution   : ``run(blob, uuids, params, *, dataset_path, capture_logs=False,
              capture_autotune=False) -> normalized_result_dict``
Profiling   : ``profile(blob, uuid, opts, *, dataset_path, env_pairs=None) -> str``
              ``list_ncu_options() -> str``
Sanitizer   : ``sanitize(blob, uuid, opts, *, dataset_path) -> str``
Cheat-check : ``cheat_check(blob, uuids, *, dataset_path, n_iters=4) -> dict``
Constants   : ``STATUS_*``; ``MODAL_IMAGE_REGISTRY`` / ``MODAL_PYTHON`` /
              ``MODAL_PACKAGE_PIN`` / ``MODAL_EXTRA_PIN``; ``DATASET_PATH_ENV`` /
              ``LEGACY_DATASET_PATH_ENV``; ``NCU_NVTX_RANGE``

The solution **blob** is the ``solution.json`` text. ``params`` is the
``BenchmarkConfig`` kwargs dict (``run`` does ``BenchmarkConfig(**params)``
internally). ``run`` builds a single-operator trace set from ``uuids``, runs the
engine, and flattens the result to the normalized dict below.

Normalized result dict (``run``'s output; consumed by the benchmark-agnostic
scoring / baseline code in ``bench_utils``)::

    {definition_name: {workload_uuid: {
        "status": <str>,                  # one of STATUS_* below
        "solution": <str>,
        "axes": {<axis>: <value>, ...},
        "latency_ms": <float>,            # present when PASSED
        "reference_latency_ms": <float>,
        "speedup_factor": <float>,
        "max_abs_error": <float|"NaN">,   # present when correctness ran
        "max_rel_error": <float|"NaN">,
        "error_log": <str>,               # present for non-PASSED workloads
        "log": <str>,                     # present with capture_logs
    }}}

A port must make ``run`` yield this shape; the scoring (``compute_score``) and
baseline (``load_baseline`` / ``save_baseline``) logic in ``bench_utils`` then
works unchanged.
"""

# No module-level flashinfer_bench import: every function below imports what it
# needs from the benchmark at call time (function scope). Importing this module
# therefore does NOT require flashinfer_bench to be installed — only the plain
# constants below resolve at import, so a port or a no-package host (e.g. the
# macOS modal-only setup that drops flashinfer-python) can still import the
# launcher scripts. Only an actual run / pack / profile / sanitize / cheat-check
# *call* loads the benchmark. Because no benchmark types cross the public surface
# (everything is str / dict / list), Modal cloudpickles only builtins — there is
# no by-reference module-path identity to resolve in the container.

# --- Status enum (the active benchmark's per-workload outcome strings) -------
# Mirrors flashinfer_bench's Evaluation status `.value`s. Kept as plain strings
# so non-adapter code can compare without importing the benchmark's enum.
STATUS_PASSED = "PASSED"
STATUS_COMPILE_ERROR = "COMPILE_ERROR"
STATUS_INCORRECT_NUMERICAL = "INCORRECT_NUMERICAL"
STATUS_RUNTIME_ERROR = "RUNTIME_ERROR"
STATUS_TIMEOUT = "TIMEOUT"

# --- Dataset discovery -------------------------------------------------------
# Env var spawn.py / bench_utils consult for the trace-set path (local backend).
DATASET_PATH_ENV = "AKO_DATASET_PATH"
LEGACY_DATASET_PATH_ENV = "FIB_DATASET_PATH"

# --- NCU profiling -----------------------------------------------------------
# The benchmark's NCU agent wraps the profiled call in this NVTX range and
# filters ncu to it. Surfaced for the profiler-ncu skill's "No kernels were
# profiled" diagnosis. (Defined by flashinfer_bench/agents/ncu.py.)
NCU_NVTX_RANGE = "flashinfer_bench_ncu_profile"

# --- Modal image pins --------------------------------------------------------
# Base CI image + dependency pins for the Modal backend, matching the bare-metal
# eval environment. The Modal runner scripts build their images from these (each
# layers its own extra pip installs on top). A port that runs on Modal points
# these at its own benchmark's image / package.
MODAL_IMAGE_REGISTRY = "flashinfer/flashinfer-ci-cu132:20260401-2c675fb"
MODAL_PYTHON = "3.12"
MODAL_PACKAGE_PIN = (
    "flashinfer-bench @ https://github.com/flashinfer-ai/flashinfer-bench/"
    "archive/f7b4d8d185625ab2d609233a1a06e99ee18a0c6b.tar.gz"
)
MODAL_EXTRA_PIN = (
    "flashinfer-python @ "
    "https://github.com/flashinfer-ai/flashinfer/archive/refs/heads/main.tar.gz"
)


def run_benchmark_all(bench_trace_set, config):
    """Run the benchmark engine over a prepared TraceSet, returning a result TraceSet.

    Internal helper called by ``run`` (above): ``bench_trace_set`` is the
    single-operator TraceSet ``run`` assembles from the requested uuids and
    ``config`` is a ``BenchmarkConfig``; the returned object is flattened by
    ``_extract_results`` into the normalized result dict.
    """
    # Function-local import: the engine submodule is only needed where the
    # benchmark actually runs (local GPU host or Modal container), never at the
    # module-import sites that just want the constants/types above.
    from flashinfer_bench.bench.benchmark import Benchmark

    return Benchmark(bench_trace_set, config).run_all(dump_traces=True)


def run_ncu(solution, workload, **kwargs):
    """Run the benchmark's NCU profiling agent on one (solution, workload)."""
    from flashinfer_bench.agents import flashinfer_bench_run_ncu

    return flashinfer_bench_run_ncu(solution, workload, **kwargs)


def list_ncu_options():
    """Return the benchmark's NCU set/section listing (`ncu --list-sets` etc.)."""
    from flashinfer_bench.agents import flashinfer_bench_list_ncu_options

    return flashinfer_bench_list_ncu_options()


def run_sanitizer(solution, workload, **kwargs):
    """Run the benchmark's compute-sanitizer agent on one (solution, workload)."""
    from flashinfer_bench.agents import flashinfer_bench_run_sanitizer

    return flashinfer_bench_run_sanitizer(solution, workload, **kwargs)


# ===========================================================================
# Plain-data public surface (the data-contract seam)
# ===========================================================================
# Everything below takes/returns only plain data — ``str`` solution-blobs,
# ``list[str]`` workload uuids, ``dict`` params, and the normalized result dict.
# No flashinfer_bench types cross this boundary: FIB objects are constructed and
# consumed entirely inside these functions (function-local imports). This is what
# lets a port reimplement the benchmark behind ~8 plain-data functions, and it
# keeps the Modal host->container boundary free of cloudpickled pydantic objects.


def list_workloads(dataset_path, definition):
    """Return ``[{"uuid": str, "axes": dict}, ...]`` for ``definition``, in dataset order."""
    from flashinfer_bench import TraceSet

    trace_set = TraceSet.from_path(dataset_path)
    entries = trace_set.workloads.get(definition, [])
    return [{"uuid": w.workload.uuid, "axes": dict(w.workload.axes)} for w in entries]


def pack(source_dir, build_cfg, *, name, definition, author):
    """Pack kernel sources from ``source_dir`` into a solution-blob (solution.json text).

    ``build_cfg`` keys: ``language``, ``entry_point``, ``destination_passing_style``
    (default False), ``target_hardware`` (default ``["cuda"]``).
    """
    from flashinfer_bench import BuildSpec
    from flashinfer_bench.agents import pack_solution_from_files

    spec = BuildSpec(
        language=build_cfg["language"],
        target_hardware=build_cfg.get("target_hardware", ["cuda"]),
        entry_point=build_cfg["entry_point"],
        destination_passing_style=build_cfg.get("destination_passing_style", False),
    )
    solution = pack_solution_from_files(
        path=str(source_dir), spec=spec, name=name,
        definition=definition, author=author,
    )
    return solution.model_dump_json(indent=2)


def solution_meta(blob):
    """``{"name", "definition", "author"}`` from a solution-blob.

    The single sanctioned place that introspects a blob's internals, so callers
    can treat the blob as opaque.
    """
    from flashinfer_bench import Solution

    sol = Solution.model_validate_json(blob)
    return {"name": sol.name, "definition": sol.definition, "author": sol.author}


def run(blob, uuids, params, *, dataset_path, capture_logs=False, capture_autotune=False):
    """Run the benchmark engine over the workloads named by ``uuids``.

    Reconstructs the solution from ``blob``, builds a single-operator trace set from
    the requested uuids (dataset order preserved), constructs ``BenchmarkConfig(**params)``,
    runs the engine, and returns the normalized result dict. When
    ``capture_autotune=True`` returns ``{"results": <dict>, "autotune_log": <str>}``.
    """
    from flashinfer_bench import BenchmarkConfig, Solution, TraceSet

    solution = Solution.model_validate_json(blob)
    config = BenchmarkConfig(**params)

    trace_set = TraceSet.from_path(dataset_path)
    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")
    definition = trace_set.definitions[solution.definition]
    if not uuids:
        raise ValueError("run() called with no workload uuids — nothing to benchmark")
    uuid_set = set(uuids)
    workloads = [w for w in trace_set.workloads.get(solution.definition, [])
                 if w.workload.uuid in uuid_set]
    # Enforce exactly-the-requested uuids. The caller resolves the uuid list from
    # docs/workloads.jsonl (host-side); this runs against the dataset at
    # dataset_path (the Modal volume, in the cloud backend). If those two sources
    # disagree — stale docs, partial volume upload, wrong-definition uuid — a pure
    # membership filter would silently run a SUBSET, corrupting the score/baseline
    # (compute_score averages over whatever ran). Raise loudly instead so the
    # divergence surfaces rather than masquerading as a quietly-smaller run.
    found = {w.workload.uuid for w in workloads}
    missing = uuid_set - found
    if missing:
        raise ValueError(
            f"{len(missing)}/{len(uuid_set)} requested workload uuid(s) not found in the "
            f"dataset for definition '{solution.definition}' (e.g. {sorted(missing)[0]!r}). "
            f"The selection source (docs/workloads.jsonl) and the execution dataset "
            f"({dataset_path}) may have diverged."
        )

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    if capture_autotune:
        import contextlib
        import io
        import os

        prior_autotune = os.environ.get("TRITON_PRINT_AUTOTUNING")
        os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
        buf = io.StringIO()
        # Best-effort: Triton writes autotune lines via Python's stderr. C-level
        # writes to fd 2 bypass this, but upstream Triton uses the Python logger.
        try:
            with contextlib.redirect_stderr(buf):
                result_trace_set = run_benchmark_all(bench_trace_set, config)
            results = _extract_results(result_trace_set, solution.definition,
                                       capture_all_logs=capture_logs)
            return {"results": results, "autotune_log": buf.getvalue()}
        finally:
            # Restore so in-process subsequent runs (e.g. variance-check followed
            # by a normal bench in the same Python session) don't inherit autotune
            # printing.
            if prior_autotune is None:
                os.environ.pop("TRITON_PRINT_AUTOTUNING", None)
            else:
                os.environ["TRITON_PRINT_AUTOTUNING"] = prior_autotune

    result_trace_set = run_benchmark_all(bench_trace_set, config)
    return _extract_results(result_trace_set, solution.definition,
                            capture_all_logs=capture_logs)


def profile(blob, uuid, opts, *, dataset_path, env_pairs=None):
    """NCU-profile one workload. Replaces the rich-typed run_ncu wrapper on the surface."""
    # Set caller env (e.g. NO_GRAPH) BEFORE importing the benchmark, so module-level
    # env gates the kernel evaluates at its own import time (run_ncu builds/imports
    # the kernel below) see the requested values. Matches the old remote-body order
    # (env first, then the FIB import) — the seam refactor must not reorder this.
    # Snapshot prior values so in-process subsequent calls don't inherit caller-set
    # env (e.g. NO_GRAPH=1 leaking into a later non-profile bench in the same shell).
    import os
    prior_env = {}
    if env_pairs:
        for k, v in env_pairs.items():
            prior_env[k] = os.environ.get(k)
            os.environ[k] = str(v)
    try:
        from flashinfer_bench import Solution

        solution = Solution.model_validate_json(blob)
        workload = _find_workload(dataset_path, solution.definition, uuid)
        return run_ncu(solution, workload, trace_set_path=dataset_path, **opts)
    finally:
        for k, v in prior_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def sanitize(blob, uuid, opts, *, dataset_path):
    """compute-sanitizer one workload. Replaces the rich-typed run_sanitizer wrapper."""
    from flashinfer_bench import Solution

    solution = Solution.model_validate_json(blob)
    workload = _find_workload(dataset_path, solution.definition, uuid)
    return run_sanitizer(solution, workload, trace_set_path=dataset_path, **opts)


def cheat_check(blob, uuids, *, dataset_path, n_iters=4):
    """Varying-inputs correctness audit over the probe ``uuids``. Returns a plain dict.

    Mutates inputs in place across ``n_iters`` and flags kernels whose outputs don't
    change (cached / capture-stale returns). The probe-slice *selection* is the
    caller's job (benchmark-agnostic indexing); this owns only build + gen + run.
    """
    import torch

    from flashinfer_bench import Solution, TraceSet
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
    from flashinfer_bench.compile import BuilderRegistry

    solution = Solution.model_validate_json(blob)
    trace_set = TraceSet.from_path(dataset_path)
    if solution.definition not in trace_set.definitions:
        return {"status": "ERROR",
                "reason": f"definition {solution.definition!r} not in trace set"}
    definition = trace_set.definitions[solution.definition]
    all_wl = trace_set.workloads.get(solution.definition, [])
    if not all_wl:
        return {"status": "ERROR", "reason": f"no workloads for {solution.definition!r}"}
    uuid_set = set(uuids)
    probe = [w for w in all_wl if w.workload.uuid in uuid_set]
    if not probe:
        return {"status": "ERROR", "reason": "no workloads match the requested probe uuids"}

    registry = BuilderRegistry.get_instance()
    runnable = registry.build(definition, solution)
    device = "cuda:0"

    out = {"status": "PASS", "definition": solution.definition,
           "n_iters": n_iters, "workloads": {}}
    overall_pass = True

    for trace in probe:
        wl = trace.workload
        wl_key = wl.uuid[:8]
        entry = {"axes": dict(wl.axes), "n_iters": n_iters}
        try:
            safe_tensors = None
            if any(inp.type == "safetensors" for inp in wl.inputs.values()):
                safe_tensors = load_safetensors(definition, wl, trace_set.root)
            inputs = gen_inputs(definition, wl, device, safe_tensors)

            # Two-shot warmup: lets JIT compile and CUDA Graph capture; after warmup
            # any caching keyed on tensor addresses is primed.
            res0 = runnable.call_value_returning(*inputs)
            torch.cuda.synchronize()
            res0 = runnable.call_value_returning(*inputs)
            torch.cuda.synchronize()
            del res0

            hashes = []
            for _ in range(n_iters):
                _mutate_inputs_inplace(inputs)
                res = runnable.call_value_returning(*inputs)
                torch.cuda.synchronize()
                out_list = list(res) if isinstance(res, tuple) else [res]
                hashes.append(_hash_outputs(out_list))
                del res, out_list

            unique = len(set(hashes))
            entry["unique_hashes"] = unique
            entry["all_iters_differ"] = all(
                hashes[i] != hashes[i - 1] for i in range(1, len(hashes))
            )
            if not entry["all_iters_differ"]:
                overall_pass = False
                entry["status"] = "FAIL"
                entry["reason"] = (
                    f"only {unique}/{n_iters} unique outputs across mutated inputs"
                    " — kernel may be returning cached / capture-stale output"
                )
            else:
                entry["status"] = "PASS"
        except Exception as e:
            overall_pass = False
            entry["status"] = "ERROR"
            entry["reason"] = f"{type(e).__name__}: {e}"
        finally:
            out["workloads"][wl_key] = entry

    runnable.cleanup()
    if not overall_pass:
        out["status"] = "FAIL"
    return out


# --- adapter-private helpers (not part of the public surface) ----------------

def _find_workload(dataset_path, definition, uuid):
    """Resolve a single Workload object by uuid from the dataset."""
    from flashinfer_bench import TraceSet

    trace_set = TraceSet.from_path(dataset_path)
    for w in trace_set.workloads.get(definition, []):
        if w.workload.uuid == uuid:
            return w.workload
    raise ValueError(f"Workload uuid {uuid!r} not found for definition {definition!r}")


def _truncate_log(log, max_chars=3000):
    """Truncate log to the last max_chars characters, preserving line boundaries."""
    if not log or len(log) <= max_chars:
        return log
    truncated = log[-max_chars:]
    nl = truncated.find("\n")
    if nl != -1 and nl < 200:
        truncated = truncated[nl + 1:]
    return f"[...truncated...]\n{truncated}"


def _extract_results(result_trace_set, definition_name, *, capture_all_logs=False):
    """Flatten a result TraceSet's Trace/Evaluation objects into the normalized dict."""
    import math

    traces = result_trace_set.traces.get(definition_name, [])
    results = {definition_name: {}}
    for trace in traces:
        if not trace.evaluation:
            continue
        entry = {
            "status": trace.evaluation.status.value,
            "solution": trace.solution,
            "axes": dict(trace.workload.axes),
        }
        if trace.evaluation.performance:
            entry["latency_ms"] = trace.evaluation.performance.latency_ms
            entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
            entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
        if trace.evaluation.correctness:
            max_abs = trace.evaluation.correctness.max_absolute_error
            max_rel = trace.evaluation.correctness.max_relative_error
            entry["max_abs_error"] = "NaN" if (max_abs is not None and math.isnan(max_abs)) else max_abs
            entry["max_rel_error"] = "NaN" if (max_rel is not None and math.isnan(max_rel)) else max_rel
        log_text = getattr(trace.evaluation, "log", "")
        if trace.evaluation.status.value != "PASSED" and log_text:
            entry["error_log"] = _truncate_log(log_text)
        elif capture_all_logs and log_text:
            entry["log"] = _truncate_log(log_text, max_chars=20000)
        results[definition_name][trace.workload.uuid] = entry
    return results


def _mutate_inputs_inplace(inputs):
    """Mutate float / packed inputs in place. Skips int32/int64 (likely indices).

    Preserves tensor pointers so honest CUDA-graph captures keyed on addresses still
    replay correctly. Cheating kernels (cute-skip, cache return) ignore these
    mutations and produce byte-identical outputs across iters.
    """
    import torch

    for t in inputs:
        if not isinstance(t, torch.Tensor):
            continue
        if t.dtype == torch.float8_e4m3fn:
            tmp = (torch.randn(t.shape, dtype=torch.bfloat16, device=t.device) * 0.5)
            t.copy_(tmp.to(torch.float8_e4m3fn))
        elif t.dtype == torch.float8_e5m2:
            tmp = (torch.randn(t.shape, dtype=torch.bfloat16, device=t.device) * 0.5)
            t.copy_(tmp.to(torch.float8_e5m2))
        elif t.is_floating_point():
            t.normal_()
        elif t.dtype == torch.int8:
            # Packed FP8+scale layouts: randomize byte content; the kernel parses
            # the layout from current bytes.
            t.random_(-128, 128)
        elif t.dtype == torch.uint8:
            t.random_(0, 256)
        # int32/int64 -> likely indices/seq_lens/block_table/cu_seqlens — skip to
        # avoid out-of-bounds reads.


def _hash_outputs(outputs):
    import hashlib

    import torch

    h = hashlib.sha256()
    for o in outputs:
        if isinstance(o, torch.Tensor):
            buf = o.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            h.update(buf)
        else:
            h.update(repr(o).encode())
    return h.hexdigest()
