"""
FlashInfer-Bench Local Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks locally.
Caches reference baseline on first run for stable, efficient subsequent runs.
"""

import argparse
import os
import sys
from functools import partial
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.benchmark_adapter as adapter
from scripts.bench_utils import (
    find_group_axis,
    get_trace_set_path,
    parse_int_filter,
    run_ab_compare,
    run_and_report,
    run_variance_check,
)
from scripts.pack_solution import pack_solution


def run_benchmark(blob: str, uuids: list, params: dict, *, capture_logs: bool = False) -> dict:
    """Thin run_fn: run the requested uuids through the adapter locally."""
    return adapter.run(blob, uuids, params, dataset_path=get_trace_set_path(),
                       capture_logs=capture_logs)


def main():
    """Pack solution and run benchmark."""
    # Line-buffer stdout so per-run progress prints stream to caller when piped.
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="Run benchmark with optional trajectory tracking")
    parser.add_argument("--label", default=None,
                        help="Label for trajectory tracking; also gates ITERATIONS.md protocol")
    parser.add_argument("--force-baseline", action="store_true",
                        help="Force re-profiling of reference baseline")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Only print score summary and per-group breakdown")
    parser.add_argument("--first", type=int, default=0, metavar="N",
                        help="Only run first N workloads (quick test mode, does not cache baseline)")
    parser.add_argument("--group", type=str, default=None, metavar="VALUES",
                        help="Only run workloads matching group axis values (comma-separated or range, e.g. --group 8,16 or --group 32-901)")
    parser.add_argument("--exclude-group", type=str, default=None, metavar="VALUES",
                        help="Exclude workloads matching group axis values (e.g. --exclude-group 1,14107)")
    parser.add_argument("--index", type=str, default=None, metavar="INDICES",
                        help="Only run specific workloads by index (e.g. --index 0,3,5 or --index 2-8)")
    parser.add_argument("--variance-check", type=int, default=0, metavar="N",
                        help="Run unchanged solution N times to measure across-run noise (>=2)")
    parser.add_argument("--smoke", action="store_true",
                        help="Run one workload per distinct group-axis bucket (covers every "
                             "group with minimum workload count). Does not cache baseline.")
    parser.add_argument("--ab-compare", dest="ab_compare", type=str, default=None,
                        metavar="LABEL",
                        help="Compare current solution to a labeled trajectory snapshot back-to-back "
                             "in the same process. Drift cancels, so deltas are tight. "
                             "Use instead of --variance-check when cross-session drift would swamp signal.")
    parser.add_argument("--capture-logs", dest="capture_logs", action="store_true",
                        help="Also capture stdout/stderr for PASSED workloads "
                             "(default: only non-PASSED). Use when diagnosing silent "
                             "performance regressions where kernel.py print(...) output "
                             "would otherwise be discarded by the isolated-runner redirect.")
    args = parser.parse_args()

    label = args.label

    # Parse group/exclude/index filters
    group_axis = ""
    group_values = None
    exclude_group_values = None
    workload_indices = None

    if args.group or args.exclude_group or args.smoke:
        group_axis = find_group_axis()
        if not group_axis and (args.group or args.exclude_group):
            print("ERROR: --group/--exclude-group requires a variable axis in definition.json, but none found.",
                  file=sys.stderr)
            sys.exit(1)
        # --smoke with no group_axis falls back silently to first workload.
    if args.group:
        group_values = parse_int_filter(args.group)
    if args.exclude_group:
        exclude_group_values = parse_int_filter(args.exclude_group)
    if args.index:
        workload_indices = parse_int_filter(args.index)

    # Isolate Triton JIT cache to this project
    os.environ.setdefault("TRITON_CACHE_DIR", str(PROJECT_ROOT / ".triton_cache"))

    if not args.quiet:
        print("Packing solution from source files...")
    solution_path = pack_solution(quiet=args.quiet)
    solution_blob = solution_path.read_text()

    if not args.quiet:
        meta = adapter.solution_meta(solution_blob)
        print(f"\nLoaded: {meta['name']} ({meta['definition']})")

    run_fn = partial(run_benchmark, capture_logs=args.capture_logs)

    # Filters are resolved to uuids inside the orchestrators (select_workload_uuids).
    filters = dict(
        max_workloads=args.first, group_values=group_values,
        exclude_group_values=exclude_group_values, workload_indices=workload_indices,
        smoke=args.smoke,
    )

    if args.ab_compare:
        run_ab_compare(
            solution_blob, run_fn,
            label=args.ab_compare,
            backend="local",
            quiet=args.quiet,
            current_label=label,
            **filters,
        )
        return

    if args.variance_check > 0:
        run_variance_check(
            solution_blob, run_fn,
            n_runs=args.variance_check,
            backend="local",
            quiet=args.quiet,
            label=label,
            **filters,
        )
        return

    run_and_report(
        solution_blob, run_fn,
        force_baseline=args.force_baseline,
        label=label,
        backend="local",
        quiet=args.quiet,
        **filters,
    )


if __name__ == "__main__":
    main()
