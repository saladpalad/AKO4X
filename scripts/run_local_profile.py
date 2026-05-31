"""
FlashInfer-Bench NCU Profiler Runner.

Profiles a solution on specific workloads using NVIDIA Nsight Compute (NCU).
Reuses pack_solution() to build Solution objects and the flashinfer-bench NCU agent API.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.benchmark_adapter as adapter
from scripts.bench_utils import find_group_axis, get_trace_set_path, parse_int_filter, tail_truncate_output
from scripts.pack_solution import pack_solution


def load_workloads():
    """Load workloads from the trace set. Returns (operator_name, list of {uuid, axes} dicts)."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    config_path = PROJECT_ROOT / "config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    operator = config["solution"]["definition"]

    entries = adapter.list_workloads(get_trace_set_path(), operator)
    if not entries:
        print(f"Error: No workloads found for operator '{operator}'", file=sys.stderr)
        sys.exit(1)
    return operator, entries


def filter_workloads(entries, indices=None, group_values=None, exclude_group_values=None):
    """Filter workloads by index, group, and exclusion. Returns list of (original_index, entry)."""
    group_axis = find_group_axis() if (group_values or exclude_group_values) else ""

    indexed = list(enumerate(entries))

    if indices is not None:
        index_set = set(indices)
        indexed = [(i, e) for i, e in indexed if i in index_set]

    if group_values and group_axis:
        group_set = set(group_values)
        indexed = [(i, e) for i, e in indexed if e["axes"].get(group_axis) in group_set]

    if exclude_group_values and group_axis:
        exclude_set = set(exclude_group_values)
        indexed = [(i, e) for i, e in indexed if e["axes"].get(group_axis) not in exclude_set]

    return indexed


def list_workloads(entries, indices=None, group_values=None, exclude_group_values=None):
    """Print workload table with indices, optionally filtered."""
    indexed = filter_workloads(entries, indices, group_values, exclude_group_values)

    total = len(entries)
    shown = len(indexed)
    label = f" (filtered {shown}/{total})" if shown < total else ""
    print(f"Workloads ({total} total{label}):\n")
    print(f"{'Index':<7} {'UUID':<12} {'Axes'}")
    print(f"{'-----':<7} {'----':<12} {'----'}")
    for i, e in indexed:
        uuid_prefix = e["uuid"][:8]
        axes_str = ", ".join(f"{k}={v}" for k, v in sorted(e["axes"].items()))
        print(f"{i:<7} {uuid_prefix:<12} {axes_str}")


def profile_workloads(args, entries):
    """Run NCU profiler on selected workloads via the adapter."""
    # Isolate Triton JIT cache to this project
    os.environ.setdefault("TRITON_CACHE_DIR", str(PROJECT_ROOT / ".triton_cache"))

    indexed = filter_workloads(entries, args.indices, args.group_values, args.exclude_group_values)
    if not indexed:
        print("Error: No workloads match the specified filters.", file=sys.stderr)
        sys.exit(1)

    trace_set_path = get_trace_set_path()

    print("Packing solution from source files...")
    solution_blob = pack_solution().read_text()

    for idx, e in indexed:
        uuid = e["uuid"]
        axes = e["axes"]
        axes_str = ", ".join(f"{k}={v}" for k, v in sorted(axes.items()))
        print(f"\nProfiling workload {idx}: {uuid[:8]}...")
        print(f"  Axes: {axes_str}")
        print(f"  Set: {args.set}, Page: {args.page}")
        if args.kernel_name:
            print(f"  Kernel filter: {args.kernel_name}")
        print()

        # Intentionally do NOT forward max_lines — flashinfer_bench's truncation
        # is head-biased and drops the metrics tables we actually want. Apply
        # tail-truncation locally instead (see bench_utils.tail_truncate_output).
        opts = {"set": args.set, "page": args.page, "timeout": args.timeout}
        if args.kernel_name:
            opts["kernel_name"] = args.kernel_name
        if args.sections:
            opts["sections"] = args.sections

        result = adapter.profile(solution_blob, uuid, opts, dataset_path=trace_set_path)
        if args.max_lines is not None:
            print(tail_truncate_output(result, args.max_lines))
        else:
            print(result)

        # Auto-save profile output with metadata
        profiles_dir = PROJECT_ROOT / "profiles"
        profiles_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        profile_data = {
            "timestamp": datetime.now().isoformat(),
            "workload_index": idx,
            "workload_uuid": uuid,
            "axes": dict(axes),
            "ncu_set": args.set,
            "ncu_page": args.page,
            "kernel_filter": args.kernel_name,
            "backend": "local",
            "output": result,
        }
        profile_path = profiles_dir / f"w{idx}_{timestamp}.json"
        profile_path.write_text(json.dumps(profile_data, indent=2))
        print(f"\nProfile saved to: {profile_path}", file=sys.stderr)


def main():
    # Line-buffer stdout so progress prints stream to caller when piped.
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        description="NCU profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python scripts/run_local_profile.py --list                         # list workloads
  python scripts/run_local_profile.py --index 0                     # NCU profile workload 0
  python scripts/run_local_profile.py --index 0 --set full          # full metrics
  python scripts/run_local_profile.py --list --exclude-group 1,14107  # filtered list
  python scripts/run_local_profile.py --ncu-options                  # list NCU sets/sections
""",
    )

    # Workload selection
    parser.add_argument("--index", type=str, default=None, metavar="INDICES",
                        help="Workload indices to profile (e.g. 0, 0,3,5, or 2-8)")
    parser.add_argument("--list", action="store_true",
                        help="List workloads with indices")
    parser.add_argument("--group", type=str, default=None, metavar="VALUES",
                        help="Filter by group axis values (e.g. 8,16 or 32-901)")
    parser.add_argument("--exclude-group", type=str, default=None, metavar="VALUES",
                        help="Exclude by group axis values (e.g. 1,14107)")

    # Mode
    parser.add_argument("--ncu-options", action="store_true",
                        help="List available NCU sets and sections")

    # NCU options
    parser.add_argument("--set", default="detailed",
                        help="NCU section set to collect (default: detailed)")
    parser.add_argument("--page", default="details", choices=["raw", "details", "source"],
                        help="NCU output page format (default: details)")
    parser.add_argument("--kernel-name", default=None,
                        help="Filter by kernel name (full-match regex; use .*name.* for substring)")
    parser.add_argument("--sections", nargs="+", default=None,
                        help="Additional NCU sections to collect beyond the set")
    parser.add_argument("--max-lines", type=int, default=None,
                        help="Keep only the last N lines (tail-truncate; metrics live at the end)")
    parser.add_argument("--timeout", type=int, default=180,
                        help="Timeout in seconds for NCU profiling (default: 180). Raise for reference kernels or --set full.")

    args = parser.parse_args()

    # Parse filter values
    args.indices = parse_int_filter(args.index) if args.index else None
    args.group_values = parse_int_filter(args.group) if args.group else None
    args.exclude_group_values = parse_int_filter(args.exclude_group) if args.exclude_group else None

    if args.ncu_options:
        print(adapter.list_ncu_options())
    elif args.list:
        _, entries = load_workloads()
        list_workloads(entries, args.indices, args.group_values, args.exclude_group_values)
    elif args.indices is not None:
        _, entries = load_workloads()
        profile_workloads(args, entries)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
