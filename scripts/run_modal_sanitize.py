"""
FlashInfer-Bench Compute-Sanitizer Runner (Modal Backend).

Thin wrapper around flashinfer_bench_run_sanitizer. `--tool` forwards to
the `sanitizer_types` argument; `all` passes None (flashinfer-bench
default = all four tools). Output is post-processed by
summarize_sanitizer_noise to surface a "NOISE FILTER" banner before the
raw sanitizer text.

Usage:
    modal run scripts/run_modal_sanitize.py --list
    modal run scripts/run_modal_sanitize.py --index 0 --tool memcheck
"""

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

_config_path = PROJECT_ROOT / "config.toml"
if _config_path.exists():
    with open(_config_path, "rb") as _f:
        _config = tomllib.load(_f)
    _GPU_TYPE = _config.get("build", {}).get("gpu", "b200").upper()
else:
    _GPU_TYPE = "B200"

import modal

# Adapter (sole flashinfer_bench importer): only plain image-pin CONSTANTS are
# imported at module top — no benchmark types cross the seam. This module-level
# import runs both locally (PROJECT_ROOT on sys.path → `scripts.` resolves) and in
# the container, where Modal re-imports this module to find the remote function but
# PROJECT_ROOT is "/" so `scripts` isn't importable — the adapter is in the image at
# /root/project/scripts (add_local_file below), so fall back to the bare name.
# Re-raise non-`scripts` errors (e.g. flashinfer_bench genuinely absent locally).
try:
    from scripts.benchmark_adapter import (
        MODAL_PACKAGE_PIN,
        MODAL_IMAGE_REGISTRY,
        MODAL_PYTHON,
    )
except ModuleNotFoundError as exc:
    if exc.name not in ("scripts", "scripts.benchmark_adapter"):
        raise
    sys.path.insert(0, "/root/project/scripts")
    from benchmark_adapter import (
        MODAL_PACKAGE_PIN,
        MODAL_IMAGE_REGISTRY,
        MODAL_PYTHON,
    )

app = modal.App("flashinfer-bench-compute-sanitizer")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

# Base image aligned with bare-metal eval (flashinfer-ci-cu132). See
# run_modal.py for rationale + rollback. Only extras layered on top.
_pip_deps = ["torch", "numpy", "cupti-python", "triton", "tilelang", "apache-tvm-ffi"]

image = (
    modal.Image.from_registry(
        MODAL_IMAGE_REGISTRY,
        add_python=MODAL_PYTHON,
    )
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install(*_pip_deps)
    .pip_install(MODAL_PACKAGE_PIN)
    .pip_install("torch-c-dlpack-ext")
    .add_local_file(str(PROJECT_ROOT / "scripts" / "benchmark_adapter.py"), "/root/project/scripts/benchmark_adapter.py")
)


# ---------------------------------------------------------------------------
# Modal remote function
# ---------------------------------------------------------------------------

@app.function(image=image, gpu=f"{_GPU_TYPE}:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_sanitizer_remote(blob: str, workload_uuid: str, san_kwargs: dict) -> str:
    """Run compute-sanitizer inside Modal container and return output."""
    import sys as _sys
    _sys.path.insert(0, "/root/project/scripts")
    from benchmark_adapter import sanitize

    return sanitize(blob, workload_uuid, san_kwargs, dataset_path=TRACE_SET_PATH)


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

def _load_workloads_jsonl():
    """Load workloads from docs/workloads.jsonl."""
    jsonl_path = PROJECT_ROOT / "docs" / "workloads.jsonl"
    if not jsonl_path.exists():
        sys.exit(f"Error: {jsonl_path} not found")
    workloads = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                workloads.append(json.loads(line))
    return workloads


def _filter_workloads(entries, indices=None, group_values=None, exclude_group_values=None):
    """Filter workloads by index and group axis. Returns list of (original_index, entry)."""
    from scripts.bench_utils import find_group_axis
    group_axis = find_group_axis() if (group_values or exclude_group_values) else ""

    indexed = list(enumerate(entries))

    if indices is not None:
        index_set = set(indices)
        indexed = [(i, e) for i, e in indexed if i in index_set]

    if group_values and group_axis:
        group_set = set(group_values)
        indexed = [(i, e) for i, e in indexed if e["workload"]["axes"].get(group_axis) in group_set]

    if exclude_group_values and group_axis:
        exclude_set = set(exclude_group_values)
        indexed = [(i, e) for i, e in indexed if e["workload"]["axes"].get(group_axis) not in exclude_set]

    return indexed


def _list_workloads(entries, indices=None, group_values=None, exclude_group_values=None):
    """Print workload table with indices, optionally filtered."""
    indexed = _filter_workloads(entries, indices, group_values, exclude_group_values)
    total = len(entries)
    shown = len(indexed)
    label = f" (filtered {shown}/{total})" if shown < total else ""
    print(f"Workloads ({total} total{label}):\n")
    print(f"{'Index':<7} {'UUID':<12} {'Axes'}")
    print(f"{'-----':<7} {'----':<12} {'----'}")
    for i, entry in indexed:
        wl = entry["workload"]
        uuid_prefix = wl["uuid"][:8]
        axes_str = ", ".join(f"{k}={v}" for k, v in sorted(wl["axes"].items()))
        print(f"{i:<7} {uuid_prefix:<12} {axes_str}")


_VALID_TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck", "all")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    index: str = None,
    list: bool = False,
    tool: str = "memcheck",
    timeout: int = 300,
    max_lines: int = -1,
    group: str = None,
    exclude_group: str = None,
):
    """compute-sanitizer on specific workloads via Modal."""
    # Line-buffer stdout so progress prints stream to caller when piped.
    sys.stdout.reconfigure(line_buffering=True)

    import os

    from scripts.bench_utils import parse_int_filter, summarize_sanitizer_noise

    if tool not in _VALID_TOOLS:
        print(f"Error: --tool must be one of {_VALID_TOOLS} (got '{tool}')", file=sys.stderr)
        sys.exit(1)

    indices = parse_int_filter(index) if index else None
    group_values = parse_int_filter(group) if group else None
    exclude_group_values = parse_int_filter(exclude_group) if exclude_group else None

    if list:
        entries = _load_workloads_jsonl()
        _list_workloads(entries, indices, group_values, exclude_group_values)
        return

    if index is None:
        print("Usage: modal run scripts/run_modal_sanitize.py --index <N>")
        print("       modal run scripts/run_modal_sanitize.py --index <N> --tool initcheck")
        print("       modal run scripts/run_modal_sanitize.py --list")
        print(f"       tools: {', '.join(_VALID_TOOLS)}")
        sys.exit(1)

    os.environ.setdefault("TRITON_CACHE_DIR", str(PROJECT_ROOT / ".triton_cache"))

    entries = _load_workloads_jsonl()
    indexed = _filter_workloads(entries, indices, group_values, exclude_group_values)
    if not indexed:
        print("Error: No workloads match the specified filters.", file=sys.stderr)
        sys.exit(1)

    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_blob = pack_solution().read_text()

    # Map --tool to the flashinfer_bench_run_sanitizer sanitizer_types argument.
    # "all" → None (flashinfer-bench default: run all four); otherwise a 1-element list.
    sanitizer_types = None if tool == "all" else [tool]

    for idx, wl_entry in indexed:
        wl = wl_entry["workload"]
        uuid = wl["uuid"]
        axes = wl["axes"]
        axes_str = ", ".join(f"{k}={v}" for k, v in sorted(axes.items()))

        print(f"\nSanitizing workload {idx}: {uuid[:8]}...")
        print(f"  Axes: {axes_str}")
        print(f"  Tool: {tool}")
        print(f"  GPU:  {_GPU_TYPE}")
        print()

        san_kwargs = {
            "sanitizer_types": sanitizer_types,
            "timeout": timeout,
        }
        if max_lines > 0:
            san_kwargs["max_lines"] = max_lines

        result = run_sanitizer_remote.remote(solution_blob, uuid, san_kwargs)
        print(summarize_sanitizer_noise(result))

        # Auto-save sanitizer output for later inspection.
        san_dir = PROJECT_ROOT / "sanitizer"
        san_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data = {
            "timestamp": datetime.now().isoformat(),
            "workload_index": idx,
            "workload_uuid": uuid,
            "axes": dict(axes),
            "tool": tool,
            "backend": f"modal-{_GPU_TYPE.lower()}",
            "output": result,
        }
        out_path = san_dir / f"w{idx}_{tool}_{timestamp}.json"
        out_path.write_text(json.dumps(data, indent=2))
        print(f"\nSanitizer output saved to: {out_path}", file=sys.stderr)
