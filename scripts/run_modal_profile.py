"""
FlashInfer-Bench Profiler Runner (Modal Backend).

Profiles a solution on specific workloads using NVIDIA Nsight Compute (NCU)
on Modal cloud GPU.

Usage:
    modal run scripts/run_modal_profile.py --list
    modal run scripts/run_modal_profile.py --index 0
    modal run scripts/run_modal_profile.py --index 0 --set full --page raw
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

# Read config locally; on Modal remote container config.toml doesn't exist
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

app = modal.App("flashinfer-bench-ncu-profile")

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
# Modal remote functions
# ---------------------------------------------------------------------------

@app.function(image=image, gpu=f"{_GPU_TYPE}:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_ncu_remote(
    blob: str,
    workload_uuid: str,
    ncu_kwargs: dict,
    env_pairs: dict = None,
) -> str:
    """Run NCU profiling inside Modal container and return output.

    ``env_pairs`` are set on ``os.environ`` BEFORE flashinfer_bench imports
    the kernel module, so module-level env gates like
    ``_NO_GRAPH = bool(os.environ.get("NO_GRAPH"))`` evaluate against the
    caller's requested value. Primary use case: toggle a kernel's
    NO_GRAPH escape hatch so NCU sees every cuLaunchKernel directly
    instead of a single opaque cuGraphLaunch.
    """
    import sys as _sys
    _sys.path.insert(0, "/root/project/scripts")
    from benchmark_adapter import profile

    return profile(blob, workload_uuid, ncu_kwargs,
                   dataset_path=TRACE_SET_PATH, env_pairs=env_pairs)


@app.function(image=image, gpu=f"{_GPU_TYPE}:1", timeout=120)
def list_ncu_options_remote() -> str:
    """Query `ncu --list-sets` + `--list-sections` on the Modal image.

    Run this on Modal (not locally) because the Modal image has `ncu`
    pre-installed; most local dev hosts don't. Sections vary by ncu
    version, so the Modal answer is what the agent actually needs.
    """
    import sys as _sys
    _sys.path.insert(0, "/root/project/scripts")
    from benchmark_adapter import list_ncu_options
    return list_ncu_options()


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

def _load_config():
    """Load config from config.toml."""
    config_path = PROJECT_ROOT / "config.toml"
    if not config_path.exists():
        sys.exit("Error: config.toml not found")
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def _load_workloads_jsonl():
    """Load workloads from docs/workloads.jsonl (local file, no volume needed)."""
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


def _list_ncu_options():
    """List available NCU sets and sections — queries the Modal image's ncu."""
    import shutil as _shutil
    if _shutil.which("ncu"):
        # Local ncu available (fast path): reports local-host sections.
        # Caveat: may differ from the Modal image's version.
        from scripts.benchmark_adapter import list_ncu_options
        print(list_ncu_options())
    else:
        # Fallback: query on Modal where ncu is installed.
        print("(Querying Modal image — ncu not available locally...)")
        print(list_ncu_options_remote.remote())


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    index: str = None,
    list: bool = False,
    ncu_options: bool = False,
    set: str = "detailed",
    page: str = "details",
    kernel_name: str = "",
    sections: str = "",
    timeout: int = 180,
    max_lines: int = -1,
    group: str = None,
    exclude_group: str = None,
    env: str = "",
):
    """Profiler — NCU for specific workloads via Modal.

    --env KEY=VAL[,KEY2=VAL2]: set environment variables inside the Modal
    container before the kernel module is imported. Primary use: toggle
    a kernel's NO_GRAPH escape hatch so NCU sees individual kernel
    launches instead of one opaque cuGraphLaunch. Example:
        bash scripts/profile.sh --index 40 --env NO_GRAPH=1
    """
    # Line-buffer stdout so progress prints stream to caller when piped.
    sys.stdout.reconfigure(line_buffering=True)

    import os

    from scripts.bench_utils import parse_int_filter, tail_truncate_output

    # Parse filter values
    indices = parse_int_filter(index) if index else None
    group_values = parse_int_filter(group) if group else None
    exclude_group_values = parse_int_filter(exclude_group) if exclude_group else None

    # Parse --env KEY=VAL[,KEY=VAL…] into a dict. Empty default = no overrides.
    env_pairs = {}
    if env:
        for pair in env.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                print(f"Error: --env entries must be KEY=VAL, got {pair!r}",
                      file=sys.stderr)
                sys.exit(1)
            k, v = pair.split("=", 1)
            env_pairs[k.strip()] = v.strip()

    if ncu_options:
        _list_ncu_options()
        return

    if list:
        entries = _load_workloads_jsonl()
        _list_workloads(entries, indices, group_values, exclude_group_values)
        return

    if index is None:
        print("Usage: modal run scripts/run_modal_profile.py --index <N>")
        print("       modal run scripts/run_modal_profile.py --list")
        print("       modal run scripts/run_modal_profile.py --ncu-options")
        sys.exit(1)

    os.environ.setdefault("TRITON_CACHE_DIR", str(PROJECT_ROOT / ".triton_cache"))

    # Load and filter workloads
    entries = _load_workloads_jsonl()
    indexed = _filter_workloads(entries, indices, group_values, exclude_group_values)
    if not indexed:
        print("Error: No workloads match the specified filters.", file=sys.stderr)
        sys.exit(1)

    # Pack solution
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_blob = pack_solution().read_text()

    for idx, wl_entry in indexed:
        wl = wl_entry["workload"]
        uuid = wl["uuid"]
        axes = wl["axes"]
        axes_str = ", ".join(f"{k}={v}" for k, v in sorted(axes.items()))

        print(f"\nProfiling workload {idx}: {uuid[:8]}...")
        print(f"  Axes: {axes_str}")
        print(f"  Set: {set}, Page: {page}")
        print(f"  GPU: {_GPU_TYPE}")
        if kernel_name:
            print(f"  Kernel filter: {kernel_name}")
        if env_pairs:
            print(f"  Env: {env_pairs}")
        print()

        ncu_kwargs = {"set": set, "page": page, "timeout": timeout}
        if kernel_name:
            ncu_kwargs["kernel_name"] = kernel_name
        if sections:
            ncu_kwargs["sections"] = [s.strip() for s in sections.split(",")]
        # Intentionally do NOT forward max_lines — flashinfer_bench's truncation
        # is head-biased and drops the metrics tables we actually want. Apply
        # tail-truncation locally instead (see bench_utils.tail_truncate_output).

        result = run_ncu_remote.remote(
            solution_blob, uuid, ncu_kwargs, env_pairs=env_pairs or None,
        )
        if max_lines > 0:
            print(tail_truncate_output(result, max_lines))
        else:
            print(result)

        # Auto-save NCU profile output
        profiles_dir = PROJECT_ROOT / "profiles"
        profiles_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        profile_data = {
            "timestamp": datetime.now().isoformat(),
            "workload_index": idx,
            "workload_uuid": uuid,
            "axes": dict(axes),
            "ncu_set": set,
            "ncu_page": page,
            "kernel_filter": kernel_name or None,
            "backend": f"modal-{_GPU_TYPE.lower()}",
            "output": result,
        }
        profile_path = profiles_dir / f"w{idx}_{timestamp}.json"
        profile_path.write_text(json.dumps(profile_data, indent=2))
        print(f"\nProfile saved to: {profile_path}", file=sys.stderr)
