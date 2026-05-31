"""Modal cheat-check: vary input values across iters, verify outputs change.

Detects (per AKO4X/reference TRAPS.md cross-references and flashinfer-bench #414):
  - CuTe DSL `@cute.kernel.launch()` skipped on torch.cuda.graph.replay() (stale output)
  - Bare `<<<grid, block>>>` chevron without `at::cuda::getCurrentCUDAStream()`
    (legacy stream-0 silently outside graph capture)
  - Result memoization keyed on input pointers/shapes (cached output return)
  - Stream injection where output isn't actually produced per call

Approach (modeled on https://github.com/flashinfer-ai/flashinfer-bench/pull/413):
keep tensor pointers stable across iterations (so honest address-keyed CUDA Graph
caches replay normally), but mutate tensor *values* in-place between iters. An
honest kernel produces a fresh output each iter; a cheating kernel returns
byte-identical output regardless of input changes.

Usage:
    modal run AKO4X/scripts/cheat_check_modal.py --kernel-dir /path/to/<repo>/<kernel>

Pass criterion: every consecutive iteration's output hash differs from the
previous iteration's, on every probed workload.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import modal

try:
    import tomllib
except ImportError:  # py<3.11 fallback
    import tomli as tomllib

# v1 path-discovery convention (see scripts/CLAUDE.md). Parent-only script:
# this is the AKO4X root, on sys.path so it can reuse the canonical packer
# instead of re-implementing it (drifted once — commit 8bcd3f2).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Modal image pins come from the adapter (the sole flashinfer_bench importer).
# This module-level import runs both locally (PROJECT_ROOT on sys.path →
# `scripts.` resolves) and in the container, where Modal re-imports this module
# to find the remote function but PROJECT_ROOT is "/" so `scripts` isn't
# importable — the adapter is in the image at /root/project/scripts
# (add_local_file below), so fall back to the bare name. Re-raise non-`scripts`
# errors (e.g. flashinfer_bench genuinely absent locally).
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

app = modal.App("ako4x-cheat-check")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=False)
TRACE_SET_PATH = "/data"

# Base image aligned with bare-metal eval (flashinfer-ci-cu132). See
# run_modal.py for rationale + rollback. Mirrors run_modal_sanitize.py's
# block (same base + pinned flashinfer-bench, so this audit builds kernels
# under the same toolchain the scored bench does) plus nvidia-cutlass-dsl
# (the audit must build the CuTe-DSL kernels whose @cute.kernel.launch()
# graph-replay skip it exists to detect).
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
    .run_commands("apt-get update && apt-get install -y git clang && pip install wheel")
    .run_commands("pip install --no-deps nvidia-cutlass-dsl>=4.3.4")
    .add_local_file(str(PROJECT_ROOT / "scripts" / "benchmark_adapter.py"), "/root/project/scripts/benchmark_adapter.py")
)


# Plain in-container helper — NOT @app.function. The Modal surface is the
# CheatCheck class below; per-invocation GPU override needs Cls.with_options()
# (modal.Function has no .with_options() — Codex review BLOCK). This body runs
# remotely because CheatCheck.run (a @modal.method) calls it inside the container.
def cheat_check_remote(
    solution_json: str,
    n_iters: int = 4,
    n_workloads_probe: int = 2,
) -> dict:
    """Run the varying-inputs cheat check inside Modal.

    Probe-slice selection (smallest + largest, or evenly-spaced) is plain
    indexing and stays here; the build + gen-inputs + mutate + hash loop is the
    FIB-coupled part and lives in benchmark_adapter.cheat_check.
    """
    import sys as _sys
    _sys.path.insert(0, "/root/project/scripts")
    from benchmark_adapter import cheat_check, list_workloads, solution_meta

    definition = solution_meta(solution_json)["definition"]
    entries = list_workloads(TRACE_SET_PATH, definition)
    if not entries:
        return {"status": "ERROR", "reason": f"no workloads for {definition!r}"}

    # Pick a representative slice: smallest + largest (cheats usually surface
    # consistently; testing both extremes catches shape-conditional skips).
    n = len(entries)
    if n <= n_workloads_probe:
        probe = entries
    elif n_workloads_probe <= 2:
        probe = [entries[0], entries[-1]]
    else:
        # Evenly-spaced indices across the full workload range so probes cover
        # shape-conditional kernel paths (e.g. fast vs slow-path).
        idx = [round(i * (n - 1) / (n_workloads_probe - 1)) for i in range(n_workloads_probe)]
        probe = [entries[i] for i in idx]

    probe_uuids = [e["uuid"] for e in probe]
    return cheat_check(solution_json, probe_uuids, dataset_path=TRACE_SET_PATH, n_iters=n_iters)


@app.cls(image=image, gpu="B200:1", timeout=1800,
         volumes={TRACE_SET_PATH: trace_volume})
class CheatCheck:
    """Modal surface for the cheat check. gpu="B200:1" is only the default;
    main() overrides it per kernel via CheatCheck.with_options(gpu=...) so the
    audit runs on the campaign's locked hardware."""

    @modal.method()
    def run(self, solution_json: str, n_iters: int = 4,
            n_workloads_probe: int = 2) -> dict:
        return cheat_check_remote(solution_json, n_iters, n_workloads_probe)


@app.local_entrypoint()
def main(kernel_dir: str, n_iters: int = 4, n_workloads_probe: int = 2):
    """Local entrypoint. Packs the kernel locally then runs the check on Modal."""
    kdir = Path(kernel_dir).resolve()
    if not (kdir / "config.toml").exists():
        print(f"ERROR: {kdir}/config.toml not found", file=sys.stderr)
        sys.exit(2)

    # Audit on the campaign's locked GPU — the kernel's config.toml [build]
    # gpu is authoritative (spawn.py always writes it; resolve_gpu makes it
    # required for the modal backend). Parent-only check: refuse to guess,
    # since a wrong/missing GPU silently audits on the wrong hardware and
    # invalidates the result (Codex FIX-NOW).
    with open(kdir / "config.toml", "rb") as f:
        _gpu = tomllib.load(f).get("build", {}).get("gpu")
    if not _gpu:
        print(f"ERROR: {kdir}/config.toml has no [build] gpu — refusing to "
              f"guess; this audit must run on the campaign's locked hardware",
              file=sys.stderr)
        sys.exit(2)
    gpu = _gpu.upper()

    print(f"[local] packing solution from {kdir}")
    from scripts.pack_solution import build_solution

    blob, _, _ = build_solution(root=kdir)
    sol_json = blob
    print(f"[local] dispatching to Modal {gpu} (n_iters={n_iters})")
    result = CheatCheck.with_options(gpu=f"{gpu}:1")().run.remote(
        sol_json, n_iters, n_workloads_probe
    )

    print()
    print("=" * 64)
    print(f"OVERALL: {result['status']}   ({result.get('definition', '?')})")
    print("=" * 64)
    for wl_id, e in result.get("workloads", {}).items():
        ax = e.get("axes", {})
        ax_str = ", ".join(f"{k}={v}" for k, v in ax.items())
        line = f"  workload {wl_id}  [{ax_str}]: {e.get('status', '?')}"
        if "unique_hashes" in e:
            line += f"  ({e['unique_hashes']}/{e.get('n_iters')} unique)"
        print(line)
        if e.get("reason"):
            print(f"    reason: {e['reason']}")
    if result.get("reason"):
        print(f"reason: {result['reason']}")

    if result.get("status") != "PASS":
        sys.exit(1)
