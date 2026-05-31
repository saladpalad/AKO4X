# scripts/ — shared runtime core

This directory holds the **shared runtime** that backs the kernel-optimization SKILLs. Multiple SKILLs (`bench`, `profiler-ncu`, `sanitizer`) consume the same runtime functions — notably `bench_utils.py` for workload loading, baseline freshness, scoring.

## Modify contract (closed-loop)

- **Editing `.claude/skills/<name>/SKILL.md` or supporting docs ≠ editing runtime.** SKILL.md describes *what* to do; `scripts/<file>.py` implements *how*. Editing one without the other is fine if the change is purely descriptive (e.g., correcting a misleading description, adding a workflow tip). Behavior changes need both.
- **Editing runtime API → must be paired with `## COUPLED references` updates** in every SKILL that lists the changed file. Consistency gets checked across runs.
- **Frozen-for-comparability behavior** lives in `bench_utils.py` and is enforced across runs:
  - `compute_score()`
  - `load_baseline()` / `save_baseline()` freshness logic
  - per-operator tolerance semantics (driven by `config.toml`'s `[benchmark]` table, populated at spawn time)
  - Patches touching these are rejected across runs. To change scoring → start fresh + re-measure baselines.
- **Non-frozen runtime is mutable**: error-message text, output formatting, debug helpers, profile/sanitize wrappers, workload-filter syntax. Edit freely; no scope check.

## Files

| File | Used by SKILLs | Notes |
|---|---|---|
| `benchmark_adapter.py` | (all, indirectly) | **The sole `flashinfer_bench` importer** — the benchmark seam. Exposes **plain-data functions** (`run` / `pack` / `solution_meta` / `list_workloads` / `profile` / `list_ncu_options` / `sanitize` / `cheat_check`): only `str`/`list`/`dict` cross it, no benchmark types. Holds the Modal-image + dataset-env constants. Porting to another benchmark = rewrite this one file. |
| `bench_utils.py` | bench, profiler-ncu, sanitizer | Shared core — workload loading, baseline I/O, scoring. Frozen segments above. Reaches the benchmark only through `benchmark_adapter`. |
| `run_local.py` / `run_modal.py` | bench | Backend dispatch. |
| `run_local_profile.py` / `run_modal_profile.py` | profiler-ncu | NCU wrappers. |
| `run_local_sanitize.py` / `run_modal_sanitize.py` | sanitizer | compute-sanitizer wrappers. |
| `cheat_check_modal.py` | (parent-only, NOT shipped to child) | Independent correctness audit invoked as `modal run /path/to/parent/scripts/cheat_check_modal.py`. Listed here for cross-reference; the file does not exist inside a spawned child. |
| `pack_solution.py` | (spawn-time / submit) | Not closed-loop-touched. |
| `diff_trajectory.py` | (general trajectory analysis) | `bash scripts/diff.sh` shim. |

`PROJECT_ROOT = Path(__file__).parent.parent` is the v1 path-discovery convention. Future v2 may rewrite this to walk-up `config.toml` so SKILL atomic boundaries can include runtime; that's out of scope for the closed-loop minimal prototype.
