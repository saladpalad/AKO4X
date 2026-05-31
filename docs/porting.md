# Porting AKO4X to a different benchmark

AKO4X ships wired to **FlashInfer-Bench** as its default benchmark, but the
benchmark is a swappable component (that's the **X**). This is the complete
checklist for pointing AKO4X at a different benchmark. It's written to be
followed end-to-end by a coding agent (or a human) in one pass.

If you only want the short version: **rewrite one file** (`scripts/benchmark_adapter.py`)
so its plain-data functions resolve to your benchmark, rewrite the `benchmark`
skill and `evaluation.toml`, swap the dependency. The rest of this page is the
detail behind that.

## Design philosophy: ports adapt to AKO, not the other way around

A few principles that shape the rest of this guide:

- **The contract is FIB-derived by origin.** AKO grew up around FlashInfer-Bench,
  and the contract carries that history: `definition` / `workload` / `uuid` /
  `axes` vocabulary, a `STATUS_PASSED = "PASSED"` string literal, a
  `{definition: {uuid: {...}}}` normalized result shape, FIB's Trace-envelope as
  the on-disk `workloads.jsonl` form, a handful of hardcoded values in
  `bench_utils.py`. We don't pretend otherwise.

- **Ports adapt to the contract, not the other way around.** A port usually
  lands in two places: rewrite `scripts/benchmark_adapter.py` to resolve to
  your benchmark's runtime, and — if your dataset layout differs from
  `definitions/<cat>/<op>.json` + `workloads/<cat>/<op>.jsonl` — write a
  spawn-time transform on the `spawn.py` side that materializes a synthetic
  AKO-shaped tree from your native layout. For example, porting to a
  flat-pyfile benchmark like KernelBench (each operator a standalone `.py`
  module under `level<N>/`) would translate to ~80 lines in `spawn.py` that
  materialize a synthetic `definitions/level<N>/<op>.json` +
  `workloads/level<N>/<op>.jsonl` tree from the source — runs in ~1s at spawn,
  leaves the rest of the harness untouched.

- **This is deliberate.** Rebuilding AKO around a benchmark-neutral data model
  would multiply the surface to maintain (a neutral type layer plus N adapters)
  without removing the underlying assumptions — FIB-shaped scoring semantics,
  baseline-freshness logic, the per-uuid result lookup in
  `bench_utils.compute_score` — that the closed-loop campaign machinery relies
  on. Adapting at the edges is genuinely simpler than re-shaping the core.

- **Single-active-benchmark, by design.** One adapter is compiled in at a
  time. There's no per-spawn benchmark selector flag, no parallel adapter
  registry. The repo is a *fork-and-adapt* template; an AKO4FIB and an AKO4KB
  are siblings, not branches of the same runtime. See [CLAUDE.md](../CLAUDE.md)'s
  "single-active-benchmark assumption".

- **Multi-benchmark coexistence and a fully benchmark-neutral data model are
  deferred under YAGNI.** Both are real future directions; neither is on the
  critical path until a concrete use case forces them (e.g. an external user
  wanting AKO4FIB + AKO4KB from the same checkout). Until then this guide
  assumes the fork-and-adapt model.

## The swap surface

Everything benchmark-specific is concentrated in a small, named set of places:

| Concern | Lives in |
|---|---|
| Benchmark runtime (run / pack / profile / sanitize / cheat-check / discovery) | **`scripts/benchmark_adapter.py`** — the *only* module that imports the benchmark package |
| Benchmark behavior the agent reads (status enum, scoring, baseline rule, `config.toml` schema, workload / fresh-input model) | `templates/skills/benchmark/` (`SKILL.md` + `benchmark.md`) |
| Per-operator eval defaults + correctness tolerances | `templates/benchmark/evaluation.toml` |
| Which template dir is the active benchmark | `BENCHMARK_DIR` constant at the top of `spawn.py` |
| How operators / workloads / baselines are discovered from the dataset | `spawn.py` (`resolve_dataset`, `discover_operator`, `list_operators`, `discover_expert_baseline`) |
| Python dependency | `pyproject.toml` |

The scoring math (`compute_score`), baseline caching (`load_baseline` /
`save_baseline`), workload filtering, trajectory tracking, the A/B + variance
methodology, and every per-DSL kernel-writing skill are **benchmark-agnostic** —
you should not need to touch them. They only ever see `str` / `list[str]` /
`dict`, never a benchmark type.

## The adapter contract: what you must implement

The adapter exposes **8 public functions**. You do **not** have to implement all
8. They split into three groups by how load-bearing they are:

### The solution blob is opaque and porter-defined

Before the function list, the one concept that makes the whole seam make sense:
the **solution blob** is just a `str` (persisted as `solution.json`). Only three
functions ever look inside it — `pack` creates it, `solution_meta` reads its
identity, `run` reconstructs and executes it. Everything outside the adapter
(`bench_utils`, the runners, `pack_solution.py`) treats the blob as an **opaque
string** it carries around but never parses.

So **you define the blob format**. It can be your benchmark's native solution
object serialized to JSON, the raw kernel source with a header, a base64'd
tarball — anything text-serializable, as long as your own `pack` / `solution_meta`
/ `run` agree on it. This is the answer to *"my benchmark has no `solution.json`
concept"*: you still implement `pack`, but it emits whatever your `run` can
consume. (`pack` can't be a no-op — something has to produce what `run` reads —
but it is in no way tied to FlashInfer-Bench's format.)

### Required (the core `bench.sh` loop won't work without these)

These three drive correctness + scoring + baseline caching — i.e. every
`bash scripts/bench.sh` run:

- **`pack(source_dir, build_cfg, *, name, definition, author) -> blob:str`** —
  pack the kernel sources under `source_dir` into a solution blob. `build_cfg`
  carries `language` / `entry_point` / `destination_passing_style` (default
  `False`) / `target_hardware` (default `["cuda"]`). `destination_passing_style`
  is a FIB BuildSpec flag: `True` means the kernel writes into a caller-provided
  output tensor (the engine allocates and passes it in), `False` means the
  kernel allocates and returns its own output. If your benchmark has no analogue
  it's safe metadata to drop in `pack` and ignore in `run`; `bench_utils`
  hardcodes `False` for its reference pack (see *Things `bench_utils.py`
  hardcodes that your `pack` must cope with* below).
- **`solution_meta(blob) -> {"name", "definition", "author"}`** — read a blob's
  identity. The one sanctioned place that introspects a blob's internals, so every
  other caller can keep the blob opaque.
- **`run(blob, uuids, params, *, dataset_path, capture_logs=False,
  capture_autotune=False) -> dict`** — build a single-operator trace set from
  `uuids`, run the engine with `BenchmarkConfig(**params)` (or your benchmark's
  equivalent — `params` is validated by being splatted into your config
  constructor), and return the **normalized result dict** (below). **This one
  function owns trace-set assembly + the engine call + result extraction** — a
  port never has to make its native objects mimic another benchmark's attribute
  graph. `capture_autotune=True` returns `{"results": <dict>, "autotune_log": <str>}`.
  **`run` must raise on uuid mismatch**: if any uuid in `uuids` is missing from
  the dataset, raise — don't silently run a subset, which would corrupt the
  score/baseline. The FIB adapter enforces this (`benchmark_adapter.py:213-233`);
  a port must do the same.

  **What `params` actually contains.** `bench_utils` builds `params` from the
  child's `config.toml [benchmark]` and sends a fixed key set: `warmup_runs`,
  `iterations`, `num_trials`, `atol`, `rtol`, `required_matched_ratio`,
  `use_isolated_runner`, `timeout_seconds`, `profile_baseline`. If your
  benchmark's config constructor doesn't accept these names, **filter or alias
  inside `run`** — don't splat blindly, or you get `TypeError: unexpected kwarg`.
  Keys your benchmark has no analogue for are safe to drop (document which in
  the `benchmark` SKILL).

### Discovery (trivial; needed only if you keep the aux tools)

- **`list_workloads(dataset_path, definition) -> [{"uuid","axes"}, ...]`** —
  enumerate the operator's workloads in dataset order. Note `bench.sh` does **not**
  call this (it reads the spawn-time `docs/workloads.jsonl`); it's the workload
  enumerator for `profile.sh` / `sanitize.sh` / cheat-check. A few lines to write,
  so implement it unless you're stubbing all three aux tools below.

### Optional (auxiliary tools; safe to stub)

Each of these backs one auxiliary command. Stub any your benchmark can't support
and **only that command degrades — the core bench/score/baseline loop is
unaffected.** The stub must return the shape the caller expects (so it formats /
prints instead of crashing):

| Function | If unsupported, return | Command that degrades |
|---|---|---|
| `profile(blob, uuid, opts, *, dataset_path, env_pairs=None) -> str` | `"NCU profiling not supported for <benchmark>"` | `profile.sh` |
| `list_ncu_options() -> str` | `"NCU not supported for <benchmark>"` | `profile.sh --ncu-options` |
| `sanitize(blob, uuid, opts, *, dataset_path) -> str` | `"compute-sanitizer not supported for <benchmark>"` | `sanitize.sh` |
| `cheat_check(blob, uuids, *, dataset_path, n_iters=4) -> dict` | `{"status": "SKIPPED", "reason": "..."}` | the Modal cheat-check audit |

A stub is one `return` statement — no benchmark import, no boilerplate:

```python
def profile(blob, uuid, opts, *, dataset_path, env_pairs=None):
    return "NCU profiling not supported for <benchmark>"

def list_ncu_options():
    return "NCU not supported for <benchmark>"

def sanitize(blob, uuid, opts, *, dataset_path):
    return "compute-sanitizer not supported for <benchmark>"

def cheat_check(blob, uuids, *, dataset_path, n_iters=4):
    return {"status": "SKIPPED", "reason": "cheat-check not supported for <benchmark>"}
```

### The normalized result dict (`run`'s output)

`run` must return this shape — it's the contract the benchmark-agnostic scoring /
baseline code in `bench_utils` consumes:

```python
{definition_name: {workload_uuid: {
    "status": <str>,                 # one of the STATUS_* values
    "solution": <str>,
    "axes": {<axis>: <value>, ...},
    "latency_ms": <float>,           # present when PASSED
    "reference_latency_ms": <float>,
    "speedup_factor": <float>,
    "max_abs_error": <float|"NaN">,  # present when correctness ran
    "max_rel_error": <float|"NaN">,
    "error_log": <str>,              # present for non-PASSED workloads
    "log": <str>,                    # present with capture_logs
}}}
```

Inside `run`, flatten your benchmark's native result objects into this dict
however you like. `compute_score` (arithmetic mean of `speedup_factor` over PASSED
workloads) and the baseline I/O then work unchanged.

### Constants

Plain constants at the module top (no benchmark import): the `STATUS_*` strings
(your benchmark's per-workload outcomes, mirrored as plain strings);
`DATASET_PATH_ENV` (the env var `spawn.py` and `bench_utils.get_trace_set_path`
consult for the dataset path on the local backend — defaults to
`"AKO_DATASET_PATH"`) and `LEGACY_DATASET_PATH_ENV` (a fallback env-var name
checked second; FIB-era setups exported `FIB_DATASET_PATH` and spawned children
keep working without re-exporting — set it to `""` if you have no legacy
exports to honor); `NCU_NVTX_RANGE` (the NVTX range your profiler wraps the
call in — surfaced for the `profiler-ncu` skill's "no kernels profiled"
diagnosis); and the Modal image pins `MODAL_IMAGE_REGISTRY` / `MODAL_PYTHON` /
`MODAL_PACKAGE_PIN` / `MODAL_EXTRA_PIN`.

**`STATUS_PASSED` is load-bearing as a literal string.** `bench_utils.compute_score`
filters PASSED workloads by `result["status"] == "PASSED"` — a string literal,
not the constant. **`STATUS_PASSED` MUST equal the string `"PASSED"`.** Other
`STATUS_*` values aren't read by `bench_utils`; they just need to be unique
informative strings that show up in `error_log` / display. If your benchmark's
native success enum is `OK` or `Success`, map it to `"PASSED"` inside `run`.

If you stub `profile`, `NCU_NVTX_RANGE` can be any non-empty string (it's only
quoted in the profiler-ncu skill's diagnosis text). If you have no Modal image
for your benchmark, the `MODAL_*_PIN` constants must still be present (the
launchers import them at module load) — empty strings work, but document that
the Modal backend is unsupported so a porter doesn't run into a confusing
image-build failure.

### Keep the function-local import discipline

**Nothing at module top imports the benchmark** — only the plain constants live
there, and every function imports what it needs from the benchmark at call time.
This keeps the adapter (and therefore every launcher that imports it) importable
even on a host *without* the benchmark package installed — e.g. the macOS
modal-only setup in `compat/overrides-macos.txt`, which drops `flashinfer-python`.
Only an actual `run` / `pack` / `profile` / `sanitize` / `cheat_check` *call*
loads the benchmark. And because no benchmark types cross the public surface,
Modal cloudpickles only builtins — there's no by-reference module identity to
resolve in the container.

## Minimum viable port

The smallest swap that gives you a working benchmark loop:

1. Implement `run`, `pack`, `solution_meta`, and `list_workloads` over your
   benchmark; **stub** `profile`, `list_ncu_options`, `sanitize`, `cheat_check`
   (return the "not supported" shapes above).
2. Rewrite the `benchmark` skill (step 2 below) and `evaluation.toml` (step 3).
3. Swap the dependency in `pyproject.toml` (step 5); adjust the `spawn.py` dataset
   functions only if your dataset layout differs (step 4).

Result: `bash scripts/bench.sh` runs end-to-end — correctness, scoring, baseline
caching, A/B and variance all work — while `profile.sh` / `sanitize.sh` /
cheat-check print "not supported". Implement the Tier-2 functions later when you
want the profiling / sanitizer / audit tools too.

## Swap checklist

### 1. Rewrite `scripts/benchmark_adapter.py`

Re-implement the public functions per **The adapter contract** above so they
resolve to your benchmark's runtime. The module's own docstring is the canonical
spec for the surface — keep it in sync. Required: `pack` / `solution_meta` /
`run`; trivial: `list_workloads`; stubbable: `profile` / `list_ncu_options` /
`sanitize` / `cheat_check`.

### 2. Rewrite the `benchmark` skill — `templates/skills/benchmark/`

Replace the body of `SKILL.md` + `benchmark.md` with your benchmark's reference:
status enum, scoring formula, reference/baseline rule, `config.toml` schema,
workload + input-freshness model, the NCU NVTX range, and any silent-skip failure
mode. Keep the skill **named `benchmark`** (see "Keeping the slot name") so the
generic DSL skills that point at "the `benchmark` skill" keep resolving.

### 3. Rewrite `templates/benchmark/evaluation.toml`

`[default]` = your benchmark's eval params; per-`op_type` sections = tolerance
overrides. `spawn.py` merges these into each child's `config.toml [benchmark]`.

**Required key names.** `bench_utils._load_benchmark_config()` reads a fixed key
set from `[benchmark]`: `baseline_iterations`, `solution_iterations`,
`num_trials`, `warmup_runs`, `atol`, `rtol`, `required_matched_ratio`,
`use_isolated_runner`, `timeout_seconds`. **Your `[default]` MUST emit these
names**, even if your benchmark calls them differently internally — bench_utils
won't see renamed keys and silently uses its defaults (no error, just wrong
numbers). If your benchmark has no analogue for a key (e.g. per-workload
tolerance instead of global), set it to a benign value (`0`, `1`) and ignore it
in your adapter's `run`.

### 4. Update the dataset-layout functions in `spawn.py` (only if your layout differs)

`resolve_dataset` / `discover_operator` / `list_operators` /
`discover_expert_baseline` encode the FlashInfer-Trace directory layout
(`definitions/<cat>/<op>.json`, `workloads/<cat>/<op>.jsonl`,
`solutions/<author>/<cat>/<op>/<solution>.json`). If your benchmark's dataset
is laid out differently, adjust these four functions. This is the one
benchmark-specific surface *outside* the adapter (it's parent-side and never
copied into children).

**Recommended pattern: a spawn-time transform.** If your benchmark's native
layout doesn't match `definitions/<cat>/<op>.json` + `workloads/<cat>/<op>.jsonl`,
the clean fix is **not** to thread shape-translation through
`benchmark_adapter.py` (which would entangle the adapter with two layouts).
Instead, materialize a synthetic AKO-shaped tree at spawn time from your
native source. Worked sketch — porting to KernelBench (flat
`<repo>/KernelBench/level<N>/<problem>.py` tree): add an
`ensure_dataset_synth()` helper in `spawn.py` that reads the flat source tree
and writes a synthetic
`<cache>/definitions/level<N>/<op>.json` + `<cache>/workloads/level<N>/<op>.jsonl`
under a cache dir alongside `spawn.py`, then have `resolve_dataset` return
that cache root. Downstream consumers (`list_operators` / `discover_operator` /
`bench_utils._load_workloads` / the adapter) all see the AKO shape; the native
shape stays at the spawn-time boundary. ~80 lines, ~1s at spawn, idempotent
(rebuild only when the source file-count changes).

**`discover_expert_baseline` is optional.** If your benchmark has no notion of
an "expert solution per operator" (e.g. KernelBench, where the PyTorch reference
IS the baseline), stub it to always return `None` — bench_utils gracefully falls
through to the reference baseline. Default behavior scans
`solutions/<author>/<op_type>/<op>/*.json` across all authors and picks the
lex-first match (a single-author fork like `mlsys26-contest` keeps resolving
because its `baseline` author sorts ahead of model-keyed authors like `llama1b`);
pass `--baseline <path>` explicitly to override when multiple authors contribute.

**`docs/workloads.jsonl` envelope shape.** `bench_utils._load_workloads()` reads
each JSONL line as `{"workload": {"uuid": ..., "axes": {...}, ...}}` (the
FIB-Trace `Trace`-envelope shape, not the raw workload shape). If your
benchmark's native workload format differs, your spawn.py must transform at
write time — wrap each line in a `{"workload": <orig>}` envelope before writing
to the child. The envelope shape is an implicit contract between spawn-time
output and bench_utils' reader; porting.md doesn't document it elsewhere.

**Treat the dataset path as read-only.** Any synthetic shim data you generate
(e.g. emitting AKO-shaped definitions / workloads from a flat-pyfile benchmark
like KernelBench) should live under the AKO repo root or an explicit cache dir,
not under the benchmark's source tree.

### 5. Update `pyproject.toml`

Replace the `flashinfer-bench` dependency with your benchmark's package (and
adjust the Linux-only `flashinfer-python` / toolchain extras as needed).

### 6. (Modal backend only) Update the image pins **and the image-build chain**

Point `MODAL_IMAGE_REGISTRY` + the `MODAL_*_PIN` constants in the adapter at an
image/package that has your benchmark installed. Each Modal runner builds its
image from these and `add_local_file`s `benchmark_adapter.py` into
`/root/project/scripts/` (the remote bodies import only the adapter); keep that if
your adapter is imported remotely.

**The pins are not the whole story.** `scripts/run_modal.py` /
`run_modal_profile.py` / `run_modal_sanitize.py` each build a `modal.Image`
with a `.pip_install(...)` chain that includes flashinfer-specific extras
(deep-gemm patches, cutlass-headers shims, etc.). When porting, strip the chain
to your benchmark's actual install recipe. Likewise replace the hardcoded Modal
identifiers that name the active benchmark:

- `modal.App("flashinfer-bench")` in `run_modal.py` — rename to your port's app
- `modal.Volume.from_name("flashinfer-trace", ...)` in all three runners —
  rename to your port's data volume

If your benchmark has **no public Modal-compatible image**, set the
`MODAL_*_PIN` constants to empty strings and document the unsupported backend in
your `benchmark` SKILL — the local backend can still work.

## What you do NOT touch

- The generic DSL skills (`triton` / `cuda` / `cute-dsl` / `tilelang` / `cpp`)
  and `profiler-ncu` / `sanitizer` — kernel-writing knowledge, benchmark-agnostic.
- The `bench` skill — generic noise-aware methodology (A/B, variance, filters).
- The runners (`run_local*.py` / `run_modal*.py`), `pack_solution.py`, and
  `cheat_check_modal.py` — they go through the adapter and only pass/receive plain
  data; they need editing only if you add or remove an adapter function.
- `bench_utils.py`'s scoring (`compute_score`) and baseline (`load_baseline` /
  `save_baseline`) logic.
- `master/` (the closed-loop orchestration — see [closed-loop.md](closed-loop.md)).

### …with these caveats: the benchmark name leaks into a few "untouched" places

Two narrow exceptions to the rules above — places where the active benchmark's
*name* (not its types or behavior) shows up and needs updating during a port:

- The `bench` skill's `SKILL.md` body has one line naming the active benchmark
  (it points at the `benchmark` skill for benchmark-specific details). Update
  the name; the methodology body stays.
- DSL skills under `templates/skills/{cute-dsl,...}` occasionally reference
  paths from FIB's install layout (e.g. `<flashinfer-install>/data/cutlass/...`).
  Grep for the old benchmark name in `templates/skills/` and fix path
  references — the kernel-writing knowledge stays, only the FIB-specific paths
  rotate.
- `scripts/CLAUDE.md` describes the adapter as "the sole `flashinfer_bench`
  importer" — comment-only, rotate the name.

### Things `bench_utils.py` hardcodes that your `pack` must cope with

`bench_utils._pack_reference()` and `_pack_snapshot()` call `adapter.pack(...)`
with literal `build_cfg` values:

```python
{"language": "triton", "entry_point": "kernel.py::run",
 "destination_passing_style": False}
```

These are FIB-era defaults baked into `bench_utils.py`, which is **FROZEN**
(`compute_score` / baseline math). Your `pack` must accept this dict — either
honor it, treat the keys as advisory metadata, or detect "this is the reference
pack" via the name suffix (FIB convention: `-reference-baseline`) and re-tag.
If your benchmark requires faithful `language` / `entry_point` values for the
reference path, you'll have to alias inside `pack`; you can't fix it in
bench_utils.

## Keeping the slot name

The template dir (`templates/benchmark/`) and skill (`templates/skills/benchmark/`)
use the stable, benchmark-neutral name **`benchmark`** on purpose: the generic
skills reference "the `benchmark` skill", and `spawn.py` reads
`BENCHMARK_DIR = "benchmark"`. Keep that name and you touch zero generic skills.
If you insist on renaming the slot to your benchmark's name, rename both dirs,
update `BENCHMARK_DIR`, and grep the generic skills for `` `benchmark` `` to fix
the pointers.

## Frozen-for-comparability

Within a single closed-loop campaign, the scoring formula and baseline freshness
rule are **frozen** so the master can compare rounds (see
`templates/closed-loop-scope.md` and the `benchmark` skill's "Frozen for bench
comparability"). Changing scoring/baseline behavior means starting a fresh
campaign with re-measured baselines — that's expected when you swap benchmarks.

## Verify the port

Steps 1–3 and 6 are static / spawn-time checks and run on any host (no GPU
required). Steps 4–5 invoke `bash scripts/bench.sh` / `profile.sh` /
`sanitize.sh`, which compile and execute the kernel — they require a GPU host
(local NVIDIA or the Modal backend if you wired one up in step 6 above).

1. **Sole importer**: `grep -rn "<your_benchmark_pkg>" scripts/` returns hits only
   in `benchmark_adapter.py`, the same way `flashinfer_bench` does today.
2. **Import-safe launchers**: importing each `run_modal*.py` on a no-GPU host must
   not fail at module load — and, since no benchmark import lives at module top, it
   must not even require the benchmark package to be installed (only an actual
   run/pack/profile/sanitize/cheat-check call loads it).
3. **Spawn**: `python spawn.py --operator <op> --backend modal --gpu <g> --name port_smoke`
   populates a child `config.toml [benchmark]` from `templates/benchmark/evaluation.toml`
   and copies `scripts/benchmark_adapter.py` into the child.
4. **Run** (the required path): in the child, `bash scripts/bench.sh --first 1`
   (compile/correctness), then a full `bash scripts/bench.sh`.
5. **Stub path** (if you stubbed Tier-2): `bash scripts/profile.sh --index 0`
   prints your "not supported" string and exits cleanly — and a full `bash
   scripts/bench.sh` still produces a score. If you implemented them, exercise
   `profile.sh --index 0` and `sanitize.sh --index 0` once each instead.
6. **Final name-leak grep.** Comments and skill bodies aren't load-bearing for
   behavior, but they keep step 1's grep honest as a completeness signal:
   ```
   grep -rn "<old_benchmark_pkg>" docs/ templates/ *.md README.md
   grep -rn 'modal.Volume.from_name(' scripts/
   grep -rn 'modal.App('              scripts/
   ```
   The first should be reduced to references-by-name in porting.md / README. The
   Modal identifiers should name your port, not the old benchmark.
