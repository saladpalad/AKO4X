# Benchmarking Reference

How to drive the bench harness from the AKO4X shim — commands, modes, filters, output interpretation, noise methodology. The benchmark's own specifics (scoring, status enum, workload model, baseline rule, `config.toml`) live in the `benchmark` skill.

## Iteration Tuning

The first benchmark run profiles the reference implementation, which can be very slow (pure Python on large workloads). Always use a timeout, especially on the first run:

```bash
timeout 600 bash scripts/bench.sh    # 10-minute timeout
```

Iteration counts are configured in the `[benchmark]` section of `config.toml` — full schema in the `benchmark` skill. To run fewer workloads when benchmarking is too slow, use `--index`, `--group`, `--smoke`, or `--first N` (see *Workload Filtering* below).

## Workload Filtering

Use filters to run a subset of workloads for fast iteration. All filter modes skip baseline caching — partial results are not saved to `baseline.json`. Filters compose in order: `--index` → `--group` → `--exclude-group` → `--smoke` → `--first`.

> ⚠️ **Smoke-test noise warning.** Running on ≤3 workloads typically shows **2-3x variance** vs the full-bench mean due to cold-cache / Triton JIT / container cold-start jitter (Modal: tenant + thermal; local: driver + cuBLAS lazy-init). Use small-workload filters for **compile/correctness checks only** — do **not** use them to decide whether a code change is a signal or regression. For that, run the full bench or use `--variance-check N`. The bench prints an explicit warning at the end of runs with ≤3 workloads; heed it.

### `--first N` — Quick test

```bash
bash scripts/bench.sh --first 1 --label "debug"   # single workload, fast compile check
bash scripts/bench.sh --first 3                     # first 3 workloads
```

Useful for quickly validating that your kernel compiles and produces correct output.

### `--smoke` — One workload per group bucket

```bash
bash scripts/bench.sh --smoke                       # e.g. 5 workloads for group_axis={1,2,6,7,8}
bash scripts/bench.sh --smoke --exclude-group 1     # skip one bucket
```

Picks the first workload for each distinct value of the operator's `group_axis` (see `docs/definition.json` → `variable_axis`). For an operator with group axis `num_tokens` and values `{1, 2, 6, 7, 8}`, `--smoke` runs 5 workloads total — one per bucket — giving correctness coverage across all shapes at roughly `5/23` of full-bench cost. If the operator has no variable axis, `--smoke` falls back to the first workload only (same as `--first 1`). Composes with other filters; applied after `--group`/`--exclude-group`.

### `--group VALUES` — Filter by group axis

```bash
bash scripts/bench.sh --group 8                    # only workloads where group axis = 8
bash scripts/bench.sh --group 8,16                 # group axis = 8 or 16
bash scripts/bench.sh --group 32-901               # group axis in range [32, 901]
bash scripts/bench.sh --group 8,16,128-512         # mix of exact values and ranges
bash scripts/bench.sh --group 8 --first 2          # first 2 workloads from group 8
```

Filters workloads by the variable axis (the axis that varies across workload groups, e.g. `num_experts`, `seq_len`). Use this to target specific workload sizes without running the full suite.

### `--exclude-group VALUES` — Exclude by group axis

```bash
bash scripts/bench.sh --exclude-group 1,14107      # skip outlier workloads
bash scripts/bench.sh --exclude-group 1-16          # skip small workloads
```

Excludes workloads matching the specified group axis values. Useful for ignoring edge-case workloads (e.g., very small or very large inputs) while benchmarking the typical range. Supports the same comma-separated and range syntax as `--group`.

### `--index INDICES` — Select by workload index

```bash
bash scripts/bench.sh --index 0,3,5               # run workloads 0, 3, and 5
bash scripts/bench.sh --index 2-8                   # run workloads 2 through 8
```

Selects specific workloads by their index (same indices shown by `bash scripts/profile.sh --list`). Useful when you know exactly which workloads to test.

Trajectory is still saved for all filter modes, so you can inspect `results.json` for error details.

> **Note on `docs/workloads.jsonl`.** The tensor `path` fields inside are relative to the dataset root (`./blob/...`), not the child env. The benchmark runner resolves them automatically: local backend reads from `config.toml`'s `dataset_path`, Modal reads from the `/data` volume. Open the dataset-side file if you need to inspect raw tensors; `docs/workloads.jsonl` is a UUID / axes / shape reference only.

## A/B Compare (`--ab-compare <label>`) — preferred for sub-1x deltas

```bash
bash scripts/bench.sh --ab-compare iter-5   # current vs labeled snapshot
```

Runs the **current solution** and the **trajectory snapshot for `<label>`** back-to-back in the **same process / Modal container**. Because both runs share one cold-start + one thermal state + one tenant-load sample, cross-session drift (typically a few % CV on Modal — see the `benchmark` skill for baseline mechanics) nearly cancels — the A/B delta is tight even when absolute scores drift.

Output: both final scores, per-group delta, top-5 per-workload movers (by `|Δ|`).

**When to use — this is the first-line tool for iterating under noise:**
- **Any sub-1x delta decision.** A single `--label` run whose headline moved by +0.5..+1x is almost certainly noise on this op. Re-run as `--ab-compare <prior-label>`; the same-container Δ is the authoritative signal.
- Quickly validating a revert ("is B worse than A, for real?").
- Debugging a suspected regression that showed up across sessions.

**Common false-positive pattern:** a labeled bench shows +1x for a new
config that looks like a breakthrough; A/B-compare against the prior
labeled trajectory then shows Δ ≈ 0. The +1x was pure Modal drift; had
the agent trusted the standalone bump and committed, a no-op change
would have been "merged" based on noise.

**How label lookup works:** `bash scripts/bench.sh --ab-compare iter-5` matches any `trajectory/*iter-5*/` dir (latest by mtime wins). Each labeled bench creates one such dir, so labels propagate automatically. On a fresh spawn before any labeled iter has been run, `trajectory/` is empty — but `docs/prior/variants/<anchor>/` holds the parent kernel seeded from a prior session, and the lookup falls back there automatically. So `bash scripts/bench.sh --ab-compare <parent-anchor>` works on spawn for drift-cancelled comparison against the previous anchor.

**Constraint:** the snapshot must have kernel sources directly in the run dir (the default for labeled bench runs). `--variance-check` snapshots that only contain `noise-floor.json` are filtered out automatically.

> **Label sanitation.** Labels containing `/` or newlines are sanitized to `_` at trajectory-dir creation (otherwise they silently create nested directories, breaking `--ab-compare` lookup). Keep labels short (<30 chars) and avoid unusual punctuation for easiest re-lookup.

## Variance Check (`--variance-check N`)

```bash
bash scripts/bench.sh --variance-check 3   # run unchanged solution 3 times
```

Runs the **same unchanged solution N times** back-to-back to measure across-run noise (Modal cold-start, GPU thermal effects, neighbour load). Reports:

- **Overall**: mean / std / CV / range across the N runs.
- **Top 5 most variable workloads** (by CV), so you can identify outliers.
- **Per-group noise** (e.g., `batch_size=1: 46.28 ± 0.12x (CV 0.3%)`, …): lets you see whether noise is spread evenly or concentrated in one group — useful when deciding whether a group-specific speedup change is real.

Useful when you want a tight noise floor number rather than the single-Δ drift-cancelled measurement that `--ab-compare` gives. Runs N times → costs `N × bench`. Result is saved to `trajectory/noise-floor.json` (fields: `scores`, `score_mean/std/cv`, `per_workload`, `per_group`, `group_axis`). When you pass `--label <name>` alongside `--variance-check N`, a second copy is written to `trajectory/<sanitized-name>/noise-floor.json` (so noise-floor history is preserved across iters).

**Combine with workload filters for faster iteration.** Filters are threaded through unchanged, so:

```bash
bash scripts/bench.sh --variance-check 3 --smoke            # 1 wl/group × 3 runs
bash scripts/bench.sh --variance-check 3 --first 5          # 5 wls × 3 runs
bash scripts/bench.sh --variance-check 3 --group 15,30      # B=15/30 only × 3 runs
```

This is the "mid-cost validation path" between a single smoke test (2-3x noise) and a full 128-workload variance check (~15-20 min on Modal). The noise-floor JSON reflects the filtered subset — note the filter in your `--label` if it matters for later diffing.

> **Cold-start note.** The very first run after a fresh Modal container (empty JIT cache, GPU not yet at steady temperature, cuBLAS/CUTLASS not yet lazy-initialized) often scores noticeably lower than subsequent runs. If that bias would confuse your comparison, run a throwaway `bash scripts/bench.sh` (any mode) before `--variance-check N` so the loop starts in steady state. Use your judgement — this is not applied automatically.

> **Variance-check ↔ labeled bench: same baseline, different per-workload numbers (observed).** Running `--variance-check N` immediately after a labeled bench on identical code with the cached baseline can give per-workload `speedup_factor` values that diverge substantially from the labeled bench (3–4× lower per-workload at small-batch indices has been observed, with overall score also diverging — e.g. 0.76× variance-check vs 1.11× labeled bench on the same kernel). Likely root cause is autotune-config drift across the two paths even though both share `inject_baseline()`. Treat the OVERALL score and CV from `--variance-check` as the session noise-floor signal (cross-run reproducibility), but do NOT compare its per-workload or per-group `speedup_factor` numbers against a labeled bench's — they are not apples-to-apples. Use `--ab-compare <label>` when you need a drift-cancelled per-workload comparison. The variance-check output prints a `Baseline:` line so the source is at least visible; if it doesn't match what the labeled bench reported, that's another sign the numbers aren't comparable.

> **`Run N: HARNESS ANOMALY` line (Modal session-reuse failure).** On the modal backend, `--variance-check N` can silently lose runs when Modal returns an empty traces dict for a second/third invocation (no container init output reaches the variance-check loop) — `compute_score()` then produces `passed=0, failed=0, error=0, final_score=None`. The print branch in `bench_utils.py::run_variance_check` distinguishes this from a real "all failed" outcome and emits `Run N: HARNESS ANOMALY (empty results; per-definition workload count = {...})` to stderr. When you see it, just retry with a fresh `bash scripts/bench.sh --variance-check N` invocation — it is not a kernel signal.

> **`FINAL SCORE: HARNESS ANOMALY` on a single labeled bench (same shape).** A single `bash scripts/bench.sh` invocation can hit the same Modal session-level failure (e.g. `cudaErrorDevicesUnavailable` on the reference baseline, "No healthy workers available after baseline setup"). The print branch in `bench_utils.py::print_results` surfaces `FINAL SCORE: HARNESS ANOMALY (empty results; Passed: 0/0)` for this `passed=0, failed=0, error=0` shape instead of an opaque `FINAL SCORE: INVALID (), Passed: 0/0`. Retry with a fresh `bash scripts/bench.sh` — not a kernel signal.

## Error Diagnostics

When a workload fails (RUNTIME_ERROR, COMPILE_ERROR, INCORRECT_NUMERICAL, etc.), the benchmark now prints the error traceback inline (up to 20 lines for the first 3 failed workloads). The full error log is saved in `trajectory/*/results.json` under the `error_log` field for each failed workload.

This eliminates the need to guess why a kernel failed — check the traceback first.

### Diagnosing silent performance regressions (`--capture-logs`)

The isolated runner redirects each workload's stdout/stderr into a tempfile and only preserves it for non-PASSED workloads by default. This means a `print(...)` you add to `kernel.py` is **discarded** when the kernel runs correctly but slowly — exactly the case where diagnostic output is most useful.

`bash scripts/bench.sh --capture-logs` surfaces captured stdout/stderr for PASSED workloads too, saved to `trajectory/*/results.json` under the `log` field (truncated to 20000 chars per workload, last-line preserved). Use when:

- A structural change causes an unexpected regression (e.g. graph capture silently failing → 4 kernels launch eagerly)
- You added `print(...)` probes in `kernel.py` to verify runtime behaviour
- `profile.sh` / `sanitize.sh` didn't isolate the cause and you need runtime trace data

Default-off because each PASSED workload log can include 10-30 lines of pytorch/triton/cupti init noise; with 128 workloads that's ~1-2MB extra in `results.json`.

## Trajectory Tracking

Each labeled run saves a snapshot to `trajectory/YYYYMMDD_HHMMSS_label/`:

- All files from `solution/`
- `config.toml`
- `results.json` with full benchmark data (per-workload latencies, speedups, scores, metadata)

Use `bash scripts/bench.sh --label "description"` to create labeled snapshots.

## Per-call overhead floor

The bench output's "~80µs fixed per-call overhead" warning text is a real
floor — the time between Python entering `run()` and the first GPU
instruction executing. When overhead is the floor, the two highest-leverage
fixes — auditing pre-kernel GPU↔CPU syncs, and capturing stable shapes into
a CUDA graph — are CUDA-runtime recipes: see the `cuda` skill ("Per-call
overhead: audit GPU↔CPU syncs" and "Capture stable shapes into a CUDA
graph"). One silent-failure mode of the graph-capture fix is detection-worthy
on its own:

### Silent kernel skipping under graph capture

If any kernel in a captured pipeline silently fails to enter the CUDA
graph (runs once during capture, skipped on all replays), the
destination tensor keeps coincidentally-correct stale bytes from
capture-time execution across replays. Why this passes correctness
silently and inflates the headline (the cascade through the
benchmark's input-replay model) is documented in the
`benchmark` skill under "Silent-skip cascade".

**Detection (any one triggering = bug confirmed):**

- **Zero-output replay** — `output.zero_()` after warmup, `runner()`,
  `torch.cuda.synchronize()`, assert `output.abs().sum() > 0`.
- **Poison-cell** — write a sentinel (e.g. `output[0,0,0] = -1e9`),
  replay, check the cell was rewritten.
- **Varying inputs** — reuse same tensor addresses but mutate contents
  between replays (`q.normal_()`); assert output differs across calls.

**Known causes** (language-specific fixes documented in the corresponding
DSL skill; per-operator evidence in each archive's `TRAPS.md`):

- `@cute.kernel.launch()` — CuTe DSL's TVM-FFI stream binding does not
  pick up capture mode (flashinfer-bench issue #414).
- CUDA chevron `<<<grid, block>>>` without explicit stream — legacy
  null-stream is never in capture mode.
