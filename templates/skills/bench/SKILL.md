---
name: bench
description: Run performance benchmarks to get a verdict on whether a kernel change actually helped. Use --ab-compare for sub-1x deltas (drift cancels in same container), --variance-check for noise floor, and the subset modes (--first, --smoke; --extremes is modal-backend only) for compile-correctness probes only — those are NOT performance verdicts. The only authoritative source for "is this change real". Benchmark specifics (config.toml schema, status enum, scoring formula, baseline rule, fresh-inputs contract) live in the `benchmark` skill.
---

# Bench

Single source of performance verdicts on this AKO4X harness. Command entry: `bash scripts/bench.sh`. The active benchmark is flashinfer-bench — its schema, behavior, and frozen-for-comparability segments live in the `benchmark` skill.

Detailed reference: `benchmark.md`. Top-level workflow:

```bash
bash scripts/bench.sh                          # full bench (the verdict)
bash scripts/bench.sh --label "iter-N desc"    # full + trajectory snapshot
bash scripts/bench.sh --ab-compare <prior>     # drift-cancelled Δ vs labeled snapshot
bash scripts/bench.sh --ab-compare <prior> --label <new>   # same + save current as <new> for chaining
bash scripts/bench.sh --variance-check 3       # measure session noise floor
bash scripts/bench.sh --first 1                # compile/correctness only (NOT perf verdict)
```

## Methodology under noise

Generic benchmarking methodology — applies regardless of the benchmark underneath:

- **A/B compare** (`--ab-compare <prior-label>`) runs the current solution and a labeled trajectory snapshot back-to-back in the same process / container; cross-session drift cancels in the delta. **First-line tool for any sub-1x decision.**
- **Variance check** (`--variance-check N`) runs the current solution N times against itself to measure the session's noise floor. Use to set the threshold below which deltas should be treated as noise.
- **Drift-cancellation reasoning**: any single-run `--label` headline can move by per-session drift (Modal is typically a few % CV) without code change. A standalone +1x bump is **not** evidence of a real improvement on a noisy backend.
- **Subset filters** (`--first`, `--smoke`, `--index`, `--group`; `--extremes` is modal-backend only) are for compile / correctness probes only — running on ≤3 workloads typically shows 2-3x variance vs the full-bench mean and is NOT a performance verdict.

## Frozen for bench comparability

The active benchmark's scoring formula, baseline freshness rule, and tolerance behavior are **frozen across multi-run campaigns** so results stay comparable. Edits to those get rejected across runs. The benchmark-specific frozen items are listed in the `benchmark` skill under "Frozen for bench comparability".

## COUPLED references

- Runtime core: `scripts/bench_utils.py` (shared by bench / profile / sanitize)
- Local backend: `scripts/run_local.py`
- Modal backend: `scripts/run_modal.py`
- Correctness audit: `scripts/cheat_check_modal.py` — **modal-only**; no local equivalent. Skipped on local-backend runs.

Editing `scripts/bench_utils.py` is allowed for non-frozen behavior (e.g., output formatting, error diagnostics). The frozen segments listed in the `benchmark` skill are not.
