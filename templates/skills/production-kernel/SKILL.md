---
name: production-kernel
description: Enforce production promotion for CUDA, Triton, CuTe, TileLang, and GPU extension candidates. Use when optimizing, reviewing, validating, benchmarking, profiling, or promoting a kernel that must remain numerically stable, training-safe, stream-correct, deployable, debuggable, and understandable to humans—not merely fast on leaderboard inputs.
---

# Production Kernel

Treat benchmark correctness as the first gate, never the production contract.

## Workflow

1. Read `.ako4x/production.toml` and `references/gates.md`.
2. Do not begin optimization until `ako4x-lab doctor` has produced real, parseable NCU and NSYS reports.
3. Preserve the configured reference, baseline, benchmark, profiler, and gate commands. Never weaken evaluation to admit a candidate.
   Protected evaluation sources are hash-bound before the optimization turn.
4. Record a hypothesis before each architectural experiment and every completed benchmark in `ITERATIONS.md`.
5. Use NCU/NSYS evidence to select optimization work. On B200/H100, query `KernelWiki` for relevant mechanisms and upstream examples.
6. Keep performance and production evidence separate. Quarantine any faster candidate that fails a hard gate.
7. Leave the exact measured candidate unchanged. Promotion binds all evidence to its source hash.

## Kernel requirements

- Keep a readable human-owned router and named algorithm paths.
- Document supported shapes, dtypes, layouts, devices, aliasing, mutation, stream semantics, concurrency, and fallback behavior.
- Keep a correctness-first fallback for unsupported or risky paths.
- Never cache numerical outputs by object identity or pointer.
- Never hide asynchronous work, auxiliary streams, or computation outside the measured dependency chain.
- Avoid process-global backend mutations; restore unavoidable state with scoped cleanup.
- Preserve failing seeds, first failing training step, profiler reports, environment fingerprints, and minimal reproducers.

Use `ako4x-lab status --watch` for live state and `ako4x-lab promote` only after the run reaches `PROMOTABLE`.
