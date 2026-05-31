---
name: benchmark
description: Reference for the active benchmark harness — what it IS and how it behaves (the active benchmark is flashinfer-bench). Covers config.toml structure (`[solution]`/`[build]`/`[benchmark]` tables), the status enum, the workload / fresh-inputs model, reference-baseline + scoring mechanics, tolerance keys, TVM-FFI builder link limits, and the silent-skip cascade. Invoke whenever you decode a bench status string, hit an unfamiliar config.toml field, suspect "correctness passed but the headline is implausible", or need to know what's frozen before proposing a bench edit — do NOT guess field or status semantics. Bench commands and noise methodology (A/B compare, variance check, drift cancellation) live in the `bench` skill.
---

# Benchmark

The active benchmark harness — what this project's `bench` skill is a frontend for. Reference for **what the benchmark is and how it behaves** — distinct from the `bench` skill which covers **how to drive it from the harness shim**. The active benchmark is **flashinfer-bench**; this skill is the single home for its schema, status, scoring, and frozen-for-comparability behavior. Detailed body: `benchmark.md`.

When this skill applies:
- Looking up a `config.toml` field you don't recognize.
- Decoding a status string from bench output (`COMPILE_ERROR`, `INCORRECT_NUMERICAL`, etc.).
- Reasoning about why correctness "passed" but the headline looks too good (silent-skip cascade ends here).
- Checking what's frozen for bench comparability (scoring/baseline/tolerance) before proposing a harness edit.
- Understanding the TVM-FFI link limit for C++/CUDA (no compile-time linking) and which kernel libraries are disallowed (cuBLAS / cuDNN) vs. allowed building blocks (CUTLASS, the DSLs) — see "Valid solution: write your own kernel".
