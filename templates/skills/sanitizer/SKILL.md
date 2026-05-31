---
name: sanitizer
description: Run NVIDIA compute-sanitizer (memcheck / racecheck / initcheck / synccheck). FIRST action on `INCORRECT_NUMERICAL` or flaky output — rolling back destroys the evidence; sanitizer often points at the exact line. Also useful for diagnosing race conditions in multi-block kernels and uninitialized-memory reads.
---

# Sanitizer

Wrapper around `compute-sanitizer`. Command entry: `bash scripts/sanitize.sh`.

Detailed reference: `sanitizer.md`. Top-level commands:

```bash
bash scripts/sanitize.sh --list
bash scripts/sanitize.sh --index 5                       # default: memcheck
bash scripts/sanitize.sh --index 5 --tool racecheck
bash scripts/sanitize.sh --index 5 --tool initcheck
bash scripts/sanitize.sh --index 5 --tool synccheck
bash scripts/sanitize.sh --index 5 --tool all            # sequential, separate timeout each
```

## Key constraints

- **Run sanitizer FIRST on `INCORRECT_NUMERICAL`** — before any rollback. Rolling back overwrites the failing kernel and destroys the only evidence of which line went wrong.
- **`cuGetProcAddress_v2` noise inflates ERROR SUMMARY** — these are PyTorch / triton / cupti probing optional CUDA driver entry points at import time. The wrapper's `>>> NOISE FILTER <<<` banner subtracts them; trust the "apparently-real kernel hits" count.
- **Real kernel errors always name a kernel / thread / block / address**. If a hit only shows Python host frames, it's noise.
- **CUDA-graph-captured kernels** show up but with sparse stack info; if traces are unhelpful, temporarily disable graph capture (env var or skip `g.replay()` path) for an eager-mode report, fix, then re-enable.

## COUPLED references

- Local backend: `scripts/run_local_sanitize.py`
- Modal backend: `scripts/run_modal_sanitize.py`
- Shared runtime: `scripts/bench_utils.py` (workload loading)
