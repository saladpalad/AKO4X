---
name: profiler-ncu
description: Run NVIDIA Nsight Compute (NCU) per-kernel profiling for register pressure, occupancy, stall-reason, memory-throughput, and IPC analysis. Use BEFORE architecting an optimization fix — not only after — when a hypothesis about microarchitectural behavior needs verification. NCU `Duration` is NOT comparable to bench timing; use NCU for ratios only.
---

# Profiler (NCU)

Wrapper around `ncu`. Command entry: `bash scripts/profile.sh`.

Detailed reference: `ncu.md`. Top-level commands:

```bash
bash scripts/profile.sh --list
bash scripts/profile.sh --index 5                          # default --set detailed --page details
bash scripts/profile.sh --index 5 --set full
bash scripts/profile.sh --index 5 --kernel-name ".*name.*"
bash scripts/profile.sh --index 5 --sections LaunchStats,Occupancy
bash scripts/profile.sh --index 5 --env NO_GRAPH=1         # modal backend; on local, prefix the run with NO_GRAPH=1
```

## Key constraints

- **NCU under graph capture is largely blind** — graph-launched kernels go through `cuGraphLaunch` and the `--kernel-name` filter often returns "No kernels were profiled". If your kernel uses `torch.cuda.CUDAGraph`, install a module-level `_NO_GRAPH = bool(os.environ.get("NO_GRAPH"))` gate AND set `NO_GRAPH=1` — modal backend via `--env NO_GRAPH=1`, local backend by exporting it in the shell (`run_local_profile.py` has no `--env` flag). See `ncu.md` "CUDA graph capture interaction".
- **Don't profile the reference** — the unoptimized Python implementation launches dozens of small kernels; profiling produces unhelpful noise. Make at least one optimization pass first.
- **NCU's "Est. Local Speedup" overestimates SMEM-staging fixes** on register-resident reduction kernels. Treat the field as an upper bound, not a prediction.

## COUPLED references

- Local backend: `scripts/run_local_profile.py`
- Modal backend: `scripts/run_modal_profile.py`
- Shared runtime: `scripts/bench_utils.py` (workload loading, dataset resolution)

Per-operator NCU traps (e.g. SMEM-staging regressions, PDL waves-per-SM thresholds) live in `docs/prior/TRAPS.md` (when your operator has a prior archive).
