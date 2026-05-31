# Profiler

> ⚠️ **NCU Duration is NOT comparable to bench timing.** Profiling overhead
> inflates kernel duration by 5-20× typical. Empirical example on B200:
> Triton fwd reports 11.58 µs in NCU vs ~1.5 µs in bench; CuTe reduce reports
> 13.15 µs vs ~0.9 µs. **Use NCU for ratios only** — occupancy%, IPC, warp
> stall reasons, memory throughput %, cache hit rates, block limits. Treat
> the absolute `Duration` field as a "kernel launched" sanity signal, not a
> perf number. Cross-reference absolute latency with `bash scripts/bench.sh`
> output.

> 🚨 **"No kernels were profiled" has TWO common causes — graph capture
> is only one of them.** Before reaching for `NO_GRAPH=1`, check what
> the kernel actually does:
>
> 1. **NVTX-include filter (active benchmark's NCU agent)** — the
>    benchmark's profiling agent runs the profiled call inside a fixed
>    `torch.cuda.nvtx.range(...)` and passes a matching `--nvtx
>    --nvtx-include` filter to ncu (the exact range name + agent source
>    live in the `benchmark` skill). If your filter / call
>    pattern doesn't land kernels inside that range, ncu reports "No
>    kernels were profiled" and lists what it saw at launch time under
>    "Available Kernels". Diagnose by re-running without
>    `--kernel-name` — if "Available Kernels" lists your kernel names
>    but no profile data appears, it's NVTX scope, NOT graph capture.
>    `NO_GRAPH=1` does not help here (no graph involved).
>
> 2. **CUDA graph capture** — applies ONLY if `solution/kernel.py`
>    itself uses `torch.cuda.CUDAGraph` / `torch.cuda.graph(g)` /
>    `g.replay()`. The fix is the two-step recipe below: install the
>    `_NO_GRAPH` gate in your kernel + pass `--env NO_GRAPH=1`. If
>    your kernel doesn't capture graphs, installing the gate is a
>    no-op — don't go down this path.
>
> Full pattern in "CUDA graph capture interaction" below.

Wrapper around NVIDIA Nsight Compute (`ncu`). Supports the `--index` / `--group` / `--exclude-group` workload filters (not `bench.sh`'s `--smoke` / `--first` / `--extremes`).

## Commands

```bash
bash scripts/profile.sh --list
bash scripts/profile.sh --index 5                                 # default: --set detailed --page details
bash scripts/profile.sh --index 5 --set full                      # NCU preset (basic|detailed|full|…)
bash scripts/profile.sh --index 5 --page raw                      # ncu output format (raw|details|source)
bash scripts/profile.sh --index 5 --kernel-name "my_kernel"       # full-match regex; use .*name.* for substring
bash scripts/profile.sh --index 5 --sections LaunchStats,Occupancy # add specific sections on top of --set
bash scripts/profile.sh --index 5 --timeout 120
bash scripts/profile.sh --index 5 --max-lines 200
bash scripts/profile.sh --ncu-options                             # list sets + sections available on this ncu build
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--index` | (required) | Workload indices (e.g. `5`, `0,3,5`, `2-8`) |
| `--list` | — | Print workload table with indices |
| `--group` | (all) | Filter by group axis values |
| `--exclude-group` | (none) | Exclude by group axis values |
| `--ncu-options` | — | `ncu --list-sets` + `ncu --list-sections` |
| `--set` | `detailed` | ncu `--set` preset (forwarded as-is) |
| `--page` | `details` | ncu `--page` format (forwarded as-is) |
| `--kernel-name` | (all) | ncu `--kernel-name` regex (forwarded as-is) |
| `--sections` | (none) | Comma-list of section names added via ncu `--section` (forwarded as-is) |
| `--max-lines` | (unlimited) | Keep the last N lines (tail-truncate; metrics live at the end, so head-truncation would drop them) |
| `--timeout` | `180` | Timeout in seconds. Reference / unoptimized kernels can need `300`+ because the Python path launches many small kernels and NCU does 9 passes each. |

Output auto-saves to `profiles/w<index>_<timestamp>.json` (full, untruncated — inspect with `jq -r .output profiles/w0_*.json | tail -500` if `--max-lines` cut off something you need).

> **Profile an optimized kernel, not the reference.** The reference implementation launches dozens of small elementwise/gemmk1 kernels; profiling it produces thousands of lines of unrelated data. If `solution/` is still the reference, make one optimization pass first, then profile.

## Discover → Select sections

`--set` picks an ncu preset bundle; `--sections` adds individual sections on top. To find names valid on the image that runs the profile:

```bash
bash scripts/profile.sh --ncu-options         # enumerate
bash scripts/profile.sh --index 5 --sections <name1>,<name2>,...
```

Section names (e.g. `LaunchStats`, `Occupancy`, `SchedulerStats`, `WarpStateStats`, `MemoryWorkloadAnalysis`, `SourceCounters`) vary by ncu version — always confirm via `--ncu-options` before relying on one.

## Wrapper-specific behavior

### CUDA graph capture interaction

**Precondition for this whole section**: your `solution/kernel.py`
actually uses `torch.cuda.CUDAGraph`. If it doesn't, `NO_GRAPH=1` is
inert — re-read the top-of-file 🚨 about NVTX-scope mis-attribution
first.

If the solution uses `torch.cuda.CUDAGraph`, kernels launch via `cuGraphLaunch`, not `cuLaunchKernel`. Consequences for this wrapper:

- **`--kernel-name` filter may return "No kernels were profiled"** — graph-captured launches don't surface the same symbol info as direct launches, so ncu's kernel regex can miss them.
- **Full profile (no `--kernel-name`) may still miss kernels** — under graph capture, NCU has been observed profiling only the first kernel of a replay even without any kernel-name filter; subsequent kernels never surface. Do not trust graph-captured "Available Kernels" listings as complete.
- **Workflow for bottleneck attribution**: add a module-level env gate in `solution/kernel.py` that skips the graph-capture branches when set (e.g. `_DISABLE_GRAPH = bool(os.environ.get("DSA_NO_GRAPH"))`), run the profile with `DSA_NO_GRAPH=1` to launch every kernel eagerly so NCU sees them all, then unset the env var for labeled benches — the graph capture itself is a meaningful wall-time win worth keeping in the production path. This is cheaper than writing `torch.cuda.Event`-pair timing harness.

  **Generic recipe**: `_DISABLE_GRAPH = bool(os.environ.get("NO_GRAPH"))` at module scope, then `if not _DISABLE_GRAPH:` around your `torch.cuda.graph(g):` capture and `g.replay()` branches.

  **Preferred invocation (modal backend)**: pass `--env NO_GRAPH=1` to `scripts/profile.sh` to set the env var inside the Modal container without editing the kernel source:
  ```bash
  bash scripts/profile.sh --index 40 --env NO_GRAPH=1 --kernel-name ".*my_kernel.*"
  ```
  The `--env` flag accepts `KEY=VAL` pairs, comma-separated. **Local backend**: there is no `--env` flag — export `NO_GRAPH=1` in the shell before `bash scripts/profile.sh …` (the module-level gate reads it identically). Fall back to hand-editing `kernel.py` only if the gate has to control code paths beyond graph capture (e.g. dispatching entire alternate kernels).

### "Est. Local Speedup" is optimistic for SMEM-staging fixes

NCU's per-section "Est. Local Speedup: N%" assumes you can free the
bottleneck without paying any new cost. For the common "raise occupancy
by SMEM-staging registers" fix, the estimate ignores the SMEM round-trip
+ `__syncthreads` barrier that the staging introduces. On register-
resident reduction kernels with small per-thread state (~32 regs of
working tile), staging often regresses despite NCU predicting double-
digit local speedup.

Per-operator evidence and the underlying mechanism live in the relevant
`docs/prior/TRAPS.md` "NCU's 'Est. Local Speedup' overestimates
SMEM-staging fixes" entries (when your operator has a prior archive).

How to use the estimate: treat it as an *upper bound* on what's
possible if the proposed fix is free. For SMEM-staging proposals on
register-resident kernels, always measure the staged variant before
committing — the predicted speedup typically does not materialize.

### "Est. Local Speedup" does NOT cover CTA-coarsening / packing fixes

When NCU diagnoses a kernel as under-loaded (low occupancy, high "Est.
Local Speedup"), the implied fix is to RAISE occupancy — lower
reg/thread, smaller per-CTA tile, more warps per CTA, or smaller
per-CTA workload. The estimate does NOT cover the inverse direction:
packing multiple logical work units into one CTA to amortize launch +
warp-init overhead.

Packing a kernel REDUCES total CTA count, which reduces wave count.
On already-under-loaded kernels in the few-CTAs-per-SM regime, reducing
CTA count below ~1 wave on the target SM count removes the warp-
scheduler's primary latency-hiding mechanism (across-CTA / across-wave
instruction overlap on the same SM). Per-CTA work amortization buys
back some of the launch overhead but per-CTA latency now runs fully
exposed. Net result is plateau-at-parent or regression.

Concrete signature: a small reduce/epilogue kernel with grid
`(N_FAST × B)`, occupancy ~10%, "Est. Local Speedup" 90%+. Packing
`k` logical units per CTA (grid `(N_FAST/k × B)`) does NOT capture the
estimate on small-batch workloads where `(N_FAST/k) × B < num_SMs` —
those drop below 1 wave and lose across-CTA latency hiding entirely.

How to use the estimate: treat occupancy bottlenecks as actionable
ONLY by intra-CTA fixes (raise warps/CTA, reduce reg/thread, smaller
tile). For CTA-count-reduction proposals on already-under-loaded
kernels, ALWAYS sanity-check `CTA_count / num_SMs ≥ 1 wave` AFTER the
reduction on the smallest-batch workload — if it drops below 1, the
proposal will lose to the parent regardless of per-CTA work win.
