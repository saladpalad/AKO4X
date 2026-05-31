# Cross-variant TRAPS — dsa-topk-indexer

Toolchain / measurement-methodology facts that apply to **every** variant
in this archive, regardless of which one is anchor. Created 2026-04-23
(reflected from gdn_decode_v0 session); each entry has a **Why** so
future sessions can judge whether their context flips the fact.

---

## Measurement methodology

### `@cute.kernel` is not captured into `torch.cuda.graph` — `evict_last_v4` has an inflated headline (2026-04-23)

**Fact:** `@cute.kernel.launch()` does **not** participate in CUDA
graph capture. Calling it inside `with torch.cuda.graph(g):` makes
the kernel **execute immediately** during the capture block, but the
launch is **not recorded into the graph**. Subsequent `g.replay()`
does not run the CuTe kernel. For `evict_last_v4`, the CuTe DSL radix
writes the final topk output tensor, so replay leaves `output` stale
from capture time — and `flashinfer-bench` uses fixed inputs per
workload, so stale output coincidentally matches the reference,
correctness passes, and CUPTI only counts the Triton score kernel's
GPU time. Reported 41.64× is inflated; honest per-call work is
~30-35× (matches `v8_radix_bt256` 45.55× and `fast_split_v6` 44.82×,
both of which actually capture all their kernels).

**Why:** TVM-FFI's environment-stream mechanism that CuTe DSL uses
for stream binding doesn't pick up `torch.cuda.graph()`'s capture
stream. The empty-graph warning only surfaces when the CuTe kernel
is captured alone; with a Triton anchor ahead of it, the graph is
non-empty → PyTorch suppresses the warning → silent failure.

**Affected variant in this archive:**
- `evict_last_v4` (41.64×, 2026-04-19) — Triton FP8 score +
  CuTe DSL radix; radix is the output writer.

**Unaffected variants (all kernels captured honestly):**
- `v8_radix_bt256` (45.55×) — pure CUDA radix, current anchor.
- `fast_split_v6` (44.82×) — pure Triton + CUDA split.
- `warp_coop_v3` (40.7×) — CUDA-only radix, pre-`evict_last`.

**How to apply:**
- Don't use `evict_last_v4`'s headline as the credibility bar for
  future variants; use `v8_radix_bt256` or `fast_split_v6`.
- When exploring CuTe DSL radix in a new variant, run at least one
  of the three detection tests from sibling
  `../dsa-sparse-attention/TRAPS.md` (zero-output sanity,
  poison-cell test, varying-inputs test) before trusting the
  speedup.
- For the full mechanism + quantitative evidence (CUPTI 4.5 µs vs
  Event 6.2 µs on Triton-fwd-only graph; pair kernel 9.5 µs when
  both reduce steps in graph), see
  `../dsa-sparse-attention/TRAPS.md` section "`@cute.kernel` is not
  captured into `torch.cuda.graph`" and flashinfer-bench issue #414.
- After `nvidia-cutlass-dsl` upstream fixes `.launch()` to respect
  capture mode, re-measure `evict_last_v4` to determine if the
  evict_last K+scale pattern is genuinely useful vs
  `v8_radix_bt256` (same-container AB recommended since
  cross-session drift is ~±1× on B200).

---

### Gate sub-1× `--ab-compare` wins on `--variance-check 3+` before committing (2026-04-22)

**Fact:** A `--ab-compare <prior>` that shows positive per-group
deltas but a sub-1× headline delta can reverse sign under a 3-run
variance-check. The per-group positive signal can be dominated by
individual ±1-4× workload swings that cancel in the mean.

**Why:** Full-bench cross-container noise on Modal B200 is ~1-2%;
any headline delta smaller than that falls inside the noise band and
needs multiple same-container runs to average out. Single `ab-compare`
runs back-to-back in one container remove drift from two code
versions but not the per-workload jitter that adds up to ~1-2% at
full-bench scale.

**Seen in:**
- v8 session (2026-04-22): a parallel session reported +0.84× for
  `num_stages=2` on the Triton score kernel with "moves every
  slow-path group consistently upward"; v8's own `--ab-compare` vs
  iter-1 showed +0.4× avg across slow-path groups. The 3-run
  variance-check then said **−0.74×** (45.00 ± 0.16% CV vs baseline
  45.74 ± 0.03% CV). Change reverted.

**How to apply:**
- Any optimization claim whose headline delta is smaller than ~1×
  (~2% at these speedups) must be gated on `--variance-check 3+`
  before labeling the iter as a win — no matter how consistent the
  per-group `ab-compare` signal looks.
- A positive-per-group, sub-1× headline `ab-compare` is not the same
  signal as a variance-checked mean; treat the two as separate
  verdicts that can disagree.
- If `--variance-check` results have overlapping CVs between A and
  B, the change is indistinguishable from noise regardless of the
  headline sign.

---

### `my_kernel<<<grid, block>>>` without an explicit stream is silently skipped by `torch.cuda.graph` capture (2026-04-24)

**Framework-level mechanism** (bench fixed-inputs × silent-kernel-skip ×
CUPTI blindness = inflated headline with passing correctness): see
`templates/benchmark.md` "Silent kernel skipping under graph capture"
for the generalized detection recipe. Entry below is the chevron/
null-stream cause + v9 iter-5 evidence + how-to-apply for this archive.

**Fact:** A chevron launch of the form
`my_kernel<<<grid, block>>>(...)` inside a torch-extension function
compiled via `torch.utils.cpp_extension.load_inline` (or TVM FFI)
targets the legacy/null stream (stream id 0). Inside a
`with torch.cuda.graph(g, stream=s):` block, `s` is put into capture
mode — but stream 0 is **not**. The launch therefore **runs
immediately during the capture block** (output bytes are written to
the destination tensor), but the launch is **not recorded into the
graph**. Subsequent `g.replay()`s do not re-run the kernel.

When this is the **only** issue in the capture block, `capture_end()`
prints `UserWarning: The CUDA Graph is empty. This usually means that
the graph was attempted to be captured on wrong device or stream.`
and CUPTI timing later raises
`ValueError: No kernel activities recorded for iteration 0`. When at
least one other kernel in the same `with` block captures correctly
(e.g. a Triton kernel that routes through PyTorch's current stream
automatically), the "empty" warning is **suppressed**; you get a
graph with only the Triton kernel in it, and the chevron-launched
kernel silently runs once during capture then never again. Correctness
still passes in `flashinfer-bench` because `use_isolated_runner = true`
holds inputs fixed per workload and the capture-time eager write
leaves the right bytes in the destination buffer.

**Fix:** route the launch through PyTorch's current stream explicitly:

```cuda
#include <ATen/cuda/CUDAContext.h>

void my_wrapper(torch::Tensor in, torch::Tensor out) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    my_kernel<<<grid, block, /*shmem_bytes=*/0, stream>>>(...);
}
```

`at::cuda::getCurrentCUDAStream()` returns the stream that PyTorch has
placed into capture mode (or the normal current stream outside a
capture block), so the launch is captured into the graph and re-runs
on every replay.

**Why:** The legacy/null stream has CUDA-wide sync semantics that are
explicitly incompatible with per-stream capture; CUDA leaves it out
of capture mode regardless of which stream is captured. PyTorch's
Triton and torch-op launchers fetch the current stream explicitly so
they never hit this. Raw `<<<>>>` inside load_inline-compiled code
does not, so it falls through.

**Seen in:**
- v9 iter-5 (2026-04-24) debugging a new CUDA bf16 wmma score
  kernel: correctness passed on warmup (matched_ratio=1.0), CUPTI
  timing then raised `No kernel activities recorded for iteration 0`.
  Adding `at::cuda::getCurrentCUDAStream()` to both the new score
  wrapper and the anchor's radix wrapper eliminated the empty-graph
  warning and made the kernels visible to CUPTI.
- **Verified NOT applicable to the v8 anchor (2026-04-25, v0 session).**
  `variants/v8_radix_bt256/kernel.py:460` launches
  `radix_select_topk_kernel<BT><<<B, BT>>>(...)` with the no-stream
  pattern. Earlier reading suspected silent-skip; verification ran
  `DSA_NO_GRAPH=1` (eager) and default (graph) modes and compared
  per-workload timing on the slowest workloads — they are identical
  (B=31 a52c09bc: 0.003-0.004ms in both modes). The radix IS being
  captured & replayed honestly on this image; the v8 headline 45.55x
  stands as-is, no interpretation correction needed. The proposed
  patch (add `at::cuda::getCurrentCUDAStream()` to the chevron) was
  applied and measured: **-4× regression for slow workloads**
  (0.004ms → 0.015ms in eager mode; cached-static-stream variant only
  ~25% slower at 0.004ms but still net-negative). Root cause:
  `c10::cuda::getCurrentCUDAStream()` adds ~12µs/call per launch when
  uncached, ~1µs/call cached vs the legacy stream 0 (chevron default).
  At a per-call radix work of ~3-4µs, this overhead dominates.
  **Do not apply the patch to v8.** The kernel header's
  `DSA_NO_GRAPH` rationale was rewritten to attribute the gate to
  NCU profiler ergonomics rather than capture brokenness. The TRAP
  itself remains valid as a general fact for any NEW `.cu` added to
  the captured graph; only the v8-specific application is closed.

- **Re-confirmed by submission-v4 official bare-metal eval (2026-04-25
  v6 session).** `FISAC-full-agent/dsa_indexer` at the `submission-v4`
  tag uses the same `radix_select_topk_kernel<BT><<<B, BT>>>` no-stream
  pattern (kernel.py:259 of that tag) and **passed all 128 workloads'
  correctness on the official bare-metal CUDA 13.2 evaluator**. If the
  chevron were silently skipped on replay, capture-time eager output
  would be stale across iter 1+ of the same workload (input-fixed → still
  passes correctness within one workload), but **stale output cannot
  match the reference for a different workload** with different inputs.
  All 128 workloads passing means each workload's output was either
  freshly recomputed at the start of that workload's evaluation OR
  the chevron was actually captured. Combined with the
  DSA_NO_GRAPH=1 timing-equivalence test above (which would catch
  per-workload skipping), this is independent confirmation that the
  v8/v4 chevron pattern is captured on the contest stack. Future
  agents flagging this pattern as a v8-specific risk should re-read
  this entry before proposing the patch — the patch has been measured
  and rejected on perf grounds.

**How to apply:**
1. Any new `.cu` compiled via `load_inline` / TVM FFI with chevron
   launches MUST fetch `at::cuda::getCurrentCUDAStream()` and pass as
   the 4th chevron arg. Treat bare `<<<grid, block>>>` as a style
   error in any file that might be called under graph capture.
2. Detection tests before trusting a multi-kernel graph-captured
   speedup:
   - `output.zero_()` between warmup and the first timed iteration
     and confirm correctness on replay (poison-cell test — if
     correctness still passes, the kernel responsible for writing
     `output` isn't being re-run).
   - NCU the graph replay with no `--kernel-name` filter. Any kernel
     listed under "Available Kernels" but absent from the per-kernel
     profile section is not actually captured.
   - `--env NO_GRAPH=1` compare (or hand-edited `_DISABLE_GRAPH = True`):
     the delta in per-workload latency should be roughly
     `#captured_kernels × per-launch_overhead (~40-60µs)`. If the
     delta is smaller than expected for the number of kernels in the
     pipeline, something isn't captured.
3. When reviving the v8 anchor's radix under graph capture (e.g.
   alongside an inline-PTX fp8 score rewrite), patch line 460
   simultaneously and variance-check the full pipeline — don't A/B
   one change at a time because the two kernels' graph-capture
   interaction is the unit of measurement.
