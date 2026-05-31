# Cross-variant gotchas — gdn-prefill

Cross-variant facts that outlive any single variant. If a future session
"discovers" any of these on its own, that's a signal the warning didn't
land — please rewrite for clarity.

## NCU register spill is a red herring (kkt_solve)

`_kkt_solve_kernel` reports 254 regs/thread + 1536 local-memory spill
requests under NCU on B200. **Three different attacks on this in
cuda_graph_v1's session all regressed.** The spilled values are
short-lived intermediates that the compiler is handling correctly;
attacking the spill via num_warps, tile size, or fp32→bf16 in-place
shadowing each made things worse:

- num_warps=8 (iter-6): −35% — sub-matmul output tile [16, 128] only
  needs one warpgroup; the second stalls.
- BV_wu=BK_wu=64 (iter-5): −33% — doubles the V/K-tile loop count in
  phases 5-6, which already issue 10 sub-matmuls per iteration.
- In-place fp32→bf16 shadow of `b_AiXX` (iter-7): −24% — Triton SSA
  keeps both bindings; the apparent register relief never materializes.

The `b_AiXX_b` aliases in cuda_graph_v1's kkt_solve LOOK redundant;
they are necessary fp32→bf16 caches. Keep them.

## BV_rec sweet spot is 16 — both directions regress

State_recurrence's per-V-tile MMA is `[BT=64, K=64] @ [K=64, BV] →
[BT, BV]`. Sweet spot is BV_rec=16 (8 V-tiles per head):

- BV_rec=8 (iter-14): −9% — sub-MMA fills 1/8 of native tcgen05
  64×128 tile; underutilization swamps the parallelism gain.
- BV_rec=32 (iter-9 → kept temporarily): only +3% over BV=64.
- BV_rec=64 (iter-4): the iter-4 baseline; less parallelism but
  better MMA utilization per block.
- BV_rec=128 (iter-3): registers spill; state accumulator is too big.

If you're tempted to push BV_rec smaller for occupancy on a different
GPU generation, re-measure — the MMA shape is what sets the floor.

## Module-level state requires `use_isolated_runner = true`

The `_GRAPH_CACHE` and `_CHUNK_META_CACHE` dicts are module-level. If a
runner reuses one Python process across workloads (persistent runner),
cache entries from workload A may alias into workload B's reads. Set
`[benchmark] use_isolated_runner = true` in `config.toml`. spawn.py
defensively patches this on new children. The default in
`scripts/bench_utils.py:64` is now `True` since 2026-04-23 (commit
`838eead`); children spawned before that date carry the old `False`
default until re-spawned.

## CUPTI span counts memcpys — every skipped `.clone()` and `.copy_()` pays

flashinfer-bench's `bench_gpu_time_with_cupti` enables tracing on
CONCURRENT_KERNEL, MEMCPY, and MEMSET activities launched in the iter's
CPU-timestamp window, then computes span as
`max(activity_end) - min(activity_start)` across them. **Every memcpy
in your `run()` is span.** Three patterns from the cuda_graph_v2
session (each was independently a full iteration's worth of work):

- `return output.clone(), new_state.clone()` = 2 GPU memcpys per iter
  that the framework's `time_runnable` immediately discards. Returning
  static-buffer refs from the replay path is safe — `time_runnable`
  drops the tuple each iter, and `check_correctness` synchronizes
  before reading. Saved iter-2 +27% on this op.
- Within a trial's CUPTI timing loop, the framework calls `run()` with
  the SAME input tensor objects for all 5 timed iters. Compare
  `(q.data_ptr(), k.data_ptr(), …)` to the last call's tuple in the
  replay path; skip every `_foreach_copy_` on match. CPU-side
  data_ptr() reads (~175ns each, ~1.4µs total for 8 inputs) are free
  from span. Saved iter-3 +58% on this op. **⚠️ Removed in v7
  (2026-04-25)** — the optimization depends on the eval harness
  property "same data_ptr ⇒ same data" and silently produces wrong
  outputs under any eval that mutates inputs in-place while reusing
  tensor objects (PR #413 direction). Borderline-violates strict
  reading of the contest rule "skipping it because inputs look the
  same is not allowed" even though the skipped operation is a memcpy
  rather than a compute kernel — see new entry "Pointer-keyed
  skip-copy is borderline by spirit-reading and silently fails under
  input-randomized eval" below for the full v7 decision-of-record.
  The 58% iter-3 figure is now historical.
- `torch._foreach_copy_` on a mixed-dtype list silently falls back to N
  per-tensor `cudaMemcpyAsync` (one MEMCPY activity each). Group by
  dtype so each group fuses into one `multi_tensor_apply_kernel`
  (one CONCURRENT_KERNEL activity).

Broader: on any multi-kernel op measured via CUPTI span, audit per-iter
memcpys before chasing kernel-body micro-optimizations. They can be
30%+ of span on small-T workloads.

## Kernel specialization by T-bucket — kkt_solve has dead MMAs for T ≤ 48

The FLA chunked algorithm decomposes BT=64 into 4 sub-chunks of BC=16.
For T_seq ≤ BC, only sub-chunk 0 has valid data; for T_seq ≤ 2*BC,
only sub-chunks 0+1; etc. Phases 2-4 of `_kkt_solve_kernel` (KKᵀ,
forward-substitute, off-diag A_inv) have runtime
`if i_tc1 < T_seq:` guards that skip dead sub-chunks. **Phases 5 and
6 (u, w compute) DO NOT** — they always run all 10 sub-block matmuls
regardless of T_seq, so 9 of 10 are dead work on masked data when
T_seq ≤ 16.

cuda_graph_v2 added three specialized kernels
(`_kkt_solve_tiny_kernel`, `_kkt_solve_tiny2_kernel`,
`_kkt_solve_tiny3_kernel`) with compile-time-constant sub-chunk counts
of 1/2/3, doing 1/3/6 MMAs per phase. Dispatched by max(T_seq) at
graph capture (one-time GPU→CPU sync for multi-seq workloads).
Cumulative +8% on mean across iter-8/9/11.

Broader: when a kernel is chunk-decomposed, audit every phase for
missing sub-block guards — the `if` cascade may only be in the most
visible place. The compute phases that issue many independent
matmuls are the easiest to leave half-guarded.

## Fused single-chunk kernel eliminates h_buf roundtrip for NT=1 — but regresses for NT>1

When `max(T_seq) ≤ BT`, every sequence fits in one chunk and
`_state_recurrence_kernel` + `_fwd_o_kernel` can be merged into a
single `_fused_single_chunk_kernel` that keeps `h_snap` in registers,
computes `v_new + q@h + tril(q@k^T*G)@v_new` inline, writes output
directly. Eliminates the `h_buf` HBM write+read AND one
graph-captured launch. v2 iter-5 +6%.

Does NOT generalize to NT>1: the fused per-block serial work becomes
`state_chunk_time + output_chunk_time` per chunk, roughly doubling
per-block time. The separate `_fwd_o_kernel`'s natural chunk-level
parallelism would otherwise absorb that output cost (it has
`(NV_o, NT, HV)` blocks vs the fused kernel's `(NV_rec, N*HV)`).
Estimated ~2× regression for single-seq T=8192. v2 dispatches the
fused kernel only when `max(T_seq) ≤ BT`.

Broader: cross-kernel fusion is only a win when the consumer's
parallelism CAN be absorbed by the producer's grid. If the consumer
gets parallelism from a dimension the producer doesn't expose, the
fused kernel pays serial-loop cost for what was previously parallel
work.
Under CUDA graph capture, the launch-saving argument for fusion is
MUCH weaker than it looks on paper: graph-replay inter-kernel gaps
are <1µs, and h_buf HBM traffic at small NT is <1MB (sub-µs at 8TB/s
HBM). v3 iter-1 re-tested NT=2 multi-chunk fusion with a purpose-built
`_fused_multichunk_kernel` and ab-compared at +0.16% vs no-fuse (i.e.
pure drift noise). If you're tempted to re-try multi-chunk fusion for
a "save one launch" reason, first measure how much launch-overhead
actually shows up in span (v3 probe: near-zero).

## Chevron CUDA launches need an explicit stream under `torch.cuda.graph` (2026-04-24)

See `../dsa-topk-indexer/TRAPS.md` section "`my_kernel<<<grid, block>>>`
without an explicit stream is silently skipped by `torch.cuda.graph`
capture" for the full mechanism + fix + detection tests.

Current gdn-prefill variants (`cuda_graph_v1`, `cuda_graph_v2`,
`cuda_graph_v3`) are Triton-only, which routes through PyTorch's
current stream via the Triton launcher and is **not** affected. But
if a future variant adds a hand-written CUDA kernel (e.g. an fp8
mma.sync chunk kernel replacing the Triton `_chunk_fwd_*_kernel`)
alongside the Triton kernels inside the existing graph-capture path,
apply the `at::cuda::getCurrentCUDAStream()` fix pre-emptively.
Detection recipe: `output.zero_()` between warmup and first timed
iter, check correctness; NCU graph replay with no kernel-name filter,
confirm every "Available Kernel" appears in the per-kernel section.

## Unified K=N MMA beats split-k halves when both halves are in-register state

v3 iter-7/iter-8 eliminated split-K accumulators in `_state_recurrence_kernel`
and `_fused_single_chunk_kernel`. Before: state tile stored as
`b_h1 [BV, 64] + b_h2 [BV, 64]` halves; each chunk body issued TWO
tl.dot calls chained by `b_v += tl.dot(b_w2, tl.trans(b_h2))`. The
accumulation created an SSA data dependency that Triton's scheduler
cannot rewrite post-hoc — the second dot cannot start until the first
writes `b_v`.

After: single `b_h [BV, K=128]` tile, ONE tl.dot with k=K. Same total
tcgen05 micro-issue count (8 k-iters on native k=16 bf16) but no
dependency chain — the compiler schedules it as a single big MMA with
better load/store overlap. Empirical wins: +0.16x on state_rec
(iter-7, large-T gains 10-22% per workload — T=5709 +22%, T=3999
+16%), +0.10x on fused kernel (iter-8, small-T gains +3-14%).

Portable to any Triton kernel that:
- stores a state tile as two k-halves for historical reasons (FLA
  ports targeting older GPUs without native k=128 bf16, or to avoid
  register budget blowups on tiles that no longer exceed budget), AND
- has multiple MMAs that read both halves (the `b_v += dot(b_w2, ...)`
  pattern).

Anti-pattern: do NOT merge k-halves when the two halves are loaded
asynchronously from separate HBM regions. Split-k exists specifically
to overlap async loads with compute — collapsing destroys the
pipeline. Only collapse when both halves are in-register state that
would be loaded together anyway.

## tl.join + permute + reshape is the deterministic hcat/vcat idiom

Triton's `tl.cat(a, b)` docstring says "Current implementation of cat
supports only `can_reorder=True`" — output order is unspecified, so
cat cannot be used when you need positional control (e.g. building a
block-lower-triangular matrix). Use `tl.join(a, b)` instead, which
stacks tensors along a new MINOR dim, then permute the join dim next
to your target dim and reshape to merge:

```python
# hcat (column concat): [M, N] + [M, N] → [M, 2N]
joined = tl.join(a, b)                      # [M, N, 2]
perm = tl.permute(joined, (0, 2, 1))        # [M, 2, N]
out = tl.reshape(perm, (M, 2*N))            # [M, 2N]

# vcat (row concat): [M, N] + [M, N] → [2M, N]
joined = tl.join(a, b)                      # [M, N, 2]
perm = tl.permute(joined, (2, 0, 1))        # [2, M, N]
out = tl.reshape(perm, (2*M, N))            # [2M, N]
```

v3 iter-9 used this to build a `[BT=64, BT=64]` block-lower-triangular
A_inv tensor from 10 `[BC=16, BC=16]` blocks (4 diag + 6 off-diag, with
the missing 6 upper-tri blocks filled as zeros). One big `[BT, BT] @
[BT, BV]` matmul then replaced 10 sub-matmuls, going from m=16
(1/4 native 64m) to m=64 (native), with 4 micro-issues instead of 10
heavily-under-utilized ones.

Anti-pattern: do NOT use `tl.cat` when output order matters — its
`can_reorder=True` default silently permutes elements and will pass
a correctness smoke test on small tiles while failing on larger ones.

Register budget warning: building `[64, 64]` bf16 ≈ 8 KB in flight
per CTA. Check occupancy impact if the surrounding kernel is register-
limited (kkt_solve is; v3 iter-9 still netted +0.06x because the
build cost was <5% of the saved matmul cost).

## Padding zero-blocks into a big-matmul refactor only wins when most blocks are valid

v3 iter-9's big-matmul refactor in `_kkt_solve_kernel` phases 5/6 won
+0.06x. v3 iter-10 tried to port the same pattern to
`_kkt_solve_tiny3_kernel` (T_seq ≤ 3*BC, only 6 of 10 sub-blocks
valid) by padding row-block 3 with zeros; regressed −0.04x, with the
T=48 bucket dropping 5.15→4.73x.

The big matmul computes all 64×64 output positions regardless of how
many sub-blocks have nonzero data. When ≤3/4 of the sub-blocks are
live, the wasted compute on zero rows outweighs the savings from
reducing issue count (10 small→4 big MMAs). Break-even roughly at
valid-block-count = 8-9 of 10 — below that, keep the per-sub-block
form.

Broader: big-matmul refactors via padding help when
`(valid_blocks / total_blocks)^2 > issue_ratio`, where issue_ratio
is `big_issues / sum(small_issues)`. For kkt_solve full (all 10 live),
this is `1 > 0.4` — big wins. For tiny3 (6 live), `0.36 > 0.4` fails
— padded big loses.

## `num_warps=8` on `_fwd_o_kernel` is a CORRECTNESS trap, not just a perf regression

Prior sessions documented `num_warps=8` as a PERF regression on
`_kkt_solve_kernel` (−35%, cuda_graph_v1 iter-6) and
`_state_recurrence_kernel` (−9%, cuda_graph_v3 iter-6) — in both cases
the second warp-group stalls when the MMA output tile only needs one
warp-group. cuda_graph_v4's session (ako4fib-run-prefill1, 2026-04-25)
iter-8 tried `num_warps=8` on `_fwd_o_kernel` paired with `BV_o=128`.
It did NOT regress perf; it produced **`INCORRECT_NUMERICAL` output on
5 workloads**: T=4124 N=15; T=8192 N=20, 32, 43, 57.

Mechanism: Triton's MMA tile-picker selects a different `mma.sync`
variant under `num_warps=8` for fwd_o's specific output tile shape,
and that variant has a numerical bug or race for this shape. The
kernel compiles cleanly; the correctness check fails at runtime. Same
mechanism family as the dsa-topk-indexer fp8 tile-picker trap (see
`../dsa-topk-indexer/TRAPS.md`).

Implication: `num_warps=8` on fwd_o is BANNED on correctness grounds,
not merely "slower." Always verify PASSED count after any `num_warps`
change — compile success ≠ correctness. Broader: whenever a Triton
config knob changes tile-picker behavior (num_warps, num_stages in
some cases, dtype), PASSED count must be the primary gate before any
perf interpretation. A "fast but 95/100" result is not a win; it's a
silent correctness regression.

## PDL borderline-wave producers cause consumer-block contention regression

Program-Dependent Launch (`launch_pdl=True` plus `gdc_launch_dependents()`
at producer tail and `gdc_wait()` before consumer's first load) lets
consumer blocks preempt idle SMs while the producer's tail is still
draining. It wins when producer grid is well-saturated (>2 waves,
plenty of slack SMs during drain) OR well-undersaturated (<1 wave,
consumer gets all 148 SMs to itself). It LOSES in the borderline
range 1.0–1.5 waves: consumer blocks dispatched onto SMs that still
hold producer L1 / shmem / register state — resource contention cost
exceeds the launch-hide saving.

Evidence from cuda_graph_v4 (ako4fib-run-prefill1 iter-11): PDL
chained across `kkt_solve` / `_state_recurrence_kernel` / `_fwd_o_kernel`
plus the tiny_kkt → fused_single_chunk chain. Net +0.06x across 100
workloads (AB-compare drift-free), but two workloads regressed sharply:

- **T=1800 N=3: −0.38x** — producer ~1.3 waves, consumer ~1.6 waves.
- **T=973 N=2: −0.17x** — producer ~0.86 waves, consumer ~1.6 waves.

iter-14 ablation removed only `_fwd_o_kernel`'s `gdc_wait()` to test:
the two outliers recovered (T=1800 +0.35x, T=973 +0.16x) but medium-T
wins across many other workloads dropped −0.05 to −0.15 each. Net
−0.04x vs the full chain → full PDL chain kept (iter-11 state).

Implication: when adding PDL to a new kernel chain, audit wave count
(`grid_blocks / num_SMs`, scaled by per-block resource footprint if
it's not one-block-per-SM) for EVERY producer→consumer pair. If any
pair sits in the 1.0–1.5-wave borderline, expect a per-workload
regression on that config; consider per-workload `USE_PDL: tl.constexpr`
gating at graph-capture time. PDL is a resource-sharing tradeoff, not
a free launch-hide — it assumes producer has fully drained before
consumer needs the SM.

Detection recipe: when a PDL change shows mean +Δ but a wide per-
workload dispersion, run `--ab-compare` with PDL removed from one
kernel at a time to identify which consumer is paying the contention
cost. Cross-check with the wave-count formula; any borderline pair
is the likely culprit.

## FP8 e4m3 casting: safe for bounded inputs, corrupts derived intermediates

Casting bf16 MMA operands to `tl.float8e4nv` (e4m3, range ±448) before
`tl.dot` doubles tcgen05 throughput. The practical question is whether
the operand's dynamic range fits ±448 without scaling.

**Safe** (cuda_graph_v5 iter-2, +0.024x): MMA operands that come
directly from model activations or weights. Bf16 projections in
transformer-family workloads are typically bounded by upstream
normalization to roughly [-10, 10]; they sit deep inside e4m3 range.
Cast is no-op on error, wins on throughput. 100/100 PASS in fwd_o's
three MMAs (q@hᵀ, q@k, A@v_new).

**Unsafe — silent corruption** (cuda_graph_v5 iter-3, 3/5 smoke fails
at abs_err 2-3e-02 vs atol 1e-02): MMA operands that are DERIVED
intermediates — matrix-inverse outputs, polynomial-expansion results,
or any tensor whose magnitude depends on the conditioning of an
upstream solve. Concrete evidence from kkt_solve phases 5/6:
`b_Ai_full` is the inverse of unit lower-triangular (I + A).
Off-diagonal entries come from `I − A + A² − ...` expansion; when
‖A‖ approaches 1 they amplify by 2-20× and exceed ±448. Cast
saturates silently; errors compound through state recurrence to
2-3e-02 magnitude on T=8192 multi-seq workloads.

Implication: verify operand's dynamic range before fp8-casting. Rule
of thumb — cast input data (activations, weights, bf16 projections);
do not cast derived intermediates (inverses, recurrence outputs,
factorization results) without per-tile scaling. The fix path for
intermediates is `tl.dot(a, b, scale_a, scale_b)` with tile-max
pre-cast scaling — high implementation cost vs uncertain net win;
deferred in cuda_graph_v5 Open directions.

Cross-reference: TRAPS #10 (Triton MMA tile-picker variants under
`(M, N, K, dtype, num_warps)`) is a separate trap. Both apply to any
FP8 change — always verify PASSED count, not just score, after dtype
switches.

---

## Pointer-keyed skip-copy is borderline by spirit-reading and silently fails under input-randomized eval (added 2026-04-25, v7)

**Fact:** `cuda_graph_v5` shipped with a hot-path optimization that
elided the input → static-buffer copy when
`(q.data_ptr(), k.data_ptr(), …)` matched the previous call's tuple:

```python
cur_ptrs = (q.data_ptr(), k.data_ptr(), v.data_ptr(), …)
if cur_ptrs != g['last_ptrs']:
    torch._foreach_copy_([g['q'], …], [q, …])
    g['last_ptrs'] = cur_ptrs
g['graph'].replay()
```

The captured `g['graph']` reads from the static staging buffers
(`g['q']`, `g['k']`, …) — not from the user's tensors directly — so
the staging buffers must hold current input data before replay. The
pointer-keyed skip relies on a load-bearing assumption: **the eval
harness reuses the same tensor objects across iterations within one
workload AND those tensors hold the same data across iters**.

The current `flashinfer-bench` eval satisfies both: it loads
safetensor inputs once per workload and passes the same tensor object
across all 5 timed iters with stable contents. The skip-copy is
correct under this eval.

**Why removed in v7:** there are two problems with relying on this:

1. **Strict-reading rule violation.** The official contest rule says
   "skipping it because 'inputs look the same' or returning cached
   results from a prior call is not allowed." `it` literally refers
   to the kernel call ("every kernel call in the timing loop must do
   real per-iteration computation"), and the skipped operation here
   is a memcpy, not a compute kernel — so spirit-reading allows it.
   But the *condition* under which the skip triggers is exactly the
   "inputs look the same" pattern the rule names. Reasonable
   reviewers can read this either way.
2. **Silent failure under input-randomized eval.** If the eval ever
   mutates input tensors in-place between iters while reusing the
   same tensor objects (PR #413 direction:
   https://github.com/flashinfer-ai/flashinfer-bench/pull/413), the
   skip-copy condition still triggers (same data_ptr) but the
   staging buffers retain stale bytes from the prior iter. The
   captured graph would replay against stale data and produce wrong
   outputs. The kernel has no detection path — it's a silent
   correctness failure, caught only by the eval harness's
   `max_abs_error` tolerance check.

**Functionally analogous to** the `moe-fp8-block-scale` quarantine
(see that archive's TRAPS "Hoisting routing out of the captured
graph + pointer-keyed reuse"), with one important distinction: the
moe routing-skip elides a *compute kernel*'s GPU time from CUPTI's
span (clear cheat under any reading), while gdn-prefill's skip-copy
elides only a *memcpy*'s span (compute kernels still launch and
produce fresh output every call, just on potentially-stale staging
data). The moe case is forbidden by both letter and spirit; the
gdn-prefill case is borderline by letter and probably fine by spirit
under the current eval.

**v7 fix:** always issue the foreach_copy + state.copy_ + cu.copy_
before `g['graph'].replay()`. Cost: ~20µs/call at T=14107 (~1% of
headline) and ~0 at small T.

**How to apply going forward:**
- Treat any future hot-path optimization keyed on `*.data_ptr()`
  matching a prior call's pointer with skepticism — it's a tell that
  the optimization assumes eval-side input stability. If the
  optimization elides a compute kernel, it's a clear cheat (see the
  moe TRAPS); if it elides only a memory operation, it's still
  borderline by strict-reading and silently fails under
  input-randomized eval, so prefer to leave it out.
- If you genuinely need a per-call refresh-skip optimization, key it
  on shape (which can change but must legitimately invalidate the
  cache), not on input data pointers.

**Seen in:** v7 session (2026-04-25), revising the cuda_graph_v3
session's iter-3 skip-copy win (originally documented at +58% on
iter-3) after team review for cheat-pattern consistency with the
v6 moe_fp8 quarantine.
