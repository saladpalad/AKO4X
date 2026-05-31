# Cross-variant gotchas — gdn-decode

Cross-variant facts that outlive any single variant. If a future session
"discovers" any of these on its own, that's a signal the warning didn't
land — please rewrite for clarity.

Created 2026-04-24 (v1 + v1-2 concurrent sessions); each entry has a
**Why** so future sessions can judge whether their context flips the
fact.

## Eviction-policy hints are combinatorial with grid layout

**Fact:** Marking `q/k` as `evict_last` and `state/new_state` as
`evict_first` on `tl.load` / `tl.store` is a sub-percent **finishing
touch** once the grid already puts CTAs sharing the same (b, h) adjacent
in issue order. Applied WITHOUT a grid layout that creates reuse, the
hints are net-negative.

**Why:** Eviction policies only matter if there's something to persist.
The default grid `(B*HV, V/BV)` interleaves (b, h) pairs across CTAs,
so adjacent CTAs need different q/k — nothing for `evict_last` to
preserve, and marking streaming state `evict_first` against an already
evicted baseline just adds complexity. Once SWAP_GRID at B≥32 puts the
8 CTAs for the same (b, h) adjacent, q/k are reusable across those 8,
and the hints push 64 KB streaming bytes off the cache path they'd
otherwise occupy.

**Seen in:** v1-2 iter-5 tried the hints alone on top of the prior
anchor (no SWAP_GRID) — drift-free A/B Δ = −0.52% (within noise but
definitely not positive). v1 iter-10 stacked them on top of iter-7's
SWAP_GRID — per-B Δ B=32 +1.0%, B=48 +0.6%, B=64 +0.5% (small but
real at large B where state streaming dominates).

**How to apply:** Before trying eviction hints on any future
gdn-decode variant, verify the grid actually creates per-CTA reuse
for the tensor you want to persist. If the default pid order
interleaves the reuse pattern you care about, fix the grid first;
the hints come after.

## Triton's `num_stages ≥ 2` regresses short-trip register-resident loops

**Fact:** Adding a V-loop (`for j in range(BV // BV_INNER)`) with
`num_stages=2` to pipeline state loads through SMEM regresses this
kernel by 10-15% at B≥16, even when the loop has only 2 iterations.

**Why:** The prior anchor keeps `state_tile[BV, K]` register-resident
throughout the forward math — one big load, two reductions, one RMW
write. Introducing a loop with `num_stages=2` forces Triton's
pipeliner to allocate SMEM double-buffers for the state chunk, adding
register↔SMEM round-trips on every iteration. Triton's alias analysis
is also too conservative to schedule load-(i+1) ahead of store-(i) on
the shared `new_state` buffer, so the pipeline overlap doesn't
materialize.

**Seen in:** Two independent concurrent sessions hit this
independently. v1 iter-1 (BV=16 GROUP_V=2 num_stages=2) measured
−28% at B=16, −14-20% at B≥32. v1 iter-2 (BV=8 GROUP_V=2
num_stages=2, grid-preserving variant) recovered to −10-15% at B≥16
but still regressed. v1-2 iter-2 independently rediscovered this
pattern with BV_INNER=8, num_stages=2 → −6 to −20% per B.

**How to apply:** On fp32 state read-modify-write kernels with
short inner loops (≤4 iterations), keep state register-resident.
Don't try to coax pipelining with `num_stages ≥ 2` unless you can
also avoid the SMEM hop entirely — which Triton can't on NVIDIA
(its async/TMA primitives go through SMEM). Warp specialization or
a CUDA+TMA rewrite is the only known path to real pipelining on
this shape.

## Graph-cache pointer aliasing requires `use_isolated_runner = true`

**Fact:** Every gdn-decode variant so far uses an input-pointer-keyed
CUDA graph cache (`_graph_cache[key]` where `key` includes
`q.data_ptr(), k.data_ptr(), …`). On persistent benchmark runners
(single Python process across workloads), PyTorch's caching allocator
can recycle tensor addresses between workloads — a replay for workload
A then uses stale pointers baked into the graph.

**Why:** `torch.cuda.graph()` captures the literal pointer values from
the kernel launch arguments. Once captured, `g.replay()` reuses those
bytes. If PyTorch's allocator hands the same address to workload B,
the cache key matches but the buffer contents are B's, and the graph's
kernel runs against B's data as if it were A's.

**Seen in:** Inherited from v0's anchor session. Documented at
`variants/triton_bv_dispatch_graph/config.toml:18-23` as a correctness
flag, not a perf knob.

**How to apply:** Any new variant that reuses the pointer-keyed
graph-cache pattern must set `[benchmark] use_isolated_runner = true`
in its `config.toml`. Don't drop this flag on the assumption that
graph replay is "just" an optimization — correctness depends on it.

## Modal session drift: ±5-15% cross-container, ±1% within-container

**Fact:** The same unchanged gdn-decode kernel benched in different
Modal containers can report absolute speedup numbers differing by up
to ~15% (vs ~1% for three-run variance-check within one container).
Cross-session `diff.sh` on identical trajectories shows ±10% apparent
"delta" with zero code change.

**Why:** Modal B200 tenancy, thermal state, driver lazy-init, and
CUDA runtime warm-up drift between container allocations. Within one
container the JIT cache is warm, the driver is initialized, and the
allocator is stable; across containers all of these reset. This is
operator-specific amplification of the general B200 drift documented
in `../dsa-sparse-attention/TRAPS.md` (which reports ~±1x on a
45× baseline — same relative magnitude).

**Seen in:** v1-2 session `--ab-compare iter-0` on byte-identical code
gave Δ = −0.68%. v1-2 cross-session cold runs of the same v0 anchor
ranged 1.08× to 1.13× headline. v1 session iter-10's full-bench
1.14× vs variance-check 1.13× was confirmed as session-drift artifact,
not real improvement.

**How to apply:** Sub-percent claims on gdn-decode must use
`--ab-compare <label>` (same-container, drift cancels) or
`--variance-check N`. Never trust a single-session headline difference
< 5% as signal. Concretely: any full-bench score movement ≤ 1.5×
within a session's same code is noise; always re-verify via A/B.

## Persistent outer-loop breaks SWAP_GRID's DRAM row-buffer coherence

**Fact:** Converting the default `(B*HV, V/BV)` grid into a
persistent-kernel outer loop (`for work_id in tl.range(pid,
TOTAL_WORK, GRID_SIZE, num_stages=N)`) regresses this kernel 24-37%
at B=48 for BOTH `NUM_STAGES=1` and `NUM_STAGES=2`. Two distinct
failure modes — don't confuse with the inner V-loop trap above.

**Why:** Two separate mechanisms stack on this shape. (1) With
`NUM_STAGES≥2`, Triton's pipeliner allocates SMEM double-buffer on the
`[BV,K]` fp32 state tile (16 KB/block) — same mechanism as the
V-loop trap, now also hitting the outer persistent loop. (2) With
`NUM_STAGES=1` the SMEM cost disappears, but the grid reordering
destroys SWAP_GRID's win: consecutive `program_id`s (which round-robin
onto adjacent SMs) now receive `work_id`s `GRID_SIZE` apart. Each
CTA's loop-iters cycle through completely DIFFERENT `(b, h)` pairs —
no two consecutive iters share the q/k L2 footprint that SWAP_GRID's
non-persistent layout carefully lined up for DRAM row-buffer coherence.
The +0.58% mean that SWAP_GRID bought evaporates; large-B regresses.

**Seen in:** v2 iter-1 (2026-04-24). At B=48 subset:
`NUM_STAGES=2` → 0.635×, `NUM_STAGES=1` → 0.757× (vs anchor 1.01×).
Same pattern at B=64. Independent failure modes confirmed by
N=2-vs-N=1 A/B within the same session.

**How to apply:** On memory-bound kernels whose win depends on
grid-order-driven L2/DRAM reuse (like SWAP_GRID on this shape),
persistent-kernel outer loops are a net negative. Persistent wins
only when (a) compute/iter ≥ HBM fetch latency, AND (b) the
persistent loop's iter ordering preserves — not breaks — the
grid-order reuse pattern. gdn-decode fails both conditions. If a
future variant drops SWAP_GRID (e.g., replaces it with a fundamentally
different layout), re-evaluate; the trap is specifically against
persistent-on-top-of-SWAP_GRID, not persistent in general.

## NCU's "Est. Local Speedup" overestimates SMEM-staging fixes

**Fact:** When NCU reports a kernel is register-occupancy-bound and
estimates a high "Est. Local Speedup" from raising occupancy, do not
trust the number for the obvious "stage register-resident state into
SMEM" fix. On this kernel, NCU at iter-5 (B=48) showed achieved
occupancy 25%, theoretical 31%, "Block Limit Registers = 5", and
"Est. Local Speedup: 68%". Staging the 32-reg state_tile into 16 KB
SMEM regressed −16% at B=32 instead.

**Why:** The estimate models the occupancy benefit but treats the
staging cost as zero. SMEM-staging adds, per cell touched: 1× LDS
on the read path of each reduction + 1× LDS on the write path +
1× STS at the load + 1× `__syncthreads` barrier. On a register-
resident reduction kernel where the working tile already fits in
~32 regs/thread, the staging traffic + barrier cost exceeds the
register-pressure savings, and achieved occupancy doesn't actually
double either (other live ranges fill the freed registers).

**Seen in:** v3-2 iter-6 (2026-04-25). NCU prediction 68% local
speedup → measured −16% per the iter-6 dead-end in
`variants/cuda_bv32_register_resident/kernel.py` header.

**How to apply:** Treat NCU's "Est. Local Speedup" as an upper bound
on what's possible if the proposed fix is free. For SMEM-staging
fixes on register-resident reduction kernels with small per-thread
state, always measure the staged variant before committing — the
predicted speedup typically does not materialize.

## BV / CTA-count sweet spot at ~1024 CTAs/wave on B200 for this shape

**Fact:** On this fp32-state RMW shape the BV/B dispatch is best
tuned to keep CTA count near `B * HV * V/BV ≈ 1024` per wave.
Empirically:
  - B=16 BV=16 → 1024 CTAs: wins drift-free +7% vs Triton.
  - B=16 BV=32 →  512 CTAs: regresses −10% (under-fill, 3.5 CTAs/SM).
  - B=32 BV=16 → 2048 CTAs: regresses (per-CTA overhead × count).
  - B=32 BV=32 → 1024 CTAs: wins drift-free +5% (the iter-5 anchor
    breakthrough).

**Why:** This kernel sits between launch-overhead-dominated (small B)
and HBM-bandwidth-bound (large B). Per-CTA setup (gate compute, q/k
decode, address arithmetic) is non-trivial; halving CTAs via a larger
tile lifts efficiency *until* SM utilization drops. B200 has 148 SMs;
at ~5 blocks/SM occupancy the well-filled lower edge sits around
`5 × 148 ≈ 740` resident CTAs (~1.4 waves at 1024). Going below
~3 CTAs/SM under-fills.

**Seen in:** v3-2 iter-5 win (BV=32 at B=32, +7.9% drift-free) and
v3-2 iter-9 regression (BV=32 at B=16, −10% on the B=16 subset).

**How to apply:** When picking BV per-B for any future variant on
this op, target CTA count near 1024 per wave; verify SM fill
(`grid_size / num_SMs / blocks_per_SM`) before going lower. The
"halve CTAs via larger tile" lever works once; applied a second
time (e.g., BV=64 at B=64) under-fills again unless block size
also grows.

## TileLang's pipelining + warp-spec doesn't fit matvec/decode reductions

**Fact:** Porting gdn-decode through TileLang's pipelining +
warp-specialization passes regressed across multiple iters
(1.01× → 0.78× at the worst). Pure Triton register-resident
remained ahead of the TileLang variants throughout that exploration.

**Why:** TileLang's pipeliner targets `T.gemm`-shaped matmuls; matvec
reductions (one query row × matrix-shaped state) don't trigger the
warp-specialization pass, so the kernel pays the SMEM-hop cost of
TileLang's pipeline staging without the pipelining win. The 10–15%
SMEM-hop overhead consumes any gains from issue-rate or async-load
benefits.

**Seen in:** ako4fib-run-gdn_decode_v3 iter-6/7/8 (2026-04 series).
Three independent attempts (TileLang port, TileLang + pipelining,
TileLang + warp specialization) all regressed.

**How to apply:** For matvec-shaped decode kernels (`q · state`
where `q` is a single row), CUDA register-resident is the correct
shape; do not port through TileLang expecting pipelining wins. If
TileLang is otherwise required (e.g., to compose with existing
TileLang infra), use it as a transport for register-resident
patterns (no `T.Pipelined`, no `T.warpspec`), not as a pipelining
lever.
