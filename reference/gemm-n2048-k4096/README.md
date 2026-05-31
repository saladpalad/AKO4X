# gemm-n2048-k4096 — Archive

GEMM `C = A @ B.T` on Qwen3-30B-A3B `attn.o_proj` shape (`A:[M,K] × B:[N,K]`, N=2048, K=4096, fp16). Variable axis is `M`; 29 workloads span M = 1 → 16294 (rough regimes: tiny M ≤ 32, ~13 workloads; mid 63–969, ~7; huge 8828–16294, 8). Baseline is cuBLAS via PyTorch `torch.matmul(A, B.T)`.

Campaign started 2026-05-20 on Modal B200, CUDA 13.0. **Closed 2026-05-21 after 3 rounds with no DSL variant beating cuBLAS** — see "Campaign closure" below.

## Canonical baseline

`baseline.json` — captured during round-1 iter-1 reference profile on Modal B200 / CUDA 13.0 / 2026-05-20T23:09:49 (auto-promoted from the Round-0 environment-only shell). 29 workloads, `source: "expert"`. cuBLAS reference shape: a flat ~9µs floor across every M ≤ ~289 workload (~18 workloads — the bench harness uses `use_cuda_graph=False` so this is a *bare-kernel cold-L2 measurement floor under CUPTI hardware tracing*, not a graph-replay floor), rising to ~153µs at M=16294.

## Rounds

### Round 1 (2026-05-20) — no archive

Sub picked **structural specialization for M ≤ 32** and chose Triton. iter-1 implemented a 2-launch chain (split-K kernel with SPLIT_K=8, BLOCK_M∈{16,32}, BLOCK_N=128, BLOCK_K=64 + fused reduce_cast). Benched **0.848x** (regression), 29/29 PASS — every M ≤ 32 workload landed at ~15-16µs versus the cuBLAS ~9µs floor.

**Closed structural lever**: 2-launch chains (split-K + post-reduce) can't beat cuBLAS on small M for this shape. cuBLAS at M=1..32 is ~22% of B200 peak HBM BW (1.8 TB/s effective on 16 MB B), so the 9µs floor is a *bare-kernel cold-L2 measurement floor* under CUPTI hardware tracing (harness uses `use_cuda_graph=False`, see `flashinfer_bench/bench/timing.py:80`), NOT a bandwidth limit and NOT a graph-replay floor. Each extra kernel launch costs ~2-3µs of GPU-side kernel-start latency (not the ~1µs that a graph-replay cost model would predict — the round-1 narrative initially wrote 1µs assuming graph replay; corrected after round-2 verified the harness setting). Either way, a second launch is structurally unbridgeable here.

Sub also wrote an iter-2 kernel (single-launch, BLOCK_N=16 → 128 disjoint N-tiles → ~97% B200 SM utilization in one wave, no atomics, no scatter buffer; predicted ~5-6µs and ~1.27x headline) but the bench never ran — late-session permission denials blocked smoke tests and git commits after ~$5.65 / 99K output-tokens of phase-1 use. PROPOSALS.md == `none`. iter-2's prediction was left as a refutable open hypothesis for round 2.

### Round 2 (2026-05-20) — no archive

Sub picked **Lever 1 (single-launch small-M Triton)** to put a number on round-1's iter-2 prediction. iter-1 implemented a single-launch autotuned Triton kernel for M ≤ 64 (autotune sweep over BLOCK_M ∈ {16,32,64}, BLOCK_N ∈ {16,32,128,256}, num_warps ∈ {2,4,8}, num_stages ∈ {3,4}), dispatching to torch.matmul for M > 64. Benched **0.717x** (regression, *worse* than round-1's 2-launch attempt), 29/29 PASS — small-M workloads landed at ~22µs vs cuBLAS ~9µs (Triton at ~7-9% peak HBM, cuBLAS at ~22% peak). Sub committed the iter-1 result then reverted to baseline so the campaign anchor stays the unmodified reference. Picked config: `BLOCK_M=16, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=4`.

**Closed structural lever**: Triton single-launch GEMM at small M (M ≤ ~289 for N=2048, K=4096, fp16 on B200) cannot beat cuBLAS — sub-1's prediction was refuted. The cuBLAS small-M floor is real kernel-level efficiency (TMA + persistent + tcgen05 likely) that Triton at this shape doesn't reproduce, not a launch-overhead floor that single-launch trivially clears. Same lever in different generic-DSL forms (Tilelang, CUTLASS Python autotune) is unlikely to clear it for the same reason.

**Remaining open levers**: (3) huge-M (M > 8K, 8 workloads, cuBLAS at ~80% peak fp16 tensor — *thin* headroom, would need a specific structural advantage beyond cuBLAS heuristic), (4) DSL pivot to CUTLASS / cute-dsl with tcgen05 (the BF16/FP16-peak unlock on Blackwell; high iteration cost — multiple cute-dsl gotchas — but highest unlock potential remaining).

### Round 3 (2026-05-21) — no archive (best campaign kernel: 0.954x, sub-archived in child env)

Sub picked **Lever 4 (cute-dsl tcgen05)** per the round-2 brief, vendoring CUTLASS's `examples/python/CuTeDSL/blackwell/dense_gemm{,_persistent}.py` into `solution/sm100_*.py` and dispatching `M < 1024 → torch.matmul` (small-M closed in rounds 1-2) / `M ≥ 1024 → compiled cute-dsl kernel`. Six iterations climbed a tile/cluster/persistence surface:

| Iter | Config | Score | Lesson |
|---|---|---|---|
| 1 | DenseGemmKernel, `mma_tiler=(128,256)`, `cluster=(2,1)`, 2cta, TMA-store | 0.905x | tcgen05 path works on B200; first config not competitive |
| 2 | swap → PersistentDenseGemmKernel (warp specialization) | 0.935x | persistent + warp-spec lifts huge-M avg 0.66 → 0.78 |
| 3 | bump `mma_tiler=(256,256)` | **0.954x** | max-legal MMA tile amortizes per-cta sync best |
| 4 | `cluster=(2,2)` (A multicast on N) | 0.954x | wash overall; only helps M > ~14000 where A exceeds B200 L2 (~120 MB) |
| 5 | `mma_tiler=(256,128)` | 0.896x | smaller `cta_tile_N` starves MMA throughput — confirms iter-3 is optimal |
| 6-smoke | 1cta vs 2cta probe at M=16294 | 0.698x (1cta) | 2cta is structurally better here — more output per cta-pair via TMEM-shared accumulator |

**Best config (iter-3)**: persistent cute-dsl + `mma_tiler=(256,256)` + `cluster=(2,1)` + `use_2cta_instrs=True` + TMA-store; M<1024 → torch.matmul. **0.954x** (29/29 PASS, `abs_err=rel_err=0`). Closed the campaign-stated "meaningful result" bar (within ±10% of cuBLAS on full-shape mean) — but did NOT beat cuBLAS, so no source-of-truth variant archived. Sub self-archived the kernel + vendored persistent class in the child env at `docs/prior/variants/iter3-persistent-tcgen05/` for forensic reference (wrap-up bundles the full child env into `experiments/gemm_n2048_k4096/`).

**Closed Lever 4** — with a specific ceiling explanation from NCU on iter-3 M=16294 (`profiles/w12_20260521_003548.json`): `sm__throughput=83%`, `sm__pipe_tensor_cycles_active=54%` (MMA pipe idle 46%), DRAM bytes_read = 948 MB (4.4× the 217 MB input footprint → heavy L2 miss rate), DRAM BW = 4.86 TB/s read (63% of B200 HBM3e peak), dominant stall `long_scoreboard` (memory-load wait to TMEM/SMEM), mean active warps 1.38/16. The vendored `PersistentTileSchedulerParams` uses linear raster traversal; cuBLAS likely uses a swizzled (raster/Hilbert) traversal that captures much more L2 reuse on huge-M. Beating cuBLAS at this shape would require a custom tile scheduler with swizzled L2-reuse-aware traversal — substantial surgery on the ~1.6 KLOC vendored kernel.

## Campaign closure

Three rounds (2026-05-20 → 2026-05-21). No DSL variant beat cuBLAS; the campaign anchor remains the unmodified `torch.matmul(A, B.T)` reference (1.0x by construction). All four structural levers identified in round 2's open-lever list closed:

- **L0 (2-launch small-M)** refuted in r1 (0.848x) — bare-kernel cold-L2 measurement floor under CUPTI, not graph-replay launch overhead; second launch costs ~2-3µs that cannot be amortized.
- **L1 (single-launch small-M)** refuted in r2 (0.717x) — cuBLAS's small-M floor is real kernel-level efficiency Triton can't reproduce, not a launch-pattern artifact.
- **L3 (huge-M Triton)** not pursued directly; r3's cute-dsl run on huge-M (the same workloads, with a *better* DSL for the regime) only reached 0.855x huge-M avg / 0.954x overall. A Triton autotune attempt is unlikely to clear what tcgen05 + persistent + max-legal MMA tile didn't.
- **L4 (cute-dsl tcgen05)** closed in r3 with a specific ceiling at 0.954x and the structural-surgery cost to break it documented above.

Conclusion: **cuBLAS is the right primitive for fp16 GEMM at N=2048, K=4096 on B200** within the budget of a closed-loop campaign that vendors but does not rewrite NVIDIA's reference Blackwell kernels. Reopening this campaign should require a qualitatively new lever — most likely a from-scratch custom tile scheduler with swizzled L2-reuse-aware traversal on top of the round-3 kernel, or a different precision / dtype (bf16 quant-aware, fp8 if the model variant allows). Re-running L0/L1/L4-without-scheduler-surgery will not move it.

