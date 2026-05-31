# Fused fwd+reduce via CuTe DSL cluster launch — design

## Motivation

Current two-kernel architecture:
```
  fwd (TileLang)  →  PO/PM/PL in HBM  →  reduce (CuTe DSL)  →  OUT/LSE
```

Costs this incurs (estimated on B200, CUDA graph-replay steady state):
- Kernel launch overhead for reduce: ~0.5–1 µs
- PO write + read through HBM: 2 MB × 2 ≈ 4 MB traffic ≈ 0.5–1 µs at L2/HBM
- Graph edge between nodes: ~0.5 µs

Total potential savings from fusion: **~1.5–2.5 µs / 8 µs ≈ 20–30%**, translating to roughly +15–20x on the headline score.

## Design

### Overall structure
Single CuTe DSL kernel, cluster-launched.

- **Grid**: `(T, 1, 1)` — one cluster per token.
- **Cluster**: `(NS, 1, 1)` = 16 blocks per cluster.
  - Each block handles one (tok, split) — same sharding as current fwd.
  - Cluster spans 16 SMs of a single or multiple TPCs (Blackwell GPCs contain up to 18 SMs; cluster-16 fits within one GPC on the SM100 layout we care about).
- **Threads per block**: 256 (match TileLang fwd).

### Phases within the kernel

```text
┌─ Phase A — Per-split attention (computed independently per block) ──────┐
│ 1. Load Q_nope, Q_pe into block-local SMEM (shared across splits if     │
│    block_in_cluster_idx == 0 broadcasts via DSMEM; otherwise per-block).│
│ 2. Sparse gather BI=128 K entries using Indices[tok, split*BI:...]      │
│    into KV_s, Kp_s (block-local SMEM).                                  │
│ 3. WGMMA: S = Q · K^T  + Q_pe · Kpe^T                                   │
│ 4. Row-max → m_i                                                        │
│ 5. exp2 + row-sum → l_i                                                 │
│ 6. WGMMA: O = softmax(S) · V                                            │
└─────────────────────────────────────────────────────────────────────────┘

  cluster_arrive(); cluster_wait();   ← single cluster-wide barrier

┌─ Phase B — Intra-cluster reduce (one "lead" block per cluster) ─────────┐
│ 7. Block 0 reads other blocks' (m_i, l_i, O_i) via DSMEM                │
│    (mapa_shared_cluster on each peer's smem pointer).                   │
│ 8. m_g = max_s m_i_s                                                    │
│ 9. l_g = Σ exp2(m_i_s − m_g) · l_i_s                                    │
│ 10. For each h, d: O_g[h,d] = Σ exp2(m_i_s − m_g) · O_i_s[h,d] / l_g    │
│ 11. Write OUT[tok, :, :], LSE[tok, :].                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why the math works unchanged
Softmax is shift-invariant; the online combine `m_g, l_g` is identical to
the current reduce kernel's per-element computation. The only difference
is the data path: partials stay in on-chip SMEM instead of PO/PM/PL HBM.

### DSMEM access pattern
Each peer block stores its `(m_i, l_i, O_i)` at a known SMEM offset.
Block 0 uses `cute.arch.mapa` (or `mapa_shared_cluster`) to get a pointer
into peer block `s`'s SMEM and reads directly.

Per output element (h, d):
- 16 peer DSMEM reads of `m_i_s`, `l_i_s`
- 16 peer DSMEM reads of `O_i_s[h, d]`
- Compute weighted sum + divide

Per (tok, h, d) element: 48 DSMEM reads (vs 48 L1/L2 reads today).

### Per-block SMEM footprint (must stay ≤ 228 KB / SM)

| Buffer | Size | Notes |
|---|---|---|
| Qn_s (bf16 16×512) | 16 KB | Can be dedup via DSMEM if block_in_cluster_idx=0 loads |
| Qp_s (bf16 16×64) | 2 KB | Same |
| KV_s (bf16 128×512) | 128 KB | Per-block — K entries differ per split |
| Kp_s (bf16 128×64) | 16 KB | Per-block |
| S_s (bf16 16×128) | 4 KB | Per-block |
| m_s / l_s (fp32 16) | 64 B each | In DSMEM-visible region |
| O_s (fp32 16×512) | 32 KB | In DSMEM-visible region (peer reads) |
| **Total** | **~198 KB** | Fits within 228 KB cap |

## Complexity of implementation

Reference points from the CUTLASS 4.x CuTe DSL examples tree:
- `blackwell/dense_gemm.py` — 1890 lines (simplest Blackwell GEMM)
- `blackwell/dense_gemm_persistent.py` — 2000+ lines
- `blackwell/fmha.py` — 3100 lines (dense FMHA, closest analogue)

A fused sparse-attention kernel must add on top of FMHA:
- Sparse-gather K (not just contiguous TMA)
- Split-K + cluster-level cross-split combine
- BF16 pre-scale for log2 domain
- LSE output

Rough estimate: 2500–3500 lines of CuTe DSL code, plus 2–5 days of
Modal-based debugging cycles (each iteration is 4–8 min of remote
compile+bench).

## Out-of-scope for this session — why

1. Single Modal bench cycle is 4–8 min; a 500-iter debug cycle is infeasible in one session.
2. First-time CuTe DSL MMA tiling + cluster code is high-risk: any bug gives
   `INCORRECT_NUMERICAL` or compile failure, and CuTe's diagnostics are MLIR-level.
3. Alternative structured search (parameter tuning, reduce loop merges, etc.)
   has already converged to local optimum; further gains from there are noise.

## Minimal next-step experiments (low-risk, high-signal)

If resuming this direction, validate the infrastructure piecewise:

1. **Cluster-launch the existing reduce** with `cluster=(1,1,4)` and DSMEM-shared PM/PL.
   This alone is ~20 lines of code. Confirms cluster launch works; no expected perf win since PM/PL are L1-cached already (marginal traffic).

2. **Port one T.gemm to CuTe DSL MMA** in a stand-alone test, using the FMHA
   example as template for the Blackwell MMA atom setup.

3. **Add intra-cluster PO partial reduce** to replace the separate reduce kernel, still with TileLang fwd. Cluster=(NS,1,1) on the fwd grid.

Only after (2) succeeds is a full fused kernel tractable.

## Expected speedup breakdown (if implemented)

| Saving | µs | % of 8 µs budget |
|---|---|---|
| Eliminate reduce kernel launch | ~0.7 | 9% |
| Eliminate PO HBM round-trip | ~0.7 | 9% |
| Graph edge between nodes | ~0.3 | 4% |
| Potential better warp scheduling in one kernel | ~0.3 | 4% |
| **Total** | **~2.0** | **~25%** |

Translated to score: current 76.7x → **~96x expected**.
