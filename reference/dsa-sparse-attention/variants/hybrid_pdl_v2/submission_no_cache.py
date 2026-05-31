"""submission_no_cache — SOL-compat derivative of hybrid_pdl_v2.

═══════════════════════════════════════════════════════════════════════════
0) SOL-compat preface (this file's reason for existing)
─────────────────────────────────────────────────────────────────────────
This file is a stripped-down sibling of `kernel.py` (hybrid_pdl_v2 anchor)
intended for callers OTHER than the FIB benchmark — production LLM
serving, the SOL-ExecBench online judge, persistent-buffer cheat-checks.

The hybrid_pdl_v2 anchor relies on the FIB harness's
`use_isolated_runner = true` contract: each workload runs in a fresh
subprocess, so module-level state (`_last_si_ptr`, `_graph_cache`) is
clean per workload. Within a workload, all 5 input tensor pointers
change together at trial boundaries (FIB allocates fresh tensors per
trial). Under that contract, the `_last_si_ptr` integer fast-path at
kernel.py L561-564 is correct:

    si_ptr = sparse_indices.data_ptr()
    if si_ptr == _last_si_ptr and _last_graph is not None:
        _last_graph.replay()
        return _last_out, _last_lse

Under any OTHER caller — buffer-pool reuse, persistent activation
caches, indptr re-use across workloads with different content — the
`sparse_indices` slot is often the last to be re-allocated (small,
frequently-reused int32). So `si_ptr` can match while `q_nope` /
`q_pe` / `ckv_cache` / `kpe_cache` have moved. The captured graph
embeds the OLD addresses for the 4 other tensors; on replay it reads
from those addresses — which now point to freed memory or different
content → INCORRECT_NUMERICAL or segfault. The full 6-tuple key
check at L567+ would catch the move, but the fast-path bypasses it.

`cheat_check_modal.py` does NOT detect this — it mutates inputs
in-place keeping pointers stable; the fast-path correctly hits and
replay reads mutated memory → PASS. Forensic-identified via static
audit (2026-05-23 ten-kernel sweep), not yet measured-failing.

**What this file changes vs hybrid_pdl_v2 anchor:**
- Drops the `_last_si_ptr` int fast-path (kernel.py L560-564) — always
  goes through the full 6-tuple key check at L567+. Cost: one dict
  lookup per call instead of one int compare; negligible on this
  operator's microsecond-scale latency. Robust under buffer-pool reuse.
- Everything else preserved: graph cache, eager fallback,
  `_DISABLE_GRAPH` env flag, all kernel definitions, all helpers.

**Expected score vs anchor:** anchor was 58.34 ± 0.04× (CV 0.06%, B200
3-run variance-check, 2026-04-25). The fast-path saved ~50-100 ns per
call (one int compare vs one dict.get). Across ~100 timed iters per
workload, total saving was <10 µs — well below CV. Expected
~58.3× ± 0.1× under FIB, indistinguishable from anchor within session
drift.

**Cross-campaign sibling derivatives** (same SOL-compat purpose,
different operators):
- `reference/mla-paged-prefill-causal-h16-ckv512-kpe64-ps1/variants/
  iter5-triton-graph-tilelang-archive/submission_no_cache.py`
- `reference/moe-fp8-block-scale-ds-routing-topk8-ng8-kg4-e32-h7168-i2048/
  variants/iter4-autotune-tactic-sweep-gated/submission_no_cache.py`

See `reference/dsa-sparse-attention/TRAPS.md` section "`_last_si_ptr`
int fast-path is buffer-pool-fragile" for full forensic.

═══════════════════════════════════════════════════════════════════════════
1) Implementation
─────────────────────────────────────────────────────────────────────────
Imports the canonical hybrid_pdl_v2 kernel.py from the same directory
to reuse all kernel definitions, helpers, and module state (`_graph_cache`,
`_static_out`, `_static_lse`, etc.). The only override is `run()`.

If you need a fully self-contained file (no `kernel.py` dependency),
copy the canonical kernel.py and inline-patch the run() body to drop
L560-564.
"""

import importlib.util
from pathlib import Path

import torch

_kernel_path = Path(__file__).parent / "kernel.py"
_spec = importlib.util.spec_from_file_location(
    "_hybrid_pdl_v2_canonical", _kernel_path
)
_canonical = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_canonical)


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """SOL-compat replacement for hybrid_pdl_v2.run().

    Drops the `_last_si_ptr` int fast-path (kernel.py L560-564); every
    call falls into the full 6-tuple key check, which keys on all 5
    input tensor data_ptrs + Tv. Safe under buffer-pool reuse:
    cache hit requires ALL 5 pointers to match → graph reads from
    those addresses → reads current content at those addresses
    (correct by graph-replay semantics). Cache miss → recapture.
    """
    si_ptr = sparse_indices.data_ptr()
    Tv = q_nope.shape[0]
    key = (
        q_nope.data_ptr(),
        q_pe.data_ptr(),
        ckv_cache.data_ptr(),
        kpe_cache.data_ptr(),
        si_ptr,
        Tv,
    )

    g = _canonical._graph_cache.get(key)
    if g is not None:
        _canonical._last_graph = g
        _canonical._last_si_ptr = si_ptr
        _canonical._last_out = _canonical._static_out[Tv]
        _canonical._last_lse = _canonical._static_lse[Tv]
        g.replay()
        return _canonical._last_out, _canonical._last_lse

    dev = q_nope.device
    ckv_flat = ckv_cache.reshape(-1, _canonical._DC)
    kpe_flat = kpe_cache.reshape(-1, _canonical._DP)

    if Tv not in _canonical._static_out:
        _canonical._static_out[Tv] = torch.empty(
            (Tv, _canonical._NH, _canonical._DC),
            dtype=torch.bfloat16,
            device=dev,
        )
        _canonical._static_lse[Tv] = torch.empty(
            (Tv, _canonical._NH), dtype=torch.float32, device=dev
        )
    output = _canonical._static_out[Tv]
    lse_o = _canonical._static_lse[Tv]
    _canonical._get_bufs(dev)

    use_triton = Tv < _canonical._TRITON_THRESH

    if _canonical._DISABLE_GRAPH:
        if use_triton:
            _canonical._launch_triton(
                Tv, q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices,
                sm_scale, output, lse_o,
            )
        else:
            _canonical._launch_tl(
                Tv, q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices,
                output, lse_o,
            )
        return output, lse_o

    cnt = _canonical._graph_cnt.get(key, 0) + 1
    _canonical._graph_cnt[key] = cnt

    if use_triton:
        _canonical._launch_triton(
            Tv, q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices,
            sm_scale, output, lse_o,
        )
    else:
        _canonical._launch_tl(
            Tv, q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices,
            output, lse_o,
        )

    if cnt >= 2:
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            if use_triton:
                _canonical._launch_triton(
                    Tv, q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices,
                    sm_scale, output, lse_o,
                )
            else:
                _canonical._launch_tl(
                    Tv, q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices,
                    output, lse_o,
                )
        _canonical._graph_cache[key] = g
        _canonical._last_graph = g
        _canonical._last_si_ptr = si_ptr
        _canonical._last_out = output
        _canonical._last_lse = lse_o

    return output, lse_o
