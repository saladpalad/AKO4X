# Cross-variant gotchas — moe-fp8-block-scale-ds-routing-topk8-ng8-kg4-e32-h7168-i2048

Cross-variant facts that outlive any single variant. If a future session
"discovers" any of these on its own, that's a signal the warning didn't
land — please rewrite for clarity.

## `data_ptr()`-keyed CUDA graph cache — FIB-contract-validated, not portable to non-isolated runtimes

**Status (TL;DR):** Contest-legal per official confirmation. Do NOT
remove from FIB submission. The +0.11x (round-3 graph) and +0.85%
(round-4 autotune) lifts are legitimate exploitation of the published
benchmark contract. Listed here so future readers porting the kernel
to a non-isolated runtime know to swap to `submission_no_cache.py`.

**Official confirmation (yongwww, 2026-04-19, MLSys 2026 FlashInfer
NVIDIA Track organizer, response to direct question on Track B
rules):**
> "Reusing a captured CUDA Graph when shapes and captured tensor
> addresses remain stable within the same isolated subprocess. In one
> implementation, this is keyed conservatively on all relevant tensor
> addresses; in another implementation, there is also an
> address-stability-based hot path that replays the previously
> captured graph when a stable sparse-index pointer is observed. In
> both cases, replay still executes the full GPU work on the current
> tensor contents and does not return memoized outputs. … i think the
> techs you mentioned above are all valid."

The iter4 anchor's `_GRAPH_BY_KEY` keyed on `(num_tokens, device.index)`
with `hidden_states.data_ptr()` re-validation is the second variant
described in the question — single-pointer stability detection.
Explicitly approved.

---

**Structural fact (for hypothetical non-FIB callers):**
iter3c / iter4 (anchor) cache the captured CUDA graph keyed on
`(num_tokens, device.index)` and re-validate via `hidden_states.data_ptr()`.
This is **correct only under runtimes that provide per-workload
subprocess isolation + fresh tensor allocation** — for the contest
the FIB harness with `use_isolated_runner=true` provides this
guarantee, and within a single trial 100 iters share the same input
tensor pointers, so the data_ptr probe correctly detects trial-
boundary tensor re-allocation within the same subprocess.

On any caller that **reuses `hidden_states.data_ptr()` across calls with
different content or different sibling-tensor pointers** — production
LLM serving (persistent activation pools), the SOL-ExecBench online
judge (one process serving multiple workloads from a shared pool), the
proposed-but-not-yet-built persistent-buffer cheat-check — the cache
returns a graph captured against stale routing/weight tensor pointers
and replays it. Replay re-reads `hidden_states` memory (so primary-input
mutation is reflected), **but the captured launches still encode the
ORIGINAL pointers for `routing_logits`, `routing_bias`,
`gemm1_weights*`, `gemm2_weights*`** — only `hs_ptr` is re-validated,
not the other 7 input tensors.

Failure mode under partial reuse:
1. Workload-0: `hidden_states@A`, `routing_logits@B`, `gemm1_weights@C`,
   ... → capture graph; cache `{(T, 0): {graph, hs_ptr=A, ...}}`.
2. Workload-1: `hidden_states@A` (pool reuse), `routing_logits@D`
   (fresh), `gemm1_weights@E` (fresh), ... → key `(T, 0)` matches;
   `hs_ptr=A` matches → **no recapture**. Replay reads `hidden_states`
   from A (workload-1's content, correct) but `routing_logits` from B
   (workload-0's stale ptr — either freed, in which case segfault, or
   reallocated to different content, in which case INCORRECT_NUMERICAL).

**Why:** `data_ptr()` is a memory-address identity. The graph capture
embeds device pointers from ALL eight input tensors into the
`cudaGraphNode_t` kernel-launch descriptors. CUDA graph replay does not
re-read Python-side bindings — it re-launches against the embedded
pointers. The FIB harness amortises the ~10ms capture cost over 100
timed iters per trial; that amortisation is what makes the +0.11x
(round-3) and +0.85% (round-4) lifts real. Under FIB the harness happens
to allocate all input tensors fresh per workload, so the data_ptr
re-validation on the primary input is sufficient — but that sufficiency
is a property of the harness's allocation pattern, not of the kernel.

**Seen in:** iter3c (introduced graph capture + data_ptr trial-boundary
detection — was the +0.11x lift); iter4 anchor (inherited; layered
tactic autotune on top — also embeds tactic in captured launches).
**Not directly observed failing**: every cheat-check / SOL-ExecBench
exposure of this family to-date has been against the FIB harness or
not yet attempted. The hazard is forensic-identified, not measured.

**How to apply:**
1. For the **FIB / contest submission**, the cache is **explicitly
   approved** by yongwww (2026-04-19) — keep it. The +0.11x (graph)
   + +0.85% (autotune) lifts are real and contest-legitimate within
   the FIB contract.
2. For **any other consumer** (SOL submission, prod serving,
   persistent-buffer cheat-check), ship a `submission_no_cache.py`
   derivative next to the variant that strips `_GRAPH_BY_KEY` and
   always uses `_eager_call`. Reference:
   `variants/iter4-autotune-tactic-sweep-gated/submission_no_cache.py`.
   The +0.11x graph win is sacrificed; the +0.85% tactic autotune lift
   is preserved (tactic cache `_TACTIC_BY_KEY` is keyed on shape, not
   content — content-independent at the cubin level). This is a
   **portability variant**, NOT a correctness fix.
3. When introducing **new** module-level graph caches in this family,
   ask: "if the data_ptr of any input is reused with different content
   OR a sibling input's data_ptr changes, would this cache return a
   stale graph?" If yes, validate ALL input pointers (not just one),
   OR use a content fingerprint (expensive), OR don't cache the graph.
4. **`cheat_check_modal.py` does NOT detect this trap.** It mutates
   inputs in-place keeping pointers stable (lines 71-99). Under that
   pattern, graph replay re-reads the mutated `hidden_states` memory
   and produces a correct fresh output → PASS. The trap requires
   *partial reuse across workloads* — a different scenario than
   cheat_check's same-workload-mutate pattern. A persistent-buffer
   probe (one shared `hidden_states` buffer reused across workloads,
   per-workload fresh `routing_logits` / `gemm*_weights`) would catch
   it; not yet implemented.

## `use_shuffled_weight=True` + `WeightLayout::MajorK` + BlockScale is unsupported

**Fact:** Round-5 iter5b tried `_USE_SHUFFLED_WEIGHT=True` on top of the
iter4 anchor (which uses `WeightLayout::MajorK` + `Fp8QuantizationType::DeepSeekFp8`
block-scale). Two variant paths both failed:

- **v1 (weights-only shuffle):** `INCORRECT_NUMERICAL` with
  `abs_err=1.02e+06`. Mechanism: the cubin's `use_shuffled_weight=True`
  fast path reads weights at shuffled row positions, but per-block
  scales are read at the **original** (unshuffled) row positions. Weight
  / scale row mismatch → catastrophic numerical drift.
- **v2 (weights + block-level scale shuffle):** kernel TIMEOUT.
  Mechanism: block-level scale shuffle is mathematically wrong —
  post-shuffle, row blocks mix samples from the input top-half and
  bottom-half; a single per-block scale cannot simultaneously
  represent both regions' value ranges.

**Why:** trtllm wires `use_shuffled_weight=True` ONLY in combination
with `MatrixLayout::BlockMajorK` (verified at
`csrc/trtllm_fused_moe_kernel_launcher.cu:1444`). The
`MajorK + BlockScale + ShuffledWeight` combination is **not** an
intended code path through the public binding; the cubin's shuffled
weight loader and the block-scale loader compute their row indices
independently, with no cross-check that they're using the same row
ordering.

**Seen in:** r5 iter5b v1 (INCORRECT_NUMERICAL), iter5b v2 (TIMEOUT).
Refuted with two failure modes by independent mechanisms.

**How to apply:**
1. Do **not** set `_USE_SHUFFLED_WEIGHT=True` while keeping
   `_WEIGHT_LAYOUT=WeightLayout.MajorK.value` and the DeepSeek FP8
   block-scale quant type. The kernel will either silently produce
   garbage (v1 path) or hang (v2 path).
2. The honest path to `use_shuffled_weight=True` requires:
   (a) switching `_WEIGHT_LAYOUT` to `WeightLayout.BlockMajorK`;
   (b) re-quantizing weights from the original `(num_experts, N, K)`
       layout into the `BlockMajorK` block-tiled form;
   (c) verifying the block-scale layout matches the shuffled rows
       (likely needs kernel-side reverse-engineering — not just
       Python-side `_WEIGHT_LAYOUT` flag).
   Multi-iter weight transform work; out of scope for in-round levers
   on the iter4 anchor; natural for a CUTLASS/CuTe-DSL fresh-kernel
   campaign.

## `enable_pdl=False` is correctness-load-bearing at borderline T=1

**Fact:** Round-5 iter5a disabled PDL (`enable_pdl=False`) hoping for a
perf delta. T=1 dropped below the matched_ratio 0.9 gate →
`INCORRECT_NUMERICAL`. Upper bound on lift is **zero at T=1** (the
correctness gate triggers); ≤ vendor-default elsewhere.

**Why:** Without PDL (Programmatic Dependent Launch), the cubin's
serialized schedule shifts FP8 rounding boundaries on the matmul
epilogue. At T=1, the FIB `matched_ratio` is already close to the 0.9
tolerance gate even with PDL=True (see iter3c's T=1 eager-fallback
rationale — capture-mode FP8 perturbation already grazes the tolerance).
The PDL-off perturbation pushes past tolerance and the workload fails
the correctness check. Larger T amortises the per-token FP8 quantisation
error and stays within tolerance, but no perf lift was observed there
either — PDL is correctness-driving, not perf-saving.

**Seen in:** r5 iter5a (refuted on first labeled bench at T=1).

**How to apply:** Treat PDL as a **correctness invariant**, not a perf
knob. Do not toggle `enable_pdl=False` searching for lift. If the
underlying B200 PDL availability ever flips (e.g., driver upgrade
removing PDL semantics), re-verify the entire T-spread against the
matched_ratio gate before treating the kernel as production-ready.

## trtllm tactic schema is `[tile_N, config]`, not `[gemm1_tactic, gemm2_tactic]`

**Fact:** The `tactic` arg to `moe_op.trtllm_fp8_block_scale_moe(...)`
is a 2-element list `[tile_N, config]`. The story that it is
`[gemm1_tactic, gemm2_tactic]` (which the r3 anchor's open-direction
note implied, and which the r4 sub initially treated as the autotune
search space) applies to a **different code path**: `cutlass_fused_moe`,
not `trtllm_fp8_block_scale_moe`.

**Why:** Source verification (r4 sub):
- `csrc/trtllm_fused_moe_kernel_launcher.cu:143-148` — Python-side
  convention `tactic = [tile_N, config]`.
- `csrc/trtllm_fused_moe_kernel_launcher.cu:2249-2313` — host-side
  enumeration `trtllm_get_valid_moe_configs(...)` returns
  `Array<Array<int64_t>>` of `[tile_N, config]` pairs per
  (dtype_act, dtype_weights, fp8_quant_type, top_k, hidden_size,
  intermediate_size, num_local_experts, act_type, use_shuffled_weight,
  weight_layout, num_tokens).

The vendor heuristic `selectDefaultTileN` is
`nextPowerOfTwo(num_tokens * top_k / num_local_experts)`, clamped to
supported tile_N values. For our `num_local_experts=32, top_k=8`:
T=14107 → tile_N=4096, T=901 → 256, T=80 → 32. The default heuristic
picks the **smallest** matching candidate and does not explore nearby
tiles ±1; that gap is exactly where iter4's per-shape sweep lands its
lift at T=11948 (+0.047x) and T=14107 (+0.068x).

**Seen in:** r3 anchor open-direction note (misframed as
`[gemm1_tactic, gemm2_tactic]`); r4 sub forensic correction via source
read; r4 anchor `iter4-autotune-tactic-sweep-gated` implements the
correct schema.

**How to apply:**
1. When extending the autotune sweep, enumerate `[tile_N, config]`
   pairs only. The valid set is per-shape; query via
   `moe_op.trtllm_get_valid_moe_configs(...)` (pure host-side, no GPU
   work; safe to call any time — typical ~50 candidates per shape).
2. The `cutlass_fused_moe` code path is in flashinfer but is a
   **different kernel** with a **different tactic shape** —
   `[gemm1_cfg, gemm2_cfg]` there. Cross-reading source for that path
   would mislead future levers on the trtllm path.
3. If the cubin gets updated and the schema changes, the autotune sweep
   will start throwing on tactic validation — that is the canonical
   "schema changed under us" signal. Re-read the launcher source before
   patching.
