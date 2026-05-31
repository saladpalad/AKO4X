# Lessons Convention

How experience from one session becomes useful to the next.

This convention intentionally has no machine-readable tags, no precondition
graphs, no version fingerprints. Experience is soft — the archive's job is
to give a new Agent enough narrative + concrete source code to orient,
not to mechanize reasoning.

## Archive layout

One archive per operator family at `reference/<family>/`:

- `README.md` — short pointer (~20 lines) identifying the current anchor
  variant, A/B baseline, fallbacks, and canonical-baseline provenance.
- `baseline.json` — frozen per-workload reference latencies; the
  denominator for speedups in every variant's `result.json`.
- `variants/<name>/` — preserved working kernels, one directory per
  variant.
- `TRAPS.md` (optional) — cross-variant toolchain / methodology facts
  that apply regardless of which variant is anchor. Create only when the
  first such fact is found.

Each `variants/<name>/` contains:

- `kernel.py` (or equivalent source) — **the ground-truth evidence**.
  Header comment carries the variant's narrative (see below).
- `config.toml` — ready-to-pack config, including any required flags.
- `result.json` — latest measurement.
- `variance.json` — variance-check data where captured.

## The variant header comment

Each variant's `kernel.py` starts with a structured header comment. This
is where lessons live — **not** in a separate markdown file. Colocating
source, narrative, and WHEN makes each variant a self-contained unit
that future sessions can read, copy, and extend.

The header covers five sections:

1. **Identity** — score + toolchain + date of last measurement;
   any required config flag.
2. **Delta from prior anchor** — one paragraph of narrative:
   what this variant adds at architecture level.
3. **Lessons on this variant** — each lesson = WHAT + WHY + two-layer
   WHEN (see below). N lessons; no cap, but drop anything whose WHY is
   just restating WHAT.
4. **Dead-ends tried on this variant** — expectation priors, not
   prohibitions (see below). Typically 5–10 entries; parameter-sweep
   long-tails fold into one line.
5. **Open directions** — narrative of where a future session continuing
   this line might go. Not a priority list, not a todo.

Session-level iteration logs (`ITERATIONS.md`) stay in the sub-env — do
not copy them into the archive.

## WHEN in two layers

Every performance lesson applies under specific conditions. Record two
layers:

- **Narrow WHEN** — the exact operational condition in this operator
  (e.g. `T≥6 where reduce grid ≥ 384 blocks`).
- **Broad WHEN** — the principle-level condition that transfers
  (e.g. `grid >> SM count, so L2 co-residency matters across blocks`).

Narrow is precise but brittle; broad is portable. A future Agent matches
current state to narrow if it's on the same operator, to broad if it's
on a different one. Both layers defend against applying a lesson where
it doesn't hold.

## Dead-ends are expectation priors, not prohibitions

Past sessions re-verify each other's dead-ends on new toolchains — this
is healthy behavior. A dead-end's job is to:

1. Tell the next Agent what to expect if they retry, and
2. Explain **why**, so they can judge whether their changed context
   flips the expectation.

**Not** to forbid re-trial. A dead-end entry without a **Why** is noise
— drop it.

Dead-ends scope to the variant they were tried on. They do NOT propagate
forward automatically when a new variant is born; the new state may have
different interactions. Cross-variant universal gotchas (toolchain facts,
measurement methodology) go in `TRAPS.md`, not in any variant's header.

## What does NOT go in a variant header

- Parameter-sweep long-tails (`num_warps ∈ {2,4,8,16}`, `BI ∈ {64,128}`,
  etc.) — fold into one line: *"parameter sweeps around {A, B, C}
  regressed on this variant; retry only with new reasoning."*
- External references (papers, doc URLs) — look them up as needed; don't
  keep stale link lists.
- Cross-session measurement history — `result.json` / `variance.json`
  and git log hold this.
- Cross-variant facts — `TRAPS.md`.

## End-of-session workflow

If a session produces a new winning variant:

1. Preserve the kernel under `reference/<family>/variants/<name>/`.
2. Write/update its header comment following the five sections above.
3. Update `reference/<family>/README.md` if the anchor shifted.
4. If the session found a toolchain or measurement fact that applies
   regardless of variant, append one entry to `TRAPS.md` (create it if
   it doesn't exist).

## Example header

```
# cute_reduce_v6 — reference kernel.py header
#
# Identity
#   75.60 ± 0.08x variance-check 5 (Modal B200, CUDA 13.2, 2026-04-20).
#   Config requires [benchmark] use_isolated_runner = true on
#   persistent-runner environments.
#
# Delta from cute_reduce_v5
#   Four orthogonal wins stacked on v5's iter-1 split-loop CuTe DSL reduce:
#   cooperative launch + exp2/log2 domain + PO layout transpose + T=1
#   masked K loads. No algorithmic change to the reduce pattern itself.
#
# Lessons on this variant
#
#   +0.55x cooperative launch on CuTe reduce
#     How:           kernel.launch(cooperative=True)
#     Why:           co-resident blocks share L2 across PO loads
#     WHEN narrow:   T≥6 where reduce grid ≥ 384 blocks; within noise at T=1
#     WHEN broad:    any kernel with grid >> SM count where cross-block L2
#                    reuse dominates per-block memory traffic
#
#   +6.63x T=1-only masked K loads in Triton fwd
#     How:           USE_MASK constexpr, tl.load(mask=valid, other=0.0)
#     Why:           at T=1 most sparse_indices are -1; gather dominates cost
#     WHEN narrow:   Tv == 1 only (dispatch USE_MASK=True)
#     WHEN broad:    sparse-gather kernels where invalid-entry ratio is high
#     Anti-pattern:  DO NOT enable at T=2 — mask overhead > savings (-2.5x)
#
#   [other lessons elided]
#
# Dead-ends tried on this variant
#   Each is an expectation prior. Re-verify cheaply if you suspect your
#   toolchain shifted; do not trust blindly.
#
#   - PO rmem prefetch: 80 regs/thread; occupancy drop > pipelining win.
#   - use_pdl=True on reduce: 1/23 INCORRECT_NUMERICAL — races with fwd writes.
#   - cluster=[1,1,2]: -0.9x; adjacent D_chunks don't share enough data.
#   - D_CHUNK ∈ {64, 256} swept under coop launch: all regress vs 128.
#   - Parameter sweeps around {num_warps, num_stages, BI, threads} all
#     regressed in prior sessions; retry only with new reasoning.
#
# Open directions
#   CuTe DSL fwd with tensor cores remains the structural lever. v8
#   attempt blocked at SMEM layout: plain row-major makes ldsm atom
#   reject for bf16 D_head≥64. Swizzled composed layout is the
#   precondition. See git log of this variant for partial scaffolding.
```

## Accepting softness

This convention has no machine-readable metadata because soft judgment
cannot be mechanized. When a detail doesn't fit the template, write it
as narrative inside the relevant section rather than inventing a schema
field — forced structure drops the nuance that made the detail worth
preserving.

If a past lesson turns out to be wrong, **edit the variant header in
place** rather than appending a contradicting note below. Keep each
variant header current, not layered.
