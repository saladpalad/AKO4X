# AKO4X — Developer Guide

Template repository for spawning GPU kernel optimization environments.
**Not** an optimization environment — use `spawn.py` to create one.
User-facing docs: [README.md](README.md) and [docs/](docs/).

## Architecture

Five layers:

1. **`spawn.py`** — CLI. Creates child environments from templates + dataset + scripts. `--agent codex|claude` selects the native task filename, skill directory, and launcher. `--profile production` also materializes the externally sourced KDA/style skills and strict project adapter.
2. **`templates/`** — canonical sources copied / rendered into each child.
   - **`task.md`** — frozen task identity + Workflow, with `{{PLACEHOLDER}}` substitutions.
   - **`retrospective.md`** — phase-2 closed-loop prompt.
   - **`agent/`** — `<agent>.json` (`codex.json` / `claude.json`, selected via `spawn.py --agent`) + `lessons-convention.md` + Claude-only `hooks/` + `commands/`.
   - **`iterations.md`** — iteration-log template.
   - **`benchmark/evaluation.toml`** — benchmark-bound bench defaults + per-`op_type` tolerance overrides. `templates/benchmark/` is the active benchmark's template dir; its name is the stable slot `spawn.py` reads via the `BENCHMARK_DIR` constant.
   - **`skills/<name>/{SKILL.md, <doc>.md}`** — 9 SKILLs (bench, benchmark, profiler-ncu, sanitizer, triton, cuda, cute-dsl, tilelang, cpp). `bench` carries generic noise-aware methodology; `benchmark` carries the active benchmark's schema (config.toml, status enum, scoring, baseline rule, fresh-inputs contract; default content is flashinfer-bench). The bench/benchmark split's load-bearing role is master FROZEN-scope enforcement — master reads `benchmark` SKILL's "Frozen for bench comparability" section at step 7.

   **Single-active-benchmark assumption.** The repo assumes one active benchmark at a time (bench-runner + task set, both swap together) — no `benchmarks/` plural-container, no runtime selector flag. Multi-benchmark *coexistence* (several behind a per-spawn selector) is a different, larger thing — deferred under YAGNI until a second benchmark is actually needed.

   **Benchmark decoupling.** The benchmark is decoupled behind one seam: `scripts/benchmark_adapter.py` is the sole `flashinfer_bench` importer, and the generic skills point at the stable `benchmark` slot (not a benchmark-specific name), so a swap does NOT touch the runners, the generic DSL skills, or `bench_utils.py`'s scoring math. Switching benchmarks = rewrite `scripts/benchmark_adapter.py` (its plain-data public functions — `run` / `pack` / `solution_meta` / `list_workloads` / `profile` / `list_ncu_options` / `sanitize` / `cheat_check`, with only `str` / `list` / `dict` crossing the seam — plus the Modal-image and dataset-env constants) + the `benchmark` skill's content + `templates/benchmark/evaluation.toml` + the `flashinfer-bench` dependency in `pyproject.toml`. `scripts/bench_utils.py` keeps the frozen `compute_score` / `load_baseline` / `save_baseline` math, which operates on the adapter's normalized result dict and is benchmark-agnostic (no benchmark types cross into it). Full agent-followable procedure: **`docs/porting.md`**.
3. **`scripts/`** — Most files copied into children (canonical list lives in `spawn.py`'s explicit copy allowlist). Sub-visible: `CLAUDE.md` (shared-runtime-core contract for closed-loop), `benchmark_adapter.py` (the sole `flashinfer_bench` importer — the benchmark seam; everything else reaches the benchmark through it), `bench_utils.py` (shared core, frozen-for-comparability segments around `compute_score` / `load_baseline` / `save_baseline`), `run_local.py` / `run_modal.py` (runners), `run_{local,modal}_{profile,sanitize}.py` (NCU + sanitizer wrappers), `pack_solution.py`, `diff_trajectory.py`. **Parent-only** (NOT copied to children): `cheat_check_modal.py` (modal-only correctness audit, invoked as `modal run …/cheat_check_modal.py`) and `backfill_parent_txt.py` (one-shot variant-lineage filler over `reference/`).
4. **Closed-loop scaffolding (`master/`, opt-in via `master/MASTER.md`)** —
   - **`master/master.py`** — thin IO layer: 8 functions (`init_campaign`, `read_campaign_mode`, `spawn_child`, `run_sub_phase1`, `send_retrospective_prompt`, `archive_variant`, `archive_failed`, `append_ledger`), no decision logic. Importable as `import master` from repo root via `master/__init__.py` re-export.
   - **`master/MASTER.md`** — master CC system prompt + 10-step round loop with **two modes**: Mode 2 default = no harness modification, sub does phase-1 kernel optimization only; Mode 3 opt-in = harness co-evolution, sub additionally writes `PROPOSALS.md` in phase-2 and master evidence-gates and applies accepted edits.
   - **`master/harness-ledger.md`** — append-only timeline of harness edits + Mode-2 round-summary lines.

   **Session semantics.** `ako4x/agent_runtime.py` constructs native Claude or Codex start/resume commands and extracts the resumable session/thread ID. A sub session is two-phase in Mode 3 (kernel optimization → harness retrospective in the same session), one-phase in Mode 2. Sub's harness proposals land in `<child>/PROPOSALS.md`; the master reads the file directly.

   **Campaign lock.** Mode, GPU, backend, agent, and profile are locked at Round 0 in `reference/<family>/baseline.json`'s `environment` block; legacy baselines are additively migrated.

   **`scripts/campaign_start.py`** — separate parent-side helper for one-time campaign-branch setup. It installs `MASTER.md` as the selected agent's root guidance (`AGENTS.md` or `CLAUDE.md`) and records agent/profile in the campaign commit. Production adapter/skill validation runs before branch creation.

5. **Production supervisor (`ako4x/`, `templates/production/`)** — generic repository adapter, pluggable execution backend, SQLite/WAL telemetry, isolated Git worktree lanes, and strict evidence state machine. NCU and NSYS must each produce and parse smoke/baseline/candidate reports. Promotion requires all named gates and an unchanged candidate hash. `docs/production.md` is the operator guide; code is the authority.

### Reference archive contract

`reference/<family>/` is the closed loop's persistent memory across rounds — `spawn.py` seeds each new child from it, and the master maintains it (steps 8–9 of each round).

**Family naming.** Current convention: `family == operator name`,
kebab-cased — so `reference/mla-paged-decode-h16-ckv512-kpe64-ps1/`
holds variants for operator `mla_paged_decode_h16_ckv512_kpe64_ps1`.
Each operator is fully isolated (no cross-shape variant sharing within a
kernel class). Legacy directories under the older kernel-class convention
(`dsa-sparse-attention`, `gdn-decode`, etc.) remain as-is — they predate
the per-operator scheme and bundle multiple shapes; the auto-discovery in
`spawn.py` (underscore-prefix match) keeps them working. The deferred
"shared variant pool within a kernel class" extension (cross-operator
variant sharing) is left for a future version.

Each `reference/<family>/` holds working kernel variants, a `README.md`
anchor pointer, `baseline.json`, optionally a `TRAPS.md` for
cross-variant toolchain facts, and `_failed/<round-id>/` for closed-loop
crash/timeout transcripts (created lazily). Lessons live in each variant's
`kernel.py` header comment — not in separate markdown. Each variant
carries a single-line `parent.txt` (parent variant name or `null` for
roots). Follow
[templates/agent/lessons-convention.md](templates/agent/lessons-convention.md)
when writing or updating a variant header: five sections
(Identity / Delta / Lessons / Dead-ends / Open directions), each lesson
carries a two-layer WHEN, dead-ends are expectation priors rather than
prohibitions, edits go in place.

When a closed-loop campaign is running, the master CC maintains
`README.md` (anchor + history) and `TRAPS.md` (silent-bug patterns) at
step 8 of each round — append new variants, rotate the anchor when a
new variant beats the current one, and append new TRAPS entries when
step-8 sanity check finds a previously-undocumented silent-skip pattern.

## Design rules

The harness is prompt + scripts consumed by **sub CC** (phase-1 kernel work / phase-2 retrospective) and **master CC** (round loop). The rules below shape its current form — derived from surveying real spawned envs and a cleanup pass on `closed-loop-v1` (May 2026). Apply them when editing. Five thematic buckets: how to **decide** what to add/cut; how to design **sub-facing prompts**; how to split work between **master and sub**; what's in **scope** to change; and the **boundaries & contracts** the rest of the system relies on.

### Decision-making

- **Empirics over speculation.** Survey real spawned envs before designing harness structure. Two findings drove this campaign's cleanup: 12 of 17 HINTS.md files in spawned envs were unmodified empty templates → HINTS.md dropped as a customization layer; closed-loop spawns produced zero ITERATIONS.md entries under the prior 4-tier protocol → collapsed to "one Summary row + free-form `## Notes`". Structure that isn't used is noise.

- **Attention budget is finite.** Every line of prompt loaded into sub competes with the kernel work it's meant to support. A 200-line ITERATIONS.md eats 5-10% of a long-session context window; pristine empty-template scaffolding pollutes every spawn; meta-commentary about Claude Code's own mechanisms is redundant. Keep what's load-bearing; cut what isn't.

- **Children are disposable.** This repo is source of truth. Fix here, re-spawn.

### Sub-facing prompt design

- **Required-substance, not required-fields.** Multi-field templates with required slots invite "going through the motions" — agents fill boilerplate to satisfy form rather than reason. State the substance the master needs to see (e.g., "scope + phase-1 evidence visible somewhere in your proposal"), not which named field it must appear in. Fields are suggested shape; the rule is substance. Applied to `templates/retrospective.md` proposal contract and `templates/iterations.md` Summary table.

- **Address sub for sub.** Sub-facing prompts (`task.md`, SKILLs, `closed-loop-scope.md` + `retrospective.md` when injected in phase-2) use audience-appropriate language: "the master" not internal orchestration jargon; no meta-commentary about mechanisms the selected agent already inhabits; no HTML-comment scaffolding; no references to paths sub can't see. The child sees its native task/skill paths (`AGENTS.md` + `.agents/skills/` for Codex, `CLAUDE.md` + `.claude/skills/` for Claude), while source-of-truth paths remain `templates/task.md` and `templates/skills/`. `spawn.py` is canonical for translation.

### Master/sub division of labor

- **Master CC is an agent, not a parser.** Master reads `PROPOSALS.md` directly with its Read tool and reasons holistically — no regex on assistant replies, no field-extraction, no `Proposal` dataclass. When master needs implementation-dependent info (path mappings, child-population logic), it Reads `spawn.py` as canonical source rather than consulting a snapshot table in MASTER.md.

- **Translation belongs with the more capable / contextual actor.** Sub CC can't see the parent repo — don't ask it to know parent paths or mapping rules. Master has the full repo plus judgment; path translation, proposal gating, and edge-case interpretation are master-side. Sub uses child-form paths; master translates.

### Scope policy

- **Default MUTABLE; FROZEN is small and named.** Only protect what anchors round-to-round comparability (task identity + active benchmark's campaign baseline). Allowlists that auto-reject "everything else" drift on every new file type added. Privilege boundaries (e.g., `master/`) are conceptually separate from FROZEN — they survive as reject categories in the audit taxonomy, not as additional FROZEN buckets. See `templates/closed-loop-scope.md`.

- **Master is reactive-only (v1).** Master CC doesn't self-propose harness edits — harness improvements come exclusively through sub's phase-2 retrospective. Rationale: validate the sub→master proposal channel as a sufficient source of harness improvements before adding a parallel master-side one; master's attention each round is already on parent selection / gating / archival, and a self-proposal stream would compete with that without yet earning the seat. The deferred self-proposal direction (gated on accumulated cross-round failure signal that master's per-round view can't see) is left for a future version. MASTER.md step 1 carries only the behavioral constraint ("you never self-propose"); the version label and rationale live here, not there.

- **Closed-loop FROZEN scope**: edits to task identity or active-benchmark scoring / baseline behavior are rejected by the master step-7 gate (ledger reason `out-of-scope: ...`). Full bucket list: [`templates/closed-loop-scope.md`](templates/closed-loop-scope.md) (sub-facing) + [`templates/skills/benchmark/benchmark.md`](templates/skills/benchmark/benchmark.md) § "Frozen for bench comparability" (benchmark-specific items, master-facing). The general principle behind this constraint is *Default MUTABLE; FROZEN is small and named* above.

### Boundaries & contracts

- **Scripts must be self-contained**: `scripts/` is copied into children. No imports from parent.
- **Operator data is external**: `definition.json`, `workloads.jsonl` come from the dataset at spawn time.
- **bench_utils.py is host-side only**: Used by both runners on the parent host (and inside spawned children), but NOT included in the Modal image — only `benchmark_adapter.py` is `add_local_file`'d into the container. The bench loop lives in `adapter.run`; bench_utils' Modal-side callers (host-side `from scripts.bench_utils import ...` in `run_modal*.py`) run outside `@app.function`. The "no heavy deps" rule remains useful for fast host imports, but is not Modal-image-dictated.
- **config.toml merges evaluation overrides**: `populate_child()` reads `templates/benchmark/evaluation.toml` (`[default]` = benchmark defaults, per-op_type sections = overrides) and writes the merged dict into the child's `[benchmark]` section. The dir name comes from the `BENCHMARK_DIR` constant at the top of `spawn.py`.
- **ITERATIONS.md is a minimal-overhead log**: `templates/iterations.md` requires only a Summary row per labeled bench plus a free-form `## Notes` section for pre-commit `Expected:` statements, dead-end records, and end-of-session synthesis. No tier dispatch / per-iter detail template / hook enforcement. Keeping the writing burden low is load-bearing — it preserves attention budget for the kernel work ITERATIONS.md is meant to support, not compete with. (For the empirics behind the 4-tier collapse, see *Empirics over speculation* above.)
- **task.md is template-body-only**: `templates/task.md` is the invariant body rendered into every child env — only its placeholders (`{{OPERATOR}}`, `{{GPU_NAME}}`, `{{PRIOR_LESSONS_BLOCK}}`) vary across spawns. AKO has two customization layers with distinct audiences and timing: (1) `templates/task.md` (cross-operator template, read by sub CC at session start), (2) initial prompt to `claude` (per-session interactive — for closed-loop, master CC's phase-1 prompt). Operator-specific human prior — orienting past a cryptic operator name, custom focus, must-use-this-DSL — belongs in (2), never in (1). For cross-session operator wisdom (dead ends from prior sessions, anchor variant pointers), use the `reference/<family>/` archive — spawn.py renders it into task.md's `## Operator` section via `{{PRIOR_LESSONS_BLOCK}}` automatically. (A prior third layer — a per-spawn `HINTS.md` file — was dropped in 2026-05; see *Empirics over speculation* above for the ablation.) This rule lives here (not in `task.md`) because the audience is humans / Claude editing the harness; sub CC doesn't need to be told.
- **SKILLs are single-source at `templates/skills/<name>/<doc>.md`**: spawn.py copies them to the selected agent's native skills directory. Production-only external skills are copied verbatim from sources declared in `templates/production/skills.toml`, then tree-hashed into `.ako4x/skills.lock.json`. `bench_utils.py` error messages reference skills by name so runtime stays frontend-neutral.
- **Advisory review hook**: `templates/agent/hooks/advisory-review.sh` fires after every N labeled benches (configurable via `config.toml [advisory] frequency`, default 3), printing a static self-review prompt to stderr. No gate, no blocking — purely advisory. Same prompt available as `/review` slash command (`templates/agent/commands/review.md`); both share a single content file. Scope: priming under-used tools / resources back into recent attention.
