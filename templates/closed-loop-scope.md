# Closed-loop scope rules

Some things in your harness must stay fixed so the master can compare results across runs; everything else is fair game for harness improvement. **Default is MUTABLE — only the items below are FROZEN.**

## FROZEN — bench-comparability anchors

**Task identity** in your `CLAUDE.md`: `## Objective`, `## Operator` (incl. workload list / shapes), `## Target GPU`. Editing these would invalidate cross-run comparability — out of scope for any single retrospective.

**Benchmark scoring & baseline behavior** — the active benchmark's scoring formula, baseline freshness rule, and tolerance behavior. The benchmark-specific list of frozen functions / files / keys lives in the active benchmark SKILL's `## Frozen for bench comparability` section. Same reason: editing them would shift the speedup denominator and break comparability.

## MUTABLE — everything else

Anything else visible in your child env is fair game — provided it doesn't touch a FROZEN segment above. Examples: `ITERATIONS.md`, SKILL docs under `.claude/skills/`, slash commands under `.claude/commands/`, scripts under `scripts/` (minus the bench-bound functions named in the active benchmark SKILL), and your `CLAUDE.md`'s intro paragraph + `## Workflow` section.

## Path forms in `scope:`

Use whatever path you see in your own child env (`.claude/skills/<name>/...`, `scripts/<f>.py`, `ITERATIONS.md`, `CLAUDE.md`, etc.). The master translates child-side paths to source-of-truth when applying.
