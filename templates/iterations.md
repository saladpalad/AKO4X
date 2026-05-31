# Iteration Log

Append a Summary row for every labeled bench. That's the only required action.

When a change is worth pre-committing a hypothesis to (architectural shift, sweep you want to track, dead-end probe), write `Expected: ...` in `## Notes` BEFORE running the bench. The point is to catch retrofitted explanations afterward — no required format. Good `Expected` lines name a mechanism, an affected dimension, and a predicted delta (e.g. "Fusing X saves 1 launch × 19 workloads — expect +2-3% on small seq only").

At session end, write a brief synthesis in `## Notes`: kernel state, remaining bottlenecks, dead ends to skip ("needs tuning" ≠ dead end), what's worth trying next session.

## Summary

| Iter | Title | Score | Passed | Notes |
|------|-------|-------|--------|-------|

## Notes

Free-form workspace for pre-commit `Expected:` statements, dead-end records, end-of-session synthesis, anything else useful to this iter or the next session.
