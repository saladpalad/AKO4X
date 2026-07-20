# AKO4X Production Kernel Policy

Optimize the configured candidate for measured performance while preserving every production gate.

- Read the repository guidance and `.ako4x/production.toml` before editing.
- Use the `production-kernel` and `cuda-kernel-style` skills for every candidate.
- On B200/H100 work, query `kernelwiki` before selecting a structural design.
- Use `ncu-report-skill` to interpret real NCU reports. Do not infer a bottleneck without profile evidence.
- Do not edit benchmark, profiler, gate, reference, or baseline commands to make a candidate pass.
- Do not edit any `[integrity].protected` source; the supervisor hash-checks it throughout the run.
- Keep a readable human-owned router, named algorithm paths, correctness-first fallback, and concise design notes.
- Record each measured attempt in `ITERATIONS.md`, including failures and dead ends.
- Never use hidden streams, stale outputs, memoized numerical results, pointer-keyed results, or asynchronous work outside the measured dependency chain.
- Leave the exact best candidate at the configured candidate path when finished.
