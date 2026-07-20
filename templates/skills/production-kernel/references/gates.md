# Production promotion gates

The executable harness requires all gates below. Project adapters choose the commands; agents may add tests but may not weaken or remove existing ones.

| Gate | Minimum evidence |
|---|---|
| correctness | Official shapes, routed paths, edge shapes, trusted reference |
| numerical | Multiple seeds, scales, difficult conditioning, finite outputs, higher-precision invariants where practical |
| api-lifetime | Input immutability, independent outputs, same-object refresh, allocator/pointer reuse, output survival after later calls |
| stream | Non-default current stream and honest event versus device-synchronized wall timing |
| concurrency | Two concurrent callers/streams without shared mutable output state |
| process-state | Backend settings unchanged after import, success, and failure |
| benchmark-integrity | No output replay, input fingerprinting, hidden work, evaluator mutation, or timing-window escape |
| training-integration | Deterministic repeated-use loop comparing trajectory, drift, finite values, and final state |
| clean-deployment | Isolated/offline import with pinned dependencies and node-local caches |
| fallback | Unsupported inputs route to a tested fallback or precise early error |
| reviewability | Named paths, source map/design note, stale experiments removed, generated code attributable |

NCU and NSYS each require three independently parsed reports: smoke, baseline, and exact candidate. A file merely existing is insufficient; the configured parser must successfully read it.

Promotion is content-addressed. Any gate, benchmark, or profile that mutates the candidate invalidates the run.

For a repository-specific mini training loop, use
`ako4x.training.compare_training_trajectories` to compare named loss, output,
gradient, and state observables over repeated updates. Preserve its JSON report as
gate evidence, including failures.
