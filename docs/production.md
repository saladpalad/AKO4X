# Production kernel campaigns

The production profile adds an executable promotion boundary around AKO4X. It
works in any Git repository and with either Codex or Claude. A candidate cannot
be promoted until the harness has collected and parsed real NCU and NSYS reports,
run every required gate, benchmarked the candidate, and verified that its source
has not changed since that evidence was collected.

This profile complements the existing AKO4X search loop. It does not replace a
repository's reference implementation, test data, training loop, or deployment
build; those become commands in the project adapter.

## Prerequisites

- A Git repository containing the candidate and reference implementation.
- `ncu` and `nsys` available on the profiling backend.
- Repository-specific commands for correctness, stress, integration, and build
  validation.
- KDA's `KernelWiki` and `ncu-report-skill`, plus `cuda-kernel-style`. AKO4X
  copies these skills verbatim and records source paths, file counts, and tree
  hashes in `.ako4x/skills.lock.json`.

Install AKO4X, then inspect skill resolution:

```bash
pip install -e /path/to/AKO4X
ako4x-lab skills
```

If a skill is in a nonstandard location, pass a repeatable override when
creating a lane:

```bash
--skill-source KernelWiki=/path/to/KernelWiki
--skill-source ncu-report-skill=/path/to/ncu-report-skill
--skill-source cuda-kernel-style=/path/to/cuda-kernel-style
```

Missing required skills are a hard error.

## Configure any repository

From the target repository:

```bash
ako4x-lab init .
```

Edit `.ako4x/production.toml`. Commands are argv arrays, not shell strings. The
checked-in template deliberately contains empty arrays so an unconfigured
campaign fails rather than silently running weak checks.

Keep `project.root` and repository-script paths relative (`root = ".."` for the
standard location), then commit the candidate, adapter, and tests before creating
a lane. Lane creation rejects dirty trees, absolute source-worktree paths, and
candidates that escape the isolated worktree.

The adapter must define:

- `[integrity].protected` paths covering reference, benchmark, gate, and test
  sources that the optimizing agent must not change;
- a stable baseline command and candidate benchmark command;
- NCU and NSYS smoke, baseline, and candidate commands;
- the report path produced by each profiler command;
- a parser command for each exact report;
- all required promotion gates.

Use each command's optional `evidence = [...]` list for result JSON, training
traces, failure reproducers, or build manifests. Required evidence is copied into
the run directory and hash-verified; profiler reports are always snapshotted.

The required gates are `correctness`, `numerical`, `api-lifetime`, `stream`,
`concurrency`, `process-state`, `training-integration`, `clean-deployment`,
`benchmark-integrity`, `fallback`, and `reviewability`.

Each gate should test behavior, not merely scan source. In particular:

- numerical: scale, conditioning, dtype, cancellation, NaN/Inf, odd shapes,
  deterministic seeds, forward and backward error where applicable;
- training integration: repeated optimizer steps against the reference while
  checking drift, gradients, loss trajectory, and memory growth;
- stream/concurrency: non-default current streams, overlapping invocations,
  output lifetime, and synchronization ownership;
- process/deployment: fresh processes, cache misses, clean builds, supported
  architectures, allocator pressure, and fallback paths;
- benchmark integrity: fresh randomized inputs, canaries, honest dependency-chain
  timing, and rejection of output replay or evaluator-specific detection;
- reviewability: source-size/complexity policy, documented invariants, bounded
  specialization, and actionable failures.

`ako4x.training.compare_training_trajectories` is the reusable training gate
core. Give it independently seeded, stateful reference and candidate steppers;
each real update returns named numeric observables such as `loss`, `output`,
`grad/<name>`, and `state/<name>`. It checks every step for shape/dtype changes,
NaN/Inf, absolute/relative drift, accumulating bias, and writes the first failing
seed/step/field to JSON. The repository still owns the meaningful mini-model,
optimizer, input distribution, and field-specific tolerances.

Before any profiler or agent starts, AKO4X hashes the production config and all
protected paths into the run record. It rechecks them before validation, after
every gate, after the benchmark, after each candidate profile, at archive, and at
promotion. Candidate and protected paths may not overlap.

Profiler commands must create non-empty reports. AKO4X deletes stale reports
before each run, parses them through the profiler's native import/stats command,
and separately executes the configured project parser. Capture commands must
explicitly invoke the matching tool and reports must use `.ncu-rep` /
`.nsys-rep`. Having `ncu` or `nsys` on `PATH` is not sufficient.

For a remote or scheduled GPU, implement a backend factory with the
`module:factory` interface from `ako4x.backends`. The backend must materialize
reports at the configured local paths before returning so artifact hashes and
parsers remain authoritative.

## Verify the adapter

```bash
ako4x-lab doctor --config .ako4x/production.toml
```

Doctor runs real NCU and NSYS smoke captures and parsers. It does not begin
optimization. A failure is recorded in `.ako4x/runs.sqlite` and its command
stdout/stderr remain under `.ako4x/runs/<run-id>/`.

## Hands-on and autonomous lanes

Each lane is an isolated Git worktree and branch. Both modes use the same hard
sequence: profiler preflight, baseline, baseline profiles, optimization,
production gates, benchmark, candidate profiles, then `PROMOTABLE`.

Create a hands-on Codex lane:

```bash
ako4x-lab lane-create human \
  --project . --config .ako4x/production.toml \
  --agent codex --mode hands-on
ako4x-lab lane-run human --project .
```

`lane-run` collects baseline evidence before opening the interactive agent. When
the session exits, it validates the resulting candidate automatically. Codex
lanes use `workspace-write` with approval policy `never`: routine worktree
commands do not prompt, while attempts outside the granted workspace fail and
return to the agent instead of escalating to full machine access.

Create an autonomous lane:

```bash
ako4x-lab lane-create auto \
  --project . --config .ako4x/production.toml \
  --agent codex --mode autonomous
ako4x-lab lane-run auto --project . --timeout 18000
```

Run the two `lane-run` commands in separate terminals for concurrent human and
autonomous search. They use different worktrees and branches.

Follow a lane from another terminal using its worktree path from `lane-create`:

```bash
ako4x-lab status --project /path/to/lane-worktree --watch
```

Telemetry is SQLite/WAL-backed and includes lifecycle transitions, commands,
agent events, and heartbeats. Autonomous Codex output records the native thread
ID so the transcript is resumable.

## Promotion

A completed lane stops at `PROMOTABLE`. Promotion is explicit:

```bash
ako4x-lab promote \
  --config /path/to/lane-worktree/.ako4x/production.toml \
  --run-id <run-id>
```

The promotion bundle contains the candidate, environment fingerprint, evidence
location, and source hash. Promotion fails if even one candidate byte changed
after validation.

## Closed-loop AKO4X campaigns

`spawn.py --agent codex --profile production` installs native `AGENTS.md`,
Codex skills, the production policy, and the project adapter in every child.
For a production closed-loop campaign, customize
`templates/production/project.toml` for the active benchmark before running:

```bash
python scripts/campaign_start.py \
  --operator <operator> --gpu b200 --backend local \
  --mode 2 --agent codex --profile production
```

Campaign setup fails before changing Git state if the adapter or required skills
are incomplete. The master archive gate accepts only a `PROMOTABLE`/`PROMOTED`
run whose validated source still matches the exact kernel selected for archive.

Use `--profile standard` only when intentionally running the original
speed/correctness workflow without the production promotion guarantees.
