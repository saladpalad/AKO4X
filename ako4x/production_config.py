"""Strict, command-based production harness configuration."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


REQUIRED_GATES = (
    "correctness",
    "numerical",
    "api-lifetime",
    "stream",
    "concurrency",
    "process-state",
    "benchmark-integrity",
    "training-integration",
    "clean-deployment",
    "fallback",
    "reviewability",
)


@dataclasses.dataclass(frozen=True)
class CommandSpec:
    argv: tuple[str, ...]
    timeout: int = 900
    env: dict[str, str] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], *, label: str, required: bool = True) -> "CommandSpec | None":
        argv = tuple(str(item) for item in raw.get("command", []))
        if not argv:
            if required:
                raise ValueError(f"{label} requires a non-empty command array")
            return None
        timeout = int(raw.get("timeout", 900))
        if timeout <= 0:
            raise ValueError(f"{label} timeout must be positive")
        env = {str(key): str(value) for key, value in raw.get("env", {}).items()}
        return cls(argv=argv, timeout=timeout, env=env)


@dataclasses.dataclass(frozen=True)
class ProfileSpec:
    name: str
    executable: str
    smoke: CommandSpec
    smoke_report: Path
    baseline: CommandSpec
    baseline_report: Path
    candidate: CommandSpec
    candidate_report: Path
    parse_smoke: CommandSpec
    parse_baseline: CommandSpec
    parse_candidate: CommandSpec


@dataclasses.dataclass(frozen=True)
class GateSpec:
    name: str
    command: CommandSpec
    evidence: tuple[Path, ...]


@dataclasses.dataclass(frozen=True)
class ProductionConfig:
    path: Path
    project_root: Path
    candidate: Path
    protected_paths: tuple[Path, ...]
    backend_plugin: str
    backend_options: dict[str, Any]
    baseline: CommandSpec
    baseline_evidence: tuple[Path, ...]
    benchmark: CommandSpec
    benchmark_evidence: tuple[Path, ...]
    optimizer: CommandSpec | None
    profilers: tuple[ProfileSpec, ...]
    gates: tuple[GateSpec, ...]


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_production_config(path: Path) -> ProductionConfig:
    path = path.resolve()
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if int(data.get("version", 0)) != 1:
        raise ValueError("production config requires version = 1")

    project = data.get("project", {})
    root_value = str(project.get("root", "."))
    project_root = _resolve(path.parent, root_value)
    candidate_value = str(project.get("candidate", "")).strip()
    if not candidate_value:
        raise ValueError("[project].candidate is required")
    candidate = _resolve(project_root, candidate_value)

    integrity = data.get("integrity", {})
    protected_values = integrity.get("protected", [])
    if not protected_values:
        raise ValueError("[integrity].protected requires repository test/reference paths")
    protected_paths = tuple(_resolve(project_root, str(value)) for value in protected_values)
    for protected in protected_paths:
        if not protected.is_relative_to(project_root):
            raise ValueError(f"protected integrity path escapes project.root: {protected}")
        if candidate == protected or candidate.is_relative_to(protected) or protected.is_relative_to(candidate):
            raise ValueError(
                f"candidate and protected integrity path overlap: {candidate} / {protected}"
            )

    backend = data.get("backend", {})
    backend_plugin = str(backend.get("plugin", "local"))
    backend_options = dict(backend.get("options", {}))

    commands = data.get("commands", {})
    baseline_raw = commands.get("baseline", {})
    benchmark_raw = commands.get("benchmark", {})
    baseline = CommandSpec.from_mapping(baseline_raw, label="commands.baseline")
    benchmark = CommandSpec.from_mapping(benchmark_raw, label="commands.benchmark")
    baseline_evidence = tuple(
        _resolve(project_root, str(value)) for value in baseline_raw.get("evidence", [])
    )
    benchmark_evidence = tuple(
        _resolve(project_root, str(value)) for value in benchmark_raw.get("evidence", [])
    )
    optimizer = CommandSpec.from_mapping(commands.get("optimizer", {}), label="commands.optimizer",
                                         required=False)
    assert baseline and benchmark

    profiler_specs: list[ProfileSpec] = []
    profiler_table = data.get("profilers", {})
    for name in ("ncu", "nsys"):
        raw = profiler_table.get(name)
        if not isinstance(raw, dict):
            raise ValueError(f"[profilers.{name}] is required")
        executable = str(raw.get("executable", name)).strip()
        if Path(executable).name != name:
            raise ValueError(
                f"profilers.{name}.executable must resolve to a binary named {name!r}"
            )
        stage_specs: dict[str, CommandSpec] = {}
        reports: dict[str, Path] = {}
        parsers: dict[str, CommandSpec] = {}
        for stage in ("smoke", "baseline", "candidate"):
            stage_raw = raw.get(stage, {})
            command = CommandSpec.from_mapping(stage_raw, label=f"profilers.{name}.{stage}")
            parse = CommandSpec.from_mapping(stage_raw.get("parse", {}),
                                             label=f"profilers.{name}.{stage}.parse")
            report_value = str(stage_raw.get("report", "")).strip()
            if not report_value:
                raise ValueError(f"profilers.{name}.{stage}.report is required")
            assert command and parse
            capture_tokens = {Path(token).name for token in command.argv}
            if name not in capture_tokens:
                raise ValueError(
                    f"profilers.{name}.{stage}.command must invoke {name!r} explicitly"
                )
            required_suffix = f".{name}-rep"
            if Path(report_value).suffix != required_suffix:
                raise ValueError(
                    f"profilers.{name}.{stage}.report must end in {required_suffix}"
                )
            stage_specs[stage] = command
            parsers[stage] = parse
            reports[stage] = _resolve(project_root, report_value)
        profiler_specs.append(ProfileSpec(
            name=name, executable=executable,
            smoke=stage_specs["smoke"], smoke_report=reports["smoke"],
            baseline=stage_specs["baseline"], baseline_report=reports["baseline"],
            candidate=stage_specs["candidate"], candidate_report=reports["candidate"],
            parse_smoke=parsers["smoke"], parse_baseline=parsers["baseline"],
            parse_candidate=parsers["candidate"],
        ))

    gates_by_name: dict[str, GateSpec] = {}
    for raw in data.get("gate", []):
        name = str(raw.get("name", "")).strip()
        if not name or name in gates_by_name:
            raise ValueError(f"gate name is missing or duplicated: {name!r}")
        command = CommandSpec.from_mapping(raw, label=f"gate.{name}")
        assert command
        evidence = tuple(_resolve(project_root, str(value)) for value in raw.get("evidence", []))
        gates_by_name[name] = GateSpec(name=name, command=command, evidence=evidence)
    missing = sorted(set(REQUIRED_GATES) - set(gates_by_name))
    if missing:
        raise ValueError("missing required production gates: " + ", ".join(missing))

    return ProductionConfig(
        path=path, project_root=project_root, candidate=candidate,
        protected_paths=protected_paths,
        backend_plugin=backend_plugin, backend_options=backend_options,
        baseline=baseline, baseline_evidence=baseline_evidence,
        benchmark=benchmark, benchmark_evidence=benchmark_evidence, optimizer=optimizer,
        profilers=tuple(profiler_specs),
        gates=tuple(gates_by_name[name] for name in REQUIRED_GATES),
    )
