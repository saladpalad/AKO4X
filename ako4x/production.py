"""Executable production-evidence pipeline for optimized kernels."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

from .backends import Backend, ExecResult, load_backend
from .events import RunStore
from .production_config import CommandSpec, GateSpec, ProductionConfig, ProfileSpec


_HASH_EXCLUDES = {".git", ".ako4x", "__pycache__", ".pytest_cache", ".mypy_cache"}


def source_hash(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"candidate does not exist: {path}")
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(
        (item for item in path.rglob("*") if item.is_file()
         and not any(part in _HASH_EXCLUDES for part in item.relative_to(path).parts)),
        key=lambda item: item.as_posix(),
    )
    for item in files:
        rel = item.name if path.is_file() else item.relative_to(path).as_posix()
        rel_bytes = rel.encode()
        data = item.read_bytes()
        digest.update(len(rel_bytes).to_bytes(8, "big"))
        digest.update(rel_bytes)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integrity_manifest(config: ProductionConfig) -> dict[str, Any]:
    paths = (config.path, *config.protected_paths)
    records = []
    digest = hashlib.sha256()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"protected integrity path does not exist: {path}")
        label = (
            path.relative_to(config.project_root).as_posix()
            if path.is_relative_to(config.project_root)
            else str(path)
        )
        sha256 = source_hash(path)
        records.append({"path": str(path), "label": label, "sha256": sha256})
        encoded = label.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(sha256))
    return {"sha256": digest.hexdigest(), "paths": records}


@dataclasses.dataclass(frozen=True)
class Evidence:
    name: str
    ok: bool
    command: tuple[str, ...]
    returncode: int
    timed_out: bool
    stdout_path: str
    stderr_path: str
    artifacts: tuple[dict[str, Any], ...]
    duration_seconds: float


class PipelineFailure(RuntimeError):
    pass


class ProductionPipeline:
    def __init__(self, config: ProductionConfig, *, store: RunStore | None = None,
                 backend: Backend | None = None) -> None:
        self.config = config
        self.store = store or RunStore(config.project_root / ".ako4x" / "runs.sqlite")
        self.backend = backend or load_backend(config.backend_plugin, config.backend_options)

    def new_run(self, *, lane: str = "candidate", agent: str = "external") -> str:
        integrity = integrity_manifest(self.config)
        run_id = self.store.start_run(self.config.project_root, lane=lane, agent=agent)
        self.store.bind_integrity(run_id, integrity["sha256"])
        self._write_json(run_id, "integrity.json", integrity)
        self._write_json(run_id, "environment.json", self.backend.fingerprint(cwd=self.config.project_root))
        self._write_json(run_id, "config.json", {
            "config": str(self.config.path),
            "candidate": str(self.config.candidate),
            "backend": self.config.backend_plugin,
        })
        return run_id

    def _run_dir(self, run_id: str) -> Path:
        path = self.config.project_root / ".ako4x" / "runs" / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_json(self, run_id: str, name: str, data: Any) -> Path:
        path = self._run_dir(run_id) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        return path

    def _execute(self, run_id: str, name: str, command: CommandSpec,
                 *, artifact_paths: Iterable[Path] = ()) -> Evidence:
        artifact_paths = tuple(artifact_paths)
        for path in artifact_paths:
            if path.is_dir():
                raise PipelineFailure(f"evidence path must be a file, got directory: {path}")
            if path.exists():
                path.unlink()
        self.store.emit(run_id, "command.started", phase=self.store.run(run_id)["state"],
                        payload={"name": name, "argv": list(command.argv)})
        started = time.monotonic()
        result = self.backend.run(list(command.argv), cwd=self.config.project_root,
                                  timeout=command.timeout, env=command.env)
        duration = time.monotonic() - started
        command_dir = self._run_dir(run_id) / "commands"
        command_dir.mkdir(parents=True, exist_ok=True)
        safe_name = name.replace("/", "_").replace(" ", "_")
        stdout_path = command_dir / f"{safe_name}.stdout.txt"
        stderr_path = command_dir / f"{safe_name}.stderr.txt"
        stdout_path.write_text(result.stdout)
        stderr_path.write_text(result.stderr)
        artifacts: list[dict[str, Any]] = []
        missing_artifact = False
        artifact_dir = self._run_dir(run_id) / "artifacts" / safe_name
        for index, path in enumerate(artifact_paths):
            exists = path.is_file() and path.stat().st_size > 0
            missing_artifact |= not exists
            snapshot = None
            artifact_hash = file_hash(path) if exists else None
            snapshot_hash = None
            if exists:
                artifact_dir.mkdir(parents=True, exist_ok=True)
                snapshot_path = artifact_dir / f"{index:02d}-{path.name}"
                shutil.copy2(path, snapshot_path)
                snapshot = str(snapshot_path)
                snapshot_hash = file_hash(snapshot_path)
                missing_artifact |= snapshot_hash != artifact_hash
            artifacts.append({
                "path": str(path),
                "snapshot": snapshot,
                "exists": exists,
                "bytes": path.stat().st_size if exists else 0,
                "sha256": artifact_hash,
                "snapshot_sha256": snapshot_hash,
            })
        evidence = Evidence(
            name=name, ok=result.ok and not missing_artifact,
            command=result.argv, returncode=result.returncode, timed_out=result.timed_out,
            stdout_path=str(stdout_path), stderr_path=str(stderr_path),
            artifacts=tuple(artifacts), duration_seconds=duration,
        )
        self._write_json(run_id, f"evidence/{safe_name}.json", dataclasses.asdict(evidence))
        self.store.emit(run_id, "command.completed", phase=self.store.run(run_id)["state"],
                        payload={"name": name, "ok": evidence.ok,
                                 "duration_seconds": duration})
        if not evidence.ok:
            detail = f"{name} failed (exit={result.returncode}, timeout={result.timed_out})"
            if missing_artifact:
                detail += "; required artifact missing or empty"
            raise PipelineFailure(detail)
        return evidence

    def _run_profile(self, run_id: str, profiler: ProfileSpec, stage: str) -> None:
        command: CommandSpec = getattr(profiler, stage)
        report: Path = getattr(profiler, f"{stage}_report")
        parser: CommandSpec = getattr(profiler, f"parse_{stage}")
        if report.exists():
            report.unlink()
        self._execute(run_id, f"{profiler.name}-{stage}", command, artifact_paths=[report])
        executable = self.backend.which(profiler.executable)
        if not executable:
            raise PipelineFailure(f"required profiler {profiler.executable!r} disappeared")
        native_argv = (
            [executable, "--import", str(report), "--page", "details"]
            if profiler.name == "ncu"
            else [executable, "stats", "--report", "cuda_gpu_kern_sum", str(report)]
        )
        self._execute(
            run_id, f"{profiler.name}-{stage}-native-parse",
            CommandSpec(argv=tuple(native_argv), timeout=parser.timeout, env=parser.env),
        )
        self._execute(run_id, f"{profiler.name}-{stage}-parse", parser)

    def preflight(self, run_id: str) -> None:
        if self.store.run(run_id)["state"] != "CREATED":
            raise ValueError("preflight requires CREATED state")
        for profiler in self.config.profilers:
            executable = self.backend.which(profiler.executable)
            if not executable:
                raise PipelineFailure(f"required profiler {profiler.executable!r} not found")
            version = self.backend.run([executable, "--version"], cwd=self.config.project_root,
                                       timeout=30)
            if not version.ok:
                raise PipelineFailure(f"{profiler.name} --version failed")
            self.store.emit(run_id, "profiler.version", phase="CREATED", payload={
                "name": profiler.name,
                "executable": executable,
                "version": (version.stdout or version.stderr).strip(),
            })
            self._run_profile(run_id, profiler, "smoke")
        self.store.transition(run_id, "PREFLIGHTED")

    def run_all(self, *, lane: str = "candidate", agent: str = "external") -> str:
        run_id = self.new_run(lane=lane, agent=agent)
        try:
            self.prepare(run_id)
            if self.config.optimizer:
                self._execute(run_id, "optimizer", self.config.optimizer)
            self.validate_candidate(run_id)
            return run_id
        except Exception as exc:
            current = self.store.run(run_id)["state"]
            if current not in {"FAILED", "PROMOTED"}:
                self.store.transition(run_id, "FAILED", payload={"error": str(exc)})
            raise

    def prepare(self, run_id: str) -> None:
        """Collect mandatory tool, baseline, and baseline-profile evidence."""
        self._verify_integrity(run_id)
        self.preflight(run_id)
        self._execute(run_id, "baseline", self.config.baseline,
                      artifact_paths=self.config.baseline_evidence)
        self.store.transition(run_id, "BASELINED")
        for profiler in self.config.profilers:
            self._run_profile(run_id, profiler, "baseline")
        self._verify_integrity(run_id)
        self.store.transition(run_id, "PROFILED_BASELINE")
        self.store.transition(run_id, "OPTIMIZING")

    def validate_candidate(self, run_id: str) -> None:
        """Validate a candidate after an optimizer or hands-on lane produced it."""
        if self.store.run(run_id)["state"] != "OPTIMIZING":
            raise ValueError("candidate validation requires OPTIMIZING state")
        self._verify_integrity(run_id)
        candidate_hash = source_hash(self.config.candidate)
        self.store.bind_source(run_id, candidate_hash)
        for gate in self.config.gates:
            self._run_gate(run_id, gate)
            self._verify_integrity(run_id)
            if source_hash(self.config.candidate) != candidate_hash:
                raise PipelineFailure(f"gate {gate.name} mutated the candidate source")
        self.store.transition(run_id, "VERIFIED")
        self._execute(run_id, "benchmark", self.config.benchmark,
                      artifact_paths=self.config.benchmark_evidence)
        self._verify_integrity(run_id)
        self.store.transition(run_id, "BENCHMARKED")
        for profiler in self.config.profilers:
            self._run_profile(run_id, profiler, "candidate")
            self._verify_integrity(run_id)
        if source_hash(self.config.candidate) != candidate_hash:
            raise PipelineFailure("candidate source changed after evidence was collected")
        self.store.transition(run_id, "PROFILED_CANDIDATE")
        self.store.transition(run_id, "PROMOTABLE", payload={"source_hash": candidate_hash})
        self._write_json(run_id, "promotion-readiness.json", {
            "run_id": run_id,
            "candidate": str(self.config.candidate),
            "source_hash": candidate_hash,
            "state": "PROMOTABLE",
        })

    def _run_gate(self, run_id: str, gate: GateSpec) -> None:
        self.store.heartbeat(run_id, phase="OPTIMIZING", detail=f"gate:{gate.name}")
        self._execute(run_id, f"gate-{gate.name}", gate.command,
                      artifact_paths=gate.evidence)

    def _verify_integrity(self, run_id: str) -> None:
        expected = self.store.run(run_id)["integrity_hash"]
        current = integrity_manifest(self.config)["sha256"]
        if not expected or current != expected:
            raise PipelineFailure(
                "protected benchmark/test/reference infrastructure changed after run start"
            )

    def promote(self, run_id: str, *, destination: Path | None = None) -> Path:
        run = self.store.run(run_id)
        if run["state"] != "PROMOTABLE":
            raise ValueError(f"run {run_id} is {run['state']}, not PROMOTABLE")
        current_hash = source_hash(self.config.candidate)
        if current_hash != run["source_hash"]:
            raise PipelineFailure("candidate no longer matches the source hash that passed evidence")
        current_integrity = integrity_manifest(self.config)["sha256"]
        if current_integrity != run["integrity_hash"]:
            raise PipelineFailure("protected infrastructure changed after evidence was collected")
        destination = destination or (
            self.config.project_root / ".ako4x" / "promotions" / current_hash[:16]
        )
        if destination.exists():
            raise FileExistsError(f"promotion destination already exists: {destination}")
        destination.mkdir(parents=True)
        candidate_destination = destination / "candidate"
        if self.config.candidate.is_dir():
            shutil.copytree(self.config.candidate, candidate_destination)
        else:
            candidate_destination.mkdir()
            shutil.copy2(self.config.candidate, candidate_destination / self.config.candidate.name)
        manifest = {
            "run_id": run_id,
            "source_hash": current_hash,
            "integrity_hash": current_integrity,
            "candidate": str(self.config.candidate),
            "evidence_dir": str(self._run_dir(run_id)),
            "environment": self.backend.fingerprint(cwd=self.config.project_root),
        }
        (destination / "promotion.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        if source_hash(candidate_destination) != current_hash:
            shutil.rmtree(destination, ignore_errors=True)
            raise PipelineFailure("promoted candidate copy failed source-hash verification")
        self.store.transition(run_id, "PROMOTED", payload={"destination": str(destination)})
        return destination
