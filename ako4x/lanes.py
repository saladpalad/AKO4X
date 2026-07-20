"""Isolated hands-on and autonomous worktree lanes."""

from __future__ import annotations

import json
import os
import re
import selectors
import shutil
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from .agent_runtime import load_agent_spec, session_id_from_transcript, start_command
from .production import ProductionPipeline
from .production_config import load_production_config
from .skill_sources import (
    load_manifest, materialize_skills, parse_overrides, resolve_skills, tree_hash,
)


SOURCE_ROOT = Path(__file__).resolve().parent.parent
PACKAGED_ASSETS = Path(__file__).resolve().parent / "_assets"
TEMPLATE_ROOT = (
    SOURCE_ROOT / "templates"
    if (SOURCE_ROOT / "templates").is_dir()
    else PACKAGED_ASSETS / "templates"
)
_LANE_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")


def _git(root: Path, argv: list[str]) -> str:
    result = subprocess.run(["git", *argv], cwd=root, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def repository_root(path: Path) -> Path:
    return Path(_git(path.resolve(), ["rev-parse", "--show-toplevel"])).resolve()


def _merge_builtin_skills(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for skill in source.iterdir():
        if not skill.is_dir():
            continue
        target = destination / skill.name
        if target.exists():
            if tree_hash(target) != tree_hash(skill):
                raise FileExistsError(
                    f"repository skill conflicts with AKO4X built-in {skill.name}: {target}"
                )
            continue
        shutil.copytree(skill, target)


def _all_commands(config: Any) -> list[Any]:
    commands = [config.baseline, config.benchmark]
    if config.optimizer:
        commands.append(config.optimizer)
    for profiler in config.profilers:
        commands.extend([
            profiler.smoke, profiler.baseline, profiler.candidate,
            profiler.parse_smoke, profiler.parse_baseline, profiler.parse_candidate,
        ])
    commands.extend(gate.command for gate in config.gates)
    return commands


def create_lane(project: Path, name: str, *, agent: str, mode: str,
                config_path: Path, skill_sources: list[str] | None = None,
                worktree_path: Path | None = None) -> dict[str, Any]:
    if not _LANE_RE.fullmatch(name):
        raise ValueError("lane name must match [a-z0-9][a-z0-9-]{0,47}")
    if mode not in {"hands-on", "autonomous"}:
        raise ValueError("lane mode must be hands-on or autonomous")
    # Validate the adapter before creating git state. Empty placeholder
    # commands therefore fail before a branch/worktree exists.
    source_config = load_production_config(config_path.resolve())
    if not source_config.candidate.exists():
        raise FileNotFoundError(f"configured candidate does not exist: {source_config.candidate}")
    root = repository_root(project)
    dirty = _git(root, ["status", "--porcelain"])
    if dirty:
        raise RuntimeError(
            "lane creation requires a clean, committed source tree so its base is "
            f"reproducible:\n{dirty}"
        )
    agent_path = TEMPLATE_ROOT / "agent" / f"{agent}.json"
    spec = load_agent_spec(agent_path, name=agent)
    destination = (worktree_path or (root.parent / f"{root.name}-ako4x-{name}")).resolve()
    if destination.exists():
        raise FileExistsError(f"lane worktree already exists: {destination}")
    branch = f"ako4x/{name}"
    _git(root, ["worktree", "add", "-b", branch, str(destination), "HEAD"])
    try:
        skills_destination = destination / spec.skills_dir
        _merge_builtin_skills(TEMPLATE_ROOT / "skills", skills_destination)
        manifest = TEMPLATE_ROOT / "production" / "skills.toml"
        overrides = parse_overrides(skill_sources or [])
        resolved, _ = resolve_skills(load_manifest(manifest), overrides=overrides, strict=True)
        materialize_skills(resolved, skills_destination,
                           lock_path=destination / ".ako4x" / "skills.lock.json")
        ako_dir = destination / ".ako4x"
        ako_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path.resolve(), ako_dir / "production.toml")
        lane_config = load_production_config(ako_dir / "production.toml")
        if not lane_config.project_root.is_relative_to(destination):
            raise ValueError(
                "worktree lanes require a relative project.root that relocates inside "
                f"the worktree, got {lane_config.project_root}"
            )
        if not lane_config.candidate.is_relative_to(destination):
            raise ValueError(
                f"candidate escapes the isolated worktree: {lane_config.candidate}"
            )
        missing_protected = [path for path in lane_config.protected_paths if not path.exists()]
        if missing_protected:
            raise FileNotFoundError(
                "protected integrity paths are missing after worktree relocation: "
                + ", ".join(str(path) for path in missing_protected)
            )
        source_root_text = str(source_config.project_root)
        for command in _all_commands(lane_config):
            if any(source_root_text in token for token in command.argv):
                raise ValueError(
                    "production command embeds the source-worktree path; use paths "
                    f"relative to project.root for lane isolation: {command.argv}"
                )
        if not lane_config.candidate.exists():
            raise FileNotFoundError(
                "candidate path is invalid after worktree relocation; keep the adapter "
                f"under .ako4x/ with project.root='..': {lane_config.candidate}"
            )
        shutil.copy2(TEMPLATE_ROOT / "production" / "AGENT_POLICY.md",
                     ako_dir / "AGENT_POLICY.md")
        if not (destination / "ITERATIONS.md").exists():
            shutil.copy2(TEMPLATE_ROOT / "iterations.md",
                         destination / "ITERATIONS.md")
        metadata = {
            "version": 1,
            "name": name,
            "mode": mode,
            "agent": agent,
            "runner": spec.runner,
            "sandbox": spec.sandbox,
            "approval_policy": spec.approval_policy,
            "branch": branch,
            "project": str(root),
            "worktree": str(destination),
            "config": str(ako_dir / "production.toml"),
            "task": str(ako_dir / "AGENT_POLICY.md"),
        }
        (ako_dir / "lane.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        registry = root / ".ako4x" / "lanes"
        registry.mkdir(parents=True, exist_ok=True)
        (registry / f"{name}.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        return metadata
    except Exception:
        _git(root, ["worktree", "remove", "--force", str(destination)])
        try:
            _git(root, ["branch", "-D", branch])
        except RuntimeError:
            pass
        raise


def load_lane(project: Path, name: str) -> dict[str, Any]:
    root = repository_root(project)
    path = root / ".ako4x" / "lanes" / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"lane metadata not found: {path}")
    return json.loads(path.read_text())


def hands_on_command(metadata: dict[str, Any]) -> list[str]:
    runner = metadata["runner"]
    worktree = metadata["worktree"]
    prompt = "Read .ako4x/AGENT_POLICY.md, then work with me hands-on to optimize the configured candidate."
    if runner == "codex":
        return [
            "codex", "-C", worktree,
            "--sandbox", metadata.get("sandbox", "workspace-write"),
            "--ask-for-approval", metadata.get("approval_policy", "never"),
            prompt,
        ]
    return ["claude", "--add-dir", worktree, prompt]


def _record_run(metadata: dict[str, Any], run_id: str) -> None:
    metadata["run_id"] = run_id
    worktree = Path(metadata["worktree"])
    local = worktree / ".ako4x" / "lane.json"
    local.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    registry = Path(metadata["project"]) / ".ako4x" / "lanes" / f"{metadata['name']}.json"
    registry.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def run_hands_on_lane(metadata: dict[str, Any]) -> str:
    """Profile baseline, open an interactive agent, then validate its output."""
    if metadata["mode"] != "hands-on":
        raise ValueError("lane is not hands-on")
    worktree = Path(metadata["worktree"])
    config = load_production_config(Path(metadata["config"]))
    pipeline = ProductionPipeline(config)
    run_id = pipeline.new_run(lane=metadata["name"], agent=metadata["agent"])
    _record_run(metadata, run_id)
    try:
        pipeline.prepare(run_id)
        pipeline.store.emit(run_id, "agent.started", phase="OPTIMIZING",
                            payload={"interactive": True})
        result = subprocess.run(hands_on_command(metadata), cwd=worktree, check=False)
        if result.returncode:
            raise RuntimeError(f"interactive agent exited {result.returncode}")
        pipeline.store.emit(run_id, "agent.completed", phase="OPTIMIZING",
                            payload={"interactive": True})
        pipeline.validate_candidate(run_id)
        return run_id
    except Exception as exc:
        state = pipeline.store.run(run_id)["state"]
        if state not in {"FAILED", "PROMOTED"}:
            pipeline.store.transition(run_id, "FAILED", payload={"error": str(exc)})
        raise


def run_autonomous_lane(metadata: dict[str, Any], *, timeout: int = 18000) -> str:
    if metadata["mode"] != "autonomous":
        raise ValueError("only autonomous lanes can be launched non-interactively")
    worktree = Path(metadata["worktree"])
    config = load_production_config(Path(metadata["config"]))
    pipeline = ProductionPipeline(config)
    run_id = pipeline.new_run(lane=metadata["name"], agent=metadata["agent"])
    _record_run(metadata, run_id)
    try:
        pipeline.prepare(run_id)
        _run_agent(metadata, pipeline, run_id, timeout=timeout)
        pipeline.validate_candidate(run_id)
        return run_id
    except Exception as exc:
        state = pipeline.store.run(run_id)["state"]
        if state not in {"FAILED", "PROMOTED"}:
            pipeline.store.transition(run_id, "FAILED", payload={"error": str(exc)})
        raise


def _run_agent(metadata: dict[str, Any], pipeline: ProductionPipeline, run_id: str,
               *, timeout: int) -> None:
    worktree = Path(metadata["worktree"])
    spec = load_agent_spec(TEMPLATE_ROOT / "agent" / f"{metadata['agent']}.json",
                           name=metadata["agent"])
    requested_id = str(uuid.uuid4()) if spec.runner == "claude" else None
    prompt = (
        "Read .ako4x/AGENT_POLICY.md and all applicable repository guidance. "
        "Optimize the candidate configured in .ako4x/production.toml. Explicitly use "
        "$production-kernel and $cuda-kernel-style; use $KernelWiki on B200/H100 and "
        "$ncu-report-skill for profile interpretation. Continue until the measured "
        "search is exhausted, then leave the exact best candidate in place."
    )
    command = start_command(spec, prompt, claude_session_id=requested_id)
    run_dir = worktree / ".ako4x" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = run_dir / "agent-transcript.jsonl"
    stderr_path = run_dir / "agent-stderr.log"
    proc = subprocess.Popen(command, cwd=worktree, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, start_new_session=True, bufsize=1)
    assert proc.stdout and proc.stderr
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
    selector.register(proc.stderr, selectors.EVENT_READ, "stderr")
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    started = time.monotonic()
    last_heartbeat = started
    timed_out = False
    while selector.get_map():
        now = time.monotonic()
        if now - started > timeout:
            timed_out = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            break
        for key, _ in selector.select(timeout=1.0):
            line = key.fileobj.readline()
            if not line:
                selector.unregister(key.fileobj)
                continue
            if key.data == "stdout":
                stdout_parts.append(line)
                _emit_agent_event(pipeline, run_id, line)
            else:
                stderr_parts.append(line)
        if now - last_heartbeat >= 30:
            pipeline.store.heartbeat(run_id, phase="OPTIMIZING", detail="agent-running")
            last_heartbeat = now
    proc.wait()
    transcript = "".join(stdout_parts)
    transcript_path.write_text(transcript)
    stderr_path.write_text("".join(stderr_parts))
    if timed_out:
        raise RuntimeError(f"agent timed out after {timeout}s")
    if proc.returncode:
        raise RuntimeError(f"agent exited {proc.returncode}; see {stderr_path}")
    session_id = session_id_from_transcript(spec, transcript, fallback=requested_id)
    pipeline.store.emit(run_id, "agent.completed", phase="OPTIMIZING",
                        payload={"session_id": session_id, "transcript": str(transcript_path)})


def _emit_agent_event(pipeline: ProductionPipeline, run_id: str, line: str) -> None:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return
    kind = str(event.get("type", "agent.output"))
    summary: dict[str, Any] = {"type": kind}
    item = event.get("item")
    if isinstance(item, dict):
        summary.update({key: item[key] for key in ("id", "type", "status") if key in item})
    pipeline.store.emit(run_id, "agent.event", phase="OPTIMIZING", payload=summary)
