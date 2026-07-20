"""Agent-neutral command construction for AKO4X child sessions.

The orchestration layer owns process lifetime and transcript persistence.  This
module only validates agent configuration, constructs commands, and extracts a
resumable session identifier from machine-readable output.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True)
class AgentSpec:
    name: str
    runner: str
    task_filename: str
    skills_dir: str
    sandbox: str = "workspace-write"
    approval_policy: str = "never"

    @classmethod
    def from_mapping(cls, name: str, data: dict[str, Any]) -> "AgentSpec":
        runner = str(data.get("runner", name)).lower()
        if runner not in {"claude", "codex"}:
            raise ValueError(f"unsupported agent runner {runner!r}")
        task_filename = str(data.get("task_filename", "")).strip()
        skills_dir = str(data.get("skills_dir", "")).strip()
        if not task_filename or Path(task_filename).name != task_filename:
            raise ValueError("task_filename must be one plain filename")
        if not skills_dir or Path(skills_dir).is_absolute() or ".." in Path(skills_dir).parts:
            raise ValueError("skills_dir must be a relative child path")
        sandbox = str(data.get("sandbox", "workspace-write"))
        if sandbox not in {"read-only", "workspace-write", "danger-full-access"}:
            raise ValueError(f"unsupported Codex sandbox {sandbox!r}")
        approval_policy = str(data.get("approval_policy", "never"))
        if approval_policy not in {"untrusted", "on-request", "never"}:
            raise ValueError(f"unsupported Codex approval policy {approval_policy!r}")
        return cls(name=name, runner=runner, task_filename=task_filename,
                   skills_dir=skills_dir, sandbox=sandbox,
                   approval_policy=approval_policy)


def load_agent_spec(config_path: Path, *, name: str | None = None) -> AgentSpec:
    data = json.loads(config_path.read_text())
    return AgentSpec.from_mapping(name or config_path.stem, data)


def start_command(spec: AgentSpec, prompt: str, *, claude_session_id: str | None = None) -> list[str]:
    if spec.runner == "claude":
        if not claude_session_id:
            raise ValueError("Claude start requires a caller-generated session id")
        return [
            "claude", "--print", "--verbose", "--output-format", "stream-json",
            "--session-id", claude_session_id, prompt,
        ]
    return [
        "codex", "exec", "--json", "--color", "never",
        "--sandbox", spec.sandbox, "--ask-for-approval", spec.approval_policy, prompt,
    ]


def resume_command(spec: AgentSpec, session_id: str, prompt: str) -> list[str]:
    if not session_id:
        raise ValueError("resume requires a session id")
    if spec.runner == "claude":
        return [
            "claude", "--resume", session_id, "--print", "--verbose",
            "--output-format", "stream-json", prompt,
        ]
    # Codex resumes with the sandbox and project context persisted on the
    # thread.  Its resume subcommand intentionally has no --sandbox flag.
    return ["codex", "exec", "resume", "--json", "--color", "never", session_id, prompt]


def session_id_from_transcript(spec: AgentSpec, transcript: str, *, fallback: str | None = None) -> str:
    if spec.runner == "claude":
        if not fallback:
            raise ValueError("Claude transcript parsing requires the supplied session id")
        return fallback

    for line in transcript.splitlines():
        try:
            event = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue
        if event.get("type") == "thread.started" and event.get("thread_id"):
            return str(event["thread_id"])
    raise ValueError("Codex transcript did not contain a thread.started event")


def write_child_agent_metadata(child_dir: Path, spec: AgentSpec) -> Path:
    ako_dir = child_dir / ".ako"
    ako_dir.mkdir(parents=True, exist_ok=True)
    path = ako_dir / "agent.json"
    path.write_text(json.dumps(dataclasses.asdict(spec), indent=2, sort_keys=True) + "\n")
    return path


def read_child_agent_metadata(child_dir: Path) -> AgentSpec:
    path = child_dir / ".ako" / "agent.json"
    if not path.is_file():
        # Backward compatibility for children created before agent metadata.
        return AgentSpec(name="claude", runner="claude", task_filename="CLAUDE.md",
                         skills_dir=".claude/skills")
    data = json.loads(path.read_text())
    return AgentSpec.from_mapping(data.get("name", "agent"), data)
