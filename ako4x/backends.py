"""Execution backends for production evidence collection."""

from __future__ import annotations

import dataclasses
import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Protocol


@dataclasses.dataclass(frozen=True)
class ExecResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


class Backend(Protocol):
    def which(self, executable: str) -> str | None: ...
    def run(self, argv: list[str], *, cwd: Path, timeout: int,
            env: dict[str, str] | None = None) -> ExecResult: ...
    def fingerprint(self, *, cwd: Path) -> dict[str, Any]: ...


class LocalBackend:
    """Run commands directly on the selected machine without shell parsing."""

    def __init__(self, **_: Any) -> None:
        pass

    def which(self, executable: str) -> str | None:
        return shutil.which(executable)

    def run(self, argv: list[str], *, cwd: Path, timeout: int,
            env: dict[str, str] | None = None) -> ExecResult:
        if not argv:
            raise ValueError("command cannot be empty")
        command_env = os.environ.copy()
        if env:
            command_env.update({str(key): str(value) for key, value in env.items()})
        try:
            result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                                    timeout=timeout, env=command_env, check=False)
            return ExecResult(tuple(argv), result.returncode, result.stdout, result.stderr)
        except subprocess.TimeoutExpired as exc:
            return ExecResult(tuple(argv), -1, _decode(exc.stdout), _decode(exc.stderr), True)

    def fingerprint(self, *, cwd: Path) -> dict[str, Any]:
        result: dict[str, Any] = {
            "backend": "local",
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "cwd": str(cwd.resolve()),
        }
        nvidia_smi = self.which("nvidia-smi")
        if nvidia_smi:
            gpu = self.run(
                [nvidia_smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                cwd=cwd, timeout=15,
            )
            if gpu.ok:
                result["gpus"] = [line.strip() for line in gpu.stdout.splitlines() if line.strip()]
        nvcc = self.which("nvcc")
        if nvcc:
            version = self.run([nvcc, "--version"], cwd=cwd, timeout=15)
            if version.ok:
                result["nvcc"] = version.stdout.strip()
        git = self.which("git")
        if git:
            commit = self.run([git, "rev-parse", "HEAD"], cwd=cwd, timeout=15)
            status = self.run([git, "status", "--porcelain=v1"], cwd=cwd, timeout=15)
            if commit.ok:
                result["git_commit"] = commit.stdout.strip()
            if status.ok:
                result["git_dirty"] = bool(status.stdout.strip())
                result["git_status"] = status.stdout.splitlines()
        return result


def _decode(value: bytes | str | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def load_backend(spec: str, options: dict[str, Any] | None = None) -> Backend:
    """Load ``module:factory``. Factories receive backend table options."""
    if spec in {"local", "ako4x.backends:LocalBackend"}:
        return LocalBackend(**(options or {}))
    module_name, separator, attribute = spec.partition(":")
    if not separator:
        raise ValueError("backend plugin must be 'module:factory' or 'local'")
    factory = getattr(importlib.import_module(module_name), attribute)
    backend = factory(**(options or {}))
    for method in ("which", "run", "fingerprint"):
        if not callable(getattr(backend, method, None)):
            raise TypeError(f"backend {spec!r} does not implement {method}()")
    return backend
