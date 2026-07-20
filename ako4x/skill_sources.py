"""Resolve and materialize external agent skills with provenance.

AKO4X's built-in skills remain under ``templates/skills``.  Large, separately
maintained knowledge bases such as KDA are resolved at spawn time, copied
verbatim into the child, and recorded in a content-addressed lock file.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


_IGNORED_PARTS = {".git", "__pycache__", ".mypy_cache", ".pytest_cache"}


@dataclasses.dataclass(frozen=True)
class SkillSource:
    name: str
    destination: str
    required: bool
    candidates: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class ResolvedSkill:
    spec: SkillSource
    source: Path
    sha256: str
    files: int


def load_manifest(path: Path) -> list[SkillSource]:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    specs: list[SkillSource] = []
    for raw in data.get("skill", []):
        name = str(raw["name"])
        destination = str(raw.get("destination", name))
        if Path(destination).name != destination or destination in {".", ".."}:
            raise ValueError(f"invalid skill destination {destination!r}")
        candidates = tuple(str(value) for value in raw.get("candidates", []))
        if not candidates:
            raise ValueError(f"skill {name!r} has no candidates")
        specs.append(SkillSource(name=name, destination=destination,
                                 required=bool(raw.get("required", True)),
                                 candidates=candidates))
    return specs


def _expand_candidate(value: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(value))).resolve()


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if any(part in _IGNORED_PARTS for part in path.relative_to(root).parts):
            continue
        if path.is_file():
            yield path


def tree_hash(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for path in _iter_files(root):
        rel = path.relative_to(root).as_posix().encode()
        digest.update(len(rel).to_bytes(8, "big"))
        digest.update(rel)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        count += 1
    return digest.hexdigest(), count


def resolve_skills(specs: Iterable[SkillSource], *, overrides: dict[str, Path] | None = None,
                   strict: bool = True) -> tuple[list[ResolvedSkill], list[str]]:
    overrides = overrides or {}
    resolved: list[ResolvedSkill] = []
    missing: list[str] = []
    for spec in specs:
        paths = [overrides[spec.name].resolve()] if spec.name in overrides else [
            _expand_candidate(candidate) for candidate in spec.candidates
        ]
        source = next((path for path in paths if (path / "SKILL.md").is_file()), None)
        if source is None:
            if spec.required:
                missing.append(f"{spec.name}: tried {', '.join(str(path) for path in paths)}")
            continue
        sha256, files = tree_hash(source)
        resolved.append(ResolvedSkill(spec=spec, source=source, sha256=sha256, files=files))
    if strict and missing:
        raise FileNotFoundError("required external skills are unavailable:\n  " + "\n  ".join(missing))
    return resolved, missing


def _copy_skill(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"skill destination already exists: {destination}")
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as tmp:
        staged = Path(tmp) / destination.name
        shutil.copytree(
            source,
            staged,
            ignore=shutil.ignore_patterns(".git", "__pycache__", ".mypy_cache", ".pytest_cache"),
            symlinks=False,
        )
        staged.rename(destination)


def materialize_skills(resolved: Iterable[ResolvedSkill], destination_root: Path,
                       *, lock_path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for item in resolved:
        destination = destination_root / item.spec.destination
        if destination.exists():
            copied_hash, copied_files = tree_hash(destination)
            if (copied_hash, copied_files) != (item.sha256, item.files):
                raise FileExistsError(
                    f"skill destination conflicts with source {item.spec.name}: {destination}"
                )
        else:
            _copy_skill(item.source, destination)
            copied_hash, copied_files = tree_hash(destination)
        if (copied_hash, copied_files) != (item.sha256, item.files):
            shutil.rmtree(destination, ignore_errors=True)
            raise RuntimeError(f"skill copy verification failed for {item.spec.name}")
        records.append({
            "name": item.spec.name,
            "destination": str(destination.relative_to(destination_root.parent.parent)),
            "source": str(item.source),
            "sha256": item.sha256,
            "files": item.files,
        })
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps({"version": 1, "skills": records}, indent=2,
                                    sort_keys=True) + "\n")
    return records


def parse_overrides(values: Iterable[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"skill override must be NAME=PATH, got {value!r}")
        result[name] = _expand_candidate(raw_path)
    return result
