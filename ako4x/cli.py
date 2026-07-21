"""Command-line entry point for AKO4X production supervision."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from .events import RunStore, STATES
from .lanes import create_lane, load_lane, run_autonomous_lane, run_hands_on_lane
from .production import ProductionPipeline
from .production_config import load_production_config
from .skill_sources import load_manifest, parse_overrides, resolve_skills


SOURCE_ROOT = Path(__file__).resolve().parent.parent
PACKAGED_ASSETS = Path(__file__).resolve().parent / "_assets"
TEMPLATE_ROOT = (
    SOURCE_ROOT / "templates"
    if (SOURCE_ROOT / "templates").is_dir()
    else PACKAGED_ASSETS / "templates"
)
RUNTIME_IGNORE = """# AKO4X production runtime (keep .ako4x/production.toml tracked)
.ako4x/runs/
.ako4x/runs.sqlite*
.ako4x/profiles/
.ako4x/promotions/
.ako4x/lanes/
.ako4x/worktrees/
"""


def _default_config(project: Path) -> Path:
    return project.resolve() / ".ako4x" / "production.toml"


def _load_pipeline(config_path: Path) -> ProductionPipeline:
    return ProductionPipeline(load_production_config(config_path))


def command_init(args: argparse.Namespace) -> int:
    project = Path(args.project).resolve()
    if not project.is_dir():
        raise FileNotFoundError(f"project directory does not exist: {project}")
    destination = _default_config(project)
    if destination.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {destination}; pass --force intentionally")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TEMPLATE_ROOT / "production" / "project.toml", destination)
    gitignore = project / ".gitignore"
    current = gitignore.read_text() if gitignore.exists() else ""
    if ".ako4x/runs.sqlite*" not in current:
        separator = "" if not current or current.endswith("\n\n") else ("\n" if current.endswith("\n") else "\n\n")
        gitignore.write_text(current + separator + RUNTIME_IGNORE)
    print(destination)
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    pipeline = _load_pipeline(Path(args.config))
    run_id = pipeline.new_run(lane="doctor", agent="none")
    try:
        pipeline.preflight(run_id)
    except Exception as exc:
        current = pipeline.store.run(run_id)["state"]
        if current not in {"FAILED", "PROMOTED"}:
            pipeline.store.transition(run_id, "FAILED", payload={"error": str(exc)})
        raise
    pipeline.store.finish_checkpoint(run_id)
    print(run_id)
    return 0


def command_run(args: argparse.Namespace) -> int:
    pipeline = _load_pipeline(Path(args.config))
    run_id = pipeline.run_all(lane=args.lane, agent=args.agent)
    print(run_id)
    return 0


def command_promote(args: argparse.Namespace) -> int:
    pipeline = _load_pipeline(Path(args.config))
    destination = pipeline.promote(args.run_id,
                                   destination=Path(args.destination).resolve() if args.destination else None)
    print(destination)
    return 0


def _status_payload(store: RunStore, run_id: str | None) -> dict:
    run = store.run(run_id) if run_id else store.latest_run()
    if run is None:
        return {"run": None, "events": []}
    return {"run": run, "events": store.events(run["id"])}


def _progress(state: str) -> str:
    width = 20
    if state == "FAILED":
        return "[FAILED]"
    completed = STATES.index(state) + 1 if state in STATES else 0
    filled = round(width * completed / len(STATES))
    return "[" + "#" * filled + "." * (width - filled) + f"] {completed}/{len(STATES)}"


def command_status(args: argparse.Namespace) -> int:
    project = Path(args.project).resolve()
    store = RunStore(project / ".ako4x" / "runs.sqlite")
    last_seq = 0
    while True:
        payload = _status_payload(store, args.run_id)
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        elif payload["run"] is None:
            print("No AKO4X runs found.")
        else:
            run = payload["run"]
            print(f"{_progress(run['state'])} {run['id']}  {run['state']}  "
                  f"{run['status']}  lane={run['lane']} agent={run['agent']}")
            for event in payload["events"]:
                if event["seq"] > last_seq:
                    print(f"  {event['seq']:04d} {event['timestamp']} {event['kind']} "
                          f"{event['phase'] or '-'} {event['payload']}")
                    last_seq = event["seq"]
        if not args.watch:
            return 0
        time.sleep(max(0.2, args.interval))


def command_skills(args: argparse.Namespace) -> int:
    manifest = TEMPLATE_ROOT / "production" / "skills.toml"
    overrides = parse_overrides(args.skill_source)
    resolved, missing = resolve_skills(load_manifest(manifest), overrides=overrides, strict=False)
    payload = {
        "resolved": [
            {"name": item.spec.name, "source": str(item.source),
             "sha256": item.sha256, "files": item.files}
            for item in resolved
        ],
        "missing": missing,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if missing else 0


def command_lane_create(args: argparse.Namespace) -> int:
    metadata = create_lane(
        Path(args.project), args.name, agent=args.agent, mode=args.mode,
        config_path=Path(args.config), skill_sources=args.skill_source,
        worktree_path=Path(args.worktree).resolve() if args.worktree else None,
        network_access=args.network_access,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"Launch through the enforced profiler/gate lifecycle: "
          f"ako4x-lab lane-run {args.name} --project {metadata['project']}")
    return 0


def command_lane_run(args: argparse.Namespace) -> int:
    metadata = load_lane(Path(args.project), args.name)
    if metadata["mode"] == "hands-on":
        run_id = run_hands_on_lane(metadata)
    else:
        run_id = run_autonomous_lane(metadata, timeout=args.timeout)
    print(run_id)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ako4x-lab",
                                     description="Production supervisor for AKO4X kernel campaigns")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Create strict production config in any repository")
    init.add_argument("project", nargs="?", default=".")
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=command_init)

    doctor = sub.add_parser("doctor", help="Require real, parseable NCU and NSYS smoke reports")
    doctor.add_argument("--config", required=True)
    doctor.set_defaults(func=command_doctor)

    run = sub.add_parser("run", help="Run baseline, gates, benchmark, and profiles")
    run.add_argument("--config", required=True)
    run.add_argument("--lane", default="candidate")
    run.add_argument("--agent", default="external")
    run.set_defaults(func=command_run)

    promote = sub.add_parser("promote", help="Promote an unchanged, evidence-bound candidate")
    promote.add_argument("--config", required=True)
    promote.add_argument("--run-id", required=True)
    promote.add_argument("--destination")
    promote.set_defaults(func=command_promote)

    status = sub.add_parser("status", help="Show run state and event history")
    status.add_argument("--project", default=".")
    status.add_argument("--run-id")
    status.add_argument("--watch", action="store_true")
    status.add_argument("--interval", type=float, default=2.0)
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=command_status)

    skills = sub.add_parser("skills", help="Resolve and hash required KDA/style skills")
    skills.add_argument("--skill-source", action="append", default=[], metavar="NAME=PATH")
    skills.set_defaults(func=command_skills)

    lane_create = sub.add_parser("lane-create", help="Create an isolated hands-on or autonomous worktree")
    lane_create.add_argument("name")
    lane_create.add_argument("--project", default=".")
    lane_create.add_argument("--config", required=True)
    lane_create.add_argument("--agent", choices=["codex", "claude"], default="codex")
    lane_create.add_argument("--mode", choices=["hands-on", "autonomous"], required=True)
    lane_create.add_argument("--worktree")
    lane_create.add_argument(
        "--network-access", action="store_true",
        help="allow the lane's Codex commands network access inside workspace-write",
    )
    lane_create.add_argument("--skill-source", action="append", default=[], metavar="NAME=PATH")
    lane_create.set_defaults(func=command_lane_create)

    lane_run = sub.add_parser("lane-run", help="Run a lane through enforced profiling and validation")
    lane_run.add_argument("name")
    lane_run.add_argument("--project", default=".")
    lane_run.add_argument("--timeout", type=int, default=18000)
    lane_run.set_defaults(func=command_lane_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"ako4x-lab: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
