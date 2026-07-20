import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from ako4x.events import RunStore
from ako4x.production import PipelineFailure, ProductionPipeline, source_hash
from ako4x.production_config import REQUIRED_GATES, load_production_config


HELPER = r'''
import pathlib
import sys

action = sys.argv[1]
if action == "write":
    path = pathlib.Path(sys.argv[2])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("real profiler report\n")
elif action == "parse":
    path = pathlib.Path(sys.argv[2])
    assert path.read_text().startswith("real profiler")
elif action == "mutate":
    pathlib.Path("submission.py").write_text("changed = True\n")
elif action == "mutate-integrity":
    pathlib.Path("policy.txt").write_text("weakened\n")
elif action == "ok":
    pass
else:
    raise SystemExit(4)
'''

PROFILER = r'''#!/usr/bin/env python3
import pathlib
import sys

args = sys.argv[1:]
if args == ["--version"]:
    print("fake-profiler 1.0")
elif args and args[0] == "capture":
    path = pathlib.Path(args[-1])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("native profiler report\n")
elif "--import" in args or (args and args[0] == "stats") or (args and args[0] == "custom-parse"):
    path = pathlib.Path(args[-1] if "--import" not in args else args[args.index("--import") + 1])
    assert path.read_text().startswith("native profiler")
else:
    raise SystemExit(5)
'''


def command(action, path=None):
    values = [sys.executable, "helper.py", action]
    if path:
        values.append(path)
    return "[" + ", ".join(json.dumps(value) for value in values) + "]"


def make_config(root: Path, *, mutate_gate: str | None = None,
                evidence_gate: str | None = None,
                mutate_integrity_gate: str | None = None) -> Path:
    parts = [
        "version = 1",
        "[project]", 'root = "."', 'candidate = "submission.py"',
        "[integrity]", 'protected = ["helper.py", "policy.txt", "ncu", "nsys"]',
        "[backend]", 'plugin = "local"',
        "[commands.baseline]", f"command = {command('ok')}",
        "[commands.benchmark]", f"command = {command('ok')}",
    ]
    for profiler in ("ncu", "nsys"):
        executable = str(root / profiler)
        parts.extend([f"[profilers.{profiler}]", f'executable = {json.dumps(executable)}'])
        for stage in ("smoke", "baseline", "candidate"):
            report = f".ako4x/profiles/{profiler}-{stage}.{profiler}-rep"
            parts.extend([
                f"[profilers.{profiler}.{stage}]",
                "command = [" + ", ".join(json.dumps(value) for value in
                                            (executable, "capture", report)) + "]",
                f'report = "{report}"',
                f"[profilers.{profiler}.{stage}.parse]",
                "command = [" + ", ".join(json.dumps(value) for value in
                                            (executable, "custom-parse", report)) + "]",
            ])
    for gate in REQUIRED_GATES:
        action = (
            "mutate" if gate == mutate_gate
            else "mutate-integrity" if gate == mutate_integrity_gate
            else "ok"
        )
        parts.extend(["[[gate]]", f'name = "{gate}"', f"command = {command(action)}"])
        if gate == evidence_gate:
            parts.append('evidence = ["stale.json"]')
    path = root / "production.toml"
    path.write_text("\n".join(parts) + "\n")
    return path


class ProductionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "helper.py").write_text(textwrap.dedent(HELPER))
        (self.root / "policy.txt").write_text("immutable\n")
        for profiler in ("ncu", "nsys"):
            path = self.root / profiler
            path.write_text(textwrap.dedent(PROFILER))
            path.chmod(0o755)
        (self.root / "submission.py").write_text("candidate = True\n")

    def tearDown(self):
        self.temp.cleanup()

    def test_full_pipeline_and_promotion(self):
        config = load_production_config(make_config(self.root))
        pipeline = ProductionPipeline(config)
        run_id = pipeline.run_all(agent="codex")
        self.assertEqual(pipeline.store.run(run_id)["state"], "PROMOTABLE")
        expected_hash = source_hash(self.root / "submission.py")
        self.assertEqual(pipeline.store.run(run_id)["source_hash"], expected_hash)
        snapshots = list((self.root / ".ako4x" / "runs" / run_id / "artifacts").rglob("*.ncu-rep"))
        self.assertEqual(len(snapshots), 3)
        destination = pipeline.promote(run_id)
        self.assertTrue((destination / "promotion.json").is_file())
        self.assertEqual(pipeline.store.run(run_id)["state"], "PROMOTED")

    def test_mutating_gate_is_rejected(self):
        config = load_production_config(make_config(self.root, mutate_gate="stream"))
        pipeline = ProductionPipeline(config)
        with self.assertRaises(PipelineFailure):
            pipeline.run_all()
        self.assertEqual(pipeline.store.latest_run()["state"], "FAILED")

    def test_missing_gate_is_configuration_error(self):
        path = make_config(self.root)
        text = path.read_text()
        marker = '[[gate]]\nname = "reviewability"'
        path.write_text(text[:text.index(marker)])
        with self.assertRaises(ValueError):
            load_production_config(path)

    def test_stale_gate_evidence_is_deleted_and_rejected(self):
        (self.root / "stale.json").write_text("old success\n")
        config = load_production_config(make_config(self.root, evidence_gate="correctness"))
        pipeline = ProductionPipeline(config)
        with self.assertRaises(PipelineFailure):
            pipeline.run_all()
        self.assertFalse((self.root / "stale.json").exists())

    def test_gate_cannot_weaken_protected_tests(self):
        config = load_production_config(
            make_config(self.root, mutate_integrity_gate="correctness")
        )
        pipeline = ProductionPipeline(config)
        with self.assertRaisesRegex(PipelineFailure, "protected"):
            pipeline.run_all()
        self.assertEqual(pipeline.store.latest_run()["state"], "FAILED")

    def test_promotion_rechecks_protected_infrastructure(self):
        config = load_production_config(make_config(self.root))
        pipeline = ProductionPipeline(config)
        run_id = pipeline.run_all()
        (self.root / "policy.txt").write_text("changed after validation\n")
        with self.assertRaisesRegex(PipelineFailure, "protected infrastructure"):
            pipeline.promote(run_id)


if __name__ == "__main__":
    unittest.main()
