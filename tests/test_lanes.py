import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ako4x.lanes import create_lane, hands_on_command, run_hands_on_lane
from ako4x.production_config import REQUIRED_GATES


def valid_config() -> str:
    parts = [
        "version = 1", "[project]", 'root = "."', 'candidate = "submission.py"',
        "[integrity]", 'protected = ["policy.txt"]',
        "[commands.baseline]", 'command = ["true"]',
        "[commands.benchmark]", 'command = ["true"]',
    ]
    for profiler in ("ncu", "nsys"):
        parts.extend([f"[profilers.{profiler}]", f'executable = "{profiler}"'])
        for stage in ("smoke", "baseline", "candidate"):
            parts.extend([
                f"[profilers.{profiler}.{stage}]", f'command = ["{profiler}"]',
                f'report = ".ako4x/{profiler}-{stage}.{profiler}-rep"',
                f"[profilers.{profiler}.{stage}.parse]", 'command = ["true"]',
            ])
    for gate in REQUIRED_GATES:
        parts.extend(["[[gate]]", f'name = "{gate}"', 'command = ["true"]'])
    return "\n".join(parts) + "\n"


class LaneTests(unittest.TestCase):
    def test_codex_hands_on_lane_materializes_skills(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
            (repo / "submission.py").write_text("candidate = True\n")
            (repo / "policy.txt").write_text("immutable tests\n")
            (repo / ".ako4x").mkdir()
            config = repo / ".ako4x" / "production.toml"
            config.write_text(valid_config().replace('root = "."', 'root = ".."'))
            subprocess.run(["git", "add", "submission.py", "policy.txt",
                            ".ako4x/production.toml"],
                           cwd=repo, check=True)
            subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=repo, check=True)

            overrides = []
            for name in ("KernelWiki", "ncu-report-skill", "cuda-kernel-style"):
                source = base / name
                source.mkdir()
                (source / "SKILL.md").write_text(
                    f"---\nname: {name}\ndescription: test\n---\ncontent\n"
                )
                overrides.append(f"{name}={source}")

            worktree = base / "worktree"
            metadata = create_lane(
                repo, "human", agent="codex", mode="hands-on", config_path=config,
                skill_sources=overrides, worktree_path=worktree,
            )
            self.assertTrue((worktree / ".agents" / "skills" / "kernelwiki" / "SKILL.md").is_file())
            self.assertTrue((worktree / ".agents" / "skills" / "production-kernel" / "SKILL.md").is_file())
            self.assertTrue((worktree / ".ako4x" / "skills.lock.json").is_file())
            self.assertEqual(metadata["mode"], "hands-on")
            launch = hands_on_command(metadata)
            self.assertEqual(launch[0], "codex")
            self.assertIn("--ask-for-approval", launch)
            self.assertIn("never", launch)

    def test_hands_on_agent_cannot_start_before_prepare(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worktree = root / "worktree"
            (worktree / ".ako4x").mkdir(parents=True)
            (root / ".ako4x" / "lanes").mkdir(parents=True)
            config = worktree / ".ako4x" / "production.toml"
            config.write_text(valid_config())
            metadata = {
                "name": "human", "mode": "hands-on", "agent": "codex",
                "runner": "codex", "worktree": str(worktree),
                "project": str(root), "config": str(config),
            }
            order = []

            class Store:
                state = "CREATED"

                def run(self, run_id):
                    return {"state": self.state}

                def emit(self, *args, **kwargs):
                    pass

                def transition(self, run_id, state, payload=None):
                    self.state = state

            class Pipeline:
                def __init__(self, config):
                    self.store = Store()

                def new_run(self, **kwargs):
                    order.append("new")
                    return "run-1"

                def prepare(self, run_id):
                    order.append("prepare")
                    self.store.state = "OPTIMIZING"

                def validate_candidate(self, run_id):
                    order.append("validate")

            def run_agent(*args, **kwargs):
                order.append("agent")
                return SimpleNamespace(returncode=0)

            with mock.patch("ako4x.lanes.load_production_config", return_value=object()), \
                 mock.patch("ako4x.lanes.ProductionPipeline", Pipeline), \
                 mock.patch("ako4x.lanes.subprocess.run", side_effect=run_agent):
                self.assertEqual(run_hands_on_lane(metadata), "run-1")
            self.assertEqual(order, ["new", "prepare", "agent", "validate"])


if __name__ == "__main__":
    unittest.main()
