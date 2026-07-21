import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ako4x.agent_runtime import AgentSpec, write_child_agent_metadata
from ako4x.events import RunStore, STATES
from ako4x.production import source_hash
from master import master


class MasterCodexTests(unittest.TestCase):
    def test_phase1_uses_codex_thread_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            child = Path(tmp)
            (child / "solution").mkdir()
            (child / "solution" / "kernel.py").write_text("value = 1\n")
            (child / "ITERATIONS.md").write_text("# iterations\n")
            write_child_agent_metadata(
                child, AgentSpec("codex", "codex", "AGENTS.md", ".agents/skills")
            )
            stdout = json.dumps({"type": "thread.started", "thread_id": "thread-42"}) + "\n"
            with mock.patch.object(master, "_run_bounded", return_value=(stdout, "", 0, False)) as run:
                result = master.run_sub_phase1(child, "optimize")
            command = run.call_args.args[0]
            self.assertEqual(command[:6], [
                "codex", "--sandbox", "workspace-write",
                "--ask-for-approval", "never", "exec",
            ])
            self.assertIn("--json", command)
            self.assertEqual(result.session_id, "thread-42")
            self.assertEqual((child / ".ako" / "session-id.txt").read_text().strip(), "thread-42")

    def test_retrospective_resumes_codex(self):
        with tempfile.TemporaryDirectory() as tmp:
            child = Path(tmp)
            write_child_agent_metadata(
                child, AgentSpec("codex", "codex", "AGENTS.md", ".agents/skills")
            )
            retrospective = child / "retro.md"
            retrospective.write_text("write proposals")
            with mock.patch.object(master, "_run_bounded", return_value=("", "", 0, False)) as run:
                master.send_retrospective_prompt(
                    child, "thread-42", retrospective_template_path=retrospective,
                    scope_path=None,
                )
            command = run.call_args.args[0]
            self.assertEqual(command[:3], ["codex", "exec", "resume"])
            self.assertIn("thread-42", command)

    def test_archive_requires_exact_production_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            child = Path(tmp)
            candidate = child / "solution"
            candidate.mkdir()
            (candidate / "kernel.py").write_text("value = 1\n")
            config_path = child / ".ako4x" / "production.toml"
            config_path.parent.mkdir()
            config_path.write_text("version = 1\n")
            store = RunStore(child / ".ako4x" / "runs.sqlite")
            run_id = store.start_run(child, agent="codex")
            store.bind_integrity(run_id, "integrity-1")
            for state in STATES[1:STATES.index("VERIFIED")]:
                store.transition(run_id, state)
            store.bind_source(run_id, source_hash(candidate))
            for state in STATES[STATES.index("VERIFIED"):STATES.index("PROMOTABLE") + 1]:
                store.transition(run_id, state)
            store.close()
            config = SimpleNamespace(project_root=child, candidate=candidate)
            with mock.patch("ako4x.production_config.load_production_config",
                            return_value=config), \
                 mock.patch("ako4x.production.integrity_manifest",
                            return_value={"sha256": "integrity-1", "paths": []}):
                evidence = master.verify_production_evidence(child)
                self.assertEqual(evidence["run_id"], run_id)
                (candidate / "kernel.py").write_text("value = 2\n")
                with self.assertRaisesRegex(RuntimeError, "no longer matches"):
                    master.verify_production_evidence(child)

    def test_closed_loop_profiles_before_agent_and_validates_after(self):
        with tempfile.TemporaryDirectory() as tmp:
            child = Path(tmp)
            (child / "solution").mkdir()
            (child / "solution" / "kernel.py").write_text("value = 1\n")
            (child / ".ako").mkdir()
            (child / ".ako" / "profile.json").write_text(
                json.dumps({"profile": "production"})
            )
            write_child_agent_metadata(
                child, AgentSpec("codex", "codex", "AGENTS.md", ".agents/skills")
            )
            order = []

            class Store:
                state = "CREATED"

                def run(self, run_id):
                    return {"state": self.state}

                def transition(self, run_id, state, payload=None):
                    self.state = state

                def close(self):
                    pass

            class Pipeline:
                def __init__(self, config):
                    self.store = Store()

                def new_run(self, **kwargs):
                    order.append("new")
                    return "prod-1"

                def prepare(self, run_id):
                    order.append("prepare")
                    self.store.state = "OPTIMIZING"

                def validate_candidate(self, run_id):
                    order.append("validate")

            def agent(*args, **kwargs):
                order.append("agent")
                stdout = json.dumps({"type": "thread.started", "thread_id": "thread-1"})
                return stdout + "\n", "", 0, False

            with mock.patch("ako4x.production_config.load_production_config",
                            return_value=object()), \
                 mock.patch("ako4x.production.ProductionPipeline", Pipeline), \
                 mock.patch.object(master, "_run_bounded", side_effect=agent):
                result = master.run_sub_phase1(child, "optimize")
            self.assertEqual(result.exit_status, 0)
            self.assertEqual(order, ["new", "prepare", "agent", "validate"])
            self.assertEqual((child / ".ako" / "production-run-id.txt").read_text().strip(),
                             "prod-1")


if __name__ == "__main__":
    unittest.main()
