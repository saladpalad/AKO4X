import json
import tempfile
import unittest
from pathlib import Path

from ako4x.agent_runtime import (
    AgentSpec,
    read_child_agent_metadata,
    resume_command,
    session_id_from_transcript,
    start_command,
    write_child_agent_metadata,
)


class AgentRuntimeTests(unittest.TestCase):
    def test_codex_start_and_resume(self):
        spec = AgentSpec(
            "codex", "codex", "AGENTS.md", ".agents/skills",
            network_access=True,
        )
        start = start_command(spec, "work")
        self.assertEqual(start[:4], [
            "codex", "-c", "sandbox_workspace_write.network_access=true", "exec",
        ])
        self.assertIn("workspace-write", start)
        self.assertIn("never", start)
        resume = resume_command(spec, "thread-1", "review")
        self.assertEqual(resume[:5], [
            "codex", "-c", "sandbox_workspace_write.network_access=true", "exec", "resume",
        ])
        self.assertNotIn("--sandbox", resume)

    def test_codex_network_setting_requires_workspace_write(self):
        spec = AgentSpec(
            "codex", "codex", "AGENTS.md", ".agents/skills",
            sandbox="read-only", network_access=True,
        )
        self.assertNotIn("sandbox_workspace_write.network_access=true",
                         start_command(spec, "work"))

    def test_network_access_must_be_boolean(self):
        with self.assertRaises(ValueError):
            AgentSpec.from_mapping("bad", {
                "runner": "codex", "task_filename": "AGENTS.md",
                "skills_dir": ".agents/skills", "network_access": "true",
            })

    def test_codex_session_from_jsonl(self):
        spec = AgentSpec("codex", "codex", "AGENTS.md", ".agents/skills")
        transcript = "noise\n" + json.dumps({"type": "thread.started", "thread_id": "abc"}) + "\n"
        self.assertEqual(session_id_from_transcript(spec, transcript), "abc")

    def test_metadata_round_trip(self):
        spec = AgentSpec("codex", "codex", "AGENTS.md", ".agents/skills")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_child_agent_metadata(root, spec)
            self.assertEqual(read_child_agent_metadata(root), spec)

    def test_invalid_skill_path(self):
        with self.assertRaises(ValueError):
            AgentSpec.from_mapping("bad", {
                "runner": "codex", "task_filename": "AGENTS.md", "skills_dir": "../skills"
            })


if __name__ == "__main__":
    unittest.main()
