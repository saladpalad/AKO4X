import json
import tempfile
import unittest
from pathlib import Path

from ako4x.skill_sources import SkillSource, materialize_skills, resolve_skills, tree_hash


class SkillSourceTests(unittest.TestCase):
    def test_verbatim_copy_and_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "SKILL.md").write_text("---\nname: demo\ndescription: demo\n---\nbody\n")
            (source / "knowledge.md").write_text("important\n")
            (source / ".git").mkdir()
            (source / ".git" / "ignored").write_text("ignore\n")
            spec = SkillSource("demo", "demo", True, (str(source),))
            resolved, missing = resolve_skills([spec])
            self.assertFalse(missing)
            destination = root / "child" / ".agents" / "skills"
            destination.mkdir(parents=True)
            lock = root / "child" / ".ako" / "skills.lock.json"
            materialize_skills(resolved, destination, lock_path=lock)
            self.assertEqual((destination / "demo" / "knowledge.md").read_text(), "important\n")
            self.assertFalse((destination / "demo" / ".git").exists())
            self.assertEqual(tree_hash(source), tree_hash(destination / "demo"))
            payload = json.loads(lock.read_text())
            self.assertEqual(payload["skills"][0]["sha256"], resolved[0].sha256)

    def test_required_missing_fails(self):
        spec = SkillSource("missing", "missing", True, ("/definitely/not/here",))
        with self.assertRaises(FileNotFoundError):
            resolve_skills([spec])

    def test_matching_existing_skill_is_reused_but_conflict_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "SKILL.md").write_text("same\n")
            spec = SkillSource("demo", "demo", True, (str(source),))
            resolved, _ = resolve_skills([spec])
            destination = root / "child" / ".agents" / "skills"
            (destination / "demo").mkdir(parents=True)
            (destination / "demo" / "SKILL.md").write_text("same\n")
            materialize_skills(resolved, destination,
                               lock_path=root / "child" / ".ako" / "lock.json")
            (destination / "demo" / "SKILL.md").write_text("different\n")
            with self.assertRaises(FileExistsError):
                materialize_skills(resolved, destination,
                                   lock_path=root / "child" / ".ako" / "lock2.json")


if __name__ == "__main__":
    unittest.main()
