import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from ako4x.cli import command_init


class CliTests(unittest.TestCase):
    def test_init_keeps_config_trackable_and_ignores_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".gitignore").write_text("build/\n")
            result = command_init(Namespace(project=str(root), force=False))
            self.assertEqual(result, 0)
            self.assertTrue((root / ".ako4x" / "production.toml").is_file())
            ignore = (root / ".gitignore").read_text()
            self.assertIn(".ako4x/runs.sqlite*", ignore)
            self.assertNotIn(".ako4x/production.toml", ignore.splitlines())


if __name__ == "__main__":
    unittest.main()
