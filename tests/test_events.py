import tempfile
import unittest
from pathlib import Path

from ako4x.events import RunStore, STATES


class EventStoreTests(unittest.TestCase):
    def test_strict_state_machine(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(Path(tmp) / "events.sqlite")
            run_id = store.start_run(Path(tmp), lane="auto", agent="codex")
            with self.assertRaises(ValueError):
                store.transition(run_id, "BASELINED")
            for state in STATES[1:]:
                store.transition(run_id, state)
            run = store.run(run_id)
            self.assertEqual(run["state"], "PROMOTED")
            self.assertEqual(run["status"], "complete")
            self.assertGreaterEqual(len(store.events(run_id)), len(STATES))
            store.close()

    def test_failure_is_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(Path(tmp) / "events.sqlite")
            run_id = store.start_run(Path(tmp))
            store.transition(run_id, "FAILED")
            with self.assertRaises(ValueError):
                store.transition(run_id, "PREFLIGHTED")

    def test_doctor_checkpoint_finishes_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(Path(tmp) / "events.sqlite")
            run_id = store.start_run(Path(tmp), lane="doctor")
            store.transition(run_id, "PREFLIGHTED")
            store.finish_checkpoint(run_id)
            run = store.run(run_id)
            self.assertEqual(run["state"], "PREFLIGHTED")
            self.assertEqual(run["status"], "complete")
            self.assertIsNotNone(run["finished_at"])


if __name__ == "__main__":
    unittest.main()
