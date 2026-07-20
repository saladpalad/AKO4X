import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ako4x.training import (
    Tolerance, TrainingDriftError, TrajectoryConfig, compare_training_trajectories,
)


def factory(delta: float):
    def build(seed: int):
        state = np.array([float(seed)], dtype=np.float32)

        def step(index: int):
            gradient = np.array([1.0 + delta], dtype=np.float32)
            state[:] = state - 0.1 * gradient
            return {
                "loss": state * state,
                "grad/x": gradient,
                "state/x": state.copy(),
            }

        return step

    return build


class TrainingTrajectoryTests(unittest.TestCase):
    def test_identical_stateful_training_passes(self):
        config = TrajectoryConfig(seeds=(0, 3), steps=4)
        report = compare_training_trajectories(factory(0.0), factory(0.0), config=config)
        self.assertEqual(report["status"], "passed")
        self.assertGreater(report["comparisons"], 0)

    def test_accumulating_bias_records_first_failing_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "training.json"
            config = TrajectoryConfig(
                seeds=(0,), steps=8,
                default=Tolerance(atol=2.5e-4, rtol=0.0),
                fields={
                    "loss": Tolerance(atol=1e-2, rtol=0.0),
                    "grad/x": Tolerance(atol=1e-2, rtol=0.0),
                },
            )
            with self.assertRaises(TrainingDriftError):
                compare_training_trajectories(
                    factory(1e-3), factory(0.0), config=config,
                    report_path=report_path,
                )
            report = json.loads(report_path.read_text())
            self.assertEqual(report["status"], "failed")
            self.assertIsNotNone(report["first_failure"]["step"])
            self.assertGreater(report["first_failure"]["step"], 0)
            self.assertIn(report["first_failure"]["field"],
                          {"loss", "grad/x", "state/x"})


if __name__ == "__main__":
    unittest.main()
