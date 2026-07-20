"""Framework-neutral differential training trajectory checks.

Repository tests supply stateful reference/candidate steppers. Each step returns
named observables such as loss, outputs, gradients, and parameters. AKO4X owns
finite checks, tolerance evaluation, drift accounting, and a machine-readable
first-failure report.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np


class Stepper(Protocol):
    def __call__(self, step: int) -> Mapping[str, Any]: ...


StepperFactory = Callable[[int], Stepper]


@dataclasses.dataclass(frozen=True)
class Tolerance:
    atol: float = 1e-5
    rtol: float = 1e-4
    check_dtype: bool = True


@dataclasses.dataclass(frozen=True)
class TrajectoryConfig:
    seeds: tuple[int, ...] = (0, 1, 17)
    steps: int = 32
    default: Tolerance = Tolerance()
    fields: Mapping[str, Tolerance] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("training trajectory requires at least one seed")
        if self.steps <= 0:
            raise ValueError("training trajectory steps must be positive")


class TrainingDriftError(AssertionError):
    def __init__(self, message: str, report: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.report = report


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _write_report(path: Path | None, report: Mapping[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def compare_training_trajectories(
    reference_factory: StepperFactory,
    candidate_factory: StepperFactory,
    *,
    config: TrajectoryConfig = TrajectoryConfig(),
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Compare stateful repeated-use trajectories and return a JSON-safe report.

    A repository adapter should make each factory create an independently seeded
    mini training loop. The returned stepper performs one real update and emits
    matching named observables. Recommended fields are ``loss``, ``output``,
    ``grad/<name>``, and ``state/<name>``.
    """
    report: dict[str, Any] = {
        "status": "running",
        "seeds": list(config.seeds),
        "steps": config.steps,
        "comparisons": 0,
        "max_abs": {},
        "max_rel": {},
        "first_failure": None,
    }
    current_seed: int | None = None
    current_step: int | None = None
    try:
        for seed in config.seeds:
            current_seed = seed
            reference_step = reference_factory(seed)
            candidate_step = candidate_factory(seed)
            for step in range(config.steps):
                current_step = step
                reference = dict(reference_step(step))
                candidate = dict(candidate_step(step))
                if reference.keys() != candidate.keys():
                    _fail(report, seed, step, "<keys>",
                          f"observable keys differ: {sorted(reference)} != {sorted(candidate)}")
                for field in reference:
                    expected = _array(reference[field])
                    actual = _array(candidate[field])
                    tolerance = config.fields.get(field, config.default)
                    if expected.shape != actual.shape:
                        _fail(report, seed, step, field,
                              f"shape differs: {expected.shape} != {actual.shape}")
                    if tolerance.check_dtype and expected.dtype != actual.dtype:
                        _fail(report, seed, step, field,
                              f"dtype differs: {expected.dtype} != {actual.dtype}")
                    try:
                        expected_finite = np.all(np.isfinite(expected))
                        actual_finite = np.all(np.isfinite(actual))
                    except TypeError:
                        _fail(report, seed, step, field, "observable must be numeric")
                    if not expected_finite:
                        _fail(report, seed, step, field, "reference contains NaN/Inf")
                    if not actual_finite:
                        _fail(report, seed, step, field, "candidate contains NaN/Inf")
                    work_dtype = (
                        np.complex128
                        if np.iscomplexobj(expected) or np.iscomplexobj(actual)
                        else np.float64
                    )
                    expected64 = expected.astype(work_dtype, copy=False)
                    actual64 = actual.astype(work_dtype, copy=False)
                    absolute = np.abs(actual64 - expected64)
                    scale = np.maximum(np.abs(expected64), np.finfo(np.float64).tiny)
                    relative = absolute / scale
                    max_abs = float(absolute.max(initial=0.0))
                    max_rel = float(relative.max(initial=0.0))
                    report["max_abs"][field] = max(
                        float(report["max_abs"].get(field, 0.0)), max_abs
                    )
                    report["max_rel"][field] = max(
                        float(report["max_rel"].get(field, 0.0)), max_rel
                    )
                    report["comparisons"] += int(expected.size)
                    allowed = tolerance.atol + tolerance.rtol * np.abs(expected64)
                    if not np.all(absolute <= allowed):
                        flat = int(np.argmax(absolute - allowed))
                        index = np.unravel_index(flat, absolute.shape) if absolute.shape else ()
                        _fail(
                            report, seed, step, field,
                            f"drift at index {index}: expected={expected64[index]!r}, "
                            f"actual={actual64[index]!r}, abs={absolute[index]:.6g}, "
                            f"allowed={allowed[index]:.6g}",
                        )
        report["status"] = "passed"
        _write_report(report_path, report)
        return report
    except TrainingDriftError:
        report["status"] = "failed"
        _write_report(report_path, report)
        raise
    except Exception as exc:
        report["status"] = "failed"
        report["first_failure"] = {
            "seed": current_seed,
            "step": current_step,
            "field": "<exception>",
            "reason": f"{type(exc).__name__}: {exc}",
        }
        _write_report(report_path, report)
        raise


def _fail(report: dict[str, Any], seed: int, step: int, field: str,
          reason: str) -> None:
    failure = {"seed": seed, "step": step, "field": field, "reason": reason}
    report["first_failure"] = failure
    raise TrainingDriftError(
        f"training trajectory diverged at seed={seed}, step={step}, field={field}: {reason}",
        report,
    )
