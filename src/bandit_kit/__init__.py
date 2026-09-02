"""Multi-armed bandit algorithms, experiments, and reporting."""

from .arms import (
    Arm,
    BernoulliArm,
    GaussianArm,
    arm_from_spec,
    best_arm,
    expected_payoffs,
)

__all__ = [
    "Arm",
    "BernoulliArm",
    "GaussianArm",
    "arm_from_spec",
    "best_arm",
    "expected_payoffs",
]