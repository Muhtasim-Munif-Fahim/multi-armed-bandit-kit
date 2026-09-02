"""Arm representations: parametric reward distributions for bandit problems."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence


@dataclass(frozen=True)
class Arm:
    """A single arm parameterised by a draw callable.

    ``draw`` returns a single reward sample when called with no arguments.
    ``expected_value`` is the analytic expectation used by the regret
    calculation. Arms must be hashable by ``name`` so they can be stored
    in dicts.
    """

    name: str
    expected_value: float
    draw: Callable[[], float]
    reward_range: tuple[float, float] = field(default=(0.0, 1.0))

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("arm name must be a non-empty string")
        if not math.isfinite(self.expected_value):
            raise ValueError("expected_value must be finite")
        lo, hi = self.reward_range
        if lo > hi:
            raise ValueError("reward_range lower bound must not exceed upper bound")


@dataclass(frozen=True)
class BernoulliArm(Arm):
    """An arm whose reward is a Bernoulli(p) sample."""

    p: float

    def __init__(self, name: str, p: float) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError("Bernoulli probability must be in [0, 1]")
        super().__init__(
            name=name,
            expected_value=float(p),
            reward_range=(0.0, 1.0),
        )
        object.__setattr__(self, "p", float(p))


@dataclass(frozen=True)
class GaussianArm(Arm):
    """An arm whose reward is a Gaussian(mean, std) sample."""

    mean: float
    std: float

    def __init__(self, name: str, mean: float, std: float) -> None:
        if std <= 0.0:
            raise ValueError("Gaussian std must be positive")
        rng_range = (mean - 4.0 * std, mean + 4.0 * std)
        super().__init__(
            name=name,
            expected_value=float(mean),
            reward_range=rng_range,
        )
        object.__setattr__(self, "mean", float(mean))
        object.__setattr__(self, "std", float(std))


def arm_from_spec(spec: str) -> Arm:
    """Construct an Arm from a short string spec.

    Supported formats:
      ``bern:<p>``        -> BernoulliArm with success probability p
      ``gauss:<m>:<s>``   -> GaussianArm with mean m and std s

    Specs without a recognised prefix raise ``ValueError``.
    """
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("arm spec must be a non-empty string")
    parts = spec.strip().split(":")
    kind = parts[0].lower()
    if kind == "bern" and len(parts) == 2:
        return BernoulliArm(name=spec, p=float(parts[1]))
    if kind == "gauss" and len(parts) == 3:
        mean = float(parts[1])
        std = float(parts[2])
        return GaussianArm(name=spec, mean=mean, std=std)
    raise ValueError(f"unrecognised arm spec: {spec!r}")


def best_arm(arms: Sequence[Arm]) -> Arm:
    """Return the arm with the largest ``expected_value``.

    Ties are broken by the original arm order so the result is stable.
    """
    if not arms:
        raise ValueError("at least one arm is required")
    best = arms[0]
    for arm in arms[1:]:
        if arm.expected_value > best.expected_value:
            best = arm
    return best


def expected_payoffs(arms: Sequence[Arm]) -> Dict[str, float]:
    """Map arm name to expected payoff for use in regret calculations."""
    return {arm.name: arm.expected_value for arm in arms}


__all__ = [
    "Arm",
    "BernoulliArm",
    "GaussianArm",
    "arm_from_spec",
    "best_arm",
    "expected_payoffs",
]