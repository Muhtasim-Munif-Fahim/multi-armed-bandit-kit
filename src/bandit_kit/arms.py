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
    reward_range: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("arm name must be a non-empty string")
        if not math.isfinite(self.expected_value):
            raise ValueError("expected_value must be finite")
        lo, hi = self.reward_range
        if lo > hi:
            raise ValueError("reward_range lower bound must not exceed upper bound")


class BernoulliArm(Arm):
    """An arm whose reward is a Bernoulli(p) sample.

    An optional ``seed`` makes the draws reproducible across calls so
    experiment harnesses can drive the same arm set from different seedable
    algorithms without contention.
    """

    def __init__(self, name: str, p: float, *, seed: int | None = None) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError("Bernoulli probability must be in [0, 1]")
        rng = random.Random(seed)
        super().__init__(
            name=name,
            expected_value=float(p),
            draw=lambda: 1.0 if rng.random() < float(p) else 0.0,
            reward_range=(0.0, 1.0),
        )
        self.p = float(p)


class GaussianArm(Arm):
    """An arm whose reward is a Gaussian(mean, std) sample."""

    def __init__(self, name: str, mean: float, std: float, *, seed: int | None = None) -> None:
        if std <= 0.0:
            raise ValueError("Gaussian std must be positive")
        rng = random.Random(seed)
        rng_range = (mean - 4.0 * std, mean + 4.0 * std)
        super().__init__(
            name=name,
            expected_value=float(mean),
            draw=lambda: rng.gauss(float(mean), float(std)),
            reward_range=rng_range,
        )
        self.mean = float(mean)
        self.std = float(std)


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


class LinearContextualArm:
    """An arm whose expected reward is the inner product ``theta · x``.

    Unlike :class:`BernoulliArm` / :class:`GaussianArm`, the mean is not
    stationary: callers pass a context vector to :meth:`draw` and
    :meth:`expected_given`. Observation noise is Gaussian with standard
    deviation ``noise_std`` so the model matches the linear
    ridge-regression estimator used by disjoint LinUCB and LinTS.
    """

    def __init__(
        self,
        name: str,
        theta: Sequence[float],
        noise_std: float = 0.1,
        *,
        seed: int | None = None,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("arm name must be a non-empty string")
        theta_list = [float(value) for value in theta]
        if not theta_list:
            raise ValueError("theta must be a non-empty vector")
        if not all(math.isfinite(value) for value in theta_list):
            raise ValueError("theta entries must be finite")
        if noise_std < 0.0:
            raise ValueError("noise_std must be non-negative")
        self.name = name
        self.theta = tuple(theta_list)
        self.noise_std = float(noise_std)
        self._rng = random.Random(seed)

    @property
    def dimension(self) -> int:
        return len(self.theta)

    def expected_given(self, context: Sequence[float]) -> float:
        """Return the noiseless expected reward ``theta · context``."""
        if len(context) != len(self.theta):
            raise ValueError(
                f"context length {len(context)} does not match theta length {len(self.theta)}"
            )
        return sum(weight * float(value) for weight, value in zip(self.theta, context))

    def draw(self, context: Sequence[float]) -> float:
        """Draw a (possibly noisy) reward for ``context``."""
        mean = self.expected_given(context)
        if self.noise_std == 0.0:
            return mean
        return mean + self._rng.gauss(0.0, self.noise_std)


def sample_context(
    dimension: int,
    rng: random.Random,
    *,
    intercept: bool = True,
) -> list[float]:
    """Draw a bounded context vector with coordinates in ``[-1, 1]``.

    When ``intercept`` is true the first coordinate is fixed at ``1.0`` so
    linear models can learn a per-arm baseline.
    """
    if dimension < 1:
        raise ValueError("dimension must be at least 1")
    if intercept:
        if dimension == 1:
            return [1.0]
        return [1.0] + [rng.uniform(-1.0, 1.0) for _ in range(dimension - 1)]
    return [rng.uniform(-1.0, 1.0) for _ in range(dimension)]


def make_linear_contextual_arms(
    n_arms: int,
    dimension: int,
    *,
    noise_std: float = 0.1,
    seed: int | None = None,
    intercept: bool = True,
) -> list[LinearContextualArm]:
    """Build ``n_arms`` synthetic linear-payoff arms with random thetas."""
    if n_arms < 1:
        raise ValueError("n_arms must be at least 1")
    if dimension < 1:
        raise ValueError("dimension must be at least 1")
    rng = random.Random(seed)
    arms: list[LinearContextualArm] = []
    for idx in range(n_arms):
        theta = [rng.uniform(-1.0, 1.0) for _ in range(dimension)]
        if intercept:
            theta[0] = rng.uniform(-0.25, 0.25)
        arms.append(
            LinearContextualArm(
                name=f"arm_{idx}",
                theta=theta,
                noise_std=noise_std,
                seed=rng.randrange(2**31),
            )
        )
    return arms


def best_arm_given(
    arms: Sequence[LinearContextualArm],
    context: Sequence[float],
) -> LinearContextualArm:
    """Return the arm with the largest expected payoff for ``context``.

    Ties are broken by the original arm order so the result is stable.
    """
    if not arms:
        raise ValueError("at least one arm is required")
    best = arms[0]
    best_value = best.expected_given(context)
    for arm in arms[1:]:
        value = arm.expected_given(context)
        if value > best_value:
            best = arm
            best_value = value
    return best


__all__ = [
    "Arm",
    "BernoulliArm",
    "GaussianArm",
    "LinearContextualArm",
    "arm_from_spec",
    "best_arm",
    "best_arm_given",
    "expected_payoffs",
    "make_linear_contextual_arms",
    "sample_context",
]