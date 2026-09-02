"""Bandit algorithm implementations: epsilon-greedy, UCB1, Thompson sampling.

Each algorithm exposes a ``BanditAlgorithm`` factory. The factory
returns an object that owns the algorithm's runtime state and exposes a
``select_arm(arms, step)`` method returning the index to pull plus an
``update(step)`` method that records the reward and arm index for the
chosen arm.

Algorithms expose a :class:`BanditStep` value object that captures the
chosen arm, the observed reward, and the step number so downstream
metrics can compute cumulative reward and cumulative regret without
re-scanning every step.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Sequence

from .arms import Arm, BernoulliArm


@dataclass
class BanditStep:
    """A single observation from a bandit policy."""

    arm_index: int
    arm_name: str
    reward: float
    step: int


class BanditAlgorithm:
    """A bandit algorithm with a mutable state container.

    The factory functions :func:`epsilon_greedy`, :func:`ucb1`, and
    :func:`thompson_sampling_bernoulli` return subclasses of this object.
    ``reset`` rebuilds the algorithm's state so the same factory can be
    reused across independent runs.
    """

    name: str

    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def reset(self, arms: Sequence[Arm]) -> None:
        """Reset internal state for a fresh run with ``arms``."""

    def select_arm(self, arms: Sequence[Arm], step: int) -> int:
        raise NotImplementedError

    def update(self, arms: Sequence[Arm], step: BanditStep) -> None:
        raise NotImplementedError


class EpsilonGreedy(BanditAlgorithm):
    """Epsilon-greedy: explore uniformly with probability epsilon."""

    def __init__(self, epsilon: float = 0.1, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self.epsilon = float(epsilon)
        self._counts: List[int] = []
        self._values: List[float] = []

    def reset(self, arms: Sequence[Arm]) -> None:
        self._counts = [0] * len(arms)
        self._values = [0.0] * len(arms)

    def select_arm(self, arms: Sequence[Arm], step: int) -> int:
        if any(c == 0 for c in self._counts):
            return self._counts.index(0)
        if self._rng.random() < self.epsilon:
            return self._rng.randrange(len(arms))
        best = max(self._values)
        candidates = [idx for idx, value in enumerate(self._values) if value == best]
        return candidates[0]

    def update(self, arms: Sequence[Arm], step: BanditStep) -> None:
        idx = step.arm_index
        n = self._counts[idx]
        new_n = n + 1
        self._values[idx] += (step.reward - self._values[idx]) / new_n
        self._counts[idx] = new_n


class UCB1(BanditAlgorithm):
    """Upper Confidence Bound (UCB1) policy.

    Each step picks the arm with the largest upper confidence bound
    ``values[idx] + sqrt(2 * ln(step + 1) / counts[idx])`` once every arm
    has been pulled at least once; the first ``len(arms)`` steps round-robin
    through the arms to seed the counts.
    """

    def __init__(self, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._counts: List[int] = []
        self._values: List[float] = []
        self._round_robin = 0

    def reset(self, arms: Sequence[Arm]) -> None:
        self._counts = [0] * len(arms)
        self._values = [0.0] * len(arms)
        self._round_robin = 0

    def select_arm(self, arms: Sequence[Arm], step: int) -> int:
        zero_indices = [idx for idx, count in enumerate(self._counts) if count == 0]
        if zero_indices:
            chosen = zero_indices[self._round_robin % len(zero_indices)]
            self._round_robin += 1
            return chosen
        log_term = 2.0 * math.log(step + 1)
        scores = [
            self._values[idx] + math.sqrt(log_term / self._counts[idx])
            for idx in range(len(arms))
        ]
        best = max(scores)
        candidates = [idx for idx, score in enumerate(scores) if score == best]
        return candidates[0]

    def update(self, arms: Sequence[Arm], step: BanditStep) -> None:
        idx = step.arm_index
        n = self._counts[idx]
        new_n = n + 1
        self._values[idx] += (step.reward - self._values[idx]) / new_n
        self._counts[idx] = new_n


class ThompsonBernoulli(BanditAlgorithm):
    """Thompson sampling for Bernoulli arms.

    Each step samples a Beta(alpha, beta) posterior for every arm and
    picks the arm with the largest draw. Posterior updates use the
    Beta-Bernoulli conjugate rule (success increments alpha, failure
    increments beta). For non-Bernoulli arms the algorithm falls back to
    using the arm's ``expected_value`` as a fixed sample, which keeps
    the API consistent across arm types.
    """

    def __init__(self, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._alpha: List[float] = []
        self._beta: List[float] = []

    def reset(self, arms: Sequence[Arm]) -> None:
        self._alpha = [1.0] * len(arms)
        self._beta = [1.0] * len(arms)

    def select_arm(self, arms: Sequence[Arm], step: int) -> int:
        samples: List[float] = []
        for idx, arm in enumerate(arms):
            if isinstance(arm, BernoulliArm):
                samples.append(self._rng.betavariate(self._alpha[idx], self._beta[idx]))
            else:
                samples.append(arm.expected_value)
        best = max(samples)
        candidates = [idx for idx, value in enumerate(samples) if value == best]
        return candidates[self._rng.randrange(len(candidates))]

    def update(self, arms: Sequence[Arm], step: BanditStep) -> None:
        idx = step.arm_index
        reward = step.reward
        if reward >= 1.0:
            self._alpha[idx] += 1.0
        elif reward <= 0.0:
            self._beta[idx] += 1.0
        else:
            self._alpha[idx] += reward
            self._beta[idx] += 1.0 - reward


def epsilon_greedy(epsilon: float = 0.1, *, seed: int | None = None) -> BanditAlgorithm:
    return EpsilonGreedy(epsilon=epsilon, seed=seed)


def ucb1(*, seed: int | None = None) -> BanditAlgorithm:
    return UCB1(seed=seed)


def thompson_sampling_bernoulli(*, seed: int | None = None) -> BanditAlgorithm:
    return ThompsonBernoulli(seed=seed)


__all__ = [
    "BanditAlgorithm",
    "BanditStep",
    "EpsilonGreedy",
    "UCB1",
    "ThompsonBernoulli",
    "epsilon_greedy",
    "ucb1",
    "thompson_sampling_bernoulli",
]