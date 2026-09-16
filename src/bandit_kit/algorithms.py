"""Bandit algorithm implementations: epsilon-greedy, UCB1, Thompson sampling, EXP3, LinUCB.

Each algorithm exposes a ``BanditAlgorithm`` factory. The factory
returns an object that owns the algorithm's runtime state and exposes a
``select_arm(arms, step)`` method returning the index to pull plus an
``update(step)`` method that records the reward and arm index for the
chosen arm.

Algorithms expose a :class:`BanditStep` value object that captures the
chosen arm, the observed reward, and the step number so downstream
metrics can compute cumulative reward and cumulative regret without
re-scanning every step. Contextual policies may also store the context
vector observed at that step.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Sequence

from . import _linalg
from .arms import Arm, BernoulliArm


@dataclass
class BanditStep:
    """A single observation from a bandit policy."""

    arm_index: int
    arm_name: str
    reward: float
    step: int
    context: tuple[float, ...] | None = None


class BanditAlgorithm:
    """A bandit algorithm with a mutable state container.

    The factory functions :func:`epsilon_greedy`, :func:`ucb1`,
    :func:`thompson_sampling_bernoulli`, :func:`exp3`, and :func:`linucb`
    return subclasses of this object.
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




class BayesianUCB(BanditAlgorithm):
    """Bayesian upper-confidence bound for Bernoulli arms.

    Maintains a Beta(alpha, beta) posterior per Bernoulli arm and picks
    the arm with the largest ``mu + lambda * sigma`` where ``mu`` and
    ``sigma`` are the posterior mean and standard deviation. ``lambda``
    defaults to 2.0 and is configurable per-instance. Non-Bernoulli
    arms fall back to ``expected_value`` as a fixed mean and a default
    standard deviation of 1.0 so the algorithm still runs on
    heterogeneous arm sets.
    """

    def __init__(self, lam: float = 2.0, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        if lam <= 0.0:
            raise ValueError("lam must be positive")
        self.lam = float(lam)
        self._alpha: list[float] = []
        self._beta: list[float] = []

    def reset(self, arms: Sequence[Arm]) -> None:
        self._alpha = [1.0] * len(arms)
        self._beta = [1.0] * len(arms)

    def select_arm(self, arms: Sequence[Arm], step: int) -> int:
        scores: list[float] = []
        for idx, arm in enumerate(arms):
            if isinstance(arm, BernoulliArm):
                alpha = self._alpha[idx]
                beta = self._beta[idx]
                mean = alpha / (alpha + beta)
                var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1.0))
                sigma = var ** 0.5
            else:
                mean = float(arm.expected_value)
                sigma = 1.0
            scores.append(mean + self.lam * sigma)
        best = max(scores)
        candidates = [idx for idx, value in enumerate(scores) if value == best]
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




class DecayingEpsilonGreedy(BanditAlgorithm):
    """Epsilon-greedy with an exponential epsilon decay schedule.

    ``epsilon_t = epsilon_min + (epsilon_start - epsilon_min) * decay ** t``
    so the policy explores broadly at the start of the run and
    exploits more aggressively as evidence accumulates. ``decay`` is in
    (0, 1) and ``epsilon_min`` is in [0, 1] with ``epsilon_start`` in
    (epsilon_min, 1].
    """

    def __init__(
        self,
        epsilon_start: float = 0.5,
        epsilon_min: float = 0.05,
        decay: float = 0.99,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        if not 0.0 <= epsilon_min < epsilon_start <= 1.0:
            raise ValueError(
                "epsilon_min must be in [0, 1) and 0 <= epsilon_min < epsilon_start <= 1"
            )
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be in (0, 1)")
        self.epsilon_start = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.decay = float(decay)
        self._counts: list[int] = []
        self._values: list[float] = []

    def reset(self, arms: Sequence[Arm]) -> None:
        self._counts = [0] * len(arms)
        self._values = [0.0] * len(arms)

    def _current_epsilon(self, step: int) -> float:
        return self.epsilon_min + (self.epsilon_start - self.epsilon_min) * (
            self.decay ** step
        )

    def select_arm(self, arms: Sequence[Arm], step: int) -> int:
        if any(c == 0 for c in self._counts):
            return self._counts.index(0)
        epsilon = self._current_epsilon(step)
        if self._rng.random() < epsilon:
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

    def epsilon_at(self, step: int) -> float:
        """Return the epsilon schedule value at ``step`` (0-indexed)."""
        if step < 0:
            raise ValueError("step must be non-negative")
        return self._current_epsilon(step)




class GradientBandit(BanditAlgorithm):
    """Gradient bandit with a softmax policy and a per-arm learning rate.

    The policy maintains a preference ``H[i]`` per arm and picks arms with
    probability ``softmax(H)[i]``. After every reward observation the
    preferences are updated as ``H[i] += alpha * (reward - baseline) * (1 - p[i])``
    on the chosen arm and ``H[j] -= alpha * (reward - baseline) * p[j]`` on
    every other arm, where ``baseline`` is the running mean reward
    (across all arms). ``alpha`` defaults to 0.1 and is configurable.
    Continuous rewards in any range are accepted.
    """

    def __init__(self, alpha: float = 0.1, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        self.alpha = float(alpha)
        self._preferences: list[float] = []
        self._all_rewards: list[float] = []

    def reset(self, arms: Sequence[Arm]) -> None:
        self._preferences = [0.0] * len(arms)
        self._all_rewards = []

    def _softmax(self) -> list[float]:
        prefs = self._preferences
        max_pref = max(prefs)
        exps = [float(p) for p in [pow(2.718281828459045, p - max_pref) for p in prefs]]
        total = sum(exps)
        return [value / total for value in exps]

    def select_arm(self, arms: Sequence[Arm], step: int) -> int:
        probabilities = self._softmax()
        r = self._rng.random()
        cumulative = 0.0
        for idx, prob in enumerate(probabilities):
            cumulative += prob
            if r < cumulative:
                return idx
        return len(probabilities) - 1

    def update(self, arms: Sequence[Arm], step: BanditStep) -> None:
        idx = step.arm_index
        reward = float(step.reward)
        self._all_rewards.append(reward)
        baseline = sum(self._all_rewards) / len(self._all_rewards)
        probabilities = self._softmax()
        for j in range(len(arms)):
            if j == idx:
                self._preferences[j] += self.alpha * (reward - baseline) * (1 - probabilities[j])
            else:
                self._preferences[j] -= self.alpha * (reward - baseline) * probabilities[j]


class Exp3(BanditAlgorithm):
    """EXP3 adversarial bandit (Auer, Cesa-Bianchi, Freund, Schapire).

    Maintains a weight ``w[i]`` per arm and samples arm ``i`` with
    probability ``(1 - gamma) * w[i] / sum(w) + gamma / K``. After
    observing a reward the chosen arm's weight is multiplied by
    ``exp(gamma * (reward / p[i]) / K)`` using the importance-weighted
    estimate. ``gamma`` in (0, 1] is the exploration mixing rate
    (default 0.1). Rewards are clipped to [0, 1] so the update stays
    well-defined for unbounded arm types such as Gaussians.
    """

    def __init__(self, gamma: float = 0.1, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        self.gamma = float(gamma)
        self._weights: list[float] = []
        self._last_probabilities: list[float] = []

    def reset(self, arms: Sequence[Arm]) -> None:
        self._weights = [1.0] * len(arms)
        self._last_probabilities = []

    def _distribution(self) -> list[float]:
        n = len(self._weights)
        total = sum(self._weights)
        return [
            (1.0 - self.gamma) * (weight / total) + self.gamma / n
            for weight in self._weights
        ]

    def select_arm(self, arms: Sequence[Arm], step: int) -> int:
        probabilities = self._distribution()
        self._last_probabilities = probabilities
        r = self._rng.random()
        cumulative = 0.0
        for idx, prob in enumerate(probabilities):
            cumulative += prob
            if r < cumulative:
                return idx
        return len(probabilities) - 1

    def update(self, arms: Sequence[Arm], step: BanditStep) -> None:
        n = len(self._weights)
        if len(self._last_probabilities) != n:
            self._last_probabilities = self._distribution()
        idx = step.arm_index
        reward = min(1.0, max(0.0, float(step.reward)))
        probability = self._last_probabilities[idx]
        estimated = reward / probability
        self._weights[idx] *= math.exp(self.gamma * estimated / n)
        max_weight = max(self._weights)
        if max_weight > 0.0:
            self._weights = [weight / max_weight for weight in self._weights]
        self._last_probabilities = []


class LinUCB(BanditAlgorithm):
    """Disjoint LinUCB (Li, Chu, Langford, Schapire 2010).

    Each arm maintains its own ridge-regression estimate ``theta_a`` of a
    linear payoff model ``E[r | x, a] = theta_a · x``. At every step the
    policy observes a shared context vector ``x`` and picks the arm that
    maximises

        ``theta_a · x + alpha * sqrt(x^T A_a^{-1} x)``

    where ``A_a = ridge * I + sum x x^T`` over pulls of arm ``a``.
    ``A^{-1}`` is maintained with Sherman-Morrison rank-1 updates so the
    implementation stays numpy-free.

    When no context is supplied and ``dimension == 1``, the policy uses
    the intercept ``[1.0]``. That reduces LinUCB to ridge-UCB on
    stationary (non-contextual) rewards and lets the existing experiment
    harness run it without a context stream.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        dimension: int = 1,
        ridge: float = 1.0,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        if dimension < 1:
            raise ValueError("dimension must be at least 1")
        if ridge <= 0.0:
            raise ValueError("ridge must be positive")
        self.alpha = float(alpha)
        self.dimension = int(dimension)
        self.ridge = float(ridge)
        self._ainv: List[List[List[float]]] = []
        self._b: List[List[float]] = []
        self._last_context: List[float] | None = None

    def reset(self, arms: Sequence[Arm]) -> None:
        n_arms = len(arms)
        scale = 1.0 / self.ridge
        self._ainv = [_linalg.identity(self.dimension, scale) for _ in range(n_arms)]
        self._b = [_linalg.zeros(self.dimension) for _ in range(n_arms)]
        self._last_context = None

    def _coerce_context(self, context: Sequence[float] | None) -> List[float]:
        if context is None:
            if self.dimension == 1:
                return [1.0]
            if self._last_context is not None:
                return list(self._last_context)
            raise ValueError(
                f"LinUCB requires a context vector of length {self.dimension}"
            )
        values = [float(entry) for entry in context]
        if len(values) != self.dimension:
            raise ValueError(
                f"context length {len(values)} does not match dimension {self.dimension}"
            )
        return values

    def _score(self, arm_index: int, context: Sequence[float]) -> float:
        inverse = self._ainv[arm_index]
        theta = _linalg.matvec(inverse, self._b[arm_index])
        mean = _linalg.dot(theta, context)
        variance = _linalg.quadratic_form(inverse, context)
        if variance < 0.0:
            variance = 0.0
        return mean + self.alpha * math.sqrt(variance)

    def select_arm(
        self,
        arms: Sequence[Arm],
        step: int,
        context: Sequence[float] | None = None,
    ) -> int:
        if not self._ainv:
            self.reset(arms)
        x = self._coerce_context(context)
        self._last_context = x
        scores = [self._score(idx, x) for idx in range(len(arms))]
        best = max(scores)
        candidates = [idx for idx, score in enumerate(scores) if score == best]
        return candidates[0]

    def update(
        self,
        arms: Sequence[Arm],
        step: BanditStep,
        context: Sequence[float] | None = None,
    ) -> None:
        if not self._ainv:
            self.reset(arms)
        x = self._coerce_context(context)
        idx = step.arm_index
        self._ainv[idx] = _linalg.sherman_morrison_update(self._ainv[idx], x)
        self._b[idx] = _linalg.add_vectors(
            self._b[idx], _linalg.scale_vector(x, float(step.reward))
        )

    def theta_hat(self, arm_index: int) -> List[float]:
        """Return the current ridge-regression weight vector for ``arm_index``."""
        return _linalg.matvec(self._ainv[arm_index], self._b[arm_index])


def ucb1(*, seed: int | None = None) -> BanditAlgorithm:
    return UCB1(seed=seed)


def thompson_sampling_bernoulli(*, seed: int | None = None) -> BanditAlgorithm:
    return ThompsonBernoulli(seed=seed)




def bayesian_ucb(lam: float = 2.0, *, seed: int | None = None) -> BanditAlgorithm:
    return BayesianUCB(lam=lam, seed=seed)




def decaying_epsilon_greedy(
    epsilon_start: float = 0.5,
    epsilon_min: float = 0.05,
    decay: float = 0.99,
    *,
    seed: int | None = None,
) -> BanditAlgorithm:
    return DecayingEpsilonGreedy(
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        decay=decay,
        seed=seed,
    )




def gradient_bandit(alpha: float = 0.1, *, seed: int | None = None) -> BanditAlgorithm:
    return GradientBandit(alpha=alpha, seed=seed)


def exp3(gamma: float = 0.1, *, seed: int | None = None) -> BanditAlgorithm:
    return Exp3(gamma=gamma, seed=seed)


def linucb(
    alpha: float = 1.0,
    dimension: int = 1,
    ridge: float = 1.0,
    *,
    seed: int | None = None,
) -> BanditAlgorithm:
    return LinUCB(alpha=alpha, dimension=dimension, ridge=ridge, seed=seed)


__all__ = [
    "BanditAlgorithm",
    "BanditStep",
    "EpsilonGreedy",
    "UCB1",
    "ThompsonBernoulli",
    "epsilon_greedy",
    "ucb1",
    "thompson_sampling_bernoulli",
    "bayesian_ucb",
    "BayesianUCB",
    "decaying_epsilon_greedy",
    "DecayingEpsilonGreedy",
    "gradient_bandit",
    "GradientBandit",
    "exp3",
    "Exp3",
    "linucb",
    "LinUCB",
]