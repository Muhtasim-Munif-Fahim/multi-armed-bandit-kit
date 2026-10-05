"""Bandit algorithm implementations: epsilon-greedy, UCB1, UCB-Tuned, sliding-window UCB, DiscountedUCB, UCB-V, UCB2, MOSS, Thompson sampling (Bernoulli and Gaussian), EXP3, LinUCB, LinTS, Boltzmann, KL-UCB.

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
    :func:`thompson_sampling_bernoulli`, :func:`thompson_sampling_gaussian`, :func:`exp3`, :func:`linucb`,
    :func:`lints`, :func:`boltzmann`, :func:`kl_ucb`, :func:`sliding_window_ucb`, :func:`sliding_window_thompson`, :func:`discounted_ucb`, and :func:`moss` return subclasses of this object.
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


class LinTS(BanditAlgorithm):
    """Disjoint linear Thompson sampling (Agrawal & Goyal 2013).

    Each arm keeps a Gaussian posterior for a linear payoff
    ``E[r | x, a] = theta_a · x``. With prior ``N(0, ridge^{-1} I)`` and a
    unit-variance Gaussian likelihood the posterior after pulls of arm
    ``a`` is

        ``theta_a | data ~ N(A_a^{-1} b_a, v^2 A_a^{-1})``

    where ``A_a = ridge * I + sum x x^T`` and ``b_a = sum r x``. At every
    step the policy draws one ``theta_a`` per arm and pulls

        ``argmax_a theta_a · x``.

    ``v`` scales posterior uncertainty. ``v = 0`` is greedy posterior-mean
    selection and matches LinUCB with ``alpha = 0`` on the same ridge
    design. ``A^{-1}`` is maintained with Sherman-Morrison rank-1 updates;
    samples use a Cholesky factor of ``A^{-1}`` so the implementation
    stays numpy-free.

    When no context is supplied and ``dimension == 1``, the policy uses
    the intercept ``[1.0]``. That reduces LinTS to one-dimensional
    Gaussian Thompson sampling and lets the stationary experiment harness
    run it without a context stream.
    """

    def __init__(
        self,
        v: float = 1.0,
        dimension: int = 1,
        ridge: float = 1.0,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        if v < 0.0:
            raise ValueError("v must be non-negative")
        if dimension < 1:
            raise ValueError("dimension must be at least 1")
        if ridge <= 0.0:
            raise ValueError("ridge must be positive")
        self.v = float(v)
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
                f"LinTS requires a context vector of length {self.dimension}"
            )
        values = [float(entry) for entry in context]
        if len(values) != self.dimension:
            raise ValueError(
                f"context length {len(values)} does not match dimension {self.dimension}"
            )
        return values

    def theta_hat(self, arm_index: int) -> List[float]:
        """Return the posterior mean weight vector for ``arm_index``."""
        return _linalg.matvec(self._ainv[arm_index], self._b[arm_index])

    def _draw_theta(self, arm_index: int) -> List[float]:
        mean = self.theta_hat(arm_index)
        if self.v == 0.0:
            return mean
        factor = _linalg.cholesky_lower(self._ainv[arm_index])
        noise = [self._rng.gauss(0.0, 1.0) for _ in range(self.dimension)]
        perturbation = _linalg.matvec(factor, noise)
        return _linalg.add_vectors(mean, _linalg.scale_vector(perturbation, self.v))

    def sample_theta(self, arm_index: int) -> List[float]:
        """Draw one posterior sample of the weight vector for ``arm_index``."""
        if not self._ainv:
            raise RuntimeError("sample_theta requires reset() before sampling")
        if arm_index < 0 or arm_index >= len(self._ainv):
            raise IndexError("arm_index out of range")
        return self._draw_theta(arm_index)

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
        scores = [
            _linalg.dot(self._draw_theta(idx), x) for idx in range(len(arms))
        ]
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


class Boltzmann(BanditAlgorithm):
    """Softmax / Boltzmann exploration with a decaying temperature schedule.

    At step ``t`` the policy samples arm ``i`` with probability

        ``P(i) = exp(Q[i] / tau_t) / sum_j exp(Q[j] / tau_t)``

    where ``Q[i]`` is the empirical mean reward of arm ``i`` and

        ``tau_t = temperature_min + (temperature_start - temperature_min) * decay ** t``

    High temperature is nearly uniform (explore); low temperature
    concentrates on the empirically best arm (exploit). The first
    unpulled arms are chosen in index order so every ``Q[i]`` is
    defined before softmax sampling begins. ``temperature_start`` must
    be positive; ``temperature_min`` is in ``[0, temperature_start]``
    and ``decay`` is in ``(0, 1)``. Setting ``temperature_min`` equal
    to ``temperature_start`` yields a constant-temperature policy.
    """

    _MIN_TEMPERATURE = 1e-12

    def __init__(
        self,
        temperature_start: float = 1.0,
        temperature_min: float = 0.05,
        decay: float = 0.99,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        if temperature_start <= 0.0:
            raise ValueError("temperature_start must be positive")
        if temperature_min < 0.0:
            raise ValueError("temperature_min must be non-negative")
        if temperature_min > temperature_start:
            raise ValueError("temperature_min must be <= temperature_start")
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be in (0, 1)")
        self.temperature_start = float(temperature_start)
        self.temperature_min = float(temperature_min)
        self.decay = float(decay)
        self._counts: List[int] = []
        self._values: List[float] = []

    def reset(self, arms: Sequence[Arm]) -> None:
        self._counts = [0] * len(arms)
        self._values = [0.0] * len(arms)

    def _current_temperature(self, step: int) -> float:
        return self.temperature_min + (self.temperature_start - self.temperature_min) * (
            self.decay ** step
        )

    def _softmax_probabilities(self, temperature: float) -> List[float]:
        values = self._values
        if not values:
            return []
        if temperature <= self._MIN_TEMPERATURE:
            best = max(values)
            n_best = sum(1 for value in values if value == best)
            return [1.0 / n_best if value == best else 0.0 for value in values]
        scaled = [value / temperature for value in values]
        max_scaled = max(scaled)
        exps = [math.exp(score - max_scaled) for score in scaled]
        total = sum(exps)
        return [value / total for value in exps]

    def _sample(self, probabilities: Sequence[float]) -> int:
        r = self._rng.random()
        cumulative = 0.0
        for idx, prob in enumerate(probabilities):
            cumulative += prob
            if r < cumulative:
                return idx
        return len(probabilities) - 1

    def select_arm(self, arms: Sequence[Arm], step: int) -> int:
        if any(c == 0 for c in self._counts):
            return self._counts.index(0)
        temperature = self._current_temperature(step)
        return self._sample(self._softmax_probabilities(temperature))

    def update(self, arms: Sequence[Arm], step: BanditStep) -> None:
        idx = step.arm_index
        n = self._counts[idx]
        new_n = n + 1
        self._values[idx] += (step.reward - self._values[idx]) / new_n
        self._counts[idx] = new_n

    def temperature_at(self, step: int) -> float:
        """Return the temperature schedule value at ``step`` (0-indexed)."""
        if step < 0:
            raise ValueError("step must be non-negative")
        return self._current_temperature(step)

    def probabilities_at(self, step: int) -> List[float]:
        """Return the softmax distribution at ``step`` given current values."""
        if step < 0:
            raise ValueError("step must be non-negative")
        return self._softmax_probabilities(self._current_temperature(step))


def _bernoulli_kl(p: float, q: float) -> float:
    """Return KL(Bernoulli(p) || Bernoulli(q)) in nats.

    ``p`` and ``q`` are assumed to lie in ``[0, 1]``. The divergence is
    ``+inf`` when ``q`` is on the boundary and ``p`` is not.
    """
    if q <= 0.0:
        return 0.0 if p <= 0.0 else math.inf
    if q >= 1.0:
        return 0.0 if p >= 1.0 else math.inf
    kl = 0.0
    if p > 0.0:
        kl += p * math.log(p / q)
    if p < 1.0:
        kl += (1.0 - p) * math.log((1.0 - p) / (1.0 - q))
    return kl


def _kl_ucb_threshold(time: int, c: float) -> float:
    """Return ``log(t) + c log(log(t))`` at 1-indexed time ``t``."""
    t = max(int(time), 1)
    log_t = math.log(t)
    if c == 0.0:
        return log_t
    # log(log(t)) is only positive for t > e; floor the inner log at 1
    # so the extra term stays non-negative on the first few steps.
    return log_t + c * math.log(max(log_t, 1.0))


def _kl_ucb_upper(mu: float, count: int, threshold: float, precision: float) -> float:
    """Largest ``q`` in ``[mu, 1]`` s.t. ``count * KL(mu, q) <= threshold``."""
    if count <= 0:
        return 1.0
    if mu >= 1.0:
        return 1.0
    if threshold <= 0.0:
        return mu
    lo = mu
    hi = 1.0
    for _ in range(64):
        if hi - lo <= precision:
            break
        mid = (lo + hi) / 2.0
        if count * _bernoulli_kl(mu, mid) <= threshold:
            lo = mid
        else:
            hi = mid
    return lo


class KLUCB(BanditAlgorithm):
    """KL-UCB (Garivier & Cappé, COLT 2011) for [0, 1]-bounded rewards.

    After every arm has been pulled once, each step picks the arm with
    the largest Bernoulli KL-UCB index

        ``sup { q ∈ [μ̂, 1] : N * d(μ̂, q) ≤ log(t) + c log log(t) }``

    where ``d`` is the Bernoulli KL divergence, ``μ̂`` is the empirical
    mean, ``N`` is the pull count, and ``t = step + 1``. ``c`` defaults
    to 0 (the empirically common ``log(t)`` threshold); ``c = 3``
    recovers the theoretically analysed extra ``log log(t)`` term.

    Bernoulli KL-UCB is valid for any reward in [0, 1] by Bernoulli
    domination. Rewards outside that interval are clipped, matching
    EXP3, so Gaussian arms still run through the same interface.
    """

    def __init__(
        self,
        c: float = 0.0,
        precision: float = 1e-6,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        if c < 0.0:
            raise ValueError("c must be non-negative")
        if precision <= 0.0:
            raise ValueError("precision must be positive")
        self.c = float(c)
        self.precision = float(precision)
        self._counts: List[int] = []
        self._values: List[float] = []
        self._round_robin = 0

    def reset(self, arms: Sequence[Arm]) -> None:
        self._counts = [0] * len(arms)
        self._values = [0.0] * len(arms)
        self._round_robin = 0

    def _index(self, arm_index: int, step: int) -> float:
        count = self._counts[arm_index]
        if count == 0:
            return math.inf
        threshold = _kl_ucb_threshold(step + 1, self.c)
        return _kl_ucb_upper(
            self._values[arm_index], count, threshold, self.precision
        )

    def upper_bound(self, arm_index: int, step: int) -> float:
        """Return the KL-UCB index of ``arm_index`` at 0-indexed ``step``."""
        if step < 0:
            raise ValueError("step must be non-negative")
        if arm_index < 0 or arm_index >= len(self._counts):
            raise IndexError("arm_index out of range")
        return self._index(arm_index, step)

    def select_arm(self, arms: Sequence[Arm], step: int) -> int:
        if not self._counts:
            self.reset(arms)
        zero_indices = [idx for idx, count in enumerate(self._counts) if count == 0]
        if zero_indices:
            chosen = zero_indices[self._round_robin % len(zero_indices)]
            self._round_robin += 1
            return chosen
        scores = [self._index(idx, step) for idx in range(len(arms))]
        best = max(scores)
        candidates = [idx for idx, score in enumerate(scores) if score == best]
        return candidates[0]

    def update(self, arms: Sequence[Arm], step: BanditStep) -> None:
        if not self._counts:
            self.reset(arms)
        idx = step.arm_index
        reward = min(1.0, max(0.0, float(step.reward)))
        n = self._counts[idx]
        new_n = n + 1
        self._values[idx] += (reward - self._values[idx]) / new_n
        self._counts[idx] = new_n




class UCBTuned(BanditAlgorithm):
    """UCB-Tuned (Auer, Cesa-Bianchi & Fischer, Machine Learning 2002).

    Variance-aware UCB for ``[0, 1]``-bounded rewards. After every arm has
    been pulled once, each step picks the arm with the largest index

    ``mean_i + sqrt( (ln t / n_i) * min(1/4, V_i) )``

    where ``t = step + 1``, ``n_i`` is the pull count,

    ``V_i = s_i^2 + sqrt(2 ln t / n_i)``,

    and ``s_i^2`` is the empirical second-moment variance
    ``(sum r^2)/n - mean^2`` (floored at 0). Capping the variance proxy
    at ``1/4`` recovers the Bernoulli worst case; when an arm looks
    nearly deterministic the bonus shrinks and the policy exploits
    faster than plain :class:`UCB1`.

    Rewards outside ``[0, 1]`` are clipped so Gaussian arms still run
    through the same interface (matching KL-UCB / EXP3).
    """

    def __init__(self, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._counts: List[int] = []
        self._sums: List[float] = []
        self._sum_sq: List[float] = []
        self._round_robin = 0

    def reset(self, arms: Sequence[Arm]) -> None:
        n = len(arms)
        self._counts = [0] * n
        self._sums = [0.0] * n
        self._sum_sq = [0.0] * n
        self._round_robin = 0

    def _index(self, arm_index: int, step: int) -> float:
        count = self._counts[arm_index]
        if count == 0:
            return math.inf
        t = max(step + 1, 1)
        mean = self._sums[arm_index] / count
        variance = self._sum_sq[arm_index] / count - mean * mean
        if variance < 0.0:
            variance = 0.0
        log_t = math.log(t)
        v = variance + math.sqrt(2.0 * log_t / count)
        bonus = math.sqrt((log_t / count) * min(0.25, v))
        return mean + bonus

    def upper_bound(self, arm_index: int, step: int) -> float:
        """Return the UCB-Tuned index of ``arm_index`` at 0-indexed ``step``."""
        if step < 0:
            raise ValueError("step must be non-negative")
        if arm_index < 0 or arm_index >= len(self._counts):
            raise IndexError("arm_index out of range")
        return self._index(arm_index, step)

    def select_arm(self, arms: Sequence[Arm], step: int) -> int:
        if not self._counts:
            self.reset(arms)
        zero_indices = [idx for idx, count in enumerate(self._counts) if count == 0]
        if zero_indices:
            chosen = zero_indices[self._round_robin % len(zero_indices)]
            self._round_robin += 1
            return chosen
        scores = [self._index(idx, step) for idx in range(len(arms))]
        best = max(scores)
        candidates = [idx for idx, score in enumerate(scores) if score == best]
        return candidates[0]

    def update(self, arms: Sequence[Arm], step: BanditStep) -> None:
        if not self._counts:
            self.reset(arms)
        idx = step.arm_index
        reward = min(1.0, max(0.0, float(step.reward)))
        self._counts[idx] += 1
        self._sums[idx] += reward
        self._sum_sq[idx] += reward * reward


class SlidingWindowUCB(BanditAlgorithm):
    """Sliding-window UCB1 (Garivier & Moulines, ALT 2011).

    Only the most recent ``window`` pulls contribute to each arm's empirical
    mean and count. After every arm has been pulled at least once inside the
    current window (or globally at the start), each step picks the arm with
    the largest index

    ``mean_i + sqrt(2 * ln(min(t, window)) / n_i)``

    where ``n_i`` and ``mean_i`` are formed from pulls of arm ``i`` among the
    last ``window`` steps, and ``t = step + 1``. When ``window`` is larger
    than the horizon the policy recovers ordinary :class:`UCB1`.

    The sliding window is the usual non-stationary adaptation of UCB1: old
    rewards fall out of the statistics so the policy can track a changing
    best arm. The first ``len(arms)`` steps still round-robin through arms
    that have never been pulled (including cold-start after a window wipe).
    """

    def __init__(self, window: int = 100, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError("window must be an integer >= 1")
        self.window = int(window)
        self._history: List[tuple[int, float]] = []
        self._counts: List[int] = []
        self._sums: List[float] = []
        self._round_robin = 0

    def reset(self, arms: Sequence[Arm]) -> None:
        n = len(arms)
        self._history = []
        self._counts = [0] * n
        self._sums = [0.0] * n
        self._round_robin = 0

    def _rebuild(self) -> None:
        n = len(self._counts)
        self._counts = [0] * n
        self._sums = [0.0] * n
        for arm_index, reward in self._history:
            self._counts[arm_index] += 1
            self._sums[arm_index] += reward

    def select_arm(self, arms: Sequence[Arm], step: int) -> int:
        if not self._counts:
            self.reset(arms)
        zero_indices = [idx for idx, count in enumerate(self._counts) if count == 0]
        if zero_indices:
            chosen = zero_indices[self._round_robin % len(zero_indices)]
            self._round_robin += 1
            return chosen
        t = min(step + 1, self.window)
        log_term = 2.0 * math.log(max(t, 1))
        scores = []
        for idx in range(len(arms)):
            count = self._counts[idx]
            mean = self._sums[idx] / count
            scores.append(mean + math.sqrt(log_term / count))
        best = max(scores)
        candidates = [idx for idx, score in enumerate(scores) if score == best]
        return candidates[0]

    def update(self, arms: Sequence[Arm], step: BanditStep) -> None:
        if not self._counts:
            self.reset(arms)
        self._history.append((step.arm_index, float(step.reward)))
        if len(self._history) > self.window:
            # Drop the oldest observation and rebuild window statistics.
            self._history = self._history[-self.window :]
            self._rebuild()
        else:
            idx = step.arm_index
            self._counts[idx] += 1
            self._sums[idx] += float(step.reward)









class SlidingWindowThompson(BanditAlgorithm):
    """Sliding-window Thompson sampling for non-stationary Bernoulli bandits.

    Maintains a global history of the most recent ``window`` pulls and runs
    Beta-Bernoulli Thompson sampling on the rewards that remain inside that
    window (same forgetting scheme as :class:`SlidingWindowUCB`). Each step
    samples ``Beta(alpha_i, beta_i)`` for every Bernoulli arm from the
    window-restricted posterior and picks the arm with the largest draw.
    Non-Bernoulli arms fall back to ``expected_value`` as a fixed sample,
    matching :class:`ThompsonBernoulli`.

    A short window forgets stale successes/failures so the policy can track
    a changing best arm; when ``window`` exceeds the horizon the policy
    recovers ordinary Bernoulli Thompson sampling. Pair with
    :class:`SlidingWindowUCB` / :class:`DiscountedUCB` for non-stationary
    comparisons.
    """

    def __init__(self, window: int = 100, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError("window must be an integer >= 1")
        self.window = int(window)
        self._history: List[tuple[int, float]] = []
        self._alpha: List[float] = []
        self._beta: List[float] = []

    def reset(self, arms: Sequence[Arm]) -> None:
        n = len(arms)
        self._history = []
        self._alpha = [1.0] * n
        self._beta = [1.0] * n

    def _rebuild(self) -> None:
        n = len(self._alpha)
        self._alpha = [1.0] * n
        self._beta = [1.0] * n
        for arm_index, reward in self._history:
            self._apply_reward(arm_index, reward)

    def _apply_reward(self, idx: int, reward: float) -> None:
        if reward >= 1.0:
            self._alpha[idx] += 1.0
        elif reward <= 0.0:
            self._beta[idx] += 1.0
        else:
            self._alpha[idx] += reward
            self._beta[idx] += 1.0 - reward

    def select_arm(self, arms: Sequence[Arm], step: int) -> int:
        if not self._alpha:
            self.reset(arms)
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
        if not self._alpha:
            self.reset(arms)
        self._history.append((step.arm_index, float(step.reward)))
        if len(self._history) > self.window:
            self._history = self._history[-self.window :]
            self._rebuild()
        else:
            self._apply_reward(step.arm_index, float(step.reward))


class DiscountedUCB(BanditAlgorithm):
    """Discounted UCB for non-stationary bandits (Garivier & Moulines).

    Maintains exponentially discounted pull counts and reward sums. After
    every arm has been pulled at least once, each step picks the arm with
    the largest index

    ``mean_i + sqrt(2 * ln(t) / N_i(γ))``

    where ``t = step + 1``, ``N_i(γ)`` is the discounted count of arm ``i``,
    and ``mean_i`` is the discounted reward sum divided by ``N_i(γ)``. On
    every update all counts and sums are multiplied by ``γ ∈ (0, 1]`` before
    the newest observation is added, so older rewards fade smoothly.

    When ``γ = 1`` the statistics are never discounted and the policy
    recovers ordinary :class:`UCB1`. Smaller ``γ`` forgets faster, which
    helps when the best arm changes over time. Alongside
    :class:`SlidingWindowUCB` this is the other classic Garivier & Moulines
    (ALT 2011) adaptation of UCB1 to non-stationary environments.
    """

    def __init__(self, gamma: float = 0.9, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        if not 0.0 < float(gamma) <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        self.gamma = float(gamma)
        self._disc_counts: List[float] = []
        self._disc_sums: List[float] = []
        self._round_robin = 0

    def reset(self, arms: Sequence[Arm]) -> None:
        n = len(arms)
        self._disc_counts = [0.0] * n
        self._disc_sums = [0.0] * n
        self._round_robin = 0

    def select_arm(self, arms: Sequence[Arm], step: int) -> int:
        if not self._disc_counts:
            self.reset(arms)
        zero_indices = [idx for idx, count in enumerate(self._disc_counts) if count <= 0.0]
        if zero_indices:
            chosen = zero_indices[self._round_robin % len(zero_indices)]
            self._round_robin += 1
            return chosen
        log_term = 2.0 * math.log(max(step + 1, 1))
        scores = []
        for idx in range(len(arms)):
            count = self._disc_counts[idx]
            mean = self._disc_sums[idx] / count
            scores.append(mean + math.sqrt(log_term / count))
        best = max(scores)
        candidates = [idx for idx, score in enumerate(scores) if score == best]
        return candidates[0]

    def update(self, arms: Sequence[Arm], step: BanditStep) -> None:
        if not self._disc_counts:
            self.reset(arms)
        g = self.gamma
        for idx in range(len(self._disc_counts)):
            self._disc_counts[idx] *= g
            self._disc_sums[idx] *= g
        arm_idx = step.arm_index
        self._disc_counts[arm_idx] += 1.0
        self._disc_sums[arm_idx] += float(step.reward)


class MOSS(BanditAlgorithm):
    """MOSS — Minimax Optimal Strategy in the Stochastic case (Audibert & Bubeck).

    Finite-horizon UCB-style index. After every arm has been pulled once,
    each step picks the arm with the largest index

    ``mean_a + sqrt( max(0, log(T / (n_a * K))) / (2 * n_a) )``

    where ``T`` is the known horizon, ``K`` is the number of arms, and
    ``n_a`` is the pull count of arm ``a``. The ``max(0, ·)`` floors the
    log term so arms that have already been pulled more than ``T/K`` times
    receive a zero exploration bonus (pure exploitation).

    Unlike :class:`UCB1`, the log argument shrinks with ``n_a``, which is
    what makes MOSS minimax-optimal for finite ``T``. Rewards outside
    ``[0, 1]`` are clipped so Gaussian arms still share the interface.
    """

    def __init__(self, horizon: int = 200, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be an integer >= 1")
        self.horizon = int(horizon)
        self._counts: List[int] = []
        self._sums: List[float] = []
        self._n_arms = 0
        self._round_robin = 0

    def reset(self, arms: Sequence[Arm]) -> None:
        n = len(arms)
        self._counts = [0] * n
        self._sums = [0.0] * n
        self._n_arms = n
        self._round_robin = 0

    def _index(self, arm_index: int) -> float:
        count = self._counts[arm_index]
        if count == 0:
            return math.inf
        mean = self._sums[arm_index] / count
        k = max(self._n_arms, 1)
        # log^+(T / (n_a * K))
        log_arg = self.horizon / (count * k)
        log_term = math.log(log_arg) if log_arg > 1.0 else 0.0
        if log_term < 0.0:
            log_term = 0.0
        bonus = math.sqrt(log_term / (2.0 * count))
        return mean + bonus

    def upper_bound(self, arm_index: int) -> float:
        """Return the MOSS index of ``arm_index`` given the current counts."""
        if arm_index < 0 or arm_index >= len(self._counts):
            raise IndexError("arm_index out of range")
        return self._index(arm_index)

    def select_arm(self, arms: Sequence[Arm], step: int) -> int:
        if not self._counts:
            self.reset(arms)
        zero_indices = [idx for idx, count in enumerate(self._counts) if count == 0]
        if zero_indices:
            chosen = zero_indices[self._round_robin % len(zero_indices)]
            self._round_robin += 1
            return chosen
        scores = [self._index(idx) for idx in range(len(arms))]
        best = max(scores)
        candidates = [idx for idx, score in enumerate(scores) if score == best]
        return candidates[0]

    def update(self, arms: Sequence[Arm], step: BanditStep) -> None:
        if not self._counts:
            self.reset(arms)
        idx = step.arm_index
        reward = min(1.0, max(0.0, float(step.reward)))
        self._counts[idx] += 1
        self._sums[idx] += reward





class GaussianThompson(BanditAlgorithm):
    """Thompson sampling for Gaussian arms with known observation noise.

    Maintains a Normal–Normal conjugate posterior for each arm's mean reward.
    With prior ``N(mu0, tau0^2)`` and known observation std ``sigma``, after
    observing rewards ``r_1, ..., r_n`` for an arm the posterior is

        ``mu | data ~ N(mu_n, 1 / lambda_n)``

    where ``lambda_n = 1/tau0^2 + n / sigma^2`` and
    ``mu_n = (mu0 / tau0^2 + sum(r) / sigma^2) / lambda_n``.

    Each step draws one sample from every arm's posterior and pulls the arm
    with the largest draw. ``sigma`` defaults to 1.0; a weakly informative
    prior uses ``mu0=0`` and ``tau0=1``. Seed support comes from the base
    :class:`BanditAlgorithm` RNG.
    """

    # Alias used by some callers / docs
    # ThompsonGaussian = set after class definition

    def __init__(
        self,
        mu0: float = 0.0,
        tau0: float = 1.0,
        sigma: float = 1.0,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        if tau0 <= 0.0:
            raise ValueError("tau0 must be positive")
        if sigma <= 0.0:
            raise ValueError("sigma must be positive")
        self.mu0 = float(mu0)
        self.tau0 = float(tau0)
        self.sigma = float(sigma)
        self._counts: List[int] = []
        self._sums: List[float] = []
        self._prior_precision = 1.0 / (self.tau0 * self.tau0)
        self._obs_precision = 1.0 / (self.sigma * self.sigma)

    def reset(self, arms: Sequence[Arm]) -> None:
        n = len(arms)
        self._counts = [0] * n
        self._sums = [0.0] * n

    def posterior_mean(self, arm_index: int) -> float:
        """Return the current posterior mean for ``arm_index``."""
        if arm_index < 0 or arm_index >= len(self._counts):
            raise IndexError("arm_index out of range")
        n = self._counts[arm_index]
        precision = self._prior_precision + n * self._obs_precision
        return (self._prior_precision * self.mu0 + self._sums[arm_index] * self._obs_precision) / precision

    def posterior_variance(self, arm_index: int) -> float:
        """Return the current posterior variance for ``arm_index``."""
        if arm_index < 0 or arm_index >= len(self._counts):
            raise IndexError("arm_index out of range")
        n = self._counts[arm_index]
        precision = self._prior_precision + n * self._obs_precision
        return 1.0 / precision

    def posterior_precision(self, arm_index: int) -> float:
        """Return the current posterior precision for ``arm_index``."""
        if arm_index < 0 or arm_index >= len(self._counts):
            raise IndexError("arm_index out of range")
        n = self._counts[arm_index]
        return self._prior_precision + n * self._obs_precision

    def select_arm(self, arms: Sequence[Arm], step: int) -> int:
        if not self._counts:
            self.reset(arms)
        samples: List[float] = []
        for idx in range(len(arms)):
            mean = self.posterior_mean(idx)
            var = self.posterior_variance(idx)
            samples.append(self._rng.gauss(mean, math.sqrt(var)))
        best = max(samples)
        candidates = [idx for idx, value in enumerate(samples) if value == best]
        return candidates[self._rng.randrange(len(candidates))]

    def update(self, arms: Sequence[Arm], step: BanditStep) -> None:
        if not self._counts:
            self.reset(arms)
        idx = step.arm_index
        self._counts[idx] += 1
        self._sums[idx] += float(step.reward)


ThompsonGaussian = GaussianThompson



class UCBV(BanditAlgorithm):
    """UCB-V: variance-aware UCB (Audibert, Munos & Szepesvári).

    Tracks per-arm first and second moments. After every arm has been
    pulled once, each step picks the arm with the largest index

    ``mean_i + sqrt(2 * V_i * ln(t) / n_i) + c * ln(t) / n_i``

    where ``t = step + 1``, ``n_i`` is the pull count, ``V_i`` is the
    empirical variance ``(sum r^2)/n - mean^2`` (floored at 0), and
    ``c >= 0`` is the additive exploration constant from Audibert et al.
    (default ``c = 3``, matching the classic bound for ``[0, 1]`` rewards
    with range bound ``b = 1``).

    Compared with :class:`UCBTuned`, UCB-V uses the Audibert exploration
    bonus (variance term plus an explicit ``c log(t)/n`` remainder)
    rather than the ``min(1/4, V)`` Bernoulli proxy. Rewards outside
    ``[0, 1]`` are clipped so Gaussian arms still run through the same
    interface (matching UCB-Tuned / KL-UCB / EXP3).
    """

    def __init__(self, c: float = 3.0, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        if isinstance(c, bool) or not isinstance(c, (int, float)) or float(c) < 0.0:
            raise ValueError("c must be a non-negative number")
        self.c = float(c)
        self._counts: List[int] = []
        self._sums: List[float] = []
        self._sum_sq: List[float] = []
        self._round_robin = 0

    def reset(self, arms: Sequence[Arm]) -> None:
        n = len(arms)
        self._counts = [0] * n
        self._sums = [0.0] * n
        self._sum_sq = [0.0] * n
        self._round_robin = 0

    def _index(self, arm_index: int, step: int) -> float:
        count = self._counts[arm_index]
        if count == 0:
            return math.inf
        t = max(step + 1, 1)
        mean = self._sums[arm_index] / count
        variance = self._sum_sq[arm_index] / count - mean * mean
        if variance < 0.0:
            variance = 0.0
        log_t = math.log(t)
        bonus = math.sqrt(2.0 * variance * log_t / count) + self.c * log_t / count
        return mean + bonus

    def upper_bound(self, arm_index: int, step: int) -> float:
        """Return the UCB-V index of ``arm_index`` at 0-indexed ``step``."""
        if step < 0:
            raise ValueError("step must be non-negative")
        if arm_index < 0 or arm_index >= len(self._counts):
            raise IndexError("arm_index out of range")
        return self._index(arm_index, step)

    def select_arm(self, arms: Sequence[Arm], step: int) -> int:
        if not self._counts:
            self.reset(arms)
        zero_indices = [idx for idx, count in enumerate(self._counts) if count == 0]
        if zero_indices:
            chosen = zero_indices[self._round_robin % len(zero_indices)]
            self._round_robin += 1
            return chosen
        scores = [self._index(idx, step) for idx in range(len(arms))]
        best = max(scores)
        candidates = [idx for idx, score in enumerate(scores) if score == best]
        return candidates[0]

    def update(self, arms: Sequence[Arm], step: BanditStep) -> None:
        if not self._counts:
            self.reset(arms)
        idx = step.arm_index
        reward = min(1.0, max(0.0, float(step.reward)))
        self._counts[idx] += 1
        self._sums[idx] += reward
        self._sum_sq[idx] += reward * reward



class UCB2(BanditAlgorithm):
    """UCB2 epoch-based UCB (Auer, Cesa-Bianchi & Fischer, 2002).

    After every arm has been pulled once, each *epoch* picks the arm ``j``
    that maximises

        ``mean_j + sqrt( (1+α) * ln(e * n / τ(r_j)) / (2 * τ(r_j)) )``

    where ``n`` is the total number of pulls so far, ``r_j`` is the number of
    completed epochs for arm ``j``, and

        ``τ(r) = ceil((1+α)^r)``.

    The chosen arm is then played for ``τ(r_j+1) - τ(r_j)`` consecutive
    steps before ``r_j`` is incremented. The parameter ``alpha`` ∈ ``(0, 1)``
    trades exploration (larger α → longer early epochs / larger bonus).
    Rewards outside ``[0, 1]`` are clipped so Gaussian arms still run.
    """

    def __init__(self, alpha: float = 0.1, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise ValueError("alpha must be a number in (0, 1)")
        alpha = float(alpha)
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self._counts: List[int] = []
        self._sums: List[float] = []
        self._epochs: List[int] = []
        self._total_pulls = 0
        self._round_robin = 0
        self._committed: int | None = None
        self._remaining = 0

    def reset(self, arms: Sequence[Arm]) -> None:
        n = len(arms)
        self._counts = [0] * n
        self._sums = [0.0] * n
        self._epochs = [0] * n
        self._total_pulls = 0
        self._round_robin = 0
        self._committed = None
        self._remaining = 0

    def tau(self, r: int) -> int:
        """Return ``ceil((1+α)^r)`` for epoch index ``r`` (≥ 0)."""
        if isinstance(r, bool) or not isinstance(r, int) or r < 0:
            raise ValueError("r must be a non-negative integer")
        return int(math.ceil((1.0 + self.alpha) ** r))

    def _radius(self, arm_index: int) -> float:
        r = self._epochs[arm_index]
        tau_r = max(self.tau(r), 1)
        n = max(self._total_pulls, 1)
        # a_{n,r} = sqrt( (1+α) ln(e n / τ(r)) / (2 τ(r)) )
        inside = (1.0 + self.alpha) * math.log((math.e * n) / tau_r) / (2.0 * tau_r)
        if inside < 0.0:
            inside = 0.0
        return math.sqrt(inside)

    def upper_bound(self, arm_index: int) -> float:
        """Return the UCB2 index of ``arm_index`` given current statistics."""
        if arm_index < 0 or arm_index >= len(self._counts):
            raise IndexError("arm_index out of range")
        count = self._counts[arm_index]
        if count == 0:
            return math.inf
        mean = self._sums[arm_index] / count
        return mean + self._radius(arm_index)

    def select_arm(self, arms: Sequence[Arm], step: int) -> int:
        if not self._counts:
            self.reset(arms)
        # Continue an in-progress epoch.
        if self._committed is not None and self._remaining > 0:
            return int(self._committed)
        # Warm-up: pull every arm once (no epoch accounting).
        zero_indices = [idx for idx, count in enumerate(self._counts) if count == 0]
        if zero_indices:
            chosen = zero_indices[self._round_robin % len(zero_indices)]
            self._round_robin += 1
            return chosen
        scores = [self.upper_bound(idx) for idx in range(len(arms))]
        best = max(scores)
        candidates = [idx for idx, score in enumerate(scores) if score == best]
        chosen = candidates[0]
        r = self._epochs[chosen]
        length = max(1, self.tau(r + 1) - self.tau(r))
        self._committed = chosen
        self._remaining = length
        return chosen

    def update(self, arms: Sequence[Arm], step: BanditStep) -> None:
        if not self._counts:
            self.reset(arms)
        idx = step.arm_index
        reward = min(1.0, max(0.0, float(step.reward)))
        self._counts[idx] += 1
        self._sums[idx] += reward
        self._total_pulls += 1
        if self._committed is None:
            # Warm-up updates: nothing else to do.
            return
        if idx != self._committed:
            # Defensive: ignore mismatched updates.
            return
        self._remaining -= 1
        if self._remaining <= 0:
            self._epochs[idx] += 1
            self._committed = None
            self._remaining = 0


def ucb2(alpha: float = 0.1, *, seed: int | None = None) -> BanditAlgorithm:
    return UCB2(alpha=alpha, seed=seed)


def ucb_tuned(*, seed: int | None = None) -> BanditAlgorithm:
    return UCBTuned(seed=seed)


def ucb1(*, seed: int | None = None) -> BanditAlgorithm:
    return UCB1(seed=seed)


def sliding_window_ucb(window: int = 100, *, seed: int | None = None) -> BanditAlgorithm:
    return SlidingWindowUCB(window=window, seed=seed)


def sliding_window_thompson(window: int = 100, *, seed: int | None = None) -> BanditAlgorithm:
    return SlidingWindowThompson(window=window, seed=seed)



def discounted_ucb(gamma: float = 0.9, *, seed: int | None = None) -> BanditAlgorithm:
    return DiscountedUCB(gamma=gamma, seed=seed)


def ucb_v(c: float = 3.0, *, seed: int | None = None) -> BanditAlgorithm:
    return UCBV(c=c, seed=seed)






def moss(horizon: int = 200, *, seed: int | None = None) -> BanditAlgorithm:
    return MOSS(horizon=horizon, seed=seed)



def thompson_sampling_bernoulli(*, seed: int | None = None) -> BanditAlgorithm:
    return ThompsonBernoulli(seed=seed)


def thompson_sampling_gaussian(
    mu0: float = 0.0,
    tau0: float = 1.0,
    sigma: float = 1.0,
    *,
    seed: int | None = None,
) -> BanditAlgorithm:
    return GaussianThompson(mu0=mu0, tau0=tau0, sigma=sigma, seed=seed)


def thompson_gaussian(
    mu0: float = 0.0,
    tau0: float = 1.0,
    sigma: float = 1.0,
    *,
    seed: int | None = None,
) -> BanditAlgorithm:
    """Alias for :func:`thompson_sampling_gaussian`."""
    return thompson_sampling_gaussian(mu0=mu0, tau0=tau0, sigma=sigma, seed=seed)






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


def lints(
    v: float = 1.0,
    dimension: int = 1,
    ridge: float = 1.0,
    *,
    seed: int | None = None,
) -> BanditAlgorithm:
    return LinTS(v=v, dimension=dimension, ridge=ridge, seed=seed)


def boltzmann(
    temperature_start: float = 1.0,
    temperature_min: float = 0.05,
    decay: float = 0.99,
    *,
    seed: int | None = None,
) -> BanditAlgorithm:
    return Boltzmann(
        temperature_start=temperature_start,
        temperature_min=temperature_min,
        decay=decay,
        seed=seed,
    )


def softmax(
    temperature_start: float = 1.0,
    temperature_min: float = 0.05,
    decay: float = 0.99,
    *,
    seed: int | None = None,
) -> BanditAlgorithm:
    """Alias for :func:`boltzmann`."""
    return boltzmann(
        temperature_start=temperature_start,
        temperature_min=temperature_min,
        decay=decay,
        seed=seed,
    )


Softmax = Boltzmann


def kl_ucb(
    c: float = 0.0,
    precision: float = 1e-6,
    *,
    seed: int | None = None,
) -> BanditAlgorithm:
    return KLUCB(c=c, precision=precision, seed=seed)


__all__ = [
    "BanditAlgorithm",
    "BanditStep",
    "EpsilonGreedy",
    "UCB1",
    "UCBTuned",
    "SlidingWindowUCB",
    "SlidingWindowThompson",
    "DiscountedUCB",
    "UCBV",
    "UCB2",
    "MOSS",
    "ThompsonBernoulli",
    "GaussianThompson",
    "ThompsonGaussian",
    "epsilon_greedy",
    "ucb1",
    "ucb_tuned",
    "sliding_window_ucb",
    "sliding_window_thompson",
    "discounted_ucb",
    "ucb_v",
    "ucb2",
    "moss",
    "thompson_sampling_bernoulli",
    "thompson_sampling_gaussian",
    "thompson_gaussian",
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
    "lints",
    "LinTS",
    "boltzmann",
    "Boltzmann",
    "softmax",
    "Softmax",
    "kl_ucb",
    "KLUCB",
]