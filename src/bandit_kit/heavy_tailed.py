"""Robust UCB for heavy-tailed rewards (Bubeck, Cesa-Bianchi & Lugosi, 2013).

UCB1, KL-UCB and Thompson sampling assume bounded or sub-Gaussian rewards.
With heavy-tailed rewards (revenue, latency, Student-t noise with infinite
variance) a single extreme draw can wreck the empirical mean and lock a
policy onto a bad arm for a long time. *Robust UCB* only assumes a finite
moment of order ``1 + epsilon`` for some ``epsilon in (0, 1]``:

* ``estimator="truncated"`` needs ``E|X|^(1+eps) <= u``. It uses the
  truncated empirical mean, dropping the i-th sample when
  ``|X_i| > (u * i / log(1/delta))^(1/(1+eps))``, with the confidence width
  ``4 u^(1/(1+eps)) (log(1/delta) / n)^(eps/(1+eps))``.
* ``estimator="median_of_means"`` needs ``E|X - mu|^(1+eps) <= v``. It takes
  the median of ``k = floor(min(8 log(1/delta) + 1, n / 2))`` block means,
  with the width ``(12 v)^(1/(1+eps)) (16 log(1/delta) / n)^(eps/(1+eps))``.

Both use the anytime schedule ``delta = t^-2``. With ``epsilon = 1`` and
bounded variance the regret is ``O(sum log T / Delta)``, like UCB1, but
without any boundedness assumption.
"""

from __future__ import annotations

import math
from typing import List, Sequence

from .algorithms import BanditAlgorithm, BanditStep
from .arms import Arm

__all__ = [
    "RobustUCB",
    "robust_ucb",
    "truncated_mean",
    "median_of_means",
]

_ESTIMATORS = ("truncated", "median_of_means")


def _check_epsilon(epsilon: float) -> float:
    if isinstance(epsilon, bool) or not 0.0 < float(epsilon) <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")
    return float(epsilon)


def truncated_mean(samples: Sequence[float], *, epsilon: float, moment_bound: float, log_inv_delta: float) -> float:
    """Truncated empirical mean of Bubeck et al. (2013), Lemma 1.

    Sample ``i`` (1-based) is kept only when
    ``|X_i| <= (moment_bound * i / log_inv_delta)^(1/(1+epsilon))``.
    The sum of kept samples is divided by ``n``, not by the number kept.
    """
    eps = _check_epsilon(epsilon)
    if moment_bound <= 0.0:
        raise ValueError("moment_bound must be positive")
    if log_inv_delta <= 0.0:
        raise ValueError("log_inv_delta must be positive")
    n = len(samples)
    if n == 0:
        raise ValueError("samples must be non-empty")
    total = 0.0
    power = 1.0 / (1.0 + eps)
    for i, x in enumerate(samples, start=1):
        if abs(x) <= (moment_bound * i / log_inv_delta) ** power:
            total += x
    return total / n


def median_of_means(samples: Sequence[float], n_blocks: int) -> float:
    """Median of the means of ``n_blocks`` equal consecutive blocks.

    Each block holds ``len(samples) // n_blocks`` samples. The trailing
    remainder is dropped, as in the analysis. With ``n_blocks=1`` this is
    the plain mean.
    """
    n = len(samples)
    if isinstance(n_blocks, bool) or int(n_blocks) != n_blocks or n_blocks < 1:
        raise ValueError("n_blocks must be an integer >= 1")
    if n < n_blocks:
        raise ValueError("need at least n_blocks samples")
    size = n // n_blocks
    means = sorted(sum(samples[b * size:(b + 1) * size]) / size for b in range(n_blocks))
    mid = n_blocks // 2
    if n_blocks % 2:
        return means[mid]
    return 0.5 * (means[mid - 1] + means[mid])


class RobustUCB(BanditAlgorithm):
    """Robust UCB with a truncated-mean or median-of-means estimator.

    Parameters
    ----------
    epsilon:
        Moment order offset: the policy assumes a finite ``(1 + epsilon)``-th
        moment. ``1.0`` means finite variance.
    moment_bound:
        ``u`` (a raw-moment bound for ``"truncated"``) or ``v`` (a central-moment
        bound for ``"median_of_means"``). Overestimating it only widens the
        confidence intervals.
    estimator:
        ``"median_of_means"`` (the default) or ``"truncated"``.
    confidence_scale:
        Multiplier on the confidence width. ``1.0`` keeps the theoretical
        constants, which are conservative. Smaller values explore less, and
        values around ``0.1``–``0.3`` usually work better in practice.
    """

    name = "robust_ucb"

    def __init__(
        self,
        epsilon: float = 1.0,
        moment_bound: float = 1.0,
        estimator: str = "median_of_means",
        confidence_scale: float = 1.0,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        if isinstance(confidence_scale, bool) or not float(confidence_scale) > 0.0:
            raise ValueError("confidence_scale must be positive")
        self.confidence_scale = float(confidence_scale)
        self.epsilon = _check_epsilon(epsilon)
        if isinstance(moment_bound, bool) or not float(moment_bound) > 0.0 or not math.isfinite(moment_bound):
            raise ValueError("moment_bound must be a positive finite number")
        if estimator not in _ESTIMATORS:
            raise ValueError(f"estimator must be one of {_ESTIMATORS}")
        self.moment_bound = float(moment_bound)
        self.estimator = estimator
        self._rewards: List[List[float]] = []
        self._t = 0

    def reset(self, arms: Sequence[Arm]) -> None:
        self._rewards = [[] for _ in arms]
        self._t = 0

    def _log_inv_delta(self) -> float:
        # delta = t^-2 with t >= 2 so the log stays positive on the first round.
        return 2.0 * math.log(max(self._t + 1, 2))

    def index(self, arm_index: int) -> float:
        """Current robust upper confidence index of ``arm_index``."""
        if arm_index < 0 or arm_index >= len(self._rewards):
            raise IndexError("arm_index out of range")
        xs = self._rewards[arm_index]
        n = len(xs)
        if n == 0:
            return math.inf
        eps = self.epsilon
        lid = self._log_inv_delta()
        if self.estimator == "truncated":
            mean = truncated_mean(xs, epsilon=eps, moment_bound=self.moment_bound, log_inv_delta=lid)
            width = 4.0 * self.moment_bound ** (1.0 / (1.0 + eps)) * (lid / n) ** (eps / (1.0 + eps))
        else:
            k = int(min(8.0 * lid + 1.0, n / 2.0))
            mean = median_of_means(xs, max(k, 1))
            width = (12.0 * self.moment_bound) ** (1.0 / (1.0 + eps)) * (16.0 * lid / n) ** (eps / (1.0 + eps))
        return mean + self.confidence_scale * width

    def select_arm(self, arms: Sequence[Arm], step: int) -> int:
        if not self._rewards:
            self.reset(arms)
        for idx, xs in enumerate(self._rewards):
            if not xs:
                return idx
        scores = [self.index(idx) for idx in range(len(arms))]
        best = max(scores)
        return scores.index(best)

    def update(self, arms: Sequence[Arm], step: BanditStep) -> None:
        if not self._rewards:
            self.reset(arms)
        self._rewards[step.arm_index].append(float(step.reward))
        self._t += 1

    def pull_counts(self) -> List[int]:
        return [len(xs) for xs in self._rewards]


def robust_ucb(
    epsilon: float = 1.0,
    moment_bound: float = 1.0,
    estimator: str = "median_of_means",
    confidence_scale: float = 1.0,
    *,
    seed: int | None = None,
) -> BanditAlgorithm:
    """Factory for :class:`RobustUCB`."""
    return RobustUCB(
        epsilon=epsilon,
        moment_bound=moment_bound,
        estimator=estimator,
        confidence_scale=confidence_scale,
        seed=seed,
    )
