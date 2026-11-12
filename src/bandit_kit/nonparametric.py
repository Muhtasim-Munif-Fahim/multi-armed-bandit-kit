"""Randomised-history bandit policies: PHE and BESA.

* **PHE** -- Perturbed-History Exploration (Kveton, Szepesvári,
  Ghavamzadeh & Boutilier, IJCAI 2019). Before scoring an arm pulled ``s``
  times, ``ceil(a * s)`` pseudo-rewards drawn i.i.d. from ``Bernoulli(1/2)``
  are appended to its reward history and the arm with the highest mean of
  the *perturbed* history is pulled. No posterior or confidence bound is
  needed; with ``a > 1`` the expected ``n``-round regret on ``[0, 1]``
  rewards is ``O(K Delta^-1 log n)``. Rewards are mapped to ``[0, 1]`` with
  one affine map spanning all arms' ``reward_range`` values (and clipped).

* **BESA** -- Best Empirical Sampled Average (Baransi, Maillard & Mannor,
  ECML-PKDD 2014). Two arms with ``n_a < n_b`` pulls are compared fairly by
  sub-sampling ``n_a`` of arm ``b``'s rewards *without replacement* and
  comparing that sub-sample's mean with arm ``a``'s full mean; ties go to
  the less-pulled arm. ``K`` arms are handled with a balanced knock-out
  tournament of such duels. BESA has no tuning parameter and needs no
  knowledge of the reward family.
"""

from __future__ import annotations

import math
from typing import List, Sequence

from .algorithms import BanditAlgorithm, BanditStep
from .arms import Arm

__all__ = ["PHE", "BESA", "phe", "besa"]


class PHE(BanditAlgorithm):
    """Perturbed-History Exploration for bounded rewards.

    Parameters
    ----------
    perturbation_scale : float
        ``a`` in the paper: the number of Bernoulli(1/2) pseudo-rewards per
        observed reward. The regret guarantee needs ``a > 1``; the default
        ``a = 2`` is a conservative choice satisfying it (smaller values
        explore less).
    """

    name = "phe"

    def __init__(self, perturbation_scale: float = 2.0, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        if isinstance(perturbation_scale, bool) or not (
            isinstance(perturbation_scale, (int, float)) and math.isfinite(perturbation_scale)
        ) or perturbation_scale <= 0:
            raise ValueError("perturbation_scale must be a positive finite number")
        self.perturbation_scale = float(perturbation_scale)
        self._counts: List[int] = []
        self._sums: List[float] = []
        self._ranges: List[tuple[float, float]] = []
        self.last_scores: List[float] = []

    def reset(self, arms: Sequence[Arm]) -> None:
        n = len(arms)
        if n == 0:
            raise ValueError("at least one arm is required")
        self._counts = [0] * n
        self._sums = [0.0] * n
        ranges = []
        for arm in arms:
            lo, hi = arm.reward_range
            if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
                raise ValueError("PHE needs every arm to have a finite, non-degenerate reward_range")
            ranges.append((float(lo), float(hi)))
        # one shared affine map so arms stay comparable after rescaling
        lo = min(r[0] for r in ranges)
        hi = max(r[1] for r in ranges)
        self._ranges = [(lo, hi)] * n
        self.last_scores = [math.nan] * n

    def _scaled(self, idx: int, reward: float) -> float:
        lo, hi = self._ranges[idx]
        return min(1.0, max(0.0, (float(reward) - lo) / (hi - lo)))

    def _binomial_half(self, n: int) -> int:
        # exact Binomial(n, 1/2) via popcount of random bits
        total = 0
        while n > 0:
            take = min(n, 64)
            total += bin(self._rng.getrandbits(take)).count("1")
            n -= take
        return total

    def select_arm(self, arms: Sequence[Arm], step: int) -> int:
        if not self._counts:
            self.reset(arms)
        for i, c in enumerate(self._counts):
            if c == 0:
                return i
        scores = []
        for s, total in zip(self._counts, self._sums):
            extra = math.ceil(self.perturbation_scale * s)
            pseudo = self._binomial_half(extra)
            scores.append((total + pseudo) / (s + extra))
        self.last_scores = scores
        best = max(scores)
        ties = [i for i, v in enumerate(scores) if v == best]
        return ties[0] if len(ties) == 1 else self._rng.choice(ties)

    def update(self, arms: Sequence[Arm], step: BanditStep) -> None:
        if not self._counts:
            self.reset(arms)
        idx = step.arm_index
        self._counts[idx] += 1
        self._sums[idx] += self._scaled(idx, step.reward)


def phe(perturbation_scale: float = 2.0, *, seed: int | None = None) -> BanditAlgorithm:
    """Factory for :class:`PHE`."""
    return PHE(perturbation_scale=perturbation_scale, seed=seed)


class BESA(BanditAlgorithm):
    """Best Empirical Sampled Average with a knock-out tournament for ``K`` arms."""

    name = "besa"

    def __init__(self, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._history: List[List[float]] = []

    def reset(self, arms: Sequence[Arm]) -> None:
        if len(arms) == 0:
            raise ValueError("at least one arm is required")
        self._history = [[] for _ in arms]

    @property
    def counts(self) -> List[int]:
        return [len(h) for h in self._history]

    def duel(self, a: int, b: int) -> int:
        """Return the winner of a BESA duel between arms ``a`` and ``b``."""
        ha, hb = self._history[a], self._history[b]
        n = min(len(ha), len(hb))
        if n == 0:
            return a if len(ha) == 0 else b
        sa = ha if len(ha) == n else self._rng.sample(ha, n)
        sb = hb if len(hb) == n else self._rng.sample(hb, n)
        ma, mb = sum(sa) / n, sum(sb) / n
        if ma > mb:
            return a
        if mb > ma:
            return b
        if len(ha) != len(hb):
            return a if len(ha) < len(hb) else b
        return a if self._rng.random() < 0.5 else b

    def _tournament(self, idx: Sequence[int]) -> int:
        if len(idx) == 1:
            return idx[0]
        mid = len(idx) // 2
        return self.duel(self._tournament(idx[:mid]), self._tournament(idx[mid:]))

    def select_arm(self, arms: Sequence[Arm], step: int) -> int:
        if not self._history:
            self.reset(arms)
        for i, h in enumerate(self._history):
            if not h:
                return i
        order = list(range(len(self._history)))
        self._rng.shuffle(order)  # random bracket removes positional bias
        return self._tournament(order)

    def update(self, arms: Sequence[Arm], step: BanditStep) -> None:
        if not self._history:
            self.reset(arms)
        self._history[step.arm_index].append(float(step.reward))


def besa(*, seed: int | None = None) -> BanditAlgorithm:
    """Factory for :class:`BESA`."""
    return BESA(seed=seed)
