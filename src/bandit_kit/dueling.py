"""Dueling bandits: learning from pairwise preferences.

In many systems the only feedback is relative: a user clicks the result
of ranker A over ranker B in an interleaved list, or a rater prefers one
model response over another. A *dueling bandit* (Yue, Broder, Kleinberg &
Joachims 2012) picks a pair ``(i, j)`` each round and observes which one
wins, with ``P(i beats j) = P[i][j]``, ``P[i][j] + P[j][i] = 1``.

* The target is the **Condorcet winner** ``c``: the arm that beats every
  other with probability above ``1/2`` (:func:`condorcet_winner`). When it
  does not exist, :func:`copeland_winners` gives the arms beating the most
  rivals.
* With ``eps_i = P[c][i] - 1/2``, a duel ``(i, j)`` costs **strong regret**
  ``(eps_i + eps_j) / 2`` (both arms should be ``c``) and **weak regret**
  ``min(eps_i, eps_j)`` (one ``c`` suffices).

Policies (all pure Python, seeded through ``random.Random``):

* :class:`RUCB` — Relative Upper Confidence Bound (Zoghi, Whiteson, Munos &
  de Rijke, ICML 2014). Optimistic estimates
  ``U_ij = w_ij / n_ij + sqrt(alpha ln t / n_ij)`` (``1`` when unplayed)
  pick a candidate ``c`` with ``U_cj >= 1/2`` for all ``j`` and pit it
  against its strongest optimistic challenger ``argmax_j U_jc``.
* :class:`DoubleThompsonSampling` — D-TS (Wu & Liu, NeurIPS 2016). The
  first arm maximises sampled Copeland score among arms that could still
  be Copeland winners under the UCBs; the second is Thompson-sampled
  against the first among arms not confidently beaten by it.
* :class:`UniformDueling` — uniformly random pairs, as a baseline.

:func:`run_dueling` plays a policy on a :class:`PreferenceMatrix` and
:func:`compare_dueling` averages regret curves over seeded runs.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "PreferenceMatrix",
    "condorcet_winner",
    "copeland_winners",
    "DuelingPolicy",
    "RUCB",
    "DoubleThompsonSampling",
    "UniformDueling",
    "DuelingResult",
    "run_dueling",
    "compare_dueling",
]


class PreferenceMatrix:
    """A stochastic pairwise-preference environment.

    ``probs[i][j]`` is the probability that arm ``i`` wins a duel against
    ``j``. The matrix must satisfy ``P[i][j] + P[j][i] = 1`` with
    ``P[i][i] = 1/2``.
    """

    def __init__(self, probs: Sequence[Sequence[float]], seed: Optional[int] = None) -> None:
        k = len(probs)
        if k < 2:
            raise ValueError("need at least two arms")
        p = [[float(v) for v in row] for row in probs]
        for i, row in enumerate(p):
            if len(row) != k:
                raise ValueError("preference matrix must be square")
            for j, v in enumerate(row):
                if not 0.0 <= v <= 1.0:
                    raise ValueError("preferences must lie in [0, 1]")
                if abs(v + p[j][i] - 1.0) > 1e-9:
                    raise ValueError("P[i][j] + P[j][i] must equal 1")
        self.probs = p
        self.rng = random.Random(seed)

    @classmethod
    def from_utilities(
        cls,
        utilities: Sequence[float],
        link: str = "logistic",
        scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> "PreferenceMatrix":
        """Bradley-Terry (``logistic``) or ``linear`` preferences from utilities.

        ``logistic``: ``P_ij = 1 / (1 + exp(-(u_i - u_j) / scale))``;
        ``linear``: ``P_ij = clip(1/2 + (u_i - u_j) / (2 scale), 0, 1)``.
        """

        u = [float(x) for x in utilities]
        if scale <= 0:
            raise ValueError("scale must be positive")
        if link == "logistic":
            f = lambda d: 1.0 / (1.0 + math.exp(-d / scale))  # noqa: E731
        elif link == "linear":
            f = lambda d: min(1.0, max(0.0, 0.5 + d / (2.0 * scale)))  # noqa: E731
        else:
            raise ValueError("link must be 'logistic' or 'linear'")
        return cls([[f(ui - uj) for uj in u] for ui in u], seed=seed)

    @property
    def n_arms(self) -> int:
        return len(self.probs)

    def duel(self, i: int, j: int) -> bool:
        """Return ``True`` if arm ``i`` wins against ``j``."""
        return self.rng.random() < self.probs[i][j]

    def gaps(self) -> List[float]:
        """``eps_i = P[c][i] - 1/2`` relative to the Condorcet winner."""
        c = condorcet_winner(self.probs)
        if c is None:
            raise ValueError("regret needs a Condorcet winner; this matrix has none")
        return [self.probs[c][i] - 0.5 for i in range(self.n_arms)]


def condorcet_winner(probs: Sequence[Sequence[float]]) -> Optional[int]:
    """Index of the arm beating all others with probability > 1/2, else ``None``."""
    k = len(probs)
    for i in range(k):
        if all(probs[i][j] > 0.5 for j in range(k) if j != i):
            return i
    return None


def copeland_winners(probs: Sequence[Sequence[float]]) -> List[int]:
    """Arms that beat (``P > 1/2``) the largest number of rivals."""
    k = len(probs)
    scores = [sum(1 for j in range(k) if j != i and probs[i][j] > 0.5) for i in range(k)]
    top = max(scores)
    return [i for i, s in enumerate(scores) if s == top]


class DuelingPolicy:
    """Base class: keeps the win-count matrix ``wins[i][j]`` (i beat j)."""

    name = "dueling"

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.wins: List[List[int]] = []
        self.t = 0

    def reset(self, n_arms: int) -> None:
        self.rng = random.Random(self.seed)
        self.wins = [[0] * n_arms for _ in range(n_arms)]
        self.t = 0

    @property
    def n_arms(self) -> int:
        return len(self.wins)

    def select_pair(self) -> Tuple[int, int]:  # pragma: no cover - interface
        raise NotImplementedError

    def update(self, i: int, j: int, i_won: bool) -> None:
        self.t += 1
        if i == j:
            return
        if i_won:
            self.wins[i][j] += 1
        else:
            self.wins[j][i] += 1

    def _ucb(self, alpha: float) -> List[List[float]]:
        k = self.n_arms
        log_t = math.log(max(self.t, 1))
        u = [[0.5] * k for _ in range(k)]
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                n = self.wins[i][j] + self.wins[j][i]
                u[i][j] = 1.0 if n == 0 else self.wins[i][j] / n + math.sqrt(alpha * log_t / n)
        return u

    def _lcb(self, alpha: float) -> List[List[float]]:
        k = self.n_arms
        log_t = math.log(max(self.t, 1))
        lo = [[0.5] * k for _ in range(k)]
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                n = self.wins[i][j] + self.wins[j][i]
                lo[i][j] = 0.0 if n == 0 else self.wins[i][j] / n - math.sqrt(alpha * log_t / n)
        return lo

    def recommend(self) -> int:
        """Empirical Copeland winner (ties broken by overall win rate)."""
        k = self.n_arms
        best, best_key = 0, None
        for i in range(k):
            beats = sum(1 for j in range(k) if j != i and self.wins[i][j] > self.wins[j][i])
            w = sum(self.wins[i])
            n = w + sum(self.wins[j][i] for j in range(k))
            key = (beats, w / n if n else 0.0)
            if best_key is None or key > best_key:
                best, best_key = i, key
        return best


class RUCB(DuelingPolicy):
    """Relative Upper Confidence Bound (Zoghi et al., 2014)."""

    name = "rucb"

    def __init__(self, alpha: float = 0.51, seed: Optional[int] = None) -> None:
        if alpha <= 0.5:
            raise ValueError("alpha must exceed 1/2")
        super().__init__(seed)
        self.alpha = float(alpha)

    def select_pair(self) -> Tuple[int, int]:
        k = self.n_arms
        u = self._ucb(self.alpha)
        cands = [c for c in range(k) if all(u[c][j] >= 0.5 for j in range(k))]
        c = self.rng.choice(cands) if cands else self.rng.randrange(k)
        col = [u[j][c] for j in range(k)]
        top = max(col)
        d = self.rng.choice([j for j in range(k) if col[j] == top])
        return c, d


class DoubleThompsonSampling(DuelingPolicy):
    """Double Thompson Sampling (Wu & Liu, 2016)."""

    name = "dts"

    def __init__(self, alpha: float = 0.51, seed: Optional[int] = None) -> None:
        if alpha <= 0.5:
            raise ValueError("alpha must exceed 1/2")
        super().__init__(seed)
        self.alpha = float(alpha)

    def select_pair(self) -> Tuple[int, int]:
        k = self.n_arms
        u = self._ucb(self.alpha)
        lo = self._lcb(self.alpha)
        zeta = [sum(1 for j in range(k) if j != i and u[i][j] > 0.5) for i in range(k)]
        top = max(zeta)
        cands = [i for i in range(k) if zeta[i] == top]
        theta = [[0.5] * k for _ in range(k)]
        for i in range(k):
            for j in range(i + 1, k):
                x = self.rng.betavariate(self.wins[i][j] + 1, self.wins[j][i] + 1)
                theta[i][j], theta[j][i] = x, 1.0 - x
        scores = {i: sum(1 for j in range(k) if j != i and theta[i][j] > 0.5) for i in cands}
        best = max(scores.values())
        first = self.rng.choice([i for i in cands if scores[i] == best])
        sampled = {}
        for i in range(k):
            if i == first:
                sampled[i] = 0.5
            elif lo[i][first] <= 0.5:
                sampled[i] = self.rng.betavariate(self.wins[i][first] + 1, self.wins[first][i] + 1)
        top2 = max(sampled.values())
        second = self.rng.choice([i for i, v in sampled.items() if v == top2])
        return first, second


class UniformDueling(DuelingPolicy):
    """Uniformly random distinct pairs (exploration-only baseline)."""

    name = "uniform"

    def select_pair(self) -> Tuple[int, int]:
        i, j = self.rng.sample(range(self.n_arms), 2)
        return i, j


@dataclass
class DuelingResult:
    """Trajectory of one dueling-bandit run."""

    policy: str
    pairs: List[Tuple[int, int]] = field(default_factory=list)
    strong_regret: List[float] = field(default_factory=list)
    weak_regret: List[float] = field(default_factory=list)
    recommendation: int = -1
    condorcet_winner: Optional[int] = None

    @property
    def cumulative_strong_regret(self) -> float:
        return self.strong_regret[-1] if self.strong_regret else 0.0

    @property
    def cumulative_weak_regret(self) -> float:
        return self.weak_regret[-1] if self.weak_regret else 0.0


def run_dueling(policy: DuelingPolicy, env: PreferenceMatrix, horizon: int) -> DuelingResult:
    """Play ``horizon`` duels; returns cumulative strong/weak regret curves."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    gaps = env.gaps()
    policy.reset(env.n_arms)
    res = DuelingResult(policy=policy.name, condorcet_winner=condorcet_winner(env.probs))
    strong = weak = 0.0
    for _ in range(int(horizon)):
        i, j = policy.select_pair()
        policy.update(i, j, env.duel(i, j))
        strong += 0.5 * (gaps[i] + gaps[j])
        weak += min(gaps[i], gaps[j])
        res.pairs.append((i, j))
        res.strong_regret.append(strong)
        res.weak_regret.append(weak)
    res.recommendation = policy.recommend()
    return res


def compare_dueling(
    probs: Sequence[Sequence[float]],
    horizon: int = 1000,
    runs: int = 10,
    policies: Sequence[str] = ("rucb", "dts", "uniform"),
    alpha: float = 0.51,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Mean final strong/weak regret and winner-recovery rate per policy."""
    makers = {
        "rucb": lambda s: RUCB(alpha=alpha, seed=s),
        "dts": lambda s: DoubleThompsonSampling(alpha=alpha, seed=s),
        "uniform": lambda s: UniformDueling(seed=s),
    }
    out: Dict[str, Dict[str, float]] = {}
    for name in policies:
        if name not in makers:
            raise ValueError(f"unknown dueling policy: {name}")
        strong = weak = hits = 0.0
        for r in range(int(runs)):
            env = PreferenceMatrix(probs, seed=seed + 1000 * r + 1)
            res = run_dueling(makers[name](seed + r), env, horizon)
            strong += res.cumulative_strong_regret
            weak += res.cumulative_weak_regret
            hits += res.recommendation == res.condorcet_winner
        out[name] = {
            "strong_regret": strong / runs,
            "weak_regret": weak / runs,
            "winner_rate": hits / runs,
        }
    return out
