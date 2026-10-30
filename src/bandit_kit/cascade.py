"""Cascading bandits: learning to rank from clicks.

In the *cascade model* (Craswell et al. 2008) a user scans a list of ``K``
items from top to bottom and clicks the first one they find attractive.
Item ``e`` attracts with probability ``w(e)``, independently of the others.
A list ``A = (a_1, ..., a_K)`` gets a click with probability

    f(A, w) = 1 - prod_k (1 - w(a_k)),

and the optimal list holds the ``K`` most attractive items. One click gives
partial feedback. Items above the click position were examined and not
clicked (weight 0), the clicked item gets weight 1, and items below it were
never examined, so nothing is learned about them. With no click, all ``K``
items get weight 0.

Policies (Kveton, Szepesvári, Wen & Ashkan, ICML 2015; Cheung et al. 2019),
all pure Python and seeded through ``random.Random``:

* :class:`CascadeUCB1`: rank items by ``w_hat + sqrt(1.5 log t / n)``.
* :class:`CascadeKLUCB`: rank items by the largest ``q`` with
  ``n KL(w_hat, q) <= log t + 3 log log t``.
* :class:`CascadeTS`: rank items by a ``Beta(1 + clicks, 1 + skips)`` sample.
* :class:`CascadeEpsilonGreedy`: shows the empirical top-``K`` and swaps in
  random items with probability ``epsilon``.
* :class:`RandomRanker`: a uniformly random list, as a baseline.

Items that have never been observed rank first, so every item is examined
at least once.

:func:`run_cascade` plays a policy on a :class:`CascadeEnvironment` and
records the expected per-round regret ``f(A*, w) - f(A_t, w)``.
:func:`compare_cascade` averages over seeded runs.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .algorithms import _bernoulli_kl, _kl_ucb_threshold, _kl_ucb_upper

__all__ = [
    "CascadeEnvironment",
    "cascade_click_probability",
    "CascadePolicy",
    "CascadeUCB1",
    "CascadeKLUCB",
    "CascadeTS",
    "CascadeEpsilonGreedy",
    "RandomRanker",
    "CascadeResult",
    "run_cascade",
    "compare_cascade",
    "CASCADE_POLICIES",
]


def cascade_click_probability(attractions: Sequence[float], ranking: Sequence[int]) -> float:
    """``f(A, w) = 1 - prod_{e in A} (1 - w(e))``: the chance of any click."""
    miss = 1.0
    for e in ranking:
        miss *= 1.0 - attractions[e]
    return 1.0 - miss


class CascadeEnvironment:
    """A stochastic cascade-model click environment over ``L`` items."""

    def __init__(self, attractions: Sequence[float], k: int, seed: Optional[int] = None) -> None:
        w = [float(v) for v in attractions]
        if len(w) < 2:
            raise ValueError("need at least two items")
        if any(not 0.0 <= v <= 1.0 for v in w):
            raise ValueError("attraction probabilities must lie in [0, 1]")
        if not 1 <= int(k) <= len(w):
            raise ValueError("k must be between 1 and the number of items")
        self.attractions = w
        self.k = int(k)
        self.rng = random.Random(seed)

    @property
    def n_items(self) -> int:
        return len(self.attractions)

    def optimal_ranking(self) -> List[int]:
        order = sorted(range(self.n_items), key=lambda e: (-self.attractions[e], e))
        return order[: self.k]

    def optimal_value(self) -> float:
        return cascade_click_probability(self.attractions, self.optimal_ranking())

    def click(self, ranking: Sequence[int]) -> Optional[int]:
        """Simulate one user: return the 0-based click position or ``None``."""
        self._check_ranking(ranking)
        for pos, e in enumerate(ranking):
            if self.rng.random() < self.attractions[e]:
                return pos
        return None

    def _check_ranking(self, ranking: Sequence[int]) -> None:
        if len(ranking) != self.k:
            raise ValueError("ranking must contain exactly k items")
        if len(set(ranking)) != len(ranking):
            raise ValueError("ranking items must be distinct")
        if any(not 0 <= e < self.n_items for e in ranking):
            raise ValueError("ranking item out of range")


class CascadePolicy:
    """Base class: keeps per-item click counts and observation counts."""

    name = "cascade"

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.clicks: List[int] = []
        self.observations: List[int] = []
        self.k = 1
        self.t = 0

    def reset(self, n_items: int, k: int) -> None:
        self.rng = random.Random(self.seed)
        self.clicks = [0] * n_items
        self.observations = [0] * n_items
        self.k = int(k)
        self.t = 0

    @property
    def n_items(self) -> int:
        return len(self.clicks)

    def means(self) -> List[float]:
        return [c / n if n else 0.0 for c, n in zip(self.clicks, self.observations)]

    def scores(self) -> List[float]:  # pragma: no cover - interface
        raise NotImplementedError

    def select(self) -> List[int]:
        s = self.scores()
        tiebreak = [self.rng.random() for _ in s]
        order = sorted(range(self.n_items), key=lambda e: (-s[e], tiebreak[e]))
        return order[: self.k]

    def update(self, ranking: Sequence[int], click_position: Optional[int]) -> None:
        """Cascade feedback: items up to and including the click are observed."""
        self.t += 1
        last = len(ranking) - 1 if click_position is None else int(click_position)
        for pos in range(last + 1):
            e = ranking[pos]
            self.observations[e] += 1
            if click_position is not None and pos == click_position:
                self.clicks[e] += 1


class CascadeUCB1(CascadePolicy):
    name = "cascade_ucb1"

    def __init__(self, c: float = 1.5, seed: Optional[int] = None) -> None:
        if c <= 0:
            raise ValueError("c must be positive")
        super().__init__(seed)
        self.c = float(c)

    def scores(self) -> List[float]:
        log_t = math.log(max(self.t, 1))
        return [
            math.inf if n == 0 else c / n + math.sqrt(self.c * log_t / n)
            for c, n in zip(self.clicks, self.observations)
        ]


class CascadeKLUCB(CascadePolicy):
    name = "cascade_kl_ucb"

    def __init__(self, c: float = 3.0, precision: float = 1e-6, seed: Optional[int] = None) -> None:
        if c < 0:
            raise ValueError("c must be non-negative")
        super().__init__(seed)
        self.c = float(c)
        self.precision = float(precision)

    def scores(self) -> List[float]:
        threshold = _kl_ucb_threshold(max(self.t, 1), self.c)
        return [
            math.inf if n == 0 else _kl_ucb_upper(c / n, n, threshold, self.precision)
            for c, n in zip(self.clicks, self.observations)
        ]


class CascadeTS(CascadePolicy):
    name = "cascade_ts"

    def scores(self) -> List[float]:
        return [
            self.rng.betavariate(1 + c, 1 + n - c) for c, n in zip(self.clicks, self.observations)
        ]


class CascadeEpsilonGreedy(CascadePolicy):
    name = "cascade_epsilon_greedy"

    def __init__(self, epsilon: float = 0.1, seed: Optional[int] = None) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must lie in [0, 1]")
        super().__init__(seed)
        self.epsilon = float(epsilon)

    def scores(self) -> List[float]:
        return [math.inf if n == 0 else m for m, n in zip(self.means(), self.observations)]

    def select(self) -> List[int]:
        ranking = super().select()
        rest = [e for e in range(self.n_items) if e not in ranking]
        for pos in range(len(ranking)):
            if rest and self.rng.random() < self.epsilon:
                j = self.rng.randrange(len(rest))
                ranking[pos], rest[j] = rest[j], ranking[pos]
        return ranking


class RandomRanker(CascadePolicy):
    name = "random"

    def scores(self) -> List[float]:
        return [0.0] * self.n_items


CASCADE_POLICIES = {
    "cascade_ucb1": CascadeUCB1,
    "cascade_kl_ucb": CascadeKLUCB,
    "cascade_ts": CascadeTS,
    "cascade_epsilon_greedy": CascadeEpsilonGreedy,
    "random": RandomRanker,
}


@dataclass
class CascadeResult:
    policy: str
    rankings: List[List[int]] = field(default_factory=list)
    click_positions: List[Optional[int]] = field(default_factory=list)
    regret: List[float] = field(default_factory=list)

    @property
    def cumulative_regret(self) -> List[float]:
        out, total = [], 0.0
        for r in self.regret:
            total += r
            out.append(total)
        return out

    @property
    def total_regret(self) -> float:
        return float(sum(self.regret))

    @property
    def click_rate(self) -> float:
        n = len(self.click_positions)
        return sum(p is not None for p in self.click_positions) / n if n else 0.0


def run_cascade(policy: CascadePolicy, env: CascadeEnvironment, horizon: int) -> CascadeResult:
    """Play ``policy`` for ``horizon`` rounds and record expected regret."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    policy.reset(env.n_items, env.k)
    best = env.optimal_value()
    result = CascadeResult(policy=policy.name)
    for _ in range(horizon):
        ranking = list(policy.select())
        pos = env.click(ranking)
        policy.update(ranking, pos)
        result.rankings.append(ranking)
        result.click_positions.append(pos)
        result.regret.append(best - cascade_click_probability(env.attractions, ranking))
    return result


def compare_cascade(
    attractions: Sequence[float],
    k: int,
    horizon: int = 2000,
    runs: int = 10,
    policies: Optional[Sequence[str]] = None,
    seed: int = 0,
    epsilon: float = 0.1,
) -> Dict[str, Dict[str, float]]:
    """Average total regret, click rate and final top-``K`` recovery over runs.

    ``optimal_rate`` is the share of runs whose final ranking holds exactly
    the optimal item set (in any order, since order does not change ``f``).
    """
    if runs <= 0:
        raise ValueError("runs must be positive")
    names = list(CASCADE_POLICIES) if policies is None else list(policies)
    unknown = [n for n in names if n not in CASCADE_POLICIES]
    if unknown:
        raise ValueError("unknown policies: " + ", ".join(unknown))
    table: Dict[str, Dict[str, float]] = {}
    for name in names:
        regrets, rates, hits = [], [], 0
        for r in range(runs):
            env = CascadeEnvironment(attractions, k, seed=seed + 7919 * r)
            cls = CASCADE_POLICIES[name]
            pol = cls(epsilon=epsilon, seed=seed + r) if cls is CascadeEpsilonGreedy else cls(seed=seed + r)
            res = run_cascade(pol, env, horizon)
            regrets.append(res.total_regret)
            rates.append(res.click_rate)
            hits += set(res.rankings[-1]) == set(env.optimal_ranking())
        table[name] = {
            "regret": sum(regrets) / runs,
            "click_rate": sum(rates) / runs,
            "optimal_rate": hits / runs,
        }
    return table
