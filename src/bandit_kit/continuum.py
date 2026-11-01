"""Continuum-armed (Lipschitz) bandits on the interval ``[0, 1]``.

Each round the learner picks a point ``x`` in ``[0, 1]`` and observes a noisy
reward with mean ``mu(x)``. The only structural assumption is that ``mu`` is
``L``-Lipschitz: ``|mu(x) - mu(y)| <= L |x - y|``. Regret is measured against
``mu* = sup_x mu(x)``.

Policies (pure Python, seeded through ``random.Random``):

* :class:`UniformDiscretizationUCB`: fixes ``K`` evenly spaced arms, by
  default ``K = ceil((L^2 T / log T)^(1/3))``, and runs UCB1 on them. This is
  the classic ``O~(T^(2/3))`` approach (Kleinberg, NIPS 2004).
* :class:`ZoomingAlgorithm`: Kleinberg, Slivkins & Upfal ("Multi-armed
  bandits in metric spaces", STOC 2008). Each active arm ``x`` owns a
  confidence ball of radius ``r(x) = sqrt(c log T / (1 + n(x)))`` in the metric
  ``L |x - y|``. When some point is no longer covered by any ball it is
  activated. Play goes to the arm with the largest ``mu_hat(x) + 2 r(x)``.
  Arms concentrate ("zoom in") near the optimum, so regret adapts to the
  zooming dimension of the instance instead of the worst case.
* :class:`UniformDiscretizationTS`: Thompson sampling on the same grid, with
  Beta posteriors for Bernoulli rewards and Gaussian posteriors otherwise.
* :class:`RandomContinuumPolicy`: uniform random points, as a baseline.

:func:`run_continuum` records the expected per-round regret
``mu* - mu(x_t)``. :func:`compare_continuum` averages over seeded runs.
:data:`CONTINUUM_PAYOFFS` provides named Lipschitz test functions together
with their Lipschitz constants.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "ContinuumEnvironment",
    "ContinuumPolicy",
    "UniformDiscretizationUCB",
    "UniformDiscretizationTS",
    "ZoomingAlgorithm",
    "RandomContinuumPolicy",
    "ContinuumResult",
    "run_continuum",
    "compare_continuum",
    "CONTINUUM_POLICIES",
    "CONTINUUM_PAYOFFS",
    "default_discretization",
]


def _triangle(x: float) -> float:
    return max(0.1, 0.9 - 2.0 * abs(x - 0.3))


def _two_peaks(x: float) -> float:
    return max(0.1, 0.6 - 3.0 * abs(x - 0.2), 0.8 - 3.0 * abs(x - 0.75))


def _smooth(x: float) -> float:
    return 0.5 + 0.4 * math.cos(2.0 * math.pi * (x - 0.6))


def _needle(x: float) -> float:
    return max(0.4, 0.9 - 10.0 * abs(x - 0.62))


#: name -> (mean function, Lipschitz constant)
CONTINUUM_PAYOFFS: Dict[str, Tuple[Callable[[float], float], float]] = {
    "triangle": (_triangle, 2.0),
    "two_peaks": (_two_peaks, 3.0),
    "smooth": (_smooth, 0.8 * math.pi),
    "needle": (_needle, 10.0),
}


class ContinuumEnvironment:
    """Noisy rewards with mean ``mu(x)`` on ``[0, 1]``.

    ``noise="bernoulli"`` draws ``Bernoulli(mu(x))`` (``mu`` must lie in
    ``[0, 1]``); ``noise="gaussian"`` adds ``N(0, sigma^2)``. ``mu*`` is taken
    from ``optimum`` if given, otherwise from a dense grid with ``grid`` points
    (exact for the piecewise-linear presets, whose peaks lie on that grid).
    """

    def __init__(
        self,
        mean_fn: Callable[[float], float],
        noise: str = "bernoulli",
        sigma: float = 0.1,
        seed: Optional[int] = None,
        optimum: Optional[float] = None,
        grid: int = 10001,
    ) -> None:
        if noise not in ("bernoulli", "gaussian"):
            raise ValueError("noise must be 'bernoulli' or 'gaussian'")
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        if grid < 2:
            raise ValueError("grid must be at least 2")
        self.mean_fn = mean_fn
        self.noise = noise
        self.sigma = float(sigma)
        self._rng = random.Random(seed)
        if optimum is None:
            best_x, best = 0.0, -math.inf
            for i in range(grid):
                x = i / (grid - 1)
                v = float(mean_fn(x))
                if v > best:
                    best_x, best = x, v
            self.optimum, self.argmax = best, best_x
        else:
            self.optimum, self.argmax = float(optimum), None

    def mean(self, x: float) -> float:
        if not 0.0 <= x <= 1.0:
            raise ValueError("arm must lie in [0, 1]")
        return float(self.mean_fn(x))

    def pull(self, x: float) -> float:
        mu = self.mean(x)
        if self.noise == "bernoulli":
            if not 0.0 <= mu <= 1.0:
                raise ValueError("Bernoulli noise needs mu(x) in [0, 1]")
            return 1.0 if self._rng.random() < mu else 0.0
        return mu + self._rng.gauss(0.0, self.sigma)


def default_discretization(horizon: int, lipschitz: float = 1.0) -> int:
    """``K = ceil((L^2 T / log T)^(1/3))``: the regret-optimal uniform grid size."""
    if horizon < 2:
        return 1
    return max(1, math.ceil((lipschitz ** 2 * horizon / math.log(horizon)) ** (1.0 / 3.0)))


class ContinuumPolicy:
    """Base class: ``reset(horizon)``, ``select() -> x``, ``update(x, reward)``."""

    name = "continuum"

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self._rng = random.Random(seed)
        self.horizon = 1
        self.t = 0

    def reset(self, horizon: int) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self.horizon = int(horizon)
        self.t = 0
        self._rng = random.Random(self.seed)

    def select(self) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def update(self, x: float, reward: float) -> None:
        self.t += 1

    @property
    def n_arms(self) -> int:
        return 0


class _GridPolicy(ContinuumPolicy):
    def __init__(self, k: Optional[int] = None, lipschitz: float = 1.0, seed: Optional[int] = None) -> None:
        super().__init__(seed)
        if k is not None and int(k) < 1:
            raise ValueError("k must be at least 1")
        if lipschitz <= 0:
            raise ValueError("lipschitz must be positive")
        self.k = None if k is None else int(k)
        self.lipschitz = float(lipschitz)

    def reset(self, horizon: int) -> None:
        super().reset(horizon)
        k = self.k if self.k is not None else default_discretization(self.horizon, self.lipschitz)
        self.arms = [(i + 0.5) / k for i in range(k)]
        self.counts = [0] * k
        self.sums = [0.0] * k
        self.sq = [0.0] * k
        self._index = {x: i for i, x in enumerate(self.arms)}

    @property
    def n_arms(self) -> int:
        return len(self.arms)

    def update(self, x: float, reward: float) -> None:
        super().update(x, reward)
        i = self._index[x]
        self.counts[i] += 1
        self.sums[i] += reward
        self.sq[i] += reward * reward


class UniformDiscretizationUCB(_GridPolicy):
    """UCB1 on ``K`` evenly spaced points (cell centres)."""

    name = "uniform_ucb"

    def select(self) -> float:
        for i, n in enumerate(self.counts):
            if n == 0:
                return self.arms[i]
        logt = math.log(max(2, self.t + 1))
        best = max(
            range(len(self.arms)),
            key=lambda i: self.sums[i] / self.counts[i] + math.sqrt(2.0 * logt / self.counts[i]),
        )
        return self.arms[best]


class UniformDiscretizationTS(_GridPolicy):
    """Thompson sampling on ``K`` evenly spaced points.

    Uses Beta(1 + s, 1 + n - s) posteriors when every reward so far was 0 or 1,
    and a Gaussian ``N(mean, 1 / (n + 1))`` posterior otherwise.
    """

    name = "uniform_ts"

    def reset(self, horizon: int) -> None:
        super().reset(horizon)
        self._binary = True

    def update(self, x: float, reward: float) -> None:
        if reward not in (0.0, 1.0):
            self._binary = False
        super().update(x, reward)

    def select(self) -> float:
        best, best_v = 0, -math.inf
        for i in range(len(self.arms)):
            n, s = self.counts[i], self.sums[i]
            if self._binary:
                v = self._rng.betavariate(1.0 + s, 1.0 + n - s)
            else:
                v = self._rng.gauss(s / n if n else 0.0, 1.0 / math.sqrt(n + 1.0))
            if v > best_v:
                best, best_v = i, v
        return self.arms[best]


class ZoomingAlgorithm(ContinuumPolicy):
    """The zooming algorithm of Kleinberg, Slivkins & Upfal (2008) on ``[0, 1]``.

    ``c`` scales the confidence radius ``sqrt(c log T / (1 + n))``. The paper's
    analysis uses ``c = 8``; the smaller default ``c = 2`` explores less and
    works better in practice.
    """

    name = "zooming"

    def __init__(self, lipschitz: float = 1.0, c: float = 2.0, seed: Optional[int] = None) -> None:
        super().__init__(seed)
        if lipschitz <= 0:
            raise ValueError("lipschitz must be positive")
        if c <= 0:
            raise ValueError("c must be positive")
        self.lipschitz = float(lipschitz)
        self.c = float(c)

    def reset(self, horizon: int) -> None:
        super().reset(horizon)
        self.arms: List[float] = []
        self.counts: List[int] = []
        self.sums: List[float] = []
        self._index: Dict[float, int] = {}
        self._logT = math.log(max(2, self.horizon))
        self.activations: List[int] = []  # round at which each arm was activated

    @property
    def n_arms(self) -> int:
        return len(self.arms)

    def radius(self, i: int) -> float:
        return math.sqrt(self.c * self._logT / (1.0 + self.counts[i]))

    def _uncovered_point(self) -> Optional[float]:
        """A point of ``[0, 1]`` outside every confidence ball, or ``None``."""
        if not self.arms:
            return 0.5
        balls = sorted(
            (x - self.radius(i) / self.lipschitz, x + self.radius(i) / self.lipschitz)
            for i, x in enumerate(self.arms)
        )
        reach = 0.0
        first = True
        for lo, hi in balls:
            if (first and lo > 0.0) or (not first and lo > reach):
                gap_lo = 0.0 if first else reach
                return 0.5 * (gap_lo + min(lo, 1.0)) if lo <= 1.0 else 0.5 * (gap_lo + 1.0)
            reach = max(reach, hi) if not first else hi
            first = False
            if reach >= 1.0:
                return None
        return 0.5 * (reach + 1.0)

    def select(self) -> float:
        x_new = self._uncovered_point()
        if x_new is not None and x_new not in self._index:
            self._index[x_new] = len(self.arms)
            self.arms.append(x_new)
            self.counts.append(0)
            self.sums.append(0.0)
            self.activations.append(self.t)
        best = max(
            range(len(self.arms)),
            key=lambda i: (self.sums[i] / self.counts[i] if self.counts[i] else 0.0) + 2.0 * self.radius(i),
        )
        return self.arms[best]

    def update(self, x: float, reward: float) -> None:
        super().update(x, reward)
        i = self._index[x]
        self.counts[i] += 1
        self.sums[i] += reward


class RandomContinuumPolicy(ContinuumPolicy):
    """Uniformly random points in ``[0, 1]``."""

    name = "random"

    def select(self) -> float:
        return self._rng.random()


CONTINUUM_POLICIES = {
    "zooming": ZoomingAlgorithm,
    "uniform_ucb": UniformDiscretizationUCB,
    "uniform_ts": UniformDiscretizationTS,
    "random": RandomContinuumPolicy,
}


@dataclass
class ContinuumResult:
    policy: str
    arms: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    regret: List[float] = field(default_factory=list)
    n_active_arms: int = 0

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

    def final_estimate(self, last: int = 100) -> float:
        """Median of the last ``last`` points played: the policy's guess of ``argmax mu``."""
        tail = sorted(self.arms[-last:])
        if not tail:
            raise ValueError("no rounds played")
        return tail[len(tail) // 2]


def run_continuum(policy: ContinuumPolicy, env: ContinuumEnvironment, horizon: int) -> ContinuumResult:
    """Play ``policy`` for ``horizon`` rounds and record expected regret."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    policy.reset(horizon)
    result = ContinuumResult(policy=policy.name)
    for _ in range(horizon):
        x = float(policy.select())
        r = env.pull(x)
        policy.update(x, r)
        result.arms.append(x)
        result.rewards.append(r)
        result.regret.append(env.optimum - env.mean(x))
    result.n_active_arms = policy.n_arms
    return result


def _make_policy(name: str, lipschitz: float, seed: int, k: Optional[int], c: float) -> ContinuumPolicy:
    if name == "zooming":
        return ZoomingAlgorithm(lipschitz=lipschitz, c=c, seed=seed)
    if name in ("uniform_ucb", "uniform_ts"):
        return CONTINUUM_POLICIES[name](k=k, lipschitz=lipschitz, seed=seed)
    return RandomContinuumPolicy(seed=seed)


def compare_continuum(
    payoff: str = "triangle",
    horizon: int = 3000,
    runs: int = 10,
    policies: Optional[Sequence[str]] = None,
    seed: int = 0,
    noise: str = "bernoulli",
    sigma: float = 0.1,
    lipschitz: Optional[float] = None,
    k: Optional[int] = None,
    c: float = 2.0,
) -> Dict[str, Dict[str, float]]:
    """Average total regret, active arms and final ``argmax`` error over runs.

    ``payoff`` names an entry of :data:`CONTINUUM_PAYOFFS`. ``lipschitz``
    defaults to that payoff's constant.
    """
    if payoff not in CONTINUUM_PAYOFFS:
        raise ValueError("unknown payoff %r; choose from %s" % (payoff, ", ".join(CONTINUUM_PAYOFFS)))
    if runs <= 0:
        raise ValueError("runs must be positive")
    fn, L = CONTINUUM_PAYOFFS[payoff]
    L = L if lipschitz is None else float(lipschitz)
    names = list(CONTINUUM_POLICIES) if policies is None else list(policies)
    unknown = [n for n in names if n not in CONTINUUM_POLICIES]
    if unknown:
        raise ValueError("unknown policies: " + ", ".join(unknown))
    table: Dict[str, Dict[str, float]] = {}
    for name in names:
        regrets, arms, errors = [], [], []
        for r in range(runs):
            env = ContinuumEnvironment(fn, noise=noise, sigma=sigma, seed=seed + 7919 * r)
            res = run_continuum(_make_policy(name, L, seed + r, k, c), env, horizon)
            regrets.append(res.total_regret)
            arms.append(res.n_active_arms)
            errors.append(abs(res.final_estimate() - env.argmax))
        table[name] = {
            "regret": sum(regrets) / runs,
            "arms": sum(arms) / runs,
            "argmax_error": sum(errors) / runs,
        }
    return table
