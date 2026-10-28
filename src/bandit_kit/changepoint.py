"""Change-point-detection bandits for piecewise-stationary environments.

Passive non-stationary policies (:class:`~bandit_kit.algorithms.SlidingWindowUCB`,
:class:`~bandit_kit.algorithms.DiscountedUCB`) forget at a fixed rate. In a
*piecewise-stationary* problem the arm means are constant between a few
unknown breakpoints, and *actively adaptive* policies that run a
change-point detector and restart when it fires are typically better:

* :class:`CUSUMUCB` -- CUSUM-UCB (Liu, Lee & Shroff, AAAI 2018). Each arm
  runs a two-sided CUSUM test (Page 1954) on its own rewards: the first
  ``m`` samples since the arm's last reset estimate its mean ``u``; then
  ``g+ = max(0, g+ + y - u - eps)``, ``g- = max(0, g- + u - y - eps)`` and an
  alarm when either exceeds ``h`` resets that arm only. Arms are chosen by
  UCB on post-reset statistics, with uniform exploration w.p. ``alpha``.
* :class:`MUCB` -- Monitored UCB (Cao, Wen, Kveton & Xie, AISTATS 2019).
  Forced round-robin exploration takes a ``gamma`` share of rounds; every
  arm keeps its last ``w`` rewards and a change is declared when
  ``|sum(second half) - sum(first half)| > b``, which restarts *all* arms.
  ``b`` defaults to ``sqrt(w/2 * log(2 K T^2))`` from the paper.

The environment :class:`PiecewiseBernoulli` holds a ``(segments x arms)``
mean table and the breakpoints. :func:`run_piecewise` plays any
:class:`~bandit_kit.algorithms.BanditAlgorithm` (so SW-UCB / D-UCB / UCB1
can be compared on equal footing) and returns *dynamic* regret against the
best arm of the current segment; :func:`compare_piecewise` averages over
seeded runs and reports detection delay for the active policies.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

from .algorithms import BanditAlgorithm, BanditStep, DiscountedUCB, SlidingWindowUCB, UCB1
from .arms import Arm, BernoulliArm

__all__ = [
    "PiecewiseBernoulli",
    "CUSUMUCB",
    "MUCB",
    "PiecewiseResult",
    "run_piecewise",
    "compare_piecewise",
    "mucb_threshold",
]


class PiecewiseBernoulli:
    """Bernoulli arms whose means jump at fixed breakpoints.

    ``means[s][k]`` is the success probability of arm ``k`` in segment
    ``s``; segment ``s`` covers steps ``[breakpoints[s-1], breakpoints[s])``
    (``breakpoints`` has ``len(means) - 1`` strictly increasing positive
    entries).
    """

    def __init__(
        self,
        means: Sequence[Sequence[float]],
        breakpoints: Sequence[int],
        *,
        seed: int | None = None,
    ) -> None:
        table = [[float(p) for p in row] for row in means]
        if not table or not table[0]:
            raise ValueError("means must be a non-empty table")
        k = len(table[0])
        if any(len(row) != k for row in table):
            raise ValueError("every segment must list the same number of arms")
        if any(not 0.0 <= p <= 1.0 for row in table for p in row):
            raise ValueError("Bernoulli means must be in [0, 1]")
        bps = [int(b) for b in breakpoints]
        if len(bps) != len(table) - 1:
            raise ValueError("need exactly len(means) - 1 breakpoints")
        if any(b <= 0 for b in bps) or any(b2 <= b1 for b1, b2 in zip(bps, bps[1:])):
            raise ValueError("breakpoints must be positive and strictly increasing")
        self.means = table
        self.breakpoints = bps
        self.n_arms = k
        self._rng = random.Random(seed)

    def segment(self, t: int) -> int:
        """Index of the segment containing step ``t`` (0-based)."""
        s = 0
        while s < len(self.breakpoints) and t >= self.breakpoints[s]:
            s += 1
        return s

    def mean(self, arm: int, t: int) -> float:
        return self.means[self.segment(t)][arm]

    def best_mean(self, t: int) -> float:
        return max(self.means[self.segment(t)])

    def pull(self, arm: int, t: int) -> float:
        return 1.0 if self._rng.random() < self.mean(arm, t) else 0.0

    def arms(self) -> List[Arm]:
        """Placeholder :class:`Arm` objects (segment-0 means) for policy ``reset``."""
        return [BernoulliArm(f"arm{k}", self.means[0][k]) for k in range(self.n_arms)]


class CUSUMUCB(BanditAlgorithm):
    """CUSUM-UCB: per-arm two-sided CUSUM change detection + UCB."""

    name = "cusum_ucb"

    def __init__(
        self,
        m: int = 20,
        eps: float = 0.05,
        h: float = 8.0,
        alpha: float = 0.01,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        if isinstance(m, bool) or not isinstance(m, int) or m < 1:
            raise ValueError("m must be an integer >= 1")
        if eps < 0.0:
            raise ValueError("eps must be non-negative")
        if h <= 0.0:
            raise ValueError("h must be positive")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.m, self.eps, self.h, self.alpha = int(m), float(eps), float(h), float(alpha)
        self.alarms: List[int] = []
        self._n_arms = 0

    def reset(self, arms: Sequence[Arm]) -> None:
        k = len(arms)
        self._n_arms = k
        self._counts = [0] * k
        self._sums = [0.0] * k
        self._ref = [0.0] * k
        self._gpos = [0.0] * k
        self._gneg = [0.0] * k
        self.alarms = []
        self._t = 0

    def _reset_arm(self, idx: int) -> None:
        self._counts[idx] = 0
        self._sums[idx] = 0.0
        self._ref[idx] = 0.0
        self._gpos[idx] = 0.0
        self._gneg[idx] = 0.0

    def select_arm(self, arms: Sequence[Arm], step: int) -> int:
        if self._n_arms != len(arms):
            self.reset(arms)
        for idx, c in enumerate(self._counts):
            if c == 0:
                return idx
        if self._rng.random() < self.alpha:
            return self._rng.randrange(len(arms))
        n_total = sum(self._counts)
        log_term = 2.0 * math.log(max(n_total, 2))
        scores = [
            self._sums[i] / self._counts[i] + math.sqrt(log_term / self._counts[i])
            for i in range(len(arms))
        ]
        best = max(scores)
        return scores.index(best)

    def update(self, arms: Sequence[Arm], step: BanditStep) -> None:
        if self._n_arms != len(arms):
            self.reset(arms)
        idx, y = step.arm_index, float(step.reward)
        self._counts[idx] += 1
        self._sums[idx] += y
        n = self._counts[idx]
        if n <= self.m:
            self._ref[idx] = self._sums[idx] / n
            return
        u = self._ref[idx]
        self._gpos[idx] = max(0.0, self._gpos[idx] + y - u - self.eps)
        self._gneg[idx] = max(0.0, self._gneg[idx] + u - y - self.eps)
        if self._gpos[idx] > self.h or self._gneg[idx] > self.h:
            self.alarms.append(int(step.step))
            self._reset_arm(idx)


def mucb_threshold(w: int, n_arms: int, horizon: int) -> float:
    """Default M-UCB detection threshold ``sqrt(w/2 * log(2 K T^2))``."""
    if w < 2 or n_arms < 1 or horizon < 1:
        raise ValueError("need w >= 2, n_arms >= 1 and horizon >= 1")
    return math.sqrt(w / 2.0 * math.log(2.0 * n_arms * horizon ** 2))


class MUCB(BanditAlgorithm):
    """Monitored UCB: windowed mean-shift test with global restarts."""

    name = "m_ucb"

    def __init__(
        self,
        w: int = 100,
        b: float | None = None,
        gamma: float = 0.05,
        horizon: int = 10000,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        if isinstance(w, bool) or not isinstance(w, int) or w < 2 or w % 2:
            raise ValueError("w must be an even integer >= 2")
        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must be in [0, 1)")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be an integer >= 1")
        if b is not None and b <= 0.0:
            raise ValueError("b must be positive")
        self.w, self.gamma, self.horizon = int(w), float(gamma), int(horizon)
        self._b_given = b
        self.b = float(b) if b is not None else 0.0
        self.alarms: List[int] = []
        self._n_arms = 0

    def reset(self, arms: Sequence[Arm]) -> None:
        k = len(arms)
        self._n_arms = k
        self.b = float(self._b_given) if self._b_given is not None else mucb_threshold(self.w, k, self.horizon)
        self._tau = 0
        self._restart(k)
        self.alarms = []

    def _restart(self, k: int) -> None:
        self._counts = [0] * k
        self._sums = [0.0] * k
        self._buffers: List[List[float]] = [[] for _ in range(k)]

    def select_arm(self, arms: Sequence[Arm], step: int) -> int:
        k = len(arms)
        if self._n_arms != k:
            self.reset(arms)
        if self.gamma > 0.0:
            period = max(k, int(math.floor(k / self.gamma)))
            a = (step - self._tau) % period
            if a < k:
                return a
        for idx, c in enumerate(self._counts):
            if c == 0:
                return idx
        n_total = sum(self._counts)
        log_term = 2.0 * math.log(max(n_total, 2))
        scores = [
            self._sums[i] / self._counts[i] + math.sqrt(log_term / self._counts[i])
            for i in range(k)
        ]
        return scores.index(max(scores))

    def update(self, arms: Sequence[Arm], step: BanditStep) -> None:
        if self._n_arms != len(arms):
            self.reset(arms)
        idx, y = step.arm_index, float(step.reward)
        self._counts[idx] += 1
        self._sums[idx] += y
        buf = self._buffers[idx]
        buf.append(y)
        if len(buf) > self.w:
            del buf[0]
        if len(buf) == self.w:
            half = self.w // 2
            if abs(sum(buf[half:]) - sum(buf[:half])) > self.b:
                self.alarms.append(int(step.step))
                self._tau = int(step.step) + 1
                self._restart(len(arms))


@dataclass
class PiecewiseResult:
    """Outcome of one piecewise-stationary run."""

    policy: str
    choices: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dynamic_regret: List[float] = field(default_factory=list)
    alarms: List[int] = field(default_factory=list)

    @property
    def cumulative_regret(self) -> float:
        return self.dynamic_regret[-1] if self.dynamic_regret else 0.0

    def detection_delays(self, breakpoints: Sequence[int]) -> List[float]:
        """Delay from each breakpoint to the first later alarm (``inf`` if missed)."""
        out = []
        bps = list(breakpoints)
        for i, bp in enumerate(bps):
            nxt = bps[i + 1] if i + 1 < len(bps) else math.inf
            hits = [a for a in self.alarms if bp <= a < nxt]
            out.append(float(hits[0] - bp) if hits else math.inf)
        return out


def run_piecewise(policy: BanditAlgorithm, env: PiecewiseBernoulli, horizon: int) -> PiecewiseResult:
    """Play ``horizon`` rounds of ``policy`` on ``env``; dynamic regret curve."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    arms = env.arms()
    policy.reset(arms)
    res = PiecewiseResult(policy=getattr(policy, "name", type(policy).__name__))
    regret = 0.0
    for t in range(int(horizon)):
        a = policy.select_arm(arms, t)
        y = env.pull(a, t)
        policy.update(arms, BanditStep(arm_index=a, arm_name=arms[a].name, reward=y, step=t))
        regret += env.best_mean(t) - env.mean(a, t)
        res.choices.append(a)
        res.rewards.append(y)
        res.dynamic_regret.append(regret)
    res.alarms = list(getattr(policy, "alarms", []))
    return res


_POLICIES: Dict[str, Callable[..., BanditAlgorithm]] = {
    "cusum_ucb": lambda seed, horizon, **kw: CUSUMUCB(
        m=kw.get("m", 20), eps=kw.get("eps", 0.05), h=kw.get("h", 8.0), alpha=kw.get("alpha", 0.01), seed=seed
    ),
    "m_ucb": lambda seed, horizon, **kw: MUCB(
        w=kw.get("w", 100), b=kw.get("b"), gamma=kw.get("gamma", 0.05), horizon=horizon, seed=seed
    ),
    "sliding_window_ucb": lambda seed, horizon, **kw: SlidingWindowUCB(window=kw.get("window", 200), seed=seed),
    "discounted_ucb": lambda seed, horizon, **kw: DiscountedUCB(gamma=kw.get("discount", 0.99), seed=seed),
    "ucb1": lambda seed, horizon, **kw: UCB1(seed=seed),
}


def compare_piecewise(
    means: Sequence[Sequence[float]],
    breakpoints: Sequence[int],
    horizon: int = 3000,
    runs: int = 10,
    policies: Sequence[str] = ("cusum_ucb", "m_ucb", "sliding_window_ucb", "discounted_ucb", "ucb1"),
    seed: int = 0,
    **params,
) -> Dict[str, Dict[str, float]]:
    """Mean final dynamic regret, alarm count and detection stats per policy.

    ``params`` are forwarded to the policy builders (``m``, ``eps``, ``h``,
    ``alpha`` for CUSUM-UCB; ``w``, ``b``, ``gamma`` for M-UCB; ``window``
    for SW-UCB; ``discount`` for D-UCB).
    """
    if runs < 1:
        raise ValueError("runs must be positive")
    out: Dict[str, Dict[str, float]] = {}
    for name in policies:
        if name not in _POLICIES:
            raise ValueError(f"unknown piecewise policy: {name}")
        regret = alarms = detected = delay_sum = 0.0
        n_bp = 0
        for r in range(int(runs)):
            env = PiecewiseBernoulli(means, breakpoints, seed=seed + 1000 * r + 1)
            res = run_piecewise(_POLICIES[name](seed + r, int(horizon), **params), env, horizon)
            regret += res.cumulative_regret
            alarms += len(res.alarms)
            for bp, d in zip(env.breakpoints, res.detection_delays(env.breakpoints)):
                if bp >= horizon:
                    continue
                n_bp += 1
                if math.isfinite(d):
                    detected += 1
                    delay_sum += d
        out[name] = {
            "dynamic_regret": regret / runs,
            "alarms": alarms / runs,
            "detection_rate": detected / n_bp if n_bp else 0.0,
            "mean_delay": delay_sum / detected if detected else math.nan,
        }
    return out
