"""Best-arm identification (pure exploration).

Regret-minimising policies such as UCB1 or Thompson sampling spend their
pulls on arms they think are good. Best-arm identification has a different
aim: find the arm with the highest mean, and stop. A typical use is a
"which variant should we ship?" experiment. There are two settings:

* **Fixed budget.** Given ``budget`` pulls, recommend one arm so that the
  probability of picking a wrong one is small.

  - :func:`successive_rejects` (Audibert & Bubeck, COLT 2010) runs ``K - 1``
    phases of lengths ``n_k = ceil((n - K) / (logbar(K) (K + 1 - k)))``,
    with ``logbar(K) = 1/2 + sum_{i=2}^K 1/i``, and drops the empirically
    worst arm after each phase. It is parameter-free and has a near-optimal
    error rate.
  - :func:`successive_halving` (Karnin, Koren & Somekh, ICML 2013) runs
    ``ceil(log2 K)`` rounds. Each round splits its share of the budget evenly
    across the surviving arms, uses fresh samples, and keeps the top half.

* **Fixed confidence.** Stop as soon as the recommended arm is
  ``epsilon``-optimal with probability at least ``1 - delta``.

  - :func:`lucb` (LUCB1; Kalyanakrishnan, Tewari, Auer & Stone, ICML 2012)
    pulls the empirical leader and the strongest challenger by upper
    confidence bound. It stops once the leader's lower bound clears every
    challenger's upper bound minus ``epsilon``. The confidence radius is
    Hoeffding's, scaled to each arm's ``reward_range``.

  - :func:`top_two_thompson` (Russo, COLT 2016 / Operations Research 2020)
    samples from the posterior, plays the sampled best arm with probability
    ``beta`` and otherwise a *challenger*: either a resampled second best
    (TTTS) or the arm with the smallest transportation cost to the leader
    (T3C; Shang et al., AISTATS 2020). It stops with the Chernoff
    generalized-likelihood-ratio rule of Garivier & Kaufmann (COLT 2016):
    ``min_j Z_{leader,j}(t) > log(2 t (K - 1) / delta)`` (Bernoulli or
    Gaussian KL), and its sample complexity approaches the
    ``beta``-optimal lower bound.
  - :func:`lil_ucb` (Jamieson, Malloy, Nowak & Bubeck, COLT 2014) pulls the
    arm with the largest law-of-the-iterated-logarithm UCB and stops once one
    arm has received ``1 + lambda`` times as many pulls as all others
    combined.

:func:`identification_error_rate` estimates the probability that any of
these methods misidentifies the best arm by rerunning it on freshly seeded
copies of Bernoulli or Gaussian arms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

from .arms import Arm, BernoulliArm, GaussianArm, best_arm

__all__ = [
    "BestArmResult",
    "identification_error_rate",
    "chernoff_glr_statistic",
    "lil_ucb",
    "lucb",
    "top_two_thompson",
    "successive_halving",
    "successive_rejects",
    "successive_rejects_schedule",
]


@dataclass
class BestArmResult:
    """Outcome of one best-arm identification run."""

    method: str
    recommended: str
    total_pulls: int
    pulls: Dict[str, int]
    empirical_means: Dict[str, float]
    stopped: bool = True
    elimination_order: List[str] = field(default_factory=list)

    def is_correct(self, arms: Sequence[Arm]) -> bool:
        """True when the recommendation has the largest expected value."""
        best_value = best_arm(arms).expected_value
        by_name = {arm.name: arm for arm in arms}
        return by_name[self.recommended].expected_value >= best_value


def _check_arms(arms: Sequence[Arm]) -> List[Arm]:
    arms = list(arms)
    if len(arms) < 2:
        raise ValueError("best-arm identification needs at least two arms")
    names = [arm.name for arm in arms]
    if len(set(names)) != len(names):
        raise ValueError("arm names must be unique")
    return arms


def _check_budget(budget: int, k: int) -> int:
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < k:
        raise ValueError(f"budget must be an integer >= number of arms ({k})")
    return budget


class _Tally:
    def __init__(self, arms: Sequence[Arm]) -> None:
        self.arms = {arm.name: arm for arm in arms}
        self.n = {arm.name: 0 for arm in arms}
        self.s = {arm.name: 0.0 for arm in arms}

    def pull(self, name: str, times: int = 1) -> float:
        total = 0.0
        for _ in range(times):
            total += float(self.arms[name].draw())
        self.n[name] += times
        self.s[name] += total
        return total

    def mean(self, name: str) -> float:
        return self.s[name] / self.n[name] if self.n[name] else 0.0

    def means(self) -> Dict[str, float]:
        return {name: self.mean(name) for name in self.n}

    @property
    def total(self) -> int:
        return sum(self.n.values())


def successive_rejects_schedule(n_arms: int, budget: int) -> List[int]:
    """Cumulative per-arm pull counts ``n_1 <= ... <= n_{K-1}`` for Successive Rejects."""
    if n_arms < 2:
        raise ValueError("need at least two arms")
    _check_budget(budget, n_arms)
    logbar = 0.5 + sum(1.0 / i for i in range(2, n_arms + 1))
    return [
        math.ceil((budget - n_arms) / (logbar * (n_arms + 1 - k)))
        for k in range(1, n_arms)
    ]


def successive_rejects(arms: Sequence[Arm], budget: int) -> BestArmResult:
    """Fixed-budget Successive Rejects (Audibert & Bubeck, 2010)."""
    arms = _check_arms(arms)
    k = len(arms)
    schedule = successive_rejects_schedule(k, budget)
    tally = _Tally(arms)
    active = [arm.name for arm in arms]
    order: List[str] = []
    previous = 0
    for n_k in schedule:
        # Every arm gets at least one pull, even when budget == K.
        target = max(1, n_k)
        extra = target - previous
        if extra > 0:
            for name in active:
                tally.pull(name, extra)
        previous = max(previous, target)
        # Drop the empirically worst arm; ties go to the later arm.
        worst = min(reversed(active), key=tally.mean)
        active.remove(worst)
        order.append(worst)
    return BestArmResult(
        method="successive_rejects",
        recommended=active[0],
        total_pulls=tally.total,
        pulls=dict(tally.n),
        empirical_means=tally.means(),
        elimination_order=order,
    )


def successive_halving(arms: Sequence[Arm], budget: int) -> BestArmResult:
    """Fixed-budget Sequential Halving (Karnin, Koren & Somekh, 2013)."""
    arms = _check_arms(arms)
    k = len(arms)
    rounds = math.ceil(math.log2(k))
    _check_budget(budget, k * rounds)
    tally = _Tally(arms)
    active = [arm.name for arm in arms]
    order: List[str] = []
    for _ in range(rounds):
        if len(active) == 1:
            break
        per_arm = max(1, budget // (len(active) * rounds))
        round_means = {name: tally.pull(name, per_arm) / per_arm for name in active}
        keep = math.ceil(len(active) / 2)
        ranked = sorted(active, key=lambda name: (-round_means[name], active.index(name)))
        order.extend(reversed(ranked[keep:]))
        active = ranked[:keep]
    return BestArmResult(
        method="successive_halving",
        recommended=active[0],
        total_pulls=tally.total,
        pulls=dict(tally.n),
        empirical_means=tally.means(),
        elimination_order=order,
    )


def lucb(
    arms: Sequence[Arm],
    delta: float = 0.05,
    epsilon: float = 0.0,
    max_pulls: int = 100_000,
) -> BestArmResult:
    """Fixed-confidence LUCB1 (Kalyanakrishnan et al., 2012) for the single best arm.

    The confidence radius of arm ``a`` after ``u`` pulls, at round ``t``, is
    ``width_a * sqrt(log(5 K t^4 / (4 delta)) / (2 u))``, where ``width_a`` is
    the length of the arm's ``reward_range``. ``stopped`` is false when the
    run hits ``max_pulls`` before the stopping rule fires; in that case the
    current empirical leader is returned.
    """
    arms = _check_arms(arms)
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if epsilon < 0.0:
        raise ValueError("epsilon must be non-negative")
    k = len(arms)
    if isinstance(max_pulls, bool) or not isinstance(max_pulls, int) or max_pulls < k:
        raise ValueError("max_pulls must be an integer >= number of arms")
    width = {arm.name: float(arm.reward_range[1] - arm.reward_range[0]) for arm in arms}
    tally = _Tally(arms)
    names = [arm.name for arm in arms]
    for name in names:
        tally.pull(name)

    def radius(name: str, t: int) -> float:
        beta = math.log(5.0 * k * t ** 4 / (4.0 * delta))
        return width[name] * math.sqrt(beta / (2.0 * tally.n[name]))

    t = 1
    stopped = False
    while True:
        means = tally.means()
        leader = max(names, key=lambda n: (means[n], -names.index(n)))
        challengers = [n for n in names if n != leader]
        challenger = max(challengers, key=lambda n: means[n] + radius(n, t))
        if means[leader] - radius(leader, t) > means[challenger] + radius(challenger, t) - epsilon:
            stopped = True
            break
        if tally.total + 2 > max_pulls:
            break
        tally.pull(leader)
        tally.pull(challenger)
        t += 1
    means = tally.means()
    leader = max(names, key=lambda n: (means[n], -names.index(n)))
    return BestArmResult(
        method="lucb",
        recommended=leader,
        total_pulls=tally.total,
        pulls=dict(tally.n),
        empirical_means=means,
        stopped=stopped,
    )


def _kl_bernoulli(p: float, q: float, eps: float = 1e-15) -> float:
    p = min(max(p, eps), 1.0 - eps)
    q = min(max(q, eps), 1.0 - eps)
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def chernoff_glr_statistic(
    mean_a: float,
    n_a: int,
    mean_b: float,
    n_b: int,
    model: str = "bernoulli",
    sigma: float = 1.0,
) -> float:
    """Generalized likelihood ratio ``Z_{a,b}`` for "mean_a > mean_b" (Garivier & Kaufmann).

    With the pooled mean ``m = (n_a mu_a + n_b mu_b) / (n_a + n_b)``:
    ``Z = n_a d(mu_a, m) + n_b d(mu_b, m)`` if ``mu_a >= mu_b`` (else 0),
    where ``d`` is the Bernoulli KL (means in ``[0, 1]``) or the Gaussian KL
    ``(x - y)^2 / (2 sigma^2)``. Equivalently, for Gaussians,
    ``Z = (mu_a - mu_b)^2 / (2 sigma^2 (1/n_a + 1/n_b))``.
    """
    if n_a <= 0 or n_b <= 0:
        return 0.0
    if mean_a < mean_b:
        return 0.0
    if model == "gaussian":
        return (mean_a - mean_b) ** 2 / (2.0 * sigma ** 2 * (1.0 / n_a + 1.0 / n_b))
    if model != "bernoulli":
        raise ValueError("model must be 'bernoulli' or 'gaussian'")
    m = (n_a * mean_a + n_b * mean_b) / (n_a + n_b)
    return n_a * _kl_bernoulli(mean_a, m) + n_b * _kl_bernoulli(mean_b, m)


def top_two_thompson(
    arms: Sequence[Arm],
    delta: float = 0.05,
    beta: float = 0.5,
    model: str = "bernoulli",
    challenger: str = "t3c",
    sigma: float | None = None,
    max_pulls: int = 100_000,
    seed: int | None = None,
    max_resamples: int = 50,
) -> BestArmResult:
    """Fixed-confidence Top-Two Thompson Sampling with Chernoff GLR stopping.

    ``model="bernoulli"`` rescales rewards to ``[0, 1]`` with each arm's
    ``reward_range`` and uses ``Beta(1 + S, 1 + N - S)`` posteriors (exact
    for 0/1 rewards; the binomial KL is still a valid sub-Bernoulli bound
    for bounded rewards). ``model="gaussian"`` uses a flat-prior
    ``N(mean, sigma^2 / n)`` posterior with known ``sigma`` (default: half
    the widest ``reward_range``, or 1 for unbounded arms).

    ``challenger="ttts"`` resamples the posterior until a different arm is
    the argmax (at most ``max_resamples`` times, falling back to T3C);
    ``challenger="t3c"`` picks ``argmin_{j != I} Z_{I,j}``.
    """
    import random as _random

    arms = _check_arms(arms)
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must be in (0, 1)")
    if model not in ("bernoulli", "gaussian"):
        raise ValueError("model must be 'bernoulli' or 'gaussian'")
    if challenger not in ("t3c", "ttts"):
        raise ValueError("challenger must be 't3c' or 'ttts'")
    k = len(arms)
    if isinstance(max_pulls, bool) or not isinstance(max_pulls, int) or max_pulls < k:
        raise ValueError("max_pulls must be an integer >= number of arms")
    rng = _random.Random(seed)
    names = [arm.name for arm in arms]
    lo = {arm.name: float(arm.reward_range[0]) for arm in arms}
    width = {arm.name: float(arm.reward_range[1] - arm.reward_range[0]) for arm in arms}
    if model == "bernoulli":
        for arm in arms:
            if not math.isfinite(width[arm.name]) or width[arm.name] <= 0:
                raise ValueError("model='bernoulli' needs arms with a finite, non-degenerate reward_range")
    if sigma is None:
        finite = [w for w in width.values() if math.isfinite(w) and w > 0]
        sigma = max(finite) / 2.0 if finite else 1.0
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    tally = _Tally(arms)
    succ = {name: 0.0 for name in names}  # rescaled reward sums (bernoulli)

    def pull(name: str) -> None:
        r = tally.pull(name)
        if model == "bernoulli":
            succ[name] += min(max((r - lo[name]) / width[name], 0.0), 1.0)

    def stat_mean(name: str) -> float:
        if model == "bernoulli":
            return succ[name] / tally.n[name]
        return tally.mean(name)

    def sample(name: str) -> float:
        if model == "bernoulli":
            return rng.betavariate(1.0 + succ[name], 1.0 + tally.n[name] - succ[name])
        return rng.gauss(tally.mean(name), sigma / math.sqrt(tally.n[name]))

    def z(a: str, b: str) -> float:
        return chernoff_glr_statistic(stat_mean(a), tally.n[a], stat_mean(b), tally.n[b], model, sigma)

    for name in names:
        pull(name)
    stopped = False
    while True:
        t = tally.total
        means = {n: stat_mean(n) for n in names}
        leader = max(names, key=lambda n: (means[n], -names.index(n)))
        threshold = math.log(2.0 * t * (k - 1) / delta)
        if min(z(leader, j) for j in names if j != leader) > threshold:
            stopped = True
            break
        if t + 1 > max_pulls:
            break
        draws = {n: sample(n) for n in names}
        first = max(names, key=lambda n: draws[n])
        if rng.random() < beta:
            pull(first)
            continue
        second = None
        if challenger == "ttts":
            for _ in range(max_resamples):
                again = {n: sample(n) for n in names}
                cand = max(names, key=lambda n: again[n])
                if cand != first:
                    second = cand
                    break
        if second is None:
            second = min((j for j in names if j != first),
                         key=lambda j: (z(first, j), names.index(j)))
        pull(second)
    means = tally.means()
    stat = {n: stat_mean(n) for n in names}
    leader = max(names, key=lambda n: (stat[n], -names.index(n)))
    return BestArmResult(
        method=f"top_two_thompson[{challenger}]",
        recommended=leader,
        total_pulls=tally.total,
        pulls=dict(tally.n),
        empirical_means=means,
        stopped=stopped,
    )


def lil_ucb(
    arms: Sequence[Arm],
    delta: float = 0.05,
    epsilon: float = 0.01,
    beta: float = 1.0,
    lam: float | None = None,
    sigma: float | None = None,
    max_pulls: int = 1_000_000,
) -> BestArmResult:
    """Fixed-confidence lil'UCB (Jamieson et al., 2014).

    Pulls ``argmax_i mu_i + (1 + beta)(1 + sqrt(epsilon)) *
    sqrt(2 sigma^2 (1 + epsilon) log(log((1 + epsilon) n_i + 2) / delta) / n_i)``
    (the ``+ 2`` keeps the iterated logarithm positive for small ``n_i``, as in
    the authors' experiments)
    and stops when some arm satisfies ``n_i >= 1 + lam * sum_{j != i} n_j``;
    that arm is recommended. ``lam`` defaults to the theoretical
    ``((2 + beta) / beta)^2`` (9 for ``beta = 1``). ``sigma`` is the
    sub-Gaussian scale (default: half the widest ``reward_range``). The
    paper's guarantee holds for ``delta < log(1 + epsilon) / e`` after the
    ``delta`` rescaling of its Theorem 2; larger ``delta`` values give the
    widely used heuristic variant.
    """
    arms = _check_arms(arms)
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be in (0, 1)")
    if beta <= 0:
        raise ValueError("beta must be positive")
    k = len(arms)
    if isinstance(max_pulls, bool) or not isinstance(max_pulls, int) or max_pulls < k:
        raise ValueError("max_pulls must be an integer >= number of arms")
    if lam is None:
        lam = ((2.0 + beta) / beta) ** 2
    if lam <= 0:
        raise ValueError("lam must be positive")
    if sigma is None:
        finite = [float(a.reward_range[1] - a.reward_range[0]) for a in arms]
        finite = [w for w in finite if math.isfinite(w) and w > 0]
        sigma = max(finite) / 2.0 if finite else 1.0
    names = [arm.name for arm in arms]
    tally = _Tally(arms)
    for name in names:
        tally.pull(name)
    scale = (1.0 + beta) * (1.0 + math.sqrt(epsilon))

    def index(name: str) -> float:
        n = tally.n[name]
        inner = math.log((1.0 + epsilon) * n + 2.0)
        bonus = math.sqrt(2.0 * sigma ** 2 * (1.0 + epsilon) * math.log(inner / delta) / n)
        return tally.mean(name) + scale * bonus

    stopped = False
    winner = None
    while True:
        total = tally.total
        for name in names:
            if tally.n[name] >= 1.0 + lam * (total - tally.n[name]):
                winner = name
                stopped = True
                break
        if stopped or total + 1 > max_pulls:
            break
        tally.pull(max(names, key=lambda n: (index(n), -names.index(n))))
    if winner is None:
        winner = max(names, key=lambda n: (tally.n[n], tally.mean(n)))
    return BestArmResult(
        method="lil_ucb",
        recommended=winner,
        total_pulls=tally.total,
        pulls=dict(tally.n),
        empirical_means=tally.means(),
        stopped=stopped,
    )


def _reseed(arm: Arm, seed: int) -> Arm:
    if isinstance(arm, BernoulliArm):
        return BernoulliArm(arm.name, arm.p, seed=seed)
    if isinstance(arm, GaussianArm):
        return GaussianArm(arm.name, arm.mean, arm.std, seed=seed)
    raise TypeError("identification_error_rate supports BernoulliArm and GaussianArm only")


def identification_error_rate(
    method: Callable[..., BestArmResult],
    arms: Sequence[Arm],
    n_runs: int = 200,
    seed: int = 0,
    **kwargs,
) -> Dict[str, float]:
    """Monte Carlo misidentification rate of ``method`` on reseeded copies of ``arms``.

    Returns ``{"error_rate", "mean_pulls", "n_runs"}``.
    """
    arms = _check_arms(arms)
    if isinstance(n_runs, bool) or not isinstance(n_runs, int) or n_runs < 1:
        raise ValueError("n_runs must be a positive integer")
    errors = 0
    pulls = 0
    for run in range(n_runs):
        fresh = [_reseed(arm, seed + run * len(arms) + i) for i, arm in enumerate(arms)]
        result = method(fresh, **kwargs)
        errors += 0 if result.is_correct(fresh) else 1
        pulls += result.total_pulls
    return {"error_rate": errors / n_runs, "mean_pulls": pulls / n_runs, "n_runs": float(n_runs)}
