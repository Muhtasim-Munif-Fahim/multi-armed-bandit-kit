"""Run-and-evaluate harness for multi-armed bandit experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

from .algorithms import (
    BanditAlgorithm,
    BanditStep,
    LinUCB,
    bayesian_ucb,
    decaying_epsilon_greedy,
    epsilon_greedy,
    exp3,
    gradient_bandit,
    linucb,
    thompson_sampling_bernoulli,
    ucb1,
)
from .arms import (
    Arm,
    BernoulliArm,
    GaussianArm,
    LinearContextualArm,
    expected_payoffs,
    sample_context,
)


_REGISTRY: Dict[str, Callable[..., BanditAlgorithm]] = {
    "epsilon_greedy": epsilon_greedy,
    "ucb1": ucb1,
    "thompson": thompson_sampling_bernoulli,
    "bayesian_ucb": bayesian_ucb,
    "decaying_epsilon_greedy": decaying_epsilon_greedy,
    "gradient_bandit": gradient_bandit,
    "exp3": exp3,
    "linucb": linucb,
}


def available_algorithms() -> List[str]:
    """Return the canonical short names of the bundled algorithms."""
    return sorted(_REGISTRY)


@dataclass
class BanditRunResult:
    """One independent run of one algorithm on a fixed arm set."""

    algorithm: str
    seed: int
    steps: List[BanditStep] = field(default_factory=list)

    @property
    def rewards(self) -> List[float]:
        return [step.reward for step in self.steps]


@dataclass
class BanditExperiment:
    """A reusable experiment harness for a fixed arm set.

    An experiment bundles the arm list, the list of algorithms to compare,
    the number of steps and the number of independent runs, and a base
    random seed (each algorithm+run pair derives its own seed as
    ``seed * 7919 + index``). The :meth:`run` method is deterministic
    given the seed; :meth:`summarize` aggregates results into the format
    consumed by the reporting layer.
    """

    arms: List[Arm]
    algorithms: List[str]
    steps: int = 200
    runs: int = 20
    seed: int = 42
    epsilon: float = 0.1
    gamma: float = 0.1
    linucb_alpha: float = 1.0
    linucb_ridge: float = 1.0

    def __post_init__(self) -> None:
        if not self.arms:
            raise ValueError("at least one arm is required")
        if not self.algorithms:
            raise ValueError("at least one algorithm is required")
        for algo in self.algorithms:
            if algo not in _REGISTRY:
                raise ValueError(f"unknown algorithm: {algo}")
        if self.steps < 1:
            raise ValueError("steps must be at least 1")
        if self.runs < 1:
            raise ValueError("runs must be at least 1")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if self.linucb_alpha < 0.0:
            raise ValueError("linucb_alpha must be non-negative")
        if self.linucb_ridge <= 0.0:
            raise ValueError("linucb_ridge must be positive")

    def _make_algorithm(self, name: str, seed: int) -> BanditAlgorithm:
        factory = _REGISTRY[name]
        if name == "epsilon_greedy":
            return factory(epsilon=self.epsilon, seed=seed)
        if name == "exp3":
            return factory(gamma=self.gamma, seed=seed)
        if name == "linucb":
            # Stationary path: intercept-only ridge-UCB (context = [1.0]).
            return factory(
                alpha=self.linucb_alpha,
                dimension=1,
                ridge=self.linucb_ridge,
                seed=seed,
            )
        return factory(seed=seed)

    def _seeded_arms(self, seed: int) -> List[Arm]:
        """Build a fresh arm set whose draws share ``seed``."""
        seeded: List[Arm] = []
        for offset, arm in enumerate(self.arms):
            child_seed = (seed * 31 + offset) & 0xFFFFFFFF
            if isinstance(arm, BernoulliArm):
                seeded.append(
                    BernoulliArm(name=arm.name, p=arm.p, seed=child_seed)
                )
            elif isinstance(arm, GaussianArm):
                seeded.append(
                    GaussianArm(name=arm.name, mean=arm.mean, std=arm.std, seed=child_seed)
                )
            else:
                # Fallback: keep the original draw; non-deterministic arms
                # are still legal but flagged in the report.
                seeded.append(arm)
        return seeded

    def run(self) -> List[BanditRunResult]:
        """Run every algorithm ``runs`` times and return a flat list of results."""
        results: List[BanditRunResult] = []
        for algo_index, algo_name in enumerate(self.algorithms):
            for run_index in range(self.runs):
                seed = self.seed * 7919 + algo_index * 1009 + run_index
                results.append(self._run_single(algo_name, seed))
        return results

    def _run_single(self, algo_name: str, seed: int) -> BanditRunResult:
        algo = self._make_algorithm(algo_name, seed)
        arms = self._seeded_arms(seed)
        algo.reset(arms)
        result = BanditRunResult(algorithm=algo_name, seed=seed)
        for step_index in range(self.steps):
            arm_index = algo.select_arm(arms, step_index)
            arm = arms[arm_index]
            reward = arm.draw()
            step = BanditStep(
                arm_index=arm_index,
                arm_name=arm.name,
                reward=float(reward),
                step=step_index,
            )
            result.steps.append(step)
            algo.update(arms, step)
        return result

    def summarize(self, results: Sequence[BanditRunResult]) -> List[Dict[str, object]]:
        """Aggregate ``results`` into per-algorithm summary rows.

        Each row carries the algorithm name, the mean final cumulative
        reward and the mean final cumulative regret (with standard error
        across runs), the per-arm selection fractions averaged over runs,
        and the chosen ``epsilon`` for reproducibility.
        """
        payoffs = expected_payoffs(self.arms)
        oracle_payoff = max(payoffs.values())
        rows: Dict[str, List[Dict[str, object]]] = {}
        for result in results:
            rewards = result.rewards
            cum_reward = 0.0
            cum_regret = 0.0
            counts = {arm.name: 0 for arm in self.arms}
            for step in result.steps:
                cum_reward += step.reward
                cum_regret += oracle_payoff - payoffs[step.arm_name]
                counts[step.arm_name] += 1
            total = sum(counts.values()) or 1
            fractions = {arm.name: counts[arm.name] / total for arm in self.arms}
            rows.setdefault(result.algorithm, []).append({
                "final_reward": cum_reward,
                "final_regret": cum_regret,
                "selections": fractions,
                "seed": result.seed,
            })
        summary: List[Dict[str, object]] = []
        for algo, entries in rows.items():
            final_rewards = [entry["final_reward"] for entry in entries]
            final_regrets = [entry["final_regret"] for entry in entries]
            avg_fractions = {
                arm.name: sum(entry["selections"][arm.name] for entry in entries) / len(entries)
                for arm in self.arms
            }
            summary.append({
                "algorithm": algo,
                "runs": len(entries),
                "mean_final_reward": sum(final_rewards) / len(final_rewards),
                "std_final_reward": _stddev(final_rewards),
                "mean_final_regret": sum(final_regrets) / len(final_regrets),
                "std_final_regret": _stddev(final_regrets),
                "mean_arm_selection_fraction": avg_fractions,
                "epsilon": self.epsilon,
                "gamma": self.gamma,
                "linucb_alpha": self.linucb_alpha,
                "linucb_ridge": self.linucb_ridge,
            })
        summary.sort(key=lambda row: float(row["mean_final_regret"]))
        return summary


def run_experiment(
    arms: Sequence[Arm],
    *,
    algorithms: Sequence[str] = ("epsilon_greedy", "ucb1", "thompson"),
    steps: int = 200,
    runs: int = 20,
    seed: int = 42,
    epsilon: float = 0.1,
    gamma: float = 0.1,
    linucb_alpha: float = 1.0,
    linucb_ridge: float = 1.0,
) -> Tuple[BanditExperiment, List[BanditRunResult]]:
    """Convenience constructor: build an experiment, run it, return both."""
    experiment = BanditExperiment(
        arms=list(arms),
        algorithms=list(algorithms),
        steps=steps,
        runs=runs,
        seed=seed,
        epsilon=epsilon,
        gamma=gamma,
        linucb_alpha=linucb_alpha,
        linucb_ridge=linucb_ridge,
    )
    return experiment, experiment.run()


def summarize_runs(
    arms: Sequence[Arm],
    results: Sequence[BanditRunResult],
) -> List[Dict[str, object]]:
    """Aggregate a list of run results using the experiment's payoff table."""
    payoffs = expected_payoffs(arms)
    oracle_payoff = max(payoffs.values())
    by_algo: Dict[str, List[BanditRunResult]] = {}
    for result in results:
        by_algo.setdefault(result.algorithm, []).append(result)
    summary: List[Dict[str, object]] = []
    for algo, entries in by_algo.items():
        rewards = [sum(step.reward for step in r.steps) for r in entries]
        regrets = [
            sum(oracle_payoff - payoffs[step.arm_name] for step in r.steps)
            for r in entries
        ]
        avg_fractions: Dict[str, float] = {arm.name: 0.0 for arm in arms}
        for r in entries:
            counts = {arm.name: 0 for arm in arms}
            for step in r.steps:
                counts[step.arm_name] += 1
            total = sum(counts.values()) or 1
            for arm_name, count in counts.items():
                avg_fractions[arm_name] += count / total
        for arm_name in avg_fractions:
            avg_fractions[arm_name] /= len(entries)
        summary.append({
            "algorithm": algo,
            "runs": len(entries),
            "mean_final_reward": sum(rewards) / len(rewards),
            "std_final_reward": _stddev(rewards),
            "mean_final_regret": sum(regrets) / len(regrets),
            "std_final_regret": _stddev(regrets),
            "mean_arm_selection_fraction": avg_fractions,
        })
    summary.sort(key=lambda row: float(row["mean_final_regret"]))
    return summary


@dataclass
class ContextualBanditExperiment:
    """Experiment harness for linear contextual bandits.

    Each step draws a synthetic context, every algorithm observes the
    same context stream within a run, and instantaneous regret is
    ``max_a theta_a · x_t - theta_{a_t} · x_t``. Non-contextual policies
    (epsilon-greedy, UCB1, ...) still run: they ignore the context and
    treat rewards as stationary, which makes them a baseline for LinUCB.
    """

    arms: List[LinearContextualArm]
    algorithms: List[str]
    steps: int = 200
    runs: int = 20
    seed: int = 42
    epsilon: float = 0.1
    gamma: float = 0.1
    linucb_alpha: float = 1.0
    linucb_ridge: float = 1.0
    context_intercept: bool = True

    def __post_init__(self) -> None:
        if not self.arms:
            raise ValueError("at least one arm is required")
        dimensions = {arm.dimension for arm in self.arms}
        if len(dimensions) != 1:
            raise ValueError("all contextual arms must share the same theta dimension")
        if not self.algorithms:
            raise ValueError("at least one algorithm is required")
        for algo in self.algorithms:
            if algo not in _REGISTRY:
                raise ValueError(f"unknown algorithm: {algo}")
        if self.steps < 1:
            raise ValueError("steps must be at least 1")
        if self.runs < 1:
            raise ValueError("runs must be at least 1")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if self.linucb_alpha < 0.0:
            raise ValueError("linucb_alpha must be non-negative")
        if self.linucb_ridge <= 0.0:
            raise ValueError("linucb_ridge must be positive")

    @property
    def dimension(self) -> int:
        return self.arms[0].dimension

    def _make_algorithm(self, name: str, seed: int) -> BanditAlgorithm:
        factory = _REGISTRY[name]
        if name == "epsilon_greedy":
            return factory(epsilon=self.epsilon, seed=seed)
        if name == "exp3":
            return factory(gamma=self.gamma, seed=seed)
        if name == "linucb":
            return factory(
                alpha=self.linucb_alpha,
                dimension=self.dimension,
                ridge=self.linucb_ridge,
                seed=seed,
            )
        return factory(seed=seed)

    def _seeded_arms(self, seed: int) -> List[LinearContextualArm]:
        seeded: List[LinearContextualArm] = []
        for offset, arm in enumerate(self.arms):
            child_seed = (seed * 31 + offset) & 0xFFFFFFFF
            seeded.append(
                LinearContextualArm(
                    name=arm.name,
                    theta=arm.theta,
                    noise_std=arm.noise_std,
                    seed=child_seed,
                )
            )
        return seeded

    def run(self) -> List[BanditRunResult]:
        """Run every algorithm ``runs`` times and return a flat list of results."""
        results: List[BanditRunResult] = []
        for algo_index, algo_name in enumerate(self.algorithms):
            for run_index in range(self.runs):
                seed = self.seed * 7919 + algo_index * 1009 + run_index
                results.append(self._run_single(algo_name, seed))
        return results

    def _select_arm(
        self,
        algo: BanditAlgorithm,
        arms: Sequence[LinearContextualArm],
        step_index: int,
        context: Sequence[float],
    ) -> int:
        if isinstance(algo, LinUCB):
            return algo.select_arm(arms, step_index, context=context)  # type: ignore[arg-type]
        return algo.select_arm(arms, step_index)  # type: ignore[arg-type]

    def _update(
        self,
        algo: BanditAlgorithm,
        arms: Sequence[LinearContextualArm],
        step: BanditStep,
        context: Sequence[float],
    ) -> None:
        if isinstance(algo, LinUCB):
            algo.update(arms, step, context=context)  # type: ignore[arg-type]
            return
        algo.update(arms, step)  # type: ignore[arg-type]

    def _run_single(self, algo_name: str, seed: int) -> BanditRunResult:
        algo = self._make_algorithm(algo_name, seed)
        arms = self._seeded_arms(seed)
        algo.reset(arms)  # type: ignore[arg-type]
        ctx_rng = random.Random(seed * 17 + 3)
        result = BanditRunResult(algorithm=algo_name, seed=seed)
        for step_index in range(self.steps):
            context = sample_context(
                self.dimension, ctx_rng, intercept=self.context_intercept
            )
            arm_index = self._select_arm(algo, arms, step_index, context)
            arm = arms[arm_index]
            reward = arm.draw(context)
            step = BanditStep(
                arm_index=arm_index,
                arm_name=arm.name,
                reward=float(reward),
                step=step_index,
                context=tuple(context),
            )
            result.steps.append(step)
            self._update(algo, arms, step, context)
        return result

    def summarize(self, results: Sequence[BanditRunResult]) -> List[Dict[str, object]]:
        """Aggregate contextual runs into per-algorithm summary rows.

        Instantaneous regret uses the context-conditional oracle
        ``max_a theta_a · x_t``, not a stationary expected-value table.
        """
        rows: Dict[str, List[Dict[str, object]]] = {}
        for result in results:
            cum_reward = 0.0
            cum_regret = 0.0
            oracle_hits = 0
            counts = {arm.name: 0 for arm in self.arms}
            for step in result.steps:
                if step.context is None:
                    raise ValueError("contextual summarize requires a context on every step")
                expected = [arm.expected_given(step.context) for arm in self.arms]
                oracle = max(expected)
                chosen = expected[step.arm_index]
                cum_reward += step.reward
                cum_regret += oracle - chosen
                counts[step.arm_name] += 1
                if chosen == oracle:
                    oracle_hits += 1
            total = sum(counts.values()) or 1
            fractions = {arm.name: counts[arm.name] / total for arm in self.arms}
            rows.setdefault(result.algorithm, []).append({
                "final_reward": cum_reward,
                "final_regret": cum_regret,
                "oracle_hit_rate": oracle_hits / total,
                "selections": fractions,
                "seed": result.seed,
            })
        summary: List[Dict[str, object]] = []
        for algo, entries in rows.items():
            final_rewards = [entry["final_reward"] for entry in entries]
            final_regrets = [entry["final_regret"] for entry in entries]
            hit_rates = [entry["oracle_hit_rate"] for entry in entries]
            avg_fractions = {
                arm.name: sum(entry["selections"][arm.name] for entry in entries) / len(entries)
                for arm in self.arms
            }
            summary.append({
                "algorithm": algo,
                "runs": len(entries),
                "mean_final_reward": sum(final_rewards) / len(final_rewards),
                "std_final_reward": _stddev(final_rewards),
                "mean_final_regret": sum(final_regrets) / len(final_regrets),
                "std_final_regret": _stddev(final_regrets),
                "mean_oracle_hit_rate": sum(hit_rates) / len(hit_rates),
                "mean_arm_selection_fraction": avg_fractions,
                "linucb_alpha": self.linucb_alpha,
                "linucb_ridge": self.linucb_ridge,
                "dimension": self.dimension,
            })
        summary.sort(key=lambda row: float(row["mean_final_regret"]))
        return summary


def run_contextual_experiment(
    arms: Sequence[LinearContextualArm],
    *,
    algorithms: Sequence[str] = ("linucb", "ucb1", "epsilon_greedy"),
    steps: int = 200,
    runs: int = 20,
    seed: int = 42,
    epsilon: float = 0.1,
    gamma: float = 0.1,
    linucb_alpha: float = 1.0,
    linucb_ridge: float = 1.0,
    context_intercept: bool = True,
) -> Tuple[ContextualBanditExperiment, List[BanditRunResult]]:
    """Convenience constructor: build a contextual experiment, run it, return both."""
    experiment = ContextualBanditExperiment(
        arms=list(arms),
        algorithms=list(algorithms),
        steps=steps,
        runs=runs,
        seed=seed,
        epsilon=epsilon,
        gamma=gamma,
        linucb_alpha=linucb_alpha,
        linucb_ridge=linucb_ridge,
        context_intercept=context_intercept,
    )
    return experiment, experiment.run()


def _stddev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return variance ** 0.5


__all__ = [
    "BanditExperiment",
    "BanditRunResult",
    "ContextualBanditExperiment",
    "available_algorithms",
    "run_experiment",
    "run_contextual_experiment",
    "summarize_runs",
]