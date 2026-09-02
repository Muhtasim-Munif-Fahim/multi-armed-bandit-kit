"""Run-and-evaluate harness for multi-armed bandit experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

from .algorithms import (
    BanditAlgorithm,
    BanditStep,
    epsilon_greedy,
    thompson_sampling_bernoulli,
    ucb1,
)
from .arms import Arm, BernoulliArm, GaussianArm, best_arm, expected_payoffs


_REGISTRY: Dict[str, Callable[..., BanditAlgorithm]] = {
    "epsilon_greedy": epsilon_greedy,
    "ucb1": ucb1,
    "thompson": thompson_sampling_bernoulli,
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

    def _make_algorithm(self, name: str, seed: int) -> BanditAlgorithm:
        factory = _REGISTRY[name]
        if name == "epsilon_greedy":
            return factory(epsilon=self.epsilon, seed=seed)
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
) -> Tuple[BanditExperiment, List[BanditRunResult]]:
    """Convenience constructor: build an experiment, run it, return both."""
    experiment = BanditExperiment(
        arms=list(arms),
        algorithms=list(algorithms),
        steps=steps,
        runs=runs,
        seed=seed,
        epsilon=epsilon,
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


def _stddev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return variance ** 0.5


__all__ = [
    "BanditExperiment",
    "BanditRunResult",
    "available_algorithms",
    "run_experiment",
    "summarize_runs",
]