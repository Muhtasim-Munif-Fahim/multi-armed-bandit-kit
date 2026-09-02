"""Multi-armed bandit algorithms, experiments, and reporting."""

from .algorithms import (
    BanditAlgorithm,
    BanditStep,
    EpsilonGreedy,
    ThompsonBernoulli,
    UCB1,
    epsilon_greedy,
    thompson_sampling_bernoulli,
    ucb1,
)
from .arms import (
    Arm,
    BernoulliArm,
    GaussianArm,
    arm_from_spec,
    best_arm,
    expected_payoffs,
)
from .experiment import (
    BanditExperiment,
    BanditRunResult,
    available_algorithms,
    run_experiment,
    summarize_runs,
)
from .metrics import (
    arm_selection_counts,
    arm_selection_fractions,
    cumulative_regret,
    cumulative_reward,
)
from .reporting import render_markdown_report

__all__ = [
    "Arm",
    "BernoulliArm",
    "GaussianArm",
    "arm_from_spec",
    "best_arm",
    "expected_payoffs",
    "BanditAlgorithm",
    "BanditStep",
    "EpsilonGreedy",
    "UCB1",
    "ThompsonBernoulli",
    "epsilon_greedy",
    "ucb1",
    "thompson_sampling_bernoulli",
    "BanditExperiment",
    "BanditRunResult",
    "available_algorithms",
    "run_experiment",
    "summarize_runs",
    "cumulative_reward",
    "cumulative_regret",
    "arm_selection_counts",
    "arm_selection_fractions",
    "render_markdown_report",
]