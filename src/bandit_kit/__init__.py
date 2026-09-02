"""Multi-armed bandit algorithms, experiments, and reporting."""

from .arms import Arm, BernoulliArm, GaussianArm, arm_from_spec
from .algorithms import epsilon_greedy, ucb1, thompson_sampling_bernoulli
from .experiment import BanditExperiment, run_experiment, summarize_runs
from .metrics import (
    cumulative_reward,
    cumulative_regret,
    arm_selection_counts,
    arm_selection_fractions,
)

__all__ = [
    "Arm",
    "BernoulliArm",
    "GaussianArm",
    "arm_from_spec",
    "epsilon_greedy",
    "ucb1",
    "thompson_sampling_bernoulli",
    "BanditExperiment",
    "run_experiment",
    "summarize_runs",
    "cumulative_reward",
    "cumulative_regret",
    "arm_selection_counts",
    "arm_selection_fractions",
]