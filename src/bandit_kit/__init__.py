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
]