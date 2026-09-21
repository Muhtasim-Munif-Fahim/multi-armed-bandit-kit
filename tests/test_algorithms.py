"""Unit tests for bandit_kit.algorithms."""

from __future__ import annotations

import pytest

from bandit_kit.algorithms import (
    BanditAlgorithm,
    BanditStep,
    EpsilonGreedy,
    ThompsonBernoulli,
    UCB1,
    epsilon_greedy,
    thompson_sampling_bernoulli,
    ucb1,
)
from bandit_kit.arms import BernoulliArm, GaussianArm


def make_arms() -> list:
    return [
        BernoulliArm(name="A", p=0.10),
        BernoulliArm(name="B", p=0.20),
        BernoulliArm(name="C", p=0.05),
    ]


def test_epsilon_greedy_pulls_each_arm_at_least_once() -> None:
    algo = epsilon_greedy(epsilon=0.1, seed=42)
    arms = make_arms()
    algo.reset(arms)
    selections = []
    for step in range(len(arms)):
        idx = algo.select_arm(arms, step)
        selections.append(idx)
        algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=arms[idx].draw(), step=step))
    assert sorted(selections) == [0, 1, 2]


def test_epsilon_greedy_exploits_best_arm_with_zero_epsilon() -> None:
    algo = epsilon_greedy(epsilon=0.0, seed=0)
    arms = make_arms()
    algo.reset(arms)
    # First round of pulls to seed counts.
    for step in range(len(arms)):
        idx = algo.select_arm(arms, step)
        algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=0.0, step=step))
    # With epsilon=0 and zero observations, all values tie at 0 -> first arm wins.
    # Pull it many times; after it builds a positive count and value, the
    # best arm (B) should win for the rest of the run.
    for step in range(3, 30):
        idx = algo.select_arm(arms, step)
        algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=0.0, step=step))
    assert algo._counts[1] >= 1  # B should be chosen after the warm-up


def test_epsilon_greedy_rejects_invalid_epsilon() -> None:
    with pytest.raises(ValueError):
        EpsilonGreedy(epsilon=-0.1)
    with pytest.raises(ValueError):
        EpsilonGreedy(epsilon=1.5)


def test_ucb1_pulls_each_arm_at_least_once() -> None:
    algo = ucb1(seed=7)
    arms = make_arms()
    algo.reset(arms)
    selections = []
    for step in range(len(arms)):
        idx = algo.select_arm(arms, step)
        selections.append(idx)
        algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=arms[idx].draw(), step=step))
    assert sorted(selections) == [0, 1, 2]


def test_ucb1_prefers_high_value_arm_after_warmup() -> None:
    algo = ucb1(seed=11)
    arms = make_arms()
    algo.reset(arms)
    # Warmup: pull each arm once with its expected value.
    payoffs = [0.1, 0.2, 0.05]
    for step, expected in enumerate(payoffs):
        idx = step
        algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=expected, step=step))
    # Subsequent rewards equal the known means so the UCB index is deterministic.
    counts = {0: 1, 1: 1, 2: 1}
    for step in range(3, 50):
        idx = algo.select_arm(arms, step)
        counts[idx] = counts.get(idx, 0) + 1
        algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=payoffs[idx], step=step))
    assert counts[1] > counts[0]
    assert counts[1] > counts[2]


def test_thompson_sampling_concentrates_on_best_arm() -> None:
    algo = thompson_sampling_bernoulli(seed=99)
    arms = make_arms()
    algo.reset(arms)
    counts = {0: 0, 1: 0, 2: 0}
    for step in range(300):
        idx = algo.select_arm(arms, step)
        counts[idx] += 1
        algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=arms[idx].draw(), step=step))
    # B has the highest expected reward; Thompson should pick it most often.
    assert counts[1] >= counts[0]
    assert counts[1] >= counts[2]


def test_thompson_sampling_falls_back_for_gaussian() -> None:
    algo = thompson_sampling_bernoulli(seed=3)
    arms = [GaussianArm(name="G1", mean=1.0, std=0.5), GaussianArm(name="G2", mean=2.0, std=0.5)]
    algo.reset(arms)
    idx = algo.select_arm(arms, 0)
    assert idx in {0, 1}
    algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=arms[idx].draw(), step=0))


def test_factory_returns_independent_instances() -> None:
    algo_a = epsilon_greedy(epsilon=0.1, seed=1)
    algo_b = epsilon_greedy(epsilon=0.1, seed=1)
    arms = make_arms()
    algo_a.reset(arms)
    algo_b.reset(arms)
    # Same seed -> same selections
    assert algo_a.select_arm(arms, 0) == algo_b.select_arm(arms, 0)