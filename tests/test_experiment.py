"""Unit tests for bandit_kit.experiment and bandit_kit.metrics."""

from __future__ import annotations

import pytest

from bandit_kit import (
    BanditExperiment,
    BernoulliArm,
    arm_selection_counts,
    arm_selection_fractions,
    available_algorithms,
    cumulative_regret,
    cumulative_reward,
    run_experiment,
    summarize_runs,
)


def make_arms():
    return [
        BernoulliArm(name="A", p=0.10),
        BernoulliArm(name="B", p=0.20),
        BernoulliArm(name="C", p=0.05),
    ]


def test_available_algorithms_lists_bundled_policies() -> None:
    algos = available_algorithms()
    assert "epsilon_greedy" in algos
    assert "ucb1" in algos
    assert "thompson" in algos
    assert "exp3" in algos
    assert "linucb" in algos
    assert "boltzmann" in algos
    assert "softmax" in algos
    assert "kl_ucb" in algos


def test_experiment_rejects_empty_arms() -> None:
    with pytest.raises(ValueError):
        BanditExperiment(arms=[], algorithms=["epsilon_greedy"])


def test_experiment_rejects_unknown_algorithm() -> None:
    with pytest.raises(ValueError, match="unknown algorithm"):
        BanditExperiment(arms=make_arms(), algorithms=["bogus"])


def test_experiment_rejects_invalid_step_count() -> None:
    with pytest.raises(ValueError):
        BanditExperiment(arms=make_arms(), algorithms=["epsilon_greedy"], steps=0)
    with pytest.raises(ValueError):
        BanditExperiment(arms=make_arms(), algorithms=["epsilon_greedy"], runs=0)


def test_run_experiment_returns_results_for_every_algorithm_and_run() -> None:
    arms = make_arms()
    experiment, results = run_experiment(
        arms=arms, algorithms=("epsilon_greedy", "ucb1"), steps=10, runs=3, seed=42
    )
    assert len(results) == 6  # 2 algorithms * 3 runs
    assert {r.algorithm for r in results} == {"epsilon_greedy", "ucb1"}
    for result in results:
        assert len(result.steps) == 10
        assert all(step.arm_index in {0, 1, 2} for step in result.steps)


def test_run_experiment_is_deterministic_for_a_given_seed() -> None:
    arms = make_arms()
    _, results_a = run_experiment(arms=arms, steps=20, runs=2, seed=99)
    _, results_b = run_experiment(arms=arms, steps=20, runs=2, seed=99)
    for ra, rb in zip(results_a, results_b):
        assert [step.arm_index for step in ra.steps] == [step.arm_index for step in rb.steps]
        assert [step.reward for step in ra.steps] == [step.reward for step in rb.steps]


def test_summarize_returns_one_row_per_algorithm() -> None:
    arms = make_arms()
    experiment, results = run_experiment(arms=arms, steps=30, runs=4, seed=7)
    summary = summarize_runs(arms, results)
    assert {row["algorithm"] for row in summary} == {"epsilon_greedy", "ucb1", "thompson"}
    for row in summary:
        assert row["runs"] == 4
        assert row["mean_final_reward"] >= 0
        assert row["mean_final_regret"] >= 0
        assert abs(sum(row["mean_arm_selection_fraction"].values()) - 1.0) < 1e-6


def test_summarize_orders_algorithms_by_regret() -> None:
    arms = make_arms()
    _, results = run_experiment(arms=arms, steps=100, runs=10, seed=11)
    summary = summarize_runs(arms, results)
    regrets = [row["mean_final_regret"] for row in summary]
    assert regrets == sorted(regrets)


def test_thompson_collects_most_reward_over_seeded_runs() -> None:
    arms = make_arms()
    _, results = run_experiment(arms=arms, algorithms=("thompson", "epsilon_greedy"), steps=200, runs=20, seed=123)
    summary = summarize_runs(arms, results)
    by_algo = {row["algorithm"]: row for row in summary}
    assert by_algo["thompson"]["mean_final_reward"] >= by_algo["epsilon_greedy"]["mean_final_reward"] - 0.01


def test_cumulative_reward_grows_monotonically() -> None:
    rewards = [1.0, 0.0, 1.0, 1.0, 0.0]
    curve = cumulative_reward(rewards)
    assert curve == [1.0, 1.0, 2.0, 3.0, 3.0]


def test_cumulative_regret_matches_oracle_difference() -> None:
    arms = make_arms()
    selections = [0, 1, 2, 1, 1]
    rewards = [0.1, 0.2, 0.05, 0.2, 0.2]
    oracle = max(arm.expected_value for arm in arms)
    regret = cumulative_regret(rewards, arms, selections=selections)
    expected = [oracle - arm.expected_value for arm in arms]
    diff = [oracle - arm.expected_value for arm in arms]
    expected_curve = []
    running = 0.0
    for r, s in zip(rewards, selections):
        running += diff[s]
        expected_curve.append(running)
    assert regret == expected_curve


def test_arm_selection_counts_initialise_zero() -> None:
    counts = arm_selection_counts(["A", "B"])
    assert counts == {"A": 0, "B": 0}


def test_arm_selection_fractions_sum_to_one() -> None:
    arms = make_arms()
    fractions = arm_selection_fractions(["A", "B", "C"], [0, 1, 1, 2, 1], arms)
    assert abs(sum(fractions.values()) - 1.0) < 1e-6


def test_fractions_handles_empty_selections() -> None:
    arms = make_arms()
    fractions = arm_selection_fractions(["A", "B", "C"], [], arms)
    assert fractions == {"A": 0.0, "B": 0.0, "C": 0.0}