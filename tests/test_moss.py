"""Tests for MOSS (Minimax Optimal Strategy in the Stochastic case)."""

from __future__ import annotations

import math

import pytest

from bandit_kit import (
    BanditExperiment,
    BanditStep,
    BernoulliArm,
    MOSS,
    available_algorithms,
    moss,
    run_experiment,
)


def make_arms():
    return [
        BernoulliArm(name="A", p=0.10, seed=1),
        BernoulliArm(name="B", p=0.55, seed=2),
        BernoulliArm(name="C", p=0.20, seed=3),
    ]


def test_factory_and_exports() -> None:
    algo = moss(horizon=100, seed=0)
    assert isinstance(algo, MOSS)
    assert algo.horizon == 100
    assert "moss" in available_algorithms()


def test_rejects_bad_horizon() -> None:
    with pytest.raises(ValueError):
        moss(horizon=0)
    with pytest.raises(ValueError):
        moss(horizon=-5)
    with pytest.raises(ValueError):
        MOSS(horizon=True)  # type: ignore[arg-type]


def test_round_robin_until_every_arm_seen() -> None:
    algo = moss(horizon=50, seed=0)
    arms = make_arms()
    algo.reset(arms)
    chosen = [algo.select_arm(arms, step) for step in range(3)]
    assert sorted(chosen) == [0, 1, 2]


def test_index_matches_formula_on_fixed_history() -> None:
    arms = make_arms()
    horizon = 100
    algo = MOSS(horizon=horizon, seed=0)
    algo.reset(arms)
    rewards = {0: [1.0, 0.0, 1.0], 1: [0.0, 0.0], 2: [1.0]}
    step = 0
    for idx, vals in rewards.items():
        for r in vals:
            algo.update(
                arms,
                BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=r, step=step),
            )
            step += 1
    k = 3
    for idx in range(3):
        n = algo._counts[idx]
        mean = algo._sums[idx] / n
        log_arg = horizon / (n * k)
        log_term = math.log(log_arg) if log_arg > 1.0 else 0.0
        expected = mean + math.sqrt(log_term / (2.0 * n))
        assert algo.upper_bound(idx) == pytest.approx(expected)


def test_zero_bonus_when_overpulled() -> None:
    """When n_a > T/K the log term is floored at 0 so bonus is zero."""
    arms = [BernoulliArm(name="A", p=0.5, seed=1), BernoulliArm(name="B", p=0.5, seed=2)]
    horizon = 10
    algo = MOSS(horizon=horizon, seed=0)
    algo.reset(arms)
    # Pull arm 0 six times (> T/K = 5) with reward 1
    for step in range(6):
        algo.update(arms, BanditStep(arm_index=0, arm_name="A", reward=1.0, step=step))
    assert algo.upper_bound(0) == pytest.approx(1.0)


def test_clips_rewards_outside_unit_interval() -> None:
    arms = make_arms()
    algo = moss(horizon=50, seed=0)
    algo.reset(arms)
    algo.update(arms, BanditStep(arm_index=0, arm_name="A", reward=2.5, step=0))
    algo.update(arms, BanditStep(arm_index=0, arm_name="A", reward=-1.0, step=1))
    assert algo._sums[0] == pytest.approx(1.0)


def test_prefers_better_arm_in_stationary_bandit() -> None:
    experiment, results = run_experiment(
        make_arms(),
        algorithms=["moss"],
        steps=300,
        runs=15,
        seed=7,
        moss_horizon=300,
    )
    summary = experiment.summarize(results)[0]
    fractions = summary["mean_arm_selection_fraction"]
    assert fractions["B"] > fractions["A"]
    assert fractions["B"] > fractions["C"]


def test_experiment_defaults_horizon_to_steps() -> None:
    experiment = BanditExperiment(
        arms=make_arms(),
        algorithms=["moss"],
        steps=40,
        runs=1,
        seed=1,
    )
    algo = experiment._make_algorithm("moss", seed=0)
    assert isinstance(algo, MOSS)
    assert algo.horizon == 40


def test_experiment_accepts_moss() -> None:
    experiment = BanditExperiment(
        arms=make_arms(),
        algorithms=["moss", "ucb1"],
        steps=40,
        runs=2,
        seed=1,
        moss_horizon=40,
    )
    results = experiment.run()
    assert len(results) == 4
    assert {r.algorithm for r in results} == {"moss", "ucb1"}


def test_upper_bound_rejects_bad_index() -> None:
    algo = moss(horizon=20, seed=0)
    arms = make_arms()
    algo.reset(arms)
    with pytest.raises(IndexError):
        algo.upper_bound(9)
