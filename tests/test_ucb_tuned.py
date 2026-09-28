"""Tests for UCB-Tuned (variance-aware UCB)."""

from __future__ import annotations

import math

import pytest

from bandit_kit import (
    BanditExperiment,
    BanditStep,
    BernoulliArm,
    UCB1,
    UCBTuned,
    available_algorithms,
    run_experiment,
    ucb_tuned,
)


def make_arms():
    return [
        BernoulliArm(name="A", p=0.10, seed=1),
        BernoulliArm(name="B", p=0.55, seed=2),
        BernoulliArm(name="C", p=0.20, seed=3),
    ]


def test_factory_and_exports() -> None:
    algo = ucb_tuned(seed=0)
    assert isinstance(algo, UCBTuned)
    assert "ucb_tuned" in available_algorithms()


def test_round_robin_until_every_arm_seen() -> None:
    algo = ucb_tuned(seed=0)
    arms = make_arms()
    algo.reset(arms)
    chosen = [algo.select_arm(arms, step) for step in range(3)]
    assert sorted(chosen) == [0, 1, 2]


def test_index_matches_formula_on_fixed_history() -> None:
    arms = make_arms()
    algo = UCBTuned(seed=0)
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
    # At step=5 (t=6) after 6 updates, evaluate indices manually.
    t = 6
    log_t = math.log(t)
    for idx in range(3):
        n = algo._counts[idx]
        mean = algo._sums[idx] / n
        var = max(0.0, algo._sum_sq[idx] / n - mean * mean)
        v = var + math.sqrt(2.0 * log_t / n)
        expected = mean + math.sqrt((log_t / n) * min(0.25, v))
        assert algo.upper_bound(idx, step=t - 1) == pytest.approx(expected)


def test_clips_rewards_outside_unit_interval() -> None:
    arms = make_arms()
    algo = UCBTuned(seed=0)
    algo.reset(arms)
    algo.update(arms, BanditStep(arm_index=0, arm_name="A", reward=2.5, step=0))
    algo.update(arms, BanditStep(arm_index=0, arm_name="A", reward=-1.0, step=1))
    assert algo._sums[0] == pytest.approx(1.0)  # 1.0 + 0.0
    assert algo._sum_sq[0] == pytest.approx(1.0)


def test_deterministic_arm_has_smaller_bonus_than_ucb1() -> None:
    """After many identical rewards, UCB-Tuned's V falls below 1/4 so its
    bonus is strictly smaller than UCB1's sqrt(2 ln t / n)."""
    arms = [BernoulliArm(name="A", p=0.5, seed=1), BernoulliArm(name="B", p=0.5, seed=2)]
    tuned = UCBTuned(seed=0)
    plain = UCB1(seed=0)
    tuned.reset(arms)
    plain.reset(arms)
    for step in range(20):
        idx = step % 2
        reward = 1.0  # deterministic success
        record = BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=reward, step=step)
        tuned.update(arms, record)
        plain.update(arms, record)
    step = 20
    t = step + 1
    # Arm 0: mean=1, variance=0 → V = sqrt(2 ln t / n) which may still exceed 1/4
    # early; use a longer run of identical pulls to force V < 1/4.
    for step in range(20, 80):
        record = BanditStep(arm_index=0, arm_name="A", reward=1.0, step=step)
        tuned.update(arms, record)
        plain.update(arms, record)
    step = 80
    n0 = tuned._counts[0]
    mean0 = tuned._sums[0] / n0
    assert mean0 == pytest.approx(1.0)
    tuned_bonus = tuned.upper_bound(0, step) - mean0
    ucb1_bonus = math.sqrt(2.0 * math.log(step + 1) / n0)
    assert tuned_bonus < ucb1_bonus


def test_prefers_better_arm_in_stationary_bandit() -> None:
    experiment, results = run_experiment(
        make_arms(),
        algorithms=["ucb_tuned"],
        steps=300,
        runs=15,
        seed=7,
    )
    summary = experiment.summarize(results)[0]
    fractions = summary["mean_arm_selection_fraction"]
    assert fractions["B"] > fractions["A"]
    assert fractions["B"] > fractions["C"]


def test_experiment_accepts_ucb_tuned() -> None:
    experiment = BanditExperiment(
        arms=make_arms(),
        algorithms=["ucb_tuned", "ucb1"],
        steps=40,
        runs=2,
        seed=1,
    )
    results = experiment.run()
    assert len(results) == 4
    assert {r.algorithm for r in results} == {"ucb_tuned", "ucb1"}


def test_upper_bound_rejects_bad_args() -> None:
    algo = ucb_tuned(seed=0)
    arms = make_arms()
    algo.reset(arms)
    with pytest.raises(ValueError):
        algo.upper_bound(0, step=-1)
    with pytest.raises(IndexError):
        algo.upper_bound(9, step=0)
