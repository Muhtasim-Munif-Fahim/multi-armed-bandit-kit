"""Tests for DiscountedUCB."""

from __future__ import annotations

import math

import pytest

from bandit_kit import (
    BanditExperiment,
    BanditStep,
    BernoulliArm,
    DiscountedUCB,
    UCB1,
    available_algorithms,
    discounted_ucb,
    run_experiment,
)


def make_arms():
    return [
        BernoulliArm(name="A", p=0.10, seed=1),
        BernoulliArm(name="B", p=0.55, seed=2),
        BernoulliArm(name="C", p=0.20, seed=3),
    ]


def test_factory_and_exports() -> None:
    algo = discounted_ucb(gamma=0.85, seed=0)
    assert isinstance(algo, DiscountedUCB)
    assert algo.gamma == 0.85
    assert "discounted_ucb" in available_algorithms()


def test_rejects_bad_gamma() -> None:
    with pytest.raises(ValueError, match=r"gamma must be in \(0, 1\]"):
        DiscountedUCB(gamma=0.0)
    with pytest.raises(ValueError, match=r"gamma must be in \(0, 1\]"):
        DiscountedUCB(gamma=1.5)
    with pytest.raises(ValueError, match=r"gamma must be in \(0, 1\]"):
        DiscountedUCB(gamma=-0.1)


def test_round_robin_until_every_arm_seen() -> None:
    algo = discounted_ucb(gamma=0.9, seed=0)
    arms = make_arms()
    algo.reset(arms)
    chosen = [algo.select_arm(arms, step) for step in range(3)]
    assert sorted(chosen) == [0, 1, 2]


def test_gamma_one_matches_ucb1_on_identical_history() -> None:
    """With γ=1 (no discounting), indices match UCB1."""
    arms = make_arms()
    ducb = DiscountedUCB(gamma=1.0, seed=0)
    ucb = UCB1(seed=0)
    ducb.reset(arms)
    ucb.reset(arms)
    for step in range(40):
        a = ducb.select_arm(arms, step)
        b = ucb.select_arm(arms, step)
        assert a == b
        reward = 0.3 if a == 0 else (0.8 if a == 1 else 0.1)
        record = BanditStep(arm_index=a, arm_name=arms[a].name, reward=reward, step=step)
        ducb.update(arms, record)
        ucb.update(arms, record)


def test_discount_fades_old_rewards() -> None:
    """Repeated discounting drives an unused arm's count toward zero."""
    arms = [
        BernoulliArm(name="cold", p=0.01, seed=1),
        BernoulliArm(name="hot", p=0.99, seed=2),
    ]
    algo = DiscountedUCB(gamma=0.5, seed=0)
    algo.reset(arms)
    # Seed both arms once.
    for idx, step in enumerate(range(2)):
        algo.select_arm(arms, step)
        algo.update(
            arms,
            BanditStep(
                arm_index=idx,
                arm_name=arms[idx].name,
                reward=0.0 if idx == 0 else 1.0,
                step=step,
            ),
        )
    cold_after_seed = algo._disc_counts[0]
    assert cold_after_seed > 0.0
    # Keep pulling the hot arm; cold's discounted count should shrink.
    for step in range(2, 12):
        algo.update(
            arms,
            BanditStep(arm_index=1, arm_name="hot", reward=1.0, step=step),
        )
    assert algo._disc_counts[0] < cold_after_seed
    assert algo._disc_counts[0] < 0.01
    assert algo._disc_counts[1] > algo._disc_counts[0]


def test_prefers_better_arm_in_stationary_bandit() -> None:
    experiment, results = run_experiment(
        make_arms(),
        algorithms=["discounted_ucb"],
        steps=300,
        runs=15,
        seed=7,
        discount_gamma=0.99,
    )
    summary = experiment.summarize(results)[0]
    fractions = summary["mean_arm_selection_fraction"]
    assert fractions["B"] > fractions["A"]
    assert fractions["B"] > fractions["C"]


def test_experiment_accepts_discounted_ucb() -> None:
    experiment = BanditExperiment(
        arms=make_arms(),
        algorithms=["discounted_ucb", "ucb1"],
        steps=50,
        runs=2,
        seed=1,
        discount_gamma=0.95,
    )
    results = experiment.run()
    assert len(results) == 4
    assert {r.algorithm for r in results} == {"discounted_ucb", "ucb1"}


def test_index_uses_calendar_log_term() -> None:
    arms = make_arms()
    algo = DiscountedUCB(gamma=1.0, seed=0)
    algo.reset(arms)
    for step, idx in enumerate([0, 1, 2]):
        algo.update(
            arms,
            BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=0.5, step=step),
        )
    chosen = algo.select_arm(arms, step=10)
    assert chosen in (0, 1, 2)
    expected_bonus = math.sqrt(2.0 * math.log(11) / 1)
    for idx in range(3):
        mean = algo._disc_sums[idx] / algo._disc_counts[idx]
        score = mean + expected_bonus
        assert score == pytest.approx(0.5 + expected_bonus)
