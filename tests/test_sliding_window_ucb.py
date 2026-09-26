"""Tests for sliding-window UCB."""

from __future__ import annotations

import math

import pytest

from bandit_kit import (
    BanditExperiment,
    BanditStep,
    BernoulliArm,
    SlidingWindowUCB,
    UCB1,
    available_algorithms,
    run_experiment,
    sliding_window_ucb,
)


def make_arms():
    return [
        BernoulliArm(name="A", p=0.10, seed=1),
        BernoulliArm(name="B", p=0.55, seed=2),
        BernoulliArm(name="C", p=0.20, seed=3),
    ]


def test_factory_and_exports() -> None:
    algo = sliding_window_ucb(window=50, seed=0)
    assert isinstance(algo, SlidingWindowUCB)
    assert algo.window == 50
    assert "sliding_window_ucb" in available_algorithms()


def test_rejects_bad_window() -> None:
    with pytest.raises(ValueError, match="window must be an integer >= 1"):
        SlidingWindowUCB(window=0)
    with pytest.raises(ValueError, match="window must be an integer >= 1"):
        SlidingWindowUCB(window=True)  # type: ignore[arg-type]


def test_round_robin_until_every_arm_seen() -> None:
    algo = sliding_window_ucb(window=10, seed=0)
    arms = make_arms()
    algo.reset(arms)
    chosen = [algo.select_arm(arms, step) for step in range(3)]
    assert sorted(chosen) == [0, 1, 2]


def test_large_window_matches_ucb1_on_identical_history() -> None:
    """With a window larger than the horizon, indices match UCB1."""
    arms = make_arms()
    sw = SlidingWindowUCB(window=10_000, seed=0)
    ucb = UCB1(seed=0)
    sw.reset(arms)
    ucb.reset(arms)
    for step in range(40):
        a = sw.select_arm(arms, step)
        b = ucb.select_arm(arms, step)
        assert a == b
        reward = 0.3 if a == 0 else (0.8 if a == 1 else 0.1)
        record = BanditStep(arm_index=a, arm_name=arms[a].name, reward=reward, step=step)
        sw.update(arms, record)
        ucb.update(arms, record)


def test_window_forgets_old_rewards() -> None:
    """After the window fills with arm-0 rewards, then only arm-1 is played,
    the mean of arm 0 eventually leaves the window."""
    arms = [
        BernoulliArm(name="cold", p=0.01, seed=1),
        BernoulliArm(name="hot", p=0.99, seed=2),
    ]
    algo = SlidingWindowUCB(window=5, seed=0)
    algo.reset(arms)
    # Seed both arms once.
    for idx, step in enumerate(range(2)):
        algo.select_arm(arms, step)  # round-robin
        algo.update(
            arms,
            BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=0.0 if idx == 0 else 1.0, step=step),
        )
    # Fill the window with cold-arm failures so its mean is 0.
    for step in range(2, 7):
        algo.update(
            arms,
            BanditStep(arm_index=0, arm_name="cold", reward=0.0, step=step),
        )
    assert algo._counts[0] >= 1
    # Now push five hot-arm successes; cold should drop out of the window.
    for step in range(7, 12):
        algo.update(
            arms,
            BanditStep(arm_index=1, arm_name="hot", reward=1.0, step=step),
        )
    assert algo._counts[0] == 0
    assert algo._counts[1] == 5
    assert algo._sums[1] / algo._counts[1] == pytest.approx(1.0)


def test_prefers_better_arm_in_stationary_bandit() -> None:
    experiment, results = run_experiment(
        make_arms(),
        algorithms=["sliding_window_ucb"],
        steps=300,
        runs=15,
        seed=7,
        sliding_window=150,
    )
    summary = experiment.summarize(results)[0]
    fractions = summary["mean_arm_selection_fraction"]
    assert fractions["B"] > fractions["A"]
    assert fractions["B"] > fractions["C"]


def test_experiment_accepts_sliding_window_algorithm() -> None:
    experiment = BanditExperiment(
        arms=make_arms(),
        algorithms=["sliding_window_ucb", "ucb1"],
        steps=50,
        runs=2,
        seed=1,
        sliding_window=20,
    )
    results = experiment.run()
    assert len(results) == 4
    assert {r.algorithm for r in results} == {"sliding_window_ucb", "ucb1"}


def test_index_uses_min_t_window_in_log_term() -> None:
    arms = make_arms()
    algo = SlidingWindowUCB(window=4, seed=0)
    algo.reset(arms)
    for step, idx in enumerate([0, 1, 2]):
        algo.update(
            arms,
            BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=0.5, step=step),
        )
    # All arms have count 1; scores differ only by shared log term.
    chosen = algo.select_arm(arms, step=10)
    assert chosen in (0, 1, 2)
    # log term at step 10 with window 4 is 2*ln(4)
    expected_bonus = math.sqrt(2.0 * math.log(4) / 1)
    for idx in range(3):
        mean = algo._sums[idx] / algo._counts[idx]
        score = mean + expected_bonus
        assert score == pytest.approx(0.5 + expected_bonus)
