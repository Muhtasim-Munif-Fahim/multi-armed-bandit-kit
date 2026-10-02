"""Tests for UCB-V (Audibert variance-aware UCB)."""

from __future__ import annotations

import math

import pytest

from bandit_kit import (
    BanditExperiment,
    BanditStep,
    BernoulliArm,
    UCB1,
    UCBV,
    available_algorithms,
    run_experiment,
    ucb_v,
)


def make_arms():
    return [
        BernoulliArm(name="A", p=0.10, seed=1),
        BernoulliArm(name="B", p=0.55, seed=2),
        BernoulliArm(name="C", p=0.20, seed=3),
    ]


def test_factory_and_exports() -> None:
    algo = ucb_v(c=2.5, seed=0)
    assert isinstance(algo, UCBV)
    assert algo.c == 2.5
    assert "ucb_v" in available_algorithms()


def test_rejects_bad_c() -> None:
    with pytest.raises(ValueError, match="c must be a non-negative"):
        UCBV(c=-1.0)
    with pytest.raises(ValueError, match="c must be a non-negative"):
        UCBV(c=True)  # type: ignore[arg-type]


def test_round_robin_until_every_arm_seen() -> None:
    algo = ucb_v(c=3.0, seed=0)
    arms = make_arms()
    algo.reset(arms)
    chosen = [algo.select_arm(arms, step) for step in range(3)]
    assert sorted(chosen) == [0, 1, 2]


def test_index_matches_audibert_formula() -> None:
    arms = make_arms()
    algo = UCBV(c=3.0, seed=0)
    algo.reset(arms)
    rewards = [0.0, 1.0, 0.0]
    for step, idx in enumerate([0, 1, 2]):
        algo.update(
            arms,
            BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=rewards[idx], step=step),
        )
    # One more update on arm 1 so variance is informative.
    algo.update(arms, BanditStep(arm_index=1, arm_name="B", reward=1.0, step=3))
    step = 10
    t = step + 1
    log_t = math.log(t)
    for idx in range(3):
        n = algo._counts[idx]
        mean = algo._sums[idx] / n
        var = algo._sum_sq[idx] / n - mean * mean
        if var < 0.0:
            var = 0.0
        expected = mean + math.sqrt(2.0 * var * log_t / n) + 3.0 * log_t / n
        assert algo.upper_bound(idx, step) == pytest.approx(expected)


def test_tracks_second_moments() -> None:
    arms = make_arms()
    algo = UCBV(c=3.0, seed=0)
    algo.reset(arms)
    algo.update(arms, BanditStep(arm_index=0, arm_name="A", reward=0.5, step=0))
    algo.update(arms, BanditStep(arm_index=0, arm_name="A", reward=1.0, step=1))
    assert algo._counts[0] == 2
    assert algo._sums[0] == pytest.approx(1.5)
    assert algo._sum_sq[0] == pytest.approx(0.25 + 1.0)


def test_prefers_better_arm() -> None:
    experiment, results = run_experiment(
        make_arms(),
        algorithms=["ucb_v"],
        steps=300,
        runs=15,
        seed=7,
        ucb_v_c=3.0,
    )
    summary = experiment.summarize(results)[0]
    fractions = summary["mean_arm_selection_fraction"]
    assert fractions["B"] > fractions["A"]
    assert fractions["B"] > fractions["C"]


def test_experiment_accepts_ucb_v() -> None:
    experiment = BanditExperiment(
        arms=make_arms(),
        algorithms=["ucb_v", "ucb1"],
        steps=50,
        runs=2,
        seed=1,
        ucb_v_c=2.0,
    )
    results = experiment.run()
    assert len(results) == 4
    assert {r.algorithm for r in results} == {"ucb_v", "ucb1"}


def test_clips_rewards_outside_unit_interval() -> None:
    arms = make_arms()
    algo = UCBV(c=3.0, seed=0)
    algo.reset(arms)
    algo.update(arms, BanditStep(arm_index=0, arm_name="A", reward=2.5, step=0))
    algo.update(arms, BanditStep(arm_index=0, arm_name="A", reward=-1.0, step=1))
    assert algo._sums[0] == pytest.approx(1.0)  # 1.0 + 0.0 after clip
    assert algo._sum_sq[0] == pytest.approx(1.0)
