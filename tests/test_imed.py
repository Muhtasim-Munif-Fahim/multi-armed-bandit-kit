"""Tests for IMED (Honda & Takemura, 2015)."""

from __future__ import annotations

import math

import pytest

from bandit_kit import (
    IMED,
    BanditExperiment,
    BanditStep,
    BernoulliArm,
    available_algorithms,
    imed,
    run_experiment,
)
from bandit_kit.algorithms import _bernoulli_kl


def make_arms():
    return [
        BernoulliArm(name="A", p=0.10, seed=1),
        BernoulliArm(name="B", p=0.70, seed=2),
        BernoulliArm(name="C", p=0.30, seed=3),
    ]


def feed(algo, arms, pulls):
    for step, (idx, reward) in enumerate(pulls):
        algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name,
                                     reward=reward, step=step))


def test_factory_and_registry() -> None:
    algo = imed(seed=0)
    assert isinstance(algo, IMED)
    assert algo.name == "imed"
    assert "imed" in available_algorithms()


def test_rejects_bad_eps() -> None:
    for bad in (0.0, 0.5, -1.0, True):
        with pytest.raises(ValueError, match="eps"):
            IMED(eps=bad)  # type: ignore[arg-type]


def test_warm_up_pulls_each_arm_once() -> None:
    arms = make_arms()
    algo = IMED()
    algo.reset(arms)
    chosen = []
    for step in range(3):
        idx = algo.select_arm(arms, step)
        chosen.append(idx)
        feed(algo, arms, [(idx, 0.0)])
    assert sorted(chosen) == [0, 1, 2]


def test_index_matches_formula() -> None:
    arms = make_arms()
    algo = IMED()
    algo.reset(arms)
    # A: 1/4, B: 3/4, C: 1/2
    feed(algo, arms, [(0, 1), (0, 0), (0, 0), (0, 0),
                      (1, 1), (1, 1), (1, 1), (1, 0),
                      (2, 1), (2, 0)])
    assert algo.index(1) == pytest.approx(math.log(4))  # leader: KL = 0
    assert algo.index(0) == pytest.approx(4 * _bernoulli_kl(0.25, 0.75) + math.log(4))
    assert algo.index(2) == pytest.approx(2 * _bernoulli_kl(0.5, 0.75) + math.log(2))
    scores = [algo.index(i) for i in range(3)]
    assert algo.select_arm(arms, 10) == scores.index(min(scores))


def test_trailing_arm_with_few_pulls_gets_explored() -> None:
    arms = make_arms()
    algo = IMED()
    algo.reset(arms)
    # Leader B pulled 50 times at mean 0.6; C pulled once with reward 0.
    feed(algo, arms, [(1, 1.0 if i % 5 < 3 else 0.0) for i in range(50)] + [(0, 0.0), (2, 0.0)])
    # C: 1 * KL(0, 0.6) + log(1) = -log(0.4) ≈ 0.92 < log(50) ≈ 3.9
    assert algo.select_arm(arms, 52) in (0, 2)


def test_perfect_leader_does_not_produce_infinite_indices() -> None:
    arms = make_arms()
    algo = IMED()
    algo.reset(arms)
    feed(algo, arms, [(0, 0.0), (1, 1.0), (2, 0.0)])
    assert all(math.isfinite(algo.index(i)) for i in range(3))


def test_unpulled_index_and_bounds() -> None:
    arms = make_arms()
    algo = IMED()
    algo.reset(arms)
    assert algo.index(0) == -math.inf
    with pytest.raises(IndexError):
        algo.index(5)


def test_rewards_are_clipped() -> None:
    arms = make_arms()
    algo = IMED()
    algo.reset(arms)
    feed(algo, arms, [(0, 3.0), (1, -2.0)])
    assert algo._values == [1.0, 0.0, 0.0]


def test_concentrates_on_best_arm_in_experiment() -> None:
    experiment, results = run_experiment(
        make_arms(), algorithms=["imed", "ucb1"], steps=400, runs=4, seed=0
    )
    assert isinstance(experiment, BanditExperiment)
    by_algo = {}
    for res in results:
        by_algo.setdefault(res.algorithm, []).append(res)
    imed_runs = by_algo["imed"]
    best_share = sum(
        sum(1 for s in run.steps if s.arm_index == 1) / len(run.steps) for run in imed_runs
    ) / len(imed_runs)
    assert best_share > 0.75
