"""Tests for the Explore-Then-Commit policy."""

from __future__ import annotations

import pytest

from bandit_kit import (
    BanditStep,
    BernoulliArm,
    ExploreThenCommit,
    available_algorithms,
    explore_then_commit,
    run_experiment,
)


def make_arms():
    return [
        BernoulliArm(name="A", p=0.10, seed=1),
        BernoulliArm(name="B", p=0.80, seed=2),
        BernoulliArm(name="C", p=0.20, seed=3),
    ]


def test_factory_and_registry() -> None:
    algo = explore_then_commit(exploration_rounds=4, seed=0)
    assert isinstance(algo, ExploreThenCommit)
    assert algo.exploration_rounds == 4
    assert "explore_then_commit" in available_algorithms()


@pytest.mark.parametrize("bad", [0, -3, True, 2.5])
def test_rejects_bad_rounds(bad) -> None:
    with pytest.raises(ValueError):
        ExploreThenCommit(exploration_rounds=bad)  # type: ignore[arg-type]


def _drive(algo, arms, steps):
    algo.reset(arms)
    chosen = []
    for t in range(steps):
        idx = algo.select_arm(arms, t)
        reward = arms[idx].draw()
        algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=reward, step=t))
        chosen.append(idx)
    return chosen


def test_round_robin_then_commit_to_best() -> None:
    arms = make_arms()
    algo = explore_then_commit(exploration_rounds=20, seed=0)
    chosen = _drive(algo, arms, 200)
    assert chosen[:6] == [0, 1, 2, 0, 1, 2]
    assert sorted(chosen[:60].count(i) for i in range(3)) == [20, 20, 20]
    assert algo.committed_arm == 1
    assert not algo.is_exploring
    assert set(chosen[60:]) == {1}


def test_reset_clears_commitment() -> None:
    arms = make_arms()
    algo = explore_then_commit(exploration_rounds=2)
    _drive(algo, arms, 10)
    assert algo.committed_arm is not None
    algo.reset(arms)
    assert algo.is_exploring
    assert algo.select_arm(arms, 0) == 0


def test_tie_breaks_to_lowest_index() -> None:
    arms = make_arms()
    algo = ExploreThenCommit(exploration_rounds=1)
    algo.reset(arms)
    for t in range(3):
        idx = algo.select_arm(arms, t)
        algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=1.0, step=t))
    assert algo.select_arm(arms, 3) == 0


def test_optimal_exploration_rounds() -> None:
    assert ExploreThenCommit.optimal_exploration_rounds(1000, 0.5) == 67
    assert ExploreThenCommit.optimal_exploration_rounds(4, 0.5) == 1
    with pytest.raises(ValueError):
        ExploreThenCommit.optimal_exploration_rounds(100, 0.0)
    with pytest.raises(ValueError):
        ExploreThenCommit.optimal_exploration_rounds(0, 0.5)


def test_runs_in_experiment_harness() -> None:
    _, results = run_experiment(make_arms(), algorithms=["explore_then_commit"], steps=100, runs=3)
    assert len(results) == 3
    assert all(len(r.steps) == 100 for r in results)
