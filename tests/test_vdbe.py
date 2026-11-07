"""Tests for Value-Difference Based Exploration (VDBE)."""

from __future__ import annotations

import math

import pytest

from bandit_kit import (
    BanditStep,
    BernoulliArm,
    VDBE,
    available_algorithms,
    run_experiment,
    vdbe,
)


def make_arms():
    return [
        BernoulliArm(name="A", p=0.10, seed=1),
        BernoulliArm(name="B", p=0.80, seed=2),
        BernoulliArm(name="C", p=0.20, seed=3),
    ]


def test_factory_and_registry() -> None:
    algo = vdbe(delta=0.2, sigma=0.5, seed=0)
    assert isinstance(algo, VDBE)
    assert algo.delta == 0.2
    assert algo.sigma == 0.5
    assert "vdbe" in available_algorithms()


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, True])
def test_rejects_bad_delta(bad) -> None:
    with pytest.raises(ValueError):
        VDBE(delta=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0.0, -1.0, True])
def test_rejects_bad_sigma(bad) -> None:
    with pytest.raises(ValueError):
        VDBE(sigma=bad)  # type: ignore[arg-type]


def test_epsilon_rises_on_surprise_and_decays_on_calm() -> None:
    arms = make_arms()
    algo = VDBE(delta=1.0, sigma=1.0, epsilon=0.0, seed=0)
    algo.reset(arms)
    # Force pull arm 0 with Q=0, reward=1 -> |r-Q|=1, f=1-e^{-1}
    algo.update(arms, BanditStep(arm_index=0, arm_name="A", reward=1.0, step=0))
    expected = 1.0 - math.exp(-1.0)
    assert algo.epsilon == pytest.approx(expected)
    # Calm update with matching reward after Q has moved: smaller surprise
    # With delta=1, epsilon becomes exactly the new surprise.
    q = algo._values[0]
    algo.update(arms, BanditStep(arm_index=0, arm_name="A", reward=q, step=1))
    assert algo.epsilon == pytest.approx(0.0)


def test_round_robin_then_epsilon_greedy() -> None:
    arms = make_arms()
    algo = vdbe(delta=0.1, sigma=1.0, epsilon=0.0, seed=0)
    algo.reset(arms)
    chosen = []
    for t in range(3):
        idx = algo.select_arm(arms, t)
        algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=0.0, step=t))
        chosen.append(idx)
    assert chosen == [0, 1, 2]
    # epsilon stays 0 on zero-reward calm updates from Q=0, so thereafter greedy
    for t in range(3, 20):
        idx = algo.select_arm(arms, t)
        algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=0.0, step=t))
        assert idx == 0  # all values 0, lowest index wins


def test_reset_restores_epsilon() -> None:
    arms = make_arms()
    algo = VDBE(delta=1.0, sigma=1.0, epsilon=0.5, seed=0)
    algo.reset(arms)
    algo.update(arms, BanditStep(arm_index=0, arm_name="A", reward=1.0, step=0))
    assert algo.epsilon != 0.5
    algo.reset(arms)
    assert algo.epsilon == 0.5


def test_runs_in_experiment_harness() -> None:
    _, results = run_experiment(make_arms(), algorithms=["vdbe"], steps=100, runs=3)
    assert len(results) == 3
    assert all(len(r.steps) == 100 for r in results)
