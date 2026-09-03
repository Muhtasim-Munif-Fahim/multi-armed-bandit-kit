"""Tests for DecayingEpsilonGreedy."""

from __future__ import annotations

import pytest

from bandit_kit import (
    BanditStep,
    BernoulliArm,
    DecayingEpsilonGreedy,
    decaying_epsilon_greedy,
)


def make_arms():
    return [
        BernoulliArm(name="A", p=0.10, seed=1),
        BernoulliArm(name="B", p=0.20, seed=2),
        BernoulliArm(name="C", p=0.05, seed=3),
    ]


def test_factory_returns_instance() -> None:
    algo = decaying_epsilon_greedy(seed=0)
    assert isinstance(algo, DecayingEpsilonGreedy)


def test_epsilon_decays_with_step() -> None:
    algo = decaying_epsilon_greedy(
        epsilon_start=0.5, epsilon_min=0.05, decay=0.99, seed=0,
    )
    e0 = algo.epsilon_at(0)
    e10 = algo.epsilon_at(10)
    e100 = algo.epsilon_at(100)
    assert e0 == pytest.approx(0.5)
    assert 0.05 <= e100 < e10 < e0


def test_epsilon_floors_at_min() -> None:
    algo = decaying_epsilon_greedy(
        epsilon_start=0.3, epsilon_min=0.05, decay=0.5, seed=0,
    )
    assert algo.epsilon_at(0) == pytest.approx(0.3)
    # After enough steps, epsilon approaches epsilon_min but never below.
    for step in range(50, 200):
        assert algo.epsilon_at(step) >= 0.05 - 1e-9


def test_pulls_each_arm_at_least_once_then_exploits() -> None:
    algo = decaying_epsilon_greedy(
        epsilon_start=0.5, epsilon_min=0.0, decay=0.99, seed=42,
    )
    arms = make_arms()
    algo.reset(arms)
    counts = {0: 0, 1: 0, 2: 0}
    # First round: each arm must be pulled once.
    for step in range(len(arms)):
        idx = algo.select_arm(arms, step)
        counts[idx] += 1
        algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=arms[idx].draw(), step=step))
    assert sorted(counts.values())[0] == 1


def test_concentrates_on_best_arm() -> None:
    algo = decaying_epsilon_greedy(
        epsilon_start=0.01, epsilon_min=0.0, decay=0.99, seed=0,
    )
    arms = make_arms()
    algo.reset(arms)
    counts = {0: 0, 1: 0, 2: 0}
    # Pre-warm each arm with one update so values are non-zero; otherwise
    # the exploit branch would tie on all zeros and pick the first arm.
    for warm in range(3):
        for idx in range(3):
            algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=arms[idx].draw(), step=warm * 3 + idx))
            counts[idx] += 1
    for step in range(3, 300):
        idx = algo.select_arm(arms, step)
        counts[idx] += 1
        algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=arms[idx].draw(), step=step))
    # B has the highest expected reward; with the warm-up the exploit
    # branch is not tied and B should clearly dominate.
    assert counts[1] > counts[0]
    assert counts[1] > counts[2]


def test_rejects_invalid_epsilon_range() -> None:
    with pytest.raises(ValueError, match="epsilon_min"):
        DecayingEpsilonGreedy(epsilon_start=0.5, epsilon_min=0.7, decay=0.99)
    with pytest.raises(ValueError, match="epsilon_min"):
        DecayingEpsilonGreedy(epsilon_start=0.0, epsilon_min=0.0, decay=0.99)
    with pytest.raises(ValueError, match="epsilon_min"):
        DecayingEpsilonGreedy(epsilon_start=1.5, epsilon_min=0.1, decay=0.99)


def test_rejects_invalid_decay() -> None:
    with pytest.raises(ValueError, match="decay"):
        DecayingEpsilonGreedy(epsilon_start=0.5, epsilon_min=0.1, decay=0.0)
    with pytest.raises(ValueError, match="decay"):
        DecayingEpsilonGreedy(epsilon_start=0.5, epsilon_min=0.1, decay=1.0)
    with pytest.raises(ValueError, match="decay"):
        DecayingEpsilonGreedy(epsilon_start=0.5, epsilon_min=0.1, decay=-0.1)


def test_epsilon_at_rejects_negative_step() -> None:
    algo = decaying_epsilon_greedy(seed=0)
    with pytest.raises(ValueError, match="step"):
        algo.epsilon_at(-1)


def test_in_algorithm_registry() -> None:
    from bandit_kit.experiment import available_algorithms
    assert "decaying_epsilon_greedy" in available_algorithms()
