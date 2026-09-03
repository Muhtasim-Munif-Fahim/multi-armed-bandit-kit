"""Tests for GradientBandit."""

from __future__ import annotations

import pytest

from bandit_kit import (
    BanditStep,
    BernoulliArm,
    GradientBandit,
    gradient_bandit,
)


def make_arms():
    return [
        BernoulliArm(name="A", p=0.10, seed=1),
        BernoulliArm(name="B", p=0.20, seed=2),
        BernoulliArm(name="C", p=0.05, seed=3),
    ]


def test_factory_returns_instance() -> None:
    algo = gradient_bandit(alpha=0.1, seed=0)
    assert isinstance(algo, GradientBandit)


def test_softmax_probabilities_sum_to_one() -> None:
    algo = gradient_bandit(seed=0)
    arms = make_arms()
    algo.reset(arms)
    probabilities = algo._softmax()
    assert len(probabilities) == len(arms)
    assert all(0.0 <= p <= 1.0 for p in probabilities)
    assert sum(probabilities) == pytest.approx(1.0)


def test_uniform_preferences_give_uniform_probabilities() -> None:
    algo = gradient_bandit(seed=0)
    arms = make_arms()
    algo.reset(arms)
    probabilities = algo._softmax()
    for prob in probabilities:
        assert prob == pytest.approx(1.0 / len(arms), abs=1e-9)


def test_chosen_arm_preference_grows_on_high_reward() -> None:
    algo = gradient_bandit(alpha=0.5, seed=0)
    arms = make_arms()
    algo.reset(arms)
    # Pre-feed one zero reward so the baseline starts at 0.
    algo.update(arms, BanditStep(arm_index=0, arm_name="A", reward=0.0, step=0))
    # Now a reward of 1.0 is well above baseline -> arm 1's preference rises.
    initial = algo._preferences[1]
    algo.update(arms, BanditStep(arm_index=1, arm_name="B", reward=1.0, step=1))
    assert algo._preferences[1] > initial


def test_unchosen_arm_preference_drops_on_high_reward() -> None:
    algo = gradient_bandit(alpha=0.5, seed=0)
    arms = make_arms()
    algo.reset(arms)
    algo.update(arms, BanditStep(arm_index=0, arm_name="A", reward=0.0, step=0))
    initial = algo._preferences[0]
    algo.update(arms, BanditStep(arm_index=1, arm_name="B", reward=1.0, step=1))
    # Unchosen arm 0 should see its preference drop relative to before.
    assert algo._preferences[0] < initial


def test_select_arm_returns_valid_index() -> None:
    algo = gradient_bandit(seed=0)
    arms = make_arms()
    algo.reset(arms)
    for step in range(20):
        idx = algo.select_arm(arms, step)
        assert 0 <= idx < len(arms)
        algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=arms[idx].draw(), step=step))


def test_prefers_best_arm_over_horizon() -> None:
    algo = gradient_bandit(alpha=0.1, seed=0)
    arms = make_arms()
    algo.reset(arms)
    counts = {0: 0, 1: 0, 2: 0}
    for step in range(300):
        idx = algo.select_arm(arms, step)
        counts[idx] += 1
        algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=arms[idx].draw(), step=step))
    assert counts[1] > counts[0]
    assert counts[1] > counts[2]


def test_rejects_non_positive_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        GradientBandit(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        GradientBandit(alpha=-0.5)


def test_in_algorithm_registry() -> None:
    from bandit_kit.experiment import available_algorithms
    assert "gradient_bandit" in available_algorithms()
