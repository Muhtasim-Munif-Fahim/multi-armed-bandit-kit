"""Tests for regret_curve and compare_to_oracle."""

from __future__ import annotations

import pytest

from bandit_kit import (
    BernoulliArm,
    compare_to_oracle,
    regret_curve,
)


def make_arms():
    return [
        BernoulliArm(name="A", p=0.10, seed=1),
        BernoulliArm(name="B", p=0.20, seed=2),
        BernoulliArm(name="C", p=0.05, seed=3),
    ]


def test_regret_curve_with_selections() -> None:
    arms = make_arms()
    selections = [0, 1, 2, 1, 1]
    rewards = [0.1, 0.2, 0.05, 0.2, 0.2]
    curve = regret_curve(rewards, arms, selections=selections)
    expected_payoffs = {"A": 0.10, "B": 0.20, "C": 0.05}
    oracle = 0.20
    diffs = [oracle - expected_payoffs[arms[idx].name] for idx in selections]
    expected = []
    running = 0.0
    for diff in diffs:
        running += diff
        expected.append(running)
    assert curve == expected


def test_regret_curve_without_selections() -> None:
    arms = make_arms()
    rewards = [0.1, 0.2, 0.05]
    curve = regret_curve(rewards, arms)
    oracle = 0.20
    expected = []
    running = 0.0
    for r in rewards:
        running += oracle - r
        expected.append(running)
    assert curve == expected


def test_regret_curve_returns_list() -> None:
    arms = make_arms()
    curve = regret_curve([0.1, 0.2], arms)
    assert isinstance(curve, list)


def test_regret_curve_handles_empty_rewards() -> None:
    arms = make_arms()
    assert regret_curve([], arms) == []


def test_compare_to_oracle_returns_expected_keys() -> None:
    arms = make_arms()
    result = compare_to_oracle(arms, [0, 1, 1, 1, 1])
    assert set(result) == {
        "oracle_arm", "oracle_payoff", "pulls", "oracle_pulls",
        "oracle_pull_rate", "average_regret_per_step",
    }
    assert result["oracle_arm"] == "B"
    assert result["oracle_payoff"] == 0.20
    assert result["pulls"] == 5
    assert result["oracle_pulls"] == 4
    assert result["oracle_pull_rate"] == 0.8


def test_compare_to_oracle_rejects_empty_selections() -> None:
    arms = make_arms()
    with pytest.raises(ValueError, match="selections"):
        compare_to_oracle(arms, [])


def test_compare_to_oracle_zero_oracle_pulls() -> None:
    arms = make_arms()
    result = compare_to_oracle(arms, [0, 0, 2, 2, 0])
    assert result["oracle_arm"] == "B"
    assert result["oracle_pulls"] == 0
    assert result["oracle_pull_rate"] == 0.0


def test_compare_to_oracle_average_regret() -> None:
    arms = make_arms()
    # If we always pull A (p=0.10) instead of B (p=0.20), per-step regret
    # is 0.10 every time; over 10 pulls that's 1.0.
    result = compare_to_oracle(arms, [0] * 10)
    assert result["average_regret_per_step"] == pytest.approx(0.1, abs=1e-6)