"""Unit tests for bandit_kit.arms."""

from __future__ import annotations

import pytest

from bandit_kit.arms import (
    Arm,
    BernoulliArm,
    GaussianArm,
    arm_from_spec,
    best_arm,
    expected_payoffs,
)


def test_bernoulli_arm_records_probability() -> None:
    arm = BernoulliArm(name="A", p=0.7)
    assert arm.expected_value == pytest.approx(0.7)
    assert arm.p == pytest.approx(0.7)
    assert arm.reward_range == (0.0, 1.0)


def test_bernoulli_arm_rejects_invalid_probability() -> None:
    with pytest.raises(ValueError):
        BernoulliArm(name="A", p=-0.1)
    with pytest.raises(ValueError):
        BernoulliArm(name="A", p=1.1)


def test_gaussian_arm_records_mean_and_std() -> None:
    arm = GaussianArm(name="G", mean=2.0, std=0.5)
    assert arm.expected_value == pytest.approx(2.0)
    assert arm.reward_range == (0.0, 4.0)


def test_gaussian_arm_rejects_non_positive_std() -> None:
    with pytest.raises(ValueError):
        GaussianArm(name="G", mean=0.0, std=0.0)


def test_arm_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        BernoulliArm(name="", p=0.5)


def test_arm_from_spec_parses_bernoulli() -> None:
    arm = arm_from_spec("bern:0.2")
    assert isinstance(arm, BernoulliArm)
    assert arm.expected_value == pytest.approx(0.2)


def test_arm_from_spec_parses_gaussian() -> None:
    arm = arm_from_spec("gauss:1.5:0.25")
    assert isinstance(arm, GaussianArm)
    assert arm.expected_value == pytest.approx(1.5)


def test_arm_from_spec_rejects_unknown_format() -> None:
    with pytest.raises(ValueError):
        arm_from_spec("exp:0.5")


def test_arm_from_spec_rejects_malformed_gaussian() -> None:
    with pytest.raises(ValueError):
        arm_from_spec("gauss:1.0")


def test_best_arm_returns_highest_expected_value() -> None:
    arms = [BernoulliArm(name="A", p=0.1), BernoulliArm(name="B", p=0.4), BernoulliArm(name="C", p=0.2)]
    assert best_arm(arms).name == "B"


def test_best_arm_breaks_ties_by_order() -> None:
    arms = [BernoulliArm(name="A", p=0.5), BernoulliArm(name="B", p=0.5)]
    assert best_arm(arms).name == "A"


def test_best_arm_rejects_empty_list() -> None:
    with pytest.raises(ValueError):
        best_arm([])


def test_expected_payoffs_maps_name_to_value() -> None:
    arms = [BernoulliArm(name="A", p=0.1), GaussianArm(name="G", mean=2.0, std=0.5)]
    payoffs = expected_payoffs(arms)
    assert payoffs == {"A": 0.1, "G": 2.0}


def test_arm_draw_returns_in_range() -> None:
    arm = BernoulliArm(name="A", p=0.5)
    samples = [arm.draw() for _ in range(50)]
    assert all(0.0 <= sample <= 1.0 for sample in samples)