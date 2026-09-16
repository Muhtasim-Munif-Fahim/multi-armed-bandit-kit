"""Tests for Exp3 (adversarial bandit)."""

from __future__ import annotations

import pytest

from bandit_kit import (
    BanditExperiment,
    BanditStep,
    BernoulliArm,
    Exp3,
    exp3,
    run_experiment,
    summarize_runs,
)


def make_arms():
    return [
        BernoulliArm(name="A", p=0.10, seed=1),
        BernoulliArm(name="B", p=0.20, seed=2),
        BernoulliArm(name="C", p=0.05, seed=3),
    ]


def test_factory_returns_instance() -> None:
    algo = exp3(gamma=0.1, seed=0)
    assert isinstance(algo, Exp3)


def test_probabilities_sum_to_one() -> None:
    algo = exp3(seed=0)
    arms = make_arms()
    algo.reset(arms)
    probabilities = algo._distribution()
    assert len(probabilities) == len(arms)
    assert all(0.0 < p <= 1.0 for p in probabilities)
    assert sum(probabilities) == pytest.approx(1.0)


def test_equal_weights_give_uniform_probabilities() -> None:
    algo = exp3(gamma=0.3, seed=0)
    arms = make_arms()
    algo.reset(arms)
    probabilities = algo._distribution()
    for prob in probabilities:
        assert prob == pytest.approx(1.0 / len(arms), abs=1e-9)


def test_chosen_arm_weight_grows_on_high_reward() -> None:
    algo = exp3(gamma=0.5, seed=0)
    arms = make_arms()
    algo.reset(arms)
    algo.update(arms, BanditStep(arm_index=1, arm_name="B", reward=1.0, step=0))
    # After renormalization the chosen arm should be the heaviest.
    assert algo._weights[1] == max(algo._weights)
    assert algo._weights[1] == pytest.approx(1.0)


def test_unchosen_arm_weight_does_not_increase() -> None:
    algo = exp3(gamma=0.5, seed=0)
    arms = make_arms()
    algo.reset(arms)
    algo.update(arms, BanditStep(arm_index=1, arm_name="B", reward=1.0, step=0))
    # Unchosen arms keep their (renormalized) relative weight below the chosen arm.
    assert algo._weights[0] < algo._weights[1]
    assert algo._weights[2] < algo._weights[1]


def test_select_arm_returns_valid_index() -> None:
    algo = exp3(seed=0)
    arms = make_arms()
    algo.reset(arms)
    for step in range(20):
        idx = algo.select_arm(arms, step)
        assert 0 <= idx < len(arms)
        algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=arms[idx].draw(), step=step))


def test_prefers_best_arm_over_horizon() -> None:
    algo = exp3(gamma=0.1, seed=0)
    arms = make_arms()
    algo.reset(arms)
    counts = {0: 0, 1: 0, 2: 0}
    for step in range(400):
        idx = algo.select_arm(arms, step)
        counts[idx] += 1
        algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=arms[idx].draw(), step=step))
    assert counts[1] > counts[0]
    assert counts[1] > counts[2]


def test_gamma_one_samples_uniformly() -> None:
    algo = exp3(gamma=1.0, seed=0)
    arms = make_arms()
    algo.reset(arms)
    counts = {0: 0, 1: 0, 2: 0}
    for step in range(300):
        idx = algo.select_arm(arms, step)
        counts[idx] += 1
        # Always reward the best arm so a non-uniform policy would concentrate.
        reward = 1.0 if idx == 1 else 0.0
        algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=reward, step=step))
    # With gamma=1 the mixture is uniform regardless of weights.
    for count in counts.values():
        assert count == pytest.approx(100, abs=40)


def test_rejects_invalid_gamma() -> None:
    with pytest.raises(ValueError, match="gamma"):
        Exp3(gamma=0.0)
    with pytest.raises(ValueError, match="gamma"):
        Exp3(gamma=-0.1)
    with pytest.raises(ValueError, match="gamma"):
        Exp3(gamma=1.5)


def test_clips_rewards_outside_unit_interval() -> None:
    from bandit_kit import GaussianArm

    algo = exp3(gamma=0.2, seed=0)
    arms = [GaussianArm(name="G1", mean=2.0, std=0.5), GaussianArm(name="G2", mean=-1.0, std=0.5)]
    algo.reset(arms)
    idx = algo.select_arm(arms, 0)
    algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=5.0, step=0))
    assert all(weight > 0.0 for weight in algo._weights)
    algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=-3.0, step=1))
    assert all(weight > 0.0 for weight in algo._weights)


def test_in_algorithm_registry() -> None:
    from bandit_kit.experiment import available_algorithms

    assert "exp3" in available_algorithms()


def test_experiment_runs_exp3() -> None:
    arms = make_arms()
    experiment, results = run_experiment(
        arms=arms, algorithms=("exp3",), steps=25, runs=3, seed=42, gamma=0.2
    )
    assert experiment.gamma == 0.2
    assert len(results) == 3
    for result in results:
        assert result.algorithm == "exp3"
        assert len(result.steps) == 25
        assert all(step.arm_index in {0, 1, 2} for step in result.steps)
    summary = summarize_runs(arms, results)
    assert len(summary) == 1
    assert summary[0]["algorithm"] == "exp3"


def test_experiment_rejects_invalid_gamma() -> None:
    with pytest.raises(ValueError, match="gamma"):
        BanditExperiment(arms=make_arms(), algorithms=["exp3"], gamma=0.0)
    with pytest.raises(ValueError, match="gamma"):
        BanditExperiment(arms=make_arms(), algorithms=["exp3"], gamma=1.5)


def test_same_seed_is_deterministic() -> None:
    algo_a = exp3(gamma=0.1, seed=11)
    algo_b = exp3(gamma=0.1, seed=11)
    arms_a = make_arms()
    arms_b = make_arms()
    algo_a.reset(arms_a)
    algo_b.reset(arms_b)
    for step in range(30):
        idx_a = algo_a.select_arm(arms_a, step)
        idx_b = algo_b.select_arm(arms_b, step)
        assert idx_a == idx_b
        reward_a = arms_a[idx_a].draw()
        reward_b = arms_b[idx_b].draw()
        algo_a.update(arms_a, BanditStep(arm_index=idx_a, arm_name=arms_a[idx_a].name, reward=reward_a, step=step))
        algo_b.update(arms_b, BanditStep(arm_index=idx_b, arm_name=arms_b[idx_b].name, reward=reward_b, step=step))
