"""Tests for Boltzmann / softmax exploration."""

from __future__ import annotations

import pytest

from bandit_kit import (
    BanditExperiment,
    BanditStep,
    BernoulliArm,
    Boltzmann,
    ContextualBanditExperiment,
    GaussianArm,
    Softmax,
    boltzmann,
    make_linear_contextual_arms,
    run_experiment,
    softmax,
    summarize_runs,
)


def make_arms():
    return [
        BernoulliArm(name="A", p=0.10, seed=1),
        BernoulliArm(name="B", p=0.20, seed=2),
        BernoulliArm(name="C", p=0.05, seed=3),
    ]


def test_factory_returns_instance() -> None:
    algo = boltzmann(temperature_start=1.0, seed=0)
    assert isinstance(algo, Boltzmann)


def test_softmax_alias_returns_same_class() -> None:
    algo = softmax(temperature_start=1.0, seed=0)
    assert isinstance(algo, Boltzmann)
    assert Softmax is Boltzmann


def test_probabilities_sum_to_one() -> None:
    algo = boltzmann(seed=0)
    arms = make_arms()
    algo.reset(arms)
    for idx, reward in enumerate([0.1, 0.2, 0.05]):
        algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=reward, step=idx))
    probabilities = algo.probabilities_at(3)
    assert len(probabilities) == len(arms)
    assert all(0.0 <= p <= 1.0 for p in probabilities)
    assert sum(probabilities) == pytest.approx(1.0)


def test_equal_values_give_uniform_probabilities() -> None:
    algo = boltzmann(temperature_start=1.0, temperature_min=1.0, decay=0.99, seed=0)
    arms = make_arms()
    algo.reset(arms)
    for idx in range(3):
        algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=0.0, step=idx))
    probabilities = algo.probabilities_at(3)
    for prob in probabilities:
        assert prob == pytest.approx(1.0 / len(arms), abs=1e-9)


def test_low_temperature_concentrates_on_best_value() -> None:
    algo = boltzmann(temperature_start=1.0, temperature_min=0.01, decay=0.5, seed=0)
    arms = make_arms()
    algo.reset(arms)
    for idx, reward in enumerate([0.1, 0.9, 0.1]):
        algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=reward, step=idx))
    probabilities = algo._softmax_probabilities(0.01)
    assert probabilities[1] > 0.99
    assert probabilities[1] > probabilities[0]
    assert probabilities[1] > probabilities[2]


def test_high_temperature_is_nearly_uniform() -> None:
    algo = boltzmann(temperature_start=100.0, temperature_min=100.0, decay=0.99, seed=0)
    arms = make_arms()
    algo.reset(arms)
    for idx, reward in enumerate([0.1, 0.9, 0.1]):
        algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=reward, step=idx))
    probabilities = algo._softmax_probabilities(100.0)
    for prob in probabilities:
        assert prob == pytest.approx(1.0 / 3.0, abs=0.02)


def test_temperature_decays_with_step() -> None:
    algo = boltzmann(temperature_start=1.0, temperature_min=0.05, decay=0.99, seed=0)
    t0 = algo.temperature_at(0)
    t10 = algo.temperature_at(10)
    t100 = algo.temperature_at(100)
    assert t0 == pytest.approx(1.0)
    assert 0.05 <= t100 < t10 < t0


def test_temperature_floors_at_min() -> None:
    algo = boltzmann(temperature_start=0.5, temperature_min=0.05, decay=0.5, seed=0)
    assert algo.temperature_at(0) == pytest.approx(0.5)
    for step in range(50, 200):
        assert algo.temperature_at(step) >= 0.05 - 1e-9


def test_constant_temperature_when_start_equals_min() -> None:
    algo = boltzmann(temperature_start=0.4, temperature_min=0.4, decay=0.5, seed=0)
    for step in (0, 10, 100):
        assert algo.temperature_at(step) == pytest.approx(0.4)


def test_pulls_each_arm_at_least_once() -> None:
    algo = boltzmann(seed=42)
    arms = make_arms()
    algo.reset(arms)
    selections = []
    for step in range(len(arms)):
        idx = algo.select_arm(arms, step)
        selections.append(idx)
        algo.update(
            arms,
            BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=arms[idx].draw(), step=step),
        )
    assert sorted(selections) == [0, 1, 2]


def test_select_arm_returns_valid_index() -> None:
    algo = boltzmann(seed=0)
    arms = make_arms()
    algo.reset(arms)
    for step in range(20):
        idx = algo.select_arm(arms, step)
        assert 0 <= idx < len(arms)
        algo.update(
            arms,
            BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=arms[idx].draw(), step=step),
        )


def test_prefers_best_arm_over_horizon() -> None:
    algo = boltzmann(temperature_start=0.5, temperature_min=0.05, decay=0.99, seed=0)
    arms = make_arms()
    algo.reset(arms)
    counts = {0: 0, 1: 0, 2: 0}
    for step in range(400):
        idx = algo.select_arm(arms, step)
        counts[idx] += 1
        algo.update(
            arms,
            BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=arms[idx].draw(), step=step),
        )
    assert counts[1] > counts[0]
    assert counts[1] > counts[2]


def test_zero_temperature_min_is_greedy_after_warmup() -> None:
    algo = boltzmann(temperature_start=1.0, temperature_min=0.0, decay=0.1, seed=0)
    arms = make_arms()
    algo.reset(arms)
    for idx, reward in enumerate([0.1, 0.8, 0.2]):
        algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=reward, step=idx))
    # After enough decay, temperature is numerically zero -> argmax of Q.
    probabilities = algo.probabilities_at(50)
    assert probabilities[1] == pytest.approx(1.0)
    assert probabilities[0] == pytest.approx(0.0)
    assert probabilities[2] == pytest.approx(0.0)


def test_rejects_invalid_temperature_start() -> None:
    with pytest.raises(ValueError, match="temperature_start"):
        Boltzmann(temperature_start=0.0)
    with pytest.raises(ValueError, match="temperature_start"):
        Boltzmann(temperature_start=-1.0)


def test_rejects_invalid_temperature_min() -> None:
    with pytest.raises(ValueError, match="temperature_min"):
        Boltzmann(temperature_start=1.0, temperature_min=-0.1)
    with pytest.raises(ValueError, match="temperature_min"):
        Boltzmann(temperature_start=0.2, temperature_min=0.5)


def test_rejects_invalid_decay() -> None:
    with pytest.raises(ValueError, match="decay"):
        Boltzmann(temperature_start=1.0, decay=0.0)
    with pytest.raises(ValueError, match="decay"):
        Boltzmann(temperature_start=1.0, decay=1.0)
    with pytest.raises(ValueError, match="decay"):
        Boltzmann(temperature_start=1.0, decay=-0.1)


def test_temperature_at_rejects_negative_step() -> None:
    algo = boltzmann(seed=0)
    with pytest.raises(ValueError, match="step"):
        algo.temperature_at(-1)
    with pytest.raises(ValueError, match="step"):
        algo.probabilities_at(-1)


def test_handles_gaussian_rewards() -> None:
    algo = boltzmann(seed=3)
    arms = [GaussianArm(name="G1", mean=1.0, std=0.5), GaussianArm(name="G2", mean=2.0, std=0.5)]
    algo.reset(arms)
    idx = algo.select_arm(arms, 0)
    assert idx in {0, 1}
    algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=arms[idx].draw(), step=0))


def test_in_algorithm_registry() -> None:
    from bandit_kit.experiment import available_algorithms

    names = available_algorithms()
    assert "boltzmann" in names
    assert "softmax" in names


def test_experiment_runs_boltzmann() -> None:
    arms = make_arms()
    experiment, results = run_experiment(
        arms=arms,
        algorithms=("boltzmann",),
        steps=25,
        runs=3,
        seed=42,
        temperature_start=0.8,
        temperature_min=0.1,
        temperature_decay=0.95,
    )
    assert experiment.temperature_start == 0.8
    assert experiment.temperature_min == 0.1
    assert experiment.temperature_decay == 0.95
    assert len(results) == 3
    for result in results:
        assert result.algorithm == "boltzmann"
        assert len(result.steps) == 25
        assert all(step.arm_index in {0, 1, 2} for step in result.steps)
    summary = summarize_runs(arms, results)
    assert len(summary) == 1
    assert summary[0]["algorithm"] == "boltzmann"


def test_experiment_runs_softmax_alias() -> None:
    arms = make_arms()
    experiment, results = run_experiment(
        arms=arms, algorithms=("softmax",), steps=10, runs=2, seed=7
    )
    assert len(results) == 2
    assert all(result.algorithm == "softmax" for result in results)


def test_experiment_rejects_invalid_temperature() -> None:
    with pytest.raises(ValueError, match="temperature_start"):
        BanditExperiment(arms=make_arms(), algorithms=["boltzmann"], temperature_start=0.0)
    with pytest.raises(ValueError, match="temperature_min"):
        BanditExperiment(
            arms=make_arms(),
            algorithms=["boltzmann"],
            temperature_start=0.2,
            temperature_min=0.5,
        )
    with pytest.raises(ValueError, match="temperature_decay"):
        BanditExperiment(arms=make_arms(), algorithms=["boltzmann"], temperature_decay=1.0)


def test_contextual_experiment_runs_boltzmann_as_baseline() -> None:
    arms = make_linear_contextual_arms(n_arms=3, dimension=2, seed=0)
    experiment = ContextualBanditExperiment(
        arms=arms,
        algorithms=["boltzmann"],
        steps=15,
        runs=2,
        seed=1,
        temperature_start=0.8,
        temperature_min=0.1,
        temperature_decay=0.95,
    )
    results = experiment.run()
    assert experiment.temperature_start == 0.8
    assert len(results) == 2
    for result in results:
        assert result.algorithm == "boltzmann"
        assert len(result.steps) == 15
        assert all(step.context is not None for step in result.steps)
    summary = experiment.summarize(results)
    assert summary[0]["algorithm"] == "boltzmann"
    assert summary[0]["temperature_start"] == 0.8


def test_same_seed_is_deterministic() -> None:
    algo_a = boltzmann(temperature_start=1.0, seed=11)
    algo_b = boltzmann(temperature_start=1.0, seed=11)
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
        algo_a.update(
            arms_a,
            BanditStep(arm_index=idx_a, arm_name=arms_a[idx_a].name, reward=reward_a, step=step),
        )
        algo_b.update(
            arms_b,
            BanditStep(arm_index=idx_b, arm_name=arms_b[idx_b].name, reward=reward_b, step=step),
        )
