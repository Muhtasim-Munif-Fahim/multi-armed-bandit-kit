"""Tests for KL-UCB (Garivier & Cappé)."""

from __future__ import annotations

import math

import pytest

from bandit_kit import (
    BanditExperiment,
    BanditStep,
    BernoulliArm,
    ContextualBanditExperiment,
    GaussianArm,
    KLUCB,
    kl_ucb,
    make_linear_contextual_arms,
    run_experiment,
    summarize_runs,
)
from bandit_kit.algorithms import _bernoulli_kl, _kl_ucb_threshold, _kl_ucb_upper


def make_arms():
    return [
        BernoulliArm(name="A", p=0.10, seed=1),
        BernoulliArm(name="B", p=0.20, seed=2),
        BernoulliArm(name="C", p=0.05, seed=3),
    ]


def test_factory_returns_instance() -> None:
    algo = kl_ucb(c=0.0, seed=0)
    assert isinstance(algo, KLUCB)
    assert algo.c == 0.0


def test_bernoulli_kl_is_zero_on_the_diagonal() -> None:
    for p in (0.0, 0.25, 0.5, 0.8, 1.0):
        assert _bernoulli_kl(p, p) == pytest.approx(0.0)


def test_bernoulli_kl_matches_closed_forms() -> None:
    assert _bernoulli_kl(0.0, 0.3) == pytest.approx(-math.log(0.7))
    assert _bernoulli_kl(1.0, 0.4) == pytest.approx(-math.log(0.4))
    assert _bernoulli_kl(0.2, 0.0) == math.inf
    assert _bernoulli_kl(0.8, 1.0) == math.inf


def test_bernoulli_kl_is_positive_off_diagonal() -> None:
    assert _bernoulli_kl(0.2, 0.8) > 0.0
    assert _bernoulli_kl(0.9, 0.1) > 0.0


def test_kl_ucb_upper_is_at_least_the_mean() -> None:
    mu = 0.4
    upper = _kl_ucb_upper(mu, count=10, threshold=math.log(50), precision=1e-8)
    assert mu <= upper <= 1.0
    assert 10 * _bernoulli_kl(mu, upper) <= math.log(50) + 1e-6


def test_kl_ucb_upper_is_one_when_mean_is_one() -> None:
    assert _kl_ucb_upper(1.0, count=5, threshold=1.0, precision=1e-6) == pytest.approx(1.0)


def test_kl_ucb_upper_shrinks_with_more_counts() -> None:
    mu = 0.3
    threshold = math.log(100)
    wide = _kl_ucb_upper(mu, count=2, threshold=threshold, precision=1e-8)
    tight = _kl_ucb_upper(mu, count=40, threshold=threshold, precision=1e-8)
    assert tight < wide
    assert tight >= mu


def test_threshold_is_log_t_when_c_is_zero() -> None:
    assert _kl_ucb_threshold(1, 0.0) == pytest.approx(0.0)
    assert _kl_ucb_threshold(10, 0.0) == pytest.approx(math.log(10))


def test_threshold_grows_with_c() -> None:
    assert _kl_ucb_threshold(20, 3.0) > _kl_ucb_threshold(20, 0.0)


def test_pulls_each_arm_at_least_once() -> None:
    algo = kl_ucb(seed=42)
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
    algo = kl_ucb(seed=0)
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
    algo = kl_ucb(c=0.0, seed=0)
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


def test_index_is_mean_when_threshold_is_zero() -> None:
    algo = kl_ucb(c=0.0, seed=0)
    arms = make_arms()
    algo.reset(arms)
    for idx, reward in enumerate([0.1, 0.8, 0.2]):
        algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=reward, step=idx))
    # step=0 => t=1 => log(1)=0 => index equals the empirical mean.
    assert algo.upper_bound(0, 0) == pytest.approx(0.1)
    assert algo.upper_bound(1, 0) == pytest.approx(0.8)
    assert algo.upper_bound(2, 0) == pytest.approx(0.2)


def test_upper_bound_exceeds_mean_after_warmup() -> None:
    algo = kl_ucb(c=0.0, seed=0)
    arms = make_arms()
    algo.reset(arms)
    for idx, reward in enumerate([0.1, 0.2, 0.05]):
        algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=reward, step=idx))
    bound = algo.upper_bound(1, 10)
    assert bound > algo._values[1]
    assert bound <= 1.0


def test_update_increments_counts_and_means() -> None:
    algo = kl_ucb(seed=0)
    arms = make_arms()
    algo.reset(arms)
    algo.update(arms, BanditStep(arm_index=1, arm_name="B", reward=1.0, step=0))
    algo.update(arms, BanditStep(arm_index=1, arm_name="B", reward=0.0, step=1))
    assert algo._counts[1] == 2
    assert algo._values[1] == pytest.approx(0.5)
    assert algo._counts[0] == 0


def test_clips_rewards_outside_unit_interval() -> None:
    algo = kl_ucb(seed=0)
    arms = [GaussianArm(name="G1", mean=2.0, std=0.5), GaussianArm(name="G2", mean=-1.0, std=0.5)]
    algo.reset(arms)
    algo.update(arms, BanditStep(arm_index=0, arm_name="G1", reward=5.0, step=0))
    algo.update(arms, BanditStep(arm_index=1, arm_name="G2", reward=-3.0, step=1))
    assert algo._values[0] == pytest.approx(1.0)
    assert algo._values[1] == pytest.approx(0.0)


def test_handles_gaussian_arms() -> None:
    algo = kl_ucb(seed=3)
    arms = [GaussianArm(name="G1", mean=0.2, std=0.1), GaussianArm(name="G2", mean=0.8, std=0.1)]
    algo.reset(arms)
    idx = algo.select_arm(arms, 0)
    assert idx in {0, 1}
    algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=arms[idx].draw(), step=0))


def test_rejects_invalid_c() -> None:
    with pytest.raises(ValueError, match="c"):
        KLUCB(c=-0.1)


def test_rejects_invalid_precision() -> None:
    with pytest.raises(ValueError, match="precision"):
        KLUCB(precision=0.0)
    with pytest.raises(ValueError, match="precision"):
        KLUCB(precision=-1e-6)


def test_upper_bound_rejects_negative_step() -> None:
    algo = kl_ucb(seed=0)
    arms = make_arms()
    algo.reset(arms)
    with pytest.raises(ValueError, match="step"):
        algo.upper_bound(0, -1)


def test_in_algorithm_registry() -> None:
    from bandit_kit.experiment import available_algorithms

    assert "kl_ucb" in available_algorithms()


def test_experiment_runs_kl_ucb() -> None:
    arms = make_arms()
    experiment, results = run_experiment(
        arms=arms, algorithms=("kl_ucb",), steps=25, runs=3, seed=42, kl_ucb_c=3.0
    )
    assert experiment.kl_ucb_c == 3.0
    assert len(results) == 3
    for result in results:
        assert result.algorithm == "kl_ucb"
        assert len(result.steps) == 25
        assert all(step.arm_index in {0, 1, 2} for step in result.steps)
    summary = summarize_runs(arms, results)
    assert len(summary) == 1
    assert summary[0]["algorithm"] == "kl_ucb"


def test_experiment_rejects_invalid_c() -> None:
    with pytest.raises(ValueError, match="kl_ucb_c"):
        BanditExperiment(arms=make_arms(), algorithms=["kl_ucb"], kl_ucb_c=-1.0)


def test_contextual_experiment_runs_kl_ucb_as_baseline() -> None:
    arms = make_linear_contextual_arms(n_arms=3, dimension=2, seed=0)
    experiment = ContextualBanditExperiment(
        arms=arms,
        algorithms=["kl_ucb"],
        steps=15,
        runs=2,
        seed=1,
        kl_ucb_c=0.0,
    )
    results = experiment.run()
    assert experiment.kl_ucb_c == 0.0
    assert len(results) == 2
    for result in results:
        assert result.algorithm == "kl_ucb"
        assert len(result.steps) == 15
        assert all(step.context is not None for step in result.steps)
    summary = experiment.summarize(results)
    assert summary[0]["algorithm"] == "kl_ucb"
    assert summary[0]["kl_ucb_c"] == 0.0


def test_same_seed_is_deterministic() -> None:
    algo_a = kl_ucb(c=0.0, seed=11)
    algo_b = kl_ucb(c=0.0, seed=11)
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


def test_higher_c_explores_more() -> None:
    def _run(c: float) -> dict:
        algo = kl_ucb(c=c, seed=42)
        arms = make_arms()
        algo.reset(arms)
        counts = {0: 0, 1: 0, 2: 0}
        for step in range(200):
            idx = algo.select_arm(arms, step)
            counts[idx] += 1
            algo.update(
                arms,
                BanditStep(
                    arm_index=idx,
                    arm_name=arms[idx].name,
                    reward=arms[idx].draw(),
                    step=step,
                ),
            )
        return counts

    counts_low = _run(0.0)
    counts_high = _run(5.0)
    exploration_low = (counts_low[0] + counts_low[2]) / 200
    exploration_high = (counts_high[0] + counts_high[2]) / 200
    assert exploration_high >= exploration_low * 0.5
