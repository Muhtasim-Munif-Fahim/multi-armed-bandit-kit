"""Tests for Gaussian Thompson sampling."""

from __future__ import annotations

import math

import pytest

from bandit_kit import (
    BanditExperiment,
    BanditStep,
    GaussianArm,
    GaussianThompson,
    ThompsonGaussian,
    available_algorithms,
    run_experiment,
    thompson_gaussian,
    thompson_sampling_gaussian,
)


def make_gaussian_arms():
    return [
        GaussianArm(name="A", mean=0.0, std=1.0, seed=1),
        GaussianArm(name="B", mean=1.5, std=1.0, seed=2),
        GaussianArm(name="C", mean=0.3, std=1.0, seed=3),
    ]


def test_factory_and_exports() -> None:
    algo = thompson_sampling_gaussian(mu0=0.0, tau0=1.0, sigma=1.0, seed=0)
    assert isinstance(algo, GaussianThompson)
    assert algo is not None
    assert ThompsonGaussian is GaussianThompson
    assert isinstance(thompson_gaussian(seed=1), GaussianThompson)
    assert "thompson_gaussian" in available_algorithms()
    assert "gaussian_thompson" in available_algorithms()


def test_rejects_bad_hyperparameters() -> None:
    with pytest.raises(ValueError):
        thompson_sampling_gaussian(tau0=0.0)
    with pytest.raises(ValueError):
        thompson_sampling_gaussian(tau0=-1.0)
    with pytest.raises(ValueError):
        thompson_sampling_gaussian(sigma=0.0)
    with pytest.raises(ValueError):
        GaussianThompson(sigma=-0.5)


def test_prior_posterior_before_observations() -> None:
    algo = GaussianThompson(mu0=0.5, tau0=2.0, sigma=1.0, seed=0)
    arms = make_gaussian_arms()
    algo.reset(arms)
    assert algo.posterior_mean(0) == pytest.approx(0.5)
    assert algo.posterior_variance(0) == pytest.approx(4.0)
    assert algo.posterior_precision(0) == pytest.approx(0.25)


def test_conjugate_update_matches_formula() -> None:
    mu0, tau0, sigma = 0.0, 1.0, 1.0
    algo = GaussianThompson(mu0=mu0, tau0=tau0, sigma=sigma, seed=0)
    arms = make_gaussian_arms()
    algo.reset(arms)
    rewards = [1.0, 2.0, 0.0]
    for step, r in enumerate(rewards):
        algo.update(arms, BanditStep(arm_index=0, arm_name="A", reward=r, step=step))
    n = len(rewards)
    total = sum(rewards)
    prior_prec = 1.0 / (tau0 * tau0)
    obs_prec = 1.0 / (sigma * sigma)
    post_prec = prior_prec + n * obs_prec
    expected_mean = (prior_prec * mu0 + total * obs_prec) / post_prec
    expected_var = 1.0 / post_prec
    assert algo.posterior_mean(0) == pytest.approx(expected_mean)
    assert algo.posterior_variance(0) == pytest.approx(expected_var)
    assert algo.posterior_precision(0) == pytest.approx(post_prec)


def test_select_arm_is_seeded() -> None:
    arms = make_gaussian_arms()
    a1 = thompson_sampling_gaussian(seed=7)
    a2 = thompson_sampling_gaussian(seed=7)
    a1.reset(arms)
    a2.reset(arms)
    # Feed identical history
    for step in range(5):
        a1.update(arms, BanditStep(arm_index=step % 3, arm_name="x", reward=0.5, step=step))
        a2.update(arms, BanditStep(arm_index=step % 3, arm_name="x", reward=0.5, step=step))
    chosen1 = [a1.select_arm(arms, step) for step in range(10, 20)]
    chosen2 = [a2.select_arm(arms, step) for step in range(10, 20)]
    assert chosen1 == chosen2


def test_prefers_better_gaussian_arm() -> None:
    experiment, results = run_experiment(
        make_gaussian_arms(),
        algorithms=["thompson_gaussian"],
        steps=400,
        runs=20,
        seed=11,
        thompson_gaussian_sigma=1.0,
    )
    summary = experiment.summarize(results)[0]
    fractions = summary["mean_arm_selection_fraction"]
    assert fractions["B"] > fractions["A"]
    assert fractions["B"] > fractions["C"]


def test_experiment_accepts_thompson_gaussian() -> None:
    experiment = BanditExperiment(
        arms=make_gaussian_arms(),
        algorithms=["thompson_gaussian", "ucb1"],
        steps=40,
        runs=2,
        seed=1,
    )
    results = experiment.run()
    assert len(results) == 4
    assert {r.algorithm for r in results} == {"thompson_gaussian", "ucb1"}


def test_posterior_helpers_reject_bad_index() -> None:
    algo = thompson_sampling_gaussian(seed=0)
    arms = make_gaussian_arms()
    algo.reset(arms)
    with pytest.raises(IndexError):
        algo.posterior_mean(9)
    with pytest.raises(IndexError):
        algo.posterior_variance(-1)
