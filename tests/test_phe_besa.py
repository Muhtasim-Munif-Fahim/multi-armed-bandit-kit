"""Tests for Perturbed-History Exploration (PHE) and BESA."""

from __future__ import annotations

import statistics

import pytest

from bandit_kit import (
    BESA,
    PHE,
    BanditExperiment,
    BanditStep,
    BernoulliArm,
    GaussianArm,
    available_algorithms,
    besa,
    phe,
)
from bandit_kit.algorithms import BanditAlgorithm


def make_arms(seed=0):
    return [
        BernoulliArm(name="A", p=0.30, seed=seed + 1),
        BernoulliArm(name="B", p=0.70, seed=seed + 2),
        BernoulliArm(name="C", p=0.50, seed=seed + 3),
        BernoulliArm(name="D", p=0.20, seed=seed + 4),
    ]


def _drive(algo, arms, steps):
    algo.reset(arms)
    chosen = []
    for t in range(steps):
        idx = algo.select_arm(arms, t)
        reward = arms[idx].draw()
        algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=reward, step=t))
        chosen.append(idx)
    return chosen


class _Uniform(BanditAlgorithm):
    name = "uniform"

    def select_arm(self, arms, step):
        return self._rng.randrange(len(arms))

    def update(self, arms, step):
        pass


def _regret(chosen, arms):
    best = max(a.expected_value for a in arms)
    return sum(best - arms[i].expected_value for i in chosen)


def test_registry_and_factories():
    assert {"phe", "besa"} <= set(available_algorithms())
    assert isinstance(phe(perturbation_scale=1.5, seed=0), PHE)
    assert phe(1.5).perturbation_scale == 1.5
    assert isinstance(besa(seed=0), BESA)


@pytest.mark.parametrize("bad", [0, -1.0, float("inf"), True, "2"])
def test_phe_rejects_bad_scale(bad):
    with pytest.raises(ValueError):
        PHE(perturbation_scale=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("factory", [phe, besa])
def test_pulls_every_arm_once_first(factory):
    arms = make_arms()
    chosen = _drive(factory(seed=0), arms, 4)
    assert chosen == [0, 1, 2, 3]


@pytest.mark.parametrize("factory", [phe, besa])
def test_converges_to_best_arm_and_beats_uniform(factory):
    regrets, uniform = [], []
    for seed in range(6):
        arms = make_arms(10 * seed)
        chosen = _drive(factory(seed=seed), arms, 1500)
        assert chosen[-500:].count(1) / 500 > 0.85
        regrets.append(_regret(chosen, arms))
        uniform.append(_regret(_drive(_Uniform(seed=seed), make_arms(10 * seed), 1500), arms))
    assert statistics.mean(regrets) < 0.25 * statistics.mean(uniform)


def test_phe_binomial_half_and_scaling():
    algo = PHE(seed=3)
    draws = [algo._binomial_half(100) for _ in range(4000)]
    assert statistics.mean(draws) == pytest.approx(50.0, abs=0.5)
    assert statistics.pvariance(draws) == pytest.approx(25.0, rel=0.1)
    assert algo._binomial_half(0) == 0
    arms = [GaussianArm("g", 0.0, 1.0, seed=1), GaussianArm("h", 1.0, 1.0, seed=2)]
    algo.reset(arms)
    # shared range [-4, 5] across both arms
    assert algo._scaled(0, 0.5) == pytest.approx(0.5)
    assert algo._scaled(1, 0.5) == pytest.approx(0.5)
    assert algo._scaled(0, 100.0) == 1.0
    assert algo._scaled(0, -100.0) == 0.0
    chosen = _drive(PHE(seed=0), arms, 800)
    assert chosen[-300:].count(1) / 300 > 0.9


def test_phe_scores_are_perturbed_means():
    arms = make_arms()
    algo = PHE(perturbation_scale=1.0, seed=0)
    _drive(algo, arms, 40)
    assert all(0.0 <= s <= 1.0 for s in algo.last_scores)


def test_besa_duel_subsamples_and_breaks_ties_to_less_pulled():
    algo = BESA(seed=0)
    algo.reset(make_arms())
    algo._history[0] = [1.0, 0.0]
    algo._history[1] = [1.0] * 10
    assert algo.duel(0, 1) == 1  # 0.5 vs 1.0
    algo._history[1] = [0.0] * 5 + [1.0] * 5
    wins = [algo.duel(0, 1) for _ in range(2000)]
    # sub-sample of size 2 from arm 1 has mean > 0.5 w.p. 2/9, == 0.5 w.p. 5/9 (tie -> arm 0)
    assert wins.count(1) / 2000 == pytest.approx(20 / 90, abs=0.03)
    algo._history[2] = []
    assert algo.duel(2, 0) == 2
    assert algo.counts == [2, 10, 0, 0]


def test_experiment_harness_runs_both():
    exp = BanditExperiment(arms=make_arms(), algorithms=["phe", "besa"], steps=80, runs=2)
    results = exp.run()
    assert len(results) == 4 and all(len(r.steps) == 80 for r in results)
