"""Tests for Tsallis-INF (Zimmert & Seldin, 2021)."""

from __future__ import annotations

import io
import math
from contextlib import redirect_stdout

import pytest

from bandit_kit import (
    BanditExperiment,
    BanditStep,
    BernoulliArm,
    TsallisINF,
    available_algorithms,
    run_experiment,
    tsallis_inf,
)
from bandit_kit.cli import main


def make_arms():
    return [
        BernoulliArm(name="A", p=0.20, seed=1),
        BernoulliArm(name="B", p=0.75, seed=2),
        BernoulliArm(name="C", p=0.40, seed=3),
    ]


def test_factory_and_registry() -> None:
    algo = tsallis_inf(seed=0)
    assert isinstance(algo, TsallisINF)
    assert algo.name == "tsallis_inf"
    assert algo.eta_scale == 2.0
    assert tsallis_inf(estimator="rv").eta_scale == 4.0
    assert "tsallis_inf" in available_algorithms()


@pytest.mark.parametrize(
    "kwargs",
    [{"estimator": "exp"}, {"eta_scale": 0.0}, {"eta_scale": -1.0},
     {"newton_tol": 0.0}, {"max_newton_iter": 0}],
)
def test_rejects_bad_params(kwargs) -> None:
    with pytest.raises(ValueError):
        TsallisINF(**kwargs)


def test_first_round_is_uniform_and_learning_rate() -> None:
    arms = make_arms()
    algo = TsallisINF()
    algo.reset(arms)
    assert algo.probabilities() == pytest.approx([1 / 3] * 3)
    assert algo.learning_rate(4) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        algo.learning_rate(0)
    with pytest.raises(RuntimeError):
        TsallisINF().probabilities()


def test_distribution_satisfies_tsallis_fixed_point() -> None:
    arms = make_arms()
    algo = TsallisINF()
    algo.reset(arms)
    algo._losses = [3.0, 0.5, 1.7]
    algo._t = 8
    probs = algo.probabilities()
    eta = algo.learning_rate(9)
    x = algo._x
    assert x < min(algo._losses)
    raw = [4.0 / (eta * (loss - x)) ** 2 for loss in algo._losses]
    assert sum(raw) == pytest.approx(1.0, abs=1e-9)
    assert probs == pytest.approx(raw, abs=1e-9)
    # lower cumulative loss -> higher probability
    assert probs[1] > probs[2] > probs[0]


def test_iw_update_is_importance_weighted() -> None:
    arms = make_arms()
    algo = TsallisINF(seed=0)
    algo.reset(arms)
    idx = algo.select_arm(arms, 0)
    p = algo._last_probabilities[idx]
    algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=0.25, step=0))
    expected = [0.0, 0.0, 0.0]
    expected[idx] = 0.75 / p
    assert algo._losses == pytest.approx(expected)
    assert algo._t == 1


def test_rv_without_baseline_early_matches_iw() -> None:
    arms = make_arms()
    algo = TsallisINF(estimator="rv", seed=0)
    algo.reset(arms)
    idx = algo.select_arm(arms, 0)
    p = algo._last_probabilities[idx]
    assert p < algo._last_eta ** 2  # eta_1 = 4, so B = 0 for every arm
    algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=0.0, step=0))
    expected = [0.0, 0.0, 0.0]
    expected[idx] = 1.0 / p
    assert algo._losses == pytest.approx(expected)


def test_rv_baseline_once_learning_rate_is_small() -> None:
    arms = make_arms()
    algo = TsallisINF(estimator="rv", seed=0)
    algo.reset(arms)
    algo._t = 999  # eta = 4 / sqrt(1000) ~ 0.126, eta^2 ~ 0.016 < 1/3
    idx = algo.select_arm(arms, 999)
    p = algo._last_probabilities[idx]
    algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=0.0, step=999))
    for i in range(3):
        if i == idx:
            assert algo._losses[i] == pytest.approx((1.0 - 0.5) / p + 0.5)
        else:
            assert algo._losses[i] == pytest.approx(0.5)


def test_rewards_are_clipped() -> None:
    arms = make_arms()
    algo = TsallisINF(seed=1)
    algo.reset(arms)
    algo.update(arms, BanditStep(arm_index=0, arm_name="A", reward=5.0, step=0))
    assert algo._losses[0] == pytest.approx(0.0)


@pytest.mark.parametrize("estimator", ["iw", "rv"])
def test_concentrates_on_best_arm(estimator) -> None:
    experiment, results = run_experiment(
        make_arms(), algorithms=["tsallis_inf"], steps=1500, runs=3, seed=0,
        tsallis_estimator=estimator,
    )
    assert isinstance(experiment, BanditExperiment)
    share = sum(
        sum(1 for s in run.steps[-500:] if s.arm_index == 1) / 500 for run in results
    ) / len(results)
    assert share > 0.7


def test_adversarial_switch_tracks_new_best_arm() -> None:
    """Rewards flip halfway; Tsallis-INF should move toward the new best arm."""
    names = ["A", "B"]
    algo = TsallisINF(seed=3)
    arms = [BernoulliArm(name=n, p=0.5, seed=i) for i, n in enumerate(names)]
    algo.reset(arms)
    late = 0
    horizon = 4000
    for t in range(horizon):
        best = 0 if t < horizon // 4 else 1
        idx = algo.select_arm(arms, t)
        reward = 1.0 if idx == best else 0.0
        algo.update(arms, BanditStep(arm_index=idx, arm_name=names[idx], reward=reward, step=t))
        if t >= horizon - 500:
            late += idx == 1
    assert late / 500 > 0.8
    assert all(math.isfinite(v) for v in algo._losses)


def test_experiment_rejects_bad_estimator() -> None:
    with pytest.raises(ValueError, match="tsallis_estimator"):
        BanditExperiment(arms=make_arms(), algorithms=["tsallis_inf"], tsallis_estimator="x")


def test_cli_compare_includes_tsallis() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(["compare", "--arms", "bern:0.2,bern:0.7", "--steps", "60",
                     "--runs", "2", "--tsallis-estimator", "rv"])
    assert code == 0
    assert "tsallis_inf" in buf.getvalue()
