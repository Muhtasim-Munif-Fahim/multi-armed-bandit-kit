"""Tests for Top-Two Thompson Sampling (Chernoff stopping) and lil'UCB."""

from __future__ import annotations

import json
import math

import pytest

from bandit_kit import (
    BernoulliArm,
    GaussianArm,
    chernoff_glr_statistic,
    identification_error_rate,
    lil_ucb,
    lucb,
    top_two_thompson,
)
from bandit_kit.cli import main


def _arms(ps, seed=0):
    return [BernoulliArm(f"v{i}", p, seed=seed + i) for i, p in enumerate(ps)]


def test_glr_statistic_matches_closed_forms():
    # Gaussian: pooled-KL form equals (ma - mb)^2 / (2 s^2 (1/na + 1/nb)).
    za = chernoff_glr_statistic(1.0, 10, 0.4, 30, model="gaussian", sigma=2.0)
    assert za == pytest.approx(0.36 / (2 * 4.0 * (1 / 10 + 1 / 30)))
    # Bernoulli: n_a kl(ma, m) + n_b kl(mb, m) with the pooled mean.
    ma, na, mb, nb = 0.6, 50, 0.4, 70
    m = (na * ma + nb * mb) / (na + nb)

    def kl(p, q):
        return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

    assert chernoff_glr_statistic(ma, na, mb, nb) == pytest.approx(na * kl(ma, m) + nb * kl(mb, m))
    assert chernoff_glr_statistic(0.3, 10, 0.5, 10) == 0.0  # wrong order
    assert chernoff_glr_statistic(0.5, 0, 0.3, 4) == 0.0
    assert chernoff_glr_statistic(1.0, 5, 0.0, 5) > 0  # edge means are clipped
    with pytest.raises(ValueError):
        chernoff_glr_statistic(0.5, 3, 0.2, 3, model="poisson")


@pytest.mark.parametrize("challenger", ["t3c", "ttts"])
def test_top_two_identifies_best_and_stops(challenger):
    arms = _arms([0.7, 0.4, 0.3])
    res = top_two_thompson(arms, delta=0.05, challenger=challenger, seed=1)
    assert res.stopped
    assert res.recommended == "v0" and res.is_correct(arms)
    assert res.method == f"top_two_thompson[{challenger}]"
    # beta = 0.5: the best arm gets roughly half of the pulls
    share = res.pulls["v0"] / res.total_pulls
    assert 0.3 < share < 0.7


def test_top_two_is_more_sample_efficient_than_lucb():
    arms = _arms([0.5, 0.4, 0.3])
    tt = identification_error_rate(top_two_thompson, arms, n_runs=20, seed=3, delta=0.05)
    lu = identification_error_rate(lucb, arms, n_runs=20, seed=3, delta=0.05)
    assert tt["error_rate"] <= 0.05
    assert tt["mean_pulls"] < 0.5 * lu["mean_pulls"]


def test_top_two_gaussian_model_and_reward_rescaling():
    arms = [GaussianArm("hi", 1.0, 1.0, seed=1), GaussianArm("lo", 0.0, 1.0, seed=2)]
    res = top_two_thompson(arms, model="gaussian", sigma=1.0, seed=4)
    assert res.stopped and res.recommended == "hi"
    from bandit_kit.arms import Arm

    flat = [Arm("a", 1.0, lambda: 1.0, reward_range=(1.0, 1.0)), Arm("b", 0.0, lambda: 0.0, reward_range=(0.0, 0.0))]
    with pytest.raises(ValueError, match="reward_range"):
        top_two_thompson(flat, model="bernoulli")


def test_top_two_respects_max_pulls_and_validation():
    arms = _arms([0.5, 0.5])
    res = top_two_thompson(arms, max_pulls=200, seed=0)
    assert not res.stopped and res.total_pulls == 200
    with pytest.raises(ValueError):
        top_two_thompson(arms, delta=0.0)
    with pytest.raises(ValueError):
        top_two_thompson(arms, beta=1.0)
    with pytest.raises(ValueError):
        top_two_thompson(arms, challenger="ucb")
    with pytest.raises(ValueError):
        top_two_thompson(arms, max_pulls=1)
    with pytest.raises(ValueError):
        top_two_thompson([arms[0]])


def test_top_two_is_seed_reproducible():
    a = top_two_thompson(_arms([0.6, 0.45]), seed=11)
    b = top_two_thompson(_arms([0.6, 0.45]), seed=11)
    assert a.pulls == b.pulls and a.total_pulls == b.total_pulls


def test_lil_ucb_identifies_best_with_dominant_pull_share():
    arms = _arms([0.8, 0.4, 0.3])
    res = lil_ucb(arms, delta=0.05)
    assert res.stopped and res.recommended == "v0"
    others = res.total_pulls - res.pulls["v0"]
    assert res.pulls["v0"] >= 1 + 9.0 * others  # lambda = ((2 + 1) / 1)^2
    rate = identification_error_rate(lil_ucb, _arms([0.5, 0.4, 0.3]), n_runs=20, seed=5)
    assert rate["error_rate"] <= 0.05


def test_lil_ucb_options_and_validation():
    arms = _arms([0.7, 0.3])
    loose = lil_ucb(arms, lam=1.0)
    tight = lil_ucb(_arms([0.7, 0.3]), lam=9.0)
    assert loose.total_pulls < tight.total_pulls
    capped = lil_ucb(_arms([0.5, 0.5]), max_pulls=100)
    assert not capped.stopped and capped.total_pulls == 100
    for bad in ({"delta": 1.0}, {"epsilon": 0.0}, {"beta": 0.0}, {"lam": -1.0}, {"max_pulls": 1}):
        with pytest.raises(ValueError):
            lil_ucb(arms, **bad)


def test_cli_identify_new_methods(capsys):
    assert main(["identify", "--arms", "bern:0.7,bern:0.3", "--method", "t3c", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["method"] == "top_two_thompson[t3c]" and payload["stopped"]
    assert main(["identify", "--arms", "bern:0.8,bern:0.3", "--method", "lil"]) == 0
    assert "method: lil_ucb" in capsys.readouterr().out
    assert main(["identify", "--arms", "gauss:1:1,gauss:0:1", "--method", "ttts",
                 "--model", "gaussian"]) == 0


def test_cli_identify_error_rate_with_seeded_method(capsys):
    argv = ["identify", "--arms", "bern:0.7,bern:0.3", "--method", "t3c", "--runs", "5", "--json"]
    assert main(argv) == 0
    first = json.loads(capsys.readouterr().out)["error_rate"]
    assert main(argv) == 0
    second = json.loads(capsys.readouterr().out)["error_rate"]
    assert first == second and first["n_runs"] == 5.0
