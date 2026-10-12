"""Tests for best-arm identification (Successive Rejects, Sequential Halving, LUCB)."""

from __future__ import annotations

import json
import math

import pytest

from bandit_kit import (
    BernoulliArm,
    GaussianArm,
    identification_error_rate,
    lucb,
    successive_halving,
    successive_rejects,
    successive_rejects_schedule,
)
from bandit_kit.cli import main


def _bern(ps, seed=0):
    return [BernoulliArm(f"a{i}", p, seed=seed + i) for i, p in enumerate(ps)]


def test_sr_schedule_matches_formula_and_budget():
    k, n = 4, 1000
    logbar = 0.5 + 1 / 2 + 1 / 3 + 1 / 4
    expected = [math.ceil((n - k) / (logbar * (k + 1 - j))) for j in range(1, k)]
    assert successive_rejects_schedule(k, n) == expected
    result = successive_rejects(_bern([0.5, 0.4, 0.3, 0.2]), n)
    sched = expected
    # Survivors of phase j get n_j pulls; the winner gets n_{K-1}.
    assert result.total_pulls == sum(sched) + sched[-1]
    assert result.total_pulls <= n
    assert len(result.elimination_order) == k - 1
    assert result.pulls[result.recommended] == sched[-1]


def test_sr_and_sh_find_an_obvious_best_arm():
    arms = _bern([0.2, 0.9, 0.1, 0.3, 0.25])
    for method in (successive_rejects, successive_halving):
        res = method(arms, 500)
        assert res.recommended == "a1"
        assert res.is_correct(arms)
        assert res.total_pulls <= 500


def test_sh_rounds_and_budget():
    arms = _bern([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    res = successive_halving(arms, 2400)
    # 8 arms -> 3 rounds of 100 pulls per arm: 8*100 + 4*200 + 2*400 = 2400.
    assert res.total_pulls == 2400
    assert res.pulls["a7"] == 100 + 200 + 400
    assert len(res.elimination_order) == 7
    with pytest.raises(ValueError, match="budget"):
        successive_halving(arms, 10)


def test_error_rate_falls_with_budget():
    arms = _bern([0.5, 0.45, 0.4, 0.4])
    small = identification_error_rate(successive_rejects, arms, n_runs=200, seed=1, budget=100)
    large = identification_error_rate(successive_rejects, arms, n_runs=200, seed=1, budget=3000)
    assert large["error_rate"] < small["error_rate"]
    assert large["error_rate"] < 0.15
    sh = identification_error_rate(successive_halving, arms, n_runs=200, seed=1, budget=3000)
    assert sh["error_rate"] < 0.2


def test_lucb_is_delta_correct_and_adapts_to_gap():
    easy = _bern([0.8, 0.2, 0.3])
    hard = _bern([0.6, 0.5, 0.45])
    easy_stats = identification_error_rate(lucb, easy, n_runs=50, seed=2, delta=0.1)
    hard_stats = identification_error_rate(lucb, hard, n_runs=20, seed=2, delta=0.1)
    assert easy_stats["error_rate"] <= 0.1
    assert hard_stats["error_rate"] <= 0.1
    assert hard_stats["mean_pulls"] > 5 * easy_stats["mean_pulls"]
    res = lucb(easy, delta=0.1)
    assert res.stopped and res.recommended == "a0"


def test_lucb_epsilon_and_cap():
    twins = _bern([0.5, 0.5], seed=3)
    capped = lucb(twins, delta=0.05, max_pulls=500)
    assert not capped.stopped and capped.total_pulls <= 500
    relaxed = lucb(twins, delta=0.05, epsilon=0.2, max_pulls=100_000)
    assert relaxed.stopped and relaxed.total_pulls < 100_000


def test_lucb_scales_radius_to_reward_range():
    arms = [GaussianArm("g0", 1.0, 0.5, seed=1), GaussianArm("g1", 0.0, 0.5, seed=2)]
    res = lucb(arms, delta=0.05)
    assert res.stopped and res.recommended == "g0"


def test_validation():
    with pytest.raises(ValueError, match="at least two"):
        successive_rejects(_bern([0.5]), 10)
    with pytest.raises(ValueError, match="budget"):
        successive_rejects(_bern([0.5, 0.4, 0.3]), 2)
    with pytest.raises(ValueError, match="unique"):
        successive_rejects([BernoulliArm("x", 0.1), BernoulliArm("x", 0.2)], 10)
    with pytest.raises(ValueError, match="delta"):
        lucb(_bern([0.5, 0.4]), delta=0.0)
    with pytest.raises(ValueError, match="n_runs"):
        identification_error_rate(lucb, _bern([0.5, 0.4]), n_runs=0)


def test_cli_identify(capsys):
    code = main(["identify", "--arms", "bern:0.2,bern:0.8,bern:0.3", "--method", "sr",
                 "--budget", "600", "--runs", "20", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["recommended"] == "bern:0.8" and payload["correct"]
    assert payload["total_pulls"] <= 600
    assert payload["error_rate"]["n_runs"] == 20.0

    code = main(["identify", "--arms", "bern:0.9,bern:0.1", "--method", "lucb", "--delta", "0.1"])
    assert code == 0
    assert "recommended arm: bern:0.9" in capsys.readouterr().out

    assert main(["identify", "--arms", "bern:0.9,bern:0.1", "--budget", "1"]) == 2
