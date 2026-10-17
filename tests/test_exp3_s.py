"""Tests for EXP3.S (switching-adversary EXP3 with fixed share)."""

from __future__ import annotations

import math

import pytest

from bandit_kit import (
    BanditExperiment,
    BernoulliArm,
    Exp3,
    Exp3S,
    exp3_s,
    exp3_s_parameters,
)
from bandit_kit.algorithms import BanditStep
from bandit_kit.cli import main


def _arms(n: int = 3):
    return [BernoulliArm(f"a{i}", 0.5, seed=i) for i in range(n)]


def test_factory_returns_exp3_s():
    algo = exp3_s(gamma=0.2, alpha=0.05, seed=0)
    assert isinstance(algo, Exp3S)
    assert algo.gamma == 0.2 and algo.alpha == 0.05


@pytest.mark.parametrize("kwargs, match", [
    ({"gamma": 0.0}, "gamma"),
    ({"gamma": 1.5}, "gamma"),
    ({"alpha": -0.1}, "alpha"),
    ({"alpha": 1.1}, "alpha"),
])
def test_rejects_invalid_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Exp3S(**kwargs)


def test_single_update_matches_closed_form():
    arms = _arms(3)
    algo = Exp3S(gamma=0.3, alpha=0.1, seed=0)
    algo.reset(arms)
    idx = algo.select_arm(arms, 0)
    p = algo._last_probabilities[idx]
    assert p == pytest.approx(1.0 / 3.0)
    algo.update(arms, BanditStep(arm_index=idx, arm_name="x", reward=1.0, step=0))
    share = math.e * 0.1 / 3 * 3.0
    expected = [1.0 + share] * 3
    expected[idx] = math.exp(0.3 * (1.0 / p) / 3) + share
    top = max(expected)
    assert algo._weights == pytest.approx([w / top for w in expected])


def test_alpha_zero_matches_exp3():
    arms = _arms(4)
    a = Exp3S(gamma=0.2, alpha=0.0, seed=7)
    b = Exp3(gamma=0.2, seed=7)
    a.reset(arms)
    b.reset(arms)
    for t in range(60):
        ia = a.select_arm(arms, t)
        ib = b.select_arm(arms, t)
        assert ia == ib
        reward = 1.0 if ia == t % 4 else 0.0
        a.update(arms, BanditStep(arm_index=ia, arm_name="x", reward=reward, step=t))
        b.update(arms, BanditStep(arm_index=ib, arm_name="x", reward=reward, step=t))
        assert a._distribution() == pytest.approx(b._distribution())


def _post_switch_mass(algo, steps_before=400, steps_after=150):
    arms = _arms(2)
    algo.reset(arms)
    for t in range(steps_before + steps_after):
        best = 0 if t < steps_before else 1
        i = algo.select_arm(arms, t)
        algo.update(arms, BanditStep(arm_index=i, arm_name="x", reward=1.0 if i == best else 0.0, step=t))
    return algo._distribution()[1]


def test_tracks_switching_best_arm_faster_than_exp3():
    shared = sum(_post_switch_mass(Exp3S(gamma=0.1, alpha=0.01, seed=s)) for s in range(5)) / 5
    plain = sum(_post_switch_mass(Exp3(gamma=0.1, seed=s)) for s in range(5)) / 5
    # After the switch EXP3's weight on the old arm is astronomically larger,
    # so it stays stuck; the fixed share lets EXP3.S recover within ~100 steps.
    assert shared > 0.6
    assert plain < 0.2


def test_weights_stay_bounded_away_from_zero():
    arms = _arms(3)
    algo = Exp3S(gamma=0.1, alpha=0.05, seed=3)
    algo.reset(arms)
    for t in range(2000):
        i = algo.select_arm(arms, t)
        algo.update(arms, BanditStep(arm_index=i, arm_name="x", reward=1.0 if i == 0 else 0.0, step=t))
    assert min(algo._weights) > 0.0
    assert abs(sum(algo._distribution()) - 1.0) < 1e-12


def test_tuned_parameters():
    gamma, alpha = exp3_s_parameters(n_arms=5, horizon=10_000, switches=3)
    expected = math.sqrt(5 * (3 * math.log(5 * 10_000) + math.e) / ((math.e - 1) * 10_000))
    assert gamma == pytest.approx(expected)
    assert alpha == pytest.approx(1e-4)
    assert exp3_s_parameters(n_arms=10, horizon=5, switches=10)[0] == 1.0
    with pytest.raises(ValueError, match="horizon"):
        exp3_s_parameters(n_arms=2, horizon=0)
    with pytest.raises(ValueError, match="switches"):
        exp3_s_parameters(n_arms=2, horizon=10, switches=-1)


def test_experiment_runs_exp3_s_and_exp3_ix():
    exp = BanditExperiment(
        arms=[BernoulliArm("a", 0.2), BernoulliArm("b", 0.7)],
        algorithms=["exp3_s", "exp3_ix", "exp3"],
        steps=50,
        runs=3,
        seed=1,
        exp3_s_alpha=0.02,
        exp3_ix_eta=0.2,
    )
    summary = exp.summarize(exp.run())
    assert {row["algorithm"] for row in summary} == {"exp3_s", "exp3_ix", "exp3"}
    with pytest.raises(ValueError, match="exp3_s_alpha"):
        BanditExperiment(arms=[BernoulliArm("a", 0.2)], algorithms=["exp3_s"], exp3_s_alpha=2.0)
    with pytest.raises(ValueError, match="exp3_ix_eta"):
        BanditExperiment(arms=[BernoulliArm("a", 0.2)], algorithms=["exp3_ix"], exp3_ix_eta=0.0)


def test_cli_compare_includes_exp3_s(tmp_path, capsys):
    out = tmp_path / "report.md"
    code = main([
        "compare", "--arms", "bern:0.2,bern:0.6", "--steps", "30", "--runs", "2",
        "--exp3-s-alpha", "0.05", "--output", str(out),
    ])
    assert code == 0
    text = out.read_text()
    assert "exp3_s" in text and "exp3_ix" in text
    assert "alpha (exp3_s)" in text or "gamma / alpha (exp3_s)" in text


def test_cli_rejects_invalid_alpha(capsys):
    code = main(["compare", "--arms", "bern:0.2,bern:0.6", "--steps", "5", "--runs", "1", "--exp3-s-alpha", "3"])
    assert code == 2
    assert "exp3_s_alpha" in capsys.readouterr().err
