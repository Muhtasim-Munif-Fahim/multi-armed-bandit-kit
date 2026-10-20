"""Tests for off-policy evaluation (IPS, SNIPS, DM, DR, replay)."""

from __future__ import annotations

import io
import json
import math
import statistics
from contextlib import redirect_stdout

import pytest

from bandit_kit import (
    BernoulliArm,
    GaussianArm,
    LoggedEvent,
    collect_logs,
    direct_method_estimate,
    doubly_robust_estimate,
    effective_sample_size,
    empirical_reward_model,
    evaluate_policy,
    importance_weights,
    ips_estimate,
    policy_probabilities,
    replay_evaluate,
    snips_estimate,
    true_policy_value,
    ucb1,
)
from bandit_kit.algorithms import BanditStep
from bandit_kit.cli import main


def _arms(ps, seed=0):
    return [BernoulliArm(f"a{i}", p, seed=seed * 101 + i) for i, p in enumerate(ps)]


HAND_LOGS = [
    LoggedEvent(0, 1.0, 0.5),
    LoggedEvent(1, 0.0, 0.25),
    LoggedEvent(2, 1.0, 0.25),
    LoggedEvent(1, 1.0, 0.25),
]


def test_hand_computed_ips_snips_and_weights():
    target = [0.2, 0.5, 0.3]
    w = importance_weights(HAND_LOGS, target)
    assert w == pytest.approx([0.4, 2.0, 1.2, 2.0])
    ips = ips_estimate(HAND_LOGS, target)
    assert ips.value == pytest.approx((0.4 + 0 + 1.2 + 2.0) / 4)
    terms = [0.4, 0.0, 1.2, 2.0]
    assert ips.std_error == pytest.approx(statistics.stdev(terms) / 2)
    assert ips.ci_low < ips.value < ips.ci_high
    snips = snips_estimate(HAND_LOGS, target)
    assert snips.value == pytest.approx(3.6 / 5.6)
    assert ips.effective_sample_size == pytest.approx(5.6**2 / (0.16 + 4 + 1.44 + 4))
    clipped = ips_estimate(HAND_LOGS, target, clip=1.0)
    assert clipped.value == pytest.approx((0.4 + 1.0 + 1.0) / 4)
    assert clipped.extra["clipped_fraction"] == pytest.approx(0.75)
    assert clipped.estimator == "clipped_ips"


def test_hand_computed_dm_and_dr():
    target = [0.2, 0.5, 0.3]
    model = empirical_reward_model(HAND_LOGS)
    assert model.means == pytest.approx([1.0, 0.5, 1.0])
    dm = direct_method_estimate(HAND_LOGS, target)
    assert dm.value == pytest.approx(0.2 * 1 + 0.5 * 0.5 + 0.3 * 1)
    assert dm.std_error == pytest.approx(0.0)
    dr = doubly_robust_estimate(HAND_LOGS, target)
    residual_terms = [0.4 * 0, 2.0 * -0.5, 1.2 * 0, 2.0 * 0.5]
    assert dr.value == pytest.approx(dm.value + sum(residual_terms) / 4)

    def constant(context, arm):
        return 0.5

    dr_const = doubly_robust_estimate(HAND_LOGS, target, reward_model=constant)
    expected = statistics.fmean(0.5 + wi * (e.reward - 0.5) for wi, e in zip(
        importance_weights(HAND_LOGS, target), HAND_LOGS))
    assert dr_const.value == pytest.approx(expected)
    assert dr_const.extra["dm_value"] == pytest.approx(0.5)


def test_ips_snips_dr_are_unbiased_under_skewed_logging():
    ps = [0.2, 0.5, 0.8]
    logging = [0.6, 0.3, 0.1]
    target = [0.1, 0.2, 0.7]
    truth = true_policy_value(_arms(ps), target)
    assert truth == pytest.approx(0.02 + 0.1 + 0.56)
    values = {"ips": [], "snips": [], "dr": [], "dm_wrong": [], "dr_wrong": []}

    def wrong(context, arm):
        return 0.5

    for rep in range(200):
        logs = collect_logs(_arms(ps, rep), 400, logging, seed=rep)
        values["ips"].append(ips_estimate(logs, target).value)
        values["snips"].append(snips_estimate(logs, target).value)
        values["dr"].append(doubly_robust_estimate(logs, target).value)
        values["dm_wrong"].append(direct_method_estimate(logs, target, reward_model=wrong).value)
        values["dr_wrong"].append(doubly_robust_estimate(logs, target, reward_model=wrong).value)
    for key in ("ips", "snips", "dr", "dr_wrong"):
        assert statistics.fmean(values[key]) == pytest.approx(truth, abs=0.012), key
    # a misspecified model biases DM, but DR corrects it via the propensities
    assert statistics.fmean(values["dm_wrong"]) == pytest.approx(0.5)
    # SNIPS is lower variance than IPS when weights are heavy (w up to 7)
    assert statistics.pstdev(values["snips"]) < statistics.pstdev(values["ips"])


def test_confidence_interval_coverage_is_near_nominal():
    ps = [0.3, 0.6]
    target = [0.0, 1.0]
    truth = 0.6
    covered = {"ips": 0, "snips": 0}
    for rep in range(300):
        logs = collect_logs(_arms(ps, rep), 300, [0.7, 0.3], seed=1000 + rep)
        for name, est in (("ips", ips_estimate(logs, target)), ("snips", snips_estimate(logs, target))):
            covered[name] += est.ci_low <= truth <= est.ci_high
    assert covered["ips"] / 300 >= 0.9
    assert covered["snips"] / 300 >= 0.9


def test_clipping_trades_bias_for_variance():
    ps = [0.4, 0.9]
    logging = [0.95, 0.05]
    target = 1
    raw, clip = [], []
    for rep in range(200):
        logs = collect_logs(_arms(ps, rep), 300, logging, seed=rep + 5)
        raw.append(ips_estimate(logs, target).value)
        clip.append(ips_estimate(logs, target, clip=5.0).value)
    assert statistics.pstdev(clip) < statistics.pstdev(raw)
    assert statistics.fmean(clip) < 0.9 - 0.3  # clipping at 5 of a weight of 20 biases down
    assert statistics.fmean(raw) == pytest.approx(0.9, abs=0.05)


def test_contextual_policy_and_reward_model():
    # context in {0, 1}; arm 1 is best when context == 1, arm 0 otherwise
    import random

    rng = random.Random(3)
    logs = []
    for _ in range(4000):
        x = rng.randint(0, 1)
        a = rng.randint(0, 1)
        p = 0.8 if (a == x) else 0.2
        logs.append(LoggedEvent(a, 1.0 if rng.random() < p else 0.0, 0.5, (x,)))

    def oracle(context):
        return [1.0, 0.0] if context[0] == 0 else [0.0, 1.0]

    def model(context, arm):
        return 0.8 if arm == context[0] else 0.2

    report = evaluate_policy(logs, oracle, reward_model=model)
    for name in ("ips", "snips", "dm", "dr"):
        assert report[name].value == pytest.approx(0.8, abs=0.03), name
    assert report["dm"].value == pytest.approx(0.8)
    assert "clipped_ips" not in report
    assert set(evaluate_policy(logs, oracle, reward_model=model, clip=1.5)) >= {"clipped_ips"}


def test_effective_sample_size_and_policy_resolution():
    assert effective_sample_size([1, 1, 1, 1]) == pytest.approx(4)
    assert effective_sample_size([4, 0, 0, 0]) == pytest.approx(1)
    assert effective_sample_size([0, 0]) == 0.0
    assert policy_probabilities(2, None, 3) == [0.0, 0.0, 1.0]
    assert policy_probabilities(lambda c: [0.5, 0.5], None, 2) == [0.5, 0.5]
    with pytest.raises(ValueError, match="sum to 1"):
        policy_probabilities([0.5, 0.6], None, 2)
    with pytest.raises(ValueError, match="out of range"):
        policy_probabilities(3, None, 3)
    with pytest.raises(ValueError, match="probabilities"):
        policy_probabilities([1.0], None, 2)
    with pytest.raises(ValueError):
        policy_probabilities(True, None, 2)


def test_validation_errors():
    with pytest.raises(ValueError, match="propensity"):
        LoggedEvent(0, 1.0, 0.0)
    with pytest.raises(ValueError, match="arm_index"):
        LoggedEvent(-1, 1.0, 0.5)
    with pytest.raises(ValueError, match="finite"):
        LoggedEvent(0, math.nan, 0.5)
    with pytest.raises(ValueError, match="empty"):
        ips_estimate([], [1.0])
    with pytest.raises(ValueError, match="never logged"):
        ips_estimate(HAND_LOGS, [0, 0, 0, 1.0], n_arms=4)
    with pytest.raises(ValueError, match="smaller"):
        ips_estimate(HAND_LOGS, [0.5, 0.5], n_arms=2)
    with pytest.raises(ValueError, match="clip"):
        ips_estimate(HAND_LOGS, [0.2, 0.5, 0.3], clip=0)
    with pytest.raises(ValueError, match="confidence"):
        ips_estimate(HAND_LOGS, [0.2, 0.5, 0.3], confidence=1.0)
    with pytest.raises(ValueError, match="positive integer"):
        collect_logs(_arms([0.5]), 0)
    with pytest.raises(ValueError, match="finite"):
        direct_method_estimate(HAND_LOGS, 0, reward_model=lambda c, a: math.inf)


def test_collect_logs_records_propensities_and_is_reproducible():
    arms = _arms([0.1, 0.9])
    logs = collect_logs(arms, 2000, [0.25, 0.75], seed=9)
    share = sum(e.arm_index == 1 for e in logs) / len(logs)
    assert share == pytest.approx(0.75, abs=0.03)
    assert {e.propensity for e in logs} == {0.25, 0.75}
    uniform = collect_logs(_arms([0.1, 0.9]), 5, seed=1)
    assert all(e.propensity == 0.5 for e in uniform)
    again = collect_logs(_arms([0.1, 0.9]), 5, seed=1)
    assert [e.arm_index for e in uniform] == [e.arm_index for e in again]


class _FixedArm:
    name = "fixed"

    def __init__(self, arm):
        self.arm = arm
        self.updates = []

    def reset(self, arms):
        self.updates = []

    def select_arm(self, arms, step):
        return self.arm

    def update(self, arms, step: BanditStep):
        self.updates.append(step)


def test_replay_fixed_policy_matches_arm_mean_and_feeds_updates():
    arms = _arms([0.2, 0.7, 0.4])
    logs = collect_logs(arms, 3000, seed=4)
    algo = _FixedArm(1)
    result = replay_evaluate(algo, arms, logs)
    matched = [e.reward for e in logs if e.arm_index == 1]
    assert result.n_matched == len(matched)
    assert result.mean_reward == pytest.approx(statistics.fmean(matched))
    assert result.arm_counts == [0, len(matched), 0]
    assert [s.step for s in algo.updates] == list(range(len(matched)))
    assert result.match_rate == pytest.approx(len(matched) / 3000)
    capped = replay_evaluate(_FixedArm(1), arms, logs, max_matches=10)
    assert capped.n_matched == 10


def test_replay_of_ucb1_tracks_its_online_average_reward():
    ps = [0.2, 0.5, 0.8]
    horizon = 600
    online, replayed = [], []
    for rep in range(40):
        arms = _arms(ps, rep)
        algo = ucb1(seed=rep)
        algo.reset(arms)
        total = 0.0
        for t in range(horizon):
            a = algo.select_arm(arms, t)
            r = arms[a].draw()
            algo.update(arms, BanditStep(a, arms[a].name, r, t))
            total += r
        online.append(total / horizon)
        logs = collect_logs(_arms(ps, 1000 + rep), 3 * horizon + 600, seed=rep)
        result = replay_evaluate(ucb1(seed=rep), _arms(ps), logs, max_matches=horizon)
        assert result.n_matched == horizon
        replayed.append(result.mean_reward)
    assert statistics.fmean(replayed) == pytest.approx(statistics.fmean(online), abs=0.02)
    assert statistics.fmean(replayed) > 0.65


def test_replay_requires_uniform_logging():
    arms = _arms([0.3, 0.6])
    logs = collect_logs(arms, 100, [0.9, 0.1], seed=0)
    with pytest.raises(ValueError, match="uniform"):
        replay_evaluate(ucb1(), arms, logs)
    result = replay_evaluate(ucb1(), arms, logs, require_uniform=False)
    assert result.n_matched > 0
    with pytest.raises(ValueError, match="outside"):
        replay_evaluate(ucb1(), arms[:1], logs, require_uniform=False)


def test_gaussian_rewards_supported():
    arms = [GaussianArm("g0", 1.0, 1.0, seed=1), GaussianArm("g1", 3.0, 1.0, seed=2)]
    logs = collect_logs(arms, 4000, seed=3)
    assert ips_estimate(logs, 1).value == pytest.approx(3.0, abs=0.15)
    assert snips_estimate(logs, [0.5, 0.5]).value == pytest.approx(2.0, abs=0.1)


def test_cli_offpolicy_text_and_json():
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main([
            "offpolicy", "--arms", "bern:0.3,bern:0.5,bern:0.7", "--target", "2",
            "--n", "3000", "--clip", "5", "--replay", "ucb1",
        ])
    assert rc == 0
    text = buf.getvalue()
    assert "true target value: 0.7000" in text
    assert "snips" in text and "clipped_ips" in text and "replay ucb1" in text

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main([
            "offpolicy", "--arms", "bern:0.3,bern:0.5,bern:0.7", "--target", "0.1,0.1,0.8",
            "--logging", "0.6,0.3,0.1", "--n", "4000", "--json",
        ])
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["true_value"] == pytest.approx(0.64)
    est = payload["estimates"]["ips"]
    assert est["ci_low"] - 0.05 <= 0.64 <= est["ci_high"] + 0.05
    assert set(payload["estimates"]) == {"ips", "snips", "dm", "dr"}


def test_cli_offpolicy_errors():
    rc = main(["offpolicy", "--arms", "bern:0.3,bern:0.5", "--target", "0.5,0.2"])
    assert rc == 2
    rc = main(["offpolicy", "--arms", "bern:0.3,bern:0.5", "--target", "0",
               "--logging", "0.5,0.5", "--n", "50", "--replay", "ucb1"])
    assert rc == 0
