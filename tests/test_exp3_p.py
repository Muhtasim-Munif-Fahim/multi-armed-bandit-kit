"""Tests for EXP3.P (high-probability adversarial bandit)."""

from __future__ import annotations

import math

import pytest

from bandit_kit import (
    BanditExperiment,
    BanditStep,
    BernoulliArm,
    Exp3P,
    available_algorithms,
    exp3_p,
)


def _arms(ps=(0.2, 0.8, 0.4), seed=0):
    return [BernoulliArm(name=chr(65 + i), p=p, seed=seed + i) for i, p in enumerate(ps)]


def _drive(algo, arms, steps):
    algo.reset(arms)
    total = 0.0
    counts = [0] * len(arms)
    for t in range(steps):
        i = algo.select_arm(arms, t)
        r = arms[i].draw()
        total += r
        counts[i] += 1
        algo.update(arms, BanditStep(arm_index=i, arm_name=arms[i].name, reward=r, step=t))
    return total, counts


def test_factory_registry_and_default_parameters():
    algo = exp3_p(horizon=10_000, delta=0.1, seed=0)
    assert isinstance(algo, Exp3P)
    assert "exp3_p" in available_algorithms()
    algo.reset(_arms())
    k, n = 3, 10_000
    assert algo.beta == pytest.approx(math.sqrt(math.log(k / 0.1) / (n * k)))
    assert algo.eta == pytest.approx(0.95 * math.sqrt(math.log(k) / (n * k)))
    assert algo.gamma == pytest.approx(1.05 * math.sqrt(k * math.log(k) / n))


def test_short_horizon_caps_gamma_and_overrides_win():
    algo = Exp3P(horizon=2)
    algo.reset(_arms())
    assert algo.gamma == 1.0
    custom = Exp3P(horizon=100, eta=0.3, gamma=0.2, beta=0.0)
    custom.reset(_arms())
    assert (custom.eta, custom.gamma, custom.beta) == (0.3, 0.2, 0.0)


def test_distribution_valid_and_explores():
    algo = Exp3P(horizon=500, seed=1)
    arms = _arms()
    algo.reset(arms)
    _drive(algo, arms, 0)
    for t in range(200):
        p = algo._distribution()
        assert sum(p) == pytest.approx(1.0)
        assert min(p) >= algo.gamma / len(arms) - 1e-12
        i = algo.select_arm(arms, t)
        algo.update(arms, BanditStep(arm_index=i, arm_name="x", reward=arms[i].draw(), step=t))


def test_every_arm_gets_optimistic_bias():
    algo = Exp3P(horizon=100, beta=0.5, gamma=0.3, eta=0.1, seed=0)
    arms = _arms()
    algo.reset(arms)
    i = algo.select_arm(arms, 0)
    probs = list(algo._last_probabilities)
    algo.update(arms, BanditStep(arm_index=i, arm_name="x", reward=1.0, step=0))
    for j, p in enumerate(probs):
        expected = ((1.0 if j == i else 0.0) + 0.5) / p
        assert algo._gains[j] == pytest.approx(expected)


def test_long_run_no_overflow_and_identifies_best_arm():
    algo = Exp3P(horizon=20_000, seed=3)
    total, counts = _drive(algo, _arms(), 20_000)
    assert counts.index(max(counts)) == 1
    assert counts[1] / 20_000 > 0.6
    assert all(math.isfinite(g) for g in algo._gains)


def test_regret_within_high_probability_bound_on_every_seed():
    # Pseudo-regret vs. the best arm stays under the Bubeck & Cesa-Bianchi
    # (2012, Thm 3.3) bound 5.15 sqrt(n K log(K / delta)) on every seed, and
    # the average per-step regret shrinks with the horizon.
    best = 0.8
    bound = lambda n: 5.15 * math.sqrt(n * 3 * math.log(3 / 0.05))
    per_step = {}
    for steps in (1000, 8000):
        regrets = []
        for s in range(8):
            total, _ = _drive(Exp3P(horizon=steps, seed=s), _arms(seed=100 * s), steps)
            regrets.append(best * steps - total)
        assert max(regrets) < bound(steps)
        per_step[steps] = sum(regrets) / len(regrets) / steps
    assert per_step[8000] < per_step[1000]


def test_experiment_uses_steps_as_horizon():
    exp = BanditExperiment(arms=_arms(), algorithms=["exp3_p"], steps=300, runs=2, seed=0)
    algo = exp._make_algorithm("exp3_p", 0)
    assert algo.horizon == 300
    results = exp.run()
    assert len(results) == 2 and len(results[0].steps) == 300


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon": 0},
        {"horizon": True},
        {"delta": 0.0},
        {"delta": 1.0},
        {"eta": 0.0},
        {"gamma": 0.0},
        {"gamma": 1.5},
        {"beta": -0.1},
    ],
)
def test_validation(kwargs):
    with pytest.raises(ValueError):
        Exp3P(**kwargs)
