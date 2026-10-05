"""Tests for UCB2 (Auer, Cesa-Bianchi & Fischer, 2002)."""

from __future__ import annotations

import math

import pytest

from bandit_kit import (
    BanditExperiment,
    BanditStep,
    BernoulliArm,
    UCB2,
    available_algorithms,
    run_experiment,
    ucb2,
)


def make_arms():
    return [
        BernoulliArm(name="A", p=0.10, seed=1),
        BernoulliArm(name="B", p=0.55, seed=2),
        BernoulliArm(name="C", p=0.20, seed=3),
    ]


def test_factory_and_exports() -> None:
    algo = ucb2(alpha=0.2, seed=0)
    assert isinstance(algo, UCB2)
    assert algo.alpha == 0.2
    assert "ucb2" in available_algorithms()


def test_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError, match="alpha must be in"):
        UCB2(alpha=0.0)
    with pytest.raises(ValueError, match="alpha must be in"):
        UCB2(alpha=1.0)
    with pytest.raises(ValueError, match="alpha must be"):
        UCB2(alpha=True)  # type: ignore[arg-type]


def test_tau_matches_formula() -> None:
    algo = UCB2(alpha=0.5, seed=0)
    assert algo.tau(0) == 1
    assert algo.tau(1) == int(math.ceil(1.5))
    assert algo.tau(2) == int(math.ceil(1.5 ** 2))
    assert algo.tau(5) == int(math.ceil(1.5 ** 5))


def test_round_robin_until_every_arm_seen() -> None:
    algo = ucb2(alpha=0.1, seed=0)
    arms = make_arms()
    algo.reset(arms)
    chosen = [algo.select_arm(arms, step) for step in range(3)]
    assert sorted(chosen) == [0, 1, 2]


def test_index_matches_auer_formula() -> None:
    arms = make_arms()
    algo = UCB2(alpha=0.1, seed=0)
    algo.reset(arms)
    for step, idx in enumerate([0, 1, 2]):
        algo.update(
            arms,
            BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=float(idx == 1), step=step),
        )
    # After warm-up: each arm once, epochs still 0, total_pulls=3.
    assert algo._total_pulls == 3
    assert algo._epochs == [0, 0, 0]
    n = 3
    alpha = 0.1
    for idx in range(3):
        tau_r = max(1, int(math.ceil((1 + alpha) ** 0)))
        mean = algo._sums[idx] / algo._counts[idx]
        radius = math.sqrt(
            (1 + alpha) * math.log((math.e * n) / tau_r) / (2.0 * tau_r)
        )
        assert algo.upper_bound(idx) == pytest.approx(mean + radius)


def test_epoch_plays_tau_difference_times() -> None:
    arms = make_arms()
    algo = UCB2(alpha=0.5, seed=0)
    algo.reset(arms)
    # Warm-up: every arm once (order follows the UCB1-style round-robin).
    seen = []
    for step in range(3):
        idx = algo.select_arm(arms, step)
        seen.append(idx)
        algo.update(
            arms,
            BanditStep(
                arm_index=idx,
                arm_name=arms[idx].name,
                reward=1.0 if idx == 1 else 0.0,
                step=step,
            ),
        )
    assert sorted(seen) == [0, 1, 2]
    # First post-warm-up epoch: r=0 → length = tau(1)-tau(0) = ceil(1.5)-1 = 1
    length = algo.tau(1) - algo.tau(0)
    assert length == 1
    chosen = algo.select_arm(arms, 3)
    assert chosen == 1  # best empirical mean
    assert algo._remaining == length
    algo.update(arms, BanditStep(arm_index=chosen, arm_name="B", reward=1.0, step=3))
    assert algo._epochs[1] == 1
    assert algo._committed is None


def test_prefers_better_arm() -> None:
    experiment, results = run_experiment(
        make_arms(),
        algorithms=["ucb2"],
        steps=400,
        runs=12,
        seed=7,
        ucb2_alpha=0.1,
    )
    summary = experiment.summarize(results)[0]
    fractions = summary["mean_arm_selection_fraction"]
    assert fractions["B"] > fractions["A"]
    assert fractions["B"] > fractions["C"]


def test_experiment_accepts_ucb2() -> None:
    experiment = BanditExperiment(
        arms=make_arms(),
        algorithms=["ucb2", "ucb1"],
        steps=40,
        runs=2,
        seed=1,
        ucb2_alpha=0.2,
    )
    results = experiment.run()
    assert len(results) == 4
    assert {r.algorithm for r in results} == {"ucb2", "ucb1"}


def test_clips_rewards_outside_unit_interval() -> None:
    arms = make_arms()
    algo = UCB2(alpha=0.1, seed=0)
    algo.reset(arms)
    algo.update(arms, BanditStep(arm_index=0, arm_name="A", reward=2.5, step=0))
    assert algo._sums[0] == pytest.approx(1.0)
