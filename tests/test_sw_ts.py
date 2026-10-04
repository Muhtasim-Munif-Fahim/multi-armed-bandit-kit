"""Tests for sliding-window Thompson sampling (SW-TS)."""

from __future__ import annotations

import pytest

from bandit_kit import (
    BanditExperiment,
    BanditStep,
    BernoulliArm,
    SlidingWindowThompson,
    available_algorithms,
    run_experiment,
    sliding_window_thompson,
)


def make_arms():
    return [
        BernoulliArm(name="A", p=0.10, seed=1),
        BernoulliArm(name="B", p=0.55, seed=2),
        BernoulliArm(name="C", p=0.20, seed=3),
    ]


def test_factory_and_exports() -> None:
    algo = sliding_window_thompson(window=50, seed=0)
    assert isinstance(algo, SlidingWindowThompson)
    assert algo.window == 50
    assert "sliding_window_thompson" in available_algorithms()


def test_rejects_bad_window() -> None:
    with pytest.raises(ValueError, match="window must be an integer >= 1"):
        SlidingWindowThompson(window=0)
    with pytest.raises(ValueError, match="window must be an integer >= 1"):
        SlidingWindowThompson(window=-3)
    with pytest.raises(ValueError, match="window must be an integer >= 1"):
        SlidingWindowThompson(window=True)  # type: ignore[arg-type]


def test_selects_valid_arm() -> None:
    algo = sliding_window_thompson(window=20, seed=0)
    arms = make_arms()
    algo.reset(arms)
    for step in range(30):
        idx = algo.select_arm(arms, step)
        assert 0 <= idx < len(arms)
        reward = 1.0 if idx == 1 else 0.0
        algo.update(
            arms,
            BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=reward, step=step),
        )


def test_window_forgets_old_rewards() -> None:
    """After filling the window with arm-0 failures then only arm-1 successes,
    arm 0's posterior returns to the prior (alpha=beta=1)."""
    arms = [
        BernoulliArm(name="cold", p=0.01, seed=1),
        BernoulliArm(name="hot", p=0.99, seed=2),
    ]
    algo = SlidingWindowThompson(window=5, seed=0)
    algo.reset(arms)
    # Push five cold failures into the window.
    for step in range(5):
        algo.update(
            arms,
            BanditStep(arm_index=0, arm_name="cold", reward=0.0, step=step),
        )
    assert algo._beta[0] == pytest.approx(6.0)  # prior 1 + 5 failures
    # Push five hot successes; cold observations drop out of the window.
    for step in range(5, 10):
        algo.update(
            arms,
            BanditStep(arm_index=1, arm_name="hot", reward=1.0, step=step),
        )
    assert algo._alpha[0] == pytest.approx(1.0)
    assert algo._beta[0] == pytest.approx(1.0)
    assert algo._alpha[1] == pytest.approx(6.0)  # prior 1 + 5 successes
    assert algo._beta[1] == pytest.approx(1.0)
    assert len(algo._history) == 5


def test_seed_reproducibility() -> None:
    arms = make_arms()
    a = sliding_window_thompson(window=30, seed=123)
    b = sliding_window_thompson(window=30, seed=123)
    a.reset(arms)
    b.reset(arms)
    choices_a = []
    choices_b = []
    for step in range(40):
        ia = a.select_arm(arms, step)
        ib = b.select_arm(arms, step)
        choices_a.append(ia)
        choices_b.append(ib)
        assert ia == ib
        reward = 0.7 if ia == 1 else 0.2
        record = BanditStep(arm_index=ia, arm_name=arms[ia].name, reward=reward, step=step)
        a.update(arms, record)
        b.update(arms, record)
    assert choices_a == choices_b


def test_different_seeds_diverge() -> None:
    arms = make_arms()
    a = sliding_window_thompson(window=30, seed=1)
    b = sliding_window_thompson(window=30, seed=2)
    a.reset(arms)
    b.reset(arms)
    choices_a = [a.select_arm(arms, s) for s in range(20)]
    choices_b = [b.select_arm(arms, s) for s in range(20)]
    assert choices_a != choices_b


def test_prefers_better_arm_in_stationary_bandit() -> None:
    experiment, results = run_experiment(
        make_arms(),
        algorithms=["sliding_window_thompson"],
        steps=400,
        runs=20,
        seed=7,
        sliding_window=200,
    )
    summary = experiment.summarize(results)[0]
    fractions = summary["mean_arm_selection_fraction"]
    assert fractions["B"] > fractions["A"]
    assert fractions["B"] > fractions["C"]


def test_experiment_accepts_sw_ts() -> None:
    experiment = BanditExperiment(
        arms=make_arms(),
        algorithms=["sliding_window_thompson", "thompson"],
        steps=40,
        runs=2,
        seed=1,
        sliding_window=15,
    )
    results = experiment.run()
    assert len(results) == 4
    assert {r.algorithm for r in results} == {"sliding_window_thompson", "thompson"}


def test_large_window_behaves_like_thompson_prior_structure() -> None:
    """With a large window, posterior counts accumulate without forgetting."""
    arms = make_arms()
    algo = SlidingWindowThompson(window=10_000, seed=0)
    algo.reset(arms)
    for step in range(10):
        algo.update(
            arms,
            BanditStep(arm_index=1, arm_name="B", reward=1.0, step=step),
        )
    assert algo._alpha[1] == pytest.approx(11.0)
    assert algo._beta[1] == pytest.approx(1.0)
    assert len(algo._history) == 10
