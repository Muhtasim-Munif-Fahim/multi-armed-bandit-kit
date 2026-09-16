"""Tests for LinUCB, linear contextual arms, and the contextual harness."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from bandit_kit import (
    BanditExperiment,
    BanditStep,
    BernoulliArm,
    ContextualBanditExperiment,
    LinearContextualArm,
    LinUCB,
    available_algorithms,
    best_arm_given,
    cumulative_contextual_regret,
    linucb,
    make_linear_contextual_arms,
    render_contextual_markdown_report,
    run_contextual_experiment,
    run_experiment,
    sample_context,
    summarize_runs,
)
from bandit_kit._linalg import identity, matvec, sherman_morrison_update


def make_bernoulli_arms():
    return [
        BernoulliArm(name="A", p=0.10, seed=1),
        BernoulliArm(name="B", p=0.20, seed=2),
        BernoulliArm(name="C", p=0.05, seed=3),
    ]


def make_aligned_arms():
    return [
        LinearContextualArm(name="A", theta=(1.0, 0.0), noise_std=0.0, seed=0),
        LinearContextualArm(name="B", theta=(0.0, 1.0), noise_std=0.0, seed=1),
    ]


def test_factory_returns_instance() -> None:
    algo = linucb(alpha=1.0, dimension=2, seed=0)
    assert isinstance(algo, LinUCB)
    assert algo.alpha == 1.0
    assert algo.dimension == 2


def test_rejects_invalid_hyperparameters() -> None:
    with pytest.raises(ValueError, match="alpha"):
        LinUCB(alpha=-0.1)
    with pytest.raises(ValueError, match="dimension"):
        LinUCB(dimension=0)
    with pytest.raises(ValueError, match="ridge"):
        LinUCB(ridge=0.0)
    with pytest.raises(ValueError, match="ridge"):
        LinUCB(ridge=-1.0)


def test_requires_context_when_dimension_exceeds_one() -> None:
    algo = linucb(dimension=2, seed=0)
    arms = make_aligned_arms()
    algo.reset(arms)
    with pytest.raises(ValueError, match="context vector"):
        algo.select_arm(arms, 0)


def test_rejects_mismatched_context_length() -> None:
    algo = linucb(dimension=2, seed=0)
    arms = make_aligned_arms()
    algo.reset(arms)
    with pytest.raises(ValueError, match="context length"):
        algo.select_arm(arms, 0, context=(1.0,))


def test_ties_break_by_arm_order_before_updates() -> None:
    algo = linucb(alpha=1.0, dimension=2, seed=0)
    arms = make_aligned_arms()
    algo.reset(arms)
    # Prior mean is 0 for every arm, so all UCBs match and arm 0 wins.
    assert algo.select_arm(arms, 0, context=(0.3, -0.7)) == 0


def test_prefers_arm_aligned_with_context_after_learning() -> None:
    algo = linucb(alpha=0.05, dimension=2, ridge=1.0, seed=0)
    arms = make_aligned_arms()
    algo.reset(arms)
    training = [(1.0, 0.0), (0.0, 1.0)] * 10
    for step, context in enumerate(training):
        idx = 0 if context[0] >= context[1] else 1
        reward = arms[idx].expected_given(context)
        algo.update(
            arms,
            BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=reward, step=step),
            context=context,
        )
    assert algo.select_arm(arms, 100, context=(1.0, 0.0)) == 0
    assert algo.select_arm(arms, 101, context=(0.0, 1.0)) == 1
    theta_a = algo.theta_hat(0)
    theta_b = algo.theta_hat(1)
    assert theta_a[0] > theta_a[1]
    assert theta_b[1] > theta_b[0]


def test_intercept_only_concentrates_on_best_bernoulli_arm() -> None:
    algo = linucb(alpha=1.0, dimension=1, seed=0)
    arms = make_bernoulli_arms()
    algo.reset(arms)
    counts = {0: 0, 1: 0, 2: 0}
    for step in range(400):
        idx = algo.select_arm(arms, step)
        counts[idx] += 1
        algo.update(
            arms,
            BanditStep(
                arm_index=idx,
                arm_name=arms[idx].name,
                reward=arms[idx].draw(),
                step=step,
            ),
        )
    assert counts[1] > counts[0]
    assert counts[1] > counts[2]


def test_update_reuses_last_context() -> None:
    algo = linucb(alpha=1.0, dimension=2, seed=0)
    arms = make_aligned_arms()
    algo.reset(arms)
    context = (1.0, 0.0)
    idx = algo.select_arm(arms, 0, context=context)
    algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=1.0, step=0))
    # After a unit reward on (1, 0), arm `idx` should have a positive first weight.
    assert algo.theta_hat(idx)[0] > 0.0


def test_sherman_morrison_matches_explicit_inverse() -> None:
    ridge = 1.0
    inverse = identity(2, 1.0 / ridge)
    vector = [0.5, -1.5]
    updated = sherman_morrison_update(inverse, vector)
    # A = I + x x^T; invert the 2x2 explicitly.
    a00 = 1.0 + vector[0] ** 2
    a01 = vector[0] * vector[1]
    a11 = 1.0 + vector[1] ** 2
    det = a00 * a11 - a01 * a01
    expected = [[a11 / det, -a01 / det], [-a01 / det, a00 / det]]
    for i in range(2):
        for j in range(2):
            assert updated[i][j] == pytest.approx(expected[i][j], abs=1e-12)
    mapped = matvec(updated, vector)
    # (A + xx^T)^{-1} x = A^{-1} x / (1 + x^T A^{-1} x)
    denom = 1.0 + vector[0] ** 2 + vector[1] ** 2
    assert mapped[0] == pytest.approx(vector[0] / denom)
    assert mapped[1] == pytest.approx(vector[1] / denom)


def test_in_algorithm_registry() -> None:
    assert "linucb" in available_algorithms()


def test_stationary_experiment_runs_intercept_linucb() -> None:
    arms = make_bernoulli_arms()
    experiment, results = run_experiment(
        arms=arms,
        algorithms=("linucb",),
        steps=25,
        runs=3,
        seed=42,
        linucb_alpha=0.8,
    )
    assert experiment.linucb_alpha == 0.8
    assert len(results) == 3
    for result in results:
        assert result.algorithm == "linucb"
        assert len(result.steps) == 25
        assert all(step.arm_index in {0, 1, 2} for step in result.steps)
        assert all(step.context is None for step in result.steps)
    summary = summarize_runs(arms, results)
    assert summary[0]["algorithm"] == "linucb"


def test_same_seed_is_deterministic() -> None:
    algo_a = linucb(alpha=1.0, dimension=2, seed=11)
    algo_b = linucb(alpha=1.0, dimension=2, seed=11)
    arms_a = make_aligned_arms()
    arms_b = make_aligned_arms()
    algo_a.reset(arms_a)
    algo_b.reset(arms_b)
    contexts = [(0.4, -0.2), (-0.9, 0.1), (0.0, 1.0), (0.7, 0.7)]
    for step, context in enumerate(contexts):
        idx_a = algo_a.select_arm(arms_a, step, context=context)
        idx_b = algo_b.select_arm(arms_b, step, context=context)
        assert idx_a == idx_b
        reward = 1.0 if idx_a == 0 else 0.0
        algo_a.update(
            arms_a,
            BanditStep(arm_index=idx_a, arm_name=arms_a[idx_a].name, reward=reward, step=step),
            context=context,
        )
        algo_b.update(
            arms_b,
            BanditStep(arm_index=idx_b, arm_name=arms_b[idx_b].name, reward=reward, step=step),
            context=context,
        )


def test_linear_contextual_arm_expected_given() -> None:
    arm = LinearContextualArm(name="A", theta=(1.0, -2.0, 0.5), noise_std=0.0)
    assert arm.expected_given((1.0, 1.0, 2.0)) == pytest.approx(-0.0)
    assert arm.draw((2.0, 0.0, 0.0)) == pytest.approx(2.0)


def test_linear_contextual_arm_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="name"):
        LinearContextualArm(name="", theta=(1.0,))
    with pytest.raises(ValueError, match="theta"):
        LinearContextualArm(name="A", theta=())
    with pytest.raises(ValueError, match="noise_std"):
        LinearContextualArm(name="A", theta=(1.0,), noise_std=-0.1)
    arm = LinearContextualArm(name="A", theta=(1.0, 0.0), noise_std=0.0)
    with pytest.raises(ValueError, match="context length"):
        arm.expected_given((1.0,))


def test_sample_context_pins_intercept() -> None:
    import random

    rng = random.Random(0)
    context = sample_context(4, rng, intercept=True)
    assert context[0] == 1.0
    assert len(context) == 4
    assert all(-1.0 <= value <= 1.0 for value in context[1:])
    rng = random.Random(1)
    no_intercept = sample_context(3, rng, intercept=False)
    assert len(no_intercept) == 3
    assert all(-1.0 <= value <= 1.0 for value in no_intercept)


def test_make_linear_contextual_arms_and_best_arm_given() -> None:
    arms = make_linear_contextual_arms(n_arms=3, dimension=3, seed=0, intercept=True)
    assert len(arms) == 3
    assert {arm.dimension for arm in arms} == {3}
    context = [1.0, 0.5, -0.25]
    best = best_arm_given(arms, context)
    values = [arm.expected_given(context) for arm in arms]
    assert best.expected_given(context) == max(values)


def test_contextual_experiment_runs_linucb_and_baselines() -> None:
    arms = [
        LinearContextualArm(name="A", theta=(0.0, 1.0, 0.0), noise_std=0.05, seed=1),
        LinearContextualArm(name="B", theta=(0.0, -1.0, 0.0), noise_std=0.05, seed=2),
        LinearContextualArm(name="C", theta=(0.0, 0.0, 1.0), noise_std=0.05, seed=3),
    ]
    experiment, results = run_contextual_experiment(
        arms,
        algorithms=("linucb", "ucb1"),
        steps=30,
        runs=2,
        seed=7,
        linucb_alpha=1.0,
        context_intercept=True,
    )
    assert experiment.dimension == 3
    assert len(results) == 4
    for result in results:
        assert len(result.steps) == 30
        assert all(step.context is not None and len(step.context) == 3 for step in result.steps)
        assert all(step.arm_index in {0, 1, 2} for step in result.steps)
    summary = experiment.summarize(results)
    assert {row["algorithm"] for row in summary} == {"linucb", "ucb1"}
    for row in summary:
        assert 0.0 <= row["mean_oracle_hit_rate"] <= 1.0


def test_contextual_experiment_is_deterministic() -> None:
    arms = make_linear_contextual_arms(n_arms=3, dimension=3, seed=1, noise_std=0.0)
    _, results_a = run_contextual_experiment(
        arms, algorithms=("linucb",), steps=20, runs=2, seed=99
    )
    _, results_b = run_contextual_experiment(
        arms, algorithms=("linucb",), steps=20, runs=2, seed=99
    )
    for ra, rb in zip(results_a, results_b):
        assert [step.arm_index for step in ra.steps] == [step.arm_index for step in rb.steps]
        assert [step.reward for step in ra.steps] == [step.reward for step in rb.steps]
        assert [step.context for step in ra.steps] == [step.context for step in rb.steps]


def test_linucb_beats_ucb1_on_contextual_rewards() -> None:
    arms = [
        LinearContextualArm(name="A", theta=(0.0, 1.0, 0.0), noise_std=0.05, seed=1),
        LinearContextualArm(name="B", theta=(0.0, -1.0, 0.0), noise_std=0.05, seed=2),
        LinearContextualArm(name="C", theta=(0.0, 0.0, 1.0), noise_std=0.05, seed=3),
    ]
    experiment, results = run_contextual_experiment(
        arms,
        algorithms=("linucb", "ucb1"),
        steps=150,
        runs=8,
        seed=3,
        linucb_alpha=1.0,
        context_intercept=True,
    )
    summary = experiment.summarize(results)
    by_algo = {row["algorithm"]: row for row in summary}
    assert by_algo["linucb"]["mean_final_regret"] < by_algo["ucb1"]["mean_final_regret"]
    assert by_algo["linucb"]["mean_oracle_hit_rate"] > by_algo["ucb1"]["mean_oracle_hit_rate"]


def test_cumulative_contextual_regret_matches_oracle() -> None:
    arms = make_aligned_arms()
    steps = [
        BanditStep(arm_index=0, arm_name="A", reward=1.0, step=0, context=(1.0, 0.0)),
        BanditStep(arm_index=0, arm_name="A", reward=0.0, step=1, context=(0.0, 1.0)),
    ]
    curve = cumulative_contextual_regret(steps, arms)
    # Step 0: oracle A (1.0), chosen A -> 0. Step 1: oracle B (1.0), chosen A (0.0) -> 1.0
    assert curve[0] == pytest.approx(0.0)
    assert curve[1] == pytest.approx(1.0)


def test_contextual_report_includes_linucb_notes() -> None:
    arms = make_linear_contextual_arms(n_arms=2, dimension=2, seed=0, noise_std=0.0)
    experiment = ContextualBanditExperiment(
        arms=arms, algorithms=["linucb"], steps=8, runs=1, seed=1, linucb_alpha=0.7
    )
    runs = experiment.run()
    summary = experiment.summarize(runs)
    report = render_contextual_markdown_report(
        experiment=experiment, runs=runs, summary=summary
    )
    assert "Contextual linear bandit experiment" in report
    assert "LinUCB alpha: 0.7" in report
    assert "context-conditional" in report
    assert "linucb" in report


def test_cli_compare_contextual_writes_report(tmp_path: Path) -> None:
    from bandit_kit.cli import main

    destination = tmp_path / "contextual.md"
    exit_code = main([
        "compare-contextual",
        "--n-arms", "3",
        "--dim", "3",
        "--steps", "12",
        "--runs", "2",
        "--seed", "1",
        "--alpha", "0.9",
        "--algorithms", "linucb,ucb1",
        "--output", str(destination),
    ])
    assert exit_code == 0
    assert destination.exists()
    text = destination.read_text(encoding="utf-8")
    assert "Contextual linear bandit" in text
    assert "linucb" in text
    assert "ucb1" in text


def test_cli_compare_contextual_json_summary() -> None:
    from bandit_kit.cli import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main([
            "compare-contextual",
            "--n-arms", "2",
            "--dim", "2",
            "--steps", "5",
            "--runs", "1",
            "--seed", "2",
            "--algorithms", "linucb",
        ])
    assert rc == 0
    output = buf.getvalue()
    payload = json.loads(output[output.rfind("{") :])
    assert payload["algorithms"] == ["linucb"]
    assert payload["dim"] == 2
    assert payload["n_arms"] == 2


def test_cli_compare_contextual_rejects_unknown_algorithm(capsys) -> None:
    from bandit_kit.cli import main

    rc = main([
        "compare-contextual",
        "--algorithms", "bogus",
        "--steps", "5",
        "--runs", "1",
    ])
    assert rc == 2
    assert "unknown algorithm" in capsys.readouterr().err


def test_experiment_rejects_invalid_linucb_alpha() -> None:
    with pytest.raises(ValueError, match="linucb_alpha"):
        BanditExperiment(
            arms=make_bernoulli_arms(),
            algorithms=["linucb"],
            linucb_alpha=-1.0,
        )
    with pytest.raises(ValueError, match="linucb_alpha"):
        ContextualBanditExperiment(
            arms=make_aligned_arms(),
            algorithms=["linucb"],
            linucb_alpha=-0.5,
        )


def test_stationary_policies_still_registered() -> None:
    algos = available_algorithms()
    for name in ("epsilon_greedy", "ucb1", "thompson", "exp3", "linucb"):
        assert name in algos
