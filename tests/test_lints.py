"""Tests for LinTS (disjoint linear Thompson sampling)."""

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
    LinTS,
    LinUCB,
    available_algorithms,
    lints,
    linucb,
    make_linear_contextual_arms,
    render_contextual_markdown_report,
    render_markdown_report,
    run_contextual_experiment,
    run_experiment,
    summarize_runs,
)
from bandit_kit._linalg import cholesky_lower, matvec


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


def _matmul(left, right):
    n = len(left)
    return [
        [sum(left[i][k] * right[k][j] for k in range(n)) for j in range(n)]
        for i in range(n)
    ]


def _transpose(matrix):
    n = len(matrix)
    return [[matrix[j][i] for j in range(n)] for i in range(n)]


def test_cholesky_reconstructs_spd_matrix() -> None:
    matrix = [[4.0, 2.0], [2.0, 3.0]]
    lower = cholesky_lower(matrix)
    rebuilt = _matmul(lower, _transpose(lower))
    for i in range(2):
        for j in range(2):
            assert rebuilt[i][j] == pytest.approx(matrix[i][j])
    # L is lower triangular: L_00 = 2, L_10 = 1, L_11 = sqrt(2).
    assert lower[0][0] == pytest.approx(2.0)
    assert lower[1][0] == pytest.approx(1.0)
    assert lower[0][1] == pytest.approx(0.0)
    assert lower[1][1] == pytest.approx(2.0 ** 0.5)
    mapped = matvec(lower, [1.0, 0.0])
    assert mapped == pytest.approx([2.0, 1.0])


def test_factory_returns_instance() -> None:
    algo = lints(v=0.5, dimension=3, ridge=2.0, seed=0)
    assert isinstance(algo, LinTS)
    assert algo.v == 0.5
    assert algo.dimension == 3
    assert algo.ridge == 2.0


def test_rejects_invalid_hyperparameters() -> None:
    with pytest.raises(ValueError, match="v"):
        LinTS(v=-0.1)
    with pytest.raises(ValueError, match="dimension"):
        LinTS(dimension=0)
    with pytest.raises(ValueError, match="ridge"):
        LinTS(ridge=0.0)
    with pytest.raises(ValueError, match="ridge"):
        LinTS(ridge=-1.0)


def test_requires_context_when_dimension_exceeds_one() -> None:
    algo = lints(dimension=2, seed=0)
    arms = make_aligned_arms()
    algo.reset(arms)
    with pytest.raises(ValueError, match="context vector"):
        algo.select_arm(arms, 0)


def test_rejects_mismatched_context_length() -> None:
    algo = lints(dimension=2, seed=0)
    arms = make_aligned_arms()
    algo.reset(arms)
    with pytest.raises(ValueError, match="context length"):
        algo.select_arm(arms, 0, context=(1.0,))


def test_zero_v_ties_break_by_arm_order_before_updates() -> None:
    algo = lints(v=0.0, dimension=2, seed=0)
    arms = make_aligned_arms()
    algo.reset(arms)
    # Prior mean is 0 for every arm, so every linear score matches and arm 0 wins.
    assert algo.select_arm(arms, 0, context=(0.3, -0.7)) == 0


def test_zero_v_matches_greedy_linucb() -> None:
    ts = lints(v=0.0, dimension=2, ridge=1.5, seed=4)
    ucb = linucb(alpha=0.0, dimension=2, ridge=1.5, seed=9)
    arms_ts = make_aligned_arms()
    arms_ucb = make_aligned_arms()
    ts.reset(arms_ts)
    ucb.reset(arms_ucb)
    contexts = [(1.0, 0.0), (0.0, 1.0), (0.4, -0.2), (-0.3, 0.8), (0.7, 0.7)]
    for step, context in enumerate(contexts):
        idx_ts = ts.select_arm(arms_ts, step, context=context)
        idx_ucb = ucb.select_arm(arms_ucb, step, context=context)
        assert idx_ts == idx_ucb
        reward = arms_ts[idx_ts].expected_given(context)
        step_ts = BanditStep(
            arm_index=idx_ts, arm_name=arms_ts[idx_ts].name, reward=reward, step=step
        )
        step_ucb = BanditStep(
            arm_index=idx_ucb, arm_name=arms_ucb[idx_ucb].name, reward=reward, step=step
        )
        ts.update(arms_ts, step_ts, context=context)
        ucb.update(arms_ucb, step_ucb, context=context)
        for arm_index in (0, 1):
            assert ts.theta_hat(arm_index) == pytest.approx(ucb.theta_hat(arm_index))


def test_posterior_mean_matches_closed_form() -> None:
    algo = lints(v=1.0, dimension=1, ridge=1.0, seed=0)
    arms = make_bernoulli_arms()
    algo.reset(arms)
    algo.update(
        arms,
        BanditStep(arm_index=0, arm_name="A", reward=1.0, step=0),
        context=(1.0,),
    )
    # A = ridge + x^2 = 2, b = 1, mean = 1/2.
    assert algo.theta_hat(0) == pytest.approx([0.5])
    assert algo.theta_hat(1) == pytest.approx([0.0])


def test_posterior_samples_match_gaussian() -> None:
    algo = lints(v=1.0, dimension=1, ridge=1.0, seed=11)
    arms = make_bernoulli_arms()
    algo.reset(arms)
    algo.update(
        arms,
        BanditStep(arm_index=0, arm_name="A", reward=1.0, step=0),
        context=(1.0,),
    )
    draws = [algo.sample_theta(0)[0] for _ in range(4000)]
    mean = sum(draws) / len(draws)
    variance = sum((value - mean) ** 2 for value in draws) / (len(draws) - 1)
    # Posterior is N(0.5, v^2 / 2) = N(0.5, 0.5).
    assert mean == pytest.approx(0.5, abs=0.05)
    assert variance == pytest.approx(0.5, abs=0.08)


def test_samples_explore_both_arms_under_a_shared_prior() -> None:
    algo = lints(v=1.0, dimension=1, ridge=1.0, seed=3)
    arms = make_aligned_arms()
    algo.reset(arms)
    chosen = {algo.select_arm(arms, step) for step in range(80)}
    assert chosen == {0, 1}


def test_prefers_arm_aligned_with_context_after_learning() -> None:
    algo = lints(v=0.0, dimension=2, ridge=1.0, seed=0)
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
    algo = lints(v=1.0, dimension=1, seed=0)
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
    algo = lints(v=0.0, dimension=2, seed=0)
    arms = make_aligned_arms()
    algo.reset(arms)
    context = (1.0, 0.0)
    idx = algo.select_arm(arms, 0, context=context)
    algo.update(arms, BanditStep(arm_index=idx, arm_name=arms[idx].name, reward=1.0, step=0))
    assert algo.theta_hat(idx)[0] > 0.0


def test_sample_theta_requires_reset_and_valid_index() -> None:
    algo = lints(dimension=1, seed=0)
    with pytest.raises(RuntimeError, match="reset"):
        algo.sample_theta(0)
    algo.reset(make_bernoulli_arms())
    with pytest.raises(IndexError):
        algo.sample_theta(3)


def test_same_seed_is_deterministic() -> None:
    algo_a = lints(v=0.8, dimension=2, seed=11)
    algo_b = lints(v=0.8, dimension=2, seed=11)
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


def test_in_algorithm_registry() -> None:
    assert "lints" in available_algorithms()


def test_stationary_experiment_runs_intercept_lints() -> None:
    arms = make_bernoulli_arms()
    experiment, results = run_experiment(
        arms=arms,
        algorithms=("lints",),
        steps=25,
        runs=3,
        seed=42,
        lints_v=0.4,
        lints_ridge=2.0,
    )
    assert experiment.lints_v == 0.4
    assert experiment.lints_ridge == 2.0
    assert len(results) == 3
    for result in results:
        assert result.algorithm == "lints"
        assert len(result.steps) == 25
        assert all(step.arm_index in {0, 1, 2} for step in result.steps)
        assert all(step.context is None for step in result.steps)
    summary = summarize_runs(arms, results)
    assert summary[0]["algorithm"] == "lints"
    report = render_markdown_report(experiment=experiment, runs=results, summary=experiment.summarize(results))
    assert "LinTS v: 0.4" in report
    assert "ridge=2.0" in report
    assert "linear Thompson sampling" in report


def test_contextual_experiment_is_deterministic() -> None:
    arms = make_linear_contextual_arms(n_arms=3, dimension=3, seed=1, noise_std=0.0)
    _, results_a = run_contextual_experiment(
        arms, algorithms=("lints",), steps=20, runs=2, seed=99, lints_v=0.7
    )
    _, results_b = run_contextual_experiment(
        arms, algorithms=("lints",), steps=20, runs=2, seed=99, lints_v=0.7
    )
    for ra, rb in zip(results_a, results_b):
        assert [step.arm_index for step in ra.steps] == [step.arm_index for step in rb.steps]
        assert [step.reward for step in ra.steps] == [step.reward for step in rb.steps]
        assert [step.context for step in ra.steps] == [step.context for step in rb.steps]


def test_lints_beats_ucb1_on_contextual_rewards() -> None:
    arms = [
        LinearContextualArm(name="A", theta=(0.0, 1.0, 0.0), noise_std=0.05, seed=1),
        LinearContextualArm(name="B", theta=(0.0, -1.0, 0.0), noise_std=0.05, seed=2),
        LinearContextualArm(name="C", theta=(0.0, 0.0, 1.0), noise_std=0.05, seed=3),
    ]
    experiment, results = run_contextual_experiment(
        arms,
        algorithms=("lints", "ucb1"),
        steps=200,
        runs=6,
        seed=3,
        lints_v=0.5,
        context_intercept=True,
    )
    summary = experiment.summarize(results)
    by_algo = {row["algorithm"]: row for row in summary}
    assert by_algo["lints"]["mean_final_regret"] < by_algo["ucb1"]["mean_final_regret"]
    assert by_algo["lints"]["mean_oracle_hit_rate"] > by_algo["ucb1"]["mean_oracle_hit_rate"]


def test_contextual_report_includes_lints_notes() -> None:
    arms = make_linear_contextual_arms(n_arms=2, dimension=2, seed=0, noise_std=0.0)
    experiment = ContextualBanditExperiment(
        arms=arms,
        algorithms=["lints"],
        steps=8,
        runs=1,
        seed=1,
        lints_v=0.7,
        lints_ridge=1.5,
    )
    runs = experiment.run()
    summary = experiment.summarize(runs)
    report = render_contextual_markdown_report(
        experiment=experiment, runs=runs, summary=summary
    )
    assert "LinTS v: 0.7" in report
    assert "LinTS ridge: 1.5" in report
    assert "linear Thompson sampling" in report
    assert "ignores context" not in report


def test_cli_compare_contextual_runs_lints(tmp_path: Path) -> None:
    from bandit_kit.cli import main

    destination = tmp_path / "lints.md"
    exit_code = main([
        "compare-contextual",
        "--n-arms", "2",
        "--dim", "2",
        "--steps", "8",
        "--runs", "1",
        "--seed", "1",
        "--lints-v", "0.3",
        "--ridge", "2.0",
        "--algorithms", "lints",
        "--output", str(destination),
    ])
    assert exit_code == 0
    text = destination.read_text(encoding="utf-8")
    assert "LinTS v: 0.3" in text
    assert "LinTS ridge: 2.0" in text


def test_cli_compare_contextual_json_includes_lints() -> None:
    from bandit_kit.cli import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main([
            "compare-contextual",
            "--n-arms", "2",
            "--dim", "2",
            "--steps", "4",
            "--runs", "1",
            "--algorithms", "lints,linucb",
        ])
    assert rc == 0
    output = buf.getvalue()
    payload = json.loads(output[output.rfind("{") :])
    assert payload["algorithms"] == ["lints", "linucb"]


def test_experiment_rejects_invalid_lints_hyperparameters() -> None:
    with pytest.raises(ValueError, match="lints_v"):
        BanditExperiment(
            arms=make_bernoulli_arms(),
            algorithms=["lints"],
            lints_v=-0.1,
        )
    with pytest.raises(ValueError, match="lints_ridge"):
        BanditExperiment(
            arms=make_bernoulli_arms(),
            algorithms=["lints"],
            lints_ridge=0.0,
        )
    with pytest.raises(ValueError, match="lints_v"):
        ContextualBanditExperiment(
            arms=make_aligned_arms(),
            algorithms=["lints"],
            lints_v=-1.0,
        )


def test_linucb_class_is_unchanged_sibling() -> None:
    # Guard against accidentally collapsing the two linear policies.
    assert not issubclass(LinTS, LinUCB)
    assert not issubclass(LinUCB, LinTS)
