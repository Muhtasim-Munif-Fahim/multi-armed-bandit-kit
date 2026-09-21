"""Unit tests for bandit_kit.reporting and the CLI."""

from __future__ import annotations

from pathlib import Path

from bandit_kit import (
    BanditExperiment,
    BernoulliArm,
    render_markdown_report,
)


def make_arms():
    return [
        BernoulliArm(name="A", p=0.10),
        BernoulliArm(name="B", p=0.20),
        BernoulliArm(name="C", p=0.05),
    ]


def test_report_includes_configuration_block(tmp_path: Path) -> None:
    experiment = BanditExperiment(arms=make_arms(), algorithms=["epsilon_greedy", "ucb1"], steps=10, runs=2, seed=42)
    runs = experiment.run()
    summary = experiment.summarize(runs)
    report = render_markdown_report(experiment=experiment, runs=runs, summary=summary)
    assert "# Multi-armed bandit experiment" in report
    assert "## Configuration" in report
    assert "epsilon (epsilon-greedy)" in report
    assert "## Per-algorithm summary" in report
    assert "## Arm selection fractions" in report
    assert "epsilon_greedy" in report
    assert "ucb1" in report


def test_report_includes_exp3_notes() -> None:
    experiment = BanditExperiment(
        arms=make_arms(), algorithms=["exp3"], steps=10, runs=2, seed=42, gamma=0.2
    )
    runs = experiment.run()
    summary = experiment.summarize(runs)
    report = render_markdown_report(experiment=experiment, runs=runs, summary=summary)
    assert "exp3" in report
    assert "gamma (exp3): 0.2" in report
    assert "adversarial bandit" in report


def test_report_includes_boltzmann_notes() -> None:
    experiment = BanditExperiment(
        arms=make_arms(),
        algorithms=["boltzmann"],
        steps=10,
        runs=2,
        seed=42,
        temperature_start=0.8,
        temperature_min=0.1,
        temperature_decay=0.95,
    )
    runs = experiment.run()
    summary = experiment.summarize(runs)
    report = render_markdown_report(experiment=experiment, runs=runs, summary=summary)
    assert "boltzmann" in report
    assert "temperature start (boltzmann): 0.8" in report
    assert "temperature min (boltzmann): 0.1" in report
    assert "temperature decay (boltzmann): 0.95" in report
    assert "softmax" in report


def test_report_includes_kl_ucb_notes() -> None:
    experiment = BanditExperiment(
        arms=make_arms(), algorithms=["kl_ucb"], steps=10, runs=2, seed=42, kl_ucb_c=3.0
    )
    runs = experiment.run()
    summary = experiment.summarize(runs)
    report = render_markdown_report(experiment=experiment, runs=runs, summary=summary)
    assert "kl_ucb" in report
    assert "KL-UCB c: 3.0" in report
    assert "Bernoulli KL-UCB" in report


def test_report_includes_linucb_notes() -> None:
    experiment = BanditExperiment(
        arms=make_arms(), algorithms=["linucb"], steps=10, runs=2, seed=42, linucb_alpha=0.5
    )
    runs = experiment.run()
    summary = experiment.summarize(runs)
    report = render_markdown_report(experiment=experiment, runs=runs, summary=summary)
    assert "linucb" in report
    assert "LinUCB alpha: 0.5" in report
    assert "ridge-UCB" in report


def test_report_lists_arm_selection_fractions(tmp_path: Path) -> None:
    experiment = BanditExperiment(arms=make_arms(), algorithms=["epsilon_greedy"], steps=15, runs=3, seed=2)
    runs = experiment.run()
    summary = experiment.summarize(runs)
    report = render_markdown_report(experiment=experiment, runs=runs, summary=summary)
    # Every arm name appears at least once (configuration + fractions row)
    for arm in make_arms():
        assert arm.name in report


def test_cli_compare_writes_markdown_report(tmp_path: Path) -> None:
    from bandit_kit.cli import main

    destination = tmp_path / "report.md"
    exit_code = main([
        "compare",
        "--arms", "bern:0.10,bern:0.20,bern:0.05",
        "--steps", "15",
        "--runs", "2",
        "--seed", "1",
        "--gamma", "0.25",
        "--temperature-start", "0.7",
        "--output", str(destination),
    ])
    assert exit_code == 0
    assert destination.exists()
    text = destination.read_text(encoding="utf-8")
    assert "Multi-armed bandit experiment" in text
    assert "epsilon_greedy" in text
    assert "exp3" in text
    assert "boltzmann" in text
    assert "kl_ucb" in text
    assert "gamma (exp3): 0.25" in text
    assert "temperature start (boltzmann): 0.7" in text
    assert "KL-UCB c:" in text


def test_cli_compare_rejects_invalid_temperature(tmp_path: Path, capsys) -> None:
    from bandit_kit.cli import main

    exit_code = main([
        "compare",
        "--arms", "bern:0.10,bern:0.20",
        "--steps", "5",
        "--runs", "1",
        "--temperature-start", "0.0",
    ])
    assert exit_code == 2
    assert "temperature_start" in capsys.readouterr().err


def test_cli_compare_rejects_invalid_kl_ucb_c(tmp_path: Path, capsys) -> None:
    from bandit_kit.cli import main

    exit_code = main([
        "compare",
        "--arms", "bern:0.10,bern:0.20",
        "--steps", "5",
        "--runs", "1",
        "--kl-ucb-c", "-1",
    ])
    assert exit_code == 2
    assert "kl_ucb_c" in capsys.readouterr().err


def test_cli_compare_rejects_empty_arm_list(tmp_path: Path, capsys) -> None:
    from bandit_kit.cli import main

    exit_code = main([
        "compare",
        "--arms", "",
        "--steps", "10",
        "--runs", "2",
        "--seed", "1",
    ])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "at least one arm spec" in captured.err