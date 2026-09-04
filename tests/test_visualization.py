"""Unit tests for bandit_kit.visualization."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import pytest

from bandit_kit import (
    BernoulliArm,
    HAS_MATPLOTLIB,
    plot_arm_selection,
    plot_regret_curves,
    plot_reward_curves,
    save_all_plots,
)
from bandit_kit.experiment import BanditExperiment


def make_arms():
    return [
        BernoulliArm(name="A", p=0.10, seed=111),
        BernoulliArm(name="B", p=0.20, seed=222),
        BernoulliArm(name="C", p=0.05, seed=333),
    ]


def make_runs():
    experiment = BanditExperiment(
        arms=make_arms(),
        algorithms=["epsilon_greedy", "ucb1"],
        steps=50,
        runs=5,
        seed=42,
    )
    return experiment, experiment.run()


def test_has_matplotlib_constant():
    assert isinstance(HAS_MATPLOTLIB, bool)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_plot_regret_curves_returns_axes():
    import matplotlib.pyplot as plt

    experiment, runs = make_runs()
    fig, ax = plt.subplots()
    result = plot_regret_curves(experiment.arms, runs, ax=ax)
    assert result is ax
    assert len(result.lines) > 0
    plt.close(fig)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_plot_reward_curves_returns_axes():
    import matplotlib.pyplot as plt

    experiment, runs = make_runs()
    fig, ax = plt.subplots()
    result = plot_reward_curves(runs, ax=ax)
    assert result is ax
    assert len(result.lines) > 0
    plt.close(fig)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_plot_arm_selection_returns_axes():
    import matplotlib.pyplot as plt

    experiment, runs = make_runs()
    fig, ax = plt.subplots()
    result = plot_arm_selection(experiment.arms, runs, ax=ax)
    assert result is ax
    plt.close(fig)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_save_all_plots_writes_three_files(tmp_path):
    experiment, runs = make_runs()
    paths = save_all_plots(experiment.arms, runs, tmp_path)
    assert len(paths) == 3
    for path in paths:
        assert path.endswith(".png")
        import os
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_plot_regret_curves_no_ax_creates_one():
    experiment, runs = make_runs()
    ax = plot_regret_curves(experiment.arms, runs)
    import matplotlib.pyplot as plt
    plt.close(ax.figure)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_plot_reward_curves_no_ax_creates_one():
    _, runs = make_runs()
    ax = plot_reward_curves(runs)
    import matplotlib.pyplot as plt
    plt.close(ax.figure)
