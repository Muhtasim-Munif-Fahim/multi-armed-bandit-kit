"""Plotting helpers for bandit experiments, built on matplotlib.

These functions render cumulative-reward and cumulative-regret curves
from :class:`~bandit_kit.experiment.BanditRunResult` objects.  Matplotlib
is imported lazily so the rest of the package remains import-safe without
a plotting backend installed.
"""

from __future__ import annotations

from typing import Sequence, Tuple

from .arms import Arm, expected_payoffs
from .experiment import BanditRunResult
from .metrics import cumulative_regret, cumulative_reward

try:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:  # pragma: no cover - exercised only when matplotlib missing
    plt = None  # type: ignore[assignment]
    _HAS_MPL = False

import matplotlib.figure as _figure  # noqa: E402


__all__ = [
    "HAS_MATPLOTLIB",
    "plot_regret_curves",
    "plot_reward_curves",
    "plot_arm_selection",
    "save_all_plots",
]

HAS_MATPLOTLIB = _HAS_MPL


def _ensure_matplotlib() -> None:
    if not _HAS_MPL:
        raise ImportError(
            "matplotlib is required for visualization; install with: pip install matplotlib"
        )


def plot_regret_curves(
    arms: Sequence[Arm],
    results: Sequence[BanditRunResult],
    ax=None,
) -> "plt.Axes":
    """Plot the mean cumulative-regret curve (±1 SEM) per algorithm.

    Parameters
    ----------
    arms:
        The arm set used to produce *results*.
    results:
        One or more :class:`BanditRunResult` objects (typically multiple
        runs of the same algorithm, or results from several algorithms).
    ax:
        An optional matplotlib ``Axes`` to draw on.
    """
    _ensure_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))  # type: ignore[union-attr]
    payoffs = expected_payoffs(arms)
    oracle = max(payoffs.values())

    by_algo: dict[str, list[BanditRunResult]] = {}
    for result in results:
        by_algo.setdefault(result.algorithm, []).append(result)

    for algo_name, runs in sorted(by_algo.items()):
        curves = [
            cumulative_regret(
                r.rewards,
                arms,
                selections=[step.arm_index for step in r.steps],
            )
            for r in runs
        ]
        mean, sem = _mean_and_sem(curves)
        steps = range(1, len(mean) + 1)
        ax.plot(steps, mean, label=algo_name)
        lower = [m - s for m, s in zip(mean, sem)]
        upper = [m + s for m, s in zip(mean, sem)]
        ax.fill_between(steps, lower, upper, alpha=0.15)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative regret")
    ax.set_title("Cumulative regret by algorithm")
    ax.legend()
    return ax


def plot_reward_curves(
    results: Sequence[BanditRunResult],
    ax=None,
) -> "plt.Axes":
    """Plot the mean cumulative-reward curve (±1 SEM) per algorithm."""
    _ensure_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))  # type: ignore[union-attr]

    by_algo: dict[str, list[BanditRunResult]] = {}
    for result in results:
        by_algo.setdefault(result.algorithm, []).append(result)

    for algo_name, runs in sorted(by_algo.items()):
        curves = [cumulative_reward(r.rewards) for r in runs]
        mean, sem = _mean_and_sem(curves)
        steps = range(1, len(mean) + 1)
        ax.plot(steps, mean, label=algo_name)
        lower = [m - s for m, s in zip(mean, sem)]
        upper = [m + s for m, s in zip(mean, sem)]
        ax.fill_between(steps, lower, upper, alpha=0.15)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative reward")
    ax.set_title("Cumulative reward by algorithm")
    ax.legend()
    return ax


def plot_arm_selection(
    arms: Sequence[Arm],
    results: Sequence[BanditRunResult],
    ax=None,
) -> "plt.Axes":
    """Plot mean per-arm selection counts as stacked bars per algorithm."""
    _ensure_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))  # type: ignore[union-attr]

    by_algo: dict[str, list[BanditRunResult]] = {}
    for result in results:
        by_algo.setdefault(result.algorithm, []).append(result)

    arm_names = [arm.name for arm in arms]
    n_algos = len(by_algo)
    width = 0.8 / max(n_algos, 1)
    x_base = range(len(arm_names))

    for i, (algo_name, runs) in enumerate(sorted(by_algo.items())):
        counts_matrix: list[list[int]] = []
        for r in runs:
            counts = [0] * len(arm_names)
            for step in r.steps:
                counts[step.arm_index] += 1
            counts_matrix.append(counts)
        mean_counts = [
            sum(col) / len(col) for col in zip(*counts_matrix)
        ]
        offsets = [x + i * width - 0.4 for x in x_base]
        ax.bar(offsets, mean_counts, width=width, label=algo_name)
    ax.set_xticks(x_base)
    ax.set_xticklabels(arm_names)
    ax.set_xlabel("Arm")
    ax.set_ylabel("Mean selection count")
    ax.set_title("Arm selection counts by algorithm")
    ax.legend()
    return ax


def _mean_and_sem(
    curves: Sequence[Sequence[float]],
) -> Tuple[list[float], list[float]]:
    """Compute element-wise mean and standard error across curves."""
    if not curves:
        return [], []
    n = len(curves)
    if n == 1:
        return list(curves[0]), [0.0] * len(curves[0])
    length = min(len(c) for c in curves)
    mean: list[float] = []
    sem: list[float] = []
    for pos in range(length):
        values = [c[pos] for c in curves]
        m = sum(values) / n
        if n < 2:
            s = 0.0
        else:
            variance = sum((v - m) ** 2 for v in values) / (n - 1)
            s = (variance ** 0.5) / (n ** 0.5)
        mean.append(m)
        sem.append(s)
    return mean, sem


def save_all_plots(
    arms: Sequence[Arm],
    results: Sequence[BanditRunResult],
    output_dir,
) -> list[str]:
    """Write three PNG plots (regret, reward, arm selection) to *output_dir*.

    Returns a list of the saved file paths.
    """
    _ensure_matplotlib()
    import os

    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    paths: list[str] = []
    reg_ax = plot_regret_curves(arms, results)
    reg_path = os.path.join(output_dir, "regret_curves.png")
    reg_ax.figure.savefig(reg_path, dpi=150, bbox_inches="tight")
    paths.append(reg_path)
    plt.close(reg_ax.figure)  # type: ignore[union-attr]

    rew_ax = plot_reward_curves(results)
    rew_path = os.path.join(output_dir, "reward_curves.png")
    rew_ax.figure.savefig(rew_path, dpi=150, bbox_inches="tight")
    paths.append(rew_path)
    plt.close(rew_ax.figure)  # type: ignore[union-attr]

    sel_ax = plot_arm_selection(arms, results)
    sel_path = os.path.join(output_dir, "arm_selection.png")
    sel_ax.figure.savefig(sel_path, dpi=150, bbox_inches="tight")
    paths.append(sel_path)
    plt.close(sel_ax.figure)  # type: ignore[union-attr]

    return paths
