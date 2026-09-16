"""Markdown report rendering for bandit experiments."""

from __future__ import annotations

from typing import Dict, List, Sequence

from .arms import expected_payoffs
from .experiment import BanditExperiment, BanditRunResult


def render_markdown_report(
    *,
    experiment: BanditExperiment,
    runs: Sequence[BanditRunResult],
    summary: Sequence[Dict[str, object]],
) -> str:
    """Render a Markdown report comparing the bundled algorithms on the given runs.

    The report covers:
      - the experiment configuration (arms, algorithms, steps, runs, seed)
      - per-algorithm mean +/- stddev final cumulative reward and regret
      - per-algorithm per-arm selection fractions
      - the oracle expected payoff for context
    """
    payoffs = expected_payoffs(experiment.arms)
    oracle = max(payoffs.values())

    lines: List[str] = []
    lines.append("# Multi-armed bandit experiment")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Arms: {_fmt_arms(experiment.arms)}")
    lines.append(f"- Oracle expected payoff: {oracle:.4f}")
    lines.append(f"- Algorithms: {', '.join(experiment.algorithms)}")
    lines.append(f"- Steps per run: {experiment.steps}")
    lines.append(f"- Independent runs per algorithm: {experiment.runs}")
    lines.append(f"- Base seed: {experiment.seed}")
    if experiment.algorithms and "epsilon_greedy" in experiment.algorithms:
        lines.append(f"- epsilon (epsilon-greedy): {experiment.epsilon}")
    if experiment.algorithms and "exp3" in experiment.algorithms:
        lines.append(f"- gamma (exp3): {experiment.gamma}")
    lines.append("")

    lines.append("## Per-algorithm summary")
    lines.append("")
    lines.append("| Algorithm | Runs | Mean reward | Std reward | Mean regret | Std regret |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in summary:
        lines.append(
            "| {algorithm} | {runs} | {reward:.4f} | {std_reward:.4f} | {regret:.4f} | {std_regret:.4f} |".format(
                algorithm=row["algorithm"],
                runs=row["runs"],
                reward=row["mean_final_reward"],
                std_reward=row["std_final_reward"],
                regret=row["mean_final_regret"],
                std_regret=row["std_final_regret"],
            )
        )
    lines.append("")

    lines.append("## Arm selection fractions")
    lines.append("")
    arm_names = [arm.name for arm in experiment.arms]
    header = "| Algorithm | " + " | ".join(arm_names) + " |"
    sep = "| --- |" + "| ---: |" * len(arm_names)
    lines.append(header)
    lines.append(sep)
    for row in summary:
        fractions: Dict[str, float] = row["mean_arm_selection_fraction"]  # type: ignore[assignment]
        cells = " | ".join(f"{fractions[name]:.3f}" for name in arm_names)
        lines.append(f"| {row['algorithm']} | {cells} |")
    lines.append("")

    lines.append("## Per-algorithm notes")
    lines.append("")
    for row in summary:
        algo = row["algorithm"]
        if algo == "epsilon_greedy":
            lines.append(
                f"- epsilon_greedy explores with probability epsilon={experiment.epsilon} and otherwise "
                "exploits the arm with the highest empirical mean reward."
            )
        elif algo == "ucb1":
            lines.append(
                "- ucb1 picks the arm with the largest upper-confidence bound; ties on the UCB index "
                "break by arm order."
            )
        elif algo == "thompson":
            lines.append(
                "- thompson sampling maintains a Beta(1, 1) posterior per Bernoulli arm and falls back "
                "to expected_value for non-Bernoulli arms."
            )
        elif algo == "exp3":
            lines.append(
                "- exp3 (adversarial bandit) samples from a mixture of the exponential-weight "
                f"distribution and the uniform distribution (gamma={experiment.gamma}); weights "
                "are updated with importance-weighted rewards clipped to [0, 1]."
            )
    lines.append("")
    return "\n".join(lines)


def _fmt_arms(arms: Sequence) -> str:
    parts = []
    for arm in arms:
        cls = type(arm).__name__
        if hasattr(arm, "p"):
            parts.append(f"{arm.name} ({cls}, p={arm.p:.3f})")
        elif hasattr(arm, "mean"):
            parts.append(f"{arm.name} ({cls}, mean={arm.mean:.3f}, std={arm.std:.3f})")
        else:
            parts.append(f"{arm.name} ({cls}, EV={arm.expected_value:.3f})")
    return "; ".join(parts)


__all__ = ["render_markdown_report"]