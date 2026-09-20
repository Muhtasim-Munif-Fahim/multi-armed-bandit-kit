"""Markdown report rendering for bandit experiments."""

from __future__ import annotations

from typing import Dict, List, Sequence

from .arms import expected_payoffs
from .experiment import BanditExperiment, BanditRunResult, ContextualBanditExperiment


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
    if experiment.algorithms and (
        "boltzmann" in experiment.algorithms or "softmax" in experiment.algorithms
    ):
        lines.append(
            f"- temperature start (boltzmann): {experiment.temperature_start}"
        )
        lines.append(f"- temperature min (boltzmann): {experiment.temperature_min}")
        lines.append(
            f"- temperature decay (boltzmann): {experiment.temperature_decay}"
        )
    if experiment.algorithms and "linucb" in experiment.algorithms:
        lines.append(
            f"- LinUCB alpha: {experiment.linucb_alpha} (intercept-only ridge-UCB; "
            f"ridge={experiment.linucb_ridge})"
        )
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
        elif algo == "linucb":
            lines.append(
                "- linucb (disjoint linear UCB) estimates a per-arm linear payoff "
                f"model with exploration bonus alpha={experiment.linucb_alpha}. In the "
                "stationary harness it uses the intercept context [1.0] (ridge-UCB)."
            )
        elif algo in ("boltzmann", "softmax"):
            lines.append(
                "- boltzmann (softmax) samples arms with probability proportional "
                "to exp(Q / tau), where Q is the empirical mean reward and tau "
                f"decays from {experiment.temperature_start} to "
                f"{experiment.temperature_min} "
                f"(decay={experiment.temperature_decay})."
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
        elif hasattr(arm, "theta"):
            theta = ", ".join(f"{value:.3f}" for value in arm.theta)
            parts.append(f"{arm.name} ({cls}, theta=[{theta}])")
        else:
            parts.append(f"{arm.name} ({cls}, EV={arm.expected_value:.3f})")
    return "; ".join(parts)


def render_contextual_markdown_report(
    *,
    experiment,
    runs: Sequence[BanditRunResult],
    summary: Sequence[Dict[str, object]],
) -> str:
    """Render a Markdown report for a linear contextual-bandit experiment."""
    if not isinstance(experiment, ContextualBanditExperiment):
        raise TypeError("experiment must be a ContextualBanditExperiment")

    lines: List[str] = []
    lines.append("# Contextual linear bandit experiment")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Arms: {_fmt_arms(experiment.arms)}")
    lines.append(f"- Context dimension: {experiment.dimension}")
    lines.append(f"- Intercept feature: {experiment.context_intercept}")
    lines.append(f"- Algorithms: {', '.join(experiment.algorithms)}")
    lines.append(f"- Steps per run: {experiment.steps}")
    lines.append(f"- Independent runs per algorithm: {experiment.runs}")
    lines.append(f"- Base seed: {experiment.seed}")
    if "linucb" in experiment.algorithms:
        lines.append(f"- LinUCB alpha: {experiment.linucb_alpha}")
        lines.append(f"- LinUCB ridge: {experiment.linucb_ridge}")
    if "epsilon_greedy" in experiment.algorithms:
        lines.append(f"- epsilon (epsilon-greedy): {experiment.epsilon}")
    if "boltzmann" in experiment.algorithms or "softmax" in experiment.algorithms:
        lines.append(
            f"- temperature start (boltzmann): {experiment.temperature_start}"
        )
        lines.append(f"- temperature min (boltzmann): {experiment.temperature_min}")
        lines.append(
            f"- temperature decay (boltzmann): {experiment.temperature_decay}"
        )
    lines.append("")
    lines.append(
        "Regret is context-conditional: at each step the oracle is the arm "
        "with the largest ``theta · x_t``."
    )
    lines.append("")

    lines.append("## Per-algorithm summary")
    lines.append("")
    lines.append(
        "| Algorithm | Runs | Mean reward | Std reward | Mean regret | Std regret | Oracle hit rate |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summary:
        lines.append(
            "| {algorithm} | {runs} | {reward:.4f} | {std_reward:.4f} | "
            "{regret:.4f} | {std_regret:.4f} | {hit:.3f} |".format(
                algorithm=row["algorithm"],
                runs=row["runs"],
                reward=row["mean_final_reward"],
                std_reward=row["std_final_reward"],
                regret=row["mean_final_regret"],
                std_regret=row["std_final_regret"],
                hit=row["mean_oracle_hit_rate"],
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
        if algo == "linucb":
            lines.append(
                "- linucb (disjoint LinUCB) observes the context vector at every "
                f"step and picks the arm with the largest linear UCB "
                f"(alpha={experiment.linucb_alpha})."
            )
        elif algo == "ucb1":
            lines.append(
                "- ucb1 ignores context and treats rewards as stationary; it is a "
                "baseline showing the cost of dropping the linear payoff model."
            )
        elif algo == "epsilon_greedy":
            lines.append(
                "- epsilon_greedy ignores context and explores uniformly with "
                f"probability epsilon={experiment.epsilon}."
            )
        elif algo == "exp3":
            lines.append(
                "- exp3 ignores context and uses an adversarial exponential-weights policy."
            )
        elif algo in ("boltzmann", "softmax"):
            lines.append(
                "- boltzmann (softmax) ignores context and samples from a "
                "temperature-scaled softmax over empirical mean rewards."
            )
        else:
            lines.append(
                f"- {algo} is a non-contextual policy run on contextual rewards "
                "as a baseline."
            )
    lines.append("")
    return "\n".join(lines)


__all__ = ["render_markdown_report", "render_contextual_markdown_report"]