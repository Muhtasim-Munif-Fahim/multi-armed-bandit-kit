"""Standalone metric helpers for evaluating bandit policies."""

from __future__ import annotations

from typing import Dict, List, Sequence

from .arms import Arm, expected_payoffs


def cumulative_reward(rewards: Sequence[float]) -> List[float]:
    """Return the cumulative reward curve from a per-step reward series."""
    running = 0.0
    out: List[float] = []
    for reward in rewards:
        running += float(reward)
        out.append(running)
    return out


def cumulative_regret(
    rewards: Sequence[float],
    arms: Sequence[Arm],
    *,
    selections: Sequence[int] | None = None,
) -> List[float]:
    """Return the cumulative regret curve for a series of rewards.

    When ``selections`` is provided (one arm index per step), the regret
    is computed against the oracle expected payoff of the selected arm;
    otherwise the oracle is constant at ``max(arm.expected_value)`` so
    each step contributes a constant regret equal to ``oracle - mean(reward)``.
    """
    payoffs = expected_payoffs(arms)
    oracle = max(payoffs.values())
    running = 0.0
    out: List[float] = []
    if selections is None:
        for reward in rewards:
            running += oracle - float(reward)
            out.append(running)
        return out
    for reward, idx in zip(rewards, selections):
        running += oracle - payoffs[arms[idx].name]
        out.append(running)
    return out


def arm_selection_counts(arm_names: Sequence[str]) -> Dict[str, int]:
    """Initialise a zero-filled count map for the given arm names."""
    return {name: 0 for name in arm_names}


def arm_selection_fractions(
    arm_names: Sequence[str],
    selections: Sequence[int],
    arms: Sequence[Arm],
) -> Dict[str, float]:
    """Compute the per-arm selection fractions from a list of arm indices."""
    counts = arm_selection_counts(arm_names)
    for idx in selections:
        counts[arms[idx].name] += 1
    total = sum(counts.values())
    if total == 0:
        return {name: 0.0 for name in arm_names}
    return {name: count / total for name, count in counts.items()}


__all__ = [
    "cumulative_reward",
    "cumulative_regret",
    "arm_selection_counts",
    "arm_selection_fractions",
]