"""Tests for EXP3-IX."""

from __future__ import annotations

import pytest

from bandit_kit import Exp3IX, BernoulliArm, exp3_ix
from bandit_kit.algorithms import BanditStep


def test_exp3_ix_factory_and_name():
    algo = exp3_ix(eta=0.2, seed=0)
    assert isinstance(algo, Exp3IX)
    arms = [BernoulliArm("a", 0.2), BernoulliArm("b", 0.8)]
    algo.reset(arms)
    idx = algo.select_arm(arms, 0)
    assert idx in (0, 1)


def test_exp3_ix_update_changes_weights():
    algo = Exp3IX(eta=0.5, gamma=0.0, seed=1)
    arms = [BernoulliArm("a", 0.1), BernoulliArm("b", 0.9), BernoulliArm("c", 0.5)]
    algo.reset(arms)
    # Force select via update after a select
    idx = algo.select_arm(arms, 0)
    before = list(algo._weights)
    algo.update(arms, BanditStep(arm_index=idx, arm_name="a", reward=1.0, step=0))
    # After reward 1, loss=0 so hat_loss=0 -> weights unchanged for IX on reward 1
    assert algo._weights[idx] == pytest.approx(before[idx])
    # Lossy update should shrink weight
    idx2 = algo.select_arm(arms, 1)
    before2 = list(algo._weights)
    algo.update(arms, BanditStep(arm_index=idx2, arm_name="a", reward=0.0, step=1))
    assert algo._weights[idx2] < before2[idx2] + 1e-12


def test_exp3_ix_rejects_bad_eta():
    with pytest.raises(ValueError, match="eta"):
        Exp3IX(eta=0.0)


def test_exp3_ix_runs_many_steps():
    algo = exp3_ix(eta=0.1, gamma=0.05, seed=42)
    arms = [BernoulliArm("a", 0.2, seed=0), BernoulliArm("b", 0.7, seed=1)]
    algo.reset(arms)
    for t in range(50):
        i = algo.select_arm(arms, t)
        r = arms[i].draw()
        algo.update(arms, BanditStep(arm_index=i, arm_name=arms[i].name, reward=r, step=t))
    assert abs(sum(algo._distribution()) - 1.0) < 1e-9
