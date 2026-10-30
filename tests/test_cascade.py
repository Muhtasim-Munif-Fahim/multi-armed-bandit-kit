import json
import math

import pytest

from bandit_kit import (
    CascadeEnvironment,
    CascadeEpsilonGreedy,
    CascadeKLUCB,
    CascadePolicy,
    CascadeTS,
    CascadeUCB1,
    RandomRanker,
    cascade_click_probability,
    compare_cascade,
    run_cascade,
)
from bandit_kit.cli import main

W = [0.5, 0.4, 0.1, 0.05, 0.05, 0.02]


def test_click_probability_formula():
    assert cascade_click_probability(W, [0, 1]) == pytest.approx(1 - 0.5 * 0.6)
    assert cascade_click_probability(W, [1, 0]) == pytest.approx(0.7)  # order-invariant
    assert cascade_click_probability([0.0, 0.0], [0]) == 0.0


def test_environment_click_distribution_follows_cascade():
    env = CascadeEnvironment([0.3, 0.5, 0.2], k=3, seed=1)
    n = 40000
    counts = {0: 0, 1: 0, 2: 0, None: 0}
    for _ in range(n):
        counts[env.click([0, 1, 2])] += 1
    assert counts[0] / n == pytest.approx(0.3, abs=0.01)
    assert counts[1] / n == pytest.approx(0.7 * 0.5, abs=0.01)
    assert counts[2] / n == pytest.approx(0.7 * 0.5 * 0.2, abs=0.01)
    assert counts[None] / n == pytest.approx(0.7 * 0.5 * 0.8, abs=0.01)
    assert env.optimal_ranking() == [1, 0, 2]


def test_environment_validation():
    with pytest.raises(ValueError):
        CascadeEnvironment([0.5], k=1)
    with pytest.raises(ValueError):
        CascadeEnvironment([0.5, 1.2], k=1)
    with pytest.raises(ValueError):
        CascadeEnvironment([0.5, 0.2], k=3)
    env = CascadeEnvironment(W, k=2)
    for bad in ([0], [0, 0], [0, 9]):
        with pytest.raises(ValueError):
            env.click(bad)


def test_partial_feedback_update():
    pol = CascadeUCB1(seed=0)
    pol.reset(5, 3)
    pol.update([4, 2, 0], 1)  # item 4 skipped, item 2 clicked, item 0 unobserved
    assert pol.observations == [0, 0, 1, 0, 1]
    assert pol.clicks == [0, 0, 1, 0, 0]
    pol.update([1, 3, 0], None)  # no click: all three examined
    assert pol.observations == [1, 1, 1, 1, 1]
    assert pol.clicks == [0, 0, 1, 0, 0]
    assert pol.t == 2


def test_unobserved_items_are_explored_first():
    for pol in (CascadeUCB1(seed=3), CascadeKLUCB(seed=3), CascadeEpsilonGreedy(epsilon=0.0, seed=3)):
        pol.reset(6, 2)
        pol.clicks = [5, 0, 0, 0, 0, 0]
        pol.observations = [5, 1, 0, 1, 0, 1]
        pol.t = 12
        assert set(pol.select()) == {2, 4}
    ucb = CascadeUCB1(seed=0)
    ucb.reset(2, 1)
    ucb.clicks, ucb.observations, ucb.t = [3, 1], [10, 2], 12
    s = ucb.scores()
    assert s[0] == pytest.approx(0.3 + math.sqrt(1.5 * math.log(12) / 10))
    assert s[1] == pytest.approx(0.5 + math.sqrt(1.5 * math.log(12) / 2))


def test_kl_ucb_scores_are_upper_bounds_and_tighter_than_ucb1():
    kl, ucb = CascadeKLUCB(seed=0), CascadeUCB1(seed=0)
    for p in (kl, ucb):
        p.reset(3, 1)
        p.clicks, p.observations, p.t = [2, 30, 1], [50, 60, 400], 500
    for m, u, v in zip(kl.means(), kl.scores(), ucb.scores()):
        assert m <= u <= 1.0
        assert u < v  # KL-UCB is tighter for small means


def test_learning_policies_find_optimal_list_and_beat_random():
    table = compare_cascade(W, k=2, horizon=1500, runs=4, seed=5)
    assert set(table) == {"cascade_ucb1", "cascade_kl_ucb", "cascade_ts", "cascade_epsilon_greedy", "random"}
    for name in ("cascade_kl_ucb", "cascade_ts"):
        assert table[name]["optimal_rate"] == 1.0
        assert table[name]["regret"] < 0.25 * table["random"]["regret"]
    assert table["cascade_ucb1"]["regret"] < table["random"]["regret"]


def test_run_cascade_regret_bookkeeping_and_determinism():
    env = CascadeEnvironment(W, k=2, seed=11)
    res = run_cascade(CascadeTS(seed=2), env, 300)
    assert len(res.regret) == len(res.rankings) == len(res.click_positions) == 300
    assert all(r >= -1e-12 for r in res.regret)
    assert res.cumulative_regret[-1] == pytest.approx(res.total_regret)
    assert 0.0 <= res.click_rate <= 1.0
    again = run_cascade(CascadeTS(seed=2), CascadeEnvironment(W, k=2, seed=11), 300)
    assert again.rankings == res.rankings
    late = sum(res.regret[-100:]) / 100
    early = sum(res.regret[:100]) / 100
    assert late < early
    with pytest.raises(ValueError):
        run_cascade(RandomRanker(), env, 0)


def test_policy_parameter_validation():
    with pytest.raises(ValueError):
        CascadeUCB1(c=0)
    with pytest.raises(ValueError):
        CascadeKLUCB(c=-1)
    with pytest.raises(ValueError):
        CascadeEpsilonGreedy(epsilon=1.5)
    with pytest.raises(ValueError, match="unknown"):
        compare_cascade(W, 2, policies=["nope"])
    with pytest.raises(NotImplementedError):
        p = CascadePolicy()
        p.reset(3, 1)
        p.select()


def test_cli_cascade_table_json_and_errors(capsys):
    assert main(["cascade", "--attractions", "0.5,0.4,0.1,0.05", "--k", "2", "--horizon", "200", "--runs", "2"]) == 0
    out = capsys.readouterr().out
    assert "optimal list: [0, 1]" in out and "cascade_ts" in out
    assert main(["cascade", "--attractions", "0.5,0.4,0.1", "--k", "1", "--horizon", "100", "--runs", "1",
                 "--policies", "cascade_ts,random", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["optimal_ranking"] == [0] and set(data["results"]) == {"cascade_ts", "random"}
    assert main(["cascade", "--attractions", "0.5,0.4", "--k", "3"]) == 2
    assert "cascade:" in capsys.readouterr().err
