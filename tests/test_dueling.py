import json
import math

import pytest

from bandit_kit import (
    RUCB,
    DoubleThompsonSampling,
    PreferenceMatrix,
    UniformDueling,
    compare_dueling,
    condorcet_winner,
    copeland_winners,
    run_dueling,
)
from bandit_kit.cli import main

UTILS = [0.0, 0.3, 0.6, 1.0, 0.8]


def test_from_utilities_is_bradley_terry_and_valid():
    env = PreferenceMatrix.from_utilities(UTILS)
    p = env.probs
    assert p[3][0] == pytest.approx(1 / (1 + math.exp(-1.0)))
    for i in range(5):
        assert p[i][i] == 0.5
        for j in range(5):
            assert p[i][j] + p[j][i] == pytest.approx(1.0)
    lin = PreferenceMatrix.from_utilities([0.0, 1.0], link="linear", scale=2.0).probs
    assert lin[1][0] == pytest.approx(0.75)
    assert condorcet_winner(p) == 3
    assert env.gaps()[3] == 0.0 and min(env.gaps()) >= 0.0


def test_matrix_validation():
    with pytest.raises(ValueError):
        PreferenceMatrix([[0.5]])
    with pytest.raises(ValueError):
        PreferenceMatrix([[0.5, 0.7], [0.7, 0.5]])
    with pytest.raises(ValueError):
        PreferenceMatrix([[0.5, 0.6, 0.5], [0.4, 0.5]])
    with pytest.raises(ValueError):
        PreferenceMatrix.from_utilities([0, 1], link="probit")
    with pytest.raises(ValueError):
        RUCB(alpha=0.5)


def test_condorcet_cycle_and_copeland():
    cycle = [[0.5, 0.9, 0.2], [0.1, 0.5, 0.8], [0.8, 0.2, 0.5]]
    assert condorcet_winner(cycle) is None
    assert copeland_winners(cycle) == [0, 1, 2]
    with pytest.raises(ValueError, match="Condorcet"):
        PreferenceMatrix(cycle).gaps()
    four = [[0.5, 0.6, 0.6, 0.4], [0.4, 0.5, 0.6, 0.6], [0.4, 0.4, 0.5, 0.6], [0.6, 0.4, 0.4, 0.5]]
    assert condorcet_winner(four) is None
    assert copeland_winners(four) == [0, 1]


def test_regret_bookkeeping():
    env = PreferenceMatrix.from_utilities(UTILS, seed=3)
    res = run_dueling(UniformDueling(seed=1), env, 200)
    gaps = env.gaps()
    assert len(res.pairs) == 200 and all(i != j for i, j in res.pairs)
    strong = sum(0.5 * (gaps[i] + gaps[j]) for i, j in res.pairs)
    weak = sum(min(gaps[i], gaps[j]) for i, j in res.pairs)
    assert res.cumulative_strong_regret == pytest.approx(strong)
    assert res.cumulative_weak_regret == pytest.approx(weak)
    assert all(b >= a for a, b in zip(res.strong_regret, res.strong_regret[1:]))
    with pytest.raises(ValueError):
        run_dueling(RUCB(), env, 0)


@pytest.mark.parametrize("policy_cls", [RUCB, DoubleThompsonSampling])
def test_policies_find_condorcet_winner_with_low_regret(policy_cls):
    env = PreferenceMatrix.from_utilities(UTILS, seed=11)
    res = run_dueling(policy_cls(seed=4), env, 2500)
    base = run_dueling(UniformDueling(seed=4), PreferenceMatrix.from_utilities(UTILS, seed=11), 2500)
    assert res.recommendation == 3
    assert res.cumulative_strong_regret < 0.7 * base.cumulative_strong_regret
    assert res.cumulative_weak_regret <= res.cumulative_strong_regret + 1e-9
    # exploitation: late duels almost always involve the Condorcet winner
    late = res.pairs[-500:]
    assert sum(1 for p in late if 3 in p) > 400
    if policy_cls is DoubleThompsonSampling:
        assert sum(1 for p in late if p == (3, 3)) > 250
    # regret growth slows down (sublinear): second half adds less than first half
    half = len(res.strong_regret) // 2
    first, total = res.strong_regret[half - 1], res.strong_regret[-1]
    assert total - first < first


def test_policies_are_reproducible():
    a = run_dueling(DoubleThompsonSampling(seed=7), PreferenceMatrix.from_utilities(UTILS, seed=2), 300)
    b = run_dueling(DoubleThompsonSampling(seed=7), PreferenceMatrix.from_utilities(UTILS, seed=2), 300)
    assert a.pairs == b.pairs and a.strong_regret == b.strong_regret


def test_compare_dueling_orders_policies():
    p = PreferenceMatrix.from_utilities(UTILS).probs
    table = compare_dueling(p, horizon=1500, runs=3)
    assert set(table) == {"rucb", "dts", "uniform"}
    assert table["dts"]["strong_regret"] < table["uniform"]["strong_regret"]
    assert table["rucb"]["strong_regret"] < table["uniform"]["strong_regret"]
    assert table["dts"]["winner_rate"] == 1.0
    with pytest.raises(ValueError):
        compare_dueling(p, policies=["iff"])


def test_cli_duel(capsys):
    assert main(["duel", "--utilities", "0,0.5,1", "--horizon", "300", "--runs", "2"]) == 0
    out = capsys.readouterr().out
    assert "Condorcet winner: arm 2" in out and "rucb" in out and "dts" in out
    assert main(["duel", "--utilities", "0,1", "--horizon", "50", "--runs", "1",
                 "--policies", "dts", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["condorcet_winner"] == 1 and "dts" in payload["results"]
    assert main(["duel", "--utilities", "0,1", "--policies", "nope"]) == 2
