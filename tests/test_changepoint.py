import json
import math

import pytest

from bandit_kit import (
    CUSUMUCB,
    MUCB,
    UCB1,
    BanditStep,
    PiecewiseBernoulli,
    compare_piecewise,
    mucb_threshold,
    run_piecewise,
)
from bandit_kit.cli import main

MEANS = [[0.9, 0.5, 0.2], [0.1, 0.5, 0.9]]


def test_environment_segments_and_means():
    env = PiecewiseBernoulli(MEANS, [100], seed=1)
    assert env.segment(0) == 0 and env.segment(99) == 0 and env.segment(100) == 1
    assert env.mean(0, 50) == 0.9 and env.mean(0, 150) == 0.1
    assert env.best_mean(10) == 0.9 and env.best_mean(500) == 0.9
    assert all(env.pull(2, t) in (0.0, 1.0) for t in range(20))
    assert [a.name for a in env.arms()] == ["arm0", "arm1", "arm2"]


@pytest.mark.parametrize(
    "means,bps",
    [([], []), ([[0.5], [0.5, 0.2]], [10]), ([[1.5]], []), ([[0.5], [0.4]], []), ([[0.5], [0.4], [0.3]], [10, 5])],
)
def test_environment_validation(means, bps):
    with pytest.raises(ValueError):
        PiecewiseBernoulli(means, bps)


def test_cusum_detects_change_and_resets_arm_only():
    pol = CUSUMUCB(m=10, eps=0.05, h=3.0, alpha=0.0, seed=0)
    arms = PiecewiseBernoulli([[0.5, 0.5]], []).arms()
    pol.reset(arms)
    for t in range(30):
        pol.update(arms, BanditStep(0, "arm0", 1.0, t))
    pol.update(arms, BanditStep(1, "arm1", 1.0, 30))
    assert pol.alarms == []
    t = 31
    while not pol.alarms and t < 100:
        pol.update(arms, BanditStep(0, "arm0", 0.0, t))
        t += 1
    # u = 1, each 0 adds 0.95 to g-: alarm on the 4th zero
    assert pol.alarms == [34]
    assert pol._counts == [0, 1]


def test_mucb_threshold_and_restart():
    assert mucb_threshold(100, 3, 1000) == pytest.approx(math.sqrt(50 * math.log(2 * 3 * 1000 ** 2)))
    pol = MUCB(w=10, b=3.0, gamma=0.0, horizon=100, seed=0)
    arms = PiecewiseBernoulli([[0.5, 0.5]], []).arms()
    pol.reset(arms)
    for t in range(5):
        pol.update(arms, BanditStep(0, "arm0", 0.0, t))
    for t in range(5, 9):
        pol.update(arms, BanditStep(0, "arm0", 1.0, t))
    assert pol.alarms == []  # window not full yet
    pol.update(arms, BanditStep(0, "arm0", 1.0, 9))
    assert pol.alarms == [9]
    assert pol._counts == [0, 0] and pol._tau == 10


def test_mucb_forced_exploration_round_robin():
    pol = MUCB(w=4, gamma=0.5, horizon=100, seed=0)
    arms = PiecewiseBernoulli([[0.5, 0.5, 0.5]], []).arms()
    pol.reset(arms)
    # period = floor(3 / 0.5) = 6 -> steps 0,1,2 forced to arms 0,1,2
    assert [pol.select_arm(arms, t) for t in (0, 1, 2, 6, 7, 8)] == [0, 1, 2, 0, 1, 2]


@pytest.mark.parametrize(
    "kwargs",
    [dict(m=0), dict(eps=-1), dict(h=0), dict(alpha=2)],
)
def test_cusum_validation(kwargs):
    with pytest.raises(ValueError):
        CUSUMUCB(**kwargs)


@pytest.mark.parametrize("kwargs", [dict(w=3), dict(w=1), dict(gamma=1.0), dict(horizon=0), dict(b=0.0)])
def test_mucb_validation(kwargs):
    with pytest.raises(ValueError):
        MUCB(**kwargs)


def test_run_piecewise_regret_and_reproducibility():
    env1 = PiecewiseBernoulli(MEANS, [500], seed=3)
    env2 = PiecewiseBernoulli(MEANS, [500], seed=3)
    r1 = run_piecewise(CUSUMUCB(seed=4), env1, 1000)
    r2 = run_piecewise(CUSUMUCB(seed=4), env2, 1000)
    assert r1.choices == r2.choices and r1.alarms == r2.alarms
    assert len(r1.dynamic_regret) == 1000
    assert all(b >= a for a, b in zip(r1.dynamic_regret, r1.dynamic_regret[1:]))
    with pytest.raises(ValueError):
        run_piecewise(UCB1(), env1, 0)


def test_active_policies_beat_ucb1_after_abrupt_change():
    # the heavily-sampled best arm degrades: UCB1's long memory keeps it on arm 0
    table = compare_piecewise([[0.9, 0.5], [0.3, 0.5]], [1500], horizon=3000, runs=4,
                              policies=("cusum_ucb", "m_ucb", "ucb1"), seed=0, w=40, b=8.0)
    assert table["cusum_ucb"]["dynamic_regret"] < table["ucb1"]["dynamic_regret"]
    assert table["m_ucb"]["dynamic_regret"] < table["ucb1"]["dynamic_regret"]
    assert table["cusum_ucb"]["detection_rate"] == 1.0
    assert table["m_ucb"]["detection_rate"] == 1.0
    assert table["ucb1"]["alarms"] == 0.0 and math.isnan(table["ucb1"]["mean_delay"])


def test_detection_delays():
    env = PiecewiseBernoulli(MEANS, [500], seed=0)
    res = run_piecewise(CUSUMUCB(h=5.0, seed=0), env, 1500)
    delays = res.detection_delays([500])
    assert len(delays) == 1 and 0 <= delays[0] < 300


def test_compare_unknown_policy():
    with pytest.raises(ValueError):
        compare_piecewise(MEANS, [100], policies=("nope",))


def test_cli_piecewise(capsys):
    rc = main(["piecewise", "--means", "0.9,0.2;0.2,0.9", "--breakpoints", "300", "--horizon", "600",
               "--runs", "2", "--policies", "cusum_ucb,ucb1", "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert set(data) == {"cusum_ucb", "ucb1"}
    rc = main(["piecewise", "--means", "0.9,0.2;0.2,0.9", "--breakpoints", "300", "--horizon", "600", "--runs", "1"])
    assert rc == 0 and "m_ucb" in capsys.readouterr().out
    assert main(["piecewise", "--means", "0.9,0.2", "--breakpoints", "300"]) == 2
