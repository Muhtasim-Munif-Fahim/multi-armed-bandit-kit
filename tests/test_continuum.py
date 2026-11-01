import json
import math

import pytest

from bandit_kit import (
    CONTINUUM_PAYOFFS,
    ContinuumEnvironment,
    RandomContinuumPolicy,
    UniformDiscretizationTS,
    UniformDiscretizationUCB,
    ZoomingAlgorithm,
    compare_continuum,
    default_discretization,
    run_continuum,
)
from bandit_kit.cli import main


def test_payoffs_are_lipschitz_and_bounded():
    for name, (fn, L) in CONTINUUM_PAYOFFS.items():
        xs = [i / 2000 for i in range(2001)]
        vals = [fn(x) for x in xs]
        assert all(0.0 <= v <= 1.0 for v in vals), name
        slope = max(abs(vals[i + 1] - vals[i]) / (xs[i + 1] - xs[i]) for i in range(2000))
        assert slope <= L + 1e-6, name


def test_environment_optimum_and_noise():
    env = ContinuumEnvironment(CONTINUUM_PAYOFFS["triangle"][0], seed=0)
    assert env.argmax == pytest.approx(0.3)
    assert env.optimum == pytest.approx(0.9)
    draws = [env.pull(0.3) for _ in range(4000)]
    assert set(draws) <= {0.0, 1.0}
    assert sum(draws) / len(draws) == pytest.approx(0.9, abs=0.03)
    g = ContinuumEnvironment(lambda x: x, noise="gaussian", sigma=0.0, seed=1)
    assert g.pull(0.25) == 0.25 and g.optimum == 1.0
    with pytest.raises(ValueError):
        env.pull(1.5)
    with pytest.raises(ValueError):
        ContinuumEnvironment(lambda x: x, noise="cauchy")
    with pytest.raises(ValueError, match="Bernoulli"):
        ContinuumEnvironment(lambda x: 2.0, seed=0).pull(0.5)


def test_default_discretization():
    assert default_discretization(1) == 1
    T = 10000
    assert default_discretization(T) == math.ceil((T / math.log(T)) ** (1 / 3))
    assert default_discretization(T, lipschitz=4.0) > default_discretization(T)


def test_uniform_grid_uses_cell_centres():
    pol = UniformDiscretizationUCB(k=4)
    pol.reset(100)
    assert pol.arms == [0.125, 0.375, 0.625, 0.875]
    with pytest.raises(ValueError):
        UniformDiscretizationUCB(k=0)
    with pytest.raises(ValueError):
        ZoomingAlgorithm(lipschitz=0)


def test_zooming_covers_interval_and_zooms_near_optimum():
    fn, L = CONTINUUM_PAYOFFS["triangle"]
    env = ContinuumEnvironment(fn, seed=3)
    pol = ZoomingAlgorithm(lipschitz=L, seed=3)
    res = run_continuum(pol, env, 20000)
    # every point of [0, 1] lies in some confidence ball
    for x in [i / 500 for i in range(501)]:
        assert any(abs(x - a) * L <= pol.radius(i) + 1e-12 for i, a in enumerate(pol.arms))
    # most plays land near the optimum and the final estimate is accurate
    near = sum(abs(x - 0.3) < 0.1 for x in res.arms) / len(res.arms)
    assert near > 0.8
    assert abs(res.final_estimate() - 0.3) < 0.05
    # arms are denser near the optimum than far away
    close = sum(abs(a - 0.3) < 0.15 for a in pol.arms)
    assert close >= len(pol.arms) / 2


def test_zooming_gap_detection():
    pol = ZoomingAlgorithm(lipschitz=1.0)
    pol.reset(100)
    assert pol._uncovered_point() == 0.5
    pol.arms, pol.counts, pol.sums = [0.5], [10 ** 6], [0.0]
    x = pol._uncovered_point()
    assert x is not None and x < 0.5 - pol.radius(0)
    pol.arms, pol.counts, pol.sums = [0.5], [0], [0.0]
    assert pol._uncovered_point() is None


@pytest.mark.parametrize("cls", [UniformDiscretizationUCB, UniformDiscretizationTS, ZoomingAlgorithm,
                                 RandomContinuumPolicy])
def test_runs_are_reproducible(cls):
    fn, L = CONTINUUM_PAYOFFS["smooth"]
    a = run_continuum(cls(seed=5), ContinuumEnvironment(fn, seed=9), 500)
    b = run_continuum(cls(seed=5), ContinuumEnvironment(fn, seed=9), 500)
    assert a.arms == b.arms and a.rewards == b.rewards
    assert all(r >= -1e-12 for r in a.regret)
    assert a.cumulative_regret[-1] == pytest.approx(a.total_regret)


def test_learners_beat_random_and_have_sublinear_regret():
    table = compare_continuum("triangle", horizon=3000, runs=3)
    for name in ("zooming", "uniform_ucb", "uniform_ts"):
        assert table[name]["regret"] < 0.5 * table["random"]["regret"]
        assert table[name]["argmax_error"] < 0.05
    assert table["random"]["arms"] == 0


def test_zooming_beats_uniform_ucb_at_long_horizon():
    table = compare_continuum("triangle", horizon=30000, runs=1, policies=["zooming", "uniform_ucb"])
    assert table["zooming"]["regret"] < table["uniform_ucb"]["regret"]
    assert table["zooming"]["arms"] < table["uniform_ucb"]["arms"]


def test_gaussian_ts_path():
    fn, _ = CONTINUUM_PAYOFFS["smooth"]
    env = ContinuumEnvironment(fn, noise="gaussian", sigma=0.2, seed=2)
    res = run_continuum(UniformDiscretizationTS(k=10, seed=2), env, 2000)
    assert abs(res.final_estimate() - 0.6) < 0.1


def test_compare_validation():
    with pytest.raises(ValueError, match="payoff"):
        compare_continuum("bumpy")
    with pytest.raises(ValueError, match="policies"):
        compare_continuum(policies=["oracle"])
    with pytest.raises(ValueError):
        compare_continuum(runs=0)


def test_cli_continuum(capsys):
    assert main(["continuum", "--horizon", "500", "--runs", "2"]) == 0
    out = capsys.readouterr().out
    assert "zooming" in out and "argmax: 0.300" in out
    assert main(["continuum", "--horizon", "300", "--runs", "1", "--json", "--payoff", "needle",
                 "--policies", "zooming"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["argmax"] == pytest.approx(0.62) and "zooming" in data["results"]
    assert main(["continuum", "--payoff", "nope"]) == 2
