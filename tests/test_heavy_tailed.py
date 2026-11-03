import math
import statistics

import pytest

from bandit_kit import RobustUCB, StudentTArm, median_of_means, robust_ucb, truncated_mean
from bandit_kit.algorithms import UCB1, BanditStep
from bandit_kit.arms import arm_from_spec
from bandit_kit.experiment import BanditExperiment, available_algorithms


def test_median_of_means_ignores_single_outlier():
    xs = [1.0] * 9 + [1e9]
    assert median_of_means(xs, 5) == pytest.approx(1.0)
    assert median_of_means([1.0, 2.0, 3.0, 4.0], 1) == pytest.approx(2.5)
    assert median_of_means([1.0, 3.0, 5.0, 7.0], 2) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        median_of_means([1.0], 2)
    with pytest.raises(ValueError):
        median_of_means([1.0, 2.0], 0)


def test_truncated_mean_drops_extremes():
    xs = [1.0] * 9 + [1e6]
    # threshold for i=10 with u=4, eps=1, log=2 is sqrt(20) ~ 4.5, so 1e6 is dropped.
    assert truncated_mean(xs, epsilon=1.0, moment_bound=4.0, log_inv_delta=2.0) == pytest.approx(0.9)
    # a huge moment bound keeps everything (plain mean)
    plain = truncated_mean([1.0, 2.0, 3.0], epsilon=1.0, moment_bound=1e9, log_inv_delta=1.0)
    assert plain == pytest.approx(2.0)
    with pytest.raises(ValueError):
        truncated_mean([], epsilon=1.0, moment_bound=1.0, log_inv_delta=1.0)
    with pytest.raises(ValueError):
        truncated_mean([1.0], epsilon=1.5, moment_bound=1.0, log_inv_delta=1.0)


def test_student_t_arm_and_spec():
    arm = StudentTArm("t", mean=2.0, df=3.0, scale=0.5, seed=1)
    draws = [arm.draw() for _ in range(4000)]
    assert statistics.median(draws) == pytest.approx(2.0, abs=0.05)
    assert arm.expected_value == 2.0
    again = StudentTArm("t", mean=2.0, df=3.0, scale=0.5, seed=1)
    assert [again.draw() for _ in range(5)] == draws[:5]
    spec = arm_from_spec("t:0.5:1.5")
    assert isinstance(spec, StudentTArm) and spec.df == 1.5 and spec.scale == 1.0
    assert arm_from_spec("t:0:2:3").scale == 3.0
    with pytest.raises(ValueError):
        StudentTArm("bad", 0.0, df=1.0)
    with pytest.raises(ValueError):
        StudentTArm("bad", 0.0, df=2.0, scale=0.0)


def test_validation():
    with pytest.raises(ValueError):
        RobustUCB(epsilon=0.0)
    with pytest.raises(ValueError):
        RobustUCB(moment_bound=-1.0)
    with pytest.raises(ValueError):
        RobustUCB(estimator="mean")
    with pytest.raises(ValueError):
        RobustUCB(confidence_scale=0.0)


def _play(algo, arms, steps):
    algo.reset(arms)
    best = max(a.expected_value for a in arms)
    regret = 0.0
    for t in range(steps):
        i = algo.select_arm(arms, t)
        r = arms[i].draw()
        algo.update(arms, BanditStep(arm_index=i, arm_name=arms[i].name, reward=r, step=t))
        regret += best - arms[i].expected_value
    return regret


def test_initial_round_robin_and_index_shrinks():
    arms = [StudentTArm(str(i), 0.1 * i, 3.0, seed=i) for i in range(3)]
    algo = RobustUCB(seed=0)
    algo.reset(arms)
    picks = []
    for t in range(3):
        i = algo.select_arm(arms, t)
        picks.append(i)
        algo.update(arms, BanditStep(arm_index=i, arm_name=arms[i].name, reward=0.0, step=t))
    assert picks == [0, 1, 2]
    assert algo.index(0) < math.inf
    with pytest.raises(IndexError):
        algo.index(5)


@pytest.mark.parametrize("estimator", ["median_of_means", "truncated"])
def test_finds_best_arm(estimator):
    arms = [StudentTArm("a", 0.0, 2.5, seed=1), StudentTArm("b", 1.0, 2.5, seed=2)]
    algo = RobustUCB(epsilon=1.0, moment_bound=4.0, estimator=estimator, confidence_scale=0.3)
    _play(algo, arms, 1500)
    counts = algo.pull_counts()
    assert counts[1] > 0.7 * sum(counts)


def test_beats_ucb1_under_infinite_variance():
    def arms(seed):
        return [
            StudentTArm("a", 0.0, 1.6, seed=seed * 3),
            StudentTArm("b", 0.5, 1.6, seed=seed * 3 + 1),
            StudentTArm("c", 0.2, 1.6, seed=seed * 3 + 2),
        ]

    robust, plain = [], []
    for seed in range(16):
        robust.append(_play(RobustUCB(0.5, 3.0, "truncated", 0.2), arms(seed), 1500))
        plain.append(_play(UCB1(), arms(seed), 1500))
    assert statistics.mean(robust) < statistics.mean(plain)
    assert max(robust) < max(plain)


def test_registered_in_experiment():
    assert "robust_ucb" in available_algorithms()
    exp = BanditExperiment(
        arms=[arm_from_spec("t:0:2"), arm_from_spec("t:1:2")],
        algorithms=["robust_ucb"],
        steps=60,
        runs=2,
    )
    results = exp.run()
    assert len(results) == 2 and all(len(r.steps) == 60 for r in results)
    assert isinstance(robust_ucb(), RobustUCB)
