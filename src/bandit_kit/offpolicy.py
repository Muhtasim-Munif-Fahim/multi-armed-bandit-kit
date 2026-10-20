"""Off-policy evaluation of bandit policies from logged interactions.

A deployed (logging) policy ``mu`` chooses arm ``a_t`` with known propensity
``mu(a_t | x_t)`` and observes only that arm's reward ``r_t``. Off-policy
evaluation estimates the value ``V(pi) = E_x[sum_a pi(a | x) E[r | x, a]]``
of a different *target* policy ``pi`` from those logs, without deploying it.

Estimators (Dudik, Langford & Li 2011; Swaminathan & Joachims 2015;
Li, Chu, Langford & Wang 2011):

* :func:`ips_estimate`: inverse propensity scoring, ``mean(w_t r_t)`` with
  ``w_t = pi(a_t | x_t) / mu(a_t | x_t)``; unbiased when ``mu > 0`` wherever
  ``pi > 0``. ``clip`` caps the weights (biased downward, lower variance).
* :func:`snips_estimate`: self-normalised IPS, ``sum w r / sum w``;
  consistent, scale-invariant, and much lower variance when weights are
  heavy-tailed.
* :func:`direct_method_estimate`: plugs in a reward model
  ``rhat(x, a)``: ``mean_t sum_a pi(a | x_t) rhat(x_t, a)``; low variance but
  biased when the model is wrong.
* :func:`doubly_robust_estimate`: DM plus an IPS correction of the model's
  residual on the logged arm; unbiased if *either* the propensities or the
  reward model are correct.
* :func:`replay_evaluate`: the replay method for *learning* algorithms under
  uniformly random logging: stream the logs, keep only events where the
  algorithm picks the logged arm, and let it learn from those. The matched
  rewards are an unbiased sample of the algorithm's online trajectory.

Policies are given as a sequence of arm probabilities, an ``int`` (a
deterministic policy that always plays that arm), or a callable
``policy(context) -> probabilities``. Every estimator returns an
:class:`OPEEstimate` with a standard error, a normal confidence interval,
and the effective sample size ``(sum w)^2 / sum w^2`` of the weights.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Callable, Dict, List, Sequence, Union

from .algorithms import BanditAlgorithm, BanditStep
from .arms import Arm

Policy = Union[int, Sequence[float], Callable[[object], Sequence[float]]]
RewardModel = Callable[[object, int], float]

_TOL = 1e-9


@dataclass(frozen=True)
class LoggedEvent:
    """One logged interaction: the arm played, its reward and its propensity."""

    arm_index: int
    reward: float
    propensity: float
    context: tuple | None = None

    def __post_init__(self) -> None:
        if isinstance(self.arm_index, bool) or not isinstance(self.arm_index, int) or self.arm_index < 0:
            raise ValueError("arm_index must be a non-negative integer")
        if not math.isfinite(self.reward):
            raise ValueError("reward must be finite")
        if not 0.0 < self.propensity <= 1.0 + _TOL:
            raise ValueError("propensity must be in (0, 1]")


@dataclass
class OPEEstimate:
    """A policy-value estimate with its sampling uncertainty."""

    estimator: str
    value: float
    std_error: float
    ci_low: float
    ci_high: float
    n: int
    effective_sample_size: float | None = None
    extra: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "estimator": self.estimator,
            "value": self.value,
            "std_error": self.std_error,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n": self.n,
            "effective_sample_size": self.effective_sample_size,
        }
        payload.update(self.extra)
        return payload


def _validate_probabilities(probs: Sequence[float], n_arms: int | None) -> List[float]:
    values = [float(p) for p in probs]
    if n_arms is not None and len(values) != n_arms:
        raise ValueError(f"policy must give {n_arms} probabilities, got {len(values)}")
    if not values or any(not math.isfinite(p) or p < -_TOL for p in values):
        raise ValueError("policy probabilities must be finite and non-negative")
    if abs(sum(values) - 1.0) > 1e-6:
        raise ValueError("policy probabilities must sum to 1")
    return [max(p, 0.0) for p in values]


def policy_probabilities(policy: Policy, context: object, n_arms: int) -> List[float]:
    """Resolve ``policy`` to its arm distribution at ``context``."""

    if isinstance(policy, bool):
        raise ValueError("policy must be an arm index, probabilities, or a callable")
    if isinstance(policy, int):
        if not 0 <= policy < n_arms:
            raise ValueError("deterministic policy arm index out of range")
        return [1.0 if i == policy else 0.0 for i in range(n_arms)]
    if callable(policy):
        return _validate_probabilities(policy(context), n_arms)
    return _validate_probabilities(policy, n_arms)


def _infer_n_arms(logs: Sequence[LoggedEvent], n_arms: int | None) -> int:
    if not logs:
        raise ValueError("logs must not be empty")
    largest = max(event.arm_index for event in logs) + 1
    if n_arms is None:
        return largest
    if n_arms < largest:
        raise ValueError("n_arms is smaller than a logged arm index")
    return int(n_arms)


def _normal_summary(terms: Sequence[float]):
    n = len(terms)
    mean = math.fsum(terms) / n
    if n > 1:
        var = math.fsum((t - mean) ** 2 for t in terms) / (n - 1)
        se = math.sqrt(var / n)
    else:
        se = float("nan")
    return mean, se


def _interval(value: float, se: float, confidence: float):
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if not math.isfinite(se):
        return float("nan"), float("nan")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    return value - z * se, value + z * se


def importance_weights(
    logs: Sequence[LoggedEvent], target: Policy, n_arms: int | None = None
) -> List[float]:
    """Per-event weights ``pi(a_t | x_t) / mu(a_t | x_t)``."""

    k = _infer_n_arms(logs, n_arms)
    return [
        policy_probabilities(target, event.context, k)[event.arm_index] / event.propensity
        for event in logs
    ]


def effective_sample_size(weights: Sequence[float]) -> float:
    """Kish effective sample size ``(sum w)^2 / sum w^2`` (0 if all weights are 0)."""

    total_sq = math.fsum(w * w for w in weights)
    if total_sq == 0.0:
        return 0.0
    return math.fsum(weights) ** 2 / total_sq


def _check_support(logs, target, k):
    """Raise if a context-free target needs an arm that never appears in the logs.

    Arms never logged carry no information about their reward, so any
    IPS-type estimate would silently treat their value as zero. Callable
    (contextual) targets are not checked, because support depends on the
    context value.
    """

    if callable(target):
        return
    probs = policy_probabilities(target, None, k)
    logged = {event.arm_index for event in logs}
    missing = [a for a, p in enumerate(probs) if p > 0 and a not in logged]
    if missing:
        raise ValueError(
            f"target policy puts mass on arms never logged: {missing}; "
            "IPS-type estimators are not identified"
        )


def ips_estimate(
    logs: Sequence[LoggedEvent],
    target: Policy,
    *,
    n_arms: int | None = None,
    clip: float | None = None,
    confidence: float = 0.95,
) -> OPEEstimate:
    """Inverse propensity scoring estimate of ``V(target)``."""

    k = _infer_n_arms(logs, n_arms)
    _check_support(logs, target, k)
    if clip is not None and not clip > 0:
        raise ValueError("clip must be positive")
    weights = importance_weights(logs, target, k)
    used = [min(w, clip) for w in weights] if clip is not None else weights
    terms = [w * e.reward for w, e in zip(used, logs)]
    value, se = _normal_summary(terms)
    lo, hi = _interval(value, se, confidence)
    extra = {"max_weight": max(weights)}
    if clip is not None:
        extra["clipped_fraction"] = sum(w > clip for w in weights) / len(weights)
    return OPEEstimate(
        "clipped_ips" if clip is not None else "ips",
        value, se, lo, hi, len(logs), effective_sample_size(used), extra,
    )


def snips_estimate(
    logs: Sequence[LoggedEvent],
    target: Policy,
    *,
    n_arms: int | None = None,
    confidence: float = 0.95,
) -> OPEEstimate:
    """Self-normalised IPS ``sum w r / sum w`` with a delta-method standard error."""

    k = _infer_n_arms(logs, n_arms)
    _check_support(logs, target, k)
    weights = importance_weights(logs, target, k)
    total = math.fsum(weights)
    if total <= 0.0:
        raise ValueError("all importance weights are zero; SNIPS is undefined")
    value = math.fsum(w * e.reward for w, e in zip(weights, logs)) / total
    n = len(logs)
    if n > 1:
        resid = math.fsum((w * (e.reward - value)) ** 2 for w, e in zip(weights, logs))
        se = math.sqrt(resid * n / (n - 1)) / total
    else:
        se = float("nan")
    lo, hi = _interval(value, se, confidence)
    return OPEEstimate("snips", value, se, lo, hi, n, effective_sample_size(weights),
                       {"mean_weight": total / n})


def empirical_reward_model(logs: Sequence[LoggedEvent], n_arms: int | None = None,
                           prior: float = 0.0) -> RewardModel:
    """Context-free reward model: per-arm mean logged reward (``prior`` if unseen)."""

    k = _infer_n_arms(logs, n_arms)
    sums = [0.0] * k
    counts = [0] * k
    for event in logs:
        sums[event.arm_index] += event.reward
        counts[event.arm_index] += 1
    means = [sums[a] / counts[a] if counts[a] else float(prior) for a in range(k)]

    def model(context: object, arm: int) -> float:
        return means[arm]

    model.means = means  # type: ignore[attr-defined]
    return model


def _model_terms(logs, target, k, reward_model):
    dm_terms = []
    residuals = []
    for event in logs:
        probs = policy_probabilities(target, event.context, k)
        preds = [float(reward_model(event.context, a)) for a in range(k)]
        if any(not math.isfinite(p) for p in preds):
            raise ValueError("reward_model must return finite values")
        dm_terms.append(math.fsum(p * q for p, q in zip(probs, preds)))
        residuals.append(event.reward - preds[event.arm_index])
    return dm_terms, residuals


def direct_method_estimate(
    logs: Sequence[LoggedEvent],
    target: Policy,
    *,
    reward_model: RewardModel | None = None,
    n_arms: int | None = None,
    confidence: float = 0.95,
) -> OPEEstimate:
    """Direct method: average model-predicted reward under the target policy.

    The default model is :func:`empirical_reward_model` fitted on the same
    logs. The standard error reflects only context sampling, not model error.
    """

    k = _infer_n_arms(logs, n_arms)
    model = reward_model or empirical_reward_model(logs, k)
    dm_terms, _ = _model_terms(logs, target, k, model)
    value, se = _normal_summary(dm_terms)
    lo, hi = _interval(value, se, confidence)
    return OPEEstimate("dm", value, se, lo, hi, len(logs))


def doubly_robust_estimate(
    logs: Sequence[LoggedEvent],
    target: Policy,
    *,
    reward_model: RewardModel | None = None,
    n_arms: int | None = None,
    clip: float | None = None,
    confidence: float = 0.95,
) -> OPEEstimate:
    """Doubly robust estimate ``mean(DM_t + w_t (r_t - rhat(x_t, a_t)))``."""

    k = _infer_n_arms(logs, n_arms)
    _check_support(logs, target, k)
    if clip is not None and not clip > 0:
        raise ValueError("clip must be positive")
    model = reward_model or empirical_reward_model(logs, k)
    weights = importance_weights(logs, target, k)
    used = [min(w, clip) for w in weights] if clip is not None else weights
    dm_terms, residuals = _model_terms(logs, target, k, model)
    terms = [d + w * r for d, w, r in zip(dm_terms, used, residuals)]
    value, se = _normal_summary(terms)
    lo, hi = _interval(value, se, confidence)
    return OPEEstimate("dr", value, se, lo, hi, len(logs), effective_sample_size(used),
                       {"dm_value": math.fsum(dm_terms) / len(dm_terms)})


def evaluate_policy(
    logs: Sequence[LoggedEvent],
    target: Policy,
    *,
    reward_model: RewardModel | None = None,
    n_arms: int | None = None,
    clip: float | None = None,
    confidence: float = 0.95,
) -> Dict[str, OPEEstimate]:
    """Run IPS, SNIPS, DM and DR (plus clipped IPS when ``clip`` is set)."""

    common = {"n_arms": n_arms, "confidence": confidence}
    out = {
        "ips": ips_estimate(logs, target, **common),
        "snips": snips_estimate(logs, target, **common),
        "dm": direct_method_estimate(logs, target, reward_model=reward_model, **common),
        "dr": doubly_robust_estimate(logs, target, reward_model=reward_model, clip=clip, **common),
    }
    if clip is not None:
        out["clipped_ips"] = ips_estimate(logs, target, clip=clip, **common)
    return out


def true_policy_value(arms: Sequence[Arm], policy: Policy, context: object = None) -> float:
    """Exact ``sum_a pi(a) E[r_a]`` for context-free arms."""

    probs = policy_probabilities(policy, context, len(arms))
    return math.fsum(p * arm.expected_value for p, arm in zip(probs, arms))


def collect_logs(
    arms: Sequence[Arm],
    n: int,
    logging_policy: Policy | None = None,
    *,
    seed: int | None = None,
) -> List[LoggedEvent]:
    """Simulate ``n`` logged pulls from ``arms`` under a stochastic logging policy.

    ``logging_policy`` defaults to uniform. Each event records the propensity
    of the arm that was played, which is what the estimators need.
    """

    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    k = len(arms)
    if k < 1:
        raise ValueError("at least one arm is required")
    rng = random.Random(seed)
    policy = logging_policy if logging_policy is not None else [1.0 / k] * k
    logs: List[LoggedEvent] = []
    for _ in range(n):
        probs = policy_probabilities(policy, None, k)
        arm_index = rng.choices(range(k), weights=probs)[0]
        logs.append(LoggedEvent(arm_index, float(arms[arm_index].draw()), probs[arm_index]))
    return logs


@dataclass
class ReplayResult:
    """Outcome of :func:`replay_evaluate`."""

    algorithm: str
    mean_reward: float
    n_matched: int
    n_logged: int
    rewards: List[float]
    arm_counts: List[int]

    @property
    def match_rate(self) -> float:
        return self.n_matched / self.n_logged if self.n_logged else 0.0


def replay_evaluate(
    algorithm: BanditAlgorithm,
    arms: Sequence[Arm],
    logs: Sequence[LoggedEvent],
    *,
    max_matches: int | None = None,
    require_uniform: bool = True,
) -> ReplayResult:
    """Replay a (learning) bandit algorithm on uniformly logged data (Li et al. 2011).

    For each event the algorithm proposes an arm given its state; when it
    matches the logged arm the logged reward is revealed to it via
    ``update`` and recorded, otherwise the event is skipped. With uniform
    logging each matched step is distributed exactly as an online step, so
    ``mean_reward`` estimates the algorithm's average reward over its first
    ``n_matched`` steps (about ``n_logged / K`` of them).
    """

    k = len(arms)
    if not logs:
        raise ValueError("logs must not be empty")
    if any(event.arm_index >= k for event in logs):
        raise ValueError("logs reference an arm index outside arms")
    if require_uniform and any(abs(event.propensity - 1.0 / k) > 1e-6 for event in logs):
        raise ValueError(
            "replay is only unbiased under uniform logging; "
            "pass require_uniform=False to override"
        )
    if max_matches is not None and max_matches < 1:
        raise ValueError("max_matches must be positive")
    algorithm.reset(arms)
    rewards: List[float] = []
    counts = [0] * k
    for event in logs:
        t = len(rewards)
        if event.context is None:
            choice = algorithm.select_arm(arms, t)
        else:
            choice = algorithm.select_arm(arms, t, context=event.context)
        if choice != event.arm_index:
            continue
        step = BanditStep(event.arm_index, arms[event.arm_index].name, float(event.reward), t,
                          event.context)
        if event.context is None:
            algorithm.update(arms, step)
        else:
            algorithm.update(arms, step, context=event.context)
        rewards.append(float(event.reward))
        counts[event.arm_index] += 1
        if max_matches is not None and len(rewards) >= max_matches:
            break
    mean = math.fsum(rewards) / len(rewards) if rewards else float("nan")
    return ReplayResult(getattr(algorithm, "name", type(algorithm).__name__), mean,
                        len(rewards), len(logs), rewards, counts)
