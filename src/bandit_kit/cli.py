"""Command-line interface for bandit_kit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .arms import arm_from_spec, best_arm, make_linear_contextual_arms
from .algorithms import boltzmann, discounted_ucb, epsilon_greedy, exp3, exp3_ix, exp3_s, imed, kl_ucb, moss, sliding_window_ucb, sliding_window_thompson, ucb1, ucb_tuned, ucb_v, ucb2, thompson_sampling_bernoulli, thompson_sampling_gaussian, tsallis_inf
from .experiment import BanditExperiment, ContextualBanditExperiment
from .reporting import render_contextual_markdown_report, render_markdown_report


ALGORITHMS = {
    "epsilon_greedy": epsilon_greedy,
    "ucb1": ucb1,
    "ucb_tuned": ucb_tuned,
    "thompson": thompson_sampling_bernoulli,
    "thompson_gaussian": thompson_sampling_gaussian,
    "exp3": exp3,
    "exp3_ix": exp3_ix,
    "exp3_s": exp3_s,
    "boltzmann": boltzmann,
    "kl_ucb": kl_ucb,
    "imed": imed,
    "tsallis_inf": tsallis_inf,
    "sliding_window_ucb": sliding_window_ucb,
    "sliding_window_thompson": sliding_window_thompson,
    "discounted_ucb": discounted_ucb,
    "ucb_v": ucb_v,
    "ucb2": ucb2,
    "moss": moss,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bandit-kit")
    sub = parser.add_subparsers(dest="command", required=True)

    compare = sub.add_parser("compare", help="Run all algorithms and emit a markdown report")
    compare.add_argument(
        "--arms", required=True,
        help="Comma-separated list of arm specs, e.g. 'bern:0.1,bern:0.2,bern:0.05'",
    )
    compare.add_argument("--steps", type=int, default=200, help="Number of pulls per run (default: 200)")
    compare.add_argument("--runs", type=int, default=20, help="Number of independent runs (default: 20)")
    compare.add_argument("--epsilon", type=float, default=0.1, help="epsilon-greedy exploration rate (default: 0.1)")
    compare.add_argument("--gamma", type=float, default=0.1, help="EXP3 / EXP3.S exploration mixing rate (default: 0.1)")
    compare.add_argument(
        "--exp3-ix-eta",
        type=float,
        default=0.1,
        help="EXP3-IX learning rate / implicit-exploration bias eta > 0 (default: 0.1)",
    )
    compare.add_argument(
        "--exp3-s-alpha",
        type=float,
        default=0.01,
        help="EXP3.S fixed-share switching rate alpha in [0, 1] (default: 0.01; 1/T is the tuned choice)",
    )
    compare.add_argument(
        "--temperature-start",
        type=float,
        default=1.0,
        help="Boltzmann / softmax initial temperature (default: 1.0)",
    )
    compare.add_argument(
        "--temperature-min",
        type=float,
        default=0.05,
        help="Boltzmann / softmax temperature floor (default: 0.05)",
    )
    compare.add_argument(
        "--temperature-decay",
        type=float,
        default=0.99,
        help="Boltzmann / softmax temperature decay in (0, 1) (default: 0.99)",
    )
    compare.add_argument(
        "--kl-ucb-c",
        type=float,
        default=0.0,
        help="KL-UCB extra log-log coefficient (default: 0.0; use 3 for the COLT bound)",
    )
    compare.add_argument(
        "--sliding-window",
        type=int,
        default=100,
        help="Sliding-window UCB / SW-TS recent-pull window (default: 100)",
    )
    compare.add_argument(
        "--discount-gamma",
        type=float,
        default=0.9,
        help="DiscountedUCB exponential discount γ in (0, 1] (default: 0.9)",
    )
    compare.add_argument(
        "--ucb-v-c",
        type=float,
        default=3.0,
        help="UCB-V additive exploration constant c (default: 3.0)",
    )
    compare.add_argument(
        "--ucb2-alpha",
        type=float,
        default=0.1,
        help="UCB2 epoch parameter α in (0, 1) (default: 0.1)",
    )
    compare.add_argument(
        "--tsallis-estimator",
        choices=("iw", "rv"),
        default="iw",
        help="Tsallis-INF loss estimator: importance-weighted or reduced-variance (default: iw)",
    )
    compare.add_argument(
        "--moss-horizon",
        type=int,
        default=None,
        help="MOSS finite horizon T (default: same as --steps)",
    )
    compare.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    compare.add_argument(
        "--output", "-o", default=None,
        help="Write the markdown report to a file instead of stdout",
    )
    _build_best_parser(sub)
    _build_identify_parser(sub)
    _build_offpolicy_parser(sub)
    _build_contextual_parser(sub)
    return parser


def _build_identify_parser(sub) -> None:
    identify = sub.add_parser(
        "identify",
        help="Best-arm identification: Successive Rejects, Sequential Halving, or LUCB",
    )
    identify.add_argument(
        "--arms", required=True,
        help="Comma-separated arm specs, e.g. 'bern:0.5,bern:0.45,bern:0.4'",
    )
    identify.add_argument(
        "--method", choices=["sr", "sh", "lucb"], default="sr",
        help="sr = Successive Rejects, sh = Sequential Halving (fixed budget); lucb = LUCB1 (fixed confidence)",
    )
    identify.add_argument("--budget", type=int, default=2000, help="Pull budget for sr / sh (default: 2000)")
    identify.add_argument("--delta", type=float, default=0.05, help="LUCB error probability (default: 0.05)")
    identify.add_argument("--epsilon", type=float, default=0.0, help="LUCB slack (default: 0)")
    identify.add_argument("--max-pulls", type=int, default=100_000, help="LUCB pull cap (default: 100000)")
    identify.add_argument(
        "--runs", type=int, default=0,
        help="Also estimate the misidentification rate over this many reseeded runs",
    )
    identify.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    identify.add_argument("--json", action="store_true", help="Print JSON")


def cmd_identify(args: argparse.Namespace) -> int:
    """Run one best-arm identification and optionally its error rate."""
    from .arms import BernoulliArm, GaussianArm
    from .identification import (
        identification_error_rate,
        lucb,
        successive_halving,
        successive_rejects,
    )

    try:
        specs = [spec for spec in args.arms.split(",") if spec.strip()]
        arms = []
        for i, spec in enumerate(specs):
            arm = arm_from_spec(spec)
            if isinstance(arm, BernoulliArm):
                arm = BernoulliArm(arm.name, arm.p, seed=args.seed + i)
            elif isinstance(arm, GaussianArm):
                arm = GaussianArm(arm.name, arm.mean, arm.std, seed=args.seed + i)
            arms.append(arm)
        if args.method == "lucb":
            method, kwargs = lucb, {"delta": args.delta, "epsilon": args.epsilon, "max_pulls": args.max_pulls}
        else:
            method = successive_rejects if args.method == "sr" else successive_halving
            kwargs = {"budget": args.budget}
        result = method(arms, **kwargs)
        payload = {
            "method": result.method,
            "recommended": result.recommended,
            "correct": result.is_correct(arms),
            "oracle_arm": best_arm(arms).name,
            "total_pulls": result.total_pulls,
            "stopped": result.stopped,
            "pulls": result.pulls,
            "empirical_means": result.empirical_means,
        }
        if args.runs:
            payload["error_rate"] = identification_error_rate(
                method, arms, n_runs=args.runs, seed=args.seed + 1000, **kwargs
            )
    except ValueError as exc:
        print(f"identify: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"method: {payload['method']}")
        print(f"recommended arm: {payload['recommended']} (oracle: {payload['oracle_arm']})")
        print(f"total pulls: {payload['total_pulls']}" + ("" if payload["stopped"] else " (hit --max-pulls)"))
        for name, count in payload["pulls"].items():
            print(f"  {name}: pulls={count} mean={payload['empirical_means'][name]:.4f}")
        if args.runs:
            rate = payload["error_rate"]
            print(f"misidentification rate over {args.runs} runs: {rate['error_rate']:.3f}")
    return 0


def _parse_probabilities(text: str, n_arms: int) -> list[float]:
    values = [float(v) for v in text.split(",") if v.strip()]
    if len(values) != n_arms:
        raise ValueError(f"expected {n_arms} probabilities, got {len(values)}")
    return values


def _build_offpolicy_parser(sub) -> None:
    ope = sub.add_parser(
        "offpolicy",
        help="Off-policy evaluation: IPS / SNIPS / DM / DR estimates of a target policy from simulated logs",
    )
    ope.add_argument("--arms", required=True, help="Comma-separated arm specs, e.g. 'bern:0.3,bern:0.5,bern:0.7'")
    ope.add_argument(
        "--target", required=True,
        help="Target policy: an arm index (deterministic) or comma-separated probabilities",
    )
    ope.add_argument(
        "--logging", default=None,
        help="Logging policy probabilities (default: uniform)",
    )
    ope.add_argument("--n", type=int, default=5000, help="Number of logged events (default: 5000)")
    ope.add_argument("--clip", type=float, default=None, help="Also report weight-clipped IPS at this cap")
    ope.add_argument(
        "--replay", default=None, choices=sorted(ALGORITHMS),
        help="Also replay this learning algorithm on the logs (requires uniform logging)",
    )
    ope.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    ope.add_argument("--json", action="store_true", help="Print JSON")


def cmd_offpolicy(args: argparse.Namespace) -> int:
    """Simulate logged data and estimate a target policy's value off-policy."""
    from .arms import BernoulliArm, GaussianArm
    from .offpolicy import collect_logs, evaluate_policy, replay_evaluate, true_policy_value

    try:
        arms = []
        for i, spec in enumerate(s for s in args.arms.split(",") if s.strip()):
            arm = arm_from_spec(spec)
            if isinstance(arm, BernoulliArm):
                arm = BernoulliArm(arm.name, arm.p, seed=args.seed + i)
            elif isinstance(arm, GaussianArm):
                arm = GaussianArm(arm.name, arm.mean, arm.std, seed=args.seed + i)
            arms.append(arm)
        if not arms:
            raise ValueError("at least one arm spec is required")
        target = int(args.target) if "," not in args.target else _parse_probabilities(args.target, len(arms))
        logging_policy = None if args.logging is None else _parse_probabilities(args.logging, len(arms))
        logs = collect_logs(arms, args.n, logging_policy, seed=args.seed + 10_000)
        estimates = evaluate_policy(logs, target, n_arms=len(arms), clip=args.clip)
        payload = {
            "true_value": true_policy_value(arms, target),
            "n_logged": len(logs),
            "estimates": {name: est.as_dict() for name, est in estimates.items()},
        }
        if args.replay:
            algo = ALGORITHMS[args.replay](seed=args.seed)
            replay = replay_evaluate(algo, arms, logs)
            payload["replay"] = {
                "algorithm": args.replay,
                "mean_reward": replay.mean_reward,
                "n_matched": replay.n_matched,
                "match_rate": replay.match_rate,
                "arm_counts": replay.arm_counts,
            }
    except (ValueError, TypeError) as exc:
        print(f"offpolicy: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    print(f"true target value: {payload['true_value']:.4f}  (logged events: {payload['n_logged']})")
    for name, est in payload["estimates"].items():
        ess = est["effective_sample_size"]
        ess_text = f"  ESS={ess:.0f}" if ess is not None else ""
        print(
            f"  {name:>11}: {est['value']:.4f}  se={est['std_error']:.4f}  "
            f"95% CI [{est['ci_low']:.4f}, {est['ci_high']:.4f}]{ess_text}"
        )
    if args.replay:
        rep = payload["replay"]
        print(
            f"replay {rep['algorithm']}: mean reward {rep['mean_reward']:.4f} over "
            f"{rep['n_matched']} matched events (match rate {rep['match_rate']:.3f})"
        )
    return 0


def _build_best_parser(sub):
    best = sub.add_parser(
        "best",
        help="Show the arm with the highest expected payoff (the oracle)",
    )
    best.add_argument(
        "--arms", required=True,
        help="Comma-separated list of arm specs, e.g. 'bern:0.1,bern:0.2,bern:0.05'",
    )
    best.add_argument(
        "--json", action="store_true",
        help="Print the result as JSON instead of plain text",
    )


def cmd_best(args: argparse.Namespace) -> int:
    """Print the oracle arm for the given spec list."""
    arms = [arm_from_spec(spec) for spec in args.arms.split(",") if spec.strip()]
    if not arms:
        print("best: at least one arm spec is required", file=sys.stderr)
        return 2
    oracle = best_arm(arms)
    payload = {
        "oracle_arm": oracle.name,
        "oracle_payoff": oracle.expected_value,
        "arms": [{"name": arm.name, "expected_value": arm.expected_value} for arm in arms],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"oracle arm: {oracle.name}")
        print(f"oracle expected payoff: {oracle.expected_value:.4f}")
        for arm in arms:
            print(f"  {arm.name}: EV={arm.expected_value:.4f}")
    return 0


def _build_contextual_parser(sub) -> None:
    contextual = sub.add_parser(
        "compare-contextual",
        help="Run LinUCB, LinTS, and optional baselines on synthetic linear contexts",
    )
    contextual.add_argument(
        "--n-arms",
        type=int,
        default=3,
        help="Number of synthetic linear arms (default: 3)",
    )
    contextual.add_argument(
        "--dim",
        type=int,
        default=4,
        help="Context dimension (default: 4, includes intercept)",
    )
    contextual.add_argument("--steps", type=int, default=200, help="Number of pulls per run (default: 200)")
    contextual.add_argument("--runs", type=int, default=20, help="Number of independent runs (default: 20)")
    contextual.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="LinUCB exploration parameter (default: 1.0)",
    )
    contextual.add_argument(
        "--lints-v",
        type=float,
        default=1.0,
        help="LinTS posterior sampling scale (default: 1.0; 0 is greedy posterior mean)",
    )
    contextual.add_argument(
        "--ridge",
        type=float,
        default=1.0,
        help="Ridge regulariser for LinUCB and LinTS (default: 1.0)",
    )
    contextual.add_argument(
        "--noise-std",
        type=float,
        default=0.1,
        help="Gaussian observation noise on linear rewards (default: 0.1)",
    )
    contextual.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    contextual.add_argument(
        "--no-intercept",
        action="store_true",
        help="Do not pin the first context coordinate to 1.0",
    )
    contextual.add_argument(
        "--algorithms",
        default="linucb,ucb1,epsilon_greedy",
        help="Comma-separated algorithm names (default: linucb,ucb1,epsilon_greedy)",
    )
    contextual.add_argument(
        "--output", "-o", default=None,
        help="Write the markdown report to a file instead of stdout",
    )


def cmd_compare_contextual(args: argparse.Namespace) -> int:
    algorithms = [name.strip() for name in args.algorithms.split(",") if name.strip()]
    if not algorithms:
        print("compare-contextual: at least one algorithm is required", file=sys.stderr)
        return 2
    if args.n_arms < 1:
        print("compare-contextual: --n-arms must be at least 1", file=sys.stderr)
        return 2
    if args.dim < 1:
        print("compare-contextual: --dim must be at least 1", file=sys.stderr)
        return 2
    intercept = not args.no_intercept
    arms = make_linear_contextual_arms(
        n_arms=args.n_arms,
        dimension=args.dim,
        noise_std=args.noise_std,
        seed=args.seed,
        intercept=intercept,
    )
    try:
        experiment = ContextualBanditExperiment(
            arms=arms,
            algorithms=algorithms,
            steps=args.steps,
            runs=args.runs,
            seed=args.seed,
            linucb_alpha=args.alpha,
            linucb_ridge=args.ridge,
            lints_v=args.lints_v,
            lints_ridge=args.ridge,
            context_intercept=intercept,
        )
    except ValueError as exc:
        print(f"compare-contextual: {exc}", file=sys.stderr)
        return 2
    runs = experiment.run()
    summary = experiment.summarize(runs)
    report = render_contextual_markdown_report(
        experiment=experiment,
        runs=runs,
        summary=summary,
    )
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report, encoding="utf-8")
        print(f"Wrote {target}")
    else:
        print(report)
    print(
        json.dumps(
            {
                "algorithms": algorithms,
                "steps": args.steps,
                "runs": args.runs,
                "dim": args.dim,
                "n_arms": args.n_arms,
            },
            indent=2,
        )
    )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    arms = [arm_from_spec(spec) for spec in args.arms.split(",") if spec.strip()]
    if not arms:
        print("compare: at least one arm spec is required", file=sys.stderr)
        return 2
    try:
        experiment = BanditExperiment(
            arms=arms,
            algorithms=list(ALGORITHMS.keys()),
            steps=args.steps,
            runs=args.runs,
            seed=args.seed,
            epsilon=args.epsilon,
            gamma=args.gamma,
            exp3_ix_eta=args.exp3_ix_eta,
            exp3_s_alpha=args.exp3_s_alpha,
            temperature_start=args.temperature_start,
            temperature_min=args.temperature_min,
            temperature_decay=args.temperature_decay,
            kl_ucb_c=args.kl_ucb_c,
            sliding_window=args.sliding_window,
            discount_gamma=args.discount_gamma,
            ucb_v_c=args.ucb_v_c,
            ucb2_alpha=args.ucb2_alpha,
            tsallis_estimator=args.tsallis_estimator,
            moss_horizon=args.moss_horizon,
        )
    except ValueError as exc:
        print(f"compare: {exc}", file=sys.stderr)
        return 2
    runs = experiment.run()
    summary = experiment.summarize(runs)
    report = render_markdown_report(
        experiment=experiment,
        runs=runs,
        summary=summary,
    )
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report, encoding="utf-8")
        print(f"Wrote {target}")
    else:
        print(report)
    print(json.dumps({"algorithms": list(ALGORITHMS), "steps": args.steps, "runs": args.runs}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "compare":
        return cmd_compare(args)
    if args.command == "best":
        return cmd_best(args)
    if args.command == "identify":
        return cmd_identify(args)
    if args.command == "offpolicy":
        return cmd_offpolicy(args)
    if args.command == "compare-contextual":
        return cmd_compare_contextual(args)
    parser.error(f"unknown command: {args.command}")
    return 2