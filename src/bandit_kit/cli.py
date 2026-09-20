"""Command-line interface for bandit_kit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .arms import arm_from_spec, best_arm, make_linear_contextual_arms
from .algorithms import boltzmann, epsilon_greedy, exp3, ucb1, thompson_sampling_bernoulli
from .experiment import BanditExperiment, ContextualBanditExperiment
from .reporting import render_contextual_markdown_report, render_markdown_report


ALGORITHMS = {
    "epsilon_greedy": epsilon_greedy,
    "ucb1": ucb1,
    "thompson": thompson_sampling_bernoulli,
    "exp3": exp3,
    "boltzmann": boltzmann,
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
    compare.add_argument("--gamma", type=float, default=0.1, help="EXP3 exploration mixing rate (default: 0.1)")
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
    compare.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    compare.add_argument(
        "--output", "-o", default=None,
        help="Write the markdown report to a file instead of stdout",
    )
    _build_best_parser(sub)
    _build_contextual_parser(sub)
    return parser


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
        help="Run LinUCB (and optional baselines) on synthetic linear contexts",
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
        "--ridge",
        type=float,
        default=1.0,
        help="LinUCB ridge regulariser (default: 1.0)",
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
            temperature_start=args.temperature_start,
            temperature_min=args.temperature_min,
            temperature_decay=args.temperature_decay,
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
    if args.command == "compare-contextual":
        return cmd_compare_contextual(args)
    parser.error(f"unknown command: {args.command}")
    return 2