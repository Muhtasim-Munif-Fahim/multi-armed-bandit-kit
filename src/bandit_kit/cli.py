"""Command-line interface for bandit_kit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .arms import arm_from_spec, best_arm
from .algorithms import epsilon_greedy, ucb1, thompson_sampling_bernoulli
from .experiment import BanditExperiment
from .reporting import render_markdown_report


ALGORITHMS = {
    "epsilon_greedy": epsilon_greedy,
    "ucb1": ucb1,
    "thompson": thompson_sampling_bernoulli,
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
    compare.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    compare.add_argument(
        "--output", "-o", default=None,
        help="Write the markdown report to a file instead of stdout",
    )
    _build_best_parser(sub)
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


def cmd_compare(args: argparse.Namespace) -> int:
    arms = [arm_from_spec(spec) for spec in args.arms.split(",") if spec.strip()]
    if not arms:
        print("compare: at least one arm spec is required", file=sys.stderr)
        return 2
    experiment = BanditExperiment(
        arms=arms,
        algorithms=list(ALGORITHMS.keys()),
        steps=args.steps,
        runs=args.runs,
        seed=args.seed,
        epsilon=args.epsilon,
    )
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
    parser.error(f"unknown command: {args.command}")
    return 2