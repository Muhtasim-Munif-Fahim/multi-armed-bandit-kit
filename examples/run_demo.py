"""Run a demo of bandit_kit and write a Markdown report."""

from __future__ import annotations

from pathlib import Path

from bandit_kit import (
    BanditExperiment,
    BernoulliArm,
    render_markdown_report,
)


def main() -> None:
    arms = [
        BernoulliArm(name="control", p=0.10),
        BernoulliArm(name="variant_a", p=0.14),
        BernoulliArm(name="variant_b", p=0.20),
        BernoulliArm(name="variant_c", p=0.05),
    ]
    experiment = BanditExperiment(
        arms=arms,
        algorithms=["epsilon_greedy", "ucb1", "thompson"],
        steps=250,
        runs=25,
        seed=7,
        epsilon=0.1,
    )
    runs = experiment.run()
    summary = experiment.summarize(runs)
    report = render_markdown_report(
        experiment=experiment,
        runs=runs,
        summary=summary,
    )
    out = Path("examples/output/demo_report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Wrote {out}")
    for row in summary:
        print(
            f"{row['algorithm']:>15}: mean_reward={row['mean_final_reward']:.3f} "
            f"mean_regret={row['mean_final_regret']:.3f}"
        )


if __name__ == "__main__":
    main()