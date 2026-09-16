# multi-armed-bandit-kit

A small, dependency-free Python toolkit for studying multi-armed bandit
algorithms in reproducible research settings. It implements classic
stochastic policies (`epsilon-greedy`, `UCB1`, `Thompson sampling`) plus
the adversarial-bandit policy `EXP3`, provides seedable synthetic
Bernoulli and Gaussian arms, runs a configurable experiment harness, and
emits a Markdown report comparing cumulative reward, cumulative regret,
and per-arm selection rates.

## Install

```bash
pip install -e .
```

## CLI quick start

```bash
bandit-kit compare --arms 'bern:0.1,bern:0.2,bern:0.05' --steps 200 --runs 20 -o report.md
```

`compare` runs epsilon-greedy, UCB1, Thompson sampling, and EXP3. Tune
epsilon-greedy with `--epsilon` and EXP3 with `--gamma`. The generated
`report.md` reports cumulative reward and regret per algorithm, per-arm
selection fractions, and the seed used so the run can be reproduced
verbatim.

## Library quick start

```python
from bandit_kit import (
    BernoulliArm, BanditExperiment, epsilon_greedy, ucb1, thompson_sampling_bernoulli, exp3,
)

arms = [BernoulliArm(name="A", p=0.10), BernoulliArm(name="B", p=0.20), BernoulliArm(name="C", p=0.05)]
experiment = BanditExperiment(
    arms=arms,
    algorithms=["epsilon_greedy", "ucb1", "thompson", "exp3"],
    steps=200,
    runs=20,
    seed=42,
    epsilon=0.1,
    gamma=0.1,
)
runs = experiment.run()
print(experiment.summarize(runs))
```

See `examples/run_demo.py` for a complete end-to-end demo and
`tests/` for the unit-test contract.