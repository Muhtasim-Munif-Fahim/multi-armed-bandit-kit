# multi-armed-bandit-kit

A small, dependency-free Python toolkit for studying multi-armed bandit
algorithms in reproducible research settings. It implements classic
stochastic policies (`epsilon-greedy`, `UCB1`, `Thompson sampling`) plus
the adversarial-bandit policy `EXP3` and the contextual linear policy
`LinUCB`, provides seedable synthetic Bernoulli, Gaussian, and linear
contextual arms, runs a configurable experiment harness, and emits a
Markdown report comparing cumulative reward, cumulative regret, and
per-arm selection rates.

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

For **contextual** rewards, `compare-contextual` draws synthetic linear
contexts and compares disjoint LinUCB to non-contextual baselines:

```bash
bandit-kit compare-contextual --dim 4 --n-arms 3 --steps 200 --runs 20 --alpha 1.0 -o contextual.md
```

Tune LinUCB with `--alpha` (exploration bonus) and `--ridge` (ridge
regulariser). Pass `--algorithms linucb` to run LinUCB alone, or keep
the default `linucb,ucb1,epsilon_greedy` to show the cost of ignoring
context. `--no-intercept` samples every coordinate in `[-1, 1]` instead
of pinning the first feature to `1.0`.

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

### LinUCB (contextual linear bandit)

Disjoint LinUCB (Li, Chu, Langford, Schapire 2010) maintains a
ridge-regression estimate `theta_a` per arm and at each step picks the
arm that maximises `theta_a · x + alpha * sqrt(x^T A_a^{-1} x)`. Matrix
inverses are updated with Sherman-Morrison rank-1 updates, so the
implementation stays numpy-free.

```python
from bandit_kit import (
    ContextualBanditExperiment,
    make_linear_contextual_arms,
)

arms = make_linear_contextual_arms(n_arms=3, dimension=4, seed=0)
experiment = ContextualBanditExperiment(
    arms=arms,
    algorithms=["linucb", "ucb1", "epsilon_greedy"],
    steps=200,
    runs=20,
    seed=42,
    linucb_alpha=1.0,
)
runs = experiment.run()
print(experiment.summarize(runs))
```

On a stationary (non-contextual) problem the same policy is available
as `"linucb"` in `BanditExperiment`. It uses the intercept context
`[1.0]`, which reduces LinUCB to ridge-UCB and leaves the other
policies unchanged.

See `examples/run_demo.py` for a complete end-to-end demo and
`tests/` for the unit-test contract.
