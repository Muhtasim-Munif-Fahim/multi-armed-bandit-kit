# multi-armed-bandit-kit

A small, dependency-free Python toolkit for studying multi-armed bandit
algorithms in reproducible research settings. It implements classic
stochastic policies (`epsilon-greedy`, `UCB1`, `Thompson sampling`,
`KL-UCB`) plus the adversarial-bandit policy `EXP3`, the contextual
linear policies `LinUCB` and `LinTS` (linear Thompson sampling), and
Boltzmann / softmax exploration with a decaying temperature schedule.
It provides seedable synthetic Bernoulli,
Gaussian, and linear contextual arms, runs a configurable experiment
harness, and emits a Markdown report comparing cumulative reward,
cumulative regret, and per-arm selection rates.

## Install

```bash
pip install -e .
```

## CLI quick start

```bash
bandit-kit compare --arms 'bern:0.1,bern:0.2,bern:0.05' --steps 200 --runs 20 -o report.md
```

`compare` runs epsilon-greedy, UCB1, Thompson sampling, EXP3,
Boltzmann / softmax, and KL-UCB. Tune epsilon-greedy with `--epsilon`,
EXP3 with `--gamma`, Boltzmann with `--temperature-start`,
`--temperature-min`, and `--temperature-decay`, and KL-UCB with
`--kl-ucb-c`. The generated `report.md` reports cumulative reward and
regret per algorithm, per-arm selection fractions, and the seed used so
the run can be reproduced verbatim.

For **contextual** rewards, `compare-contextual` draws synthetic linear
contexts and compares disjoint LinUCB to non-contextual baselines:

```bash
bandit-kit compare-contextual --dim 4 --n-arms 3 --steps 200 --runs 20 --alpha 1.0 -o contextual.md
```

Tune LinUCB with `--alpha` (exploration bonus) and `--ridge` (ridge
regulariser, shared with LinTS). Pass `--algorithms linucb` to run
LinUCB alone, add `lints` to compare linear Thompson sampling
(`--lints-v` sets the posterior scale), or keep the default
`linucb,ucb1,epsilon_greedy` to show the cost of ignoring context.
`--no-intercept` samples every coordinate in `[-1, 1]` instead of
pinning the first feature to `1.0`.

```bash
bandit-kit compare-contextual --algorithms lints,linucb,ucb1 --lints-v 1.0 --ridge 1.0 --dim 4 --steps 200 --runs 20
```

## Library quick start

```python
from bandit_kit import (
    BernoulliArm, BanditExperiment, epsilon_greedy, ucb1, thompson_sampling_bernoulli, exp3, boltzmann, kl_ucb,
)

arms = [BernoulliArm(name="A", p=0.10), BernoulliArm(name="B", p=0.20), BernoulliArm(name="C", p=0.05)]
experiment = BanditExperiment(
    arms=arms,
    algorithms=["epsilon_greedy", "ucb1", "thompson", "exp3", "boltzmann", "kl_ucb"],
    steps=200,
    runs=20,
    seed=42,
    epsilon=0.1,
    gamma=0.1,
    temperature_start=1.0,
    temperature_min=0.05,
    temperature_decay=0.99,
    kl_ucb_c=0.0,
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

### LinTS (linear contextual Thompson sampling)

Disjoint linear Thompson sampling (Agrawal & Goyal, 2013) keeps the
same per-arm Bayesian linear regression posterior as LinUCB. With prior
`N(0, ridge^{-1} I)` and a unit-variance Gaussian likelihood,

```
A_a = ridge * I + sum x x^T
b_a = sum r x
theta_a ~ N(A_a^{-1} b_a, v^2 A_a^{-1})
```

and the policy pulls `argmax_a theta_a · x`. `v` is the posterior
sampling scale: `v = 0` is greedy posterior-mean selection and matches
LinUCB with `alpha = 0` on the same design matrix. Posterior draws use
a Cholesky factor of `A^{-1}`, which is maintained with the same
Sherman-Morrison updates as LinUCB, so the implementation stays
numpy-free.

The registry name is `"lints"`. On a stationary problem the policy uses
the intercept context `[1.0]`, the same fallback as LinUCB.

```python
from bandit_kit import ContextualBanditExperiment, make_linear_contextual_arms

arms = make_linear_contextual_arms(n_arms=3, dimension=4, seed=0)
experiment = ContextualBanditExperiment(
    arms=arms,
    algorithms=["lints", "linucb", "ucb1"],
    steps=200,
    runs=20,
    seed=42,
    lints_v=1.0,
    lints_ridge=1.0,
    linucb_alpha=1.0,
)
runs = experiment.run()
print(experiment.summarize(runs))
```

### Boltzmann / softmax exploration

Boltzmann (also called softmax action selection) maintains an empirical
mean `Q[i]` per arm and samples

```
P(i) = exp(Q[i] / tau_t) / sum_j exp(Q[j] / tau_t)
```

The temperature follows an exponential schedule

```
tau_t = temperature_min + (temperature_start - temperature_min) * decay ** t
```

High temperature is nearly uniform; as `tau` cools the policy
concentrates on the empirically best arm. Set `temperature_min` equal
to `temperature_start` for a constant-temperature policy. The registry
name is `"boltzmann"`; `"softmax"` is an alias for the same factory.

```python
from bandit_kit import BanditExperiment, BernoulliArm, boltzmann

arms = [BernoulliArm(name="A", p=0.10), BernoulliArm(name="B", p=0.20)]
experiment = BanditExperiment(
    arms=arms,
    algorithms=["boltzmann"],
    steps=200,
    runs=20,
    seed=42,
    temperature_start=1.0,
    temperature_min=0.05,
    temperature_decay=0.99,
)
```

### KL-UCB

KL-UCB (Garivier & Cappé, COLT 2011) is an index policy for
`[0, 1]`-bounded rewards. After a one-pull warmup it selects the arm
with the largest Bernoulli KL upper bound

```
sup { q in [mu_hat, 1] : N * d(mu_hat, q) <= log(t) + c * log(log(t)) }
```

where `d` is Bernoulli KL divergence. `c=0` (the default) uses the
common `log(t)` threshold; `c=3` recovers the extra `log log(t)` term
from the paper. Rewards outside `[0, 1]` are clipped, matching EXP3.

The kit already ships Beta-Bernoulli Thompson sampling
(`"thompson"` / `thompson_sampling_bernoulli`) and UCB1 (`"ucb1"`), so
KL-UCB is the additional bounded-reward UCB variant. The registry name
is `"kl_ucb"`.

```python
from bandit_kit import BanditExperiment, BernoulliArm, kl_ucb

arms = [BernoulliArm(name="A", p=0.10), BernoulliArm(name="B", p=0.20)]
experiment = BanditExperiment(
    arms=arms,
    algorithms=["kl_ucb"],
    steps=200,
    runs=20,
    seed=42,
    kl_ucb_c=0.0,
)
```

See `examples/run_demo.py` for a complete end-to-end demo and
`tests/` for the unit-test contract.
