# multi-armed-bandit-kit

A small, dependency-free Python toolkit for studying multi-armed bandit
algorithms in reproducible research settings. It implements classic
stochastic policies (`epsilon-greedy`, `UCB1`, `UCB-Tuned`, `sliding-window UCB`, `SW-TS`, `DiscountedUCB`, `UCB-V`, `UCB2`, `MOSS`, `Thompson sampling`,
`KL-UCB`, `IMED`) plus the adversarial-bandit policy `EXP3`, the best-of-both-worlds
policy `Tsallis-INF`, the contextual
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

`compare` runs epsilon-greedy, UCB1, UCB-Tuned, sliding-window UCB, SW-TS, DiscountedUCB, UCB-V, UCB2, MOSS, Thompson sampling, EXP3,
Boltzmann / softmax, and KL-UCB. Tune epsilon-greedy with `--epsilon`,
EXP3 with `--gamma`, Boltzmann with `--temperature-start`,
`--temperature-min`, and `--temperature-decay`, KL-UCB with
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


### Sliding-window UCB

Sliding-window UCB (Garivier & Moulines, ALT 2011) keeps UCB1's index but
forms the empirical mean and pull count from only the most recent `window`
observations. At step `t` the index of arm `i` is

```text
mean_i + sqrt(2 * ln(min(t, window)) / n_i)
```

where `n_i` and `mean_i` use pulls of arm `i` inside the last `window`
steps. When `window` is larger than the horizon the policy recovers
ordinary UCB1. A short window forgets stale rewards, which helps when the
best arm changes over time.

```python
from bandit_kit import sliding_window_ucb, run_experiment
from bandit_kit.arms import BernoulliArm

arms = [BernoulliArm("a", 0.1), BernoulliArm("b", 0.5), BernoulliArm("c", 0.2)]
experiment, results = run_experiment(
    arms,
    algorithms=["sliding_window_ucb", "ucb1"],
    steps=500,
    runs=20,
    sliding_window=100,
)
print(experiment.summarize(results))
```

Tune the window from the CLI with `--sliding-window` (default 100). The
registry name is `"sliding_window_ucb"`.


### Sliding-window Thompson sampling (SW-TS)

Sliding-window Thompson sampling keeps only the most recent `window` pulls
(same global history as sliding-window UCB) and runs Beta-Bernoulli Thompson
sampling on that window. Old successes and failures fall out of the Beta
posteriors, so the policy can track a non-stationary best arm. When
`window` is larger than the horizon it recovers ordinary Bernoulli Thompson
sampling. The registry name is `"sliding_window_thompson"`; it reuses the
`--sliding-window` CLI / `sliding_window` experiment parameter.

```python
from bandit_kit import sliding_window_thompson, run_experiment
from bandit_kit.arms import BernoulliArm

arms = [BernoulliArm("a", 0.1), BernoulliArm("b", 0.5), BernoulliArm("c", 0.2)]
experiment, results = run_experiment(
    arms,
    algorithms=["sliding_window_thompson", "thompson"],
    steps=500,
    runs=20,
    sliding_window=100,
)
print(experiment.summarize(results))
```


### Discounted UCB

Discounted UCB (Garivier & Moulines, ALT 2011) keeps UCB1's index but
applies an exponential discount `γ ∈ (0, 1]` to every past count and
reward. At step `t` the index of arm `i` is

```text
mean_i + sqrt(2 * ln(t) / N_i(γ))
```

where `N_i(γ)` and `mean_i` are formed from the discounted history. After
every update all statistics are multiplied by `γ` before the newest
observation is added. When `γ = 1` the policy recovers ordinary UCB1;
smaller `γ` forgets faster and tracks a changing best arm. The other
Garivier & Moulines non-stationary variant in this kit is sliding-window
UCB.

```python
from bandit_kit import discounted_ucb, run_experiment
from bandit_kit.arms import BernoulliArm

arms = [BernoulliArm("a", 0.1), BernoulliArm("b", 0.5), BernoulliArm("c", 0.2)]
experiment, results = run_experiment(
    arms,
    algorithms=["discounted_ucb", "ucb1"],
    steps=500,
    runs=20,
    discount_gamma=0.9,
)
print(experiment.summarize(results))
```

Tune the discount from the CLI with `--discount-gamma` (default 0.9). The
registry name is `"discounted_ucb"`.


### UCB-V

UCB-V (Audibert, Munos & Szepesvári) is a variance-aware UCB that tracks
per-arm second moments. After a one-pull warmup it selects

```text
mean_i + sqrt(2 * V_i * ln(t) / n_i) + c * ln(t) / n_i
```

where `V_i` is the empirical variance from the running sum of squares
and `c` is an additive exploration constant (default `3`, matching the
classic `[0, 1]` bound). Low-variance arms shrink the exploration bonus
faster than plain UCB1. Rewards outside `[0, 1]` are clipped (matching
UCB-Tuned / KL-UCB). The registry name is `"ucb_v"`.

```python
from bandit_kit import ucb_v, run_experiment
from bandit_kit.arms import BernoulliArm

arms = [BernoulliArm("a", 0.1), BernoulliArm("b", 0.5), BernoulliArm("c", 0.2)]
experiment, results = run_experiment(
    arms,
    algorithms=["ucb_v", "ucb1"],
    steps=500,
    runs=20,
    ucb_v_c=3.0,
)
print(experiment.summarize(results))
```

Tune the additive constant from the CLI with `--ucb-v-c` (default 3.0).



### UCB2

UCB2 (Auer, Cesa-Bianchi & Fischer, Machine Learning 2002) is an epoch-based
UCB. After a one-pull warm-up it repeatedly picks the arm that maximises
`mean + sqrt((1+α) ln(e n / τ(r)) / (2 τ(r)))` with `τ(r) = ceil((1+α)^r)`,
then plays that arm for `τ(r+1) - τ(r)` consecutive steps before bumping its
epoch counter. The registry name is `"ucb2"`; pass `ucb2_alpha` (default
`0.1`) to tune α.

```python
from bandit_kit import ucb2, run_experiment

experiment, results = run_experiment(
    arms,
    algorithms=["ucb2", "ucb1"],
    steps=300,
    runs=10,
    seed=0,
    ucb2_alpha=0.1,
)
```

### IMED

IMED (Indexed Minimum Empirical Divergence; Honda & Takemura, JMLR 2015)
is an asymptotically optimal policy for Bernoulli / `[0, 1]` rewards. After
a one-pull warm-up it pulls the arm with the **smallest** index
`N_a * KL(mean_a, best_mean) + log(N_a)`, where `KL` is the Bernoulli
divergence. It matches the KL-UCB regret guarantee but needs no
root-finding. The registry name is `"imed"`.

```python
from bandit_kit import imed, run_experiment

experiment, results = run_experiment(
    arms,
    algorithms=["imed", "kl_ucb", "ucb1"],
    steps=500,
    runs=10,
    seed=0,
)
```

### Tsallis-INF

Tsallis-INF (Zimmert & Seldin, JMLR 2021) is a "best of both worlds"
policy: it is minimax-optimal (`O(sqrt(K T))`) against adversarial rewards
and still gets logarithmic regret on stochastic arms, without knowing
which regime it faces. It runs online mirror descent with the 1/2-Tsallis
entropy over cumulative loss estimates `L_i` (loss = `1 - reward`, rewards
clipped to `[0, 1]`) and samples from `p_i = 4 / (eta_t (L_i - x))^2`,
where Newton's method finds the normaliser `x` and `eta_t = eta_scale / sqrt(t)`.
`estimator="iw"` (importance-weighted, `eta_scale=2`) is the default;
`estimator="rv"` uses the reduced-variance estimator (`eta_scale=4`). The
registry name is `"tsallis_inf"`. For experiments use `tsallis_estimator`,
and on the command line `--tsallis-estimator`.

```python
from bandit_kit import run_experiment

experiment, results = run_experiment(
    arms,
    algorithms=["tsallis_inf", "exp3", "ucb1"],
    steps=1000,
    runs=10,
    seed=0,
    tsallis_estimator="rv",
)
```

### UCB-Tuned

UCB-Tuned (Auer, Cesa-Bianchi & Fischer, 2002) is a variance-aware UCB
index for `[0, 1]`-bounded rewards. After a one-pull warmup it selects

```
mean_i + sqrt( (ln t / n_i) * min(1/4, V_i) )
```

with `V_i = s_i^2 + sqrt(2 ln t / n_i)` and `s_i^2` the empirical
second-moment variance. Arms that look nearly deterministic get a
smaller bonus than plain UCB1, so the policy can exploit earlier.
Rewards outside `[0, 1]` are clipped (matching KL-UCB / EXP3). The
registry name is `"ucb_tuned"`.

```python
from bandit_kit import BanditExperiment, BernoulliArm, ucb_tuned

arms = [BernoulliArm(name="A", p=0.10), BernoulliArm(name="B", p=0.55)]
experiment = BanditExperiment(
    arms=arms,
    algorithms=["ucb_tuned", "ucb1"],
    steps=300,
    runs=20,
    seed=42,
)
```


### MOSS

MOSS (Audibert & Bubeck) is a finite-horizon UCB-style policy. After every
arm has been pulled once it selects

```
mean_a + sqrt( max(0, log(T / (n_a * K))) / (2 * n_a) )
```

where `T` is the known horizon and `K` is the number of arms. Arms that
have already been pulled more than `T/K` times get a zero exploration
bonus.

```python
from bandit_kit import moss, run_experiment, BernoulliArm

arms = [BernoulliArm("A", 0.1), BernoulliArm("B", 0.5), BernoulliArm("C", 0.2)]
experiment, results = run_experiment(
    arms,
    algorithms=["moss", "ucb1"],
    steps=200,
    runs=20,
    moss_horizon=200,
)
```

Tune the horizon from the CLI with `--moss-horizon` (defaults to `--steps`).
The registry name is `"moss"`.

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
