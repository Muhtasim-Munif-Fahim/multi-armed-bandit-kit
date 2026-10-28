# multi-armed-bandit-kit

A small, dependency-free Python toolkit for studying multi-armed bandit
algorithms in reproducible research settings. It implements classic
stochastic policies (`epsilon-greedy`, `UCB1`, `UCB-Tuned`, `sliding-window UCB`, `SW-TS`, `DiscountedUCB`, `UCB-V`, `UCB2`, `MOSS`, `Thompson sampling`,
`KL-UCB`, `IMED`) plus the adversarial-bandit policies `EXP3`, `EXP3-IX`, and the switching-adversary
`EXP3.S`, the best-of-both-worlds
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

`compare` runs epsilon-greedy, UCB1, UCB-Tuned, sliding-window UCB, SW-TS, DiscountedUCB, UCB-V, UCB2, MOSS, Thompson sampling, EXP3, EXP3-IX, EXP3.S,
Boltzmann / softmax, and KL-UCB. Tune epsilon-greedy with `--epsilon`,
EXP3 / EXP3.S with `--gamma`, EXP3-IX with `--exp3-ix-eta`, EXP3.S's
fixed-share switching rate with `--exp3-s-alpha`, Boltzmann with `--temperature-start`,
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

### EXP3.S (switching adversaries)

`EXP3` competes with the single best arm in hindsight, so once one arm has
dominated for a while its weight is astronomically larger than the rest and
the policy cannot follow a change in which arm is best. `EXP3.S` (Auer,
Cesa-Bianchi, Freund & Schapire 2002) adds a *fixed share* after each
importance-weighted update:

```
w_i <- w_i * exp(gamma * x_hat_i / K) + (e * alpha / K) * W_t
```

where `W_t` is the total weight before the update. Every arm keeps a floor
proportional to the total, so the policy re-allocates within a few dozen
pulls after the best arm switches. `alpha = 0` recovers `EXP3` exactly.
`exp3_s_parameters(n_arms, horizon, switches)` returns the paper's tuned
`(gamma, alpha)` for a horizon `T` with at most `S` switches
(`alpha = 1/T`).

```python
from bandit_kit import BanditExperiment, BernoulliArm, exp3_s_parameters

gamma, alpha = exp3_s_parameters(n_arms=2, horizon=2000, switches=2)
exp = BanditExperiment(
    arms=[BernoulliArm("a", 0.3), BernoulliArm("b", 0.6)],
    algorithms=["exp3_s", "exp3_ix", "exp3"],
    steps=2000,
    gamma=gamma,
    exp3_s_alpha=alpha,
)
print(exp.summarize(exp.run()))
```

`exp3_ix` and `exp3_s` are both available in `BanditExperiment`,
`ContextualBanditExperiment`, and `bandit-kit compare`.

### Best-arm identification (Successive Rejects, Sequential Halving, LUCB, Top-Two TS, lil'UCB)

The policies above minimise regret. Pure-exploration *best-arm identification* has a different goal: run the experiment, then name the arm with the highest mean.

- `successive_rejects(arms, budget)`: fixed budget (Audibert & Bubeck, 2010). Runs `K - 1` phases of lengths `n_k = ceil((n - K) / (logbar(K) (K + 1 - k)))` and drops the empirically worst arm after each one. Needs no tuning.
- `successive_halving(arms, budget)`: fixed budget (Karnin, Koren & Somekh, 2013). Runs `ceil(log2 K)` rounds on fresh samples and keeps the top half each round.
- `lucb(arms, delta=0.05, epsilon=0.0)`: fixed confidence (LUCB1, Kalyanakrishnan et al., 2012). Pulls the leader and the strongest UCB challenger, and stops once the leader is `epsilon`-optimal with probability at least `1 - delta`.
- `top_two_thompson(arms, delta=0.05, beta=0.5, model="bernoulli", challenger="t3c")`: fixed confidence (Russo 2016; T3C, Shang et al. 2020). Each round samples the posterior and plays the sampled best arm with probability `beta`. Otherwise it plays a challenger: either the arm with the smallest transportation cost (`t3c`) or a resampled runner-up (`ttts`). It stops on the Chernoff GLR rule of Garivier & Kaufmann (2016), `min_j Z_leader,j > log(2t(K−1)/δ)`, using Bernoulli or Gaussian KL. `chernoff_glr_statistic` exposes `Z`.
- `lil_ucb(arms, delta=0.05, epsilon=0.01, beta=1.0)`: fixed confidence (Jamieson et al., 2014). It pulls the arm with the largest law-of-the-iterated-logarithm UCB, and stops once one arm has `n_i ≥ 1 + λ Σ_{j≠i} n_j`, where `λ = ((2+β)/β)²` by default.
- `identification_error_rate(method, arms, n_runs, seed, **kwargs)`: Monte Carlo estimate of the misidentification rate and the mean number of pulls, using reseeded copies of the arms.

```python
from bandit_kit import BernoulliArm, identification_error_rate, lucb, successive_rejects

arms = [BernoulliArm(f"v{i}", p, seed=i) for i, p in enumerate([0.50, 0.45, 0.40, 0.40])]
res = successive_rejects(arms, budget=3000)
print(res.recommended, res.elimination_order, res.total_pulls)

print(identification_error_rate(successive_rejects, arms, n_runs=200, budget=3000))
easy = [BernoulliArm("a", 0.8, seed=1), BernoulliArm("b", 0.3, seed=2)]
print(lucb(easy, delta=0.05).total_pulls)  # stops as soon as the gap is resolved
```

```bash
bandit-kit identify --arms bern:0.5,bern:0.45,bern:0.4 --method sr --budget 3000 --runs 200
bandit-kit identify --arms bern:0.8,bern:0.3 --method lucb --delta 0.05 --json
bandit-kit identify --arms bern:0.5,bern:0.4,bern:0.3 --method t3c --delta 0.05 --runs 50
bandit-kit identify --arms bern:0.5,bern:0.4,bern:0.3 --method lil
```

On `bern:0.5,0.4,0.3` with δ = 0.05, all three fixed-confidence methods identified the best arm in every run. Mean pulls per run were about 17,000 for LUCB1, 2,500 for Top-Two TS (T3C) and 8,000 for lil'UCB. Top-Two TS with GLR stopping is far more sample-efficient than Hoeffding-based LUCB.

### Off-policy evaluation (IPS, SNIPS, DM, DR, replay)

Estimate how a *new* policy would perform from logs collected by the deployed one, without deploying it. Each logged event records the arm played, its reward, and the logging policy's propensity for that arm (`LoggedEvent(arm_index, reward, propensity, context=None)`).

- `ips_estimate(logs, target, clip=None)`: inverse propensity scoring `mean(w r)` with `w = pi(a)/mu(a)`. It is unbiased under full support. `clip` caps the weights, trading some downward bias for lower variance.
- `snips_estimate`: self-normalised IPS `sum(w r) / sum(w)` with a delta-method standard error. It is usually far more stable when the weights are heavy-tailed.
- `direct_method_estimate(logs, target, reward_model=None)`: model-based plug-in. The default model is the per-arm empirical mean. It is biased if the model is wrong.
- `doubly_robust_estimate`: DM plus an IPS correction of the model residual. It is unbiased if either the propensities or the model are right.
- `evaluate_policy(...)`: runs all four estimators, plus clipped IPS when `clip` is given. Every result reports a standard error, a normal CI, and the Kish effective sample size `(sum w)^2 / sum w^2`.
- `replay_evaluate(algorithm, arms, logs)`: the replay method (Li et al., 2011) for *learning* algorithms under uniformly random logging. The algorithm only sees, and learns from, the events where it agrees with the logged arm. The matched rewards are an unbiased sample of its online run.
- `collect_logs(arms, n, logging_policy=None)` simulates logged data, and `true_policy_value(arms, policy)` gives the ground truth.

Policies are an arm index (deterministic), a list of probabilities, or `policy(context) -> probabilities`.

```python
from bandit_kit import BernoulliArm, collect_logs, evaluate_policy, replay_evaluate, ucb1

arms = [BernoulliArm(f"a{i}", p, seed=i) for i, p in enumerate([0.3, 0.5, 0.7])]
logs = collect_logs(arms, 5000, logging_policy=[0.6, 0.3, 0.1], seed=1)
for name, est in evaluate_policy(logs, target=[0.1, 0.1, 0.8], clip=10).items():
    print(name, round(est.value, 4), (round(est.ci_low, 3), round(est.ci_high, 3)))

uniform = collect_logs(arms, 6000, seed=2)
print(replay_evaluate(ucb1(seed=0), arms, uniform).mean_reward)
```

```bash
bandit-kit offpolicy --arms bern:0.3,bern:0.5,bern:0.7 --target 0.1,0.1,0.8 --logging 0.6,0.3,0.1 --n 5000
bandit-kit offpolicy --arms bern:0.3,bern:0.5,bern:0.7 --target 2 --clip 5 --replay ucb1 --json
```

## Dueling bandits (pairwise preferences)

When feedback is relative — interleaved ranker clicks, "which response is
better?" ratings — `bandit_kit.dueling` learns from duels. A
`PreferenceMatrix` holds `P[i][j] = P(i beats j)` (build one directly or with
`PreferenceMatrix.from_utilities(u, link="logistic")`, i.e. Bradley–Terry).
Regret is measured against the **Condorcet winner**: strong regret
`(eps_i + eps_j)/2` and weak regret `min(eps_i, eps_j)` per duel, with
`eps_i = P[c][i] - 1/2`.

- `RUCB` — Relative Upper Confidence Bound (Zoghi et al., 2014).
- `DoubleThompsonSampling` — D-TS (Wu & Liu, 2016).
- `UniformDueling` — random-pair baseline.
- `condorcet_winner`, `copeland_winners`, `run_dueling`, `compare_dueling`.

```python
from bandit_kit import DoubleThompsonSampling, PreferenceMatrix, run_dueling

env = PreferenceMatrix.from_utilities([0.0, 0.3, 0.6, 1.0, 0.8], seed=1)
res = run_dueling(DoubleThompsonSampling(seed=0), env, horizon=2000)
print(res.recommendation, res.cumulative_strong_regret, res.cumulative_weak_regret)
```

```bash
bandit-kit duel --utilities 0,0.3,0.6,1.0,0.8 --horizon 2000 --runs 10
```

## Piecewise-stationary bandits (CUSUM-UCB, M-UCB)

When arm means jump at a few unknown times, *actively adaptive* policies run
a change-point detector and restart when it fires:

* `CUSUMUCB(m, eps, h, alpha)` — CUSUM-UCB (Liu, Lee & Shroff, AAAI 2018):
  per-arm two-sided CUSUM test after `m` reference samples; an alarm resets
  that arm only. UCB on post-reset statistics plus `alpha` uniform exploration.
* `MUCB(w, b, gamma, horizon)` — Monitored UCB (Cao et al., AISTATS 2019):
  `gamma` share of forced round-robin exploration and a windowed mean-shift
  test `|Σ second half − Σ first half| > b` over each arm's last `w` rewards;
  an alarm restarts all arms. `b` defaults to `sqrt(w/2 · log(2KT²))`.
* `PiecewiseBernoulli(means, breakpoints)` environment, `run_piecewise`
  (dynamic regret vs. the best arm of the current segment, alarm times,
  detection delays) and `compare_piecewise` (also runs SW-UCB, D-UCB and UCB1).

```python
from bandit_kit import compare_piecewise
compare_piecewise([[0.9, 0.5], [0.3, 0.5]], breakpoints=[1500], horizon=3000, runs=10)
```

```bash
bandit-kit piecewise --means "0.9,0.5;0.3,0.5" --breakpoints 1500 --horizon 3000 --runs 10
```
