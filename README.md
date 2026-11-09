# The Long-Only Minimum Variance Portfolio in a One-Factor Market: Theory and Asymptotics

Reproduction of selected numerical experiments from Kercheval and Sowunmi,
*"The Long-Only Minimum Variance Portfolio in a One-Factor Market: Theory and
Asymptotics"*, Florida State University, Department of Mathematics.

The paper studies the long-only minimum-variance (LOMV) portfolio under a
one-factor covariance model `Sigma = sigma^2 * beta * beta^T + delta^2 * I`
and shows (Theorem 2.2) that its active set is the subset of lowest-beta
assets whose size `k` is found by a closed-form recursion. Section 2.3 and
Theorem 2.5 give the asymptotic active ratio `k_p/p -> F(beta*)` as the
number of assets `p -> infinity`, where `beta*` is the unique zero of a
population function `G`, together with a counterexample distribution for
which the ratio does not converge. Section 3 illustrates both results
numerically.

All three experiments below use only synthetically generated factor loadings
(`beta`), exactly as in the paper's Section 3; no market data is needed.

## Repository layout

```
src/            portfolio theory and asymptotics (flat modules)
exp/            experiment scripts (python -m exp.<name>)
data/           synthetic beta generators (no market data loader needed)
tests/          pytest unit and integration tests
results/        generated metrics, tables and figures + RESULTS.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Tests

```bash
python -m pytest
```

## Experiments

```bash
python -m exp.exp1   # Figure 1.1: GMV vs LOMV weights against beta, p = 5000
python -m exp.exp2   # Figure 3.1 / Table 3.1: active-ratio convergence (Normal / truncated Normal)
python -m exp.exp3   # Figure 3.2: non-convergence with a four-point discrete beta distribution
```

Each script writes its metrics, tables and figures directly to `results/`
and prints a JSON summary to stdout. `reproduce.sh` runs the tests and all
three experiments in sequence.

```bash
./reproduce.sh
```

## Experiment summaries

### Experiment 1 — Figure 1.1: GMV vs LOMV portfolio weights against beta

For `p = 5000` assets with `beta_i ~ N(1, 0.4^2)` iid and
`Sigma = 0.15^2 * beta * beta^T + 0.3^2 * I`, computes the long-short global
minimum-variance (GMV) weights in closed form and the long-only minimum
variance (LOMV) weights via the active-set recursion of Theorem 2.2, using
the Woodbury identity to invert the rank-one perturbed diagonal covariance
matrix in `O(p)` time. Produces `results/fig1_1_gmv_vs_lomv_weights.png` and
`results/exp1_metrics.json`.

### Experiment 2 — Figure 3.1 / Table 3.1: active-ratio convergence

For `p` in `{500, 3000, 10000}` and 400 independent trials per `(p,
distribution)`, simulates the active ratio `k_p/p` for `beta ~ N(1, 0.4^2)`
(Theorem 2.5, Case 1) and the same distribution truncated at zero (Case 3),
and compares the simulated means with the asymptotic limit `F(beta*)` and
the finite-`p` prediction `F(y_p)` of Eq. (3.1), where `y_p` solves
`G(y_p) = -nu^2/(p * sigma^2)`. Produces
`results/fig3_1_active_ratio_boxplots.png`, `results/exp2_table_3_1.csv` and
`results/exp2_metrics.json`.

### Experiment 3 — Figure 3.2: non-convergence with a four-point distribution

Using the discrete distribution `P(beta=-1)=0.05, P(beta=1)=0.15,
P(beta=2)=0.30, P(beta=5)=0.50`, whose unique zero `beta* = 2` of `G` is an
atom of `F`, simulates 400 trials per `p in {500, 3000, 10000}` and shows
that the active ratio does not converge: it clusters near `F(beta*-) = 0.20`
(the block at `beta = 2` inactive) or `F(beta*) = 0.50` (that block active).
Produces `results/fig3_2_non_convergence.png` and `results/exp3_metrics.json`.

## Results

See [`results/RESULTS.md`](results/RESULTS.md) for the full numerical
comparison with the paper, deviations, and items not reproduced.
