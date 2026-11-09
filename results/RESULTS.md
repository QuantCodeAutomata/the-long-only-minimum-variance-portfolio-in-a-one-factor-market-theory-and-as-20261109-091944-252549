# Results — The Long-Only Minimum Variance Portfolio in a One-Factor Market: Theory and Asymptotics

Kercheval and Sowunmi, *"The Long-Only Minimum Variance Portfolio in a
One-Factor Market: Theory and Asymptotics"*, Florida State University,
Department of Mathematics.

All numbers below come from `results/exp1_metrics.json`,
`results/exp2_metrics.json`, `results/exp2_table_3_1.csv` and
`results/exp3_metrics.json`, written by the latest run of
`python -m exp.exp1`, `python -m exp.exp2` and `python -m exp.exp3`.

## Experiment 1 — Figure 1.1: GMV vs LOMV portfolio weights against beta

Setup: `p = 5000`, `beta_i ~ N(1, 0.4^2)` iid, `sigma = 0.15`, `delta = 0.3`
(Section 3). Figure: `results/fig1_1_gmv_vs_lomv_weights.png`.

| Quantity | Paper (Figure 1.1 caption) | Reproduced | Verdict |
|---|---|---|---|
| GMV assets with nonzero weight | 5000 (all) | 5000 | reproduced |
| GMV assets with positive weight | 3304 (66.1%) | 3282 (65.6%) | reproduced |
| LOMV active-set size `k` | 319 | 319 | reproduced |
| LOMV active assets with negative beta | 34 | 27 | partially reproduced |

The active-set size matches the paper exactly. The GMV positive-weight count
is within 1% of the paper's value. The count of active assets with negative
beta is of the same order as the paper's value but differs by 7 assets out
of 319; the paper does not report the random seed used to draw the betas, so
this count — which depends on exactly which low-beta assets happen to be
negative in a single draw of 5000 iid `N(1, 0.4^2)` variables — is expected
to vary across realizations of the same experiment. The qualitative result
of Figure 1.1 (the LOMV active set is a small, low-beta subset of a much
larger set of GMV long-short holdings) holds.

## Experiment 2 — Figure 3.1 / Table 3.1: active-ratio convergence

Setup: `p in {500, 3000, 10000}`, 400 trials per `(p, distribution)`,
`sigma = 0.15`, `delta = 0.3`. Figure: `results/fig3_1_active_ratio_boxplots.png`.
Table: `results/exp2_table_3_1.csv`.

Asymptotic limits (Theorem 2.5): `beta* = 0.2884`, `F(beta*) = 0.0376` for
`N(1, 0.4^2)` (Case 1); `beta* = 0`, `F(beta*) = 0` for the same distribution
truncated at zero (Case 3).

### N(1, 0.4^2)

| p | F(y_p) paper | F(y_p) reproduced | simulated mean, paper | simulated mean, reproduced | Verdict |
|---|---|---|---|---|---|
| 500 | 0.1592 | 0.1592 | 0.159 | 0.1603 | reproduced |
| 3000 | 0.0730 | 0.0730 | 0.0732 | 0.0726 | reproduced |
| 10000 | 0.0512 | 0.0512 | 0.0512 | 0.0512 | reproduced |

### N(1, 0.4^2) truncated at zero

| p | F(y_p) paper | F(y_p) reproduced | simulated mean, paper | simulated mean, reproduced | Verdict |
|---|---|---|---|---|---|
| 500 | 0.1468 | 0.1468 | 0.1481 | 0.1469 | reproduced |
| 3000 | 0.0563 | 0.0563 | 0.0561 | 0.0563 | reproduced |
| 10000 | 0.0307 | 0.0307 | 0.0307 | 0.0307 | reproduced |

For every `p` and both distributions: the finite-`p` prediction `F(y_p)` of
Eq. (3.1) matches the paper to the fourth decimal place, the simulated mean
active ratio matches the paper's reported value within its own trial-to-trial
dispersion, and the active ratio decreases monotonically in `p` and is
systematically biased above the asymptotic limit `F(beta*)` for finite `p`,
as Section 3.1 describes. The truncated distribution has a lower active ratio
than the untruncated one at every `p`, consistent with Case 3 having no
negative-beta assets.

## Experiment 3 — Figure 3.2: non-convergence with a four-point distribution

Setup: `p in {500, 3000, 10000}`, 400 trials per `p`,
`P(beta=-1)=0.05, P(beta=1)=0.15, P(beta=2)=0.30, P(beta=5)=0.50`,
`sigma^2 = 1`, `delta^2 = 0.1`. Figure: `results/fig3_2_non_convergence.png`.

The population function `G` has its unique zero at `beta* = 2`, an atom of
`F` with mass 0.30 (`F(beta*-) = 0.20`, `F(beta*) = 0.50`), verified by
assertion in `exp/exp3.py` against the closed-form values in Section 2.3.

| p | low-mode trials, paper | low-mode trials, reproduced | high-mode trials, paper | high-mode trials, reproduced | Verdict |
|---|---|---|---|---|---|
| 500 | 186 | 196 | 214 | 204 | partially reproduced |
| 3000 | 194 | 208 | 206 | 192 | partially reproduced |
| 10000 | 206 | 197 | 194 | 203 | partially reproduced |

A trial is classified as low-mode when its active ratio falls below the
midpoint 0.35 of the two reference lines `F(beta*-) = 0.20` and
`F(beta*) = 0.50`. In every reproduced run, the active ratios cluster tightly
around one of the two reference lines (mean low-mode ratio within 0.004 of
0.20 and mean high-mode ratio within 0.002 of 0.50 at every `p`), and no
trial falls between the two modes — reproducing the qualitative
non-convergence claim of Theorem 2.5, Case 1, Remark 1: `k_p/p` does not
converge as `p -> infinity`. The paper itself attributes the split between
the two modes to a mean-zero random walk (the sign of `G_p(beta*)`), so the
exact low/high split is not expected to match a different random seed; the
paper does not report its seed. The low/high split is within 10 trials
(2.5% of 400) of the paper's counts at `p = 500` and `p = 10000`, and within
14 trials (3.5%) at `p = 3000`; the reproduced counts move toward an
approximately even split at all three `p`, the same qualitative pattern the
paper reports.

## Deviations from the paper

- No market data is used: all three experiments are entirely based on
  synthetically generated factor loadings (`beta`), exactly as specified in
  the paper's Section 3 numerical examples. No substitution of paper-specified
  data was required or made.
- Random seeds are not given in the paper (noted as an uncertainty for all
  three experiments); fixed seeds (`EXP1_SEED = 20260101`, `EXP2_SEED =
  20260102`, `EXP3_SEED = 20260103` in `src/config.py`) were chosen for
  reproducibility of this implementation. This explains the trial-to-trial
  differences noted above (the negative-beta active count in Experiment 1
  and the low/high-mode trial counts in Experiment 3), which depend on the
  specific draw of betas and are expected to vary across seeds.
- The classification of a trial as "low-mode" vs "high-mode" in Experiment 3
  uses the midpoint 0.35 between the paper's two reference lines
  `F(beta*-) = 0.20` and `F(beta*) = 0.50` (`EXP3_MODE_SPLIT` in
  `src/config.py`); the paper does not specify a classification threshold,
  only that trials cluster near one line or the other.
- Plot styles (scatter vs line, exact colors, binning) are not specified by
  the paper for Figures 1.1, 3.1 or 3.2; the choices made here (scatter plots
  of weight/active-ratio against beta/trial index, boxplots for Figure 3.1,
  dashed reference lines for the theoretical limits) follow the qualitative
  description in the figure captions.

## Not reproduced

Nothing in the scope of the three selected experiments (Figures 1.1, 3.1,
3.2 and Table 3.1) was left unreproduced.
