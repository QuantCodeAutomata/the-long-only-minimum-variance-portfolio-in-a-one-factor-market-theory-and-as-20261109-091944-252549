"""Parameters for every experiment, taken from the paper's Section 3 setup.

No classes or dicts: every parameter is a module-level UPPER_CASE variable
with a comment naming its meaning, unit, and source.
"""

# ---------------------------------------------------------------------------
# Shared covariance-model convention
# ---------------------------------------------------------------------------
# Sigma = sigma^2 * beta * beta^T + delta^2 * I_p  (Section 2, Eq. 2.4;
# Section 3 numerical examples). sigma and delta are asset-independent
# (delta_i = delta for all i) in every experiment here.

# ---------------------------------------------------------------------------
# Experiment 1 (Figure 1.1): GMV vs LOMV weights against beta, p = 5000
# ---------------------------------------------------------------------------
EXP1_P = 5000              # number of assets, Section 3 / Figure 1.1 caption
EXP1_SIGMA = 0.15          # factor volatility, Section 3 ("parameters taken from Clarke et al. [2011]")
EXP1_DELTA = 0.30          # constant idiosyncratic volatility delta_i, Section 3
EXP1_BETA_MEAN = 1.0       # beta_i ~ N(1, 0.4^2) iid, Figure 1.1 caption
EXP1_BETA_STD = 0.4        # beta_i ~ N(1, 0.4^2) iid, Figure 1.1 caption
EXP1_SEED = 20260101       # random seed, not specified in the paper

# ---------------------------------------------------------------------------
# Experiment 2 (Figure 3.1 / Table 3.1): active-ratio convergence
# ---------------------------------------------------------------------------
EXP2_P_VALUES = (500, 3000, 10000)   # asset-universe sizes, Section 3.1
EXP2_N_TRIALS = 400                  # independent trials per (p, distribution), Section 3.1
EXP2_SIGMA = 0.15                    # factor volatility, Section 3
EXP2_DELTA = 0.30                    # constant idiosyncratic volatility, Section 3
EXP2_BETA_MEAN = 1.0                 # beta_i ~ N(1, 0.4^2), Section 3.1
EXP2_BETA_STD = 0.4                  # beta_i ~ N(1, 0.4^2), Section 3.1
EXP2_SEED = 20260102                 # random seed, not specified in the paper

# ---------------------------------------------------------------------------
# Experiment 3 (Figure 3.2): non-convergence with a four-point distribution
# ---------------------------------------------------------------------------
EXP3_P_VALUES = (500, 3000, 10000)   # asset-universe sizes, Section 3.2
EXP3_N_TRIALS = 400                  # independent trials per p, Section 3.2
EXP3_SIGMA2 = 1.0                    # sigma^2, "arbitrarily select sigma^2 = 1", Section 3.2
EXP3_DELTA2 = 0.1                    # delta^2, "arbitrarily select delta^2 = 0.1", Section 3.2
EXP3_BETA_VALUES = (-1.0, 1.0, 2.0, 5.0)     # four-point support, Section 2.3 / 3.2
EXP3_BETA_PROBS = (0.05, 0.15, 0.30, 0.50)   # four-point probabilities, Section 2.3 / 3.2
EXP3_LOW_MODE_REF = 0.20              # F(beta*-), reference line for the inactive block, Section 2.3
EXP3_HIGH_MODE_REF = 0.50             # F(beta*), reference line for the active block, Section 2.3
EXP3_MODE_SPLIT = 0.35                # midpoint used only to classify trials as low/high mode for reporting; not a paper quantity
EXP3_SEED = 20260103                  # random seed, not specified in the paper
