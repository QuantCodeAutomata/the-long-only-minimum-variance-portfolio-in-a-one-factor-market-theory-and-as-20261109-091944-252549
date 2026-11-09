"""Figure 1.1: GMV (long-short) vs LOMV (long-only) portfolio weights against beta.

Reproduces the comparison of the long-short global minimum variance (GMV)
portfolio and the long-only minimum variance (LOMV) portfolio under a
one-factor covariance model with p = 5000 assets (paper Section 3, Figure 1.1).

Run as: python -m exp.exp1
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.generate_synthetic import generate_normal_betas
from src import config
from src.lomv import gmv_weights_one_factor, lomv_weights

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run() -> dict:
    rng = np.random.default_rng(config.EXP1_SEED)
    beta = generate_normal_betas(config.EXP1_P, config.EXP1_BETA_MEAN, config.EXP1_BETA_STD, rng)

    w_gmv = gmv_weights_one_factor(beta, config.EXP1_SIGMA, config.EXP1_DELTA)
    w_lomv, k = lomv_weights(beta, config.EXP1_DELTA, config.EXP1_SIGMA)

    num_positive_gmv = int((w_gmv > 0).sum())
    num_nonzero_gmv = int((w_gmv != 0).sum())
    num_negative_beta_active = int(((beta < 0) & (w_lomv > 0)).sum())

    metrics = {
        "p": config.EXP1_P,
        "sigma": config.EXP1_SIGMA,
        "delta": config.EXP1_DELTA,
        "beta_mean": config.EXP1_BETA_MEAN,
        "beta_std": config.EXP1_BETA_STD,
        "seed": config.EXP1_SEED,
        "gmv_num_nonzero": num_nonzero_gmv,
        "gmv_num_positive": num_positive_gmv,
        "gmv_frac_positive": num_positive_gmv / config.EXP1_P,
        "gmv_weight_mean": float(w_gmv.mean()),
        "gmv_weight_min": float(w_gmv.min()),
        "gmv_weight_max": float(w_gmv.max()),
        "lomv_active_count_k": k,
        "lomv_active_ratio": k / config.EXP1_P,
        "lomv_num_negative_beta_active": num_negative_beta_active,
        "lomv_beta_threshold": float(np.sort(beta)[k - 1]),
        "expected_lomv_active_count_paper": 319,
        "expected_lomv_negative_beta_active_paper": 34,
        "expected_gmv_num_positive_paper": 3304,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "exp1_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    order = np.argsort(beta)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(beta[order], w_gmv[order], s=4, alpha=0.5, label="GMV (long-short) weight", color="tab:red")
    ax.scatter(beta[order], w_lomv[order], s=4, alpha=0.5, label="LOMV (long-only) weight", color="tab:blue")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xlabel(r"asset beta $\beta_i$")
    ax.set_ylabel(r"portfolio weight $w_i$")
    ax.set_title(f"GMV vs LOMV portfolio weights against beta (p = {config.EXP1_P})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig1_1_gmv_vs_lomv_weights.png", dpi=150)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    run()
