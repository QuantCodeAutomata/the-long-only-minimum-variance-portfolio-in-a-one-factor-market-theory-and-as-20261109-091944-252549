"""Figure 3.1 and Table 3.1: active-ratio convergence for Normal and truncated
Normal beta distributions.

Reproduces the boxplots of the active ratio k_p/p for p in {500, 3000, 10000}
under beta ~ N(1, 0.4^2) (Theorem 2.5, Case 1) and the same distribution
truncated at zero (Theorem 2.5, Case 3), compared with the asymptotic limit
F(beta*) and the finite-p prediction F(y_p) of Eq. (3.1) (paper Section 3.1).

Run as: python -m exp.exp2
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.generate_synthetic import generate_normal_betas, generate_truncated_normal_betas
from src import config
from src.active_ratio import (
    F_normal,
    F_truncated_normal,
    G_normal,
    G_truncated_normal,
    beta_star_normal,
    finite_p_threshold,
)
from src.simulation import simulate_active_ratios

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Paper-reported reference numbers (Section 3.1, Table 3.1, Figure 3.1) used
# only for the comparison printed in results/RESULTS.md, never for computation.
PAPER_TABLE_3_1 = {
    "normal": {
        500: {"y_p": 0.6009, "F_y_p": 0.1592, "sim_mean": 0.1590},
        3000: {"y_p": 0.4185, "F_y_p": 0.0730, "sim_mean": 0.0732},
        10000: {"y_p": 0.3468, "F_y_p": 0.0512, "sim_mean": 0.0512},
    },
    "truncated": {
        500: {"y_p": 0.5891, "F_y_p": 0.1468, "sim_mean": 0.1481},
        3000: {"y_p": 0.3851, "F_y_p": 0.0563, "sim_mean": 0.0561},
        10000: {"y_p": 0.2838, "F_y_p": 0.0307, "sim_mean": 0.0307},
    },
}


def run(full: bool = True) -> dict:
    """Run the Figure 3.1 / Table 3.1 simulation.

    `full` selects the paper's configuration (p in {500, 3000, 10000}, 400
    trials); the one-factor active-set algorithm is O(p log p) per trial, so
    this configuration runs quickly and is always the one executed (no
    reduction is needed for this experiment).
    """
    rng = np.random.default_rng(config.EXP2_SEED)
    mu, std = config.EXP2_BETA_MEAN, config.EXP2_BETA_STD
    sigma, delta = config.EXP2_SIGMA, config.EXP2_DELTA
    nu2_over_sigma2 = delta ** 2 / sigma ** 2   # nu = delta for constant delta_i

    beta_star = beta_star_normal(mu, std)
    F_beta_star_normal = F_normal(beta_star, mu, std)
    F_beta_star_truncated = 0.0  # Case 3: beta* = 0, F(beta*) = 0 (paper Section 3.1)

    results: dict = {
        "beta_star_normal": beta_star,
        "F_beta_star_normal": F_beta_star_normal,
        "beta_star_truncated": 0.0,
        "F_beta_star_truncated": F_beta_star_truncated,
        "distributions": {"normal": {}, "truncated": {}},
    }

    boxdata = {"normal": [], "truncated": []}
    table_rows = []

    for p in config.EXP2_P_VALUES:
        nu2_over_p_sigma2 = nu2_over_sigma2 / p

        ratios_normal = simulate_active_ratios(
            p, config.EXP2_N_TRIALS,
            lambda n, r: generate_normal_betas(n, mu, std, r),
            sigma, delta, rng,
        )
        ratios_truncated = simulate_active_ratios(
            p, config.EXP2_N_TRIALS,
            lambda n, r: generate_truncated_normal_betas(n, mu, std, r),
            sigma, delta, rng,
        )
        boxdata["normal"].append(ratios_normal)
        boxdata["truncated"].append(ratios_truncated)

        y_p_normal = finite_p_threshold(
            lambda y: G_normal(y, mu, std), beta_star, nu2_over_p_sigma2, search_hi=beta_star + 20.0
        )
        F_y_p_normal = F_normal(y_p_normal, mu, std)

        y_p_truncated = finite_p_threshold(
            lambda y: G_truncated_normal(y, mu, std), 0.0, nu2_over_p_sigma2, search_hi=10.0
        )
        F_y_p_truncated = F_truncated_normal(y_p_truncated, mu, std)

        results["distributions"]["normal"][p] = {
            "simulated_mean": float(ratios_normal.mean()),
            "simulated_std": float(ratios_normal.std()),
            "y_p": y_p_normal,
            "F_y_p": F_y_p_normal,
        }
        results["distributions"]["truncated"][p] = {
            "simulated_mean": float(ratios_truncated.mean()),
            "simulated_std": float(ratios_truncated.std()),
            "y_p": y_p_truncated,
            "F_y_p": F_y_p_truncated,
        }
        paper_normal = PAPER_TABLE_3_1["normal"].get(p)
        paper_truncated = PAPER_TABLE_3_1["truncated"].get(p)
        table_rows.append(
            {
                "p": p,
                "normal_y_p": y_p_normal,
                "normal_F_y_p": F_y_p_normal,
                "normal_sim_mean": float(ratios_normal.mean()),
                "normal_paper_sim_mean": paper_normal["sim_mean"] if paper_normal else None,
                "truncated_y_p": y_p_truncated,
                "truncated_F_y_p": F_y_p_truncated,
                "truncated_sim_mean": float(ratios_truncated.mean()),
                "truncated_paper_sim_mean": paper_truncated["sim_mean"] if paper_truncated else None,
            }
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "exp2_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(RESULTS_DIR / "exp2_table_3_1.csv", "w") as f:
        header = list(table_rows[0].keys())
        f.write(",".join(header) + "\n")
        for row in table_rows:
            f.write(",".join(str(row[h]) for h in header) + "\n")

    _plot_figure_3_1(boxdata, F_beta_star_normal, F_beta_star_truncated)

    print(json.dumps(results, indent=2))
    return results


def _plot_figure_3_1(boxdata: dict, F_beta_star_normal: float, F_beta_star_truncated: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
    p_values = list(config.EXP2_P_VALUES)

    for ax, key, F_star, title in (
        (axes[0], "normal", F_beta_star_normal, r"$\beta \sim N(1, 0.4^2)$"),
        (axes[1], "truncated", F_beta_star_truncated, r"$\beta \sim N(1, 0.4^2)$ truncated at 0"),
    ):
        data = boxdata[key]
        bp = ax.boxplot(data, positions=range(len(p_values)), widths=0.5, showmeans=True)
        ax.axhline(F_star, color="red", linestyle="--", label=r"$F(\beta^*)$" + f" = {F_star:.4f}")
        ax.set_xticks(range(len(p_values)))
        ax.set_xticklabels([str(p) for p in p_values])
        ax.set_xlabel("p")
        ax.set_ylabel(r"active ratio $k_p/p$")
        ax.set_title(title)
        ax.legend()

    fig.suptitle("Figure 3.1: distribution of the active ratio over 400 trials")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig3_1_active_ratio_boxplots.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
