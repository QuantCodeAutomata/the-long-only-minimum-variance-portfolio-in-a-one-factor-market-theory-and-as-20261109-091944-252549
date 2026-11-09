"""Figure 3.2: non-convergence of the active ratio with a discrete four-point
beta distribution.

Reproduces the non-convergence phenomenon of Theorem 2.5, Case 1, Remark 1,
using a four-point distribution whose unique zero beta* = 2 of G is an atom
of F (paper Section 2.3 and Section 3.2). For each p in {500, 3000, 10000},
400 independent trials are classified as low-mode (the block at beta = 2
inactive, k_p/p near F(beta*-) = 0.20) or high-mode (that block active,
k_p/p near F(beta*) = 0.50).

Run as: python -m exp.exp3
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.generate_synthetic import generate_four_point_betas
from src import config
from src.active_ratio import F_discrete, G_discrete
from src.simulation import simulate_active_ratios

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Paper-reported low-mode / high-mode trial counts (Section 3.2, Figure 3.2
# table) used only for the comparison in results/RESULTS.md.
PAPER_MODE_COUNTS = {500: (186, 214), 3000: (194, 206), 10000: (206, 194)}


def run() -> dict:
    rng = np.random.default_rng(config.EXP3_SEED)
    values = np.asarray(config.EXP3_BETA_VALUES)
    probs = np.asarray(config.EXP3_BETA_PROBS)
    sigma = float(np.sqrt(config.EXP3_SIGMA2))
    delta = float(np.sqrt(config.EXP3_DELTA2))

    beta_star = 2.0  # unique zero of G for this distribution (Section 2.3, verified below)
    assert abs(G_discrete(beta_star, values, probs)) < 1e-9, "beta* = 2 must be an exact zero of G"
    assert G_discrete(beta_star - 1e-6, values, probs) > 0, "G must be positive just below beta*"
    assert G_discrete(beta_star + 1e-6, values, probs) < 0, "G must be negative just above beta*"

    F_low = F_discrete(beta_star, values, probs, strict=True)   # F(beta*-)
    F_high = F_discrete(beta_star, values, probs, strict=False)  # F(beta*)
    assert abs(F_low - config.EXP3_LOW_MODE_REF) < 1e-9
    assert abs(F_high - config.EXP3_HIGH_MODE_REF) < 1e-9

    results: dict = {
        "beta_star": beta_star,
        "F_beta_star_minus": F_low,
        "F_beta_star": F_high,
        "p_results": {},
    }
    scatter_data = {}

    for p in config.EXP3_P_VALUES:
        ratios = simulate_active_ratios(
            p, config.EXP3_N_TRIALS,
            lambda n, r: generate_four_point_betas(n, values, probs, r),
            sigma, delta, rng,
        )
        low_mode = ratios < config.EXP3_MODE_SPLIT
        high_mode = ~low_mode
        paper_counts = PAPER_MODE_COUNTS.get(p)
        results["p_results"][p] = {
            "n_low_mode": int(low_mode.sum()),
            "n_high_mode": int(high_mode.sum()),
            "paper_n_low_mode": paper_counts[0] if paper_counts else None,
            "paper_n_high_mode": paper_counts[1] if paper_counts else None,
            "mean_low_mode_ratio": float(ratios[low_mode].mean()) if low_mode.any() else None,
            "mean_high_mode_ratio": float(ratios[high_mode].mean()) if high_mode.any() else None,
        }
        scatter_data[p] = ratios

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "exp3_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    _plot_figure_3_2(scatter_data, values, probs, F_low, F_high)

    print(json.dumps(results, indent=2))
    return results


def _plot_figure_3_2(
    scatter_data: dict, values: np.ndarray, probs: np.ndarray, F_low: float, F_high: float
) -> None:
    p_values = list(config.EXP3_P_VALUES)
    fig, axes = plt.subplots(1, len(p_values) + 1, figsize=(4 * (len(p_values) + 1), 4.5))

    for ax, p in zip(axes[:-1], p_values):
        ratios = scatter_data[p]
        colors = np.where(ratios < config.EXP3_MODE_SPLIT, "tab:blue", "tab:red")
        ax.scatter(np.arange(len(ratios)), ratios, s=6, c=colors, alpha=0.6)
        ax.axhline(F_low, color="tab:blue", linestyle="--", linewidth=1, label=f"F(beta*-) = {F_low:.2f}")
        ax.axhline(F_high, color="tab:red", linestyle="--", linewidth=1, label=f"F(beta*) = {F_high:.2f}")
        ax.set_title(f"p = {p}")
        ax.set_xlabel("trial")
        ax.set_ylabel(r"active ratio $k_p/p$")
        ax.legend(fontsize=8)

    y_grid = np.linspace(-0.5, 6.5, 400)
    g_vals = [G_discrete(y, values, probs) for y in y_grid]
    axes[-1].plot(y_grid, g_vals, color="black")
    axes[-1].axhline(0.0, color="gray", linewidth=0.5)
    axes[-1].axvline(2.0, color="gray", linestyle=":", label=r"$\beta^*=2$")
    axes[-1].set_xlabel("y")
    axes[-1].set_ylabel("G(y)")
    axes[-1].set_title("G function (Eq. 2.16)")
    axes[-1].legend()

    fig.suptitle("Figure 3.2: non-convergence of the active ratio (four-point distribution)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig3_2_non_convergence.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
