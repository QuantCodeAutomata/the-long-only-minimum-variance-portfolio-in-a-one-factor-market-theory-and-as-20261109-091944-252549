"""Monte Carlo active-ratio simulation shared by Figures 3.1 and 3.2.

Custom implementation (paper Section 3: "For each p ... we repeat the
experiment 400 times"): repeatedly draws betas from a supplied generator,
applies the Theorem 2.2 active-set solver, and records k_p/p.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.lomv import active_set_size


def simulate_active_ratios(
    p: int,
    n_trials: int,
    draw_betas: Callable[[int, np.random.Generator], np.ndarray],
    sigma: float,
    delta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Active ratio k_p/p over `n_trials` independent trials of p assets.

    `draw_betas(p, rng)` must return a length-p array of iid beta draws.
    delta_i is held constant at `delta` for every asset, as in Section 3.
    """
    assert p >= 1
    assert n_trials >= 1
    ratios = np.empty(n_trials, dtype=float)
    for t in range(n_trials):
        beta = draw_betas(p, rng)
        k, _ = active_set_size(beta, delta, sigma)
        ratios[t] = k / p
    return ratios
