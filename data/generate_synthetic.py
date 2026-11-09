"""Synthetic beta generators for the paper's numerical examples (Section 3).

No market data is used in any of the three experiments: the paper's
Section 3 numerical examples are entirely based on simulated factor loadings
(betas) drawn from stated distributions, with the covariance model
Sigma = sigma^2 beta beta^T + delta^2 I. All three generators below are iid
draws, as stated in the paper.
"""

from __future__ import annotations

import numpy as np


def generate_normal_betas(p: int, mean: float, std: float, rng: np.random.Generator) -> np.ndarray:
    """p iid draws from N(mean, std^2) (Figure 1.1 caption; Section 3.1, Figure 3.1 left panel)."""
    assert p >= 1
    assert std > 0
    return rng.normal(loc=mean, scale=std, size=p)


def generate_truncated_normal_betas(
    p: int, mean: float, std: float, rng: np.random.Generator
) -> np.ndarray:
    """p iid draws from N(mean, std^2) conditioned on beta >= 0 (Section 3.1, Figure 3.1 right panel).

    Implemented by rejection sampling (redraw negative values), as described
    in the methodology ("truncated at zero ... negative draws are redrawn").
    """
    assert p >= 1
    assert std > 0
    betas = rng.normal(loc=mean, scale=std, size=p)
    negative = betas < 0
    while np.any(negative):
        betas[negative] = rng.normal(loc=mean, scale=std, size=int(negative.sum()))
        negative = betas < 0
    return betas


def generate_four_point_betas(
    p: int, values: tuple[float, ...], probs: tuple[float, ...], rng: np.random.Generator
) -> np.ndarray:
    """p iid draws from a discrete distribution with atoms `values` and masses `probs`

    (Section 2.3 / Section 3.2, Figure 3.2: P(beta=-1)=0.05, P(beta=1)=0.15,
    P(beta=2)=0.30, P(beta=5)=0.50).
    """
    assert p >= 1
    probs_arr = np.asarray(probs, dtype=float)
    assert abs(probs_arr.sum() - 1.0) < 1e-9, "probabilities must sum to 1"
    return rng.choice(np.asarray(values, dtype=float), size=p, p=probs_arr)
