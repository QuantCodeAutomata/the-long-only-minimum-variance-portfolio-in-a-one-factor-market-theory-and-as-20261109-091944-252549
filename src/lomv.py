"""GMV and LOMV portfolio solutions under a one-factor covariance model.

Sigma = sigma^2 * beta * beta^T + delta^2 * I_p  (paper Eq. 2.4). All
functions use the Woodbury identity to invert this rank-one perturbation of a
diagonal matrix in O(p) time, as suggested in the paper's numerical-examples
section, instead of forming or inverting the p x p matrix Sigma directly.

Custom implementation (paper Proposition 2.1 and Theorem 2.2): no library
(skfolio, pyportfolioopt) implements the long-only minimum-variance solution
of a one-factor model via this closed-form active-set criterion, so the
closed-form recursion of Theorem 2.2 is implemented here from scratch.
"""

from __future__ import annotations

import numpy as np


def gmv_weights_one_factor(
    beta: np.ndarray, sigma: float, delta: np.ndarray | float
) -> np.ndarray:
    """Long-short global minimum variance (GMV) weights, Eq. (1.1)/(2.3): w^LS = Sigma^-1 1 / (1^T Sigma^-1 1).

    Uses the Woodbury identity for Sigma = sigma^2 beta beta^T + delta^2 I:
        Sigma^-1 = D^-1 - (D^-1 beta)(D^-1 beta)^T / (1/sigma^2 + beta^T D^-1 beta)
    with D = diag(delta_i^2), so that Sigma^-1 1 is obtained in O(p) time
    without forming Sigma.

    Parameters
    ----------
    beta : (k,) array of factor loadings.
    sigma : factor volatility (sigma > 0).
    delta : (k,) array, or scalar, of idiosyncratic volatilities (delta_i > 0).

    Returns
    -------
    (k,) array of portfolio weights summing to 1.
    """
    beta = np.asarray(beta, dtype=float)
    k = beta.shape[0]
    assert k >= 1, "need at least one asset"
    assert sigma > 0, "sigma must be positive"
    delta = np.broadcast_to(np.asarray(delta, dtype=float), (k,))
    assert np.all(delta > 0), "delta_i must be positive"

    inv_delta2 = 1.0 / (delta ** 2)          # diag(D^-1)
    d_inv_beta = inv_delta2 * beta           # D^-1 beta
    d_inv_1 = inv_delta2                     # D^-1 1 (since 1 is all ones)
    denom = 1.0 / sigma ** 2 + float(beta @ d_inv_beta)  # 1/sigma^2 + beta^T D^-1 beta
    assert denom > 0, "Woodbury denominator must be positive for Sigma to be invertible"

    sigma_inv_1 = d_inv_1 - d_inv_beta * (float(beta @ d_inv_1) / denom)
    total = float(np.sum(sigma_inv_1))
    assert total != 0, "1^T Sigma^-1 1 must be nonzero"
    return sigma_inv_1 / total


def compute_R_sequence(beta_sorted: np.ndarray, delta_sorted: np.ndarray, sigma: float) -> np.ndarray:
    """R_i sequence of Theorem 2.2, for betas already sorted in increasing order.

    R_1 = 1/sigma^2
    R_i = 1/sigma^2 + sum_{j=1}^{i-1} (beta_j/delta_j^2)(beta_j - beta_i), i = 2..p

    computed via the recursion R_{i+1} = R_i + (beta_i - beta_{i+1}) * C_i,
    with C_i = sum_{j=1}^{i} beta_j/delta_j^2 (paper Theorem 2.2 and the
    implementation note accompanying it).
    """
    beta_sorted = np.asarray(beta_sorted, dtype=float)
    delta_sorted = np.asarray(delta_sorted, dtype=float)
    p = beta_sorted.shape[0]
    assert p >= 1
    assert np.all(np.diff(beta_sorted) >= -1e-12), "betas must be sorted ascending"

    weighted_beta = beta_sorted / (delta_sorted ** 2)   # beta_j / delta_j^2
    C = np.cumsum(weighted_beta)                        # C_i = sum_{j<=i} beta_j/delta_j^2

    R = np.empty(p, dtype=float)
    R[0] = 1.0 / sigma ** 2
    if p > 1:
        # R[i] = R_{i+1} for i = 1..p-1 (0-indexed), using C_i = C[i-1]
        R[1:] = R[0] + np.cumsum((beta_sorted[:-1] - beta_sorted[1:]) * C[:-1])
    return R


def active_set_size(beta: np.ndarray, delta: np.ndarray | float, sigma: float) -> tuple[int, np.ndarray]:
    """Number k of active assets (Theorem 2.2) and the ascending-beta sort order.

    Enforces the paper's sign convention sum(beta_i/delta_i^2) >= 0 (Eq. 2.5)
    internally; since Sigma is invariant under beta -> -beta this does not
    change the resulting active set or weights.

    Returns
    -------
    k : int, number of active assets (k = argmax{i : R_i > 0}).
    order : (p,) array of original indices sorted by (possibly sign-flipped) beta ascending.
    """
    beta = np.asarray(beta, dtype=float)
    p = beta.shape[0]
    delta = np.broadcast_to(np.asarray(delta, dtype=float), (p,))

    sign = 1.0 if float(np.sum(beta / delta ** 2)) >= 0 else -1.0
    beta_work = sign * beta

    order = np.argsort(beta_work, kind="mergesort")
    beta_sorted = beta_work[order]
    delta_sorted = delta[order]

    R = compute_R_sequence(beta_sorted, delta_sorted, sigma)
    positive = np.where(R > 0)[0]
    assert positive.size > 0, "R_1 = 1/sigma^2 > 0, so at least one asset must be active"
    k = int(positive.max()) + 1   # convert 0-indexed last-positive position to a 1-based count
    return k, order


def lomv_weights(beta: np.ndarray, delta: np.ndarray | float, sigma: float) -> tuple[np.ndarray, int]:
    """Long-only minimum variance (LOMV) weights, Theorem 2.2 parts 1-2.

    Returns the full p-vector of weights (zero for inactive assets, in the
    original asset order) and the number k of active assets.
    """
    beta = np.asarray(beta, dtype=float)
    p = beta.shape[0]
    delta_vec = np.broadcast_to(np.asarray(delta, dtype=float), (p,))

    k, order = active_set_size(beta, delta_vec, sigma)
    active_idx = order[:k]   # original indices of the k lowest-beta (sign-adjusted) assets

    w_active = gmv_weights_one_factor(beta[active_idx], sigma, delta_vec[active_idx])

    w_full = np.zeros(p, dtype=float)
    w_full[active_idx] = w_active
    assert np.all(w_full[active_idx] > 0), "all active weights must be strictly positive"
    assert abs(w_full.sum() - 1.0) < 1e-6, "LOMV weights must sum to 1"
    return w_full, k
