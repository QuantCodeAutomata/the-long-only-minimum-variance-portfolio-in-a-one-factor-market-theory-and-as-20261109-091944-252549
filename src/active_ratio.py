"""Asymptotic active-ratio theory for the LOMV portfolio (paper Section 2.2-2.3,
Theorem 2.5, and the finite-p heuristic of Section 3.1, Eq. 3.1).

Custom implementation (paper Eq. 2.16, Theorem 2.5, Eq. 3.1): the population
function G and the finite-p threshold y_p are specific to this paper and have
no equivalent in a general portfolio-optimization or statistics library;
scipy.integrate/scipy.optimize/scipy.stats are used only as numerical
primitives (integration, root finding, and the Normal/truncated-Normal
distributions themselves).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm, truncnorm


def G_normal(y: float, mu: float, std: float) -> float:
    """G(y) = int_{-inf}^{y} (x^2 - y x) dF(x), Eq. (2.16), for F = N(mu, std^2).

    Integrated from a finite lower bound (mu - 15*std, beyond which the
    Normal density is negligible to machine precision) rather than -inf:
    `quad` with an infinite lower bound and an upper bound y far from the
    density's mass can under-resolve the peak and silently return a value
    close to zero instead of raising, which this avoids.
    """
    lower = mu - 15.0 * std
    integrand = lambda x: (x ** 2 - y * x) * norm.pdf(x, loc=mu, scale=std)
    if y <= lower:
        return 0.0
    value, _ = quad(integrand, lower, y, limit=200)
    return value


def G_truncated_normal(y: float, mu: float, std: float) -> float:
    """G(y) for F = N(mu, std^2) truncated to [0, infinity) (Section 3.1, Case 3)."""
    if y <= 0:
        return 0.0
    a = (0.0 - mu) / std
    rv = truncnorm(a, np.inf, loc=mu, scale=std)
    integrand = lambda x: (x ** 2 - y * x) * rv.pdf(x)
    value, _ = quad(integrand, 0.0, y, limit=200)
    return value


def G_discrete(y: float, values: np.ndarray, probs: np.ndarray) -> float:
    """G(y) for a discrete distribution with atoms `values` and masses `probs`."""
    values = np.asarray(values, dtype=float)
    probs = np.asarray(probs, dtype=float)
    mask = values <= y
    return float(np.sum(probs[mask] * values[mask] * (values[mask] - y)))


def F_normal(y: float, mu: float, std: float) -> float:
    return float(norm.cdf(y, loc=mu, scale=std))


def F_truncated_normal(y: float, mu: float, std: float) -> float:
    if y <= 0:
        return 0.0
    a = (0.0 - mu) / std
    return float(truncnorm.cdf(y, a, np.inf, loc=mu, scale=std))


def F_discrete(y: float, values: np.ndarray, probs: np.ndarray, strict: bool = False) -> float:
    """F(y) = P(beta <= y), or P(beta < y) if strict=True (for F(beta*-))."""
    values = np.asarray(values, dtype=float)
    probs = np.asarray(probs, dtype=float)
    mask = values < y if strict else values <= y
    return float(np.sum(probs[mask]))


def beta_star_normal(mu: float, std: float, search_hi: float | None = None) -> float:
    """Unique positive zero beta* of G for F = N(mu, std^2) (Theorem 2.5, Case 1).

    Requires P(beta < 0) > 0 and E[beta] > 0, satisfied for mu=1, std=0.4 (the
    paper's Figure 3.1 setting): P(beta<0) = Phi(-mu/std) approx 0.62% > 0.
    """
    assert mu > 0, "Case 1 of Theorem 2.5 requires E[beta] > 0"
    hi = search_hi if search_hi is not None else mu + 20.0 * std
    g0 = G_normal(0.0, mu, std)
    assert g0 > 0, "G(0) must be positive under Case 1 (some negative betas)"
    g_hi = G_normal(hi, mu, std)
    assert g_hi < 0, "search bracket too small: G(hi) must be negative"
    return brentq(lambda y: G_normal(y, mu, std), 0.0, hi, xtol=1e-10)


def finite_p_threshold(
    G_func, beta_star: float, nu2_over_p_sigma2: float, search_hi: float
) -> float:
    """Solve G(y_p) = -nu^2/(p sigma^2) for y_p > beta* (Eq. 3.1).

    `G_func` is a scalar function y -> G(y). G is strictly decreasing on
    [beta*, infinity) in Cases 1 and 3, so the root is unique.
    """
    target = -nu2_over_p_sigma2
    lo = beta_star + 1e-9
    g_lo = G_func(lo)
    assert g_lo >= target - 1e-9, "target must lie below G just above beta*"
    g_hi = G_func(search_hi)
    assert g_hi < target, "search bracket too small: G(search_hi) must be below target"
    return brentq(lambda y: G_func(y) - target, lo, search_hi, xtol=1e-10)
