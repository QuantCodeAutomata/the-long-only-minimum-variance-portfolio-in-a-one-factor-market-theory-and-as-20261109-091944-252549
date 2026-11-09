"""Unit tests for src/active_ratio.py (paper Eq. 2.16, Theorem 2.5, Eq. 3.1)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm

from src.active_ratio import (
    F_discrete,
    F_normal,
    F_truncated_normal,
    G_discrete,
    G_normal,
    G_truncated_normal,
    beta_star_normal,
    finite_p_threshold,
)


def test_g_normal_matches_naive_double_integral():
    """G(y) = E[beta^2 1_{beta<=y}] - y E[beta 1_{beta<=y}] by direct numerical check."""
    mu, std, y = 1.0, 0.4, 0.3
    naive, _ = integrate.quad(
        lambda x: (x ** 2 - y * x) * norm.pdf(x, loc=mu, scale=std), mu - 15 * std, y
    )
    assert abs(G_normal(y, mu, std) - naive) < 1e-8


def test_g_normal_decreasing_near_beta_star():
    mu, std = 1.0, 0.4
    beta_star = beta_star_normal(mu, std)
    assert G_normal(beta_star - 0.05, mu, std) > 0
    assert G_normal(beta_star + 0.05, mu, std) < 0


def test_beta_star_normal_matches_paper_value():
    """Paper Section 3.1: beta* approx 0.288 for N(1, 0.4^2), F(beta*) approx 0.0376."""
    beta_star = beta_star_normal(1.0, 0.4)
    assert abs(beta_star - 0.288) < 0.01
    assert abs(F_normal(beta_star, 1.0, 0.4) - 0.0376) < 0.01


def test_g_truncated_normal_zero_at_origin():
    """Case 3 of Theorem 2.5: beta* = 0, F(beta*) = 0 for the truncated distribution."""
    mu, std = 1.0, 0.4
    assert G_truncated_normal(0.0, mu, std) == 0.0
    assert F_truncated_normal(0.0, mu, std) == 0.0
    assert G_truncated_normal(0.5, mu, std) < 0


def test_g_discrete_four_point_distribution_zero_at_two():
    """Paper Section 2.3: beta* = 2 is the unique zero of G for the four-point law."""
    values = np.array([-1.0, 1.0, 2.0, 5.0])
    probs = np.array([0.05, 0.15, 0.30, 0.50])
    assert abs(G_discrete(2.0, values, probs)) < 1e-12
    assert G_discrete(1.5, values, probs) > 0
    assert G_discrete(2.5, values, probs) < 0


def test_f_discrete_strict_vs_nonstrict_at_atom():
    """F(beta*-) = 0.20 and F(beta*) = 0.50, since beta=2 carries mass 0.30."""
    values = np.array([-1.0, 1.0, 2.0, 5.0])
    probs = np.array([0.05, 0.15, 0.30, 0.50])
    assert abs(F_discrete(2.0, values, probs, strict=True) - 0.20) < 1e-12
    assert abs(F_discrete(2.0, values, probs, strict=False) - 0.50) < 1e-12


def test_finite_p_threshold_above_beta_star_and_decreasing_in_p():
    """y_p > beta* (Eq. 3.1) and y_p -> beta* as p grows (nu^2/(p sigma^2) -> 0)."""
    mu, std = 1.0, 0.4
    sigma, delta = 0.15, 0.30
    beta_star = beta_star_normal(mu, std)
    nu2_over_sigma2 = delta ** 2 / sigma ** 2

    y_500 = finite_p_threshold(lambda y: G_normal(y, mu, std), beta_star, nu2_over_sigma2 / 500, search_hi=beta_star + 20.0)
    y_10000 = finite_p_threshold(lambda y: G_normal(y, mu, std), beta_star, nu2_over_sigma2 / 10000, search_hi=beta_star + 20.0)

    assert y_500 > beta_star
    assert y_10000 > beta_star
    assert y_10000 < y_500  # larger p => smaller finite-sample correction


def test_finite_p_threshold_rejects_oversmall_bracket():
    mu, std = 1.0, 0.4
    beta_star = beta_star_normal(mu, std)
    with pytest.raises(AssertionError):
        finite_p_threshold(lambda y: G_normal(y, mu, std), beta_star, 1e-6, search_hi=beta_star + 1e-8)
