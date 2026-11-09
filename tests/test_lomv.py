"""Unit tests for src/lomv.py: GMV and LOMV solutions (paper Prop. 2.1, Thm. 2.2)."""

from __future__ import annotations

import numpy as np
import pytest

from src.lomv import active_set_size, compute_R_sequence, gmv_weights_one_factor, lomv_weights


def _direct_gmv_weights(beta: np.ndarray, sigma: float, delta) -> np.ndarray:
    """Direct (non-Woodbury) GMV solution used as a ground truth for the fast path."""
    beta = np.asarray(beta, dtype=float)
    k = beta.shape[0]
    delta_vec = np.broadcast_to(np.asarray(delta, dtype=float), (k,))
    sigma_mat = sigma ** 2 * np.outer(beta, beta) + np.diag(delta_vec ** 2)
    ones = np.ones(k)
    sigma_inv_1 = np.linalg.solve(sigma_mat, ones)
    return sigma_inv_1 / sigma_inv_1.sum()


def test_woodbury_matches_direct_inversion():
    rng = np.random.default_rng(0)
    beta = rng.normal(1.0, 0.4, size=50)
    sigma, delta = 0.15, 0.3
    w_fast = gmv_weights_one_factor(beta, sigma, delta)
    w_direct = _direct_gmv_weights(beta, sigma, delta)
    np.testing.assert_allclose(w_fast, w_direct, atol=1e-10)


def test_gmv_weights_sum_to_one():
    rng = np.random.default_rng(1)
    beta = rng.normal(1.0, 0.4, size=200)
    w = gmv_weights_one_factor(beta, 0.15, 0.3)
    assert abs(w.sum() - 1.0) < 1e-8


def test_gmv_single_asset_weight_is_one():
    w = gmv_weights_one_factor(np.array([1.3]), 0.15, 0.3)
    np.testing.assert_allclose(w, np.array([1.0]))


def test_r_sequence_matches_definition_small_case():
    beta_sorted = np.array([-0.5, 0.2, 1.0, 1.5])
    delta_sorted = np.array([0.3, 0.3, 0.3, 0.3])
    sigma = 0.15
    R = compute_R_sequence(beta_sorted, delta_sorted, sigma)
    expected = np.empty(4)
    expected[0] = 1.0 / sigma ** 2
    for i in range(1, 4):
        s = sum(
            (beta_sorted[j] / delta_sorted[j] ** 2) * (beta_sorted[j] - beta_sorted[i])
            for j in range(i)
        )
        expected[i] = 1.0 / sigma ** 2 + s
    np.testing.assert_allclose(R, expected, atol=1e-10)


def test_active_set_is_lowest_beta_assets():
    """Theorem 2.2: the active set K = {1, ..., k} is the k lowest-beta assets."""
    rng = np.random.default_rng(2)
    beta = rng.normal(1.0, 0.4, size=500)
    k, order = active_set_size(beta, 0.3, 0.15)
    active_betas = beta[order[:k]]
    inactive_betas = beta[order[k:]]
    assert active_betas.max() <= inactive_betas.min()


def test_lomv_weights_are_nonnegative_and_sum_to_one():
    rng = np.random.default_rng(3)
    beta = rng.normal(1.0, 0.4, size=1000)
    w, k = lomv_weights(beta, 0.3, 0.15)
    assert np.all(w >= 0.0)
    assert (w > 0).sum() == k
    assert abs(w.sum() - 1.0) < 1e-6


def test_lomv_active_count_much_smaller_than_p_for_normal_betas():
    """Expected outcome of Figure 1.1: active set much smaller than p = 5000."""
    rng = np.random.default_rng(20260101)
    p = 5000
    beta = rng.normal(1.0, 0.4, size=p)
    _, k = lomv_weights(beta, 0.3, 0.15)
    assert 0 < k < p
    assert k / p < 0.15


def test_active_set_invariant_to_global_sign_flip():
    """Sigma is invariant under beta -> -beta, so the active set must be unchanged."""
    rng = np.random.default_rng(4)
    beta = rng.normal(1.0, 0.4, size=300)
    k1, _ = active_set_size(beta, 0.3, 0.15)
    k2, _ = active_set_size(-beta, 0.3, 0.15)
    assert k1 == k2


def test_gmv_rejects_nonpositive_sigma():
    with pytest.raises(AssertionError):
        gmv_weights_one_factor(np.array([1.0, 2.0]), 0.0, 0.3)


def test_r1_always_positive_so_at_least_one_active_asset():
    beta = np.array([5.0, 6.0, 7.0])
    k, _ = active_set_size(beta, 0.3, 0.15)
    assert k >= 1
