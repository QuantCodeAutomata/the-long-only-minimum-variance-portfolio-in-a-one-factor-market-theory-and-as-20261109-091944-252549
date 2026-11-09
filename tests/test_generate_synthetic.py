"""Unit tests for data/generate_synthetic.py (synthetic beta generators, Section 3)."""

from __future__ import annotations

import numpy as np
import pytest

from data.generate_synthetic import (
    generate_four_point_betas,
    generate_normal_betas,
    generate_truncated_normal_betas,
)


def test_normal_betas_shape_and_moments():
    rng = np.random.default_rng(0)
    beta = generate_normal_betas(200000, 1.0, 0.4, rng)
    assert beta.shape == (200000,)
    assert abs(beta.mean() - 1.0) < 0.01
    assert abs(beta.std() - 0.4) < 0.01


def test_normal_betas_contain_negative_values():
    """P(beta < 0) approx 0.62% for N(1, 0.4^2), Section 3.1."""
    rng = np.random.default_rng(1)
    beta = generate_normal_betas(200000, 1.0, 0.4, rng)
    frac_negative = (beta < 0).mean()
    assert abs(frac_negative - 0.0062) < 0.003


def test_truncated_normal_betas_never_negative():
    rng = np.random.default_rng(2)
    beta = generate_truncated_normal_betas(50000, 1.0, 0.4, rng)
    assert np.all(beta >= 0.0)
    assert beta.shape == (50000,)


def test_four_point_betas_only_take_specified_values():
    rng = np.random.default_rng(3)
    values = (-1.0, 1.0, 2.0, 5.0)
    probs = (0.05, 0.15, 0.30, 0.50)
    beta = generate_four_point_betas(20000, values, probs, rng)
    assert set(np.unique(beta)) <= set(values)
    # empirical frequencies close to the specified probabilities
    for v, p in zip(values, probs):
        assert abs((beta == v).mean() - p) < 0.02


def test_four_point_betas_rejects_bad_probabilities():
    rng = np.random.default_rng(4)
    with pytest.raises(AssertionError):
        generate_four_point_betas(10, (-1.0, 1.0), (0.3, 0.3), rng)
