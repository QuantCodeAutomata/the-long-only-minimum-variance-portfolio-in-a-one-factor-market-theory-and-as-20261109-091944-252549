"""Unit tests for src/simulation.py (Monte Carlo active-ratio driver, Section 3)."""

from __future__ import annotations

import numpy as np

from data.generate_synthetic import generate_normal_betas
from src.simulation import simulate_active_ratios


def test_active_ratios_within_unit_interval():
    rng = np.random.default_rng(0)
    ratios = simulate_active_ratios(
        p=50, n_trials=20,
        draw_betas=lambda n, r: generate_normal_betas(n, 1.0, 0.4, r),
        sigma=0.15, delta=0.30, rng=rng,
    )
    assert ratios.shape == (20,)
    assert np.all(ratios > 0.0)
    assert np.all(ratios <= 1.0)


def test_active_ratios_reproducible_with_fixed_seed():
    draw = lambda n, r: generate_normal_betas(n, 1.0, 0.4, r)
    ratios_a = simulate_active_ratios(60, 10, draw, 0.15, 0.30, np.random.default_rng(42))
    ratios_b = simulate_active_ratios(60, 10, draw, 0.15, 0.30, np.random.default_rng(42))
    np.testing.assert_allclose(ratios_a, ratios_b)


def test_larger_p_gives_smaller_mean_active_ratio_for_normal_betas():
    """Expected outcome of Figure 3.1: active ratio decreases toward F(beta*) as p grows."""
    draw = lambda n, r: generate_normal_betas(n, 1.0, 0.4, r)
    rng = np.random.default_rng(7)
    small = simulate_active_ratios(300, 60, draw, 0.15, 0.30, rng).mean()
    large = simulate_active_ratios(4000, 60, draw, 0.15, 0.30, rng).mean()
    assert large < small
