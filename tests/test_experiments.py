"""Reduced-scale integration tests for exp/exp1.py, exp/exp2.py, exp/exp3.py.

These run the real exp.run() entry points at small p / few trials (monkeypatched
config) to check the paper's qualitative properties end to end, without the
full-scale run used to produce results/.
"""

from __future__ import annotations

import importlib

import numpy as np

from src import config


def test_exp1_active_set_much_smaller_than_p(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "EXP1_P", 2000)
    exp1 = importlib.reload(importlib.import_module("exp.exp1"))
    monkeypatch.setattr(exp1, "RESULTS_DIR", tmp_path)
    metrics = exp1.run()
    assert metrics["p"] == 2000
    assert 0 < metrics["lomv_active_count_k"] < metrics["p"]
    assert metrics["lomv_active_ratio"] < 0.3
    assert metrics["gmv_num_nonzero"] == metrics["p"]
    assert 0.0 < metrics["gmv_frac_positive"] < 1.0
    assert (tmp_path / "exp1_metrics.json").exists()
    assert (tmp_path / "fig1_1_gmv_vs_lomv_weights.png").exists()
    importlib.reload(exp1)


def test_exp2_active_ratio_decreases_with_p(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "EXP2_P_VALUES", (200, 2000))
    monkeypatch.setattr(config, "EXP2_N_TRIALS", 20)
    exp2 = importlib.reload(importlib.import_module("exp.exp2"))
    monkeypatch.setattr(exp2, "RESULTS_DIR", tmp_path)
    results = exp2.run()
    small_mean = results["distributions"]["normal"][200]["simulated_mean"]
    large_mean = results["distributions"]["normal"][2000]["simulated_mean"]
    assert large_mean < small_mean
    assert results["F_beta_star_truncated"] == 0.0
    assert abs(results["F_beta_star_normal"] - 0.0376) < 0.01
    # truncated case has no negative betas, so it should need a smaller active set
    assert (
        results["distributions"]["truncated"][2000]["simulated_mean"]
        < results["distributions"]["normal"][2000]["simulated_mean"]
    )
    assert (tmp_path / "exp2_metrics.json").exists()
    assert (tmp_path / "exp2_table_3_1.csv").exists()
    importlib.reload(exp2)
    monkeypatch.setattr(config, "EXP2_P_VALUES", (500, 3000, 10000))
    monkeypatch.setattr(config, "EXP2_N_TRIALS", 400)


def test_exp3_bimodal_split_near_reference_lines(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "EXP3_P_VALUES", (300, 1500))
    monkeypatch.setattr(config, "EXP3_N_TRIALS", 60)
    exp3 = importlib.reload(importlib.import_module("exp.exp3"))
    monkeypatch.setattr(exp3, "RESULTS_DIR", tmp_path)
    results = exp3.run()
    assert abs(results["beta_star"] - 2.0) < 1e-9
    assert abs(results["F_beta_star_minus"] - 0.20) < 1e-9
    assert abs(results["F_beta_star"] - 0.50) < 1e-9
    for p in (300, 1500):
        row = results["p_results"][p]
        assert row["n_low_mode"] + row["n_high_mode"] == 60
        if row["mean_low_mode_ratio"] is not None:
            assert abs(row["mean_low_mode_ratio"] - 0.20) < 0.05
        if row["mean_high_mode_ratio"] is not None:
            assert abs(row["mean_high_mode_ratio"] - 0.50) < 0.05
    assert (tmp_path / "exp3_metrics.json").exists()
    assert (tmp_path / "fig3_2_non_convergence.png").exists()
    importlib.reload(exp3)
    monkeypatch.setattr(config, "EXP3_P_VALUES", (500, 3000, 10000))
    monkeypatch.setattr(config, "EXP3_N_TRIALS", 400)
