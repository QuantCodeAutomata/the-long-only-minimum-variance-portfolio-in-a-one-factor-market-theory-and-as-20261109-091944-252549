# Repository guide for coding agents

This repository reproduces three numerical experiments from Kercheval and
Sowunmi, *"The Long-Only Minimum Variance Portfolio in a One-Factor Market:
Theory and Asymptotics"*: Figure 1.1 (GMV vs LOMV portfolio weights against
beta), Figure 3.1 / Table 3.1 (active-ratio convergence for Normal and
truncated-Normal beta distributions), and Figure 3.2 (non-convergence of the
active ratio with a discrete four-point beta distribution). See
`README.md` for the paper's setting and `results/RESULTS.md` for the
numerical comparison with the paper.

## Layout

- `src/` — flat library modules: `config.py` (all experiment parameters as
  module-level `UPPER_CASE` variables, no classes or dicts), `lomv.py` (GMV
  and LOMV solvers using the Woodbury identity for the one-factor covariance
  model, Theorem 2.2), `active_ratio.py` (the population function `G`,
  Theorem 2.5 asymptotic limits, and the finite-`p` threshold `y_p` of
  Eq. 3.1), `simulation.py` (the Monte Carlo loop shared by Figures 3.1 and
  3.2).
- `exp/` — one script per experiment (`exp1.py`, `exp2.py`, `exp3.py`), run
  as `python -m exp.<name>`. Each script only calls the `src/` and `data/`
  APIs; it does not reimplement any shared logic.
- `data/` — `generate_synthetic.py` draws the beta vectors used by every
  experiment (Normal, truncated Normal, four-point discrete). No market data
  is used anywhere in this repository: every experiment in the paper's
  Section 3 is defined entirely by synthetically generated factor loadings.
- `tests/` — unit tests of `src/` on small fixtures, and reduced-scale
  integration tests of each `exp/` entry point.
- `results/` — metrics (`.json`/`.csv`) and figures (`.png`) written by the
  experiment scripts, plus `results/RESULTS.md` with the numbers compared to
  the paper. Rerunning an experiment script overwrites its files in
  `results/`; after a rerun, the numbers and verdicts in `results/RESULTS.md`
  must be checked against the new files and updated by hand.

## Commands

```bash
pip install -r requirements.txt
python -m pytest                 # full test suite
python -m exp.exp1                # Figure 1.1
python -m exp.exp2                # Figure 3.1 / Table 3.1
python -m exp.exp3                # Figure 3.2
./reproduce.sh                    # tests + all three experiments
```

All three experiments already run at the paper's full configuration
(`p = 5000` for Experiment 1; `p in {500, 3000, 10000}` with 400 trials per
`p` for Experiments 2 and 3); there is no reduced or `--full` mode to choose
between.

## Conventions

- Every experiment parameter lives in `src/config.py` as a module-level
  `UPPER_CASE` variable with a comment giving its meaning, unit and source
  (a paper section, or "not specified in the paper"). Do not hardcode a
  parameter value inside `exp/` or `src/`.
- No market data is used, so there is no `data_loader.py` and no API key is
  required. `data/generate_synthetic.py` is the only place that generates the
  beta vectors used by the experiments; do not duplicate that logic in
  `src/` or `exp/`.
- Run `python -m pytest -m "not market_data"` is not needed here (no test is
  marked `market_data`); plain `python -m pytest` runs the whole suite.
- Keep deviations from the paper (random seeds, plotting choices, trial
  classification thresholds) documented only in `results/RESULTS.md`, not as
  comments scattered through the code.
