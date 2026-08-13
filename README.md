# Change-Point Detection for Defensive Asset Allocation

Replication code for *Change-Point Detection for Defensive Asset Allocation: A
Fixed-Baseline CUSUM Overlay* (Pinhe Chen, Fort Hays State University).

The paper proposes a cumulative-sum detector whose in-control reference
distribution is estimated once on an initial window and then frozen, rather
than re-estimated from a trailing window or reset after each alarm. It
characterizes the detector on synthetic series, benchmarks it against an
offline change-point procedure on S&P 500 returns, validates it out of sample
on six non-US equity markets, and embeds it in a two-state equity–Treasury
allocation rule.

## Data

Market data are obtained from WRDS and are **not** redistributed here, in
accordance with the terms of that licence. The analysis uses:

| Series | Source | Coverage |
|---|---|---|
| S&P 500 daily returns | WRDS | 1990–2026 |
| SPY, IEF daily returns | WRDS | 2003–2026 |
| VIX | WRDS | 2003–2026 |
| Country total-return aggregates (Japan, Germany, France, Italy, Hong Kong, Brazil) | WRDS | market-specific, 1995 or 1999 onward |

Scripts expect these under `data/` as Parquet files with a `date` index and a
`log_return` column (country aggregates: CSV with `date` and `portret`). A
reader with WRDS access can reconstruct the analysis dataset from the table
above; `src/data_loaders.py` documents the exact fields consumed.

Generated output is written to `results/`. Neither directory is tracked.

## Layout

```
src/
  algorithms/       detector implementations
    cusum_fixed.py  frozen-baseline mean detector
    cusum_abs.py    frozen-baseline variance detector
    cusum.py        adaptive detector with post-alarm baseline reset
    icss.py         Inclan-Tiao iterated cumulative sums of squares
    pelt_l2.py, pelt_rbf.py   offline segmentation, exploratory only
  simulators.py     synthetic data-generating processes
  metrics/          detection metrics
  data_loaders.py   data access layer

experiments_*.py    the experiments reported in the paper
r3c_*.py, r3d_*.py, r3e_*.py, r4_*.py, r6_*.py, r7_*.py, r8_*.py
                    inference, figures, cross-market and robustness analyses

cusum-revision/     analyses added during peer review
  src/              fixed / rolling / reset baseline variants; a replication
                    of the published backtest used as a reproduction check
  scripts/          one runner per analysis, numbered in execution order

paper/figures/      figures as they appear in the paper
```

## Running

```bash
pip install -r cusum-revision/requirements.txt
```

Python 3.9 or later, with `numpy`, `pandas`, `scipy`, `statsmodels` and
`pyarrow`.

The original experiments are run individually from the repository root, for
example:

```bash
python experiments_portfolio_v2.py
python experiments_r0_icss.py
```

The review-stage analyses begin with a reproduction check that rebuilds every
published strategy from the stored signal files and compares the resulting
Sharpe ratios against the values reported in the paper, exiting non-zero on any
discrepancy beyond 0.003:

```bash
python cusum-revision/scripts/01_validate.py
```

The remaining scripts are numbered in the order they are intended to be run and
are documented in `cusum-revision/README.md`.

## Reproducibility

All scripts are deterministic given their seed arguments. The synthetic
experiments use common random numbers across detector variants, so comparisons
between variants are paired. Default hyperparameters throughout are the values
reported in the paper: threshold 8, drift 0.5, baseline window 252
observations, cooldown 60 observations.

## License

MIT. See `cusum-revision/LICENSE`.
