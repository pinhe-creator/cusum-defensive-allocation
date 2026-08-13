# Fixed-Baseline CUSUM for Defensive Asset Allocation

Revision code for *Change-Point Detection for Defensive Asset Allocation:
A Fixed-Baseline CUSUM Overlay*.

Every published strategy is rebuilt from the existing signal files using the
same backtest formula, transaction-cost convention, and log-return definition
as the original experiments. Script `01_validate.py` compares the rebuilt
Sharpe ratios against the published ones and stops if any of them differ by
more than 0.003, so the new analyses can only run on a reproduction that
matches the paper.

## Layout

Place this repository inside the project root so that `data/` and `results/`
resolve to the existing directories:

```
cpd-finance-benchmark/
  data/         spy_daily.parquet, ief_daily.parquet, spx_daily.parquet,
                uk_daily.parquet, germany_daily.parquet, japan_daily.parquet,
                korea_daily.parquet
  results/      portfolio_v2_signals.csv, portfolio_v2_metrics.csv,
                r3b_hamilton_ms_states.csv, r3b_hamilton_ms_metrics.csv,
                r0_realdata_stage2.csv
  cusum-revision/
    src/
    scripts/
```

Run every script from the project root:

```bash
cd cpd-finance-benchmark
pip install -r cusum-revision/requirements.txt
python cusum-revision/scripts/01_validate.py
```

New outputs are written to `results/rev_*.csv` and never overwrite existing
files.

## Conventions carried over from the original experiments

| Item | Value |
|---|---|
| Risk-on weights | 0.60 SPY / 0.40 IEF |
| Risk-off weights | 0.30 SPY / 0.70 IEF |
| Returns | daily log returns |
| Transaction cost | total turnover x 10 bps, charged on the rebalance day |
| First day | cash to initial allocation, turnover 1.0 |
| Weight lag | day t uses the weight set at t-1 |
| Sharpe | mean / sd(ddof=1) x sqrt(252) |
| Detector defaults | threshold 8, drift 0.5, baseline window 252, cooldown 60 |

The risk-off holding period applied to newly constructed detectors is inferred
from the run lengths of the published state series, so new and published
strategies share the same alarm-to-allocation rule. Override it with
`--duration` if needed.

## Detectors

`src/detectors.py` provides three baseline treatments of the same Page
recursion, for both the mean detector (on returns) and the variance detector
(on absolute returns):

| Function | Baseline |
|---|---|
| `cusum_fixed` | estimated once on the first 252 observations, then frozen |
| `cusum_rolling` | re-estimated at every t from the trailing 252 observations |
| `cusum_reset` | re-estimated after each alarm and cooldown |

`cusum_reset` reproduces the adaptive detector already reported in the paper;
`cusum_rolling` is the trailing-window comparator that the paper discusses but
does not evaluate.

## Scripts

**01_validate.py** rebuilds all six published strategies and checks their
Sharpe ratios against `portfolio_v2_metrics.csv` and
`r3b_hamilton_ms_metrics.csv`. Also reports drawdown, risk-off share, alarm
onsets, and risk-off run lengths. Exits non-zero on any mismatch.

```bash
python cusum-revision/scripts/01_validate.py
```

**02_synthetic_baselines.py** compares fixed, rolling, and reset baselines on
mean-shift, variance-shift, mixed, and null scenarios, reporting detection
rate, first-alarm rate, mean delay, and false-alarm rate. The CUSUM-fixed and
CUSUM-abs rows serve as controls against the published synthetic tables; align
`--n`, `--tau`, `--sigma`, and `--seed` with the original experiment if they
do not match.

```bash
python cusum-revision/scripts/02_synthetic_baselines.py --replications 1000
```

**03_portfolio_rolling.py** appends the rolling-baseline detectors to the
published comparison table. Re-checks the reproduction before running and
aborts on mismatch. Writes the daily strategy return series used downstream.

```bash
python cusum-revision/scripts/03_portfolio_rolling.py
```

**04_inference.py** computes Jobson-Korkie statistics with the Memmel
correction and paired stationary-bootstrap confidence intervals for
differences in both Sharpe ratio and maximum drawdown.

```bash
python cusum-revision/scripts/04_inference.py --replications 10000 --all-pairs
```

**05_stability.py** tests the in-control assumption on the frozen baseline
window: ICSS breaks inside the window, split-sample tests for equality of mean
and variance, sensitivity of the baseline estimates to window length, and the
drift of a trailing baseline across the full sample.

```bash
python cusum-revision/scripts/05_stability.py
```

**06_cross_market.py** repeats the overlay on the non-US equity series,
including the UK. The bond leg defaults to IEF; pass `--bond-template` to use
local bond series.

```bash
python cusum-revision/scripts/06_cross_market.py --markets uk,germany,japan,korea
```

**07_power_curve.py** sweeps shift magnitudes and reports the magnitude at
which each detector first reaches 50%, 80%, and 90% detection power.

```bash
python cusum-revision/scripts/07_power_curve.py --replications 1000
```

**08_icss_tolerance.py** matches detector alarms to the ICSS breakpoints in
`r0_realdata_stage2.csv` across tolerance windows from 20 to 120 days, both
for all breaks and for variance increases only.

```bash
python cusum-revision/scripts/08_icss_tolerance.py
```

## Runtime

Scripts 01, 03, 05, 06, and 08 finish in seconds. Script 04 takes a few
minutes at 10000 replications. Scripts 02 and 07 dominate the total: at 1000
replications each they run on the order of tens of minutes on a single core.
All scripts are deterministic given `--seed`.

## Citation

Chen, P. Change-point detection for defensive asset allocation: A
fixed-baseline CUSUM overlay.

## License

MIT
