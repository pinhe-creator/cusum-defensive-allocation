"""Robustness check R1: CUSUM-fixed alarm pattern and portfolio
performance across different baseline_window values.

Tests whether the paper finding "CUSUM-fixed achieves highest Sharpe"
depends critically on the choice baseline_window=252.

Baseline-window grid design:
    Main:       [126, 252, 504]  - 6 months, 1 year, 2 years
                These windows estimate baseline from 2003-Q1/Q2/Q3
                only (no crisis contamination).
    Diagnostic: [750, 1000]      - 3 years, 4 years
                These windows estimate baseline from 2003-2007 data,
                potentially including pre-GFC vol clusters.

We deliberately exclude baseline_window >= 1500 because such windows
would estimate the baseline from data extending into the GFC period
itself (Lehman 2008-09 occurs ~1450 trading days into our 2003+ sample).
A baseline estimated on partial-crisis data would be biased upward in
sigma and contaminate the detection benchmark.

For each baseline_window, we report:
    - Number of alarms detected on SPX 1990-2026 (self-test analog)
    - Number of alarms in portfolio sample (2003-2026)
    - Activation date (the first day the detector can produce alarms)
    - Whether COVID 2020-02 alarm is retained (paper-critical)
    - Estimated (mu, sigma) baseline values
    - Portfolio Sharpe / max_dd / Calmar at tc=10 bps

Outputs:
    results/robustness_baseline_window.csv
"""
import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from algorithms import cusum_fixed


# ====================================================================
# Configuration
# ====================================================================

# Main robustness windows. 750/1000 are diagnostic only because they
# delay detector activation and may include early pre-GFC vol clusters
# in the frozen baseline. baseline_window >= 1500 is deliberately
# excluded - those would contaminate the baseline with GFC itself.
MAIN_WINDOWS = [126, 252, 504]
DIAGNOSTIC_WINDOWS = [750, 1000]
BASELINE_WINDOWS = MAIN_WINDOWS + DIAGNOSTIC_WINDOWS

THRESHOLD = 8.0
DRIFT = 0.50
COOLDOWN = 60
SIDE = "negative"
HOLD = 20
TC_BPS = 10

W_NORMAL  = np.array([0.60, 0.40])
W_RISKOFF = np.array([0.30, 0.70])

COVID_WINDOW = (pd.Timestamp("2020-02-01"),
                pd.Timestamp("2020-03-31"))
GFC_WINDOW = (pd.Timestamp("2008-09-01"),
              pd.Timestamp("2008-12-31"))
RATEHIKE_WINDOW = (pd.Timestamp("2022-01-01"),
                   pd.Timestamp("2022-12-31"))


# ====================================================================
# Data loading
# ====================================================================

def load_data():
    spy = pd.read_parquet("data/spy_daily.parquet").rename(
        columns={"close": "spy_close", "log_return": "spy_ret"})
    ief = pd.read_parquet("data/ief_daily.parquet").rename(
        columns={"close": "ief_close", "log_return": "ief_ret"})
    spx = pd.read_parquet("data/spx_daily.parquet").rename(
        columns={"close": "spx_close", "log_return": "spx_ret"})

    spx_full = spx[["spx_ret"]].dropna()

    df_port = (spy[["spy_close", "spy_ret"]]
               .join(ief[["ief_close", "ief_ret"]], how="inner")
               .join(spx[["spx_ret"]], how="inner")
               .dropna())

    print(f"Full SPX: {len(spx_full)} obs "
          f"({spx_full.index.min().date()} to "
          f"{spx_full.index.max().date()})")
    print(f"Portfolio sample: {len(df_port)} obs "
          f"({df_port.index.min().date()} to "
          f"{df_port.index.max().date()})")
    return spx_full, df_port


def load_benchmarks():
    """Read benchmark metrics from existing portfolio_v2 results."""
    try:
        df = pd.read_csv("results/portfolio_v2_metrics.csv")
        df = df[df["tc_bps"] == TC_BPS]
        out = {}
        for _, row in df.iterrows():
            out[row["strategy"]] = {
                "sharpe": row["sharpe"],
                "max_dd": row["max_drawdown"],
                "ann_return": row["ann_return"],
                "calmar": row["calmar"],
            }
        return out
    except Exception as e:
        print(f"  (note: could not load portfolio_v2_metrics.csv: {e})")
        return {}


# ====================================================================
# Portfolio backtest (minimal version)
# ====================================================================

def state_from_alarms(alarms, n, hold=HOLD):
    state = np.zeros(n, dtype=int)
    for a in alarms:
        end = min(a + hold + 1, n)
        state[a:end] = 1
    return state


def backtest_simple(spy_ret, ief_ret, state, tc_bps=TC_BPS):
    n = len(spy_ret)
    weights = np.zeros((n, 2))
    weights[state == 0] = W_NORMAL
    weights[state == 1] = W_RISKOFF
    tc_rate = tc_bps / 1e4

    weight_prev = np.zeros((n, 2))
    weight_prev[1:] = weights[:-1]

    asset_returns = np.column_stack([spy_ret, ief_ret])
    gross_log_ret = np.sum(weight_prev * asset_returns, axis=1)

    dw = np.zeros((n, 2))
    dw[0] = weights[0]
    dw[1:] = weights[1:] - weights[:-1]
    tc_per_day = np.sum(np.abs(dw), axis=1) * tc_rate

    return gross_log_ret - tc_per_day


def crisis_loss_normalized(returns, dates, start, end):
    """Worst normalized wealth loss within crisis window.

    Wealth is normalized to 1.0 at the first trading day of the
    window (not the day before). Returns a negative number = worst
    relative loss within the window.
    """
    mask = (dates >= start) & (dates <= end)
    if not mask.any():
        return np.nan
    sub_log = returns[mask]
    cum = np.cumsum(sub_log)
    sub_wealth = np.exp(cum) / np.exp(cum[0])
    return float((sub_wealth - 1).min())


def portfolio_metrics(returns, dates):
    n = len(returns)
    ann_return = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    wealth = np.exp(np.cumsum(returns))
    running_max = np.maximum.accumulate(wealth)
    drawdown = wealth / running_max - 1
    max_dd = float(drawdown.min())
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan

    return {
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_dd": max_dd,
        "calmar": float(calmar) if not np.isnan(calmar) else np.nan,
        "gfc_loss_normalized": crisis_loss_normalized(
            returns, dates, *GFC_WINDOW),
        "covid_loss_normalized": crisis_loss_normalized(
            returns, dates, *COVID_WINDOW),
        "ratehike_loss_normalized": crisis_loss_normalized(
            returns, dates, *RATEHIKE_WINDOW),
    }


# ====================================================================
# Main
# ====================================================================

def main():
    os.makedirs("results", exist_ok=True)
    t_start = time.time()

    spx_full, df_port = load_data()

    spx_ret_full = spx_full["spx_ret"].values
    dates_full = spx_full.index

    spy_ret = df_port["spy_ret"].values
    ief_ret = df_port["ief_ret"].values
    spx_ret_port = df_port["spx_ret"].values
    dates_port = df_port.index

    print()
    print("=" * 90)
    print("R1: CUSUM-fixed robustness across baseline_window")
    print("=" * 90)
    print(f"Main windows (no crisis contamination): {MAIN_WINDOWS}")
    print(f"Diagnostic windows (longer; check downstream): "
          f"{DIAGNOSTIC_WINDOWS}")
    print(f"Excluded: bw >= 1500 (would contaminate baseline with GFC)")
    print()
    print(f"Fixed params: threshold={THRESHOLD}, drift={DRIFT}, "
          f"cooldown={COOLDOWN}, side='{SIDE}'")
    print(f"Portfolio: HOLD={HOLD}, tc_bps={TC_BPS}")
    print()

    rows = []
    for bw in BASELINE_WINDOWS:
        tier = "main" if bw in MAIN_WINDOWS else "diagnostic"

        # Full SPX detection
        if len(spx_ret_full) <= bw:
            print(f"Skipping bw={bw} on full SPX (insufficient data)")
            continue
        result_full = cusum_fixed.detect(
            spx_ret_full,
            threshold=THRESHOLD, drift=DRIFT,
            baseline_window=bw, cooldown=COOLDOWN, side=SIDE,
        )
        alarms_full = result_full["change_points"]
        dates_full_alarms = [dates_full[a].strftime("%Y-%m-%d")
                             for a in alarms_full]
        baseline_full = result_full["baseline"]
        activation_full = dates_full[bw].strftime("%Y-%m-%d")
        covid_alarm_full = any(
            COVID_WINDOW[0] <= dates_full[a] <= COVID_WINDOW[1]
            for a in alarms_full
        )

        # Portfolio sample detection + backtest
        if len(spx_ret_port) <= bw:
            print(f"  bw={bw}: portfolio sample too small "
                  f"(n={len(spx_ret_port)}), skipping portfolio metrics")
            metrics = {k: np.nan for k in [
                "ann_return", "ann_vol", "sharpe", "max_dd", "calmar",
                "gfc_loss_normalized", "covid_loss_normalized",
                "ratehike_loss_normalized",
            ]}
            n_alarms_port = np.nan
            covid_alarm_port = np.nan
            activation_port = None
            dates_port_alarms = []
            baseline_port_str = "n/a"
        else:
            result_port = cusum_fixed.detect(
                spx_ret_port,
                threshold=THRESHOLD, drift=DRIFT,
                baseline_window=bw, cooldown=COOLDOWN, side=SIDE,
            )
            alarms_port = result_port["change_points"]
            n_alarms_port = len(alarms_port)
            activation_port = dates_port[bw].strftime("%Y-%m-%d")
            covid_alarm_port = any(
                COVID_WINDOW[0] <= dates_port[a] <= COVID_WINDOW[1]
                for a in alarms_port
            )
            baseline_port = result_port["baseline"]
            baseline_port_str = (f"mu={baseline_port['mu']:.6f}, "
                                 f"sigma={baseline_port['sigma']:.6f}")
            dates_port_alarms = [dates_port[a].strftime("%Y-%m-%d")
                                 for a in alarms_port]

            state = state_from_alarms(alarms_port, len(spx_ret_port))
            returns = backtest_simple(spy_ret, ief_ret, state)
            metrics = portfolio_metrics(returns, dates_port)

        row = {
            "baseline_window": bw,
            "tier": tier,
            "activation_date_full": activation_full,
            "activation_date_portfolio": activation_port,
            "n_alarms_full_spx": len(alarms_full),
            "covid_alarm_full": covid_alarm_full,
            "baseline_mu_full": baseline_full["mu"],
            "baseline_sigma_full": baseline_full["sigma"],
            "alarm_dates_full_spx": ";".join(dates_full_alarms),
            "n_alarms_portfolio": n_alarms_port,
            "covid_alarm_portfolio": covid_alarm_port,
            "alarm_dates_portfolio": ";".join(dates_port_alarms),
            **metrics,
        }
        rows.append(row)

        print(f"baseline_window = {bw}  [{tier}]")
        print(f"  Activation date: full SPX = {activation_full}, "
              f"portfolio = {activation_port}")
        print(f"  Full SPX: {len(alarms_full)} alarms, "
              f"COVID retained: {covid_alarm_full}")
        print(f"    baseline: mu={baseline_full['mu']:.6f} "
              f"({baseline_full['mu']*100:.3f}% per day), "
              f"sigma={baseline_full['sigma']:.6f}")
        print(f"    dates: {dates_full_alarms}")
        if not np.isnan(n_alarms_port):
            print(f"  Portfolio sample: {n_alarms_port} alarms, "
                  f"COVID retained: {covid_alarm_port}")
            print(f"    baseline (port): {baseline_port_str}")
            print(f"    dates: {dates_port_alarms}")
            print(f"    Sharpe={metrics['sharpe']:.3f}, "
                  f"max_dd={metrics['max_dd']*100:.1f}%, "
                  f"calmar={metrics['calmar']:.3f}, "
                  f"ann_ret={metrics['ann_return']*100:.2f}%")
            print(f"    crisis losses (normalized): "
                  f"GFC={metrics['gfc_loss_normalized']*100:.1f}%, "
                  f"COVID={metrics['covid_loss_normalized']*100:.1f}%, "
                  f"2022={metrics['ratehike_loss_normalized']*100:.1f}%")
        print()

    df_r1 = pd.DataFrame(rows)
    df_r1.to_csv("results/robustness_baseline_window.csv", index=False)
    print(f"Saved: results/robustness_baseline_window.csv "
          f"({len(df_r1)} rows)")

    # Compact summary
    print()
    print("=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    cols = ["baseline_window", "tier", "activation_date_portfolio",
            "n_alarms_portfolio", "covid_alarm_portfolio",
            "sharpe", "max_dd", "calmar", "ann_return"]
    print(df_r1[cols].round(4).to_string(index=False))

    # Benchmark reference (read from portfolio_v2 results, not hardcoded)
    print()
    print(f"Reference benchmarks (from portfolio_v2_metrics, tc={TC_BPS}):")
    bench = load_benchmarks()
    if bench:
        for name in ["Static 60/40", "VIX threshold",
                     "Adaptive CUSUM", "CUSUM-fixed", "CUSUM-abs"]:
            if name in bench:
                b = bench[name]
                print(f"  {name:<18s} Sharpe={b['sharpe']:.3f}, "
                      f"max_dd={b['max_dd']*100:.2f}%, "
                      f"Calmar={b['calmar']:.3f}, "
                      f"ann_ret={b['ann_return']*100:.2f}%")
    else:
        print("  (run experiments_portfolio_v2.py first)")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
