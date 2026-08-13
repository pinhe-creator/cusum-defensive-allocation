"""Stage 5 v2: Portfolio backtest comparing 5 strategies.

Compares risk-off allocation strategies on SPY + IEF (2003-2026):
    1. Static 60/40      - daily rebalanced baseline
    2. VIX threshold     - industry-standard volatility trigger
    3. Adaptive CUSUM    - original cusum.py (post-alarm-reset baseline)
    4. CUSUM-fixed       - cusum_fixed.py, side="negative" (mean shift,
                           negative-only, frozen baseline)
    5. CUSUM-abs         - cusum_abs.py (variance shift, one-sided,
                           frozen baseline)

Strategy mechanics:
    Normal state:   60% SPY + 40% IEF
    Risk-off state: 30% SPY + 70% IEF
    Hold L=20 trading days after each alarm, then return to normal.

Transaction costs: tested at 0, 5, 10 bps per one-way trade per leg.

Backtest convention: portfolio starts in cash on day 0, pays TC to
establish initial position, earns no return on day 0. From day 1
onward, return at day t is determined by weights chosen at end of
day t-1 (no look-ahead). Weighted-return convention (target weights
applied to returns; no intraday drift cost charged).

IMPORTANT: CUSUM-fixed and CUSUM-abs estimate their frozen baselines
from the first 252 days of the ALIGNED sample (2003+), NOT from the
full SPX history. This means alarm patterns here may differ from
those reported in the algorithms' self-tests (which used SPX 1990+).

Outputs:
    results/portfolio_v2_metrics.csv      - all metrics across cost grid
    results/portfolio_v2_signals.csv      - per-day state for each strategy
    results/portfolio_v2_alarms.csv       - alarm-level data for appendix
    results/portfolio_v2_cumulative.png   - cumulative returns
    results/portfolio_v2_drawdown.png     - drawdown curves
    results/portfolio_v2_crisis.png       - crisis zooms
"""
import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from algorithms import cusum, cusum_fixed, cusum_abs


# ====================================================================
# Configuration
# ====================================================================

W_NORMAL  = np.array([0.60, 0.40])   # SPY, IEF
W_RISKOFF = np.array([0.30, 0.70])
RISK_OFF_HOLD = 20

VIX_RISKOFF_THRESHOLD = 30.0
VIX_NORMAL_THRESHOLD  = 20.0

# Adaptive CUSUM (original, with post-alarm baseline reset)
CUSUM_ADAPTIVE_KWARGS = {
    "threshold": 8.0, "drift": 0.50,
    "window": 100, "cooldown": 60,
}

# Fixed-baseline CUSUM, negative-only for portfolio risk-off
CUSUM_FIXED_KWARGS = {
    "threshold": 8.0, "drift": 0.50,
    "baseline_window": 252, "cooldown": 60,
    "side": "negative",
}

# Absolute-return CUSUM (variance-target, one-sided)
CUSUM_ABS_KWARGS = {
    "threshold": 8.0, "drift": 0.50,
    "baseline_window": 252, "cooldown": 60,
}

TC_GRID = [0, 5, 10]

CRISIS_WINDOWS = {
    "GFC 2008-2009":   ("2008-09-01", "2009-06-30"),
    "COVID 2020":      ("2020-02-01", "2020-06-30"),
    "Rate hikes 2022": ("2022-01-01", "2022-12-31"),
}


# ====================================================================
# Data loading
# ====================================================================

def load_aligned_data():
    """Load SPY, IEF, VIX, SPX and align on common dates."""
    spy = pd.read_parquet("data/spy_daily.parquet")
    ief = pd.read_parquet("data/ief_daily.parquet")
    vix = pd.read_parquet("data/vix_daily.parquet")
    spx = pd.read_parquet("data/spx_daily.parquet")

    spy = spy.rename(columns={"close": "spy_close",
                              "log_return": "spy_ret"})
    ief = ief.rename(columns={"close": "ief_close",
                              "log_return": "ief_ret"})
    spx = spx.rename(columns={"close": "spx_close",
                              "log_return": "spx_ret"})

    df = (spy[["spy_close", "spy_ret"]]
          .join(ief[["ief_close", "ief_ret"]], how="inner")
          .join(vix[["vix"]], how="inner")
          .join(spx[["spx_ret"]], how="inner")
          .dropna())

    print(f"Aligned data: {len(df)} obs from "
          f"{df.index.min().date()} to {df.index.max().date()}")
    return df


# ====================================================================
# Signal generators
# ====================================================================

def signal_static(n):
    return np.zeros(n, dtype=int)


def signal_vix(vix_series):
    """VIX hysteresis: enter risk-off when VIX>30, exit when VIX<20."""
    n = len(vix_series)
    state = np.zeros(n, dtype=int)
    current = 0
    for t in range(n):
        v = vix_series.iloc[t]
        if current == 0 and v > VIX_RISKOFF_THRESHOLD:
            current = 1
        elif current == 1 and v < VIX_NORMAL_THRESHOLD:
            current = 0
        state[t] = current
    return state


def state_from_alarms(alarms, n, hold=RISK_OFF_HOLD):
    """Convert alarm indices to a 0/1 state series with hold period."""
    state = np.zeros(n, dtype=int)
    for a in alarms:
        end = min(a + hold + 1, n)
        state[a:end] = 1
    return state


def signal_cusum_adaptive(spx_returns):
    """Original CUSUM (post-alarm-reset baseline)."""
    result = cusum.detect(spx_returns, **CUSUM_ADAPTIVE_KWARGS)
    alarms = result["change_points"]
    state = state_from_alarms(alarms, len(spx_returns))
    return state, alarms, result


def signal_cusum_fixed(spx_returns):
    """Fixed-baseline CUSUM, negative-only."""
    result = cusum_fixed.detect(spx_returns, **CUSUM_FIXED_KWARGS)
    alarms = result["change_points"]
    state = state_from_alarms(alarms, len(spx_returns))
    return state, alarms, result


def signal_cusum_abs(spx_returns):
    """Absolute-return CUSUM (variance-target)."""
    result = cusum_abs.detect(spx_returns, **CUSUM_ABS_KWARGS)
    alarms = result["change_points"]
    state = state_from_alarms(alarms, len(spx_returns))
    return state, alarms, result


# ====================================================================
# Backtest engine
# ====================================================================

def backtest(spy_returns, ief_returns, state, tc_bps=0):
    """Backtest a strategy. Returns net log returns after TC."""
    n = len(spy_returns)
    weights = np.zeros((n, 2))
    weights[state == 0] = W_NORMAL
    weights[state == 1] = W_RISKOFF

    tc_rate = tc_bps / 1e4

    weight_prev = np.zeros((n, 2))
    weight_prev[1:] = weights[:-1]
    # weight_prev[0] stays zero (cash on day 0)

    asset_returns = np.column_stack([spy_returns, ief_returns])
    gross_log_ret = np.sum(weight_prev * asset_returns, axis=1)

    dw = np.zeros((n, 2))
    dw[0] = weights[0]
    dw[1:] = weights[1:] - weights[:-1]
    tc_per_day = np.sum(np.abs(dw), axis=1) * tc_rate

    net_log_ret = gross_log_ret - tc_per_day
    cumulative = np.cumsum(net_log_ret)
    turnover = float(np.sum(np.abs(dw)))

    return {
        "portfolio_returns": net_log_ret,
        "turnover": turnover,
        "cumulative": cumulative,
        "state": state,
    }


# ====================================================================
# Risk metrics
# ====================================================================

def compute_metrics(portfolio_returns, trading_days_per_year=252):
    n = len(portfolio_returns)
    years = n / trading_days_per_year

    wealth = np.exp(np.cumsum(portfolio_returns))
    running_max = np.maximum.accumulate(wealth)
    drawdown = wealth / running_max - 1
    max_drawdown = float(drawdown.min())

    ann_return = portfolio_returns.mean() * trading_days_per_year
    ann_vol = portfolio_returns.std() * np.sqrt(trading_days_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    calmar = ann_return / abs(max_drawdown) if max_drawdown < 0 else np.nan

    var_95 = float(np.quantile(portfolio_returns, 0.05))
    var_99 = float(np.quantile(portfolio_returns, 0.01))
    es_95 = float(portfolio_returns[portfolio_returns < var_95].mean())
    es_99 = float(portfolio_returns[portfolio_returns < var_99].mean())

    return {
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "calmar": float(calmar) if not np.isnan(calmar) else np.nan,
        "var_95_daily": var_95,
        "var_99_daily": var_99,
        "es_95_daily": es_95,
        "es_99_daily": es_99,
        "n_obs": int(n),
        "years": float(years),
    }


def crisis_subsample_loss(portfolio_returns, dates, start, end):
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    if not mask.any():
        return {"crisis_max_dd": np.nan, "loss_from_start": np.nan}
    sub_log = portfolio_returns[mask]
    cum = np.cumsum(sub_log)
    wealth = np.exp(cum)
    running_max = np.maximum.accumulate(wealth)
    crisis_max_dd = float((wealth / running_max - 1).min())
    sub_wealth_norm = wealth / wealth[0]
    loss_from_start = float((sub_wealth_norm - 1).min())
    return {"crisis_max_dd": crisis_max_dd,
            "loss_from_start": loss_from_start}


# ====================================================================
# Main
# ====================================================================

def main():
    os.makedirs("results", exist_ok=True)
    t_start = time.time()

    df = load_aligned_data()
    n = len(df)
    spy_ret = df["spy_ret"].values
    ief_ret = df["ief_ret"].values
    vix = df["vix"]
    spx_ret = df["spx_ret"].values
    dates = df.index

    print()
    print("Generating signals...")

    # Document the fixed-baseline window(s) so paper appendix can
    # report them. CUSUM-fixed and CUSUM-abs use the first N days of
    # the aligned sample (2003+) to estimate their frozen baselines.
    # Note: this is NOT the same window used in their original
    # self-tests (1990+ SPX), so alarm dates may differ.
    bw_fixed = CUSUM_FIXED_KWARGS["baseline_window"]
    bw_abs = CUSUM_ABS_KWARGS["baseline_window"]
    if bw_fixed == bw_abs:
        bw = bw_fixed
        print(f"  Fixed baseline window: "
              f"{dates[0].date()} to {dates[bw-1].date()} "
              f"(first {bw} trading days of aligned sample)")
    else:
        print(f"  CUSUM-fixed baseline window: "
              f"{dates[0].date()} to {dates[bw_fixed-1].date()} "
              f"(first {bw_fixed} days)")
        print(f"  CUSUM-abs baseline window:   "
              f"{dates[0].date()} to {dates[bw_abs-1].date()} "
              f"(first {bw_abs} days)")

    state_static = signal_static(n)
    state_vix = signal_vix(vix)
    state_cusum_a, alarms_a, result_a = signal_cusum_adaptive(spx_ret)
    state_cusum_f, alarms_f, result_f = signal_cusum_fixed(spx_ret)
    state_cusum_x, alarms_x, result_x = signal_cusum_abs(spx_ret)

    # Report estimated baseline values from fixed-baseline detectors
    print(f"  CUSUM-fixed baseline: "
          f"mu={result_f['baseline']['mu']:.6f} "
          f"({result_f['baseline']['mu']*100:.3f}% per day), "
          f"sigma={result_f['baseline']['sigma']:.6f}")
    print(f"  CUSUM-abs baseline:   "
          f"mu_abs={result_x['baseline']['mu_abs']:.4f} "
          f"({result_x['baseline']['mu_abs']*100:.2f}% per day), "
          f"sigma_abs={result_x['baseline']['sigma_abs']:.4f}")

    alarm_dates_a = [dates[a].strftime("%Y-%m-%d") for a in alarms_a]
    alarm_dates_f = [dates[a].strftime("%Y-%m-%d") for a in alarms_f]
    alarm_dates_x = [dates[a].strftime("%Y-%m-%d") for a in alarms_x]

    print()
    print(f"  Static 60/40:     state always 0")
    print(f"  VIX threshold:    risk-off frac = "
          f"{state_vix.mean()*100:.1f}%")
    print(f"  Adaptive CUSUM:   {len(alarms_a)} alarms, risk-off frac = "
          f"{state_cusum_a.mean()*100:.1f}%")
    print(f"    dates: {alarm_dates_a}")
    print(f"  CUSUM-fixed:      {len(alarms_f)} alarms, risk-off frac = "
          f"{state_cusum_f.mean()*100:.1f}%")
    print(f"    dates: {alarm_dates_f}")
    print(f"  CUSUM-abs:        {len(alarms_x)} alarms, risk-off frac = "
          f"{state_cusum_x.mean()*100:.1f}%")
    print(f"    dates: {alarm_dates_x}")

    # Save daily signals
    signal_df = pd.DataFrame({
        "date": dates,
        "static_state": state_static,
        "vix_state": state_vix,
        "vix_level": vix.values,
        "cusum_adaptive_state": state_cusum_a,
        "cusum_fixed_state": state_cusum_f,
        "cusum_abs_state": state_cusum_x,
    })
    signal_df.to_csv("results/portfolio_v2_signals.csv", index=False)
    print(f"\nSaved daily signals: results/portfolio_v2_signals.csv")

    # Save alarm-level data for paper appendix (enriched)
    alarm_rows = []
    for name, alarm_list, result in [
        ("Adaptive CUSUM", alarms_a, result_a),
        ("CUSUM-fixed", alarms_f, result_f),
        ("CUSUM-abs", alarms_x, result_x),
    ]:
        directions = result.get("directions", [None] * len(alarm_list))
        metadata = result.get("metadata", {})
        for i, a in enumerate(alarm_list):
            alarm_rows.append({
                "strategy": name,
                "alarm_index": int(a),
                "alarm_date": dates[a].strftime("%Y-%m-%d"),
                "direction": (directions[i]
                              if i < len(directions) else None),
                "target_change": metadata.get("target_change", ""),
                "algorithm_type": metadata.get("algorithm_type", ""),
            })
    pd.DataFrame(alarm_rows).to_csv(
        "results/portfolio_v2_alarms.csv", index=False
    )
    print(f"Saved alarms: results/portfolio_v2_alarms.csv "
          f"({len(alarm_rows)} rows)")

    strategies = {
        "Static 60/40":     state_static,
        "VIX threshold":    state_vix,
        "Adaptive CUSUM":   state_cusum_a,
        "CUSUM-fixed":      state_cusum_f,
        "CUSUM-abs":        state_cusum_x,
    }

    rows = []
    backtest_results = {}
    for tc_bps in TC_GRID:
        for name, state in strategies.items():
            bt = backtest(spy_ret, ief_ret, state, tc_bps=tc_bps)
            metrics = compute_metrics(bt["portfolio_returns"])
            metrics["strategy"] = name
            metrics["tc_bps"] = tc_bps
            metrics["turnover"] = bt["turnover"]
            metrics["riskoff_fraction"] = float(state.mean())
            for crisis_name, (cs, ce) in CRISIS_WINDOWS.items():
                cl = crisis_subsample_loss(
                    bt["portfolio_returns"], dates, cs, ce
                )
                metrics[f"{crisis_name}_max_dd"] = cl["crisis_max_dd"]
                metrics[f"{crisis_name}_loss_from_start"] = (
                    cl["loss_from_start"]
                )
            rows.append(metrics)
            backtest_results[(name, tc_bps)] = bt

    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv("results/portfolio_v2_metrics.csv", index=False)
    print(f"Saved metrics: results/portfolio_v2_metrics.csv "
          f"({len(df_metrics)} rows)")

    # ---- Print tables ----
    print("\n" + "=" * 110)
    print("FULL-SAMPLE METRICS (2003-2026)")
    print("=" * 110)
    cols_main = ["strategy", "tc_bps", "ann_return", "ann_vol",
                 "sharpe", "max_drawdown", "calmar",
                 "var_95_daily", "es_95_daily",
                 "riskoff_fraction", "turnover"]
    print(df_metrics[cols_main].round(4).to_string(index=False))

    print("\n" + "=" * 110)
    print("CRISIS-PERIOD LOSSES")
    print("=" * 110)
    for crisis_name in CRISIS_WINDOWS:
        sub_cols = ["strategy", "tc_bps",
                    f"{crisis_name}_max_dd",
                    f"{crisis_name}_loss_from_start"]
        print(f"\n{crisis_name}:")
        print(df_metrics[sub_cols].round(4).to_string(index=False))

    # ---- Plots ----
    plot_tc = TC_GRID[-1]
    colors = {"Static 60/40":   "black",
              "VIX threshold":  "blue",
              "Adaptive CUSUM": "gray",
              "CUSUM-fixed":    "red",
              "CUSUM-abs":      "darkorange"}
    linestyles = {"Static 60/40":   "-",
                  "VIX threshold":  "-",
                  "Adaptive CUSUM": ":",
                  "CUSUM-fixed":    "-",
                  "CUSUM-abs":      "-"}

    # Cumulative
    fig, ax = plt.subplots(figsize=(13, 6))
    for name in strategies:
        bt = backtest_results[(name, plot_tc)]
        wealth = np.exp(bt["cumulative"])
        ax.plot(dates, wealth, label=name, color=colors[name],
                linestyle=linestyles[name], lw=1.1, alpha=0.85)
    ax.set_yscale("log")
    ax.set_title(f"Cumulative portfolio value, $1 invested "
                 f"(tc={plot_tc} bps)")
    ax.set_ylabel("Wealth (log scale)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/portfolio_v2_cumulative.png",
                dpi=130, bbox_inches="tight")
    print("\nSaved: results/portfolio_v2_cumulative.png")

    # Drawdown
    fig, ax = plt.subplots(figsize=(13, 6))
    for name in strategies:
        bt = backtest_results[(name, plot_tc)]
        wealth = np.exp(bt["cumulative"])
        running_max = np.maximum.accumulate(wealth)
        dd = wealth / running_max - 1
        ax.plot(dates, dd * 100, label=name, color=colors[name],
                linestyle=linestyles[name], lw=0.9, alpha=0.85)
    ax.set_title(f"Portfolio drawdown (tc={plot_tc} bps)")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/portfolio_v2_drawdown.png",
                dpi=130, bbox_inches="tight")
    print("Saved: results/portfolio_v2_drawdown.png")

    # Crisis zoom
    fig, axes = plt.subplots(1, len(CRISIS_WINDOWS), figsize=(16, 4.5))
    for ax, (crisis_name, (cs, ce)) in zip(axes, CRISIS_WINDOWS.items()):
        for name in strategies:
            bt = backtest_results[(name, plot_tc)]
            wealth = np.exp(bt["cumulative"])
            mask = (dates >= pd.Timestamp(cs)) & (dates <= pd.Timestamp(ce))
            if not mask.any():
                continue
            sub_w = wealth[mask] / wealth[mask][0]
            ax.plot(dates[mask], sub_w, label=name, color=colors[name],
                    linestyle=linestyles[name], lw=1.0)
        ax.set_title(crisis_name)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")
    plt.suptitle(f"Crisis wealth (normalized; tc={plot_tc} bps)")
    plt.tight_layout()
    plt.savefig("results/portfolio_v2_crisis.png",
                dpi=130, bbox_inches="tight")
    print("Saved: results/portfolio_v2_crisis.png")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
