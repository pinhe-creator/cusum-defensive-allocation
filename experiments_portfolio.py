"""Stage 5: Portfolio backtest comparing CUSUM-driven risk-off to baselines.

Compares three portfolio strategies on SPY + IEF (2003-2026):
    1. Static 60/40    - daily rebalanced baseline
    2. VIX threshold   - industry-standard volatility trigger
    3. CUSUM-driven    - CPD alarm triggers risk-off

Strategy mechanics:
    Normal state:   60% SPY + 40% IEF
    Risk-off state: 30% SPY + 70% IEF
    Hold L=20 trading days after risk-off trigger, then return to normal.

Note on rebalancing: this backtest uses a weighted-return convention,
i.e., portfolio return at day t equals weight_{t-1} dot return_t.
This implicitly assumes daily rebalancing back to target weights at
the close of each day, with no cost charged for the implicit intraday
weight drift. Transaction costs are only charged when the TARGET
weights change (state transition), not for daily drift rebalancing.
All three strategies use the same convention, so cross-strategy
comparison remains fair.

Transaction costs: tested at 0, 5, 10 bps per one-way trade per asset leg.

Backtest convention: portfolio starts in cash on day 0, pays TC to
establish initial position, earns no return on day 0. From day 1
onward, return at day t is determined by weights chosen at end of
day t-1 (no look-ahead).

Outputs:
    results/portfolio_metrics.csv      - full metrics across cost grid
    results/portfolio_signals.csv      - per-day signal states for analysis
    results/portfolio_cumulative.png   - cumulative returns
    results/portfolio_drawdown.png     - drawdown curves
    results/portfolio_crisis.png       - crisis-period zoom-ins
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

from algorithms import cusum


# ====================================================================
# Configuration
# ====================================================================

W_NORMAL  = np.array([0.60, 0.40])   # SPY, IEF
W_RISKOFF = np.array([0.30, 0.70])
RISK_OFF_HOLD = 20

VIX_RISKOFF_THRESHOLD = 30.0
VIX_NORMAL_THRESHOLD  = 20.0

CUSUM_KWARGS = {
    "threshold": 8.0, "drift": 0.50,
    "window": 100, "cooldown": 60,
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


def signal_cusum(spx_returns, hold=RISK_OFF_HOLD):
    """CUSUM alarm-driven risk-off (vectorized)."""
    result = cusum.detect(spx_returns, **CUSUM_KWARGS)
    alarms = result["change_points"]
    n = len(spx_returns)
    state = np.zeros(n, dtype=int)
    for a in alarms:
        end = min(a + hold + 1, n)
        state[a:end] = 1
    return state, alarms


# ====================================================================
# Backtest engine
# ====================================================================

def backtest(spy_returns, ief_returns, state, tc_bps=0):
    """Backtest a strategy. Returns net log returns after TC.

    Convention: day 0 = setup from cash (pay TC, no return).
                day t (t>=1): return uses weight_{t-1} (chosen at end
                of t-1). Rebalance to weight_t happens at end of day t.
    """
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
    dw[0] = weights[0]                # from cash to initial position
    dw[1:] = weights[1:] - weights[:-1]
    tc_per_day = np.sum(np.abs(dw), axis=1) * tc_rate

    net_log_ret = gross_log_ret - tc_per_day
    cumulative = np.cumsum(net_log_ret)
    turnover = float(np.sum(np.abs(dw)))

    return {
        "portfolio_returns": net_log_ret,
        "weights_spy": weights[:, 0],
        "turnover": turnover,
        "cumulative": cumulative,
        "state": state,
    }


# ====================================================================
# Risk metrics
# ====================================================================

def compute_metrics(portfolio_returns, trading_days_per_year=252):
    """Portfolio risk metrics from log return series."""
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

    # Tail metrics (in-sample; expanding-window VaR is future work)
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
    """Two measures of crisis period loss:

    crisis_max_dd: largest drawdown within crisis window (vs local peak).
    loss_from_start: worst level relative to crisis START
                     (negative; directly interpretable as 'crisis impact').
    """
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
    state_static = signal_static(n)
    state_vix = signal_vix(vix)
    state_cusum, cusum_alarms = signal_cusum(spx_ret)
    cusum_alarm_dates = [dates[a].strftime("%Y-%m-%d") for a in cusum_alarms]

    print(f"Static 60/40: state always 0")
    print(f"VIX threshold: risk-off fraction = "
          f"{state_vix.mean()*100:.1f}% of days")
    print(f"CUSUM: {len(cusum_alarms)} alarms, risk-off fraction = "
          f"{state_cusum.mean()*100:.1f}% of days")
    print(f"  CUSUM alarm dates: {cusum_alarm_dates}")

    # Save daily signal states for post-hoc analysis
    signal_df = pd.DataFrame({
        "date": dates,
        "static_state": state_static,
        "vix_state": state_vix,
        "cusum_state": state_cusum,
        "vix_level": vix.values,
    })
    signal_df.to_csv("results/portfolio_signals.csv", index=False)
    print(f"  Saved daily signals: results/portfolio_signals.csv")

    strategies = {
        "Static 60/40":   state_static,
        "VIX threshold":  state_vix,
        "CUSUM":          state_cusum,
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
    df_metrics.to_csv("results/portfolio_metrics.csv", index=False)
    print(f"\nSaved metrics: results/portfolio_metrics.csv "
          f"({len(df_metrics)} rows)")

    # ---- Print tables ----
    print("\n" + "=" * 100)
    print("FULL-SAMPLE METRICS (2003-2026)")
    print("=" * 100)
    cols_main = ["strategy", "tc_bps", "ann_return", "ann_vol",
                 "sharpe", "max_drawdown", "calmar",
                 "var_95_daily", "es_95_daily", "turnover"]
    print(df_metrics[cols_main].round(4).to_string(index=False))

    print("\n" + "=" * 100)
    print("CRISIS-PERIOD LOSSES (max_dd within window, "
          "loss_from_start = worst point vs crisis start)")
    print("=" * 100)
    for crisis_name in CRISIS_WINDOWS:
        sub_cols = ["strategy", "tc_bps",
                    f"{crisis_name}_max_dd",
                    f"{crisis_name}_loss_from_start"]
        print(f"\n{crisis_name}:")
        print(df_metrics[sub_cols].round(4).to_string(index=False))

    # ---- Plots ----
    plot_tc = TC_GRID[-1]
    colors = {"Static 60/40": "black",
              "VIX threshold": "blue",
              "CUSUM": "red"}

    fig, ax = plt.subplots(figsize=(13, 5))
    for name in strategies:
        bt = backtest_results[(name, plot_tc)]
        wealth = np.exp(bt["cumulative"])
        ax.plot(dates, wealth, label=name, color=colors[name],
                lw=1.0, alpha=0.85)
    ax.set_yscale("log")
    ax.set_title(f"Cumulative portfolio value, $1 invested "
                 f"(tc={plot_tc} bps)")
    ax.set_ylabel("Wealth (log scale)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/portfolio_cumulative.png",
                dpi=130, bbox_inches="tight")
    print("\nSaved: results/portfolio_cumulative.png")

    fig, ax = plt.subplots(figsize=(13, 5))
    for name in strategies:
        bt = backtest_results[(name, plot_tc)]
        wealth = np.exp(bt["cumulative"])
        running_max = np.maximum.accumulate(wealth)
        dd = wealth / running_max - 1
        ax.plot(dates, dd * 100, label=name, color=colors[name],
                lw=0.8, alpha=0.85)
    ax.set_title(f"Portfolio drawdown (tc={plot_tc} bps)")
    ax.set_ylabel("Drawdown (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/portfolio_drawdown.png",
                dpi=130, bbox_inches="tight")
    print("Saved: results/portfolio_drawdown.png")

    fig, axes = plt.subplots(1, len(CRISIS_WINDOWS), figsize=(15, 4))
    for ax, (crisis_name, (cs, ce)) in zip(axes, CRISIS_WINDOWS.items()):
        for name in strategies:
            bt = backtest_results[(name, plot_tc)]
            wealth = np.exp(bt["cumulative"])
            mask = (dates >= pd.Timestamp(cs)) & (dates <= pd.Timestamp(ce))
            if not mask.any():
                continue
            sub_w = wealth[mask] / wealth[mask][0]
            ax.plot(dates[mask], sub_w, label=name, color=colors[name],
                    lw=1.0)
        ax.set_title(crisis_name)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.suptitle(f"Crisis wealth (normalized; tc={plot_tc} bps)")
    plt.tight_layout()
    plt.savefig("results/portfolio_crisis.png",
                dpi=130, bbox_inches="tight")
    print("Saved: results/portfolio_crisis.png")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
