"""
r3d_portfolio_panels.py

Generate two paper figures for §6 covering all six strategies, including
the Hamilton MS-AR(1) baseline that the existing portfolio_v2 figures
omit:

  results/r3d_cumulative_drawdown.pdf  -- 2-panel: log cumulative wealth
                                          (top) and percent drawdown
                                          (bottom)
  results/r3d_crisis_panel.pdf         -- 3-panel: normalized wealth in
                                          GFC, COVID, and 2022 windows

Daily portfolio returns for all six strategies are reconstructed with the
same backtest function as experiments_portfolio_v2.py at tc_bps=10. The
reconstruction logic is identical to r3c_sharpe_inference_extended.py
(verified to bit-identical Hamilton series in that script's cross-check).

Inputs:
  results/portfolio_v2_signals.csv     -- 5-strategy daily states + VIX
  results/r3b_hamilton_ms_states.csv   -- Hamilton daily_state
  data/spy_daily.parquet               -- SPY daily log_return
  data/ief_daily.parquet               -- IEF daily log_return

Run locally; <10 seconds.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SIGNALS_PATH   = "results/portfolio_v2_signals.csv"
HAMILTON_PATH  = "results/r3b_hamilton_ms_states.csv"
SPY_PARQUET    = "data/spy_daily.parquet"
IEF_PARQUET    = "data/ief_daily.parquet"
OUT_DIR        = "results"

TC_BPS = 10
W_NORMAL  = np.array([0.60, 0.40])
W_RISKOFF = np.array([0.30, 0.70])

STRATEGY_STATE_COLS = {
    "Static 60/40":    "static_state",
    "VIX threshold":   "vix_state",
    "Adaptive CUSUM":  "cusum_adaptive_state",
    "CUSUM-fixed":     "cusum_fixed_state",
    "CUSUM-abs":       "cusum_abs_state",
}
ALL_STRATEGIES = list(STRATEGY_STATE_COLS) + ["Hamilton MS-AR(1)"]

STYLE = {
    "Static 60/40":      dict(color="black",      ls="-",  lw=1.1, alpha=0.85),
    "VIX threshold":     dict(color="tab:blue",   ls="-",  lw=1.0, alpha=0.85),
    "Adaptive CUSUM":    dict(color="gray",       ls=":",  lw=1.0, alpha=0.85),
    "CUSUM-fixed":       dict(color="tab:red",    ls="-",  lw=1.2, alpha=0.90),
    "CUSUM-abs":         dict(color="darkorange", ls="-",  lw=1.0, alpha=0.85),
    "Hamilton MS-AR(1)": dict(color="tab:green",  ls="--", lw=1.1, alpha=0.85),
}

CRISIS_WINDOWS = {
    "GFC 2008--09":      ("2008-09-01", "2009-06-30"),
    "COVID 2020":        ("2020-02-01", "2020-06-30"),
    "Rate hikes 2022":   ("2022-01-01", "2022-12-31"),
}


# ---------------------------------------------------------------------------
# Loaders + backtest (mirrors r3c_sharpe_inference_extended.py)
# ---------------------------------------------------------------------------
def load_return_series(path):
    x = pd.read_parquet(path)
    if "date" in x.columns:
        x["date"] = pd.to_datetime(x["date"])
        x = x.sort_values("date").set_index("date")
    else:
        x.index = pd.to_datetime(x.index)
        x = x.sort_index()
    if "log_return" not in x.columns:
        raise ValueError(f"{path} must contain a 'log_return' column")
    return x["log_return"]


def load_dated_csv(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").set_index("date")


def backtest_returns(spy_ret, ief_ret, state, tc_bps=TC_BPS):
    n = len(spy_ret)
    weights = np.zeros((n, 2))
    weights[state == 0] = W_NORMAL
    weights[state == 1] = W_RISKOFF
    weight_prev = np.zeros((n, 2))
    weight_prev[1:] = weights[:-1]
    gross = np.sum(weight_prev * np.column_stack([spy_ret, ief_ret]), axis=1)
    dw = np.zeros((n, 2))
    dw[0] = weights[0]
    dw[1:] = weights[1:] - weights[:-1]
    tc = np.sum(np.abs(dw), axis=1) * (tc_bps / 1e4)
    return gross - tc


# ---------------------------------------------------------------------------
# Assemble all six strategies
# ---------------------------------------------------------------------------
print("Loading inputs and reconstructing 6-strategy daily returns...")
os.makedirs(OUT_DIR, exist_ok=True)

signals = load_dated_csv(SIGNALS_PATH)
ham_states = load_dated_csv(HAMILTON_PATH)
spy_ret = load_return_series(SPY_PARQUET)
ief_ret = load_return_series(IEF_PARQUET)

df = signals.copy()
df["spy_ret"]        = spy_ret.reindex(df.index)
df["ief_ret"]        = ief_ret.reindex(df.index)
df["hamilton_state"] = ham_states["daily_state"].reindex(df.index)
assert df["spy_ret"].notna().all()
assert df["ief_ret"].notna().all()
assert df["hamilton_state"].notna().all()
print(f"  Aligned {len(df)} rows from {df.index.min().date()} to "
      f"{df.index.max().date()}")

spy_arr = df["spy_ret"].to_numpy()
ief_arr = df["ief_ret"].to_numpy()

returns = {}
for label, col in STRATEGY_STATE_COLS.items():
    returns[label] = backtest_returns(
        spy_arr, ief_arr, df[col].to_numpy(dtype=int), tc_bps=TC_BPS,
    )
returns["Hamilton MS-AR(1)"] = backtest_returns(
    spy_arr, ief_arr, df["hamilton_state"].to_numpy(dtype=int), tc_bps=TC_BPS,
)

# Pre-compute wealth and drawdown per strategy
wealth = {s: np.exp(np.cumsum(returns[s])) for s in ALL_STRATEGIES}
drawdown = {s: wealth[s] / np.maximum.accumulate(wealth[s]) - 1
            for s in ALL_STRATEGIES}
dates = df.index


# ---------------------------------------------------------------------------
# Figure 1: cumulative wealth (log) + drawdown, stacked
# ---------------------------------------------------------------------------
print("\nPlotting cumulative wealth + drawdown (2 panels)...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8.5),
                                sharex=True,
                                gridspec_kw={"height_ratios": [1.4, 1.0]})

for s in ALL_STRATEGIES:
    ax1.plot(dates, wealth[s], label=s, **STYLE[s])
ax1.set_yscale("log")
ax1.set_ylabel(r"Wealth (log scale, \$1 invested)")
ax1.set_title("Cumulative portfolio value, six strategies (TC = 10 bps)")
ax1.grid(alpha=0.3, which="both")
ax1.legend(loc="upper left", ncol=2, fontsize=9, framealpha=0.9)

for s in ALL_STRATEGIES:
    ax2.plot(dates, drawdown[s] * 100, label=s, **STYLE[s])
ax2.set_ylabel("Drawdown (\\%)")
ax2.set_xlabel("Date")
ax2.set_title("Portfolio drawdown")
ax2.grid(alpha=0.3)
ax2.axhline(0, color="black", lw=0.5, alpha=0.5)
ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
out_cw = os.path.join(OUT_DIR, "r3d_cumulative_drawdown.pdf")
plt.savefig(out_cw, bbox_inches="tight")
plt.close()
print(f"  Wrote {out_cw}")


# ---------------------------------------------------------------------------
# Figure 2: 3-crisis zoom (normalized wealth)
# ---------------------------------------------------------------------------
print("\nPlotting 3-crisis zoom (normalized wealth)...")
fig, axes = plt.subplots(1, len(CRISIS_WINDOWS), figsize=(16, 4.5),
                          sharey=False)
for ax, (crisis, (cs, ce)) in zip(axes, CRISIS_WINDOWS.items()):
    mask = (dates >= pd.Timestamp(cs)) & (dates <= pd.Timestamp(ce))
    if not mask.any():
        continue
    for s in ALL_STRATEGIES:
        sub_w = wealth[s][mask] / wealth[s][mask][0]
        ax.plot(dates[mask], sub_w, label=s, **STYLE[s])
    ax.set_title(crisis)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=30)
axes[0].set_ylabel("Wealth normalized to crisis start")
axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
plt.suptitle("Crisis-period wealth, all six strategies (TC = 10 bps)",
             y=1.02)
plt.tight_layout()
out_cr = os.path.join(OUT_DIR, "r3d_crisis_panel.pdf")
plt.savefig(out_cr, bbox_inches="tight")
plt.close()
print(f"  Wrote {out_cr}")

print("\nDone. Copy these two PDFs into paper/figures/ before recompiling.")