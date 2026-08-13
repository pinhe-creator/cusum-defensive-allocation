"""
r3e_alarm_timeline.py

Generate a single-panel paper figure visualizing when the fixed-baseline
CUSUM detectors fire on the S&P 500 over the portfolio sample
(2003-01-03 to 2026-05-22). Bridges section 5 (detection on real data)
and section 6 (portfolio application) by showing the sparse, identifiable
events that drive the defensive allocation rule.

Inputs:
  results/portfolio_v2_signals.csv   -- daily strategy states
  data/spx_daily.parquet             -- SPX with 'close' and 'log_return'

Output:
  results/r3e_alarm_timeline.pdf

Alarms are derived from state transitions 0 -> 1 in the strategy state
columns (matching the backtest's definition of an alarm event).
Run locally; <5 seconds.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SIGNALS_PATH = "results/portfolio_v2_signals.csv"
SPX_PARQUET  = "data/spx_daily.parquet"
OUT_PDF      = "results/r3e_alarm_timeline.pdf"

CRISIS_WINDOWS = {
    "GFC 2008-09":     ("2008-09-01", "2009-06-30"),
    "COVID 2020":      ("2020-02-01", "2020-06-30"),
    "Rate hikes 2022": ("2022-01-01", "2022-12-31"),
}


# ---------------------------------------------------------------------------
# Load signals + derive alarm dates from state-transition (0 -> 1)
# ---------------------------------------------------------------------------
def alarm_dates(state_series):
    """Return the dates at which state transitions from 0 to 1."""
    s = state_series.astype(int).to_numpy()
    transitions = (s[1:] == 1) & (s[:-1] == 0)
    alarm_idx = np.where(transitions)[0] + 1  # transition is observed at t
    return state_series.index[alarm_idx]


print("Loading signals + SPX...")
sig = pd.read_csv(SIGNALS_PATH, parse_dates=["date"]).sort_values("date").set_index("date")

fixed_alarms = alarm_dates(sig["cusum_fixed_state"])
abs_alarms   = alarm_dates(sig["cusum_abs_state"])
print(f"  Window: {sig.index.min().date()} to {sig.index.max().date()}  ({len(sig)} rows)")
print(f"  CUSUM-fixed alarms: {len(fixed_alarms)}")
for d in fixed_alarms:
    print(f"    {d.date()}")
print(f"  CUSUM-abs alarms:   {len(abs_alarms)}")
for d in abs_alarms:
    print(f"    {d.date()}")


# ---------------------------------------------------------------------------
# Load SPX (close on log scale)
# ---------------------------------------------------------------------------
spx = pd.read_parquet(SPX_PARQUET)
if "date" in spx.columns:
    spx["date"] = pd.to_datetime(spx["date"])
    spx = spx.set_index("date")
else:
    spx.index = pd.to_datetime(spx.index)
spx = spx.sort_index().reindex(sig.index)

if "close" in spx.columns and spx["close"].notna().any():
    y = spx["close"]
    ylabel = "S\\&P 500 index (log scale)"
elif "log_return" in spx.columns and spx["log_return"].notna().any():
    y = np.exp(spx["log_return"].cumsum())
    y = y / y.iloc[0]
    ylabel = "S\\&P 500 cumulative return (log scale, normalized)"
else:
    raise ValueError("SPX parquet must contain 'close' or 'log_return' column")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 5.5))

# crisis shading (background)
for name, (cs, ce) in CRISIS_WINDOWS.items():
    ax.axvspan(pd.Timestamp(cs), pd.Timestamp(ce),
               color="0.5", alpha=0.12, zorder=0)

# SPX line
ax.plot(sig.index, y, color="black", lw=0.9, alpha=0.85, zorder=1)

# CUSUM-abs alarms (orange dotted, drawn first so red overlays where overlap)
for d in abs_alarms:
    ax.axvline(d, color="darkorange", ls=":", lw=0.9, alpha=0.75, zorder=2)

# CUSUM-fixed alarms (red solid, foreground)
for d in fixed_alarms:
    ax.axvline(d, color="crimson", ls="-", lw=1.3, alpha=0.9, zorder=3)

ax.set_yscale("log")
ax.set_ylabel(ylabel)
ax.set_xlabel("Date")
ax.grid(alpha=0.3, which="both")
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

legend_elements = [
    Line2D([0], [0], color="black", lw=1.0, label="S\\&P 500 index"),
    Line2D([0], [0], color="crimson", lw=1.3,
           label=f"CUSUM-fixed alarms (n = {len(fixed_alarms)})"),
    Line2D([0], [0], color="darkorange", ls=":", lw=1.0,
           label=f"CUSUM-abs alarms (n = {len(abs_alarms)})"),
    Patch(facecolor="0.5", alpha=0.25, label="Crisis windows"),
]
ax.legend(handles=legend_elements, loc="upper left",
          framealpha=0.95, fontsize=9)

plt.tight_layout()
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.close()
print(f"\nWrote {OUT_PDF}")
print("Copy this PDF into paper/figures/ before recompiling.")
