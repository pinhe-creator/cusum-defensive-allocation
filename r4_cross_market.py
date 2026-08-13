"""
r4_cross_market.py

Cross-market out-of-sample validation of the fixed-baseline CUSUM
detectors. The detectors are applied with NO re-tuning---identical
hyperparameters to the S&P 500 study (threshold h=8, drift k=0.5,
baseline window w=252, cooldown c=60)---to six non-US equity markets
that never participated in any parameter choice. If the detectors fire
on each market's well-known crises, the result substantially weakens the
concern that the S&P 500 results reflect overfitting to two US crises.

Detector code is imported directly from algorithms/ so that the
machinery is bit-identical to the S&P 500 application.

Data: WRDS "World Indices by WRDS" daily country returns with dividends
(portret), one CSV per country, columns:
  fic, date, portret, portretx, n, country, currency

UK and China are excluded (export format problems per user).

Run locally; <15 seconds.
"""

import os
import sys
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ---------------------------------------------------------------------------
# Robustly load the SAME detector implementations used for the S&P 500 study,
# by file path. This avoids any dependency on package structure / __init__.py,
# and does NOT trigger the modules' __main__ self-tests (which import
# src.simulators) because importlib sets __name__ to the module name.
# ---------------------------------------------------------------------------
def _find_file(filename, search_root):
    """Return the shallowest path to `filename` under search_root, or None."""
    best = None
    best_depth = 10**9
    for dirpath, dirs, files in os.walk(search_root):
        dirs[:] = [d for d in dirs
                   if not d.startswith(".") and d != "node_modules"]
        if filename in files:
            depth = dirpath.count(os.sep)
            if depth < best_depth:
                best_depth = depth
                best = os.path.join(dirpath, filename)
    return best


def _load_module(mod_name, filepath):
    spec = importlib.util.spec_from_file_location(mod_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_root = os.getcwd()
_fixed_path = _find_file("cusum_fixed.py", _root)
_abs_path = _find_file("cusum_abs.py", _root)
if _fixed_path is None or _abs_path is None:
    sys.exit(f"Could not find cusum_fixed.py / cusum_abs.py under {_root}. "
             f"Run this script from the project root.")
print(f"Loading detectors:\n  {_fixed_path}\n  {_abs_path}\n")
cusum_fixed = _load_module("cusum_fixed", _fixed_path)
cusum_abs = _load_module("cusum_abs", _abs_path)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = "data/world_indices"          # put the per-country CSVs here
MARKETS  = ["Japan", "Germany", "France", "Italy", "HK", "Brazil"]
OUT_CSV  = "results/r4_cross_market_alarms.csv"
OUT_PDF  = "results/r4_cross_market_panel.pdf"

RETURN_COL = "portret"     # daily country return WITH dividends
DATE_COL   = "date"
N_COL      = "n"

# Hyperparameters --- IDENTICAL to the S&P 500 study (no re-tuning)
H = 8.0     # threshold
K = 0.5     # drift / slack
W = 252     # baseline window
C = 60      # cooldown

# Known crisis windows (broad, calendar-based)
CRISES = {
    "Asian 1997-98":   ("1997-07-01", "1998-12-31"),
    "GFC 2008-09":     ("2008-09-01", "2009-06-30"),
    "Euro debt 2011":  ("2011-07-01", "2012-09-30"),
    "China/EM 2015":   ("2015-06-01", "2016-02-29"),
    "COVID 2020":      ("2020-02-01", "2020-06-30"),
    "Tightening 2022": ("2022-01-01", "2022-12-31"),
}


# ---------------------------------------------------------------------------
# Loading + cleaning
# ---------------------------------------------------------------------------
def load_market(name):
    """Load one country CSV, return cleaned DataFrame with log returns."""
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path, on_bad_lines="skip")
    df.columns = [c.strip().lower() for c in df.columns]

    if RETURN_COL not in df.columns:
        alt = "portretx"
        if alt in df.columns:
            print(f"  [{name}] WARNING: '{RETURN_COL}' missing, using '{alt}'")
            ret_col = alt
        else:
            raise ValueError(f"{name}: no return column found "
                             f"(columns: {df.columns.tolist()})")
    else:
        ret_col = RETURN_COL

    df[DATE_COL] = pd.to_datetime(df[DATE_COL].astype(str), errors="coerce")
    df = df.dropna(subset=[DATE_COL])

    r = pd.to_numeric(df[ret_col], errors="coerce")
    df = df.loc[r.notna()].copy()
    df["ret"] = pd.to_numeric(df[ret_col], errors="coerce").loc[df.index].values
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df = df.loc[df["ret"] > -0.99].reset_index(drop=True)

    # magnitude auto-detection: decimal (0.012) vs percent (1.2)
    med_abs = df["ret"].abs().median()
    if med_abs > 0.05:
        df["ret"] = df["ret"] / 100.0
        scale_note = f"percent->decimal (median|r|={med_abs:.3f} -> /100)"
    else:
        scale_note = f"decimal as-is (median|r|={med_abs:.4f})"

    df["logret"] = np.log1p(df["ret"].values)
    df.attrs["scale_note"] = scale_note
    return df


def match_crises(alarm_dates, monitor_start, data_end):
    """For each crisis, mark match / miss / n-a depending on alarm presence."""
    row = {}
    ad = pd.to_datetime(list(alarm_dates))
    for crisis, (cs, ce) in CRISES.items():
        cs, ce = pd.Timestamp(cs), pd.Timestamp(ce)
        if ce < monitor_start or cs > data_end:
            row[crisis] = "n/a"
        else:
            hit = bool(((ad >= cs) & (ad <= ce)).any()) if len(ad) else False
            row[crisis] = "match" if hit else "miss"
    return row


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
print("=" * 70)
print("Cross-market out-of-sample validation (identical hyperparameters)")
print(f"  h={H}, k={K}, w={W}, c={C}  |  cusum_fixed side='negative'")
print("=" * 70)

os.makedirs("results", exist_ok=True)

results = {}
table_rows = []

for name in MARKETS:
    try:
        df = load_market(name)
    except Exception as e:
        print(f"\n[{name}] SKIPPED ({e})")
        continue

    x = df["logret"].to_numpy()
    dates = df[DATE_COL].reset_index(drop=True)

    if len(x) <= W:
        print(f"\n[{name}] SKIPPED: only {len(x)} obs (need > {W})")
        continue

    res_fixed = cusum_fixed.detect(x, threshold=H, drift=K,
                                   baseline_window=W, cooldown=C,
                                   side="negative")
    res_abs = cusum_abs.detect(x, threshold=H, drift=K,
                               baseline_window=W, cooldown=C)

    fixed_idx = res_fixed["change_points"]
    abs_idx = res_abs["change_points"]
    fixed_dates = dates.iloc[fixed_idx].tolist()
    abs_dates = dates.iloc[abs_idx].tolist()

    monitor_start = dates.iloc[W]
    data_end = dates.iloc[-1]
    min_n_baseline = (int(df[N_COL].iloc[:W].min())
                      if N_COL in df.columns else None)

    results[name] = dict(
        dates=dates, ret=df["ret"].to_numpy(),
        fixed_dates=fixed_dates, abs_dates=abs_dates,
        monitor_start=monitor_start, data_end=data_end,
    )

    print(f"\n[{name}]")
    print(f"  scale: {df.attrs['scale_note']}")
    print(f"  obs: {len(x)}  |  {dates.iloc[0].date()} -> {data_end.date()}")
    print(f"  monitoring starts: {monitor_start.date()} (after {W}-day baseline)")
    if min_n_baseline is not None:
        print(f"  min securities in baseline window: {min_n_baseline}")
    print(f"  CUSUM-fixed (neg) alarms: {len(fixed_dates)}")
    for d in fixed_dates:
        print(f"    {d.date()}")
    print(f"  CUSUM-abs alarms:         {len(abs_dates)}")
    for d in abs_dates:
        print(f"    {d.date()}")

    fixed_match = match_crises(fixed_dates, monitor_start, data_end)
    abs_match = match_crises(abs_dates, monitor_start, data_end)
    table_rows.append(dict(market=name, detector="CUSUM-fixed (neg)",
                           n_alarms=len(fixed_dates), **fixed_match))
    table_rows.append(dict(market=name, detector="CUSUM-abs",
                           n_alarms=len(abs_dates), **abs_match))

# ---------------------------------------------------------------------------
# Cross-market crisis table
# ---------------------------------------------------------------------------
tbl = pd.DataFrame(table_rows)
print("\n" + "=" * 70)
print("CROSS-MARKET CRISIS DETECTION TABLE")
print("  match = >=1 alarm in crisis window; miss = none; n/a = not covered")
print("=" * 70)
with pd.option_context("display.max_columns", None, "display.width", 200):
    print(tbl.to_string(index=False))
tbl.to_csv(OUT_CSV, index=False)
print(f"\nWrote {OUT_CSV}")

# ---------------------------------------------------------------------------
# Figure: per-market cumulative index (log scale) + alarm verticals
# ---------------------------------------------------------------------------
valid = list(results.keys())
if valid:
    ncols = 2
    nrows = int(np.ceil(len(valid) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows),
                             squeeze=False)
    for i, name in enumerate(valid):
        ax = axes[i // ncols][i % ncols]
        r = results[name]
        dates = r["dates"]
        wealth = np.cumprod(1.0 + r["ret"])
        ax.plot(dates, wealth, color="black", lw=0.8, alpha=0.85, zorder=1)
        ax.set_yscale("log")
        for crisis, (cs, ce) in CRISES.items():
            ax.axvspan(pd.Timestamp(cs), pd.Timestamp(ce),
                       color="0.5", alpha=0.10, zorder=0)
        for d in r["abs_dates"]:
            ax.axvline(d, color="darkorange", ls=":", lw=0.8, alpha=0.7, zorder=2)
        for d in r["fixed_dates"]:
            ax.axvline(d, color="crimson", ls="-", lw=1.1, alpha=0.85, zorder=3)
        ax.set_title(f"{name}  (fixed n={len(r['fixed_dates'])}, "
                     f"abs n={len(r['abs_dates'])})", fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for j in range(len(valid), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("Fixed-baseline CUSUM alarms across non-US equity markets "
                 "(identical hyperparameters; red = CUSUM-fixed, "
                 "orange = CUSUM-abs)", y=1.005, fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_PDF, bbox_inches="tight")
    plt.close()
    print(f"Wrote {OUT_PDF}")

print("\nDone.")