"""Excess-return Sharpe ratios and allocation-rule sensitivity.

Produces the Excess Sharpe column of Table 4 and Tables S13 and S14:

  results/excess_sharpe.csv          Sharpe ratios under a zero risk-free rate and on returns in
                                     excess of the daily one-month Treasury-bill rate
  results/excess_sharpe_pairs.csv    stationary-bootstrap inference on excess-return Sharpe
                                     differences for all fifteen strategy pairs
  results/allocation_sensitivity.csv one-at-a-time sensitivity of the CUSUM overlays to the
                                     risk-off holding period and the risk-off equity weight

The risk-free rate is the daily RF series of the Fama/French 3 factors file from the Kenneth R.
French Data Library. Run from the repository root:

    python3 cusum-revision/scripts/09_allocation_and_excess_sharpe.py
"""
import argparse
import io
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SIGNALS = "results/portfolio_v2_signals.csv"
ALARMS = "results/portfolio_v2_alarms.csv"
PORTFOLIO_METRICS = "results/portfolio_v2_metrics.csv"
HAM_STATES = "results/r3b_hamilton_ms_states.csv"
HAM_METRICS = "results/r3b_hamilton_ms_metrics.csv"
SPY = "data/spy_daily.parquet"
IEF = "data/ief_daily.parquet"
RF_CACHE = "data/ff_rf_daily.csv"
FRENCH_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
              "F-F_Research_Data_Factors_daily_CSV.zip")

OUT_EXCESS = "results/excess_sharpe.csv"
OUT_PAIRS = "results/excess_sharpe_pairs.csv"
OUT_ALLOC = "results/allocation_sensitivity.csv"

TC_BPS = 10
W_NORMAL = (0.60, 0.40)
RISKOFF_EQUITY_DEFAULT = 0.30
HOLD_DEFAULT = 20
ANN = 252
BLOCK = 20
SEED = 42
BONF = 1.0 - 0.05 / 15
SHARPE_TOL = 1e-6
DD_TOL = 5e-4

HOLDS = [5, 10, 20, 40, 60]
RISKOFF_EQUITY = [0.45, 0.30, 0.15, 0.00]
SWEEP_STRATEGIES = ["CUSUM-fixed", "CUSUM-abs", "Adaptive CUSUM"]

CRISES = {
    "GFC 2008-2009": ("2008-09-01", "2009-06-30"),
    "COVID 2020": ("2020-02-01", "2020-06-30"),
    "Rate hikes 2022": ("2022-01-01", "2022-12-31"),
}

STATE_COLS = {
    "Static 60/40": "static_state",
    "VIX threshold": "vix_state",
    "Adaptive CUSUM": "cusum_adaptive_state",
    "CUSUM-fixed": "cusum_fixed_state",
    "CUSUM-abs": "cusum_abs_state",
}
ALL = list(STATE_COLS) + ["Hamilton MS-AR(1)"]
PAIRS = [
    ("CUSUM-fixed", "Static 60/40"), ("CUSUM-fixed", "VIX threshold"),
    ("CUSUM-fixed", "Adaptive CUSUM"), ("CUSUM-fixed", "CUSUM-abs"),
    ("CUSUM-fixed", "Hamilton MS-AR(1)"), ("CUSUM-abs", "Static 60/40"),
    ("CUSUM-abs", "VIX threshold"), ("CUSUM-abs", "Adaptive CUSUM"),
    ("CUSUM-abs", "Hamilton MS-AR(1)"), ("Adaptive CUSUM", "Static 60/40"),
    ("Adaptive CUSUM", "VIX threshold"), ("Adaptive CUSUM", "Hamilton MS-AR(1)"),
    ("VIX threshold", "Static 60/40"), ("VIX threshold", "Hamilton MS-AR(1)"),
    ("Hamilton MS-AR(1)", "Static 60/40"),
]


def load_returns(path):
    x = pd.read_parquet(path)
    if "date" in x.columns:
        x["date"] = pd.to_datetime(x["date"])
        x = x.sort_values("date").set_index("date")
    else:
        x.index = pd.to_datetime(x.index)
        x = x.sort_index()
    return x["log_return"]


def load_dated_csv(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").set_index("date")


def backtest(spy, ief, state, riskoff_equity=RISKOFF_EQUITY_DEFAULT, tc_bps=TC_BPS):
    n = len(spy)
    w = np.zeros((n, 2))
    w[state == 0] = W_NORMAL
    w[state == 1] = (riskoff_equity, 1.0 - riskoff_equity)
    w_prev = np.zeros((n, 2))
    w_prev[1:] = w[:-1]
    gross = np.sum(w_prev * np.column_stack([spy, ief]), axis=1)
    dw = np.zeros((n, 2))
    dw[0] = w[0]
    dw[1:] = w[1:] - w[:-1]
    return gross - np.sum(np.abs(dw), axis=1) * (tc_bps / 1e4)


def sharpe(r):
    sd = np.std(r)
    return np.nan if sd <= 0 else np.mean(r) / sd * np.sqrt(ANN)


def max_dd(r):
    wealth = np.exp(np.cumsum(r))
    return float((wealth / np.maximum.accumulate(wealth) - 1).min())


def crisis_dd(r, dates, start, end):
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return max_dd(r[mask])


def state_from_alarms(alarms, n, hold):
    s = np.zeros(n, dtype=int)
    for a in alarms:
        s[a:min(a + hold + 1, n)] = 1
    return s


def parse_french(text):
    lines = text.splitlines()
    header = next(l for l in lines if "Mkt-RF" in l)
    cols = [c.strip() for c in header.split(",")]
    rf_pos = cols.index("RF")
    rows = []
    for line in lines:
        if re.match(r"^\s*\d{8}\s*,", line):
            parts = [p.strip() for p in line.split(",")]
            rows.append((parts[0], float(parts[rf_pos])))
    rf = pd.DataFrame(rows, columns=["date", "rf_pct"])
    rf["date"] = pd.to_datetime(rf["date"], format="%Y%m%d")
    rf["rf"] = rf["rf_pct"] / 100.0
    return rf[["date", "rf"]]


def load_rf(dates, rf_csv):
    if rf_csv:
        rf = pd.read_csv(rf_csv, parse_dates=["date"])
    elif Path(RF_CACHE).exists():
        rf = pd.read_csv(RF_CACHE, parse_dates=["date"])
    else:
        raw = urllib.request.urlopen(FRENCH_URL, timeout=120).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            rf = parse_french(z.read(name).decode("latin-1"))
        Path(RF_CACHE).parent.mkdir(parents=True, exist_ok=True)
        rf.to_csv(RF_CACHE, index=False)
    s = rf.set_index("date")["rf"].sort_index()
    if s.index.max() < dates.max():
        sys.exit(f"Risk-free series ends {s.index.max().date()}, "
                 f"before the sample end {dates.max().date()}.")
    aligned = s.reindex(dates)
    if int(aligned.isna().sum()) > 5:
        sys.exit(f"{int(aligned.isna().sum())} sample dates have no risk-free observation.")
    return np.log1p(aligned.ffill().to_numpy())


def joint_index_matrix(T, block, reps, seed):
    rng = np.random.default_rng(seed)
    p = 1.0 / block
    idx = np.empty((reps, T), dtype=np.int64)
    for b in range(reps):
        row = idx[b]
        row[0] = rng.integers(T)
        for t in range(1, T):
            row[t] = rng.integers(T) if rng.random() < p else (row[t - 1] + 1) % T
    return idx


def boot_diffs(ra, rb, idx):
    return np.array([sharpe(ra[i]) - sharpe(rb[i]) for i in idx])


def ci(d, level):
    a = 1.0 - level
    lo, hi = np.quantile(d, [a / 2, 1 - a / 2])
    return float(lo), float(hi)


def metrics_row(r, rf_log, dates, state):
    ex = r - rf_log
    row = {
        "ann_return": np.mean(r) * ANN,
        "ann_vol": np.std(r) * np.sqrt(ANN),
        "sharpe_zero_rf": sharpe(r),
        "ann_excess_return": np.mean(ex) * ANN,
        "sharpe_excess": sharpe(ex),
        "max_dd": max_dd(r),
        "riskoff_days": int(state.sum()),
        "riskoff_share": float(state.mean()),
    }
    for name, (s, e) in CRISES.items():
        row[f"{name}_max_dd"] = crisis_dd(r, dates, s, e)
    return row


def check_inputs(rets, states, alarm_idx, dates, T):
    pm = pd.read_csv(PORTFOLIO_METRICS)
    pm = pm[pm["tc_bps"] == TC_BPS].set_index("strategy")
    ref = {s: float(pm.loc[s, "sharpe"]) for s in STATE_COLS}
    ref["Hamilton MS-AR(1)"] = float(pd.read_csv(HAM_METRICS).iloc[0]["sharpe"])
    for s in ALL:
        if abs(sharpe(rets[s]) - ref[s]) > SHARPE_TOL:
            sys.exit(f"{s}: backtest returns are inconsistent with {PORTFOLIO_METRICS}.")
    for s in SWEEP_STRATEGIES:
        if not np.array_equal(state_from_alarms(alarm_idx[s], T, HOLD_DEFAULT), states[s]):
            sys.exit(f"{s}: alarm dates in {ALARMS} do not reproduce the state series.")
    for s in ["Static 60/40", "CUSUM-fixed"]:
        for name, (a, b) in CRISES.items():
            if abs(crisis_dd(rets[s], dates, a, b) - float(pm.loc[s, f"{name}_max_dd"])) > DD_TOL:
                sys.exit(f"{s}: {name} drawdown is inconsistent with {PORTFOLIO_METRICS}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf-csv", default=None,
                    help="CSV with columns date, rf (daily simple rate as a decimal)")
    ap.add_argument("--reps", type=int, default=10_000)
    ap.add_argument("--skip-bootstrap", action="store_true")
    args = ap.parse_args()

    sig = load_dated_csv(SIGNALS)
    ham = load_dated_csv(HAM_STATES)
    df = sig.copy()
    df["spy"] = load_returns(SPY).reindex(df.index)
    df["ief"] = load_returns(IEF).reindex(df.index)
    df["ham"] = ham["daily_state"].reindex(df.index)
    if df[["spy", "ief", "ham"]].isna().any().any():
        sys.exit("Missing asset returns or Hamilton states on signal dates.")
    dates = df.index
    T = len(df)
    spy, ief = df["spy"].to_numpy(), df["ief"].to_numpy()

    states = {s: df[c].astype(int).to_numpy() for s, c in STATE_COLS.items()}
    states["Hamilton MS-AR(1)"] = df["ham"].astype(int).to_numpy()
    rets = {s: backtest(spy, ief, st) for s, st in states.items()}

    alarms = pd.read_csv(ALARMS)
    alarm_idx = {s: alarms.loc[alarms["strategy"] == s, "alarm_index"].to_numpy()
                 for s in SWEEP_STRATEGIES}
    check_inputs(rets, states, alarm_idx, dates, T)

    rf_log = load_rf(dates, args.rf_csv)
    print(f"Sample: {T} trading days, {dates.min().date()} to {dates.max().date()}")
    print(f"Mean risk-free rate: {np.mean(rf_log) * ANN * 100:.2f}% per year")

    print("\nSharpe ratios under a zero risk-free rate and on excess returns")
    rows = [{"strategy": s, **metrics_row(rets[s], rf_log, dates, states[s])} for s in ALL]
    exdf = pd.DataFrame(rows)
    exdf["rank_zero_rf"] = exdf["sharpe_zero_rf"].rank(ascending=False).astype(int)
    exdf["rank_excess"] = exdf["sharpe_excess"].rank(ascending=False).astype(int)
    print(exdf[["strategy", "ann_return", "ann_excess_return", "ann_vol", "sharpe_zero_rf",
                "sharpe_excess", "rank_zero_rf", "rank_excess"]].round(4).to_string(index=False))
    by = exdf.set_index("strategy")
    print("\nSharpe difference from Static 60/40")
    for s in ALL[1:]:
        print(f"  {s:<20} zero rf {by.loc[s, 'sharpe_zero_rf'] - by.loc['Static 60/40', 'sharpe_zero_rf']:+.4f}"
              f"   excess {by.loc[s, 'sharpe_excess'] - by.loc['Static 60/40', 'sharpe_excess']:+.4f}")
    exdf.to_csv(OUT_EXCESS, index=False)

    if not args.skip_bootstrap:
        print(f"\nExcess-return Sharpe differences, stationary bootstrap "
              f"(B={args.reps}, expected block length {BLOCK}, seed {SEED})")
        idx = joint_index_matrix(T, BLOCK, args.reps, SEED)
        ex = {s: rets[s] - rf_log for s in ALL}
        out = []
        for a, b in PAIRS:
            d = boot_diffs(ex[a], ex[b], idx)
            lo, hi = ci(d, 0.95)
            blo, bhi = ci(d, BONF)
            out.append({"candidate": a, "baseline": b,
                        "delta_sharpe_excess": sharpe(ex[a]) - sharpe(ex[b]),
                        "ci95_lo": lo, "ci95_hi": hi, "excludes_zero_95": lo > 0 or hi < 0,
                        "ci_bonf_lo": blo, "ci_bonf_hi": bhi,
                        "excludes_zero_bonf": blo > 0 or bhi < 0})
        pdf = pd.DataFrame(out)
        print(pdf.round(4).to_string(index=False))
        pdf.to_csv(OUT_PAIRS, index=False)

    print("\nAllocation-rule sensitivity (alarm dates held fixed)")
    static = metrics_row(rets["Static 60/40"], rf_log, dates, states["Static 60/40"])
    rows = [{"strategy": "Static 60/40", "axis": "reference", "value": np.nan, **static}]
    configs = [("hold_days", h, h, RISKOFF_EQUITY_DEFAULT) for h in HOLDS] + \
              [("riskoff_equity", e, HOLD_DEFAULT, e) for e in RISKOFF_EQUITY]
    for s in SWEEP_STRATEGIES:
        for axis, val, hold, eq in configs:
            st = state_from_alarms(alarm_idx[s], T, hold)
            r = backtest(spy, ief, st, riskoff_equity=eq)
            rows.append({"strategy": s, "axis": axis, "value": val, "hold_days": hold,
                         "riskoff_equity": eq, **metrics_row(r, rf_log, dates, st)})
    alloc = pd.DataFrame(rows)
    show = ["strategy", "axis", "value", "riskoff_share", "sharpe_zero_rf", "sharpe_excess",
            "max_dd"] + [f"{c}_max_dd" for c in CRISES]
    print(alloc[show].round(4).to_string(index=False))
    alloc.to_csv(OUT_ALLOC, index=False)


if __name__ == "__main__":
    main()
