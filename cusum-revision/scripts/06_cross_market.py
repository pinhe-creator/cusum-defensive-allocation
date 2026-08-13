import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from detectors import cusum_abs, cusum_fixed, cusum_rolling
from paper_backtest import (
    TC_BPS,
    TRADING_DAYS,
    alarms_to_state,
    max_drawdown,
    sharpe,
)

W_NORMAL_EQUITY = 0.60
W_RISKOFF_EQUITY = 0.30

DATE_CANDIDATES = ["date", "Date", "DATE", "datadate", "caldt"]
LEVEL_CANDIDATES = ["total_return_index", "index_value", "close", "Close",
                    "PRC", "level", "value", "tri"]
LOG_RETURN_CANDIDATES = ["log_return", "logret", "logreturn"]
SIMPLE_RETURN_CANDIDATES = ["portret", "ret", "return", "Return", "portretx"]


def read_market(path):
    df = pd.read_csv(path)
    date_col = next((c for c in DATE_CANDIDATES if c in df.columns), None)
    if date_col is None:
        date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

    log_col = next((c for c in LOG_RETURN_CANDIDATES if c in df.columns), None)
    if log_col is not None:
        return df[log_col].astype(float).dropna(), log_col

    simple_col = next((c for c in SIMPLE_RETURN_CANDIDATES if c in df.columns), None)
    if simple_col is not None:
        s = df[simple_col].astype(float).dropna()
        if s.abs().max() > 1.0:
            s = s / 100.0
        return np.log1p(s), simple_col

    lvl_col = next((c for c in LEVEL_CANDIDATES if c in df.columns), None)
    if lvl_col is None:
        raise ValueError(f"{path}: no recognized return or level column "
                         f"({list(df.columns)})")
    return np.log(df[lvl_col].astype(float)).diff().dropna(), lvl_col


def backtest_cash(equity_log_ret, state, tc_bps=TC_BPS):
    """Equity-cash overlay. Cash earns zero; costs charged on total turnover."""
    r = np.asarray(equity_log_ret, dtype=float)
    state = np.asarray(state, dtype=int)
    n = len(r)

    w = np.where(state == 1, W_RISKOFF_EQUITY, W_NORMAL_EQUITY)
    w_prev = np.zeros(n)
    w_prev[1:] = w[:-1]

    gross = w_prev * r

    dw = np.zeros(n)
    dw[0] = w[0]
    dw[1:] = w[1:] - w[:-1]
    turnover = 2.0 * np.abs(dw)
    turnover[0] = 1.0
    tc = turnover * (tc_bps / 1e4)
    return gross - tc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/world_indices")
    ap.add_argument("--pattern", default="*.csv")
    ap.add_argument("--threshold", type=float, default=8.0)
    ap.add_argument("--drift", type=float, default=0.5)
    ap.add_argument("--baseline-window", type=int, default=252)
    ap.add_argument("--cooldown", type=int, default=60)
    ap.add_argument("--duration", type=int, default=21)
    ap.add_argument("--tc-bps", type=float, default=TC_BPS)
    ap.add_argument("--out", default="results/rev_cross_market_portfolio.csv")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not files:
        raise SystemExit(f"no files in {args.dir} matching {args.pattern}")

    kw = dict(threshold=args.threshold, drift=args.drift,
              baseline_window=args.baseline_window, cooldown=args.cooldown)
    rows = []

    for path in files:
        market = os.path.splitext(os.path.basename(path))[0]
        try:
            r, src_col = read_market(path)
        except Exception as exc:
            print(f"skipped {market}: {exc}")
            continue
        arr = r.to_numpy()
        n = len(arr)
        if n <= args.baseline_window + 100:
            print(f"skipped {market}: only {n} observations")
            continue

        static = np.zeros(n, dtype=int)
        static_ret = backtest_cash(arr, static, args.tc_bps)
        base_sharpe = sharpe(static_ret)
        base_dd = max_drawdown(static_ret)
        rows.append({
            "market": market, "strategy": "Static 60/40 cash", "n_obs": n,
            "start": str(r.index[0].date()), "end": str(r.index[-1].date()),
            "source_column": src_col,
            "sharpe": base_sharpe, "max_drawdown": base_dd,
            "risk_off_share": 0.0, "n_alarms": 0,
            "sharpe_vs_static": np.nan, "drawdown_vs_static": np.nan,
        })

        for label, fn, kwargs in [
            ("CUSUM-fixed", cusum_fixed, dict(side="neg")),
            ("CUSUM-rolling", cusum_rolling, dict(side="neg")),
            ("CUSUM-abs", cusum_abs, dict(baseline="fixed")),
        ]:
            alarms = fn(arr, **kwargs, **kw)
            st = alarms_to_state(alarms, args.duration)
            ret = backtest_cash(arr, st, args.tc_bps)
            rows.append({
                "market": market, "strategy": label, "n_obs": n,
                "start": str(r.index[0].date()), "end": str(r.index[-1].date()),
                "sharpe": sharpe(ret), "max_drawdown": max_drawdown(ret),
                "risk_off_share": float(np.mean(st == 1)),
                "n_alarms": int(np.sum(alarms)),
                "sharpe_vs_static": sharpe(ret) - base_sharpe,
                "drawdown_vs_static": max_drawdown(ret) - base_dd,
            })

    if not rows:
        raise SystemExit("no markets processed")

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print("Overlay is equity-to-cash: 0.60 equity in the risk-on state and "
          "0.30 in the risk-off state, remainder in cash at zero return. "
          "A positive drawdown_vs_static means a shallower drawdown than the "
          "static allocation.")
    print("Compare n_alarms for CUSUM-fixed against the published Table 8 "
          "alarm counts before reporting.")


if __name__ == "__main__":
    main()
