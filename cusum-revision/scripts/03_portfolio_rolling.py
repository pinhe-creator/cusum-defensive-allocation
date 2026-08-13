import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from detectors import cusum_abs, cusum_fixed, cusum_reset, cusum_rolling
from paper_backtest import (
    TC_BPS,
    alarms_to_state,
    backtest,
    load_paper_panel,
    published_sharpes,
    reconstruct_all,
    run_lengths,
    sharpe,
    summarize,
)

TOLERANCE = 0.003


def infer_duration(states):
    lens = []
    for label in ("CUSUM-fixed", "CUSUM-abs"):
        if label in states:
            lens.extend(run_lengths(states[label]))
    return int(np.median(lens)) if lens else 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tc-bps", type=float, default=TC_BPS)
    ap.add_argument("--threshold", type=float, default=8.0)
    ap.add_argument("--drift", type=float, default=0.5)
    ap.add_argument("--baseline-window", type=int, default=252)
    ap.add_argument("--cooldown", type=int, default=60)
    ap.add_argument("--duration", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    ap.add_argument("--out", default="results/rev_portfolio.csv")
    ap.add_argument("--returns-out", default="results/rev_strategy_returns.csv")
    args = ap.parse_args()

    df = load_paper_panel()
    returns, states = reconstruct_all(df, tc_bps=args.tc_bps)
    target = published_sharpes(tc_bps=args.tc_bps)

    for label, r in returns.items():
        exp = target.get(label)
        if exp is not None and abs(sharpe(r) - exp) > args.tolerance:
            print(f"ABORT: {label} reconstructed Sharpe {sharpe(r):.4f} vs "
                  f"published {exp:.4f}. Run 01_validate.py first.")
            sys.exit(1)

    duration = args.duration or infer_duration(states)
    print(f"Risk-off duration applied to new detectors: {duration} days "
          f"({'user-specified' if args.duration else 'inferred from published states'})")
    print()

    spy = df["spy_ret"].to_numpy()
    ief = df["ief_ret"].to_numpy()
    kw = dict(threshold=args.threshold, drift=args.drift,
              baseline_window=args.baseline_window, cooldown=args.cooldown)

    new = {
        "CUSUM-rolling": cusum_rolling(spy, side="neg", **kw),
        "CUSUM-reset (replicated)": cusum_reset(spy, side="neg", **kw),
        "CUSUM-fixed (replicated)": cusum_fixed(spy, side="neg", **kw),
        "CUSUM-abs-rolling": cusum_abs(spy, baseline="rolling", **kw),
    }

    rows = []
    series = {"date": df.index}
    for label in returns:
        rows.append(summarize(returns[label], states[label], label))
        series[label] = returns[label]

    for label, alarms in new.items():
        st = alarms_to_state(alarms, duration)
        r = backtest(spy, ief, st, args.tc_bps)
        row = summarize(r, st, label)
        row["n_alarms"] = int(np.sum(alarms))
        rows.append(row)
        series[label] = r

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    pd.DataFrame(series).to_csv(args.returns_out, index=False)

    cols = ["strategy", "ann_return", "ann_vol", "sharpe", "max_drawdown",
            "risk_off_share", "n_onsets"]
    print(out[[c for c in cols if c in out.columns]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print("Published strategies above are reconstructed values and match the paper.")
    print("Replicated CUSUM-fixed is a specification check on the detector code:")
    print("its Sharpe should sit close to the published CUSUM-fixed row.")


if __name__ == "__main__":
    main()
