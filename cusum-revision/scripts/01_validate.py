import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from paper_backtest import (
    TC_BPS,
    load_paper_panel,
    max_drawdown,
    onset_count,
    published_sharpes,
    reconstruct_all,
    run_lengths,
    sharpe,
)

TOLERANCE = 0.003


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tc-bps", type=float, default=TC_BPS)
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    ap.add_argument("--out", default="results/rev_validation.csv")
    args = ap.parse_args()

    df = load_paper_panel()
    returns, states = reconstruct_all(df, tc_bps=args.tc_bps)
    target = published_sharpes(tc_bps=args.tc_bps)

    print(f"Sample: {len(df)} obs, {df.index.min().date()} to {df.index.max().date()}")
    print(f"Strategies reconstructed: {len(returns)}")
    print()

    rows = []
    failures = []
    for label, r in returns.items():
        st = states[label]
        got = sharpe(r)
        exp = target.get(label, float("nan"))
        delta = got - exp if np.isfinite(exp) else float("nan")
        status = "no reference"
        if np.isfinite(delta):
            status = "match" if abs(delta) <= args.tolerance else "MISMATCH"
            if status == "MISMATCH":
                failures.append(label)
        lens = run_lengths(st)
        rows.append({
            "strategy": label,
            "sharpe_reconstructed": got,
            "sharpe_published": exp,
            "delta": delta,
            "status": status,
            "max_drawdown": max_drawdown(r),
            "risk_off_share": float(np.mean(st == 1)),
            "n_onsets": onset_count(st),
            "risk_off_run_min": int(min(lens)) if lens else 0,
            "risk_off_run_median": float(np.median(lens)) if lens else 0.0,
            "risk_off_run_max": int(max(lens)) if lens else 0,
        })

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)

    show = ["strategy", "sharpe_reconstructed", "sharpe_published", "delta",
            "status", "max_drawdown", "risk_off_share", "n_onsets"]
    print(out[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(out[["strategy", "risk_off_run_min", "risk_off_run_median",
               "risk_off_run_max"]].to_string(index=False))

    if failures:
        print()
        print(f"FAILED: {len(failures)} strategy Sharpe values differ from the "
              f"paper by more than {args.tolerance}: {failures}")
        print("Do not run the remaining scripts until this resolves.")
        sys.exit(1)

    print()
    print(f"All reconstructed Sharpe values match the paper within {args.tolerance}.")


if __name__ == "__main__":
    main()
