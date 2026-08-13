import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from detectors import cusum_abs, cusum_fixed, cusum_reset, cusum_rolling
from paper_backtest import load_log_returns

ICSS_STAGE2 = "results/r0_realdata_stage2.csv"


def load_icss_breaks(path):
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col is None:
        raise SystemExit(f"no date column in {path}: {list(df.columns)}")
    df[date_col] = pd.to_datetime(df[date_col])
    ratio_col = next((c for c in df.columns if "ratio" in c.lower()), None)
    df["direction"] = ("increase" if ratio_col is None
                       else np.where(df[ratio_col] > 1.0, "increase", "decrease"))
    return df, date_col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="data/spx_daily.parquet")
    ap.add_argument("--icss", default=ICSS_STAGE2)
    ap.add_argument("--tolerances", default="20,40,60,80,120")
    ap.add_argument("--threshold", type=float, default=8.0)
    ap.add_argument("--drift", type=float, default=0.5)
    ap.add_argument("--baseline-window", type=int, default=252)
    ap.add_argument("--cooldown", type=int, default=60)
    ap.add_argument("--out", default="results/rev_icss_tolerance.csv")
    args = ap.parse_args()

    y = load_log_returns(args.series)
    dates = y.index
    arr = y.to_numpy()

    icss_df, date_col = load_icss_breaks(args.icss)
    icss_df = icss_df[(icss_df[date_col] >= dates[0]) & (icss_df[date_col] <= dates[-1])]

    kw = dict(threshold=args.threshold, drift=args.drift,
              baseline_window=args.baseline_window, cooldown=args.cooldown)
    detectors = {
        "CUSUM-fixed": cusum_fixed(arr, side="neg", **kw),
        "CUSUM-rolling": cusum_rolling(arr, side="neg", **kw),
        "CUSUM-reset": cusum_reset(arr, side="neg", **kw),
        "CUSUM-abs": cusum_abs(arr, baseline="fixed", **kw),
        "CUSUM-abs-rolling": cusum_abs(arr, baseline="rolling", **kw),
    }

    tolerances = [int(x) for x in args.tolerances.split(",")]
    rows = []
    for subset, name in [(icss_df, "all breaks"),
                         (icss_df[icss_df["direction"] == "increase"], "increases only")]:
        targets = pd.to_datetime(subset[date_col]).to_numpy()
        for label, alarms in detectors.items():
            alarm_dates = dates[np.flatnonzero(alarms)].to_numpy()
            for tol in tolerances:
                if len(targets) == 0 or len(alarm_dates) == 0:
                    rate = np.nan
                else:
                    matched = [
                        np.any(np.abs((alarm_dates - t) / np.timedelta64(1, "D")) <= tol)
                        for t in targets
                    ]
                    rate = float(np.mean(matched))
                rows.append({
                    "reference_set": name,
                    "detector": label,
                    "tolerance_days": tol,
                    "n_reference_breaks": len(targets),
                    "n_alarms": len(alarm_dates),
                    "match_rate": rate,
                })

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
