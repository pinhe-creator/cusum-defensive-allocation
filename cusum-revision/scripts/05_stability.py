import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from paper_backtest import load_paper_panel


def it_statistic(e2):
    n = len(e2)
    total = np.sum(e2)
    if total <= 0:
        return 0.0, 0
    c = np.cumsum(e2)
    d = c / total - np.arange(1, n + 1) / n
    return float(np.sqrt(n / 2.0) * np.max(np.abs(d))), int(np.argmax(np.abs(d)))


def icss(y, critical=1.358, min_length=30):
    e2 = (np.asarray(y, dtype=float) - np.mean(y)) ** 2
    points = []

    def rec(lo, hi):
        if hi - lo < min_length:
            return
        stat, arg = it_statistic(e2[lo:hi])
        if stat > critical:
            cp = lo + arg
            points.append(cp)
            rec(lo, cp + 1)
            rec(cp + 1, hi)

    rec(0, len(e2))
    return sorted(set(points))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-window", type=int, default=252)
    ap.add_argument("--critical", type=float, default=1.358)
    ap.add_argument("--min-length", type=int, default=30)
    ap.add_argument("--windows", default="126,189,252,378,504")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    df = load_paper_panel()
    y = df["spy_ret"].to_numpy()
    w = args.baseline_window
    window = y[:w]

    breaks = icss(window, args.critical, args.min_length)
    half = w // 2
    t_stat, t_p = stats.ttest_ind(window[:half], window[half:], equal_var=False)
    f_stat = np.var(window[:half], ddof=1) / np.var(window[half:], ddof=1)
    d1, d2 = half - 1, w - half - 1
    f_p = 2.0 * min(stats.f.cdf(f_stat, d1, d2), 1.0 - stats.f.cdf(f_stat, d1, d2))

    summary = pd.DataFrame([{
        "baseline_window": w,
        "baseline_start": str(df.index[0].date()),
        "baseline_end": str(df.index[w - 1].date()),
        "mu0": float(np.mean(window)),
        "sigma0": float(np.std(window, ddof=0)),
        "icss_breaks_within_baseline": len(breaks),
        "split_mean_t": float(t_stat),
        "split_mean_p": float(t_p),
        "split_var_f": float(f_stat),
        "split_var_p": float(f_p),
    }])

    sens = pd.DataFrame([{
        "window": int(v),
        "mu0": float(np.mean(y[:int(v)])),
        "sigma0": float(np.std(y[:int(v)], ddof=0)),
    } for v in args.windows.split(",")])

    drift = pd.DataFrame([{
        "date": str(df.index[t].date()),
        "rolling_mu": float(np.mean(y[t - w:t])),
        "rolling_sigma": float(np.std(y[t - w:t], ddof=0)),
    } for t in range(w, len(y), 21)])

    os.makedirs(args.outdir, exist_ok=True)
    summary.to_csv(os.path.join(args.outdir, "rev_stability.csv"), index=False)
    sens.to_csv(os.path.join(args.outdir, "rev_stability_windows.csv"), index=False)
    drift.to_csv(os.path.join(args.outdir, "rev_stability_drift.csv"), index=False)

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
    print(sens.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
    print(f"Rolling sigma over full sample: min {drift['rolling_sigma'].min():.6f}, "
          f"median {drift['rolling_sigma'].median():.6f}, "
          f"max {drift['rolling_sigma'].max():.6f}")


if __name__ == "__main__":
    main()
