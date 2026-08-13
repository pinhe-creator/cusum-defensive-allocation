import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from detectors import cusum_abs, cusum_fixed, cusum_rolling


def power(detector, magnitude, kind, reps, n, tau, burn_in, sigma, seed):
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(reps):
        y = rng.normal(0.0, sigma, n)
        if kind == "mean":
            y[tau:] += magnitude * sigma
        else:
            y[tau:] *= magnitude
        idx = np.flatnonzero(detector(y))
        idx = idx[idx >= burn_in]
        hits += int(np.any(idx >= tau))
    return hits / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replications", type=int, default=1000)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--tau", type=int, default=1000)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--baseline-window", type=int, default=252)
    ap.add_argument("--threshold", type=float, default=8.0)
    ap.add_argument("--drift", type=float, default=0.5)
    ap.add_argument("--cooldown", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/rev_power_curve.csv")
    args = ap.parse_args()

    kw = dict(threshold=args.threshold, drift=args.drift,
              baseline_window=args.baseline_window, cooldown=args.cooldown)
    detectors = [
        ("CUSUM-fixed", lambda y: cusum_fixed(y, side="neg", **kw), "mean"),
        ("CUSUM-rolling", lambda y: cusum_rolling(y, side="neg", **kw), "mean"),
        ("CUSUM-abs", lambda y: cusum_abs(y, baseline="fixed", **kw), "variance"),
        ("CUSUM-abs-rolling", lambda y: cusum_abs(y, baseline="rolling", **kw), "variance"),
    ]

    mean_grid = np.round(np.arange(-1.50, -0.02, 0.05), 4)
    vol_grid = np.round(np.arange(1.05, 3.05, 0.05), 4)

    rows = []
    for name, fn, kind in detectors:
        grid = mean_grid if kind == "mean" else vol_grid
        for m in grid:
            rows.append({
                "detector": name,
                "shift_type": kind,
                "magnitude": float(m),
                "power": power(fn, float(m), kind, args.replications, args.n,
                               args.tau, args.baseline_window, args.sigma, args.seed),
            })

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)

    print("Magnitude at which each detector first reaches a given power level")
    for name in out["detector"].unique():
        sub = out[out["detector"] == name]
        kind = sub["shift_type"].iloc[0]
        parts = []
        for level in (0.5, 0.8, 0.9):
            hit = sub[sub["power"] >= level]
            if len(hit):
                v = hit["magnitude"].max() if kind == "mean" else hit["magnitude"].min()
                parts.append(f"{level:.0%}: {v:+.2f}")
            else:
                parts.append(f"{level:.0%}: not reached")
        print(f"  {name:20s} ({kind:8s})  " + "   ".join(parts))


if __name__ == "__main__":
    main()
