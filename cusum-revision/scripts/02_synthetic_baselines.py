import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from detectors import cusum_abs, cusum_fixed, cusum_reset, cusum_rolling


def evaluate(alarms, tau, burn_in):
    idx = np.flatnonzero(alarms)
    idx = idx[idx >= burn_in]
    post = idx[idx >= tau]
    pre = idx[idx < tau]
    return (len(post) > 0,
            len(idx) > 0 and idx[0] >= tau,
            float(post[0] - tau) if len(post) else np.nan,
            len(pre) > 0)


def run(detector, reps, n, tau, burn_in, sigma, mean_shift, vol_mult, seed):
    rng = np.random.default_rng(seed)
    det, first, delay, fa = [], [], [], []
    for _ in range(reps):
        y = rng.normal(0.0, sigma, n)
        if vol_mult != 1.0:
            y[tau:] *= vol_mult
        if mean_shift != 0.0:
            y[tau:] += mean_shift * sigma
        d, f, dl, a = evaluate(detector(y), tau, burn_in)
        det.append(d); first.append(f); delay.append(dl); fa.append(a)
    delay = np.array(delay, dtype=float)
    return {
        "detection_rate": float(np.mean(det)),
        "first_alarm_rate": float(np.mean(first)),
        "mean_delay": float(np.nanmean(delay)) if np.any(np.isfinite(delay)) else np.nan,
        "false_alarm_rate": float(np.mean(fa)),
    }


def run_null(detector, reps, n, burn_in, sigma, seed):
    rng = np.random.default_rng(seed)
    fired, first = [], []
    region = n - burn_in
    for _ in range(reps):
        y = rng.normal(0.0, sigma, n)
        idx = np.flatnonzero(detector(y))
        idx = idx[idx >= burn_in]
        fired.append(len(idx) > 0)
        first.append(float(idx[0] - burn_in) if len(idx) else float(region))
    return {
        "false_alarm_rate": float(np.mean(fired)),
        "censored_arl0": float(np.mean(first)),
        "censoring_rate": float(1.0 - np.mean(fired)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replications", type=int, default=500)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--tau", type=int, default=600)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--baseline-window", type=int, default=252)
    ap.add_argument("--threshold", type=float, default=8.0)
    ap.add_argument("--drift", type=float, default=0.5)
    ap.add_argument("--cooldown", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    kw = dict(threshold=args.threshold, drift=args.drift,
              baseline_window=args.baseline_window, cooldown=args.cooldown)
    mean_detectors = [
        ("CUSUM-fixed", lambda y: cusum_fixed(y, side="neg", **kw)),
        ("CUSUM-rolling", lambda y: cusum_rolling(y, side="neg", **kw)),
        ("CUSUM-reset", lambda y: cusum_reset(y, side="neg", **kw)),
    ]
    var_detectors = [
        ("CUSUM-abs", lambda y: cusum_abs(y, baseline="fixed", **kw)),
        ("CUSUM-abs-rolling", lambda y: cusum_abs(y, baseline="rolling", **kw)),
        ("CUSUM-abs-reset", lambda y: cusum_abs(y, baseline="reset", **kw)),
    ]
    common = dict(reps=args.replications, n=args.n, tau=args.tau,
                  burn_in=args.baseline_window, sigma=args.sigma, seed=args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    for name, fn in mean_detectors:
        for d in [-0.1, -0.25, -0.5, -1.0, -2.0]:
            r = run(fn, mean_shift=d, vol_mult=1.0, **common)
            r.update({"detector": name, "mean_shift": d})
            rows.append(r)
    t1 = pd.DataFrame(rows)[["detector", "mean_shift", "detection_rate",
                             "first_alarm_rate", "mean_delay", "false_alarm_rate"]]
    t1.to_csv(os.path.join(args.outdir, "rev_table1_mean_shift.csv"), index=False)

    rows = []
    for name, fn in var_detectors:
        for v in [1.5, 2.0, 3.0]:
            r = run(fn, mean_shift=0.0, vol_mult=v, **common)
            r.update({"detector": name, "vol_ratio": v})
            rows.append(r)
    t2 = pd.DataFrame(rows)[["detector", "vol_ratio", "detection_rate",
                             "first_alarm_rate", "mean_delay", "false_alarm_rate"]]
    t2.to_csv(os.path.join(args.outdir, "rev_table2_variance_shift.csv"), index=False)

    rows = []
    for name, fn in mean_detectors + var_detectors:
        for d, v in [(-0.25, 2.0), (-0.5, 2.0)]:
            r = run(fn, mean_shift=d, vol_mult=v, **common)
            r.update({"detector": name, "mean_shift": d, "vol_ratio": v})
            rows.append(r)
    t3 = pd.DataFrame(rows)[["detector", "mean_shift", "vol_ratio", "detection_rate",
                             "first_alarm_rate", "mean_delay", "false_alarm_rate"]]
    t3.to_csv(os.path.join(args.outdir, "rev_table3_mixed_crisis.csv"), index=False)

    rows = []
    for name, fn in mean_detectors + var_detectors:
        r = run_null(fn, args.replications, args.n, args.baseline_window,
                     args.sigma, args.seed)
        r["detector"] = name
        rows.append(r)
    t4 = pd.DataFrame(rows)[["detector", "false_alarm_rate", "censored_arl0",
                             "censoring_rate"]]
    t4.to_csv(os.path.join(args.outdir, "rev_table4_null.csv"), index=False)

    fmt = lambda x: f"{x:.4f}"
    print(f"Design: T={args.n}, change point={args.tau}, sigma0={args.sigma}, "
          f"N={args.replications}, baseline={args.baseline_window}")
    print(f"Pre-change monitoring [{args.baseline_window},{args.tau}) = "
          f"{args.tau - args.baseline_window} obs; null region = "
          f"{args.n - args.baseline_window} obs")
    print()
    print("Mean-shift series"); print(t1.to_string(index=False, float_format=fmt))
    print()
    print("Variance-shift series"); print(t2.to_string(index=False, float_format=fmt))
    print()
    print("Mixed crisis"); print(t3.to_string(index=False, float_format=fmt))
    print()
    print("Null"); print(t4.to_string(index=False, float_format=fmt))
    print()
    print("The CUSUM-fixed and CUSUM-abs rows should reproduce the published "
          "Tables 1-4. Verify before reporting the added rows.")


if __name__ == "__main__":
    main()
