"""Absolute-return Page CUSUM for online VARIANCE-shift alarms.

Variant of classical CUSUM that operates on |x_t| instead of x_t.
This makes it sensitive to variance/volatility regime shifts rather
than mean shifts, addressing the fundamental limitation of standard
CUSUM on pure variance changes (where standard mean-CUSUM achieves
F1 of only 0.24 on synthetic DGP-2).

Mechanism:
    Standardize |x_t| against baseline:
        u_t = (|x_t| - mu_|x|_baseline) / sigma_|x|_baseline
    One-sided CUSUM (only positive deviations matter for vol regime):
        S_t = max(0, S_{t-1} + u_t - drift)
    Alarm when S_t > threshold.

Why one-sided: although standardized |x_t| can be below zero
(periods of unusually calm markets), the risk-off application only
cares about upward volatility shifts. A downward deviation in |x_t|
indicates calmer markets and should not trigger a defensive
allocation. So we accumulate only positive deviations.

Why FIXED baseline (rather than adaptive):
    Same diagnostic rationale as cusum_fixed.py - rolling/adaptive
    baselines absorb pre-crisis volatility into baseline itself,
    desensitizing the detector. Fixed baseline preserves long-run
    normal volatility as the comparison reference. This is the
    appropriate choice for paper Section 6 (portfolio risk-off),
    which needs to detect deviations from long-run normalcy.

Cooldown design: same as cusum_fixed.py. Cooldown suppresses repeat
alarms but does NOT update baseline.

Target change: variance / volatility regime shift.
This algorithm is the variance-shift counterpart to cusum_fixed.py,
together providing a 'mean-shift + variance-shift' diagnostic pair.

Unlike cusum_fixed.py, this algorithm does NOT have a 'side' parameter
because the application is inherently one-sided.
"""
import time
import numpy as np


def _check_univariate(series):
    x = np.asarray(series, dtype=float)
    if x.ndim != 1:
        raise ValueError(
            f"CUSUM-abs requires 1D input, got shape={x.shape}."
        )
    if x.shape[0] == 0:
        raise ValueError("series is empty")
    if not np.all(np.isfinite(x)):
        raise ValueError("series contains NaN or infinite values")
    return x


def _safe_std(values, eps=1e-8):
    sd = float(np.std(values))
    return sd if sd > eps else eps


def detect(series, threshold=5.0, drift=0.0, baseline_window=252,
           cooldown=30):
    """CUSUM on absolute returns (variance-sensitive, fixed baseline).

    Args:
        series: 1d np.array of returns (raw; we take abs internally).
        threshold: alarm threshold on one-sided cumulative deviation.
        drift: slack on standardized |x|. Default 0.0.
        baseline_window: observations to estimate fixed (mu, sigma) of |x|.
        cooldown: observations to suppress repeat alarms.

    Returns:
        Standard detect dict with online / alarm_times metadata.
        Also returns the estimated baseline (mu_abs, sigma_abs) for
        reproducibility and debugging.
    """
    t0 = time.time()
    x = _check_univariate(series)
    abs_x = np.abs(x)
    n = len(abs_x)

    if baseline_window < 2:
        raise ValueError(f"baseline_window must be >= 2, "
                         f"got {baseline_window}")
    if n <= baseline_window:
        raise ValueError(f"series length n={n} must exceed "
                         f"baseline_window={baseline_window}")
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")
    if drift < 0:
        raise ValueError(f"drift must be >= 0, got {drift}")
    if cooldown < 0:
        raise ValueError(f"cooldown must be >= 0, got {cooldown}")

    # Fixed baseline on |x|
    mu_abs = float(np.mean(abs_x[:baseline_window]))
    sigma_abs = _safe_std(abs_x[:baseline_window])

    s = 0.0  # one-sided: vol regime means abs_x systematically higher
    scores = np.zeros(n)
    change_points = []
    cooldown_remaining = 0

    for t in range(baseline_window, n):
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            scores[t] = 0.0
            if cooldown_remaining == 0:
                s = 0.0
            continue

        u = (abs_x[t] - mu_abs) / sigma_abs
        s = max(0.0, s + u - drift)
        scores[t] = s

        if scores[t] > threshold:
            change_points.append(int(t))
            s = 0.0
            cooldown_remaining = cooldown

    return {
        "change_points": change_points,
        "scores": scores,
        "runtime_sec": time.time() - t0,
        "hyperparams": {
            "threshold": threshold,
            "drift": drift,
            "baseline_window": baseline_window,
            "cooldown": cooldown,
        },
        "baseline": {
            "mu_abs": mu_abs,
            "sigma_abs": sigma_abs,
        },
        "metadata": {
            "algorithm_type": "online",
            "output_type": "alarm_times",
            "target_change": "variance_shift",
            "diagnostic_variant": True,
        },
    }


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.simulators import (
        dgp1_gaussian_mean, dgp2_gaussian_variance,
        dgp4_garch_switching,
    )

    print("=" * 60)
    print("CUSUM-abs self-test")
    print("=" * 60)

    tol = 20

    print("\n[1] DGP-1 (mean shift) - abs dilutes mean direction:")
    print("  (|x| loses sign information, so mean-shift signal is")
    print("   diluted unless mean magnitude is large.)")
    series, true_bps = dgp1_gaussian_mean(seed=42)
    result = detect(series, threshold=8.0, drift=0.50, baseline_window=100,
                    cooldown=60)
    hits = [cp for cp in result["change_points"]
            if any(abs(cp - t) <= tol for t in true_bps)]
    print(f"  true CPs: {true_bps}")
    print(f"  detected: {result['change_points']}")
    print(f"  hits +/-{tol}: {hits}")
    print(f"  baseline: mu_abs={result['baseline']['mu_abs']:.4f}, "
          f"sigma_abs={result['baseline']['sigma_abs']:.4f}")

    print("\n[2] DGP-2 (variance shift) - the target of cusum_abs:")
    print("  (Standard CUSUM achieves only F1=0.24 here at best.)")
    series, true_bps = dgp2_gaussian_variance(seed=42)
    result = detect(series, threshold=8.0, drift=0.50, baseline_window=100,
                    cooldown=60)
    hits = [cp for cp in result["change_points"]
            if any(abs(cp - t) <= tol for t in true_bps)]
    print(f"  true CPs: {true_bps}")
    print(f"  detected: {result['change_points']}")
    print(f"  hits +/-{tol}: {hits}")
    print(f"  baseline: mu_abs={result['baseline']['mu_abs']:.4f}, "
          f"sigma_abs={result['baseline']['sigma_abs']:.4f}")

    print("\n[3] DGP-4 (GARCH switching) - financial-style vol regime:")
    series, true_bps = dgp4_garch_switching(seed=42)
    result = detect(series, threshold=8.0, drift=0.50, baseline_window=100,
                    cooldown=60)
    hits = [cp for cp in result["change_points"]
            if any(abs(cp - t) <= tol for t in true_bps)]
    print(f"  true CPs: {true_bps}")
    print(f"  detected: {result['change_points']}")
    print(f"  hits +/-{tol}: {hits}")
    print(f"  baseline: mu_abs={result['baseline']['mu_abs']:.4f}, "
          f"sigma_abs={result['baseline']['sigma_abs']:.4f}")

    print("\n[4] Real SPX quick test:")
    try:
        import pandas as pd
        spx = pd.read_parquet("data/spx_daily.parquet").dropna()
        x_spx = spx["log_return"].values
        result = detect(x_spx, threshold=8.0, drift=0.50,
                        baseline_window=252, cooldown=60)
        dates = [spx.index[i].strftime("%Y-%m-%d")
                 for i in result["change_points"]]
        print(f"  total alarms: {len(dates)}")
        print(f"  alarm dates: {dates}")
        print(f"  max score: {result['scores'].max():.2f}")
        print(f"  baseline: mu_abs={result['baseline']['mu_abs']:.4f} "
              f"({result['baseline']['mu_abs']*100:.2f}% per day), "
              f"sigma_abs={result['baseline']['sigma_abs']:.4f}")
        print(f"  metadata: {result['metadata']}")
    except Exception as e:
        print(f"  skipped real-data test: {e}")
