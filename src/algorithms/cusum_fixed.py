"""Fixed-baseline Page CUSUM for online mean-shift alarms.

Diagnostic variant of classical CUSUM where the baseline (mu_0,
sigma_0) is estimated ONCE from the initial warmup window and then
frozen for the entire series. This contrasts with the adaptive
CUSUM wrapper in cusum.py, which re-estimates the baseline after
each alarm and cooldown period (post-alarm-reset adaptive, not
rolling).

What this algorithm tests:
    Hypothesis: the sparse alarm behavior of adaptive CUSUM on real
    financial data (e.g., 1 alarm in 23 years, silent on COVID-2020)
    stems specifically from baseline re-estimation absorbing crisis
    volatility into the baseline itself. Fixing the baseline tests
    whether removing this mechanism restores detection sensitivity.

What this algorithm IS and IS NOT:
    - IS: a mean-shift detector relative to a fixed initial baseline.
          Triggers when raw returns persistently deviate (up or down)
          from the long-run mean estimated in the warmup window.
    - IS NOT: a volatility detector. On pure variance shifts where
          the mean stays at zero, CUSUM-fixed remains insensitive
          (just like adaptive CUSUM). For variance-targeted detection,
          use cusum_abs.py.

Why CUSUM-fixed may alarm during 2008 and COVID even though it is
not a vol detector: those crises feature persistent NEGATIVE returns
relative to long-run mean (not just vol spike), and CUSUM-fixed
detects this persistent mean-shift via accumulating s_neg.

Cooldown design: cooldown suppresses repeated alarms during a single
regime shift episode (a feature, not a limitation). During cooldown,
baseline does NOT update (that is the entire point of this variant);
only the CUSUM accumulators (s_pos, s_neg) are reset.

Side parameter for portfolio applications:
    side="both"     : alarms on either positive or negative cumulative
                      deviation (default; matches cusum.py for fair
                      detection-stage comparison).
    side="negative" : alarms only on negative cumulative deviation.
                      Use this in portfolio backtest to avoid false
                      risk-off triggers during bull-market regime
                      shifts (e.g., 2017 monotonic uptrend).
    side="positive" : alarms only on positive cumulative deviation
                      (rarely used; provided for symmetry).
"""
import time
import numpy as np


def _check_univariate(series):
    x = np.asarray(series, dtype=float)
    if x.ndim != 1:
        raise ValueError(
            f"CUSUM-fixed requires 1D input, got shape={x.shape}."
        )
    if x.shape[0] == 0:
        raise ValueError("series is empty")
    if not np.all(np.isfinite(x)):
        raise ValueError("series contains NaN or infinite values")
    return x


def _safe_std(values, eps=1e-8):
    sd = float(np.std(values))
    return sd if sd > eps else eps


def _score_by_side(s_pos, s_neg, side):
    """Return the score component matching the active side parameter.

    When side='negative', the user only cares about s_neg as the
    risk-relevant signal; reporting max(s_pos, s_neg) would mix in
    bull-market positive excursions that never trigger an alarm.
    """
    if side == "positive":
        return s_pos
    if side == "negative":
        return s_neg
    return max(s_pos, s_neg)


def detect(series, threshold=5.0, drift=0.0, baseline_window=252,
           cooldown=30, side="both"):
    """Page CUSUM with frozen baseline.

    Args:
        series: 1d np.array of returns.
        threshold: alarm threshold on max(S+, S-).
        drift: slack in standardized units. Default 0.0.
        baseline_window: observations to estimate fixed (mu, sigma).
        cooldown: observations to suppress repeated alarms after alarm.
        side: which direction(s) trigger alarms.
              'both' (default), 'negative', or 'positive'.
              For portfolio risk-off, use 'negative'.

    Returns:
        Standard detect dict with online / alarm_times metadata.
        Also returns the estimated baseline (mu, sigma) for
        reproducibility and debugging.
    """
    t0 = time.time()
    x = _check_univariate(series)
    n = len(x)

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
    if side not in ("both", "negative", "positive"):
        raise ValueError(f"side must be 'both', 'negative', or "
                         f"'positive', got {side}")

    # Frozen baseline from initial window
    mu = float(np.mean(x[:baseline_window]))
    sigma = _safe_std(x[:baseline_window])

    s_pos = 0.0
    s_neg = 0.0
    scores = np.zeros(n)
    change_points = []
    directions = []
    cooldown_remaining = 0

    for t in range(baseline_window, n):
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            scores[t] = 0.0
            if cooldown_remaining == 0:
                # Reset CUSUM accumulators only; baseline stays frozen
                s_pos = 0.0
                s_neg = 0.0
            continue

        z = (x[t] - mu) / sigma
        s_pos = max(0.0, s_pos + z - drift)
        s_neg = max(0.0, s_neg - z - drift)
        scores[t] = _score_by_side(s_pos, s_neg, side)

        alarm_pos = (s_pos > threshold) and side in ("both", "positive")
        alarm_neg = (s_neg > threshold) and side in ("both", "negative")

        if alarm_pos or alarm_neg:
            if alarm_pos and alarm_neg:
                direction = "positive" if s_pos >= s_neg else "negative"
            elif alarm_pos:
                direction = "positive"
            else:
                direction = "negative"
            change_points.append(int(t))
            directions.append(direction)
            s_pos = 0.0
            s_neg = 0.0
            cooldown_remaining = cooldown

    return {
        "change_points": change_points,
        "directions": directions,
        "scores": scores,
        "runtime_sec": time.time() - t0,
        "hyperparams": {
            "threshold": threshold,
            "drift": drift,
            "baseline_window": baseline_window,
            "cooldown": cooldown,
            "side": side,
        },
        "baseline": {
            "mu": mu,
            "sigma": sigma,
        },
        "metadata": {
            "algorithm_type": "online",
            "output_type": "alarm_times",
            "target_change": "mean_shift_vs_fixed_baseline",
            "diagnostic_variant": True,
        },
    }


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.simulators import dgp1_gaussian_mean, dgp2_gaussian_variance

    print("=" * 60)
    print("CUSUM-fixed self-test")
    print("=" * 60)

    print("\n[1] DGP-1 (mean shift) - should work well:")
    series, true_bps = dgp1_gaussian_mean(seed=42)
    result = detect(series, threshold=8.0, drift=0.50, baseline_window=100,
                    cooldown=60, side="both")
    tol = 20
    hits = [cp for cp in result["change_points"]
            if any(abs(cp - t) <= tol for t in true_bps)]
    print(f"  true CPs: {true_bps}")
    print(f"  detected: {result['change_points']}")
    print(f"  directions: {result['directions']}")
    print(f"  hits +/-{tol}: {hits}")
    print(f"  baseline: mu={result['baseline']['mu']:.4f}, "
          f"sigma={result['baseline']['sigma']:.4f}")

    print("\n[2] DGP-2 (variance shift) - not the target of mean-CUSUM:")
    print("  (Any alarms here reflect outlier excursions, not principled")
    print("   variance detection. Use cusum_abs.py for variance shifts.)")
    series, true_bps = dgp2_gaussian_variance(seed=42)
    result = detect(series, threshold=8.0, drift=0.50, baseline_window=100,
                    cooldown=60, side="both")
    hits = [cp for cp in result["change_points"]
            if any(abs(cp - t) <= tol for t in true_bps)]
    print(f"  true CPs: {true_bps}")
    print(f"  detected: {result['change_points']}")
    print(f"  hits +/-{tol}: {hits}")
    print(f"  baseline: mu={result['baseline']['mu']:.4f}, "
          f"sigma={result['baseline']['sigma']:.4f}")

    print("\n[3] Real SPX quick test (negative side only):")
    try:
        import pandas as pd
        spx = pd.read_parquet("data/spx_daily.parquet").dropna()
        x_spx = spx["log_return"].values
        result = detect(x_spx, threshold=8.0, drift=0.50,
                        baseline_window=252, cooldown=60,
                        side="negative")
        dates = [spx.index[i].strftime("%Y-%m-%d")
                 for i in result["change_points"]]
        print(f"  total alarms (negative only): {len(dates)}")
        print(f"  alarm dates: {dates}")
        print(f"  max score: {result['scores'].max():.2f}")
        print(f"  baseline: mu={result['baseline']['mu']:.6f} "
              f"({result['baseline']['mu']*100:.3f}% per day), "
              f"sigma={result['baseline']['sigma']:.6f}")
        print(f"  metadata: {result['metadata']}")
    except Exception as e:
        print(f"  skipped real-data test: {e}")
