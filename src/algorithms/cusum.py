"""Classical Page CUSUM (Page 1954) for online mean shift detection.

This is the classical mean-shift CUSUM. It is NOT designed to detect
variance shifts; expect poor performance on pure variance-shift DGPs
(e.g., DGP-2). A variance-sensitive variant (CUSUM on |x| or x^2) is
a separate algorithm and not provided here.
"""
import time
import numpy as np


def _check_univariate(series):
    """Classical CUSUM is univariate; reject 2D input explicitly."""
    x = np.asarray(series, dtype=float)
    if x.ndim != 1:
        raise ValueError(
            f"Classical CUSUM requires 1D input, got shape={x.shape}."
        )
    if x.shape[0] == 0:
        raise ValueError("series is empty")
    if not np.all(np.isfinite(x)):
        raise ValueError("series contains NaN or infinite values")
    return x


def _safe_std(values, eps=1e-8):
    """Standard deviation with a floor to prevent division by zero."""
    sd = float(np.std(values))
    return sd if sd > eps else eps


def detect(series, threshold=8.0, drift=0.25, window=100, cooldown=60):
    """Two-sided Page CUSUM for online mean-shift alarms.

    Maintains recursive statistics tracking positive and negative
    standardized deviations from a rolling baseline:

        z_t  = (x_t - mu_baseline) / sigma_baseline
        S+_t = max(0, S+_{t-1} + z_t - drift)
        S-_t = max(0, S-_{t-1} - z_t - drift)

    Alarm triggers when max(S+, S-) > threshold. After each alarm,
    detection is suppressed for `cooldown` observations to prevent
    clustered alarms from a single regime shift. When the cooldown
    ends, baseline is re-estimated from the most recent `window`
    observations.

    This wrapper returns ALARM TIMES, not offline changepoint estimates.
    Online alarms are systematically delayed relative to true change points;
    detection metrics must account for this asymmetry.

    Args:
        series: 1d np.array.
        threshold: alarm threshold on max(S+, S-).
            Larger values are more conservative.
        drift: slack parameter (in standardized units).
            Default 0.0; larger values reduce false alarms but increase
            detection delay and may miss small regime shifts.
            For financial benchmarks, drift is a calibration dimension:
            grid-search over [0.0, 0.1, 0.25, 0.5].
        window: baseline estimation window length.
        cooldown: observations to suppress alarms after each alarm.

    Returns:
        Standard detect dict with metadata identifying this as an
        online mean-shift detector with alarm-time output.
    """
    t0 = time.time()
    x = _check_univariate(series)
    n = len(x)

    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if n <= window:
        raise ValueError(f"series length n={n} must exceed window={window}")
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")
    if drift < 0:
        raise ValueError(f"drift must be >= 0, got {drift}")
    if cooldown < 0:
        raise ValueError(f"cooldown must be >= 0, got {cooldown}")

    # Initial baseline from first `window` observations (burn-in).
    mu = float(np.mean(x[:window]))
    sigma = _safe_std(x[:window])

    s_pos = 0.0
    s_neg = 0.0
    scores = np.zeros(n)
    change_points = []
    cooldown_remaining = 0

    for t in range(window, n):
        # Cooldown phase: suppress alarms; refresh baseline when cooldown ends.
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            scores[t] = 0.0
            if cooldown_remaining == 0:
                # Re-estimate baseline from most recent `window` observations.
                start = max(0, t - window + 1)
                mu = float(np.mean(x[start:t + 1]))
                sigma = _safe_std(x[start:t + 1])
                s_pos = 0.0
                s_neg = 0.0
            continue

        # Active detection.
        z = (x[t] - mu) / sigma
        s_pos = max(0.0, s_pos + z - drift)
        s_neg = max(0.0, s_neg - z - drift)
        scores[t] = max(s_pos, s_neg)

        if scores[t] > threshold:
            change_points.append(int(t))
            s_pos = 0.0
            s_neg = 0.0
            cooldown_remaining = cooldown

    return {
        "change_points": change_points,
        "scores": scores,
        "runtime_sec": time.time() - t0,
        "hyperparams": {
            "threshold": threshold,
            "drift": drift,
            "window": window,
            "cooldown": cooldown,
        },
        "metadata": {
            "algorithm_type": "online",
            "output_type": "alarm_times",
            "target_change": "mean_shift",
        },
    }


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.simulators import dgp1_gaussian_mean

    series, true_bps = dgp1_gaussian_mean(seed=42)
    result = detect(series, threshold=8.0, drift=0.25,
                    window=100, cooldown=60)

    print(f"CUSUM on DGP-1 (true CPs={true_bps}):")
    print(f"  detected: {result['change_points']}")
    print(f"  runtime: {result['runtime_sec']:.3f}s")
    print(f"  max score: {result['scores'].max():.2f}")

    tol = 20
    hits = [
        cp for cp in result["change_points"]
        if any(abs(cp - t) <= tol for t in true_bps)
    ]
    print(f"  hits within +/-{tol}: {hits}")
    print(f"  hyperparams used: {result['hyperparams']}")
    print(f"  metadata: {result['metadata']}")
