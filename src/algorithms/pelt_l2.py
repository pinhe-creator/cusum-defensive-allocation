"""PELT with L2 (least-squares) cost.

L2 cost detects MEAN shifts only:
    C(segment) = sum_{t in segment} (x_t - segment_mean)^2

It is blind to variance shifts: a segment with mean=0, std=5 and
a segment with mean=0, std=1 have similar L2 cost despite very
different distributions. For pure variance-shift detection, use
PELT-RBF or transform input (e.g., feed |x| or x^2 to PELT-L2).

This algorithm is included as the offline mean-shift counterpart to
classical (online) CUSUM. The comparison isolates the effect of
online-vs-offline on the same target change type (mean).
"""
import time
import numpy as np
import ruptures as rpt


def _prepare_array(series, standardize=True):
    """Convert input to finite 2D array and optionally z-score columns.

    Warning: For heavy-tailed data, sample std is unstable and
    standardize=True may introduce bias. Use both as robustness.
    """
    x = np.asarray(series, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.ndim != 2:
        raise ValueError(f"series must be 1D or 2D array, got shape={x.shape}")
    if x.shape[0] == 0:
        raise ValueError("series is empty")
    if not np.all(np.isfinite(x)):
        raise ValueError("series contains NaN or infinite values")
    if standardize:
        mu = x.mean(axis=0)
        sd = x.std(axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        x = (x - mu) / sd
    return x


def detect(series, pen=10.0, min_size=30, jump=1, standardize=True):
    """PELT with L2 (least-squares) cost.

    Targets mean shifts only. Significantly faster than PELT-RBF
    because L2 cost is parametric (just segment means and sums of
    squares), not kernel-based.

    Args:
        series: 1d (n_obs,) or 2d (n_obs, n_features) array.
        pen: PELT penalty. Larger -> fewer detected changepoints.
            Note: L2 cost on log returns may produce zero changepoints
            at typical pen values because per-segment mean differences
            are very small (~1e-3). pen often needs to be much smaller
            for raw log returns than for other data.
        min_size: minimum segment length.
        jump: subsampling step (jump=1 = full precision).
        standardize: z-score each feature before fitting.

    Returns:
        Standard detect dict with offline / location_estimates metadata.
    """
    t0 = time.time()
    x = _prepare_array(series, standardize=standardize)
    n_obs = x.shape[0]

    if min_size < 1:
        raise ValueError(f"min_size must be >= 1, got {min_size}")
    if jump < 1:
        raise ValueError(f"jump must be >= 1, got {jump}")
    if n_obs < 2 * min_size:
        raise ValueError(
            f"series too short for min_size={min_size}: n_obs={n_obs}"
        )

    algo = rpt.Pelt(model="l2", min_size=min_size, jump=jump).fit(x)
    bps = algo.predict(pen=pen)
    bps = [int(bp) for bp in bps if bp < n_obs]

    return {
        "change_points": bps,
        "scores": None,
        "runtime_sec": time.time() - t0,
        "hyperparams": {
            "pen": pen,
            "min_size": min_size,
            "jump": jump,
            "standardize": standardize,
        },
        "metadata": {
            "algorithm_type": "offline",
            "output_type": "location_estimates",
            "target_change": "mean_shift",
        },
    }


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.simulators import (
        dgp1_gaussian_mean, dgp2_gaussian_variance, dgp4_garch_switching,
    )

    print("=" * 60)
    print("PELT-L2 self-test")
    print("=" * 60)

    # DGP-1 (mean shift): should work well
    print("\n[1] DGP-1 (Gaussian mean shift) - L2's home turf:")
    series, true_bps = dgp1_gaussian_mean(seed=42)
    result = detect(series, pen=5, min_size=30, jump=1)
    tol = 20
    hits = [cp for cp in result["change_points"]
            if any(abs(cp - t) <= tol for t in true_bps)]
    print(f"  true CPs       : {true_bps}")
    print(f"  detected       : {result['change_points']}")
    print(f"  hits within {tol}: {hits}")
    print(f"  runtime        : {result['runtime_sec']:.3f}s")

    # DGP-2 (variance shift): should fail (L2 is mean-blind to variance)
    print("\n[2] DGP-2 (Gaussian variance shift) - L2's weakness:")
    series, true_bps = dgp2_gaussian_variance(seed=42)
    result = detect(series, pen=5, min_size=30, jump=1)
    hits = [cp for cp in result["change_points"]
            if any(abs(cp - t) <= tol for t in true_bps)]
    print(f"  true CPs       : {true_bps}")
    print(f"  detected       : {result['change_points']}")
    print(f"  hits within {tol}: {hits}")
    print(f"  Expected: low recall - L2 cannot see variance changes")

    # DGP-4 (GARCH): mostly variance changes - should also struggle
    print("\n[3] DGP-4 (GARCH switching) - mixed mean+variance:")
    series, true_bps = dgp4_garch_switching(seed=42)
    result = detect(series, pen=5, min_size=30, jump=1)
    hits = [cp for cp in result["change_points"]
            if any(abs(cp - t) <= tol for t in true_bps)]
    print(f"  true CPs       : {true_bps}")
    print(f"  detected       : {result['change_points']}")
    print(f"  hits within {tol}: {hits}")
    print(f"  metadata       : {result['metadata']}")
