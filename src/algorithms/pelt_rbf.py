"""PELT with RBF kernel cost."""
import time
import numpy as np
import ruptures as rpt


def _prepare_array(series, standardize=True):
    """Convert input to finite 2D array and optionally z-score columns.

    Warning: For heavy-tailed data (e.g., Student-t with df<5, or real
    financial returns), the sample std is unstable (a single extreme value
    can shift std by 30%+), and standardize=True may introduce bias.
    Paper RQ2 should run both standardize=True/False as robustness.
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
    """PELT with RBF kernel.

    Detects mean, variance, and more general distributional changes.
    In finite samples, sensitivity depends on kernel bandwidth,
    penalty, minimum segment length, and signal strength.

    Args:
        series: 1d array with shape (n_obs,) or 2d array with shape
            (n_obs, n_features).
        pen: PELT penalty. Larger values produce fewer detected changepoints.
            Note: pen=10.0 is a sanity-check default only.
            Formal benchmark must use grid-search on DGP-4 to select pen.
        min_size: Minimum segment length.
        jump: Subsampling step. jump=1 keeps full precision;
            jump>1 speeds up but restricts changepoints to a coarser grid.
        standardize: If True, z-score each feature before fitting.
            Default True for cross-dataset comparability of pen.

    Returns:
        Standard detect output dict. Offline algorithms use scores=None.
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

    algo = rpt.Pelt(model="rbf", min_size=min_size, jump=jump).fit(x)
    bps = algo.predict(pen=pen)
    # ruptures may include n_obs as sentinel endpoint; filter it out.
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
            "target_change": "distributional",
        },
    }


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.simulators import dgp2_gaussian_variance

    series, true_bps = dgp2_gaussian_variance(seed=42)
    result = detect(series, pen=5, min_size=30, jump=1)

    print(f"PELT-RBF on DGP-2 (true CPs={true_bps}):")
    print(f"  detected: {result['change_points']}")
    print(f"  runtime: {result['runtime_sec']:.2f}s")

    tol = 20
    hits = [
        cp for cp in result["change_points"]
        if any(abs(cp - t) <= tol for t in true_bps)
    ]
    print(f"  hits within +/-{tol}: {hits}")
    print(f"  hyperparams used: {result['hyperparams']}")
