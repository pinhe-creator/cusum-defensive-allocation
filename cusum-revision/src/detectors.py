import numpy as np

THRESHOLD = 8.0
DRIFT = 0.5
BASELINE_WINDOW = 252
COOLDOWN = 60


def _page(z, k, h, cooldown, side, start):
    n = len(z)
    alarms = np.zeros(n, dtype=bool)
    sp = sn = 0.0
    block = 0
    for t in range(start, n):
        if block > 0:
            block -= 1
            sp = sn = 0.0
            continue
        if not np.isfinite(z[t]):
            continue
        sp = max(0.0, sp + z[t] - k)
        sn = max(0.0, sn - z[t] - k)
        if side == "two":
            fired = sp > h or sn > h
        elif side == "neg":
            fired = sn > h
        else:
            fired = sp > h
        if fired:
            alarms[t] = True
            sp = sn = 0.0
            block = cooldown
    return alarms


def cusum_fixed(y, threshold=THRESHOLD, drift=DRIFT, baseline_window=BASELINE_WINDOW,
                cooldown=COOLDOWN, side="neg"):
    """Baseline estimated once on the first w observations and never updated."""
    y = np.asarray(y, dtype=float)
    mu = np.mean(y[:baseline_window])
    sd = np.std(y[:baseline_window], ddof=0)
    if sd <= 0:
        raise ValueError("zero baseline dispersion")
    z = (y - mu) / sd
    return _page(z, drift, threshold, cooldown, side, baseline_window)


def cusum_rolling(y, threshold=THRESHOLD, drift=DRIFT, baseline_window=BASELINE_WINDOW,
                  cooldown=COOLDOWN, side="neg"):
    """Baseline re-estimated at every t from the trailing w observations."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    z = np.full(n, np.nan)
    for t in range(baseline_window, n):
        w = y[t - baseline_window:t]
        sd = np.std(w, ddof=0)
        if sd > 0:
            z[t] = (y[t] - np.mean(w)) / sd
    return _page(z, drift, threshold, cooldown, side, baseline_window)


def cusum_reset(y, threshold=THRESHOLD, drift=DRIFT, baseline_window=BASELINE_WINDOW,
                cooldown=COOLDOWN, side="neg"):
    """Baseline re-estimated after each alarm and cooldown."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    alarms = np.zeros(n, dtype=bool)
    mu = np.mean(y[:baseline_window])
    sd = np.std(y[:baseline_window], ddof=0)
    sp = sn = 0.0
    block = 0
    for t in range(baseline_window, n):
        if block > 0:
            block -= 1
            sp = sn = 0.0
            if block == 0:
                lo = max(0, t - baseline_window + 1)
                mu = np.mean(y[lo:t + 1])
                sd = np.std(y[lo:t + 1], ddof=0)
            continue
        if sd <= 0:
            continue
        z = (y[t] - mu) / sd
        sp = max(0.0, sp + z - drift)
        sn = max(0.0, sn - z - drift)
        if side == "two":
            fired = sp > threshold or sn > threshold
        elif side == "neg":
            fired = sn > threshold
        else:
            fired = sp > threshold
        if fired:
            alarms[t] = True
            sp = sn = 0.0
            block = cooldown
    return alarms


def cusum_abs(y, baseline="fixed", **kwargs):
    a = np.abs(np.asarray(y, dtype=float))
    kwargs.setdefault("side", "pos")
    if baseline == "fixed":
        return cusum_fixed(a, **kwargs)
    if baseline == "rolling":
        return cusum_rolling(a, **kwargs)
    if baseline == "reset":
        return cusum_reset(a, **kwargs)
    raise ValueError(baseline)


VARIANTS = {
    "fixed": cusum_fixed,
    "rolling": cusum_rolling,
    "reset": cusum_reset,
}
