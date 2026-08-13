"""Detection-level evaluation metrics for change-point detection.

Compares a list of detected change points (or alarm times) against a
known list of true change points, and computes:

  - Matching (TP/FP/FN) under one-sided or symmetric tolerance windows
  - Precision, recall, F1
  - Detection delay (DD): mean, median, P90, max
  - Utility-weighted detection loss (UWDL): novel metric proposed for
    this paper. Standard F1 ignores how late a detection arrives;
    UWDL penalizes late detections via exponential decay.
  - False alarms per year: annualized rate of unmatched detections
  - Hausdorff distance: classical CPD distance metric

Matching modes (must be chosen per-algorithm, NOT one fixed default):

  - mode='one_sided':
      For online alarm-time algorithms (CUSUM, BOCPD, online PELT).
      Detected cp matches true cp tau iff  tau <= cp <= tau + window.
      Detections before tau are false positives; algorithms cannot
      "know the future". Among multiple in-window candidates, the
      earliest is chosen (because that defines detection delay).

  - mode='symmetric':
      For offline location-estimate algorithms (PELT, BinSeg, Kernel CPD).
      Detected cp matches true cp tau iff |cp - tau| <= window.
      Offline outputs are location estimates, so |error| is symmetric.
      Among multiple in-window candidates, the NEAREST to tau is chosen
      (because nearest minimizes location error).

Use evaluate_result() to auto-detect mode from algorithm metadata.
Direct evaluate() requires manual mode specification.
"""
import numpy as np


# ====================================================================
# Mode inference from algorithm metadata
# ====================================================================

def infer_matching_mode(result, default="symmetric"):
    """Infer correct matching mode from algorithm output metadata.

    Args:
        result: dict returned by an algorithm's detect() function.
        default: mode to use if metadata is missing or ambiguous.

    Returns:
        'one_sided' if algorithm outputs alarm times (online),
        'symmetric' if it outputs location estimates (offline).
    """
    if not isinstance(result, dict):
        return default
    metadata = result.get("metadata", {})
    output_type = metadata.get("output_type", "")
    algorithm_type = metadata.get("algorithm_type", "")
    if output_type == "alarm_times" or algorithm_type == "online":
        return "one_sided"
    if output_type == "location_estimates" or algorithm_type == "offline":
        return "symmetric"
    return default


# ====================================================================
# Core matching
# ====================================================================

def match_detections(detected, true_cps, window=20, mode="symmetric"):
    """Match detected change points to true change points.

    For each true CP (processed in order), assigns one unmatched
    detection that falls in its tolerance window:
      - one_sided mode: assigns EARLIEST in-window detection
        (defines detection delay correctly).
      - symmetric mode: assigns NEAREST in-window detection
        (minimizes location error; avoids earlier-is-always-better bias).

    Args:
        detected: list[int] or 1D array of detected indices.
        true_cps: list[int] of true CP indices. May be empty (null DGP).
        window: tolerance window size (in observations).
        mode: 'one_sided' or 'symmetric'.

    Returns:
        dict with keys: tp, fp, fn, matched_pairs, unmatched_detected,
        unmatched_true, mode, window.
    """
    if mode not in ("one_sided", "symmetric"):
        raise ValueError(
            f"mode must be 'one_sided' or 'symmetric', got {mode}"
        )
    if window < 0:
        raise ValueError(f"window must be >= 0, got {window}")

    detected_sorted = sorted(int(d) for d in detected)
    true_sorted = sorted(int(t) for t in true_cps)

    used_detected = set()
    matched_pairs = []
    unmatched_true = []

    for tau in true_sorted:
        if mode == "one_sided":
            lo, hi = tau, tau + window
        else:  # symmetric
            lo, hi = tau - window, tau + window

        # Collect all unused in-window candidates
        candidates = []
        for i, cp in enumerate(detected_sorted):
            if i in used_detected:
                continue
            if lo <= cp <= hi:
                candidates.append((i, cp))

        if not candidates:
            unmatched_true.append(tau)
            continue

        # Pick best candidate by mode-specific rule
        if mode == "one_sided":
            # earliest defines detection delay
            best_i, best_cp = candidates[0]
        else:  # symmetric: nearest minimizes location error
            best_i, best_cp = min(candidates, key=lambda z: abs(z[1] - tau))

        used_detected.add(best_i)
        matched_pairs.append((tau, best_cp))

    unmatched_detected = [
        detected_sorted[i] for i in range(len(detected_sorted))
        if i not in used_detected
    ]

    return {
        "tp": len(matched_pairs),
        "fp": len(unmatched_detected),
        "fn": len(unmatched_true),
        "matched_pairs": matched_pairs,
        "unmatched_detected": unmatched_detected,
        "unmatched_true": unmatched_true,
        "mode": mode,
        "window": window,
    }


# ====================================================================
# Precision / recall / F1
# ====================================================================

def precision_recall_f1(match_result):
    """Compute precision, recall, and F1 from match_detections output.

    Conventions:
        - No true CPs and no detections: precision=1, recall=1, f1=1
        - No true CPs but some detections: precision=0, recall=NaN, f1=0
        - True CPs exist but no detections: precision=NaN, recall=0, f1=0

    Note: On null DGPs (no true CPs), F1 is not informative. Use
    false_alarms_per_year as the main metric instead.
    """
    tp = match_result["tp"]
    fp = match_result["fp"]
    fn = match_result["fn"]

    n_detected = tp + fp
    n_true = tp + fn

    if n_detected == 0 and n_true == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if n_detected == 0:
        return {"precision": np.nan, "recall": 0.0, "f1": 0.0}
    if n_true == 0:
        return {"precision": 0.0, "recall": np.nan, "f1": 0.0}

    precision = tp / n_detected
    recall = tp / n_true
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


# ====================================================================
# Detection delay
# ====================================================================

def detection_delay(match_result):
    """Detection delay statistics from matched pairs.

    delay_i = detected_cp_i - true_cp_i

    In one_sided mode, delay >= 0 by construction.
    In symmetric mode, delay can be negative (location estimate
    placed before true CP); reported as-is.

    Returns:
        dict with mean_dd, median_dd, p90_dd, max_dd, n_matched.
        All NaN if no matches.
    """
    pairs = match_result["matched_pairs"]
    if not pairs:
        return {
            "mean_dd": np.nan, "median_dd": np.nan,
            "p90_dd": np.nan, "max_dd": np.nan,
            "n_matched": 0,
        }
    delays = np.array([d - t for (t, d) in pairs], dtype=float)
    return {
        "mean_dd": float(np.mean(delays)),
        "median_dd": float(np.median(delays)),
        "p90_dd": float(np.percentile(delays, 90)),
        "max_dd": float(np.max(delays)),
        "n_matched": len(pairs),
    }


# ====================================================================
# UWDL (paper's novel metric)
# ====================================================================

def utility_weighted_detection_loss(match_result, kappa=10.0,
                                    missed_penalty=1.0):
    """Utility-weighted detection loss.

    For each true CP:
        - If matched in one_sided mode:
            loss_i = 1 - exp(-DD_i / kappa), where DD_i = max(0, cp - tau)
        - If matched in symmetric mode:
            loss_i = 1 - exp(-|cp - tau| / kappa)
            (absolute error, since offline outputs are location estimates
             without temporal direction)
        - If unmatched (missed):
            loss_i = missed_penalty

    UWDL = mean of loss_i over all true CPs.

    Properties:
        - Perfect detection (cp == tau): loss = 0
        - Large error: loss approaches 1
        - Missed CP: loss = missed_penalty (default 1.0)
        - UWDL in [0, 1] when missed_penalty=1

    Args:
        match_result: dict from match_detections (must contain 'mode').
        kappa: penalty decay time scale (default 10 trading days).
        missed_penalty: penalty for unmatched true CPs (default 1.0).

    Returns:
        float UWDL. NaN if no true CPs.
    """
    if kappa <= 0:
        raise ValueError(f"kappa must be > 0, got {kappa}")

    pairs = match_result["matched_pairs"]
    n_missed = match_result["fn"]
    n_true = len(pairs) + n_missed
    mode = match_result.get("mode", "symmetric")

    if n_true == 0:
        return np.nan

    losses = []
    for tau, cp in pairs:
        raw_dd = cp - tau
        if mode == "one_sided":
            dd = max(0.0, raw_dd)  # cannot be negative by construction
        else:  # symmetric: use absolute location error
            dd = abs(raw_dd)
        losses.append(1.0 - np.exp(-dd / kappa))
    losses.extend([missed_penalty] * n_missed)
    return float(np.mean(losses))


# ====================================================================
# False alarms per year
# ====================================================================

def false_alarms_per_year(match_result, n_obs, trading_days_per_year=252):
    """Annualized rate of false alarms (unmatched detections).

    false_alarms_per_year = fp / (n_obs / trading_days_per_year)

    Note: this is a frequency (events per year), not a rate in the
    statistical sense (proportion). The classical name "FPR per year"
    is a misnomer common in CPD literature; we use the more precise
    name false_alarms_per_year.

    Args:
        match_result: dict from match_detections.
        n_obs: total length of the series.
        trading_days_per_year: 252 for daily equities (default),
            365 for daily crypto, etc.
    """
    if n_obs <= 0:
        return np.nan
    years = n_obs / trading_days_per_year
    if years <= 0:
        return np.nan
    return match_result["fp"] / years


# ====================================================================
# Hausdorff distance
# ====================================================================

def hausdorff_distance(detected, true_cps):
    """Classical Hausdorff distance between two CP sets.

    H(A, B) = max( max_{a in A} min_{b in B} |a - b|,
                   max_{b in B} min_{a in A} |a - b| )

    Note: Hausdorff is dominated by the worst false positive.
    For online alarm-time algorithms with many FPs, Hausdorff can be
    misleading as a primary ranking metric. Report it for compatibility
    with classical CPD literature, but do not use it as the main
    cross-algorithm ranking criterion.

    Conventions:
        - Both empty: 0
        - One empty: inf
    """
    a = np.array(sorted(int(d) for d in detected), dtype=float)
    b = np.array(sorted(int(t) for t in true_cps), dtype=float)

    if a.size == 0 and b.size == 0:
        return 0.0
    if a.size == 0 or b.size == 0:
        return float("inf")

    d_ab = max(np.min(np.abs(a_i - b)) for a_i in a)
    d_ba = max(np.min(np.abs(b_i - a)) for b_i in b)
    return float(max(d_ab, d_ba))


# ====================================================================
# Convenience: all metrics in one call
# ====================================================================

def evaluate(detected, true_cps, n_obs, window=20, mode="symmetric",
             kappa=10.0, trading_days_per_year=252):
    """Compute all detection metrics in one call.

    Mode must be specified explicitly. Use evaluate_result() to
    auto-infer mode from algorithm metadata.
    """
    match = match_detections(detected, true_cps,
                             window=window, mode=mode)
    prf = precision_recall_f1(match)
    dd = detection_delay(match)
    uwdl = utility_weighted_detection_loss(match, kappa=kappa)
    fa_y = false_alarms_per_year(
        match, n_obs, trading_days_per_year=trading_days_per_year
    )
    h = hausdorff_distance(detected, true_cps)

    return {
        "tp": match["tp"],
        "fp": match["fp"],
        "fn": match["fn"],
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "mean_dd": dd["mean_dd"],
        "median_dd": dd["median_dd"],
        "p90_dd": dd["p90_dd"],
        "max_dd": dd["max_dd"],
        "uwdl": uwdl,
        "false_alarms_per_year": fa_y,
        "fpr_per_year": fa_y,  # backward-compatible alias
        "hausdorff": h,
        "n_detected": len(detected),
        "n_true": len(true_cps),
        "n_obs": n_obs,
        "window": window,
        "mode": mode,
        "kappa": kappa,
    }


def evaluate_result(result, true_cps, n_obs, window=20,
                    kappa=10.0, trading_days_per_year=252,
                    offline_mode="symmetric"):
    """Evaluate an algorithm result dict, auto-inferring matching mode.

    This is the preferred entry point for benchmark loops. It reads
    'metadata' from the algorithm's detect() output to decide the
    correct matching mode automatically.

    Args:
        result: dict from an algorithm's detect() function.
        true_cps: list[int] of true CPs.
        n_obs: int, series length.
        window: tolerance window.
        kappa: UWDL decay constant.
        trading_days_per_year: for false_alarms_per_year normalization.
        offline_mode: default mode when metadata is missing.

    Returns:
        dict combining evaluate() output with algorithm runtime info.
    """
    detected = result["change_points"]
    mode = infer_matching_mode(result, default=offline_mode)
    out = evaluate(
        detected, true_cps, n_obs=n_obs, window=window,
        mode=mode, kappa=kappa,
        trading_days_per_year=trading_days_per_year,
    )
    out["runtime_sec"] = result.get("runtime_sec", np.nan)
    out["matching_mode_inferred"] = mode
    return out


# ====================================================================
# Self-test
# ====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: perfect detection (symmetric)")
    print("=" * 60)
    res = evaluate([250, 500, 750], [250, 500, 750],
                   n_obs=1000, window=20, mode="symmetric")
    for k in ["tp", "fp", "fn", "precision", "recall", "f1",
              "mean_dd", "uwdl", "false_alarms_per_year", "hausdorff"]:
        print(f"  {k:>22}: {res[k]}")
    print()

    print("=" * 60)
    print("Test 2: PELT-RBF style (offline, symmetric)")
    print("=" * 60)
    print("  detected=[242, 500, 752], true=[250, 500, 750]")
    print("  Expected: all 3 in-window, nearest matching => tp=3, fp=0, fn=0")
    res = evaluate([242, 500, 752], [250, 500, 750],
                   n_obs=1000, window=20, mode="symmetric")
    for k in ["tp", "fp", "fn", "f1", "mean_dd", "uwdl"]:
        print(f"  {k:>22}: {res[k]}")
    print()

    print("=" * 60)
    print("Test 3: same data but ONE-SIDED mode (online-style)")
    print("=" * 60)
    print("  Expected: 242 < 250 is false positive => tp=2, fp=1, fn=1")
    res = evaluate([242, 500, 752], [250, 500, 750],
                   n_obs=1000, window=20, mode="one_sided")
    for k in ["tp", "fp", "fn", "f1", "mean_dd", "uwdl"]:
        print(f"  {k:>22}: {res[k]}")
    print()

    print("=" * 60)
    print("Test 4: nearest-matching corner case")
    print("=" * 60)
    print("  true=[250], detected=[231, 249], window=20, symmetric")
    print("  Old earliest-match would pick 231 (DD=-19, wrong);")
    print("  new nearest-match picks 249 (|err|=1, correct).")
    res = evaluate([231, 249], [250], n_obs=1000, window=20,
                   mode="symmetric")
    for k in ["tp", "fp", "fn", "mean_dd", "uwdl"]:
        print(f"  {k:>22}: {res[k]}")
    print()

    print("=" * 60)
    print("Test 5: CUSUM-style (one_sided)")
    print("=" * 60)
    print("  detected=[243, 314, 412, 503, 594, 754, 846]")
    print("  Expected: 243 early (FP), 503 hits 500, 754 hits 750.")
    print("  CP 250 is missed. => tp=2, fp=5, fn=1")
    res = evaluate(
        [243, 314, 412, 503, 594, 754, 846], [250, 500, 750],
        n_obs=1000, window=20, mode="one_sided",
    )
    for k in ["tp", "fp", "fn", "precision", "recall", "f1",
              "mean_dd", "uwdl", "false_alarms_per_year"]:
        print(f"  {k:>22}: {res[k]}")
    print()

    print("=" * 60)
    print("Test 6: null DGP, 5 false alarms")
    print("=" * 60)
    res = evaluate([100, 200, 300, 400, 500], [], n_obs=1000, window=20,
                   mode="one_sided")
    for k in ["tp", "fp", "fn", "precision", "recall", "f1",
              "false_alarms_per_year"]:
        print(f"  {k:>22}: {res[k]}")
    print()

    print("=" * 60)
    print("Test 7: missed detection")
    print("=" * 60)
    res = evaluate([], [500], n_obs=1000, window=20, mode="one_sided")
    for k in ["tp", "fp", "fn", "precision", "recall", "f1",
              "uwdl", "hausdorff"]:
        print(f"  {k:>22}: {res[k]}")
    print()

    print("=" * 60)
    print("Test 8: evaluate_result with mock metadata")
    print("=" * 60)
    mock_offline = {
        "change_points": [242, 500, 752],
        "runtime_sec": 1.45,
        "metadata": {"algorithm_type": "offline",
                     "output_type": "location_estimates"},
    }
    mock_online = {
        "change_points": [243, 314, 412, 503, 594, 754, 846],
        "runtime_sec": 0.001,
        "metadata": {"algorithm_type": "online",
                     "output_type": "alarm_times"},
    }
    print("  Offline algorithm (PELT-style):")
    res = evaluate_result(mock_offline, [250, 500, 750], n_obs=1000)
    print(f"    inferred mode: {res['matching_mode_inferred']}")
    print(f"    tp={res['tp']}, fp={res['fp']}, fn={res['fn']}, "
          f"f1={res['f1']:.3f}")
    print()
    print("  Online algorithm (CUSUM-style):")
    res = evaluate_result(mock_online, [250, 500, 750], n_obs=1000)
    print(f"    inferred mode: {res['matching_mode_inferred']}")
    print(f"    tp={res['tp']}, fp={res['fp']}, fn={res['fn']}, "
          f"f1={res['f1']:.3f}")
