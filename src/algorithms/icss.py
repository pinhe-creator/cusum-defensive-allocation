"""
ICSS Algorithm: Iterated Cumulative Sums of Squares
====================================================

Reference:
    Inclán, C., & Tiao, G. C. (1994). Use of cumulative sums of squares for
    retrospective detection of changes of variance. Journal of the American
    Statistical Association, 89(427), 913-923.

Description:
    Classical OFFLINE retrospective detector for variance changes in time
    series. Operates on full-sample normalization, so it cannot be used in
    a sequential / online setting without look-ahead bias.

    For a series r_1, ..., r_T with E[r_t] = 0, define
        C_t = sum_{i=1}^{t} r_i^2
        D_t = C_t / C_T - t / T,         t = 1, ..., T   (D_T = 0)
    Under H_0 (constant variance), sqrt(T/2) * D_t behaves like a Brownian
    bridge. The test statistic is
        D* = max_{1 <= t <= T-1} |sqrt(T/2) * D_t|.
    Critical value for 5% level is approximately D*_crit = 1.358.

    ICSS extends single-changepoint detection to MULTIPLE change-points via
    iterative segmentation: detect strongest break, split into two segments,
    repeat on each segment until no further breaks pass the threshold.

Important caveat (paper must acknowledge):
    Classical ICSS assumes IID returns under H_0. Under conditional
    heteroskedasticity (GARCH-type volatility clustering), the test
    statistic exhibits well-documented size distortion and over-rejects
    H_0; see Andreou & Ghysels (2002, J. Applied Econometrics) and Sansó,
    Aragó, Carrion (2004) for evidence and proposed corrections (kappa_1
    and kappa_2 statistics). Daily equity log returns exhibit strong
    volatility clustering, so this implementation will likely detect
    MORE change-points than the true regime structure warrants. We retain
    classical ICSS as a CLASSICAL REFERENCE BENCHMARK to illustrate the
    boundary case "what offline retrospective detection finds, with
    full-sample normalization and no GARCH correction".

Role in this paper:
    Algorithm 6 -- classical retrospective variance-break benchmark.
    Used in Stage 1 (synthetic) and Stage 2 (real-data static detection).
    EXCLUDED from Stage 5 (portfolio backtest) because full-sample
    normalization induces look-ahead bias that disqualifies it from
    sequential decision settings.

Indexing convention:
    All change-point indices are 0-indexed (Python array positions).
    A change-point index k denotes the BOUNDARY between pre-break
    observations returns[0..k] (inclusive) and post-break observations
    returns[k+1..T-1] (inclusive). This matches the convention used in
    the other detectors in this codebase.

Author: Pinhe Chen, Fort Hays State University
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Critical values for ICSS test statistic D*
# Source: Inclán & Tiao (1994), Table 1
# -----------------------------------------------------------------------------
ICSS_CRITICAL_VALUES = {
    0.01: 1.628,   # 1% significance
    0.05: 1.358,   # 5% significance (default)
    0.10: 1.224,   # 10% significance
}


# -----------------------------------------------------------------------------
# Input validation
# -----------------------------------------------------------------------------
def _check_univariate(series) -> np.ndarray:
    """
    Validate and coerce input to a clean 1-D float64 ndarray.

    Raises
    ------
    ValueError
        If input is not 1-D, is empty, or contains NaN / inf.
    """
    x = np.asarray(series, dtype=float).flatten()
    if x.ndim != 1:
        raise ValueError(f"ICSS requires 1-D input, got shape={x.shape}.")
    if len(x) == 0:
        raise ValueError("series is empty")
    if not np.all(np.isfinite(x)):
        raise ValueError(
            "series contains NaN or infinite values; "
            "drop or impute before passing to ICSS"
        )
    return x


def _compute_D_statistic(returns: np.ndarray) -> Tuple[np.ndarray, float, int]:
    """
    Compute the ICSS D-statistic for a segment of returns.

    The segment is locally de-meaned internally (classical ICSS assumes
    mean-zero observations within each segment, not across the full sample).

    For locally-centered returns r_1, ..., r_T:
        C_t = sum_{i=1}^t r_i^2
        D_t = C_t / C_T - t / T,        t = 1, ..., T  (D_T = 0)
        D* = max_t |sqrt(T/2) * D_t|

    Parameters
    ----------
    returns : np.ndarray
        1-D array of returns. Will be locally de-meaned internally.

    Returns
    -------
    D_series : np.ndarray
        Array of D_t values for t = 1, ..., T (length T). The last value
        D_T is exactly 0.
    D_star : float
        Test statistic = max |sqrt(T/2) * D_t|.
    k_star : int
        0-indexed array position of the maximum (candidate change-point).
        If k_star = i, the candidate break separates returns[0..i] from
        returns[i+1..T-1].
    """
    returns = np.asarray(returns, dtype=float)
    T = len(returns)
    if T < 4:
        return np.zeros(T), 0.0, 0

    # Local (segment-level) de-mean. Classical ICSS assumes E[r] = 0
    # within each segment; using full-sample mean would inflate the SOS
    # by n_S * (mu_local - mu_full)^2 and shift the D-statistic shape.
    returns = returns - np.mean(returns)

    # Cumulative sum of squares
    C = np.cumsum(returns ** 2)
    C_T = C[-1]
    if C_T <= 0:
        return np.zeros(T), 0.0, 0

    # D_t for t = 1, ..., T. Array position i corresponds to math t = i+1.
    # The last value D_T = C_T/C_T - T/T = 0.
    t_array = np.arange(1, T + 1)
    D_series = C / C_T - t_array / T

    scaled = np.sqrt(T / 2.0) * np.abs(D_series)
    D_star = float(scaled.max())
    k_star = int(scaled.argmax())  # 0-indexed array position

    return D_series, D_star, k_star


def _detect_single_changepoint(
    returns: np.ndarray,
    significance: float = 0.05,
) -> Tuple[bool, int, float]:
    """
    Single-changepoint test on one segment.

    Returns
    -------
    is_significant : bool
        True if D* exceeds critical value at the chosen significance level.
    k_star : int
        0-indexed array position of the candidate change-point within
        this segment.
    D_star : float
        Test statistic value.
    """
    if significance not in ICSS_CRITICAL_VALUES:
        raise ValueError(
            f"significance must be in {list(ICSS_CRITICAL_VALUES.keys())}, "
            f"got {significance}"
        )
    crit = ICSS_CRITICAL_VALUES[significance]
    _, D_star, k_star = _compute_D_statistic(returns)
    return (D_star > crit), k_star, D_star


def icss_detect(
    returns,
    significance: float = 0.05,
    min_segment_length: int = 30,
) -> Dict:
    """
    Iterated Cumulative Sums of Squares (ICSS), Step 1 only.

    Algorithm:
      Apply D-statistic test recursively to segments. Each significant
      break splits its parent segment into two sub-segments; recursion
      continues until no segment yields D* > critical_value. We accept
      a change-point only if BOTH sides of the split would have at least
      `min_segment_length` observations -- this discards unreliable
      edge-of-segment break candidates that can arise from boundary
      behavior of the Brownian-bridge approximation.

    Note on Step 2:
      Step 2 of Inclán & Tiao (1994) is a refinement that adjusts
      change-point locations using neighboring breakpoints. Step 2 is
      omitted here because our use case is benchmarking, not preferred
      estimation of variance-break locations.

    Parameters
    ----------
    returns : array-like
        1-D series of returns. Must be finite (no NaN, no inf). Local
        de-meaning is applied within each segment; user does not need
        to pre-process the mean.
    significance : float
        Significance level for breakpoint test. One of {0.01, 0.05, 0.10}.
        Default 0.05.
    min_segment_length : int
        Minimum number of observations on EACH side of an accepted break,
        and minimum length of a segment to be considered for further
        splitting. Default 30.

    Returns
    -------
    Dict with keys:
        "changepoints": List[int]
            Sorted list of 0-indexed change-point positions. A change-point
            k denotes the boundary returns[0..k] | returns[k+1..T-1].
        "n_changepoints": int
            Number of detected change-points.
        "D_star_values": List[float]
            D* test statistic value for each detected change-point.
        "metadata": dict
            Algorithm metadata.
    """
    returns = _check_univariate(returns)
    T = len(returns)

    # Recursive segmentation using a stack-based approach
    changepoints: List[int] = []
    D_star_values: List[float] = []

    stack: List[Tuple[int, int]] = [(0, T - 1)]  # (start_idx, end_idx) inclusive

    while stack:
        start, end = stack.pop()
        segment_length = end - start + 1

        # Cannot host a break that leaves >= min_segment_length on each side
        if segment_length < 2 * min_segment_length:
            continue

        # Slice raw returns; _compute_D_statistic handles local de-mean
        segment = returns[start : end + 1]
        is_significant, k_local, D_star = _detect_single_changepoint(
            segment, significance=significance
        )

        if not is_significant:
            continue

        # Edge-break filter: require min_segment_length observations on
        # BOTH sides of the candidate break.
        # Pre-break observations: segment[0 .. k_local]   -> length k_local + 1
        # Post-break observations: segment[k_local+1 .. end_local] -> length segment_length - k_local - 1
        left_len = k_local + 1
        right_len = segment_length - k_local - 1
        if left_len < min_segment_length:
            continue
        if right_len < min_segment_length:
            continue

        # Accept change-point
        k_global = start + k_local
        changepoints.append(k_global)
        D_star_values.append(D_star)

        # Recurse on left and right sub-segments
        # Left:  [start,       k_global]      length = left_len
        # Right: [k_global+1,  end]           length = right_len
        stack.append((start, k_global))
        stack.append((k_global + 1, end))

    # Sort chronologically
    sorted_indices = np.argsort(changepoints)
    changepoints_sorted = [changepoints[i] for i in sorted_indices]
    D_star_sorted = [D_star_values[i] for i in sorted_indices]

    return {
        "changepoints": changepoints_sorted,
        "n_changepoints": len(changepoints_sorted),
        "D_star_values": D_star_sorted,
        "metadata": {
            "algorithm_type": "offline_retrospective",
            "output_type": "changepoint_set",
            "target_change": "variance",
            "diagnostic_variant": "classical_ICSS_step1_only",
            "reference": "Inclan & Tiao (1994) JASA 89(427), 913-923",
            "significance_level": significance,
            "critical_value": ICSS_CRITICAL_VALUES[significance],
            "min_segment_length": min_segment_length,
            "n_observations": T,
            "test_statistic": "D_star = max |sqrt(T/2) * D_t|",
            "lookahead": True,
            "portfolio_eligible": False,
            "known_size_distortion_under_GARCH": True,
            "demean_strategy": "segment_local",
        },
    }


def icss_score_series(
    returns,
    significance: float = 0.05,
) -> Dict:
    """
    Compute the ICSS D-statistic SERIES for the full sample.

    For visualization / Stage 1 scoring only -- does NOT return a
    change-point set (use icss_detect for that).

    Returns
    -------
    Dict with keys:
        "D_series": np.ndarray (length T)
            Raw D_t values for t = 1, ..., T. Final value D_T = 0.
        "scaled_abs_D": np.ndarray (length T)
            sqrt(T/2) * |D_t| -- directly comparable to critical value.
        "critical_value": float
            Critical value at chosen significance level.
        "max_D_star": float
            Maximum of scaled_abs_D.
        "max_location": int
            0-indexed array position where max occurs.
        "metadata": dict
    """
    returns = _check_univariate(returns)
    T = len(returns)

    # _compute_D_statistic does its own local de-mean.
    # For a single-segment call (the full sample), local de-mean is
    # equivalent to full-sample de-mean.
    D_series, D_star, k_star = _compute_D_statistic(returns)
    scaled_abs_D = np.sqrt(T / 2.0) * np.abs(D_series)

    return {
        "D_series": D_series,
        "scaled_abs_D": scaled_abs_D,
        "critical_value": ICSS_CRITICAL_VALUES[significance],
        "max_D_star": D_star,
        "max_location": k_star,
        "metadata": {
            "algorithm_type": "offline_retrospective",
            "output_type": "test_statistic_series",
            "target_change": "variance",
            "diagnostic_variant": "classical_ICSS_D_statistic_series",
            "significance_level": significance,
            "demean_strategy": "segment_local",
        },
    }


# -----------------------------------------------------------------------------
# Pipeline-compatible wrapper
# -----------------------------------------------------------------------------
def detect(
    series,
    significance: float = 0.05,
    min_segment_length: int = 30,
) -> Dict:
    """
    Pipeline-compatible ICSS wrapper. Matches the standard benchmark
    interface used by the other detectors in this codebase.

    Returns
    -------
    Dict with standard fields:
        "change_points": List[int]   -- 0-indexed boundary positions
        "scores":        np.ndarray  -- scaled |D_t| series (for plotting)
        "runtime_sec":   float       -- wall-clock detection time
        "hyperparams":   dict        -- (significance, min_segment_length)
        "metadata":      dict        -- includes lookahead=True,
                                        portfolio_eligible=False

    Caveat: ICSS is OFFLINE retrospective. The "scores" array uses
    full-sample normalization (C_T) and therefore embeds look-ahead.
    Do NOT pass this output into any sequential / online portfolio
    routine.
    """
    t0 = time.time()

    x = _check_univariate(series)
    detect_result = icss_detect(
        x,
        significance=significance,
        min_segment_length=min_segment_length,
    )
    score_result = icss_score_series(x, significance=significance)

    return {
        "change_points": detect_result["changepoints"],
        "scores": score_result["scaled_abs_D"],
        "runtime_sec": time.time() - t0,
        "hyperparams": {
            "significance": significance,
            "min_segment_length": min_segment_length,
        },
        "metadata": {
            **detect_result["metadata"],
            "D_star_values": detect_result["D_star_values"],
            "n_changepoints": detect_result["n_changepoints"],
        },
    }


# -----------------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Test 1: single break at index 499 (sigma 0.01 -> 0.03)
    n1, n2 = 500, 500
    sigma1, sigma2 = 0.01, 0.03
    returns = np.concatenate([
        rng.normal(0, sigma1, size=n1),
        rng.normal(0, sigma2, size=n2),
    ])
    out = detect(returns, significance=0.05, min_segment_length=30)
    print("[Test 1: single break at index 499]")
    print(f"  Detected {out['metadata']['n_changepoints']} change-points:")
    for cp, D in zip(out["change_points"], out["metadata"]["D_star_values"]):
        print(f"    cp = {cp:4d}   D* = {D:.3f}")
    print(f"  runtime = {out['runtime_sec']*1000:.2f} ms")
    print(f"  metadata.portfolio_eligible = {out['metadata']['portfolio_eligible']}")

    # Test 2: three true breaks at indices 299, 599, 899
    rng2 = np.random.default_rng(7)
    segments = [
        rng2.normal(0, 0.010, 300),
        rng2.normal(0, 0.025, 300),
        rng2.normal(0, 0.010, 300),
        rng2.normal(0, 0.040, 300),
    ]
    returns2 = np.concatenate(segments)
    out2 = detect(returns2, significance=0.05, min_segment_length=30)
    print("\n[Test 2: three true breaks at indices 299, 599, 899]")
    print(f"  Detected {out2['metadata']['n_changepoints']} change-points:")
    for cp, D in zip(out2["change_points"], out2["metadata"]["D_star_values"]):
        print(f"    cp = {cp:4d}   D* = {D:.3f}")

    # Test 3: no-change baseline
    rng3 = np.random.default_rng(99)
    no_change = rng3.normal(0, 0.015, 1000)
    out3 = detect(no_change, significance=0.05, min_segment_length=30)
    print("\n[Test 3: no-change baseline, 1000 obs]")
    print(f"  Detected {out3['metadata']['n_changepoints']} change-points "
          "(expect 0 with possible Type I errors)")
    for cp, D in zip(out3["change_points"], out3["metadata"]["D_star_values"]):
        print(f"    cp = {cp:4d}   D* = {D:.3f}")

    # Test 4: edge-break behavior
    # Two true breaks at indices 49 and 949 are within min_segment_length=100
    # of the series boundary. The edge-break filter discards any break whose
    # sides have fewer than min_segment_length observations. The exact
    # detection outcome depends on where the D-statistic maximum lands -
    # which can fall on an edge (filtered) or in the interior (accepted
    # as a spurious break introduced by the retrospective statistic).
    rng4 = np.random.default_rng(13)
    edge_test = np.concatenate([
        rng4.normal(0, 0.01, 50),
        rng4.normal(0, 0.04, 900),
        rng4.normal(0, 0.01, 50),
    ])
    out4 = detect(edge_test, significance=0.05, min_segment_length=100)
    print("\n[Test 4: edge-break filter, true breaks at indices 49 and 949,"
          " min_segment_length=100]")
    print(f"  Detected {out4['metadata']['n_changepoints']} change-points.")
    print("  Note: any reported break here reflects either correct edge")
    print("  filtering (0 cps) or interior approximation of the edge break")
    print("  produced by the retrospective statistic; both outcomes are")
    print("  consistent with classical ICSS behavior on this geometry.")
    for cp, D in zip(out4["change_points"], out4["metadata"]["D_star_values"]):
        print(f"    cp = {cp:4d}   D* = {D:.3f}")

    # Test 5: finite-input check
    print("\n[Test 5: finite-input validation]")
    bad = np.array([0.01, 0.02, np.nan, 0.03])
    try:
        detect(bad)
        print("  ERROR: expected ValueError for NaN input, none raised")
    except ValueError as e:
        print(f"  OK: ValueError raised for NaN input: {e}")
    bad2 = np.array([0.01, np.inf, 0.03])
    try:
        detect(bad2)
        print("  ERROR: expected ValueError for inf input, none raised")
    except ValueError as e:
        print(f"  OK: ValueError raised for inf input: {e}")

    # Test 6: off-by-one boundary check
    # Construct a segment of length 60 with a break at index 29 (left = 30,
    # right = 30, both exactly equal to min_segment_length=30).
    # Under the corrected boundary, the break should be ACCEPTABLE.
    rng6 = np.random.default_rng(55)
    boundary_test = np.concatenate([
        rng6.normal(0, 0.01, 30),   # indices 0..29
        rng6.normal(0, 0.10, 30),   # indices 30..59
    ])
    out6 = detect(boundary_test, significance=0.05, min_segment_length=30)
    print("\n[Test 6: off-by-one boundary - break at index 29 with"
          " min_segment_length=30 (each side exactly equal)]")
    print(f"  Detected {out6['metadata']['n_changepoints']} change-points "
          "(expect 1 - the break with both sides exactly 30 obs)")
    for cp, D in zip(out6["change_points"], out6["metadata"]["D_star_values"]):
        print(f"    cp = {cp:4d}   D* = {D:.3f}")
