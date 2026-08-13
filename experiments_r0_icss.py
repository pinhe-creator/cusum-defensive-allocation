"""
R0 Experiment: ICSS (Inclán-Tiao 1994) as classical reference benchmark
========================================================================

Purpose:
    Evaluate the classical OFFLINE ICSS algorithm on:
      - Stage 1: synthetic benchmark (variance shift simulations,
                 including multi-break scenarios)
      - Stage 2: real S&P 500 data (1990-2026)

    ICSS is NOT included in Stage 5 portfolio backtest because it uses
    full-sample normalization (look-ahead bias). It serves as classical
    reference benchmark to:
      (a) validate that online variance detectors (CUSUM-abs in
          particular) recover the major variance breakpoints found by
          ICSS, and
      (b) pre-empt the reviewer challenge "why didn't you compare against
          CUSUM of squares / ICSS?"

    Known caveat acknowledged in module docstring and paper:
        Classical ICSS over-rejects under conditional heteroskedasticity
        (Andreou & Ghysels 2002, Sansó et al. 2004). S&P 500 daily log
        returns exhibit strong GARCH-type clustering, so the detected
        change-point count is likely an UPPER BOUND on true regime
        breaks.

Outputs (results/r0_*):
    - r0_synthetic_stage1.csv     : F1 / delay / over-detection by scenario
    - r0_realdata_stage2.csv      : detected change-points + pre/post dates
                                    + sigma ratios
    - r0_realdata_overlap.csv     : overlap with CUSUM-fixed / CUSUM-abs
    - r0_D_statistic_series.png   : visualization of D* series

Usage:
    cd /Users/chenpinhe/Downloads/cpd-finance-benchmark/
    python experiments_r0_icss.py

Author: Pinhe Chen, Fort Hays State University
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Match the import convention used by other experiment scripts in this
# codebase (experiments_portfolio_v2.py, experiments_robustness_*.py).
sys.path.insert(0, "src")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from algorithms import icss


# =============================================================================
# Configuration
# =============================================================================
RESULTS_DIR = Path("results")
DATA_DIR = Path("data")
SIGNIFICANCE = 0.05

# min_segment_length: 30 for synthetic (1.5 months daily), 252 for real (1 year)
MIN_SEG_SYNTHETIC = 30
MIN_SEG_REAL = 252

# Tolerance for matching detected vs true change-points in synthetic scoring
TOLERANCE_SYNTHETIC = 20

# Tolerance (in calendar days) for ICSS-vs-online-detector overlap
OVERLAP_TOLERANCE_DAYS = 60

# Online detectors to compare against ICSS in the overlap analysis.
# Restricted to variance-targeted online detectors (CUSUM-abs is the direct
# online counterpart to ICSS) and the paper's main contribution detector
# (CUSUM-fixed; included for completeness even though its target is the
# mean rather than the variance). Adaptive CUSUM is excluded because it
# also targets the mean and is not the paper's main contribution.
KEEP_DETECTORS = {"CUSUM-fixed", "CUSUM-abs"}

# Target-change annotation used in the overlap CSV.
DETECTOR_TARGETS = {
    "CUSUM-abs":   "variance",  # direct online counterpart to ICSS
    "CUSUM-fixed": "mean",      # paper main contribution; not a variance detector
}

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Generic data generator supporting variable number of breaks
# =============================================================================
def generate_variance_shift_series(
    segment_lengths: List[int],
    segment_sigmas: List[float],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[int]]:
    """
    Generate a series with piecewise-constant variance.

    Returns
    -------
    returns : np.ndarray
    true_breaks : List[int]
        0-indexed positions of true breakpoints. A break at position k
        means returns[0..k] are pre-break, returns[k+1..] post-break.
        For a series with K segments, there are K-1 breakpoints.
    """
    assert len(segment_lengths) == len(segment_sigmas), \
        "segment_lengths and segment_sigmas must have same length"

    segments = []
    true_breaks = []
    cursor = 0
    for i, (n, sigma) in enumerate(zip(segment_lengths, segment_sigmas)):
        segments.append(rng.normal(0, sigma, size=n))
        cursor += n
        if i < len(segment_lengths) - 1:
            true_breaks.append(cursor - 1)

    returns = np.concatenate(segments)
    return returns, true_breaks


def compute_F1_and_delay(
    detected: List[int],
    true_breaks: List[int],
    tolerance: int,
) -> Tuple[float, Optional[float]]:
    """
    F1 score with tolerance, plus signed mean detection delay.

    Matching rule: for each detected change-point d, match it to the
    UNMATCHED true break t with smallest |d - t|, provided |d - t| <=
    tolerance. This is "nearest unmatched" matching -- strictly better
    than "first-found-in-order" greedy matching, though still not
    Hungarian-optimal.

    Delay is signed mean of (detection - true break) across matched
    pairs. Offline detectors can produce negative delay when the test
    statistic peaks just before the true regime change.
    """
    if len(true_breaks) == 0:
        F1 = 1.0 if len(detected) == 0 else 0.0
        return F1, None

    TP = 0
    matched_true = set()
    matched_distances = []

    for d in detected:
        candidates = [
            (abs(d - t), j, t)
            for j, t in enumerate(true_breaks)
            if j not in matched_true and abs(d - t) <= tolerance
        ]
        if candidates:
            _, j_best, t_best = min(candidates)
            TP += 1
            matched_true.add(j_best)
            matched_distances.append(d - t_best)

    FP = len(detected) - TP
    FN = len(true_breaks) - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    delay = float(np.mean(matched_distances)) if matched_distances else None
    return F1, delay


# =============================================================================
# STAGE 1: synthetic benchmark
# =============================================================================
def evaluate_synthetic_scenarios(n_replications: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)

    # Each scenario: (name, segment_lengths, segment_sigmas)
    scenarios = [
        ("S1_small_var_shift",
            [500, 500], [0.010, 0.015]),                 # 1.5x sigma
        ("S2_medium_var_shift",
            [500, 500], [0.010, 0.020]),                 # 2.0x sigma
        ("S3_large_var_shift",
            [500, 500], [0.010, 0.030]),                 # 3.0x sigma
        ("S4_no_change",
            [1000],     [0.010]),
        ("S5_two_breaks",
            [400, 300, 300], [0.010, 0.025, 0.012]),
        ("S6_three_breaks",
            [300, 300, 200, 200], [0.010, 0.025, 0.010, 0.030]),
    ]

    rows = []

    for scen in scenarios:
        name, seg_lens, seg_sigmas = scen
        n_true_breaks = len(seg_lens) - 1
        F1_scores = []
        n_detected_list = []
        delays = []

        for _ in range(n_replications):
            returns, true_breaks = generate_variance_shift_series(
                seg_lens, seg_sigmas, rng
            )

            result = icss.detect(
                returns,
                significance=SIGNIFICANCE,
                min_segment_length=MIN_SEG_SYNTHETIC,
            )
            detected = result["change_points"]
            n_detected_list.append(len(detected))

            F1, delay = compute_F1_and_delay(
                detected, true_breaks, tolerance=TOLERANCE_SYNTHETIC
            )
            F1_scores.append(F1)
            if delay is not None:
                delays.append(delay)

        # Signed over-detection: mean(n_detected) - n_true_breaks
        # Positive = ICSS over-reports; negative = under-reports.
        over_detection_mean = float(np.mean(n_detected_list) - n_true_breaks)

        rows.append({
            "scenario": name,
            "n_segments": len(seg_lens),
            "n_true_breaks": n_true_breaks,
            "total_length": sum(seg_lens),
            "segment_lengths": str(seg_lens),
            "segment_sigmas": str(seg_sigmas),
            "n_replications": n_replications,
            "F1_mean": float(np.mean(F1_scores)),
            "F1_std": float(np.std(F1_scores)),
            "n_detected_mean": float(np.mean(n_detected_list)),
            "n_detected_std": float(np.std(n_detected_list)),
            "over_detection_mean": over_detection_mean,
            "delay_mean": float(np.mean(delays)) if delays else np.nan,
            "delay_std": float(np.std(delays)) if delays else np.nan,
        })

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "r0_synthetic_stage1.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[Stage 1] Saved to {out_path}")
    print(df[["scenario", "n_true_breaks", "F1_mean", "F1_std",
              "n_detected_mean", "over_detection_mean",
              "delay_mean"]].to_string(index=False))
    return df


# =============================================================================
# STAGE 2: real S&P 500 data
# =============================================================================
def evaluate_real_data(parquet_path: str = "data/spx_daily.parquet"):
    if not Path(parquet_path).exists():
        print(f"[Stage 2] Data file {parquet_path} not found. Skipping.")
        return None

    df = pd.read_parquet(parquet_path)
    print(f"\n[Stage 2] Loaded {len(df)} rows from {parquet_path}")
    print(f"          Date range: {df.index.min()} to {df.index.max()}")

    if "log_return" not in df.columns:
        df["log_return"] = np.log(df["close"]).diff()

    returns_series = df["log_return"].dropna()
    returns = returns_series.values
    dates = returns_series.index

    result = icss.detect(
        returns,
        significance=SIGNIFICANCE,
        min_segment_length=MIN_SEG_REAL,
    )

    cps = result["change_points"]
    D_stars = result["metadata"]["D_star_values"]

    print(f"\n[Stage 2] ICSS detected {len(cps)} change-points.")
    print(f"          Runtime: {result['runtime_sec']:.3f} sec")
    print(f"          (Caveat: classical ICSS over-rejects under GARCH;"
          f" this count is an UPPER bound on true regime breaks.)")

    # Pre/post sigma windows respecting the change-point convention:
    # cp_idx is the LAST index of pre-break (inclusive); post-break starts
    # at cp_idx + 1. Each window uses 252 observations.
    #
    # Date output convention (point 2 of v4 review):
    #   cp_pre_date  = dates[cp_idx]          last day of pre-break regime
    #   cp_post_date = dates[cp_idx + 1]      first day of post-break regime
    # Overlap matching uses cp_post_date because online alarms necessarily
    # fire on or after the first day of the new regime.
    cp_data = []
    n_obs = len(returns)
    for cp_idx, D_star in zip(cps, D_stars):
        cp_pre_date = dates[cp_idx]
        cp_post_date = dates[cp_idx + 1] if (cp_idx + 1) < n_obs else pd.NaT

        pre_start = max(0, cp_idx - 251)
        pre_end_excl = cp_idx + 1
        sigma_pre = returns[pre_start:pre_end_excl].std()

        post_start = cp_idx + 1
        post_end_excl = min(n_obs, post_start + 252)
        sigma_post = returns[post_start:post_end_excl].std()

        cp_data.append({
            "cp_index": int(cp_idx),
            "cp_pre_date": cp_pre_date,
            "cp_post_date": cp_post_date,
            "D_star": D_star,
            "sigma_pre_252d": sigma_pre,
            "sigma_post_252d": sigma_post,
            "sigma_ratio": (sigma_post / sigma_pre) if sigma_pre > 0 else np.nan,
        })

    cp_df = pd.DataFrame(cp_data)
    out_path = RESULTS_DIR / "r0_realdata_stage2.csv"
    cp_df.to_csv(out_path, index=False)
    print(f"[Stage 2] Saved to {out_path}")
    print(cp_df.to_string(index=False))

    # D* series plot.
    # Note: the plotted statistic is the FULL-SAMPLE ICSS D-statistic
    # (single-pass), while vertical lines mark breakpoints selected by
    # RECURSIVE ICSS segmentation (local D-statistics on each sub-segment).
    # These two need not be congruent.
    scaled_D = result["scores"]
    crit = icss.ICSS_CRITICAL_VALUES[SIGNIFICANCE]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, scaled_D, color="black", linewidth=0.7,
            label=r"Full-sample $\sqrt{T/2} \cdot |D_t|$")
    ax.axhline(crit, color="red", linestyle="--", linewidth=1.0,
               label=f"Critical value ({SIGNIFICANCE} level)")
    for cp_idx in cps:
        ax.axvline(dates[cp_idx], color="blue", linestyle=":",
                   linewidth=0.6, alpha=0.6)
    ax.plot([], [], color="blue", linestyle=":", linewidth=1.0,
            label="Recursive ICSS breakpoints")
    ax.set_xlabel("Date")
    ax.set_ylabel(r"Scaled $|D_t|$  (full-sample)")
    ax.set_title("ICSS on S&P 500 daily log-returns:  "
                 "full-sample statistic and recursive breakpoints")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = RESULTS_DIR / "r0_D_statistic_series.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[Stage 2] Saved plot to {plot_path}")

    return cp_df


# =============================================================================
# Overlap analysis with CUSUM-fixed / CUSUM-abs
# =============================================================================
def overlap_with_online_detectors(
    icss_cp_df: pd.DataFrame,
    alarms_csv_path: str = "results/portfolio_v2_alarms.csv",
    tolerance_days: int = OVERLAP_TOLERANCE_DAYS,
):
    """
    For each ICSS change-point, find the closest CUSUM-fixed and CUSUM-abs
    alarm within tolerance_days. Matching uses the POST-BREAK date of each
    ICSS change-point (first day of new regime), because online alarms
    necessarily fire on or after the first day of the new regime.

    Recognized alarm CSV layouts (auto-detected):
      1. Long with `detector` column:
            alarm_date | detector | ...
      2. Long with `strategy` column (this codebase's convention):
            alarm_date | strategy | ...    (e.g., portfolio_v2_alarms.csv)
      3. Wide format:
            date | CUSUM_fixed | CUSUM_abs | ...

    After parsing, filters to KEEP_DETECTORS = {CUSUM-fixed, CUSUM-abs}.
    Adaptive CUSUM (mean-target with adaptive baseline) is excluded
    because it is not the paper's main contribution and not the
    variance-target reference for ICSS comparison.
    """
    if not Path(alarms_csv_path).exists():
        print(f"\n[Overlap] {alarms_csv_path} not found. Skipping overlap step.")
        return None

    raw = pd.read_csv(alarms_csv_path)
    print(f"\n[Overlap] Loaded {len(raw)} rows from {alarms_csv_path}")
    print(f"          Columns: {list(raw.columns)}")

    online_alarms = None

    # Layout 1: long with `detector`
    if {"alarm_date", "detector"}.issubset(raw.columns):
        online_alarms = raw.copy()
        online_alarms["alarm_date"] = pd.to_datetime(online_alarms["alarm_date"])

    # Layout 2: long with `strategy` (this codebase's convention)
    elif {"alarm_date", "strategy"}.issubset(raw.columns):
        online_alarms = raw.rename(columns={"strategy": "detector"}).copy()
        online_alarms["alarm_date"] = pd.to_datetime(online_alarms["alarm_date"])

    # Layout 3: wide format
    else:
        date_col = None
        for c in raw.columns:
            if c.lower() in ("date", "alarm_date", "timestamp"):
                date_col = c
                break
        if date_col is not None:
            detector_cols = [c for c in raw.columns
                             if c != date_col and (
                                 "cusum" in c.lower() or
                                 "fixed" in c.lower() or
                                 "abs" in c.lower())]
            if detector_cols:
                long_rows = []
                raw[date_col] = pd.to_datetime(raw[date_col])
                for _, row in raw.iterrows():
                    for d in detector_cols:
                        if pd.notna(row[d]) and bool(row[d]):
                            long_rows.append({
                                "alarm_date": row[date_col],
                                "detector": d,
                            })
                online_alarms = pd.DataFrame(long_rows)

    if online_alarms is None or len(online_alarms) == 0:
        print("[Overlap] Could not parse alarms CSV into "
              "(alarm_date, detector) records. Skipping.")
        print("          Adjust the parser in overlap_with_online_detectors()"
              " or rename your columns to 'alarm_date' and 'detector'.")
        return None

    all_detectors = sorted(online_alarms["detector"].unique())
    print(f"[Overlap] All detectors in file: {all_detectors}")

    # Apply KEEP filter
    online_alarms = online_alarms[
        online_alarms["detector"].isin(KEEP_DETECTORS)
    ].copy()

    detectors = sorted(online_alarms["detector"].unique())
    print(f"[Overlap] Kept for ICSS comparison: {detectors}")
    if len(detectors) == 0:
        print(f"[Overlap] None of KEEP_DETECTORS={sorted(KEEP_DETECTORS)} "
              f"found in alarm CSV. Skipping.")
        return None

    overlap_rows = []
    for _, row in icss_cp_df.iterrows():
        # Use POST-break date because online alarms cannot fire before the
        # new regime begins
        if pd.isna(row.get("cp_post_date")):
            continue
        icss_date = pd.to_datetime(row["cp_post_date"])

        for det in detectors:
            det_alarms = online_alarms[
                online_alarms["detector"] == det
            ]["alarm_date"].sort_values().reset_index(drop=True)
            if len(det_alarms) == 0:
                continue

            deltas = (det_alarms - icss_date).dt.days
            within = deltas[deltas.abs() <= tolerance_days]
            if len(within) > 0:
                closest_pos = within.abs().idxmin()
                overlap_rows.append({
                    "icss_post_date": icss_date,
                    "icss_D_star": row["D_star"],
                    "icss_sigma_ratio": row["sigma_ratio"],
                    "detector": det,
                    "detector_target": DETECTOR_TARGETS.get(det, "unknown"),
                    "matched_alarm_date": det_alarms.loc[closest_pos],
                    "delta_days": int(deltas.loc[closest_pos]),
                    "match_status": "matched",
                })
            else:
                overlap_rows.append({
                    "icss_post_date": icss_date,
                    "icss_D_star": row["D_star"],
                    "icss_sigma_ratio": row["sigma_ratio"],
                    "detector": det,
                    "detector_target": DETECTOR_TARGETS.get(det, "unknown"),
                    "matched_alarm_date": pd.NaT,
                    "delta_days": np.nan,
                    "match_status": "no_match",
                })

    overlap_df = pd.DataFrame(overlap_rows)
    out_path = RESULTS_DIR / "r0_realdata_overlap.csv"
    overlap_df.to_csv(out_path, index=False)
    print(f"\n[Overlap] Saved to {out_path}")

    matched = overlap_df[overlap_df["match_status"] == "matched"]
    print("\n[Overlap] Summary (ICSS breaks recovered by each online detector"
          f" within +/- {tolerance_days} days):")
    print(f"          Note: ICSS targets variance; CUSUM-abs is its direct")
    print(f"          online counterpart. CUSUM-fixed targets the mean and")
    print(f"          is reported for completeness as the paper's main")
    print(f"          contribution detector.")
    for det in detectors:
        det_total = len(overlap_df[overlap_df["detector"] == det])
        det_matched = matched[matched["detector"] == det]
        if det_total == 0:
            continue
        pct = 100.0 * len(det_matched) / det_total
        median_delta = (det_matched["delta_days"].median()
                        if len(det_matched) > 0 else np.nan)
        target = DETECTOR_TARGETS.get(det, "?")
        print(f"  {det:14s} ({target:8s})   matched {len(det_matched):2d}/"
              f"{det_total:2d} ({pct:5.1f}%)   "
              f"median delta = {median_delta:+.0f} days")

    return overlap_df


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("=" * 72)
    print("R0 Experiment: ICSS (Inclán-Tiao 1994) -- classical retrospective")
    print("                variance-break benchmark")
    print("=" * 72)
    print("Caveat: classical ICSS is known to over-reject under GARCH-type")
    print("        conditional heteroskedasticity (Andreou & Ghysels 2002;")
    print("        Sansó et al. 2004). Detected change-point count on")
    print("        financial returns is an UPPER bound on true regime breaks.")
    print("=" * 72)

    # Stage 1: synthetic benchmark
    stage1_df = evaluate_synthetic_scenarios(n_replications=200, seed=42)

    # Stage 2: real S&P 500 data
    stage2_df = evaluate_real_data(parquet_path="data/spx_daily.parquet")

    # Cross-comparison with online variance detector (CUSUM-abs) and
    # paper main contribution (CUSUM-fixed). Adaptive CUSUM excluded.
    if stage2_df is not None:
        overlap_df = overlap_with_online_detectors(
            stage2_df,
            alarms_csv_path="results/portfolio_v2_alarms.csv",
            tolerance_days=OVERLAP_TOLERANCE_DAYS,
        )

    print("\n" + "=" * 72)
    print("R0 complete. Outputs in results/r0_*")
    print("=" * 72)
