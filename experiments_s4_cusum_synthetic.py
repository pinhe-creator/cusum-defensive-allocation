#!/usr/bin/env python3
"""
experiments_s4_cusum_synthetic.py

Section 4 (Synthetic benchmark) -- detection-power / false-alarm
characterization of the PROPOSED sequential detectors on controlled data
with known change points.

Complements the ICSS synthetic validation in experiments_r0_icss.py
(Stage 1). Here we validate the detectors actually deployed in the
portfolio overlay (Section 6), using the SAME hyperparameters, so the
synthetic results characterize the deployed configuration.

Three matched data-generating processes:
  - CUSUM-fixed is a NEGATIVE-MEAN detector -> negative mean-shift series
  - CUSUM-abs   is a VARIANCE  detector     -> variance-shift series
  - MIXED crisis (negative mean + variance increase) -> realistic crisis
    signal; run BOTH detectors to show the downside channel (CUSUM-fixed)
    and the variance channel (CUSUM-abs).

Design (matching the real setup):
  - sigma0 = 0.01 (1% daily vol, typical equity scale)
  - series length T = 1000; the post-change regime begins at INDEX 600
    (the 601st observation), which is AFTER the 252-observation baseline
    window, so the frozen baseline is estimated on a pure pre-change regime.
  - monitoring region is [baseline_window, T).
  - detector hyperparameters identical to experiments_portfolio_v2.py.

Common random numbers (CRN): for each replication a single base noise
vector is drawn and reused across all shift magnitudes, so the power
curves differ only in the injected shift, not in the underlying noise.

Metrics reported per scenario (mean over N replications):
  - detection_rate        : P(>=1 alarm in the post-change region)        [any-post power]
  - first_alarm_detection_rate : P(the FIRST monitoring alarm is at/after t*)
                            [strict: excludes paths that false-alarm first]
  - mean_delay, delay_std : delay of first post-change alarm (ddof=1)      [ARL1]
  - false_alarm_rate      : P(>=1 alarm in the pre-change region [bw, t*))
  - mean_n_false          : mean number of pre-change alarms

No-change (null) characterization:
  - false_alarm_rate over the whole monitoring region [bw, T)
  - ARL0_censored_mean    : mean obs from start of monitoring to first false
                            alarm, CENSORED at T for no-alarm paths (a lower
                            bound / censored estimate, NOT the true ARL0)

Outputs (results/): summary CSVs + replication-level CSVs for reproducibility.

Run locally (Monte Carlo, ~1-2 min):
    python experiments_s4_cusum_synthetic.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- import detector implementations (same layout as portfolio_v2) ----------
try:
    from algorithms import cusum_fixed, cusum_abs
except Exception:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parent / "src"))
    from algorithms import cusum_fixed, cusum_abs


# ===========================================================================
# Configuration -- keep IN SYNC with experiments_portfolio_v2.py
# ===========================================================================
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_REPLICATIONS = 500
SEED = 42
SIGMA0 = 0.01            # baseline daily volatility
T_LEN = 1000             # series length
T_CP_INDEX = 600         # post-change regime begins at this 0-based index

# Detector hyperparameters -- MUST match the portfolio deployment.
CUSUM_FIXED_KWARGS = {
    "threshold": 8.0, "drift": 0.50,
    "baseline_window": 252, "cooldown": 60, "side": "negative",
}
CUSUM_ABS_KWARGS = {
    "threshold": 8.0, "drift": 0.50,
    "baseline_window": 252, "cooldown": 60,
}

# Single source of truth for the baseline window; guard against drift.
BASELINE_WINDOW = CUSUM_FIXED_KWARGS["baseline_window"]
assert BASELINE_WINDOW == CUSUM_ABS_KWARGS["baseline_window"], \
    "baseline_window must match across detectors for a fair comparison"

# Scenario grids
MEAN_SHIFTS_K = [0.1, 0.25, 0.5, 1.0, 2.0]   # post-change mean = -k * sigma0
VAR_RATIOS = [1.5, 2.0, 3.0]                 # post-change sigma = r * sigma0
MIXED_SCENARIOS = [                          # (mean_k, var_ratio)
    (0.25, 2.0),
    (0.50, 2.0),
]

# Monte Carlo standard error for a probability estimate p:
#   SE = sqrt(p(1-p)/N);  worst case p=0.5 -> SE = 0.5/sqrt(N)
MC_SE_WORST = 0.5 / np.sqrt(N_REPLICATIONS)


# ===========================================================================
# Metric extraction
# ===========================================================================
def get_alarms(result):
    """Extract alarm indices from a detector result dict (defensive)."""
    if isinstance(result, dict):
        for key in ("change_points", "alarms", "alarm_indices", "alarm_times"):
            if key in result:
                return list(result[key])
        raise KeyError(
            f"No alarm key found in detector result; keys={list(result)}")
    return list(result)


def rep_metrics(alarms):
    """Per-replication metrics from a sorted alarm list."""
    alarms = sorted(alarms)
    monitoring = [a for a in alarms if a >= BASELINE_WINDOW]
    pre = [a for a in monitoring if a < T_CP_INDEX]          # false alarms
    post = [a for a in monitoring if a >= T_CP_INDEX]        # detections
    first_mono = monitoring[0] if monitoring else None
    return {
        "detected_any_post": 1 if post else 0,
        "first_alarm_detection": 1 if (first_mono is not None
                                       and first_mono >= T_CP_INDEX) else 0,
        "delay": (post[0] - T_CP_INDEX) if post else np.nan,
        "any_false": 1 if pre else 0,
        "n_false": len(pre),
        "first_mono_alarm": first_mono if first_mono is not None else np.nan,
    }


def summarize(scenario_label, detector, extra, rep_dicts):
    delays = [d["delay"] for d in rep_dicts if not np.isnan(d["delay"])]
    row = {
        "detector": detector,
        "scenario": scenario_label,
        "n_replications": len(rep_dicts),
        "detection_rate": float(np.mean([d["detected_any_post"] for d in rep_dicts])),
        "first_alarm_detection_rate": float(np.mean([d["first_alarm_detection"] for d in rep_dicts])),
        "mean_delay": float(np.mean(delays)) if delays else np.nan,
        "delay_std": float(np.std(delays, ddof=1)) if len(delays) > 1 else 0.0,
        "false_alarm_rate": float(np.mean([d["any_false"] for d in rep_dicts])),
        "mean_n_false": float(np.mean([d["n_false"] for d in rep_dicts])),
    }
    row.update(extra)
    return row


# ===========================================================================
# Evaluation loops (CRN: base noise drawn once per replication)
# ===========================================================================
def evaluate_mean_shift():
    rng = np.random.default_rng(SEED)
    acc = {k: [] for k in MEAN_SHIFTS_K}
    rep_rows = []
    for i in range(N_REPLICATIONS):
        base = rng.normal(0.0, SIGMA0, size=T_LEN)        # CRN base
        for k in MEAN_SHIFTS_K:
            x = base.copy()
            x[T_CP_INDEX:] += -k * SIGMA0
            res = cusum_fixed.detect(x, **CUSUM_FIXED_KWARGS)
            m = rep_metrics(get_alarms(res))
            acc[k].append(m)
            rep_rows.append({"detector": "CUSUM-fixed",
                             "scenario": f"mean_shift_-{k}sigma",
                             "mean_shift_k": k, "rep": i, **m})
    rows = []
    for k in MEAN_SHIFTS_K:
        rows.append(summarize(
            f"mean_shift_-{k}sigma", "CUSUM-fixed",
            {"mean_shift_k": k, "post_change_mean": -k * SIGMA0}, acc[k]))
        r = rows[-1]
        print(f"  [CUSUM-fixed] mean -{k}sigma: power={r['detection_rate']:.3f} "
              f"first-alarm={r['first_alarm_detection_rate']:.3f} "
              f"delay={r['mean_delay']:.1f} FA={r['false_alarm_rate']:.3f}")
    pd.DataFrame(rows).to_csv(
        RESULTS_DIR / "s4_cusum_fixed_meanshift.csv", index=False)
    pd.DataFrame(rep_rows).to_csv(
        RESULTS_DIR / "s4_cusum_fixed_meanshift_replications.csv", index=False)
    return rows


def evaluate_var_shift():
    rng = np.random.default_rng(SEED + 1)
    acc = {r: [] for r in VAR_RATIOS}
    rep_rows = []
    for i in range(N_REPLICATIONS):
        z = rng.normal(0.0, 1.0, size=T_LEN)              # CRN standard normals
        for r in VAR_RATIOS:
            x = np.empty(T_LEN)
            x[:T_CP_INDEX] = z[:T_CP_INDEX] * SIGMA0
            x[T_CP_INDEX:] = z[T_CP_INDEX:] * (r * SIGMA0)
            res = cusum_abs.detect(x, **CUSUM_ABS_KWARGS)
            m = rep_metrics(get_alarms(res))
            acc[r].append(m)
            rep_rows.append({"detector": "CUSUM-abs",
                             "scenario": f"var_shift_{r}x",
                             "var_ratio": r, "rep": i, **m})
    rows = []
    for r in VAR_RATIOS:
        rows.append(summarize(
            f"var_shift_{r}x", "CUSUM-abs",
            {"var_ratio": r, "post_change_sigma": r * SIGMA0}, acc[r]))
        rr = rows[-1]
        print(f"  [CUSUM-abs] var {r}x: power={rr['detection_rate']:.3f} "
              f"first-alarm={rr['first_alarm_detection_rate']:.3f} "
              f"delay={rr['mean_delay']:.1f} FA={rr['false_alarm_rate']:.3f}")
    pd.DataFrame(rows).to_csv(
        RESULTS_DIR / "s4_cusum_abs_varshift.csv", index=False)
    pd.DataFrame(rep_rows).to_csv(
        RESULTS_DIR / "s4_cusum_abs_varshift_replications.csv", index=False)
    return rows


def evaluate_mixed_crisis():
    """Negative mean + variance increase; run BOTH detectors."""
    rng = np.random.default_rng(SEED + 4)
    acc = {(k, r, det): [] for (k, r) in MIXED_SCENARIOS
           for det in ("CUSUM-fixed", "CUSUM-abs")}
    rep_rows = []
    for i in range(N_REPLICATIONS):
        z = rng.normal(0.0, 1.0, size=T_LEN)              # CRN
        for (k, r) in MIXED_SCENARIOS:
            x = np.empty(T_LEN)
            x[:T_CP_INDEX] = z[:T_CP_INDEX] * SIGMA0
            x[T_CP_INDEX:] = z[T_CP_INDEX:] * (r * SIGMA0) - k * SIGMA0
            for det, mod, kw in [
                ("CUSUM-fixed", cusum_fixed, CUSUM_FIXED_KWARGS),
                ("CUSUM-abs", cusum_abs, CUSUM_ABS_KWARGS),
            ]:
                res = mod.detect(x, **kw)
                m = rep_metrics(get_alarms(res))
                acc[(k, r, det)].append(m)
                rep_rows.append({"detector": det,
                                 "scenario": f"mixed_-{k}sig_{r}xvar",
                                 "mean_shift_k": k, "var_ratio": r,
                                 "rep": i, **m})
    rows = []
    for (k, r) in MIXED_SCENARIOS:
        for det in ("CUSUM-fixed", "CUSUM-abs"):
            rows.append(summarize(
                f"mixed_-{k}sig_{r}xvar", det,
                {"mean_shift_k": k, "var_ratio": r}, acc[(k, r, det)]))
            rr = rows[-1]
            print(f"  [{det}] mixed mean-{k}sig var{r}x: "
                  f"power={rr['detection_rate']:.3f} "
                  f"first-alarm={rr['first_alarm_detection_rate']:.3f} "
                  f"delay={rr['mean_delay']:.1f} FA={rr['false_alarm_rate']:.3f}")
    pd.DataFrame(rows).to_csv(
        RESULTS_DIR / "s4_cusum_fixed_mixed_crisis.csv", index=False)
    pd.DataFrame(rep_rows).to_csv(
        RESULTS_DIR / "s4_mixed_crisis_replications.csv", index=False)
    return rows


def evaluate_no_change():
    """False-alarm rate and censored ARL0 under the null for both detectors."""
    rows = []
    rep_rows = []
    for name, detector, kwargs, seed in [
        ("CUSUM-fixed", cusum_fixed, CUSUM_FIXED_KWARGS, SEED + 2),
        ("CUSUM-abs", cusum_abs, CUSUM_ABS_KWARGS, SEED + 3),
    ]:
        rng = np.random.default_rng(seed)
        gaps, any_alarm, censored = [], [], 0
        for i in range(N_REPLICATIONS):
            x = rng.normal(0.0, SIGMA0, size=T_LEN)
            alarms = [a for a in get_alarms(detector.detect(x, **kwargs))
                      if a >= BASELINE_WINDOW]
            has = 1 if alarms else 0
            any_alarm.append(has)
            gap = (alarms[0] - BASELINE_WINDOW) if alarms \
                else (T_LEN - BASELINE_WINDOW)   # censored
            if not alarms:
                censored += 1
            gaps.append(gap)
            rep_rows.append({"detector": name, "rep": i,
                             "any_false_alarm": has,
                             "first_alarm_gap": gap,
                             "censored": 0 if alarms else 1})
        rows.append({
            "detector": name,
            "scenario": "no_change",
            "n_replications": N_REPLICATIONS,
            "false_alarm_rate": float(np.mean(any_alarm)),
            "ARL0_censored_mean": float(np.mean(gaps)),
            "censoring_rate": censored / N_REPLICATIONS,
            "monitoring_length": T_LEN - BASELINE_WINDOW,
        })
        print(f"  [{name}] no-change: FA_rate={np.mean(any_alarm):.3f} "
              f"censored-ARL0>={np.mean(gaps):.0f} "
              f"(censored {censored}/{N_REPLICATIONS})")
    pd.DataFrame(rows).to_csv(
        RESULTS_DIR / "s4_nochange_falsealarm.csv", index=False)
    pd.DataFrame(rep_rows).to_csv(
        RESULTS_DIR / "s4_nochange_falsealarm_replications.csv", index=False)
    return rows


def main():
    print("=" * 70)
    print("Section 4: CUSUM synthetic detection-power / false-alarm benchmark")
    print(f"  N={N_REPLICATIONS}, T={T_LEN}, t*(index)={T_CP_INDEX}, "
          f"baseline_window={BASELINE_WINDOW}, sigma0={SIGMA0}")
    print(f"  Monte Carlo SE (worst case, p=0.5): {MC_SE_WORST:.4f}")
    print("=" * 70)
    print("\n[1/4] CUSUM-fixed on negative mean-shift series:")
    evaluate_mean_shift()
    print("\n[2/4] CUSUM-abs on variance-shift series:")
    evaluate_var_shift()
    print("\n[3/4] Mixed crisis (negative mean + variance increase):")
    evaluate_mixed_crisis()
    print("\n[4/4] No-change (null) false-alarm characterization:")
    evaluate_no_change()
    print("\nDone. Summary + replication-level CSVs in results/s4_*.csv")
    print("Paste the summary CSVs back to write Section 4.")


if __name__ == "__main__":
    main()