"""
R2 Experiment: OAT parameter sensitivity sweep for CUSUM-fixed
====================================================================

Purpose:
    Address the QF reviewer concern that the CUSUM-fixed Sharpe and
    drawdown improvements documented in Section 6 might be an artifact
    of a specific default hyperparameter choice. This script runs a
    one-at-a-time (OAT) sensitivity sweep over four detector
    hyperparameters at six levels each.

Design:
    24 evaluated rows = 21 unique parameter tuples + 3 duplicate
    evaluations of the default tuple (8.0, 0.50, 252, 60). The default
    appears once in each of the four OAT sweeps (one of the four
    counts as the unique evaluation, the other three are exact
    duplicates that serve as a determinism sanity check).

    This is a one-at-a-time sensitivity sweep, NOT a full factorial
    grid (which would be 6^4 = 1296 configurations). OAT cannot detect
    parameter interactions; it only verifies that the headline result
    is not driven by a single arbitrary default along any one axis.

Sweep dimensions and levels (default in bold, at index 2):
    threshold        in [4, 6, *8*, 10, 12, 16]            0.5x-2x default
    drift            in [0.25, 0.40, *0.50*, 0.60, 0.75, 1.00]  0.5x-2x default
    baseline_window  in [63, 126, *252*, 378, 504, 756]    0.25x-3x default
    cooldown         in [20, 40, *60*, 90, 120, 180]       0.33x-3x default

    For each sweep, the other three parameters are held at default.

Scope:
    This experiment varies detector hyperparameters only. The portfolio
    risk-off holding rule (RISK_OFF_HOLD = 20 trading days) is kept
    fixed at the headline value used in Section 6. Allocation-policy
    sensitivity is a separate experiment.

Return convention:
    R2 does not reconstruct returns manually. It imports
    load_aligned_data(), backtest(), and compute_metrics() directly
    from experiments_portfolio_v2.py, so the return convention (log
    returns) and the backtest engine are identical to the headline
    portfolio_v2 results.

Paper claim supported:
    "Holding the other parameters at their defaults, the CUSUM-fixed
    portfolio Sharpe and drawdown metrics are stable across one-at-a-
    time perturbations on each of the four detector hyperparameters.
    Across 21 unique configurations evaluated at TC = 10 bps, Sharpe
    remains within [X, Y] and max drawdown remains within [A%, B%],
    compared to the default values 0.581 and -33.22% and the static
    60/40 benchmark of 0.518 and -36.13%."

Outputs:
    results/r2_oat_sensitivity.csv          - 24 rows x ~21 columns
                                              (includes full detector
                                              kwargs per row)
    results/r2_oat_sensitivity_panel.png    - 2x4 panel (Sharpe, MaxDD)
                                              x (4 sweep axes), dpi=160
    results/r2_oat_sensitivity_panel.pdf    - same panel as vector PDF
                                              for LaTeX inclusion

Usage:
    cd /Users/chenpinhe/Downloads/cpd-finance-benchmark/
    python experiments_r2_param_grid.py

Author: Pinhe Chen, Fort Hays State University
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from algorithms import cusum_fixed
from experiments_portfolio_v2 import (
    load_aligned_data,
    signal_static,
    state_from_alarms,
    backtest,
    compute_metrics,
    crisis_subsample_loss,
    RISK_OFF_HOLD,
    CUSUM_FIXED_KWARGS,
    CRISIS_WINDOWS,
)


# =============================================================================
# Configuration
# =============================================================================
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TC_BPS = 10                     # Matches paper headline Sharpe 0.581
DEFAULT_PARAMS = dict(CUSUM_FIXED_KWARGS)
# Expect: {threshold: 8.0, drift: 0.5, baseline_window: 252,
#          cooldown: 60, side: "negative"}

# OAT sweep grid: each axis has 6 levels, with the default at index 2
SWEEP_GRID: Dict[str, List] = {
    "threshold":       [4.0, 6.0, 8.0, 10.0, 12.0, 16.0],
    "drift":           [0.25, 0.40, 0.50, 0.60, 0.75, 1.00],
    "baseline_window": [63, 126, 252, 378, 504, 756],
    "cooldown":        [20, 40, 60, 90, 120, 180],
}

# Expected default Sharpe at TC=10 bps. Loaded dynamically from
# portfolio_v2_metrics.csv if available; falls back to hardcoded value.
EXPECTED_DEFAULT_SHARPE_FALLBACK = 0.5812
SANITY_TOL = 0.005
METRICS_CSV = RESULTS_DIR / "portfolio_v2_metrics.csv"

OUT_CSV = RESULTS_DIR / "r2_oat_sensitivity.csv"
OUT_PNG = RESULTS_DIR / "r2_oat_sensitivity_panel.png"
OUT_PDF = RESULTS_DIR / "r2_oat_sensitivity_panel.pdf"


def load_expected_default_sharpe() -> tuple:
    """
    Load CUSUM-fixed Sharpe at TC=10 bps from portfolio_v2_metrics.csv.
    Falls back to EXPECTED_DEFAULT_SHARPE_FALLBACK if the file is
    missing, fails to parse, or does not contain a CUSUM-fixed row.

    Returns
    -------
    (sharpe_value, source_label) : tuple[float, str]
        source_label is either "portfolio_v2_metrics.csv" or
        "hardcoded fallback". Used for audit-trail console logging.
    """
    if not METRICS_CSV.exists():
        return EXPECTED_DEFAULT_SHARPE_FALLBACK, "hardcoded fallback"
    try:
        df = pd.read_csv(METRICS_CSV)
        if "tc_bps" in df.columns:
            df = df[df["tc_bps"] == TC_BPS]
        # Find strategy + sharpe columns
        strat_col = next(
            (c for c in df.columns
             if c.lower() in ("strategy", "strategy_name", "name")),
            None,
        )
        sharpe_col = next(
            (c for c in df.columns
             if c.lower() in ("sharpe", "sharpe_ratio")),
            None,
        )
        if strat_col is None or sharpe_col is None:
            return EXPECTED_DEFAULT_SHARPE_FALLBACK, "hardcoded fallback"
        # Match CUSUM-fixed (allow variants like "CUSUM-fixed", "CUSUM_fixed")
        # Use regex=False for both replace and contains to avoid surprise
        # regex behavior across pandas versions.
        mask = (
            df[strat_col].astype(str)
            .str.lower()
            .str.replace("_", "-", regex=False)
            .str.contains("cusum-fixed", regex=False)
        )
        rows = df[mask]
        if len(rows) == 0:
            return EXPECTED_DEFAULT_SHARPE_FALLBACK, "hardcoded fallback"
        return float(rows[sharpe_col].iloc[0]), "portfolio_v2_metrics.csv"
    except Exception:
        return EXPECTED_DEFAULT_SHARPE_FALLBACK, "hardcoded fallback"


# =============================================================================
# One-config evaluation
# =============================================================================
def run_one_config(
    spx_ret: np.ndarray,
    spy_ret: np.ndarray,
    ief_ret: np.ndarray,
    dates: pd.DatetimeIndex,
    params: Dict,
    tc_bps: float,
) -> Dict:
    """
    Run a full CUSUM-fixed detection + backtest + metrics for one
    parameter configuration. Returns a flat dict of metrics including
    crisis-period drawdowns.
    """
    result = cusum_fixed.detect(spx_ret, **params)
    alarms = result["change_points"]

    state = state_from_alarms(alarms, len(spx_ret), hold=RISK_OFF_HOLD)
    bt = backtest(spy_ret, ief_ret, state, tc_bps=tc_bps)

    # Defensive: portfolio_returns and dates must align for crisis_subsample
    if len(bt["portfolio_returns"]) != len(dates):
        raise ValueError(
            f"portfolio_returns length ({len(bt['portfolio_returns'])}) "
            f"!= dates length ({len(dates)}); backtest convention changed?"
        )

    metrics = compute_metrics(bt["portfolio_returns"])

    risk_off_frac = float(state.mean())
    n_alarms = int(len(alarms))

    # Crisis-period drawdowns
    crisis_dd = {}
    for label, (start, end) in CRISIS_WINDOWS.items():
        cl = crisis_subsample_loss(bt["portfolio_returns"], dates, start, end)
        # Sanitize label for column name: "GFC 2008-2009" -> "GFC_2008_2009"
        col_key = label.replace(" ", "_").replace("-", "_")
        crisis_dd[f"crisis_dd_{col_key}"] = cl["crisis_max_dd"]

    return {
        **metrics,
        "n_alarms": n_alarms,
        "risk_off_frac": risk_off_frac,
        **crisis_dd,
    }


# =============================================================================
# Main
# =============================================================================
def main():
    t0 = time.time()
    print("=" * 72)
    print("R2: OAT parameter sensitivity sweep for CUSUM-fixed")
    print("=" * 72)
    print(f"TC = {TC_BPS} bps")
    print(f"Default params: {DEFAULT_PARAMS}")
    print(f"Sweep axes:     {list(SWEEP_GRID.keys())}")
    print(f"Levels per axis: 6 (default at index 2)")
    print(f"Total configs:  {sum(len(v) for v in SWEEP_GRID.values())}")
    print()

    # ---- Load data ----
    df = load_aligned_data()
    n = len(df)
    spy_ret = df["spy_ret"].values
    ief_ret = df["ief_ret"].values
    spx_ret = df["spx_ret"].values
    dates = df.index

    # ---- Static baseline ----
    print("[Baseline] Computing Static 60/40 reference...")
    static_state = signal_static(n)
    static_bt = backtest(spy_ret, ief_ret, static_state, tc_bps=TC_BPS)
    static_metrics = compute_metrics(static_bt["portfolio_returns"])
    static_sharpe = static_metrics["sharpe"]
    static_maxdd = static_metrics["max_drawdown"]
    print(f"  Static 60/40: Sharpe = {static_sharpe:.4f}, "
          f"MaxDD = {static_maxdd:.4f}")

    # ---- Sanity check on default config ----
    expected_default_sharpe, expected_source = load_expected_default_sharpe()
    print(f"\n[Sanity] Verifying default config matches expected Sharpe "
          f"(source: {expected_source})...")
    default_res = run_one_config(spx_ret, spy_ret, ief_ret, dates,
                                  DEFAULT_PARAMS, TC_BPS)
    default_delta = default_res["sharpe"] - expected_default_sharpe
    sanity_ok = abs(default_delta) < SANITY_TOL
    status = "OK" if sanity_ok else "MISMATCH"
    print(f"  Default: Sharpe = {default_res['sharpe']:.4f}  "
          f"(expected {expected_default_sharpe:.4f}, delta {default_delta:+.4f}, "
          f"{status})")
    if not sanity_ok:
        print(f"  WARNING: default config Sharpe differs from expected by "
              f"more than {SANITY_TOL}. Subsequent results may not align with")
        print(f"  portfolio_v2_metrics.csv. Investigate before paper inclusion.")

    # ---- Sweep ----
    print(f"\n[Sweep] Running 24 OAT evaluations "
          f"(4 axes x 6 levels; 21 unique tuples + 3 duplicate defaults)...")
    rows = []
    for axis_name, levels in SWEEP_GRID.items():
        default_val_for_axis = DEFAULT_PARAMS[axis_name]
        for level in levels:
            params = dict(DEFAULT_PARAMS)
            params[axis_name] = level
            is_default = (level == default_val_for_axis)

            res = run_one_config(spx_ret, spy_ret, ief_ret, dates,
                                  params, TC_BPS)

            row = {
                # Sweep metadata
                "sweep_axis":   axis_name,
                "param_value":  level,
                "is_default":   is_default,
                # Full detector parameter tuple (self-contained per row)
                "threshold":       params["threshold"],
                "drift":           params["drift"],
                "baseline_window": params["baseline_window"],
                "cooldown":        params["cooldown"],
                "side":            params.get("side", ""),
                # Backtest setting
                "tc_bps":       TC_BPS,
                # Metrics
                **res,
                # Deltas vs static baseline
                "delta_sharpe_vs_static":
                    res["sharpe"] - static_sharpe,
                "delta_maxdd_vs_static":
                    res["max_drawdown"] - static_maxdd,
            }
            rows.append(row)

            marker = "  <-- DEFAULT" if is_default else ""
            print(f"  {axis_name:<18s} = {str(level):<7s}  "
                  f"Sharpe={res['sharpe']:+.4f}  "
                  f"MaxDD={res['max_drawdown']:+.4f}  "
                  f"n_alarms={res['n_alarms']:>3d}  "
                  f"riskoff={res['risk_off_frac']:.3f}{marker}")

    # ---- Save CSV ----
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\n[Save] {OUT_CSV} ({len(df_out)} rows, {len(df_out.columns)} cols)")

    # ---- Unique-tuple view for stability summary ----
    # The default tuple (8.0, 0.50, 252, 60) appears 4 times across the
    # 4 OAT sweeps. For unbiased stability statistics we deduplicate to
    # 21 unique parameter tuples. The full df_out is still used for the
    # determinism check (which requires the 4 default duplicates).
    df_unique = df_out.drop_duplicates(
        subset=["threshold", "drift", "baseline_window", "cooldown", "side"]
    ).copy()
    print(f"  Unique parameter tuples: {len(df_unique)} (expected 21)")

    # ---- Determinism check on the default duplicates ----
    print(f"\n[Determinism] Verifying default-tuple duplicates produce "
          f"identical metrics...")
    default_rows = df_out[df_out["is_default"]].copy()
    print(f"  Found {len(default_rows)} default rows "
          f"(expected 4 = 1 unique + 3 duplicate).")
    if len(default_rows) >= 2:
        sharpe_range = default_rows["sharpe"].max() - default_rows["sharpe"].min()
        maxdd_range = (default_rows["max_drawdown"].max()
                       - default_rows["max_drawdown"].min())
        print(f"  Sharpe across defaults: range = {sharpe_range:.2e} "
              f"(should be 0 for determinism)")
        print(f"  MaxDD across defaults:  range = {maxdd_range:.2e}")
        if sharpe_range > 1e-10 or maxdd_range > 1e-10:
            print(f"  WARNING: detector is not deterministic across calls; "
                  f"investigate seed handling.")

    # ---- Stability summary (computed on unique tuples) ----
    print(f"\n[Stability summary]  (computed on {len(df_unique)} unique "
          f"parameter tuples, default counted once)")
    sharpe_min = df_unique["sharpe"].min()
    sharpe_max = df_unique["sharpe"].max()
    sharpe_median = df_unique["sharpe"].median()
    maxdd_min = df_unique["max_drawdown"].min()
    maxdd_max = df_unique["max_drawdown"].max()
    maxdd_median = df_unique["max_drawdown"].median()
    n_alarms_min = df_unique["n_alarms"].min()
    n_alarms_max = df_unique["n_alarms"].max()
    riskoff_min = df_unique["risk_off_frac"].min()
    riskoff_max = df_unique["risk_off_frac"].max()

    print(f"  Sharpe        in [{sharpe_min:.4f}, {sharpe_max:.4f}]  "
          f"median = {sharpe_median:.4f}  "
          f"width = {sharpe_max - sharpe_min:.4f}")
    print(f"  MaxDD         in [{maxdd_min:.4f}, {maxdd_max:.4f}]  "
          f"median = {maxdd_median:.4f}  "
          f"width = {maxdd_max - maxdd_min:.4f}")
    print(f"  n_alarms      in [{n_alarms_min}, {n_alarms_max}]")
    print(f"  risk_off_frac in [{riskoff_min:.4f}, {riskoff_max:.4f}]")
    print()
    print(f"  Anchor:       Static Sharpe = {static_sharpe:.4f}, "
          f"Static MaxDD = {static_maxdd:.4f}")
    print(f"  Default:      CUSUM-fixed Sharpe = {default_res['sharpe']:.4f}, "
          f"MaxDD = {default_res['max_drawdown']:.4f}")
    print()

    n_better_sharpe = int((df_unique["sharpe"] > static_sharpe).sum())
    n_less_severe_maxdd = int((df_unique["max_drawdown"] > static_maxdd).sum())
    n_total = len(df_unique)
    print(f"  Configs with Sharpe > Static: {n_better_sharpe}/{n_total} "
          f"({100*n_better_sharpe/n_total:.0f}%)")
    print(f"  Configs with less severe MaxDD than Static: "
          f"{n_less_severe_maxdd}/{n_total} "
          f"({100*n_less_severe_maxdd/n_total:.0f}%)")

    # ---- Worst-case sweep per axis ----
    print(f"\n[Worst-case per axis]")
    for axis_name in SWEEP_GRID.keys():
        sub = df_out[df_out["sweep_axis"] == axis_name]
        wc_sharpe_idx = sub["sharpe"].idxmin()
        wc_maxdd_idx = sub["max_drawdown"].idxmin()
        print(f"  {axis_name:<18s}: "
              f"worst Sharpe = {sub.loc[wc_sharpe_idx, 'sharpe']:+.4f} "
              f"at {axis_name} = {sub.loc[wc_sharpe_idx, 'param_value']}, "
              f"worst MaxDD = {sub.loc[wc_maxdd_idx, 'max_drawdown']:+.4f} "
              f"at {axis_name} = {sub.loc[wc_maxdd_idx, 'param_value']}")

    # ---- Plot 2x4 panel ----
    print(f"\n[Plot] Generating 2x4 panel figure...")
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))

    axis_titles = {
        "threshold":       "Threshold h",
        "drift":           "Drift k",
        "baseline_window": "Baseline window (days)",
        "cooldown":        "Cooldown (days)",
    }

    for col_idx, axis_name in enumerate(SWEEP_GRID.keys()):
        sub = df_out[df_out["sweep_axis"] == axis_name].copy()
        sub = sub.sort_values("param_value")
        x = sub["param_value"].values
        default_val = DEFAULT_PARAMS[axis_name]

        # ---- Row 0: Sharpe ----
        ax = axes[0, col_idx]
        ax.plot(x, sub["sharpe"].values, 'o-', color='C0',
                linewidth=1.5, markersize=6, label='CUSUM-fixed')
        ax.axhline(static_sharpe, color='gray', linestyle='--',
                   linewidth=1, alpha=0.8,
                   label=f"Static = {static_sharpe:.3f}")
        ax.axvline(default_val, color='red', linestyle=':',
                   linewidth=1, alpha=0.6,
                   label=f"default = {default_val}")
        ax.set_xlabel(axis_titles[axis_name])
        if col_idx == 0:
            ax.set_ylabel("Annualized Sharpe ratio")
        ax.set_title(f"Sharpe vs {axis_titles[axis_name]}")
        ax.grid(alpha=0.3)
        if col_idx == 0:
            ax.legend(fontsize=8, loc="best")

        # ---- Row 1: MaxDD ----
        ax = axes[1, col_idx]
        ax.plot(x, sub["max_drawdown"].values, 's-', color='C3',
                linewidth=1.5, markersize=6, label='CUSUM-fixed')
        ax.axhline(static_maxdd, color='gray', linestyle='--',
                   linewidth=1, alpha=0.8,
                   label=f"Static = {static_maxdd:.3f}")
        ax.axvline(default_val, color='red', linestyle=':',
                   linewidth=1, alpha=0.6)
        ax.set_xlabel(axis_titles[axis_name])
        if col_idx == 0:
            ax.set_ylabel("Maximum drawdown")
        ax.set_title(f"MaxDD vs {axis_titles[axis_name]}")
        ax.grid(alpha=0.3)
        if col_idx == 0:
            ax.legend(fontsize=8, loc="best")

    plt.suptitle(
        f"R2: OAT Parameter Sensitivity for CUSUM-fixed "
        f"(TC = {TC_BPS} bps, 21 unique tuples + 3 duplicate defaults)",
        fontsize=13, y=1.00,
    )
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    plt.savefig(OUT_PDF, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT_PNG}")
    print(f"  Saved: {OUT_PDF}")

    print(f"\n{'=' * 72}")
    print(f"R2 complete. Total time: {time.time() - t0:.1f}s")
    print(f"Outputs: {OUT_CSV.name}, {OUT_PNG.name}, {OUT_PDF.name}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
