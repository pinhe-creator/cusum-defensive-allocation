"""CUSUM hyperparameter grid search.

Sweeps (threshold, drift) on fixed (window, cooldown). Evaluates on
all 6 DGPs with auto-inferred matching mode. Identifies:
    - Per-DGP best config (F1-based for non-null, FA-based for null)
    - Overall robust config: best mean F1 rank across non-null DGPs,
      restricted to configs with bounded false alarms on null DGPs.

Outputs:
    results/cusum_grid_raw.csv                  per-(config, dgp, rep)
    results/cusum_grid_summary_by_config_dgp.csv  aggregated by (cfg, dgp)
    results/cusum_grid_per_dgp.csv              best config per DGP
    results/cusum_grid_robust_rank.csv          overall robust ranking
    results/cusum_grid_summary.png              F1 heatmaps per DGP
"""
import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulators import (
    dgp0_gaussian_null, dgp0_student_t_null,
    dgp1_gaussian_mean, dgp2_gaussian_variance,
    dgp3_student_t, dgp4_garch_switching,
)
from metrics.detection import evaluate_result
from algorithms import cusum


# ====================================================================
# Configuration
# ====================================================================

N_REPLICATIONS = 20
N_OBS = 1000
DETECTION_WINDOW = 20
KAPPA = 10

THRESHOLD_GRID = [3.0, 5.0, 8.0, 10.0, 15.0]
DRIFT_GRID = [0.0, 0.1, 0.25, 0.5]
FIXED_WINDOW = 100
FIXED_COOLDOWN = 60

# Risk-aware filter: configs exceeding this FA/year on null DGPs are
# excluded from robust ranking (so we don't pick configs that look
# good on non-null DGPs only because they fire on everything).
MAX_FA_PER_YEAR_NULL = 1.0

NULL_DGPS = ["dgp0_gaussian", "dgp0_student_t"]
NON_NULL_DGPS = ["dgp1_mean", "dgp2_variance",
                 "dgp3_student_t", "dgp4_garch"]

DGPS = [
    ("dgp0_gaussian",   dgp0_gaussian_null,    {}),
    ("dgp0_student_t",  dgp0_student_t_null,   {}),
    ("dgp1_mean",       dgp1_gaussian_mean,    {}),
    ("dgp2_variance",   dgp2_gaussian_variance, {}),
    ("dgp3_student_t",  dgp3_student_t,        {}),
    ("dgp4_garch",      dgp4_garch_switching,  {}),
]


# ====================================================================
# Main experiment
# ====================================================================

def main():
    os.makedirs("results", exist_ok=True)
    rows = []
    t_start = time.time()

    n_configs = len(THRESHOLD_GRID) * len(DRIFT_GRID)
    total_runs = n_configs * len(DGPS) * N_REPLICATIONS
    run_count = 0

    print(f"Sweeping {n_configs} configs x {len(DGPS)} DGPs x "
          f"{N_REPLICATIONS} reps = {total_runs} runs")
    print()

    for threshold in THRESHOLD_GRID:
        for drift in DRIFT_GRID:
            kwargs = {
                "threshold": threshold,
                "drift": drift,
                "window": FIXED_WINDOW,
                "cooldown": FIXED_COOLDOWN,
            }

            for dgp_name, dgp_fn, dgp_kwargs in DGPS:
                for rep in range(N_REPLICATIONS):
                    series, true_cps = dgp_fn(
                        n_obs=N_OBS, seed=rep, **dgp_kwargs
                    )
                    run_count += 1

                    try:
                        result = cusum.detect(series, **kwargs)
                    except Exception as e:
                        print(f"  [error] threshold={threshold} "
                              f"drift={drift} {dgp_name} rep={rep}: {e}")
                        continue

                    metrics = evaluate_result(
                        result, true_cps, n_obs=N_OBS,
                        window=DETECTION_WINDOW, kappa=KAPPA,
                    )
                    metrics["runtime_sec"] = result.get("runtime_sec", np.nan)
                    metrics["threshold"] = threshold
                    metrics["drift"] = drift
                    metrics["dgp"] = dgp_name
                    metrics["replication"] = rep
                    rows.append(metrics)

                    if run_count % 240 == 0:
                        elapsed = time.time() - t_start
                        pct = 100 * run_count / total_runs
                        print(f"  progress: {run_count}/{total_runs} "
                              f"({pct:.1f}%) elapsed={elapsed:.1f}s")

    df_raw = pd.DataFrame(rows)
    df_raw.to_csv("results/cusum_grid_raw.csv", index=False)
    print(f"\nSaved raw: results/cusum_grid_raw.csv ({len(df_raw)} rows)")

    # ----------------------------------------------------------------
    # Aggregate by (threshold, drift, dgp)
    # ----------------------------------------------------------------
    agg = (
        df_raw.groupby(["threshold", "drift", "dgp"])
              .agg({"f1": "mean", "precision": "mean", "recall": "mean",
                    "mean_dd": "mean", "uwdl": "mean",
                    "false_alarms_per_year": "mean",
                    "runtime_sec": "mean"})
              .reset_index()
              .round(3)
    )
    agg.to_csv("results/cusum_grid_summary_by_config_dgp.csv", index=False)
    print(f"Saved grid summary: "
          f"results/cusum_grid_summary_by_config_dgp.csv ({len(agg)} rows)")

    # ----------------------------------------------------------------
    # Per-DGP best config
    # ----------------------------------------------------------------
    best_per_dgp = []
    for dgp in NON_NULL_DGPS:
        sub = agg[agg["dgp"] == dgp].sort_values("f1", ascending=False)
        best = sub.iloc[0]
        best_per_dgp.append({
            "dgp": dgp,
            "criterion": "max F1",
            "threshold": best["threshold"],
            "drift": best["drift"],
            "f1": best["f1"],
            "precision": best["precision"],
            "recall": best["recall"],
            "false_alarms_per_year": best["false_alarms_per_year"],
        })
    for dgp in NULL_DGPS:
        sub = agg[agg["dgp"] == dgp].sort_values(
            "false_alarms_per_year", ascending=True
        )
        best = sub.iloc[0]
        # For null DGPs, F1/precision/recall are not informative —
        # omit them to avoid misinterpretation.
        best_per_dgp.append({
            "dgp": dgp,
            "criterion": "min false_alarms_per_year",
            "threshold": best["threshold"],
            "drift": best["drift"],
            "f1": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "false_alarms_per_year": best["false_alarms_per_year"],
        })
    df_best = pd.DataFrame(best_per_dgp)
    df_best.to_csv("results/cusum_grid_per_dgp.csv", index=False)

    print("\n" + "=" * 80)
    print("BEST CONFIG PER DGP")
    print("=" * 80)
    print(df_best.to_string(index=False))

    # ----------------------------------------------------------------
    # Robust config: best mean F1 rank across non-null DGPs,
    # restricted to configs with bounded FA on null DGPs.
    # ----------------------------------------------------------------
    # Step 1: Compute mean FA on null DGPs per config.
    null_fa = (
        agg[agg["dgp"].isin(NULL_DGPS)]
        .groupby(["threshold", "drift"])["false_alarms_per_year"]
        .mean()
        .reset_index()
        .rename(columns={"false_alarms_per_year": "null_fa_mean"})
    )

    # Step 2: Pivot F1 on non-null DGPs.
    pivot_f1 = agg[agg["dgp"].isin(NON_NULL_DGPS)].pivot_table(
        index=["threshold", "drift"], columns="dgp", values="f1"
    )
    # Drop configs that failed on any non-null DGP (NaN).
    pivot_f1_clean = pivot_f1.dropna(how="any")

    # Step 3: Rank F1 within each non-null DGP (lower rank = better).
    ranks = pivot_f1_clean.rank(ascending=False, method="average")
    mean_rank = ranks.mean(axis=1)

    # Step 4: Merge with null FA, filter, and rank.
    df_rank = mean_rank.reset_index()
    df_rank.columns = ["threshold", "drift", "mean_rank_non_null"]
    df_rank = df_rank.merge(pivot_f1_clean.reset_index(),
                            on=["threshold", "drift"], how="left")
    df_rank = df_rank.merge(null_fa, on=["threshold", "drift"], how="left")
    df_rank["passes_null_filter"] = (
        df_rank["null_fa_mean"] <= MAX_FA_PER_YEAR_NULL
    )
    df_rank = df_rank.sort_values("mean_rank_non_null")
    df_rank.to_csv("results/cusum_grid_robust_rank.csv", index=False)
    print(f"\nSaved robust rank: results/cusum_grid_robust_rank.csv")

    print("\n" + "=" * 80)
    print(f"TOP-5 ROBUST CONFIGS by non-null F1 rank")
    print(f"(passes_null_filter=True means null_fa_mean <= "
          f"{MAX_FA_PER_YEAR_NULL})")
    print("=" * 80)
    print(df_rank.head(5).to_string(index=False))

    # Top-1 config that ALSO passes null filter
    passing = df_rank[df_rank["passes_null_filter"]]
    print("\n" + "=" * 80)
    if len(passing) == 0:
        print("WARNING: No config passes the null-FA filter.")
        print(f"Consider relaxing MAX_FA_PER_YEAR_NULL "
              f"(currently {MAX_FA_PER_YEAR_NULL})")
    else:
        print(f"TOP-3 ROBUST + RISK-AWARE CONFIGS "
              f"(passes null filter, ranked by non-null F1)")
        print("=" * 80)
        print(passing.head(3).to_string(index=False))

    # ----------------------------------------------------------------
    # Visualization: F1 heatmap per non-null DGP
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, dgp in zip(axes.flat, NON_NULL_DGPS):
        sub = agg[agg["dgp"] == dgp].pivot(
            index="threshold", columns="drift", values="f1"
        )
        im = ax.imshow(sub.values, cmap="RdYlGn", vmin=0, vmax=1,
                       aspect="auto")
        ax.set_xticks(range(len(sub.columns)))
        ax.set_xticklabels([f"{d:.2f}" for d in sub.columns])
        ax.set_yticks(range(len(sub.index)))
        ax.set_yticklabels([f"{t:.0f}" for t in sub.index])
        ax.set_xlabel("drift")
        ax.set_ylabel("threshold")
        ax.set_title(f"{dgp}: F1")
        for i in range(sub.shape[0]):
            for j in range(sub.shape[1]):
                val = sub.values[i, j]
                color = "white" if val < 0.4 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=9)
        plt.colorbar(im, ax=ax)
    plt.suptitle(f"CUSUM F1 vs (threshold, drift), "
                 f"window={FIXED_WINDOW}, cooldown={FIXED_COOLDOWN}, "
                 f"N={N_REPLICATIONS}")
    plt.tight_layout()
    plt.savefig("results/cusum_grid_summary.png",
                dpi=130, bbox_inches="tight")
    print("\nSaved heatmaps: results/cusum_grid_summary.png")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
