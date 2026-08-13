"""Stage 1 mini-experiment: 2 algorithms x 6 DGPs benchmark.

Generates N independent series per DGP, runs each algorithm,
evaluates with auto-inferred matching mode, and aggregates
into an (algorithm x DGP) results table.

Outputs:
    results/stage1_raw.csv     - per-replication results
    results/stage1_summary.csv - aggregated by (algorithm, DGP)
    results/stage1_heatmap.png - F1 heatmap (non-null DGPs only)
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
from algorithms import pelt_rbf, pelt_l2, cusum


# ====================================================================
# Configuration
# ====================================================================

N_REPLICATIONS = 20    # Small first run; scale up to 1000 later.
N_OBS = 1000
DETECTION_WINDOW = 20
KAPPA = 10

# DGP roster: (name, function, kwargs)
DGPS = [
    ("dgp0_gaussian",   dgp0_gaussian_null,    {}),
    ("dgp0_student_t",  dgp0_student_t_null,   {}),
    ("dgp1_mean",       dgp1_gaussian_mean,    {}),
    ("dgp2_variance",   dgp2_gaussian_variance, {}),
    ("dgp3_student_t",  dgp3_student_t,        {}),
    ("dgp4_garch",      dgp4_garch_switching,  {}),
]

# Algorithm roster: (name, module, kwargs)
ALGOS = [
    ("PELT-RBF",  pelt_rbf,  {"pen": 5, "min_size": 30, "jump": 5}),
    ("PELT-L2",   pelt_l2,   {"pen": 5, "min_size": 30, "jump": 5}),
    ("CUSUM",     cusum,     {"threshold": 8.0, "drift": 0.50,
                              "window": 100, "cooldown": 60}),
]


# ====================================================================
# Main experiment
# ====================================================================

def main():
    os.makedirs("results", exist_ok=True)
    rows = []
    t_start = time.time()

    total_runs = len(DGPS) * len(ALGOS) * N_REPLICATIONS
    run_count = 0

    for dgp_name, dgp_fn, dgp_kwargs in DGPS:
        for rep in range(N_REPLICATIONS):
            # Explicitly pass n_obs so future changes to N_OBS propagate.
            series, true_cps = dgp_fn(n_obs=N_OBS, seed=rep, **dgp_kwargs)

            for algo_name, algo_module, algo_kwargs in ALGOS:
                run_count += 1

                # Run algorithm; catch errors so one failure does not
                # kill the whole benchmark.
                try:
                    result = algo_module.detect(series, **algo_kwargs)
                except Exception as e:
                    print(f"  [error] {algo_name} on {dgp_name} "
                          f"rep={rep}: {e}")
                    continue

                # evaluate_result auto-infers matching mode from
                # algorithm metadata and already includes runtime_sec.
                metrics = evaluate_result(
                    result, true_cps, n_obs=N_OBS,
                    window=DETECTION_WINDOW, kappa=KAPPA,
                )
                metrics["n_detected_raw"] = len(
                    result.get("change_points", [])
                )
                metrics["algorithm"] = algo_name
                metrics["dgp"] = dgp_name
                metrics["replication"] = rep
                rows.append(metrics)

                if run_count % 20 == 0:
                    elapsed = time.time() - t_start
                    pct = 100 * run_count / total_runs
                    print(f"  progress: {run_count}/{total_runs} "
                          f"({pct:.1f}%) elapsed={elapsed:.1f}s")

    df_raw = pd.DataFrame(rows)
    df_raw.to_csv("results/stage1_raw.csv", index=False)
    print(f"\nSaved raw: results/stage1_raw.csv ({len(df_raw)} rows)")

    # ----------------------------------------------------------------
    # Aggregate by (algorithm, dgp)
    # ----------------------------------------------------------------
    agg_funcs = {
        "f1": "mean",
        "precision": "mean",
        "recall": "mean",
        "mean_dd": "mean",
        "uwdl": "mean",
        "false_alarms_per_year": "mean",
        "runtime_sec": "mean",
        "tp": "sum",
        "fp": "sum",
        "fn": "sum",
    }
    # Defensive: include matching mode only if it was recorded.
    if "matching_mode_inferred" in df_raw.columns:
        agg_funcs["matching_mode_inferred"] = "first"

    df_summary = (
        df_raw.groupby(["algorithm", "dgp"])
              .agg(agg_funcs)
              .reset_index()
              .round(3)
    )
    df_summary.to_csv("results/stage1_summary.csv", index=False)
    print(f"Saved summary: results/stage1_summary.csv "
          f"({len(df_summary)} rows)")

    print("\n" + "=" * 80)
    print("STAGE 1 SUMMARY (mean over replications)")
    print("=" * 80)
    print(df_summary.to_string(index=False))

    # ----------------------------------------------------------------
    # F1 heatmap (non-null DGPs only; F1 not meaningful for null DGPs)
    # ----------------------------------------------------------------
    non_null = ["dgp1_mean", "dgp2_variance",
                "dgp3_student_t", "dgp4_garch"]
    df_heat = df_summary[df_summary["dgp"].isin(non_null)]
    pivot_f1 = df_heat.pivot(index="algorithm", columns="dgp", values="f1")

    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(pivot_f1.values, cmap="RdYlGn", vmin=0, vmax=1,
                   aspect="auto")
    ax.set_xticks(range(len(pivot_f1.columns)))
    ax.set_xticklabels(pivot_f1.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot_f1.index)))
    ax.set_yticklabels(pivot_f1.index)
    ax.set_title(
        f"Stage 1: mean F1 over {N_REPLICATIONS} replications\n"
        f"(offline=symmetric, online=one-sided; non-null DGPs)"
    )
    for i in range(pivot_f1.shape[0]):
        for j in range(pivot_f1.shape[1]):
            val = pivot_f1.values[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=11)
    plt.colorbar(im, ax=ax, label="F1")
    plt.tight_layout()
    plt.savefig("results/stage1_heatmap.png", dpi=130, bbox_inches="tight")
    print(f"\nSaved heatmap: results/stage1_heatmap.png")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
