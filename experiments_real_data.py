"""Stage 2 experiment: 3 algorithms on real SPX data.

Runs PELT-RBF, PELT-L2, and CUSUM on the full SPX daily log returns
1990-2026 and evaluates against two event-anchor sets:
    - strict:   NBER recession starts + crash days (< -7%)
    - episodes: same anchors merged with min_gap_days=30

Note on terminology: anchors are pre-specified event windows,
NOT exact true changepoints. tp/fp/fn in metrics tables here mean:
    tp = aligned detections (within window of an anchor)
    fp = unaligned detections (outside any anchor window)
    fn = missed anchors

Outputs:
    results/real_data_detections.csv     - per-algorithm detection lists
    results/real_data_metrics_strict.csv - metrics under strict anchors
    results/real_data_metrics_episode.csv - metrics under episode anchors
    results/real_data_timeline.png       - timeline visualization
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

from data_loaders import download_spx
from ground_truth import build_ground_truth, ground_truth_indices
from metrics.detection import evaluate_result
from algorithms import pelt_rbf, pelt_l2, cusum


# ====================================================================
# Configuration
# ====================================================================

DETECTION_WINDOW = 20
KAPPA = 10
TRADING_DAYS_PER_YEAR = 252

ALGOS = [
    ("PELT-RBF",  pelt_rbf,
     {"pen": 5, "min_size": 30, "jump": 5}),
    ("PELT-L2",   pelt_l2,
     {"pen": 5, "min_size": 30, "jump": 5}),
    ("CUSUM",     cusum,
     {"threshold": 8.0, "drift": 0.50,
      "window": 100, "cooldown": 60}),
]


def evaluate_anchor_set(detections, anchors_indices, n_obs, label):
    """Evaluate all algorithms against one anchor set."""
    rows = []
    for algo_name, result in detections.items():
        metrics = evaluate_result(
            result, anchors_indices, n_obs=n_obs,
            window=DETECTION_WINDOW, kappa=KAPPA,
            trading_days_per_year=TRADING_DAYS_PER_YEAR,
        )
        metrics["algorithm"] = algo_name
        metrics["anchor_set"] = label
        metrics["n_detected"] = len(result["change_points"])
        rows.append(metrics)
    return pd.DataFrame(rows)


def main():
    os.makedirs("results", exist_ok=True)

    # ---- Load data ----
    df = download_spx()
    series = df["log_return"].values
    n_obs = len(series)
    print(f"Loaded SPX: {n_obs} obs from "
          f"{df.index.min().date()} to {df.index.max().date()}")

    # ---- Build two anchor sets ----
    events_strict = build_ground_truth(df)
    events_episode = build_ground_truth(df, min_gap_days=30)
    anchors_strict_idx = ground_truth_indices(events_strict, df)
    anchors_episode_idx = ground_truth_indices(events_episode, df)
    print(f"Strict anchors:  {len(anchors_strict_idx)}")
    print(f"Episode anchors: {len(anchors_episode_idx)}")
    print()
    print("Episode anchors (used as primary evaluation):")
    for date_str in sorted(events_episode.keys()):
        print(f"  {date_str}  {events_episode[date_str]}")
    print()

    # ---- Run all algorithms ----
    detections = {}
    for algo_name, algo_module, algo_kwargs in ALGOS:
        print(f"Running {algo_name}...")
        t0 = time.time()
        try:
            result = algo_module.detect(series, **algo_kwargs)
        except Exception as e:
            print(f"  [error] {algo_name}: {e}")
            continue
        elapsed = time.time() - t0
        detections[algo_name] = result
        print(f"  detected {len(result['change_points'])} CPs in "
              f"{elapsed:.1f}s")

    # ---- Save detection list with algorithm_type ----
    detection_rows = []
    for algo_name, result in detections.items():
        algo_type = result.get("metadata", {}).get("algorithm_type", "")
        for cp_idx in result["change_points"]:
            detection_rows.append({
                "algorithm": algo_name,
                "algorithm_type": algo_type,
                "cp_index": cp_idx,
                "cp_date": df.index[cp_idx].strftime("%Y-%m-%d"),
            })
    df_detections = pd.DataFrame(detection_rows)
    df_detections.to_csv("results/real_data_detections.csv", index=False)
    print(f"\nSaved detections: results/real_data_detections.csv "
          f"({len(df_detections)} rows)")

    # ---- Evaluate against both anchor sets ----
    df_metrics_strict = evaluate_anchor_set(
        detections, anchors_strict_idx, n_obs, "strict"
    )
    df_metrics_episode = evaluate_anchor_set(
        detections, anchors_episode_idx, n_obs, "episode"
    )

    df_metrics_strict.to_csv(
        "results/real_data_metrics_strict.csv", index=False
    )
    df_metrics_episode.to_csv(
        "results/real_data_metrics_episode.csv", index=False
    )

    # ---- Print results with disclaimer ----
    cols = ["algorithm", "anchor_set", "n_detected",
            "tp", "fp", "fn",
            "f1", "precision", "recall",
            "mean_dd", "uwdl", "false_alarms_per_year",
            "matching_mode_inferred"]

    print("\n" + "=" * 90)
    print("Note: tp/fp/fn here mean aligned detections, "
          "unaligned detections, and missed anchors.")
    print("Anchors are pre-specified event windows, "
          "NOT exact true changepoints.")
    print("=" * 90)
    print("METRICS: STRICT ANCHOR SET (13 anchors)")
    print("=" * 90)
    print(df_metrics_strict[cols].to_string(index=False))

    print("\n" + "=" * 90)
    print("METRICS: EPISODE ANCHOR SET (9 episodes, min_gap=30 days)")
    print("=" * 90)
    print(df_metrics_episode[cols].to_string(index=False))

    # ---- Timeline visualization ----
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)

    # Top panel: SPX price with anchors
    axes[0].plot(df.index, df["close"], lw=0.6, color="black")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("SPX (log scale)")
    axes[0].set_title("SPX with strict anchors (blue) and algorithm "
                      "detections (colored)")
    for idx in anchors_strict_idx:
        axes[0].axvline(df.index[idx], color="blue",
                        alpha=0.4, lw=1.5, linestyle="--")

    # Bottom panel: algorithm detections
    colors = {"PELT-RBF": "red", "PELT-L2": "orange", "CUSUM": "green"}
    y_levels = {"PELT-RBF": 0.85, "PELT-L2": 0.65, "CUSUM": 0.45}

    axes[1].set_ylim(0, 1)
    axes[1].set_yticks(list(y_levels.values()))
    axes[1].set_yticklabels(list(y_levels.keys()))

    # Anchor lines on bottom panel for reference
    for idx in anchors_strict_idx:
        axes[1].axvline(df.index[idx], color="blue",
                        alpha=0.2, lw=1, linestyle="--")

    for algo_name, result in detections.items():
        color = colors.get(algo_name, "gray")
        y = y_levels[algo_name]
        cps = result["change_points"]
        for cp_idx in cps:
            axes[1].plot(df.index[cp_idx], y, marker="|",
                         markersize=15, color=color, mew=2)

    axes[1].set_xlabel("Date")
    axes[1].set_title("Algorithm detection times (anchors in blue)")

    plt.tight_layout()
    plt.savefig("results/real_data_timeline.png",
                dpi=130, bbox_inches="tight")
    print(f"\nSaved timeline: results/real_data_timeline.png")


if __name__ == "__main__":
    main()
