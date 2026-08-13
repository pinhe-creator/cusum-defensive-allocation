"""Robustness check R2: CUSUM-fixed and CUSUM-abs sensitivity to
(threshold, drift) hyperparameters.

Tests whether the main paper results from experiments_portfolio_v2
depend critically on the choice (threshold=8, drift=0.5). These
defaults were inherited from the original adaptive CUSUM grid search
and were never tuned specifically for fixed-baseline detectors.

Grid (per detector):
    threshold in {6, 8, 10, 12}
    drift in {0.25, 0.50, 0.75}
    -> 12 configs each, 24 total

Fixed for this experiment:
    baseline_window = 252  (R1 verified robust across 126-1000)
    cooldown = 60
    hold = 20
    tc_bps = 10
    side = "negative" for CUSUM-fixed (portfolio convention)

Detectors tested:
    CUSUM-fixed (mean shift, fixed baseline, negative-only)
    CUSUM-abs (variance shift, fixed baseline, one-sided)

For each (detector, threshold, drift) config we report:
    - n_alarms (full SPX 1990-2026 and portfolio 2003-2026)
    - alarm_dates (full and portfolio, semicolon-separated)
    - covid_alarm and gfc_alarm retention (paper-critical)
    - baseline mu/sigma values
    - portfolio metrics: Sharpe, max_dd, calmar, ann_return
    - crisis losses (normalized): GFC, COVID, 2022 rate hikes
    - riskoff_fraction

Outputs:
    results/robustness_cusum_fixed_grid.csv
    results/robustness_cusum_abs_grid.csv
    results/robustness_grid_sharpe_fixed.png    (heatmap)
    results/robustness_grid_sharpe_abs.png      (heatmap)
    results/robustness_grid_maxdd_fixed.png     (heatmap)
    results/robustness_grid_maxdd_abs.png       (heatmap)

Relationship to R1: R2 conditions on the main baseline_window=252;
R1 (experiments_robustness_baseline.py) separately evaluates
baseline-window sensitivity. We do not run a joint 5x4x3 grid because
R1 results show baseline_window has small marginal effect on Sharpe
(0.539-0.581 range) so a joint grid would be redundant.
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

from algorithms import cusum_fixed, cusum_abs


# ====================================================================
# Configuration
# ====================================================================

THRESHOLDS = [6, 8, 10, 12]
DRIFTS = [0.25, 0.50, 0.75]
BASELINE_WINDOW = 252
COOLDOWN = 60
SIDE = "negative"
HOLD = 20
TC_BPS = 10

W_NORMAL  = np.array([0.60, 0.40])
W_RISKOFF = np.array([0.30, 0.70])

# Acute crisis windows for retention checks
COVID_WINDOW = (pd.Timestamp("2020-02-01"),
                pd.Timestamp("2020-03-31"))
GFC_WINDOW = (pd.Timestamp("2008-09-01"),
              pd.Timestamp("2008-12-31"))
RATEHIKE_WINDOW = (pd.Timestamp("2022-01-01"),
                   pd.Timestamp("2022-12-31"))


# ====================================================================
# Data loading
# ====================================================================

def load_data():
    spy = pd.read_parquet("data/spy_daily.parquet").rename(
        columns={"close": "spy_close", "log_return": "spy_ret"})
    ief = pd.read_parquet("data/ief_daily.parquet").rename(
        columns={"close": "ief_close", "log_return": "ief_ret"})
    spx = pd.read_parquet("data/spx_daily.parquet").rename(
        columns={"close": "spx_close", "log_return": "spx_ret"})

    spx_full = spx[["spx_ret"]].dropna()
    df_port = (spy[["spy_close", "spy_ret"]]
               .join(ief[["ief_close", "ief_ret"]], how="inner")
               .join(spx[["spx_ret"]], how="inner")
               .dropna())

    print(f"Full SPX: {len(spx_full)} obs")
    print(f"Portfolio sample: {len(df_port)} obs "
          f"({df_port.index.min().date()} to "
          f"{df_port.index.max().date()})")
    return spx_full, df_port


def load_benchmarks():
    try:
        df = pd.read_csv("results/portfolio_v2_metrics.csv")
        df = df[df["tc_bps"] == TC_BPS]
        out = {}
        for _, row in df.iterrows():
            out[row["strategy"]] = {
                "sharpe": row["sharpe"],
                "max_dd": row["max_drawdown"],
                "ann_return": row["ann_return"],
                "calmar": row["calmar"],
            }
        return out
    except Exception:
        return {}


# ====================================================================
# Backtest
# ====================================================================

def state_from_alarms(alarms, n, hold=HOLD):
    state = np.zeros(n, dtype=int)
    for a in alarms:
        end = min(a + hold + 1, n)
        state[a:end] = 1
    return state


def backtest_simple(spy_ret, ief_ret, state, tc_bps=TC_BPS):
    n = len(spy_ret)
    weights = np.zeros((n, 2))
    weights[state == 0] = W_NORMAL
    weights[state == 1] = W_RISKOFF
    tc_rate = tc_bps / 1e4

    weight_prev = np.zeros((n, 2))
    weight_prev[1:] = weights[:-1]

    asset_returns = np.column_stack([spy_ret, ief_ret])
    gross_log_ret = np.sum(weight_prev * asset_returns, axis=1)

    dw = np.zeros((n, 2))
    dw[0] = weights[0]
    dw[1:] = weights[1:] - weights[:-1]
    tc_per_day = np.sum(np.abs(dw), axis=1) * tc_rate
    return gross_log_ret - tc_per_day


def crisis_loss_normalized(returns, dates, start, end):
    """Worst normalized loss within crisis window (normalized to
    wealth at first day of window = 1.0)."""
    mask = (dates >= start) & (dates <= end)
    if not mask.any():
        return np.nan
    sub_log = returns[mask]
    cum = np.cumsum(sub_log)
    sub_wealth = np.exp(cum) / np.exp(cum[0])
    return float((sub_wealth - 1).min())


def portfolio_metrics(returns, state, dates):
    ann_return = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    wealth = np.exp(np.cumsum(returns))
    running_max = np.maximum.accumulate(wealth)
    drawdown = wealth / running_max - 1
    max_dd = float(drawdown.min())
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan
    return {
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_dd": max_dd,
        "calmar": float(calmar) if not np.isnan(calmar) else np.nan,
        "riskoff_fraction": float(state.mean()),
        "gfc_loss": crisis_loss_normalized(
            returns, dates, *GFC_WINDOW),
        "covid_loss": crisis_loss_normalized(
            returns, dates, *COVID_WINDOW),
        "ratehike_loss": crisis_loss_normalized(
            returns, dates, *RATEHIKE_WINDOW),
    }


# ====================================================================
# Run one detector across the grid
# ====================================================================

def run_grid(detector_name, detect_fn, spx_full, df_port):
    """Run one detector across the (threshold, drift) grid.

    detect_fn: callable(series, threshold, drift) -> result dict.
    Returns DataFrame with one row per (threshold, drift).
    """
    spx_ret_full = spx_full["spx_ret"].values
    dates_full = spx_full.index

    spy_ret = df_port["spy_ret"].values
    ief_ret = df_port["ief_ret"].values
    spx_ret_port = df_port["spx_ret"].values
    dates_port = df_port.index

    rows = []
    for threshold in THRESHOLDS:
        for drift in DRIFTS:
            # Full SPX detection
            result_full = detect_fn(spx_ret_full, threshold, drift)
            alarms_full = result_full["change_points"]
            covid_full = any(
                COVID_WINDOW[0] <= dates_full[a] <= COVID_WINDOW[1]
                for a in alarms_full
            )
            gfc_full = any(
                GFC_WINDOW[0] <= dates_full[a] <= GFC_WINDOW[1]
                for a in alarms_full
            )

            # Portfolio detection + backtest
            result_port = detect_fn(spx_ret_port, threshold, drift)
            alarms_port = result_port["change_points"]
            covid_port = any(
                COVID_WINDOW[0] <= dates_port[a] <= COVID_WINDOW[1]
                for a in alarms_port
            )
            gfc_port = any(
                GFC_WINDOW[0] <= dates_port[a] <= GFC_WINDOW[1]
                for a in alarms_port
            )

            state = state_from_alarms(alarms_port,
                                      len(spx_ret_port))
            returns = backtest_simple(spy_ret, ief_ret, state)
            metrics = portfolio_metrics(returns, state, dates_port)

            # Get baseline values
            baseline = result_port.get("baseline", {})
            if "mu" in baseline:
                baseline_str = (f"mu={baseline['mu']:.6f}, "
                                f"sigma={baseline['sigma']:.6f}")
            elif "mu_abs" in baseline:
                baseline_str = (f"mu_abs={baseline['mu_abs']:.4f}, "
                                f"sigma_abs={baseline['sigma_abs']:.4f}")
            else:
                baseline_str = "n/a"

            # Save alarm dates for paper appendix and post-hoc analysis
            alarm_dates_full_str = ";".join(
                dates_full[a].strftime("%Y-%m-%d") for a in alarms_full
            )
            alarm_dates_port_str = ";".join(
                dates_port[a].strftime("%Y-%m-%d") for a in alarms_port
            )

            row = {
                "detector": detector_name,
                "threshold": threshold,
                "drift": drift,
                "n_alarms_full": len(alarms_full),
                "n_alarms_port": len(alarms_port),
                "alarm_dates_full": alarm_dates_full_str,
                "alarm_dates_port": alarm_dates_port_str,
                "covid_alarm_full": covid_full,
                "covid_alarm_port": covid_port,
                "gfc_alarm_full": gfc_full,
                "gfc_alarm_port": gfc_port,
                "baseline_summary": baseline_str,
                **metrics,
            }
            rows.append(row)
    return pd.DataFrame(rows)


# ====================================================================
# Heatmap plotting
# ====================================================================

def plot_heatmap(df, metric, detector_name, vmin=None, vmax=None,
                 cmap="RdYlGn", fmt=".3f", out_path=None,
                 better_high=True):
    """Plot a (threshold x drift) heatmap of `metric`."""
    pivot = df.pivot(index="threshold", columns="drift",
                     values=metric)
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap_eff = cmap if better_high else cmap + "_r"
    im = ax.imshow(pivot.values, aspect="auto",
                   cmap=cmap_eff,
                   vmin=vmin, vmax=vmax,
                   origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("drift")
    ax.set_ylabel("threshold")
    ax.set_title(f"{metric} grid — {detector_name}\n"
                 f"baseline_window={BASELINE_WINDOW}, "
                 f"cooldown={COOLDOWN}, hold={HOLD}, tc={TC_BPS}bps")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, format(val, fmt),
                    ha="center", va="center", fontsize=10,
                    color="black")

    plt.colorbar(im, ax=ax, label=metric)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ====================================================================
# Main
# ====================================================================

def main():
    os.makedirs("results", exist_ok=True)
    t_start = time.time()

    spx_full, df_port = load_data()

    print()
    print("=" * 90)
    print("R2: (threshold, drift) robustness grid")
    print("=" * 90)
    print(f"Grid: threshold in {THRESHOLDS}, drift in {DRIFTS}")
    print(f"Fixed: baseline_window={BASELINE_WINDOW}, "
          f"cooldown={COOLDOWN}, hold={HOLD}, tc_bps={TC_BPS}")
    print(f"CUSUM-fixed uses side='{SIDE}'")
    print()

    # ---- CUSUM-fixed grid ----
    def detect_fixed(series, threshold, drift):
        return cusum_fixed.detect(
            series, threshold=threshold, drift=drift,
            baseline_window=BASELINE_WINDOW, cooldown=COOLDOWN,
            side=SIDE,
        )

    print("Running CUSUM-fixed grid ...")
    df_fixed = run_grid("CUSUM-fixed", detect_fixed, spx_full, df_port)
    df_fixed.to_csv("results/robustness_cusum_fixed_grid.csv",
                    index=False)

    # ---- CUSUM-abs grid ----
    def detect_abs(series, threshold, drift):
        return cusum_abs.detect(
            series, threshold=threshold, drift=drift,
            baseline_window=BASELINE_WINDOW, cooldown=COOLDOWN,
        )

    print("Running CUSUM-abs grid ...")
    df_abs = run_grid("CUSUM-abs", detect_abs, spx_full, df_port)
    df_abs.to_csv("results/robustness_cusum_abs_grid.csv",
                  index=False)

    # ---- Print summary tables ----
    print()
    print("=" * 90)
    print("CUSUM-fixed grid")
    print("=" * 90)
    cols = ["threshold", "drift", "n_alarms_port", "covid_alarm_port",
            "gfc_alarm_port", "sharpe", "max_dd", "calmar",
            "ann_return", "riskoff_fraction"]
    print(df_fixed[cols].round(4).to_string(index=False))

    print()
    print("=" * 90)
    print("CUSUM-abs grid")
    print("=" * 90)
    print(df_abs[cols].round(4).to_string(index=False))

    # ---- Benchmark reference ----
    bench = load_benchmarks()
    if bench:
        print()
        print(f"Reference benchmarks (tc={TC_BPS}):")
        for name in ["Static 60/40", "VIX threshold",
                     "Adaptive CUSUM"]:
            if name in bench:
                b = bench[name]
                print(f"  {name:<18s} Sharpe={b['sharpe']:.3f}, "
                      f"max_dd={b['max_dd']*100:.2f}%, "
                      f"Calmar={b['calmar']:.3f}, "
                      f"ann_ret={b['ann_return']*100:.2f}%")

    # ---- Pass/Fail summary ----
    print()
    print("=" * 90)
    print("ROBUSTNESS SUMMARY")
    print("=" * 90)
    static_sharpe = bench.get("Static 60/40", {}).get("sharpe", 0.518)
    static_maxdd = bench.get("Static 60/40", {}).get("max_dd", -0.3613)
    vix_sharpe = bench.get("VIX threshold", {}).get("sharpe", 0.495)

    for detector_name, df in [("CUSUM-fixed", df_fixed),
                              ("CUSUM-abs", df_abs)]:
        n_total = len(df)
        n_beat_static_sharpe = (df["sharpe"] > static_sharpe).sum()
        n_beat_vix_sharpe = (df["sharpe"] > vix_sharpe).sum()
        n_beat_static_dd = (df["max_dd"] > static_maxdd).sum()
        n_covid_port = df["covid_alarm_port"].sum()
        n_gfc_port = df["gfc_alarm_port"].sum()
        n_overtrigger = (df["riskoff_fraction"] > 0.15).sum()
        n_undertrigger = (df["n_alarms_port"] < 3).sum()
        print(f"\n{detector_name} ({n_total} configs):")
        print(f"  Sharpe > Static ({static_sharpe:.3f}): "
              f"{n_beat_static_sharpe}/{n_total}")
        print(f"  Sharpe > VIX ({vix_sharpe:.3f}): "
              f"{n_beat_vix_sharpe}/{n_total}")
        print(f"  max_dd better than Static "
              f"({static_maxdd*100:.1f}%): "
              f"{n_beat_static_dd}/{n_total}")
        print(f"  COVID alarm retained (portfolio): "
              f"{n_covid_port}/{n_total}")
        print(f"  GFC alarm retained (portfolio): "
              f"{n_gfc_port}/{n_total}")
        print(f"  Over-trigger (riskoff_frac > 15%): "
              f"{n_overtrigger}/{n_total}")
        print(f"  Under-trigger (< 3 portfolio alarms): "
              f"{n_undertrigger}/{n_total}")
        if df["sharpe"].notna().any():
            best_idx = df["sharpe"].idxmax()
            best = df.loc[best_idx]
            print(f"  Best config: threshold={best['threshold']}, "
                  f"drift={best['drift']} -> "
                  f"Sharpe={best['sharpe']:.3f}, "
                  f"max_dd={best['max_dd']*100:.1f}%, "
                  f"n_alarms_port={best['n_alarms_port']}")

    # ---- Heatmaps ----
    print()
    print("Generating heatmaps...")
    plot_heatmap(df_fixed, "sharpe", "CUSUM-fixed",
                 out_path="results/robustness_grid_sharpe_fixed.png",
                 better_high=True, fmt=".3f")
    plot_heatmap(df_abs, "sharpe", "CUSUM-abs",
                 out_path="results/robustness_grid_sharpe_abs.png",
                 better_high=True, fmt=".3f")
    plot_heatmap(df_fixed, "max_dd", "CUSUM-fixed",
                 out_path="results/robustness_grid_maxdd_fixed.png",
                 better_high=True, fmt=".3f")
    plot_heatmap(df_abs, "max_dd", "CUSUM-abs",
                 out_path="results/robustness_grid_maxdd_abs.png",
                 better_high=True, fmt=".3f")
    print("  Saved 4 heatmaps to results/")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
