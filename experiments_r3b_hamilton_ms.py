"""
R3b Experiment: Hamilton (1989) MS-AR(1) baseline portfolio strategy
=====================================================================

Purpose:
    Add a 6th portfolio strategy to the portfolio comparison table:
    a Hamilton (1989) Markov-switching AR(1) regime-detection baseline.
    Quantitative Finance reviewers consistently ask "why not compare
    against Markov-switching?"; R3b preempts this by adding the
    industry-standard econometric alternative.

    The key comparison claim is NOT that CUSUM-fixed estimates regimes
    more accurately than Hamilton MS-AR(1). It is that CUSUM-fixed, as
    a portfolio risk-off trigger, achieves competitive Sharpe and
    drawdown metrics against the Hamilton baseline, despite being
    structurally simpler (sequential change-point detection vs full
    Markov-switching model estimation).

Specification (Hamilton-style Markov-switching autoregression):
    A two-regime Markov-switching autoregression with regime-switching
    intercepts c_{s_t} and variances sigma_{s_t}, and a common AR(1)
    coefficient phi. In statsmodels MarkovAutoregression regression
    parameterization:

        y_t = c_{s_t} + phi * y_{t-1} + sigma_{s_t} * eps_t

    where s_t in {0, 1} follows a 2-state Markov chain with transition
    probabilities p[0->0] and p[1->1] estimated from data. This is
    equivalent to Hamilton's (1989) mean-adjusted form
    (y_t - mu_{s_t}) = phi * (y_{t-1} - mu_{s_{t-1}}) + sigma_{s_t} eps_t
    under the bijection c_{s_t} = mu_{s_t} * (1 - phi), but we report
    the regression-form parameters here since those are what
    statsmodels estimates directly.

    Configuration flags:
    - switching_ar = False (AR coefficient phi constant across regimes)
    - switching_variance = True (sigma_{s_t} varies by regime; this is
      the key feature for distinguishing low-vol from high-vol states)
    - switching_trend = True (default; intercept c_{s_t} varies by regime)

Estimation:
    statsmodels.tsa.regime_switching.markov_autoregression.
    MarkovAutoregression, EM algorithm with search_reps random
    initializations. Estimated on monthly SPX log returns (daily log
    returns resampled by sum to monthly).

Backtest convention:
    Rolling expanding-window estimation. For each month t after the
    burn-in period, fit MS-AR(1) on data {y_1, ..., y_t}, extract the
    filtered probability P(s_t = high_vol | y_{1:t}), and set the
    daily portfolio state for month t+1 to risk_off if this exceeds
    0.5. The signal is held constant through each month. This is a
    no-look-ahead, real-time-implementable design that matches the
    sequential nature of CUSUM-fixed.

    The filtered marginal probability is used rather than the smoothed
    probability, because smoothed probabilities use future data and
    would introduce look-ahead bias into the backtest.

Label-switching defense:
    EM iterations can assign state labels (0 vs 1) inconsistently
    across re-fits ("label switching"). To identify the physical
    high-volatility state robustly, we extract the fitted variances
    sigma_0^2 and sigma_1^2 from each re-fit and define the high-vol
    state as the one with the larger variance. The signal is then
    based on P(s_t = high_vol_state | y_{1:t}) regardless of the
    EM-assigned label.

Burn-in:
    36 months (3 years). Signal generation begins after month 37.
    The first 36 months of the daily backtest are held at risk_on
    (Static 60/40 allocation), matching the burn-in convention.

Outputs:
    results/r3b_hamilton_ms_metrics.csv      - 1-row summary metrics
    results/r3b_hamilton_ms_states.csv       - daily signal series
    results/r3b_hamilton_ms_diagnostics.csv  - monthly fit parameters
    results/r3b_hamilton_ms_panel.png        - 3-panel visualization
    results/r3b_hamilton_ms_panel.pdf        - same as PDF for LaTeX

Usage:
    cd /Users/chenpinhe/Downloads/cpd-finance-benchmark/
    python experiments_r3b_hamilton_ms.py

    Runtime: approximately 20-30 minutes (244 monthly EM fits with
    em_iter=20 and search_reps=5). For faster debugging, reduce
    SEARCH_REPS to 2 in the config block.

Author: Pinhe Chen, Fort Hays State University
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Suppress statsmodels convergence and numerical warnings during the
# rolling fit. We capture EM failures via try/except and report them
# separately in diagnostics, so suppressing warnings here is safe.
warnings.filterwarnings("ignore")

from statsmodels.tsa.regime_switching.markov_autoregression import (
    MarkovAutoregression,
)

from experiments_portfolio_v2 import (
    load_aligned_data,
    signal_static,
    backtest,
    compute_metrics,
    crisis_subsample_loss,
    CRISIS_WINDOWS,
)


# =============================================================================
# Configuration
# =============================================================================
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Hamilton 1989 spec
K_REGIMES = 2
AR_ORDER = 1
SWITCHING_AR = False           # AR coefficient constant across states
SWITCHING_VARIANCE = True      # Variance switches by state (key feature)

# Estimation
BURN_IN_MONTHS = 36            # 3 years burn-in
EM_ITER = 20                   # EM iterations per fit
SEARCH_REPS = 5                # Random EM restarts (paper-defensible robustness)

# Signal generation
P_HIGH_VOL_THRESHOLD = 0.5

# Backtest
TC_BPS = 10                    # Matches paper main results
STRATEGY_NAME = "Hamilton_MS_AR1"

# Progress reporting
PROGRESS_EVERY = 20            # Print status every N months

# Output paths
OUT_METRICS = RESULTS_DIR / "r3b_hamilton_ms_metrics.csv"
OUT_STATES = RESULTS_DIR / "r3b_hamilton_ms_states.csv"
OUT_DIAGNOSTICS = RESULTS_DIR / "r3b_hamilton_ms_diagnostics.csv"
OUT_PANEL_PNG = RESULTS_DIR / "r3b_hamilton_ms_panel.png"
OUT_PANEL_PDF = RESULTS_DIR / "r3b_hamilton_ms_panel.pdf"


# =============================================================================
# Label-switching defense: identify high-vol state by sigma magnitude
# =============================================================================
def identify_high_vol_state(results) -> Optional[int]:
    """
    Identify which regime index (0 or 1) corresponds to the high-volatility
    state. EM label-switching means the index can swap between fits, so we
    use the fitted variance magnitude as the physical identifier:
    high_vol_state = argmax(sigma^2).

    Note: statsmodels MarkovAutoregression returns params as a plain
    ndarray; we use results.model.param_names to find sigma2 indices.

    Returns None if identification fails (parameters not present or fit
    failed).
    """
    if results is None:
        return None
    try:
        param_names = results.model.param_names
        params = results.params
        # Find indices of sigma2[0], sigma2[1], ... in param_names
        sigma_indices = [i for i, name in enumerate(param_names)
                         if name.startswith("sigma2[")]
        if len(sigma_indices) >= 2:
            sigmas = np.array([params[i] for i in sigma_indices])
            return int(np.argmax(sigmas))
        # Fallback: identify by most negative mean (crisis = negative drift)
        const_indices = [i for i, name in enumerate(param_names)
                         if name.startswith("const[")]
        if len(const_indices) >= 2:
            consts = np.array([params[i] for i in const_indices])
            return int(np.argmin(consts))
        return 1
    except Exception:
        return None


def extract_filtered_p_high_vol(results, high_vol_state: Optional[int]
                                  ) -> Optional[float]:
    """
    Extract P(s_t = high_vol_state | y_{1:t}) for the most recent period
    from the filtered marginal probabilities. Returns None on failure.

    Note: results.filtered_marginal_probabilities is an ndarray of shape
    (T_effective, k_regimes); the last row is the most recent filtered
    probability.
    """
    if results is None or high_vol_state is None:
        return None
    try:
        filt = results.filtered_marginal_probabilities
        # filt shape can be (T_eff, K) or (K, T_eff) depending on
        # statsmodels version; detect orientation by k_regimes
        if filt.shape[-1] == K_REGIMES:
            # shape (T_eff, K), last row is what we want
            p_values = filt[-1]
        elif filt.shape[0] == K_REGIMES:
            # shape (K, T_eff), last column is what we want
            p_values = filt[:, -1]
        else:
            return None
        if len(p_values) > high_vol_state:
            return float(p_values[high_vol_state])
        return None
    except Exception:
        return None


def extract_state_parameters(results) -> Dict[str, float]:
    """
    Extract per-state means and standard deviations for diagnostics,
    sorted by variance (low-vol state first, high-vol state second).
    Returns NaN values if extraction fails.
    """
    out = {
        "mu_low_vol":     float("nan"),
        "mu_high_vol":    float("nan"),
        "sigma_low_vol":  float("nan"),
        "sigma_high_vol": float("nan"),
    }
    if results is None:
        return out
    try:
        param_names = results.model.param_names
        params = results.params
        sigma_indices = [i for i, name in enumerate(param_names)
                         if name.startswith("sigma2[")]
        const_indices = [i for i, name in enumerate(param_names)
                         if name.startswith("const[")]
        if len(sigma_indices) < 2 or len(const_indices) < 2:
            return out
        sigmas2 = np.array([params[i] for i in sigma_indices])
        consts = np.array([params[i] for i in const_indices])
        low_idx = int(np.argmin(sigmas2))
        high_idx = int(np.argmax(sigmas2))
        out["mu_low_vol"]     = float(consts[low_idx])
        out["mu_high_vol"]    = float(consts[high_idx])
        out["sigma_low_vol"]  = float(np.sqrt(sigmas2[low_idx]))
        out["sigma_high_vol"] = float(np.sqrt(sigmas2[high_idx]))
    except Exception:
        pass
    return out


# =============================================================================
# Safe fitting wrapper
# =============================================================================
def fit_ms_ar1_safe(history: pd.Series,
                     em_iter: int = EM_ITER,
                     search_reps: int = SEARCH_REPS):
    """
    Safely fit MS-AR(1) with random restarts. Returns the results
    object or None on any failure (numerical, convergence, etc.).
    """
    try:
        model = MarkovAutoregression(
            history.values,
            k_regimes=K_REGIMES,
            order=AR_ORDER,
            trend="c",
            switching_ar=SWITCHING_AR,
            switching_variance=SWITCHING_VARIANCE,
        )
        results = model.fit(
            em_iter=em_iter,
            search_reps=search_reps,
            disp=False,
        )
        return results
    except Exception:
        return None


# =============================================================================
# Rolling expanding-window estimation
# =============================================================================
def rolling_estimate_monthly_signal(
    monthly_returns: pd.Series,
    burn_in: int = BURN_IN_MONTHS,
    verbose: bool = True,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Rolling expanding-window MS-AR(1) estimation.

    For each month t in [burn_in, n_months), fit MS-AR(1) on the
    expanding window y_{1:t} and use the resulting filtered probability
    P(s_t = high_vol | y_{1:t}) to set the signal for month t+1.

    Returns
    -------
    monthly_signal : pd.Series
        Monthly risk-off signal indexed by month-end dates. Value at
        month m is the signal *for month m*, which is determined by the
        fit using data through month m-1. The first `burn_in + 1`
        months hold signal = 0.
    diagnostics : pd.DataFrame
        Per-month fit parameters and diagnostic flags for paper appendix.
    """
    n_months = len(monthly_returns)
    monthly_signal = pd.Series(0, index=monthly_returns.index, dtype=int)

    diag_rows: List[Dict] = []
    n_fits_attempted = 0
    n_fits_failed = 0
    last_valid_signal = 0  # Carry forward on EM failure
    t_loop_start = time.time()

    for t in range(burn_in, n_months):
        history = monthly_returns.iloc[: t + 1]   # data through month t
        n_fits_attempted += 1

        results = fit_ms_ar1_safe(history)
        high_vol_state = identify_high_vol_state(results)
        p_high_vol = extract_filtered_p_high_vol(results, high_vol_state)
        state_params = extract_state_parameters(results)

        if p_high_vol is None:
            n_fits_failed += 1
            signal_for_next_month = last_valid_signal
            fit_success = False
        else:
            signal_for_next_month = int(p_high_vol > P_HIGH_VOL_THRESHOLD)
            last_valid_signal = signal_for_next_month
            fit_success = True

        # Apply signal to NEXT month (no within-month look-ahead)
        if t + 1 < n_months:
            monthly_signal.iloc[t + 1] = signal_for_next_month

        diag_rows.append({
            "month_end_date":     monthly_returns.index[t],
            "n_months_used":      t + 1,
            "fit_success":        fit_success,
            "used_carry_forward": (not fit_success),
            "high_vol_state_idx": (high_vol_state
                                    if high_vol_state is not None else -1),
            "p_high_vol":         (p_high_vol
                                    if p_high_vol is not None else float("nan")),
            "signal_next_month":  signal_for_next_month,
            **state_params,
        })

        if verbose and (t - burn_in + 1) % PROGRESS_EVERY == 0:
            elapsed = time.time() - t_loop_start
            p_str = (f"{p_high_vol:.3f}"
                     if p_high_vol is not None else "  NaN")
            print(f"  [{t - burn_in + 1:>3d}/{n_months - burn_in}] "
                  f"{monthly_returns.index[t].date()}  "
                  f"P_high_vol={p_str}  "
                  f"signal_next={signal_for_next_month}  "
                  f"failures={n_fits_failed}  "
                  f"elapsed={elapsed:.0f}s")

    print(f"\n  Total fits attempted: {n_fits_attempted}")
    print(f"  Failures (carried forward): {n_fits_failed} "
          f"({100*n_fits_failed/max(n_fits_attempted, 1):.1f}%)")

    diagnostics = pd.DataFrame(diag_rows)
    return monthly_signal, diagnostics


# =============================================================================
# Monthly -> Daily signal carrier
# =============================================================================
def monthly_to_daily_signal(
    monthly_signal: pd.Series,
    daily_dates: pd.DatetimeIndex,
) -> np.ndarray:
    """
    Map each daily date to the signal for its calendar month.

    monthly_signal is indexed by month-end dates. Its value at month-end
    of month m IS the signal that applies to every trading day within
    that same calendar month m. This is the convention required by the
    backtest logic in main(): a fit at the end of month t produces a
    signal stored at the t+1 month-end index, and that signal applies
    to all trading days within calendar month t+1.

    Using pandas Period('M') matching ensures the signal applies to the
    entire calendar month, not just from month-end onward (which is
    what a naive ffill on month-end dates would produce, introducing an
    approximately-one-month systematic lag).

    Leading periods with no monthly signal (before the first non-zero
    monthly value) are filled with 0 (risk_on / Static 60/40).
    """
    # Re-index monthly_signal by calendar month period
    monthly_by_period = monthly_signal.copy()
    monthly_by_period.index = monthly_by_period.index.to_period("M")

    # Map each daily date's calendar month to the monthly signal
    daily_periods = daily_dates.to_period("M")
    daily_signal = pd.Series(daily_periods).map(monthly_by_period)
    daily_signal = daily_signal.fillna(0).astype(int)

    return daily_signal.values


# =============================================================================
# Main
# =============================================================================
def main():
    t0 = time.time()
    print("=" * 72)
    print("R3b: Hamilton (1989) MS-AR(1) baseline portfolio strategy")
    print("=" * 72)
    print(f"Spec:      MS-AR(1), K={K_REGIMES} states, order={AR_ORDER}")
    print(f"           switching_ar={SWITCHING_AR}, "
          f"switching_variance={SWITCHING_VARIANCE}")
    print(f"Burn-in:   {BURN_IN_MONTHS} months")
    print(f"EM:        em_iter={EM_ITER}, search_reps={SEARCH_REPS}")
    print(f"Signal:    P(high_vol | y_{{1:t}}) > {P_HIGH_VOL_THRESHOLD} "
          f"-> risk_off (next month)")
    print(f"Carrier:   Monthly -> daily, constant within month "
          f"(no within-month look-ahead)")
    print(f"TC:        {TC_BPS} bps")
    print("=" * 72)

    # ---- Load aligned data (delegates to portfolio_v2) ----
    df = load_aligned_data()
    daily_spx_ret = df["spx_ret"]
    daily_dates = df.index

    # ---- Resample daily to monthly (sum of log returns) ----
    # Use "ME" (month-end, pandas 2.2+) with fallback to "M" (deprecated
    # but still functional in 2.2 and earlier).
    try:
        monthly_returns = daily_spx_ret.resample("ME").sum()
    except (ValueError, TypeError):
        monthly_returns = daily_spx_ret.resample("M").sum()
    n_months = len(monthly_returns)
    print(f"\nMonthly SPX log returns: {n_months} months, "
          f"{monthly_returns.index.min().date()} to "
          f"{monthly_returns.index.max().date()}")

    if BURN_IN_MONTHS >= n_months:
        print(f"FATAL: burn-in ({BURN_IN_MONTHS}) >= total months ({n_months})")
        sys.exit(1)

    n_backtest_months = n_months - BURN_IN_MONTHS
    print(f"Backtest period: signal generated for months "
          f"{BURN_IN_MONTHS + 1} to {n_months} "
          f"({n_backtest_months} months, ~{n_backtest_months/12:.1f} years)")

    # ---- Rolling estimation ----
    print(f"\n[Rolling estimation] Fitting MS-AR(1) on expanding window...")
    print(f"  Progress reported every {PROGRESS_EVERY} months. "
          f"Estimated runtime: 20-30 min.")
    t1 = time.time()
    monthly_signal, diagnostics = rolling_estimate_monthly_signal(
        monthly_returns, burn_in=BURN_IN_MONTHS, verbose=True
    )
    elapsed_fit = time.time() - t1
    print(f"  Estimation time: {elapsed_fit:.1f}s "
          f"({elapsed_fit/max(n_backtest_months, 1):.2f}s/month avg)")

    # ---- Monthly to daily carrier ----
    print(f"\n[Carrier] Converting monthly signal to daily portfolio state...")
    daily_state = monthly_to_daily_signal(monthly_signal, daily_dates)
    risk_off_frac = float(daily_state.mean())
    print(f"  Daily state: {len(daily_state)} days, "
          f"risk_off fraction = {risk_off_frac:.4f}")

    # State transitions (analog of "alarms" for paper comparison)
    state_transitions = int(np.sum(np.diff(daily_state) != 0))
    print(f"  State transitions (analog of alarm count): {state_transitions}")

    # ---- Backtest ----
    print(f"\n[Backtest] Running portfolio backtest at TC={TC_BPS} bps...")
    spy_ret = df["spy_ret"].values
    ief_ret = df["ief_ret"].values
    bt = backtest(spy_ret, ief_ret, daily_state, tc_bps=TC_BPS)
    metrics = compute_metrics(bt["portfolio_returns"])

    # ---- Static baseline for comparison ----
    static_state = signal_static(len(df))
    static_bt = backtest(spy_ret, ief_ret, static_state, tc_bps=TC_BPS)
    static_metrics = compute_metrics(static_bt["portfolio_returns"])

    # ---- Crisis-period drawdowns ----
    crisis_dd = {}
    for label, (start, end) in CRISIS_WINDOWS.items():
        cl = crisis_subsample_loss(bt["portfolio_returns"], daily_dates,
                                    start, end)
        col_key = label.replace(" ", "_").replace("-", "_")
        crisis_dd[f"crisis_dd_{col_key}"] = cl["crisis_max_dd"]

    # ---- Summary printout ----
    print(f"\n{'=' * 72}")
    print(f"R3b RESULTS: Hamilton MS-AR(1) vs Static 60/40")
    print(f"{'=' * 72}")
    print(f"  Sharpe ratio:       {metrics['sharpe']:+.4f}  "
          f"(Static {static_metrics['sharpe']:+.4f}, "
          f"delta {metrics['sharpe'] - static_metrics['sharpe']:+.4f})")
    print(f"  Max drawdown:       {metrics['max_drawdown']:+.4f}  "
          f"(Static {static_metrics['max_drawdown']:+.4f}, "
          f"delta {metrics['max_drawdown'] - static_metrics['max_drawdown']:+.4f})")
    print(f"  Annualized return:  {metrics['ann_return']:+.4f}")
    print(f"  Annualized vol:     {metrics['ann_vol']:+.4f}")
    print(f"  Calmar ratio:       {metrics['calmar']:+.4f}")
    print(f"  Risk-off fraction:  {risk_off_frac:.4f}")
    print(f"  State transitions:  {state_transitions}")
    print(f"\n  Crisis-period max drawdowns:")
    for k, v in crisis_dd.items():
        label = k.replace("crisis_dd_", "").replace("_", " ")
        print(f"    {label:<22s} {v:+.4f}")

    # ---- Save outputs ----
    print(f"\n[Save outputs]")

    # 1. Metrics CSV (single row, parallel to portfolio_v2_metrics schema)
    metrics_row = {
        "strategy": STRATEGY_NAME,
        "tc_bps":   TC_BPS,
        **metrics,
        "risk_off_frac":   risk_off_frac,
        "n_alarms":        state_transitions,
        **crisis_dd,
        "delta_sharpe_vs_static":
            metrics["sharpe"] - static_metrics["sharpe"],
        "delta_maxdd_vs_static":
            metrics["max_drawdown"] - static_metrics["max_drawdown"],
        "burn_in_months":  BURN_IN_MONTHS,
        "em_iter":         EM_ITER,
        "search_reps":     SEARCH_REPS,
        "p_high_vol_threshold": P_HIGH_VOL_THRESHOLD,
        "n_fits_failed":   int((~diagnostics["fit_success"]).sum()),
    }
    pd.DataFrame([metrics_row]).to_csv(OUT_METRICS, index=False)
    print(f"  {OUT_METRICS}  ({len(metrics_row)} columns, 1 row)")

    # 2. States CSV (daily): for portfolio comparison + plotting
    # Build daily P(high_vol) by mapping each daily date to the
    # filtered probability inferred from the fit AT THE END OF THE SAME
    # CALENDAR MONTH (anchor to fit month, not signal month). This
    # gives the diagnostic interpretation "in month m, the MS-AR(1)
    # filtered probability of being in the high-vol state was X".
    # We use Period('M') matching to avoid the month-end forward-fill
    # lag that would result from reindex(method='ffill').
    monthly_p_by_period = diagnostics.set_index("month_end_date")["p_high_vol"].copy()
    monthly_p_by_period.index = monthly_p_by_period.index.to_period("M")
    daily_p_high_vol = pd.Series(daily_dates.to_period("M")).map(monthly_p_by_period)

    states_df = pd.DataFrame({
        "date":                  daily_dates,
        "p_high_vol_fit_month":  daily_p_high_vol.values,
        "daily_state":           daily_state,
        "portfolio_log_return":  bt["portfolio_returns"],
    })
    states_df.to_csv(OUT_STATES, index=False)
    print(f"  {OUT_STATES}  ({len(states_df)} rows)")

    # 3. Diagnostics CSV (monthly per-fit parameters)
    diagnostics.to_csv(OUT_DIAGNOSTICS, index=False)
    print(f"  {OUT_DIAGNOSTICS}  ({len(diagnostics)} rows)")

    # ---- Plot 3-panel figure ----
    print(f"\n[Plot] Generating 3-panel figure...")
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    # Panel 1: SPX monthly log returns
    ax = axes[0]
    ax.plot(monthly_returns.index, monthly_returns.values,
            linewidth=0.8, color="black", alpha=0.7)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_ylabel("SPX monthly log return")
    ax.set_title("Monthly SPX log returns (input to MS-AR(1))")
    ax.grid(alpha=0.3)

    # Panel 2: filtered P(high_vol) over time
    ax = axes[1]
    valid = diagnostics[diagnostics["fit_success"]].copy()
    ax.plot(valid["month_end_date"], valid["p_high_vol"],
            linewidth=1.2, color="C3",
            label=r"$P(s_t = \mathrm{high\_vol} | y_{1:t})$")
    ax.axhline(P_HIGH_VOL_THRESHOLD, color="gray", linestyle="--",
               linewidth=1, alpha=0.7,
               label=f"threshold = {P_HIGH_VOL_THRESHOLD}")
    ax.set_ylabel("Filtered P(high-vol)")
    ax.set_title("MS-AR(1) filtered probability of high-volatility state "
                 "(rolling expanding-window)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

    # Panel 3: cumulative wealth + risk-off shading
    ax = axes[2]
    cum_wealth = np.exp(np.cumsum(bt["portfolio_returns"]))
    static_cum = np.exp(np.cumsum(static_bt["portfolio_returns"]))
    ax.plot(daily_dates, cum_wealth, linewidth=1.2, color="C0",
            label=STRATEGY_NAME)
    ax.plot(daily_dates, static_cum, linewidth=1.2, color="gray",
            linestyle="--", label="Static 60/40")

    # Shade contiguous risk-off blocks
    in_block = False
    block_start = None
    for i in range(len(daily_state)):
        if daily_state[i] == 1 and not in_block:
            block_start = daily_dates[i]
            in_block = True
        elif daily_state[i] == 0 and in_block:
            ax.axvspan(block_start, daily_dates[i], alpha=0.15,
                       color="C3")
            in_block = False
    if in_block:
        ax.axvspan(block_start, daily_dates[-1], alpha=0.15, color="C3")

    ax.set_ylabel("Cumulative wealth")
    ax.set_xlabel("Date")
    ax.set_title(f"Cumulative wealth: {STRATEGY_NAME} vs Static 60/40  "
                 "(red shading = risk-off periods)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

    plt.suptitle(
        f"R3b: Hamilton (1989) MS-AR(1) Baseline "
        f"(TC = {TC_BPS} bps, burn-in = {BURN_IN_MONTHS} months)",
        fontsize=13, y=1.00,
    )
    plt.tight_layout()
    plt.savefig(OUT_PANEL_PNG, dpi=160, bbox_inches="tight")
    plt.savefig(OUT_PANEL_PDF, bbox_inches="tight")
    plt.close()
    print(f"  {OUT_PANEL_PNG}")
    print(f"  {OUT_PANEL_PDF}")

    print(f"\n{'=' * 72}")
    print(f"R3b complete. Total time: {time.time() - t0:.1f}s")
    print(f"Strategy added: {STRATEGY_NAME}")
    print(f"  Sharpe={metrics['sharpe']:+.4f}, "
          f"MaxDD={metrics['max_drawdown']:+.4f}, "
          f"risk_off={risk_off_frac:.4f}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
