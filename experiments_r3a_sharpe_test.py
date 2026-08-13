"""
R3a Experiment: Sharpe ratio bootstrap CI + Jobson-Korkie-Memmel test
======================================================================

Purpose:
    Provide statistical significance backing for the paper's main claim
    that CUSUM-fixed Sharpe (0.581) is greater than Static 60/40 Sharpe
    (0.518). Without this test, a reviewer can argue the 12% improvement
    is consistent with sampling noise.

Methods:
    1. Stationary bootstrap (Politis & Romano 1994) for per-strategy
       annualized Sharpe 95% CI. Block bootstrap is required because
       daily returns are autocorrelated; iid bootstrap would understate
       uncertainty.
    2. Jobson-Korkie test with Memmel (2003) correction for pairwise
       Sharpe equality. Parametric reference only; assumes joint
       normality, which is violated for daily financial returns.
    3. Pairwise paired-bootstrap Sharpe difference CI with two-level
       inference:
         - Primary (uncorrected): two-sided 95% CI excludes zero on the
           positive side.
         - Bonferroni: two-sided (1 - alpha/m)% CI excludes zero, where
           m = number of pairs (10) and alpha = 0.05, giving a 99.5% CI.
       P(boot diff <= 0) reported as a descriptive bootstrap probability
       and NOT used in any significance judgment.

    The bootstrap CI is the primary inference tool; JK-Memmel is the
    parametric reference. We avoid centered-bootstrap p-values because
    the Sharpe ratio is a nonlinear statistic and the simpler CI-
    exclusion criterion is more transparent.

Return-type convention:
    portfolio_v2 computes portfolio return as
        net_log_ret = sum(weights * asset_log_return) - tc_per_day
    so the Sharpe value reported in portfolio_v2_metrics.csv is computed
    from LOG returns. R3a follows the same convention.

Strategies tested (from portfolio_v2):
    Static_60_40, VIX_threshold, Adaptive_CUSUM, CUSUM_fixed, CUSUM_abs

Pairs tested (10 total):
    CUSUM_fixed > {Static, VIX, Adaptive, CUSUM-abs}            (4 pairs)
    CUSUM_abs   > {Static, VIX, Adaptive}                       (3 pairs)
    Adaptive    > {Static, VIX}                                  (2 pairs)
    VIX         > Static                                          (1 pair)

Multiple comparison correction:
    Bonferroni-adjusted alpha for 10 pairs: 0.05/10 = 0.005

Inputs:
    results/portfolio_v2_signals.csv     (state for each strategy daily)
    data/spy_daily.parquet               (SPY daily, has 'log_return' col)
    data/ief_daily.parquet               (IEF daily, has 'log_return' col)
    results/portfolio_v2_metrics.csv     (used to load expected Sharpe;
                                          fallback to hardcoded if missing)

Outputs (results/r3a_*):
    r3a_sharpe_ci.csv         : per-strategy bootstrap CI
    r3a_jobson_korkie.csv     : pairwise JK-Memmel tests
    r3a_bootstrap_diff.csv    : pairwise bootstrap difference CIs

Usage:
    cd /Users/chenpinhe/Downloads/cpd-finance-benchmark/
    python experiments_r3a_sharpe_test.py

    For faster debug runs, set N_BOOTSTRAP = 2000 in the config block.

Author: Pinhe Chen, Fort Hays State University
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

# Match the import convention used by other experiment scripts.
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# Configuration
# =============================================================================
RESULTS_DIR = Path("results")
DATA_DIR = Path("data")

SIGNALS_CSV = RESULTS_DIR / "portfolio_v2_signals.csv"
SPY_PATH = DATA_DIR / "spy_daily.parquet"
IEF_PATH = DATA_DIR / "ief_daily.parquet"
METRICS_CSV = RESULTS_DIR / "portfolio_v2_metrics.csv"

# If portfolio_v2_daily_returns.csv exists, use it directly; otherwise
# reconstruct from signals + SPY/IEF.
DAILY_RETURNS_CSV = RESULTS_DIR / "portfolio_v2_daily_returns.csv"

# Bootstrap parameters
N_BOOTSTRAP = 10000        # For debug runs, lower to 2000 for ~5x speedup
N_BOOTSTRAP_SENSITIVITY = 3000  # Sensitivity check uses fewer reps
BOOTSTRAP_BLOCK_LENGTHS = [20, 50, 100]   # for sensitivity
PRIMARY_BLOCK_LENGTH = 20
CONFIDENCE_LEVEL = 0.95     # Two-sided level used to derive CI percentiles
ANNUALIZATION = 252
SEED = 42

# Reconstruction control
# If True and DAILY_RETURNS_CSV exists, load it directly.
# If False (default), always reconstruct from signals + SPY/IEF parquet,
# which is safer because it guarantees alignment with portfolio_v2's
# log-return convention.
USE_EXISTING_DAILY_RETURNS = False

# Transaction cost (one-way, basis points). Matches portfolio_v2 default.
TC_BPS = 10

# State -> SPY weight mapping (risk-on 60/40 vs risk-off 30/70)
STATE_TO_SPY_WEIGHT = {0: 0.60, 1: 0.30}

# Column mapping: signals CSV state column -> strategy display name
SIGNAL_TO_STRATEGY = {
    "static_state":         "Static_60_40",
    "vix_state":            "VIX_threshold",
    "cusum_adaptive_state": "Adaptive_CUSUM",
    "cusum_fixed_state":    "CUSUM_fixed",
    "cusum_abs_state":      "CUSUM_abs",
}

STRATEGY_ORDER = [
    "Static_60_40",
    "VIX_threshold",
    "Adaptive_CUSUM",
    "CUSUM_fixed",
    "CUSUM_abs",
]

# Pairs to test: (candidate, baseline). Order matters for one-sided p-values
# (H1: SR_candidate > SR_baseline).
PAIRS = [
    # Main claim and primary comparisons
    ("CUSUM_fixed",    "Static_60_40"),    # MAIN CLAIM
    ("CUSUM_fixed",    "VIX_threshold"),
    ("CUSUM_fixed",    "Adaptive_CUSUM"),
    ("CUSUM_fixed",    "CUSUM_abs"),
    # CUSUM-abs comparisons
    ("CUSUM_abs",      "Static_60_40"),
    ("CUSUM_abs",      "VIX_threshold"),
    ("CUSUM_abs",      "Adaptive_CUSUM"),
    # Lower-priority comparisons (still reported for completeness)
    ("Adaptive_CUSUM", "Static_60_40"),
    ("Adaptive_CUSUM", "VIX_threshold"),
    ("VIX_threshold",  "Static_60_40"),
]

# Hardcoded expected Sharpe values (fallback if metrics.csv cannot be parsed).
# These come from session summary at TC = 10 bps.
EXPECTED_SHARPE_FALLBACK = {
    "Static_60_40":   0.518,
    "VIX_threshold":  0.495,
    "Adaptive_CUSUM": 0.538,
    "CUSUM_fixed":    0.581,
    "CUSUM_abs":      0.552,
}
VALIDATION_TOL = 0.02

# Display-name -> snake_case mapping for parsing portfolio_v2_metrics.csv
METRICS_NAME_MAP = {
    "Static 60/40":   "Static_60_40",
    "VIX threshold":  "VIX_threshold",
    "Adaptive CUSUM": "Adaptive_CUSUM",
    "CUSUM-fixed":    "CUSUM_fixed",
    "CUSUM-abs":      "CUSUM_abs",
}

# Output paths
OUT_SHARPE_CI = RESULTS_DIR / "r3a_sharpe_ci.csv"
OUT_JK_MEMMEL = RESULTS_DIR / "r3a_jobson_korkie.csv"
OUT_BOOT_DIFF = RESULTS_DIR / "r3a_bootstrap_diff.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Sharpe ratio
# =============================================================================
def compute_sharpe(returns, annualization=ANNUALIZATION) -> float:
    """Annualized Sharpe ratio assuming risk-free rate = 0."""
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return float("nan")
    sigma = r.std(ddof=1)
    if sigma == 0 or not np.isfinite(sigma):
        return float("nan")
    return float(r.mean() / sigma * np.sqrt(annualization))


# =============================================================================
# Expected Sharpe loading (dynamic from metrics.csv with fallback)
# =============================================================================
def load_expected_sharpes() -> Dict[str, float]:
    """
    Load expected Sharpe ratios for reconstruction validation.

    Priority:
      1. results/portfolio_v2_metrics.csv (dynamic, filtered to TC=TC_BPS)
      2. EXPECTED_SHARPE_FALLBACK (hardcoded from session summary)
    """
    if not METRICS_CSV.exists():
        print(f"  [Expected Sharpe] {METRICS_CSV} not found, "
              f"using hardcoded fallback")
        return dict(EXPECTED_SHARPE_FALLBACK)

    try:
        df = pd.read_csv(METRICS_CSV)
    except Exception as e:
        print(f"  [Expected Sharpe] Failed to read {METRICS_CSV}: {e}")
        return dict(EXPECTED_SHARPE_FALLBACK)

    # Filter to TC_BPS if column exists
    if "tc_bps" in df.columns:
        df = df[df["tc_bps"] == TC_BPS].copy()

    # Find Sharpe column (could be 'sharpe', 'Sharpe', or 'sharpe_ratio')
    sharpe_col = None
    for c in df.columns:
        if c.lower() in ("sharpe", "sharpe_ratio"):
            sharpe_col = c
            break
    if sharpe_col is None:
        print(f"  [Expected Sharpe] No Sharpe column found in metrics CSV, "
              f"using fallback")
        return dict(EXPECTED_SHARPE_FALLBACK)

    # Find strategy column
    strat_col = None
    for c in df.columns:
        if c.lower() in ("strategy", "strategy_name", "name"):
            strat_col = c
            break
    if strat_col is None:
        print(f"  [Expected Sharpe] No strategy column found in metrics CSV, "
              f"using fallback")
        return dict(EXPECTED_SHARPE_FALLBACK)

    out: Dict[str, float] = {}
    for _, row in df.iterrows():
        display_name = str(row[strat_col]).strip()
        key = METRICS_NAME_MAP.get(display_name)
        if key:
            out[key] = float(row[sharpe_col])

    if len(out) < 3:
        # Couldn't parse most strategies; fall back
        print(f"  [Expected Sharpe] Only parsed {len(out)} strategies "
              f"from metrics CSV, using fallback")
        return dict(EXPECTED_SHARPE_FALLBACK)

    print(f"  [Expected Sharpe] Loaded {len(out)} strategies from {METRICS_CSV}")
    return out


# =============================================================================
# Daily-return reconstruction from signals
# =============================================================================
def _load_asset_log_returns(parquet_path: Path, name: str) -> pd.Series:
    """
    Load an asset parquet and return its daily LOG returns series.

    portfolio_v2 uses log returns throughout, so R3a must use log returns
    for the reconstruction to match the reported Sharpe values.
    """
    df = pd.read_parquet(parquet_path)
    if "log_return" in df.columns:
        s = df["log_return"].copy()
    elif "close" in df.columns:
        s = np.log(df["close"]).diff()
    elif "return" in df.columns:
        # Convert simple to log
        s = np.log1p(df["return"])
    else:
        raise ValueError(
            f"{name} parquet must have 'log_return', 'return', or 'close' column"
        )
    s.name = name
    return s.dropna()


def reconstruct_daily_log_returns(
    signals_df: pd.DataFrame,
    spy_log_returns: pd.Series,
    ief_log_returns: pd.Series,
    tc_bps: float = TC_BPS,
) -> pd.DataFrame:
    """
    Reconstruct daily portfolio LOG returns for each strategy from state
    signals. Matches the portfolio_v2 convention:

        gross_log_ret[t] = w_spy[t-1] * spy_log[t] + (1-w_spy[t-1]) * ief_log[t]
        tc[t]            = total_turnover[t] * (tc_bps/1e4)
        net_log_ret[t]   = gross_log_ret[t] - tc[t]

    where total_turnover[t] is:
        - Day 0: |w_spy[0] - 0| + |w_ief[0] - 0| = 1.0
        - Day t (t >= 1): |w_spy[t] - w_spy[t-1]| + |w_ief[t] - w_ief[t-1]|
                       = 2 * |w_spy[t] - w_spy[t-1]|
        Rebalance is modeled as occurring at the close of day t. The TC for
        that rebalance is charged against day t's net return. Day t+1's
        gross return then uses w_spy[t].
    """
    common = (
        signals_df.index
        .intersection(spy_log_returns.index)
        .intersection(ief_log_returns.index)
        .sort_values()
    )
    spy_r = spy_log_returns.loc[common].values
    ief_r = ief_log_returns.loc[common].values
    T = len(common)
    tc_unit = tc_bps / 1e4

    out: Dict[str, np.ndarray] = {}
    for state_col, strat_name in SIGNAL_TO_STRATEGY.items():
        if state_col not in signals_df.columns:
            print(f"  WARNING: {state_col} not in signals CSV. Skipping.")
            continue

        states = signals_df[state_col].loc[common].astype(int).values
        if set(np.unique(states)) - {0, 1}:
            print(f"  WARNING: {state_col} has values outside {{0,1}}; "
                  f"unique = {np.unique(states)}")

        w_spy = np.array([STATE_TO_SPY_WEIGHT[s] for s in states])

        ret = np.zeros(T)
        # Day 0: cash -> initial allocation. Total turnover = 1.0.
        ret[0] = -tc_unit

        for t in range(1, T):
            gross = w_spy[t-1] * spy_r[t] + (1.0 - w_spy[t-1]) * ief_r[t]
            if w_spy[t] != w_spy[t-1]:
                tc_t = 2.0 * abs(w_spy[t] - w_spy[t-1]) * tc_unit
            else:
                tc_t = 0.0
            ret[t] = gross - tc_t

        out[strat_name] = ret

    return pd.DataFrame(out, index=common)


# =============================================================================
# Stationary bootstrap (Politis-Romano 1994)
# =============================================================================
def stationary_bootstrap_sharpe(
    returns_array: np.ndarray,
    block_length: float,
    n_reps: int,
    seed: int,
    annualization: int = ANNUALIZATION,
    progress_every: int = 1000,
    progress_label: str = "",
) -> np.ndarray:
    """
    Stationary bootstrap (Politis-Romano 1994) joint sampling of all
    K strategies, preserving cross-sectional correlations.

    Block lengths are geometric with mean `block_length`. A new block
    starts whenever an independent Bernoulli(1/block_length) draw fires.
    Within a block, indices increment by 1 (mod T).

    Parameters
    ----------
    returns_array : np.ndarray
        Shape (T, K). Daily log returns for K strategies over T periods.
    block_length : float
        Expected block length (= 1/p in geometric switching).
    n_reps : int
        Number of bootstrap replications.
    seed : int
        Random seed.

    Returns
    -------
    boot_sharpes : np.ndarray
        Shape (n_reps, K). Annualized Sharpe ratios from each replicate.
    """
    T, K = returns_array.shape
    rng = np.random.default_rng(seed)
    p = 1.0 / block_length
    sqrt_ann = np.sqrt(annualization)

    boot_sharpes = np.empty((n_reps, K))

    for b in range(n_reps):
        if progress_every > 0 and (b + 1) % progress_every == 0:
            label = f" ({progress_label})" if progress_label else ""
            print(f"    bootstrap{label} {b+1}/{n_reps}")

        # Determine block boundaries via switch probabilities.
        switches = rng.random(T) < p
        switches[0] = True
        block_seq_starts = np.where(switches)[0]
        block_data_starts = rng.integers(0, T, size=len(block_seq_starts))

        # Fill index sequence block by block (~50 blocks for T=5884, p=1/20)
        idx = np.empty(T, dtype=np.int64)
        for k in range(len(block_seq_starts)):
            bs = block_seq_starts[k]
            be = (block_seq_starts[k+1]
                  if k+1 < len(block_seq_starts) else T)
            block_len = be - bs
            idx[bs:be] = (block_data_starts[k] + np.arange(block_len)) % T

        boot_data = returns_array[idx]
        mu = boot_data.mean(axis=0)
        sigma = boot_data.std(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sr = np.where(sigma > 0, mu / sigma * sqrt_ann, 0.0)
        boot_sharpes[b] = sr

    return boot_sharpes


# =============================================================================
# Jobson-Korkie test with Memmel (2003) correction
# =============================================================================
def jobson_korkie_memmel(
    r_i: np.ndarray,
    r_j: np.ndarray,
    annualization: int = ANNUALIZATION,
) -> Dict:
    """
    Jobson-Korkie-Memmel test for Sharpe ratio equality.

    Memmel (2003) corrected variance estimator:
        theta = (1/T) * [
            2 * (1 - rho_ij)
            + 0.5 * (SR_i_d^2 + SR_j_d^2 - 2 * rho_ij^2 * SR_i_d * SR_j_d)
        ]

    Note rho_ij^2 in the cross term (not rho_ij as in JK 1981).
    SR_d denotes daily (un-annualized) Sharpe.

    Test statistic z = (SR_i_d - SR_j_d) / sqrt(theta) -> N(0, 1) under H0.

    Caveat for paper: JK-Memmel assumes joint normality of returns,
    which is violated for daily financial data (fat tails, volatility
    clustering). Reported as parametric reference; the bootstrap
    difference test below is the primary inferential tool.
    """
    r_i = np.asarray(r_i, dtype=float)
    r_j = np.asarray(r_j, dtype=float)
    T = len(r_i)
    assert len(r_j) == T, "Returns must have equal length"

    mu_i, mu_j = r_i.mean(), r_j.mean()
    sigma_i = r_i.std(ddof=1)
    sigma_j = r_j.std(ddof=1)

    # Defensive: zero variance => Sharpe undefined
    if sigma_i <= 0 or sigma_j <= 0 or not np.isfinite(sigma_i) or not np.isfinite(sigma_j):
        return {
            "sr_i_annual": float("nan"),
            "sr_j_annual": float("nan"),
            "sr_diff_annual": float("nan"),
            "rho_ij": float("nan"),
            "theta": float("nan"),
            "z_stat": float("nan"),
            "p_two_sided": float("nan"),
            "p_one_sided_i_gt_j": float("nan"),
        }

    sr_i_d = mu_i / sigma_i
    sr_j_d = mu_j / sigma_j

    sqrt_ann = np.sqrt(annualization)
    sr_i_ann = sr_i_d * sqrt_ann
    sr_j_ann = sr_j_d * sqrt_ann

    rho = float(np.corrcoef(r_i, r_j)[0, 1])

    theta = (1.0 / T) * (
        2.0 * (1.0 - rho)
        + 0.5 * (sr_i_d**2 + sr_j_d**2 - 2.0 * rho**2 * sr_i_d * sr_j_d)
    )

    if theta <= 0:
        z = float("nan")
        p_two = float("nan")
        p_one = float("nan")
    else:
        z = (sr_i_d - sr_j_d) / np.sqrt(theta)
        p_two = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
        p_one = 1.0 - stats.norm.cdf(z)

    return {
        "sr_i_annual": sr_i_ann,
        "sr_j_annual": sr_j_ann,
        "sr_diff_annual": sr_i_ann - sr_j_ann,
        "rho_ij": rho,
        "theta": theta,
        "z_stat": float(z),
        "p_two_sided": float(p_two),
        "p_one_sided_i_gt_j": float(p_one),
    }


# =============================================================================
# Main
# =============================================================================
def load_or_reconstruct_returns() -> pd.DataFrame:
    """Returns DataFrame of daily LOG returns, columns in STRATEGY_ORDER."""
    if USE_EXISTING_DAILY_RETURNS and DAILY_RETURNS_CSV.exists():
        print(f"[Load] USE_EXISTING_DAILY_RETURNS=True. Loading "
              f"{DAILY_RETURNS_CSV} directly.")
        print(f"       WARNING: assuming this file contains daily LOG returns")
        print(f"       matching portfolio_v2 convention. If it is in simple-")
        print(f"       return units, validation against expected Sharpe will")
        print(f"       fail. Set USE_EXISTING_DAILY_RETURNS=False to force")
        print(f"       reconstruction from raw signals + SPY/IEF parquet.")
        df = pd.read_csv(DAILY_RETURNS_CSV, index_col=0, parse_dates=True)
    else:
        if DAILY_RETURNS_CSV.exists():
            print(f"[Load] Found {DAILY_RETURNS_CSV} but USE_EXISTING_DAILY_RETURNS=False.")
            print(f"       Reconstructing from raw data to guarantee alignment.")
        else:
            print(f"[Load] {DAILY_RETURNS_CSV} not found. Reconstructing from "
                  f"signals + SPY/IEF log returns.")

        if not SIGNALS_CSV.exists():
            print(f"  FATAL: {SIGNALS_CSV} not found.")
            sys.exit(1)
        if not SPY_PATH.exists() or not IEF_PATH.exists():
            print(f"  FATAL: SPY or IEF parquet missing.")
            sys.exit(1)

        signals = pd.read_csv(SIGNALS_CSV, parse_dates=["date"]).set_index("date")
        spy_log = _load_asset_log_returns(SPY_PATH, "SPY")
        ief_log = _load_asset_log_returns(IEF_PATH, "IEF")
        print(f"  SPY log returns: {len(spy_log)} obs, "
              f"{spy_log.index.min().date()} to {spy_log.index.max().date()}")
        print(f"  IEF log returns: {len(ief_log)} obs, "
              f"{ief_log.index.min().date()} to {ief_log.index.max().date()}")
        print(f"  Reconstructing with TC = {TC_BPS} bps...")
        df = reconstruct_daily_log_returns(signals, spy_log, ief_log,
                                            tc_bps=TC_BPS)

    available = [c for c in STRATEGY_ORDER if c in df.columns]
    if len(available) < len(STRATEGY_ORDER):
        missing = [c for c in STRATEGY_ORDER if c not in df.columns]
        print(f"  WARNING: missing strategies in returns DataFrame: {missing}")
    return df[available]


def validate_reconstruction(
    point_sharpes: Dict[str, float],
    expected: Dict[str, float],
) -> bool:
    """Compare computed Sharpe vs expected. Returns True if all pass."""
    print(f"\n[Validation] Reconstructed vs expected Sharpe "
          f"(tolerance = {VALIDATION_TOL}):")
    print(f"  {'Strategy':<18s} {'Computed':>10s} {'Expected':>10s} "
          f"{'Delta':>8s} {'Status':>10s}")

    all_ok = True
    for strat in STRATEGY_ORDER:
        if strat not in point_sharpes:
            continue
        computed = point_sharpes[strat]
        exp = expected.get(strat, float("nan"))
        delta = computed - exp
        status = "OK" if abs(delta) < VALIDATION_TOL else "MISMATCH"
        if status == "MISMATCH":
            all_ok = False
        print(f"  {strat:<18s} {computed:>10.4f} {exp:>10.4f} "
              f"{delta:>+8.4f} {status:>10s}")

    if not all_ok:
        print("\n  WARNING: reconstruction does not match expected Sharpe.")
        print("  Possible causes:")
        print("    - TC convention differs (try TC_BPS = 5 or 0)")
        print("    - Weight scheme differs from 60/40 vs 30/70")
        print("    - Annualization differs (252 vs 250)")
        print("  Statistical tests below still run on the reconstructed")
        print("  series, but interpret results with this caveat in mind.")
    return all_ok


def main():
    print("=" * 72)
    print("R3a: Sharpe ratio bootstrap CI + Jobson-Korkie-Memmel test")
    print("=" * 72)
    print(f"Annualization: {ANNUALIZATION}    Bootstrap reps: {N_BOOTSTRAP}")
    print(f"Primary block length: {PRIMARY_BLOCK_LENGTH}    "
          f"Sensitivity: {BOOTSTRAP_BLOCK_LENGTHS}")
    print(f"Multiple comparison: Bonferroni for {len(PAIRS)} pairs, "
          f"adjusted alpha = {0.05/len(PAIRS):.4f}")
    print(f"Return convention: LOG returns (matches portfolio_v2)")
    print("=" * 72)

    # ---- Load expected Sharpe ----
    expected_sharpes = load_expected_sharpes()

    # ---- Load / reconstruct returns ----
    returns_df = load_or_reconstruct_returns()
    available = list(returns_df.columns)
    print(f"\n[Data] {len(returns_df)} obs, {len(available)} strategies: "
          f"{available}")
    print(f"       Date range: {returns_df.index.min().date()} to "
          f"{returns_df.index.max().date()}")

    # ---- Point Sharpe + validation ----
    point_sharpes = {
        s: compute_sharpe(returns_df[s].values) for s in available
    }
    validate_reconstruction(point_sharpes, expected_sharpes)

    returns_array = returns_df.values

    # ---- Bootstrap CIs ----
    print(f"\n[Bootstrap] Running stationary bootstrap "
          f"(block_length={PRIMARY_BLOCK_LENGTH}, n_reps={N_BOOTSTRAP})...")
    primary_boot = stationary_bootstrap_sharpe(
        returns_array, PRIMARY_BLOCK_LENGTH, N_BOOTSTRAP, SEED,
        progress_label=f"bl={PRIMARY_BLOCK_LENGTH}",
    )

    alpha = 1.0 - CONFIDENCE_LEVEL
    lo_pct = 100.0 * alpha / 2.0
    hi_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_rows = []
    for k, strat in enumerate(available):
        ci_lo, ci_hi = np.percentile(primary_boot[:, k], [lo_pct, hi_pct])
        ci_rows.append({
            "strategy": strat,
            "sharpe_point": point_sharpes[strat],
            "sharpe_boot_mean": float(primary_boot[:, k].mean()),
            "sharpe_boot_se": float(primary_boot[:, k].std(ddof=1)),
            f"sharpe_ci_lo_{lo_pct:.1f}pct": float(ci_lo),
            f"sharpe_ci_hi_{hi_pct:.1f}pct": float(ci_hi),
            "ci_width": float(ci_hi - ci_lo),
            "confidence_level": CONFIDENCE_LEVEL,
            "block_length": PRIMARY_BLOCK_LENGTH,
            "n_bootstrap": N_BOOTSTRAP,
        })
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(OUT_SHARPE_CI, index=False)
    print(f"[Bootstrap] Primary CI saved to {OUT_SHARPE_CI}")
    ci_lo_col = f"sharpe_ci_lo_{lo_pct:.1f}pct"
    ci_hi_col = f"sharpe_ci_hi_{hi_pct:.1f}pct"
    print(ci_df[["strategy", "sharpe_point", ci_lo_col, ci_hi_col,
                 "ci_width"]].to_string(index=False))

    # Sensitivity to block length (uses fewer reps; check CI width stability)
    print(f"\n[Bootstrap] Sensitivity check across block lengths "
          f"{BOOTSTRAP_BLOCK_LENGTHS} (n_reps={N_BOOTSTRAP_SENSITIVITY}):")
    sens_results: Dict[int, np.ndarray] = {PRIMARY_BLOCK_LENGTH: primary_boot}
    for bl in BOOTSTRAP_BLOCK_LENGTHS:
        if bl != PRIMARY_BLOCK_LENGTH:
            sens_results[bl] = stationary_bootstrap_sharpe(
                returns_array, bl, N_BOOTSTRAP_SENSITIVITY, SEED + bl,
                progress_label=f"bl={bl}",
            )
    print(f"  {'Strategy':<18s}", end="")
    for bl in BOOTSTRAP_BLOCK_LENGTHS:
        print(f"  bl={bl:<3d} {int(CONFIDENCE_LEVEL*100)}% CI       ", end="")
    print()
    for k, strat in enumerate(available):
        print(f"  {strat:<18s}", end="")
        for bl in BOOTSTRAP_BLOCK_LENGTHS:
            lo, hi = np.percentile(sens_results[bl][:, k], [lo_pct, hi_pct])
            print(f"  [{lo:>+5.3f}, {hi:>+5.3f}]   ", end="")
        print()

    # ---- Jobson-Korkie-Memmel pairwise tests ----
    bonferroni_alpha = 0.05 / len(PAIRS)
    print(f"\n[JK-Memmel] Pairwise Sharpe equality tests "
          f"(H1: SR_candidate > SR_baseline, one-sided)")
    print(f"  Parametric reference; primary inference relies on the")
    print(f"  paired stationary-bootstrap Sharpe difference CI below.")
    print(f"  Significance flags: * = p<0.05 uncorrected, "
          f"** = p<{bonferroni_alpha:.4f} Bonferroni")
    print()
    print(f"  {'Candidate':<16s} {'>':<3s} {'Baseline':<16s} "
          f"{'ΔSR':>9s} {'rho':>7s} {'z':>8s} "
          f"{'p_1side':>10s} {'p_2side':>10s} {'sig':>5s}")

    jk_rows = []
    for (si, sj) in PAIRS:
        if si not in available or sj not in available:
            continue
        r_i = returns_df[si].values
        r_j = returns_df[sj].values
        res = jobson_korkie_memmel(r_i, r_j)

        p1 = res["p_one_sided_i_gt_j"]
        if p1 < bonferroni_alpha:
            sig = "**"
        elif p1 < 0.05:
            sig = "*"
        else:
            sig = ""

        print(f"  {si:<16s} {'>':<3s} {sj:<16s} "
              f"{res['sr_diff_annual']:>+9.4f} {res['rho_ij']:>+7.3f} "
              f"{res['z_stat']:>+8.3f} "
              f"{p1:>10.4f} {res['p_two_sided']:>10.4f} {sig:>5s}")

        jk_rows.append({
            "strategy_i_candidate": si,
            "strategy_j_baseline":  sj,
            "sr_i_annual": res["sr_i_annual"],
            "sr_j_annual": res["sr_j_annual"],
            "sr_diff_annual": res["sr_diff_annual"],
            "rho_ij": res["rho_ij"],
            "theta": res["theta"],
            "z_stat": res["z_stat"],
            "p_one_sided_i_gt_j": res["p_one_sided_i_gt_j"],
            "p_two_sided": res["p_two_sided"],
            "significant_uncorrected_5pct":
                res["p_one_sided_i_gt_j"] < 0.05,
            "significant_bonferroni":
                res["p_one_sided_i_gt_j"] < bonferroni_alpha,
        })

    jk_df = pd.DataFrame(jk_rows)
    jk_df.to_csv(OUT_JK_MEMMEL, index=False)
    print(f"\n[JK-Memmel] Saved to {OUT_JK_MEMMEL}")

    # ---- Bootstrap difference: CI-based inference ----
    # Bonferroni-adjusted CI percentiles for multiple-comparison criterion.
    # 10 pairs -> alpha_bonf = 0.005 -> 99.5% CI -> percentiles 0.25/99.75.
    bonf_lo_pct = 100.0 * bonferroni_alpha / 2.0
    bonf_hi_pct = 100.0 * (1.0 - bonferroni_alpha / 2.0)
    bonf_ci_level = 1.0 - bonferroni_alpha

    print(f"\n[Bootstrap diff] Pairwise Sharpe difference CIs "
          f"(block_length={PRIMARY_BLOCK_LENGTH}, n_reps={N_BOOTSTRAP})")
    print(f"  Primary criterion:    two-sided {int(CONFIDENCE_LEVEL*100)}% CI "
          f"excludes zero on the positive side")
    print(f"  Bonferroni criterion: two-sided {bonf_ci_level*100:.1f}% CI "
          f"excludes zero on the positive side")
    print(f"                        ({len(PAIRS)} pairs, "
          f"alpha_bonf = {bonferroni_alpha:.4f})")
    print(f"  P(diff <= 0) reported as a descriptive bootstrap probability,")
    print(f"  NOT used in any formal significance judgment. JK-Memmel above")
    print(f"  is the parametric reference.")
    print()
    print(f"  {'Candidate':<16s} {'>':<3s} {'Baseline':<16s} "
          f"{'ΔSR_obs':>9s}  {'95% CI for ΔSR':>22s} "
          f"{'99.5% CI lo':>12s} {'P(d<=0)':>10s} {'sig':>5s}")

    diff_rows = []
    for (si, sj) in PAIRS:
        if si not in available or sj not in available:
            continue
        i = available.index(si)
        j = available.index(sj)
        diffs = primary_boot[:, i] - primary_boot[:, j]
        diff_obs = point_sharpes[si] - point_sharpes[sj]
        ci_lo, ci_hi = np.percentile(diffs, [lo_pct, hi_pct])
        bonf_ci_lo, bonf_ci_hi = np.percentile(diffs, [bonf_lo_pct, bonf_hi_pct])

        # Descriptive paired-bootstrap probability that candidate is NOT
        # better than baseline. NOT used in any significance judgment;
        # reported as interpretive aid only.
        prob_diff_le_zero_desc = float((diffs <= 0).mean())

        # Primary criterion: two-sided 95% CI excludes zero on the positive
        # side. Equivalent to a one-sided test at alpha ~ 0.025 (conservative
        # relative to a one-sided 0.05 test).
        bootstrap_ci_positive = bool(ci_lo > 0.0)

        # Bonferroni-adjusted criterion: two-sided 99.5% CI excludes zero
        # on the positive side. This is the statistically self-consistent
        # multiple-comparison adjustment for a CI-based criterion.
        bootstrap_bonferroni_ci_positive = bool(bonf_ci_lo > 0.0)

        if bootstrap_bonferroni_ci_positive:
            sig = "**"
        elif bootstrap_ci_positive:
            sig = "*"
        else:
            sig = ""

        print(f"  {si:<16s} {'>':<3s} {sj:<16s} "
              f"{diff_obs:>+9.4f}  [{ci_lo:>+8.4f}, {ci_hi:>+8.4f}] "
              f"{bonf_ci_lo:>+12.4f} {prob_diff_le_zero_desc:>10.4f} {sig:>5s}")

        diff_rows.append({
            "strategy_i_candidate": si,
            "strategy_j_baseline":  sj,
            "diff_observed": diff_obs,
            "diff_boot_mean": float(diffs.mean()),
            f"diff_ci_lo_{lo_pct:.1f}pct": float(ci_lo),
            f"diff_ci_hi_{hi_pct:.1f}pct": float(ci_hi),
            f"diff_bonf_ci_lo_{bonf_lo_pct:.3f}pct": float(bonf_ci_lo),
            f"diff_bonf_ci_hi_{bonf_hi_pct:.3f}pct": float(bonf_ci_hi),
            "confidence_level": CONFIDENCE_LEVEL,
            "bonferroni_confidence_level": bonf_ci_level,
            "bootstrap_prob_diff_le_zero_descriptive": prob_diff_le_zero_desc,
            "bootstrap_ci_excludes_zero_positive": bootstrap_ci_positive,
            "bootstrap_bonferroni_ci_excludes_zero_positive":
                bootstrap_bonferroni_ci_positive,
            "significant_uncorrected_5pct": bootstrap_ci_positive,
            "significant_bonferroni": bootstrap_bonferroni_ci_positive,
            "block_length": PRIMARY_BLOCK_LENGTH,
            "n_bootstrap": N_BOOTSTRAP,
        })

    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv(OUT_BOOT_DIFF, index=False)
    print(f"\n[Bootstrap diff] Saved to {OUT_BOOT_DIFF}")

    # ---- Main claim summary ----
    print("\n" + "=" * 72)
    print("MAIN CLAIM SUMMARY: CUSUM-fixed vs Static 60/40")
    print("=" * 72)

    main_jk = next(
        (r for r in jk_rows
         if r["strategy_i_candidate"] == "CUSUM_fixed"
         and r["strategy_j_baseline"] == "Static_60_40"),
        None,
    )
    main_diff = next(
        (r for r in diff_rows
         if r["strategy_i_candidate"] == "CUSUM_fixed"
         and r["strategy_j_baseline"] == "Static_60_40"),
        None,
    )

    if main_jk and main_diff:
        print(f"  CUSUM-fixed Sharpe:               {main_jk['sr_i_annual']:.4f}")
        print(f"  Static 60/40 Sharpe:              {main_jk['sr_j_annual']:.4f}")
        print(f"  Difference:                       {main_jk['sr_diff_annual']:+.4f}")
        print(f"  Cross-correlation:                {main_jk['rho_ij']:+.4f}")
        print()
        print(f"  JK-Memmel z-statistic:            {main_jk['z_stat']:+.3f}")
        print(f"  JK-Memmel p (one-sided):          "
              f"{main_jk['p_one_sided_i_gt_j']:.4f}  (parametric reference)")
        diff_lo_col = f"diff_ci_lo_{lo_pct:.1f}pct"
        diff_hi_col = f"diff_ci_hi_{hi_pct:.1f}pct"
        bonf_lo_col = f"diff_bonf_ci_lo_{bonf_lo_pct:.3f}pct"
        bonf_hi_col = f"diff_bonf_ci_hi_{bonf_hi_pct:.3f}pct"
        print(f"  Bootstrap {int(CONFIDENCE_LEVEL*100)}% CI for diff:        "
              f"[{main_diff[diff_lo_col]:+.4f}, "
              f"{main_diff[diff_hi_col]:+.4f}]")
        print(f"  Bonferroni {bonf_ci_level*100:.1f}% CI for diff:    "
              f"[{main_diff[bonf_lo_col]:+.4f}, "
              f"{main_diff[bonf_hi_col]:+.4f}]")
        print(f"  95% CI excludes 0 (pos):          "
              f"{main_diff['bootstrap_ci_excludes_zero_positive']}")
        print(f"  Bonferroni CI excludes 0 (pos):   "
              f"{main_diff['bootstrap_bonferroni_ci_excludes_zero_positive']}")
        print(f"  P(boot diff <= 0):                "
              f"{main_diff['bootstrap_prob_diff_le_zero_descriptive']:.4f}  "
              f"(descriptive)")
        print()

        # Inference based on bootstrap CI (primary) + JK-Memmel (reference).
        # Two independent CI-based criteria:
        #   - 95% CI exclusion: equivalent to ~ one-sided alpha=0.025 test
        #   - 99.5% CI exclusion: Bonferroni-adjusted for 10 pairs
        # JK-Memmel reported but does not drive the verdict.
        ci_supports = main_diff['bootstrap_ci_excludes_zero_positive']
        bonf_ci_supports = main_diff['bootstrap_bonferroni_ci_excludes_zero_positive']
        p_jk = main_jk['p_one_sided_i_gt_j']

        if bonf_ci_supports:
            print(f"  VERDICT: CUSUM-fixed Sharpe significantly > Static under")
            print(f"           Bonferroni-adjusted {bonf_ci_level*100:.1f}% bootstrap CI.")
            print(f"           Strongest possible inference under dependence-")
            print(f"           robust block-bootstrap. Paper can claim Sharpe")
            print(f"           improvement is statistically distinguishable from")
            print(f"           zero even after multiple-comparison correction.")
        elif ci_supports and p_jk < 0.05:
            print(f"  VERDICT: CUSUM-fixed Sharpe > Static at uncorrected 95% CI.")
            print(f"           Both bootstrap CI excludes zero AND JK-Memmel p < 0.05.")
            print(f"           Paper claim: 'nominally significant under both")
            print(f"           parametric and dependence-robust inference,")
            print(f"           though not under Bonferroni adjustment.'")
        elif ci_supports:
            print(f"  VERDICT: 95% bootstrap CI excludes zero but JK-Memmel p = "
                  f"{p_jk:.4f} >= 0.05.")
            print(f"           Paper claim: 'Sharpe difference is supported by")
            print(f"           dependence-robust bootstrap CI, with marginal")
            print(f"           parametric significance.'")
        elif p_jk < 0.05:
            print(f"  VERDICT: JK-Memmel p < 0.05 but bootstrap CI includes zero.")
            print(f"           Bootstrap is the primary inference; paper claim:")
            print(f"           'parametric significance, but Sharpe improvement is")
            print(f"           NOT distinguishable from zero under dependence-")
            print(f"           robust block-bootstrap inference.'")
        else:
            print(f"  VERDICT: CUSUM-fixed Sharpe improvement is NOT statistically")
            print(f"           distinguishable from Static 60/40 under either the")
            print(f"           bootstrap CI or JK-Memmel test. The 12% Sharpe gain")
            print(f"           is consistent with sampling noise given the high")
            print(f"           cross-correlation and sample size.")
            print(f"           Paper claim should emphasize point-estimate Sharpe")
            print(f"           + drawdown reduction + parameter-grid robustness,")
            print(f"           NOT statistical Sharpe significance.")

    print("\n" + "=" * 72)
    print("R3a complete.")
    print("Outputs: r3a_sharpe_ci.csv, r3a_jobson_korkie.csv, r3a_bootstrap_diff.csv")
    print("=" * 72)


if __name__ == "__main__":
    main()