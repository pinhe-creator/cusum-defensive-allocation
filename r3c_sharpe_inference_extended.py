"""
r3c_sharpe_inference_extended.py  (v3)

Addresses three reviewer concerns on the §6.5 Sharpe inference:

  (P12) Include Hamilton MS-AR(1) in the pairwise comparisons:
        full 15 pairs (6 strategies choose 2), in a paper-aligned order.

  (P11) Block-length sensitivity for the stationary bootstrap: rerun the
        CUSUM-fixed vs Static pair at block lengths 10, 20, 60.

  Bonferroni at 15 comparisons: 1 - 0.05/15 ≈ 0.99667 (NOT 99.5%, which
  was for 10 pairs). Both 95% and Bonferroni-15 percentile CIs are
  reported.

All six strategies' daily portfolio returns are reconstructed here with
the same backtest function as experiments_portfolio_v2.py at TC_BPS = 10
(Hamilton imports the very same function, so reconstruction is identical
to the values already on disk -- the reconstruction just makes the
uniformity visible to a reviewer). Reconstructed Sharpes are hard-checked
against portfolio_v2_metrics.csv and r3b_hamilton_ms_metrics.csv; the
script aborts if any mismatch exceeds 0.003.

Inputs:
  results/portfolio_v2_signals.csv     -- 5 strategy daily states + VIX
  results/portfolio_v2_metrics.csv     -- per-strategy Sharpe for x-check
  results/r3b_hamilton_ms_states.csv   -- Hamilton daily_state column
  results/r3b_hamilton_ms_metrics.csv  -- Hamilton Sharpe for x-check
  data/spy_daily.parquet, data/ief_daily.parquet  -- daily log returns

Outputs (placed in results/, not /tmp/, for the replication package):
  results/r3c_sharpe_inference_15pairs.csv
  results/r3c_block_length_sensitivity.csv

Runtime: ~75s for B = 10000.
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SIGNALS_PATH     = "results/portfolio_v2_signals.csv"
PV2_METRICS_PATH = "results/portfolio_v2_metrics.csv"
HAMILTON_PATH    = "results/r3b_hamilton_ms_states.csv"
HAMILTON_METRICS = "results/r3b_hamilton_ms_metrics.csv"
SPY_PARQUET      = "data/spy_daily.parquet"
IEF_PARQUET      = "data/ief_daily.parquet"

OUT_DIR          = "results"
OUT_15PAIRS      = os.path.join(OUT_DIR, "r3c_sharpe_inference_15pairs.csv")
OUT_BLOCKLEN     = os.path.join(OUT_DIR, "r3c_block_length_sensitivity.csv")

TC_BPS = 10
W_NORMAL  = np.array([0.60, 0.40])
W_RISKOFF = np.array([0.30, 0.70])
TRADING_DAYS_PER_YEAR = 252

B = 10_000
PAIR_BLOCK_LEN = 20                       # primary expected block length
BLOCK_LENGTHS  = [10, 20, 60]             # for P11 sensitivity
SEED = 42

N_PAIRS_FOR_BONFERRONI = 15
BONF_LEVEL = 1.0 - 0.05 / N_PAIRS_FOR_BONFERRONI   # 0.99667

SHARPE_XCHECK_TOL = 0.003                 # abort if reconstruction differs

STRATEGY_STATE_COLS_PV2 = {
    "Static 60/40":    "static_state",
    "VIX threshold":   "vix_state",
    "Adaptive CUSUM":  "cusum_adaptive_state",
    "CUSUM-fixed":     "cusum_fixed_state",
    "CUSUM-abs":       "cusum_abs_state",
}
ALL_STRATEGIES = list(STRATEGY_STATE_COLS_PV2) + ["Hamilton MS-AR(1)"]

# Paper-aligned pair order (CUSUM-fixed focal candidate first)
PAIR_ORDER = [
    ("CUSUM-fixed",       "Static 60/40"),
    ("CUSUM-fixed",       "VIX threshold"),
    ("CUSUM-fixed",       "Adaptive CUSUM"),
    ("CUSUM-fixed",       "CUSUM-abs"),
    ("CUSUM-fixed",       "Hamilton MS-AR(1)"),
    ("CUSUM-abs",         "Static 60/40"),
    ("CUSUM-abs",         "VIX threshold"),
    ("CUSUM-abs",         "Adaptive CUSUM"),
    ("CUSUM-abs",         "Hamilton MS-AR(1)"),
    ("Adaptive CUSUM",    "Static 60/40"),
    ("Adaptive CUSUM",    "VIX threshold"),
    ("Adaptive CUSUM",    "Hamilton MS-AR(1)"),
    ("VIX threshold",     "Static 60/40"),
    ("VIX threshold",     "Hamilton MS-AR(1)"),
    ("Hamilton MS-AR(1)", "Static 60/40"),
]
assert len(PAIR_ORDER) == N_PAIRS_FOR_BONFERRONI


# ---------------------------------------------------------------------------
# Defensive loaders
# ---------------------------------------------------------------------------
def load_return_series(path):
    """Load a parquet asset file; return a Series of daily log returns
    indexed by date. Handles either date-as-index or date-as-column."""
    x = pd.read_parquet(path)
    if "date" in x.columns:
        x["date"] = pd.to_datetime(x["date"])
        x = x.sort_values("date").set_index("date")
    else:
        x.index = pd.to_datetime(x.index)
        x = x.sort_index()
    if "log_return" not in x.columns:
        raise ValueError(f"{path} must contain a 'log_return' column "
                         f"(found: {list(x.columns)})")
    return x["log_return"]


def load_dated_csv(path):
    """Read a CSV with a 'date' column; return DataFrame with date as index."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").set_index("date")


def validate_binary_state(series, name):
    assert series.notna().all(), f"{name} has missing states"
    raw_vals = set(series.dropna().unique().tolist())
    if not raw_vals.issubset({0, 1, 0.0, 1.0, True, False}):
        raise ValueError(f"{name} contains non-binary values: {sorted(raw_vals)}")
    return series.astype(int).to_numpy()


# ---------------------------------------------------------------------------
# Backtest replication and Sharpe
# ---------------------------------------------------------------------------
def backtest_returns(spy_ret, ief_ret, state, tc_bps=TC_BPS):
    """Replicates experiments_portfolio_v2.py::backtest() exactly. Same
    function is also used inside experiments_r3b_hamilton_ms.py (it
    imports it), so applying this here to all 6 strategies is a uniform
    pipeline."""
    n = len(spy_ret)
    weights = np.zeros((n, 2))
    weights[state == 0] = W_NORMAL
    weights[state == 1] = W_RISKOFF

    weight_prev = np.zeros((n, 2))
    weight_prev[1:] = weights[:-1]   # day-1 lag; weight_prev[0] = cash

    asset_ret = np.column_stack([spy_ret, ief_ret])
    gross = np.sum(weight_prev * asset_ret, axis=1)

    dw = np.zeros((n, 2))
    dw[0] = weights[0]
    dw[1:] = weights[1:] - weights[:-1]
    tc = np.sum(np.abs(dw), axis=1) * (tc_bps / 1e4)

    return gross - tc


def annualized_sharpe(r):
    sd = np.std(r, ddof=1)
    if sd <= 0:
        return np.nan
    return (np.mean(r) / sd) * np.sqrt(TRADING_DAYS_PER_YEAR)


# ---------------------------------------------------------------------------
# Stationary bootstrap with JOINT resampling
# ---------------------------------------------------------------------------
def stationary_bootstrap_indices(T, expected_block_length, rng):
    """Politis-Romano (1994): geometric block lengths with mean
    expected_block_length, wrapped around the series."""
    p = 1.0 / expected_block_length
    idx = np.empty(T, dtype=np.int64)
    idx[0] = rng.integers(T)
    for t in range(1, T):
        if rng.random() < p:
            idx[t] = rng.integers(T)
        else:
            idx[t] = (idx[t - 1] + 1) % T
    return idx


def generate_joint_index_matrix(T, block_length, B, seed):
    """Pre-generate B resample index arrays. All pairwise Sharpe
    differences are evaluated on the same B paths, which is the
    'applied jointly across all strategies' design referenced in §6.5."""
    rng = np.random.default_rng(seed)
    idx_mat = np.empty((B, T), dtype=np.int64)
    for b in range(B):
        idx_mat[b] = stationary_bootstrap_indices(T, block_length, rng)
    return idx_mat


def bootstrap_diffs_for_pair(r_a, r_b, idx_mat):
    """Sharpe-difference vector across the joint bootstrap paths in
    idx_mat. Computed once per pair; multiple CIs (different levels)
    are then derived from the same vector."""
    B = idx_mat.shape[0]
    diffs = np.empty(B)
    for b in range(B):
        idx = idx_mat[b]
        diffs[b] = annualized_sharpe(r_a[idx]) - annualized_sharpe(r_b[idx])
    return diffs


def ci_from_diffs(diffs, level):
    """Two-sided percentile CI at the requested level."""
    alpha = 1.0 - level
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Load and assemble
# ---------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
print("Loading inputs...")

signals = load_dated_csv(SIGNALS_PATH)
ham_states = load_dated_csv(HAMILTON_PATH)
spy_ret = load_return_series(SPY_PARQUET)
ief_ret = load_return_series(IEF_PARQUET)

df = signals.copy()
df["spy_ret"]       = spy_ret.reindex(df.index)
df["ief_ret"]       = ief_ret.reindex(df.index)
df["hamilton_state"] = ham_states["daily_state"].reindex(df.index)

assert df["spy_ret"].notna().all(),       "SPY returns have gaps after alignment"
assert df["ief_ret"].notna().all(),       "IEF returns have gaps after alignment"
assert df["hamilton_state"].notna().all(), "Hamilton daily_state has gaps"
print(f"  Aligned {len(df)} rows from {df.index.min().date()} to "
      f"{df.index.max().date()}")

# Reconstruct daily portfolio returns for all 6 strategies via the same
# backtest function and tc_bps. Hamilton is reconstructed (not read) for
# pipeline uniformity; both routes give the same numbers.
state_map = {
    s: validate_binary_state(df[c], c)
    for s, c in STRATEGY_STATE_COLS_PV2.items()
}
state_map["Hamilton MS-AR(1)"] = validate_binary_state(
    df["hamilton_state"], "hamilton_state"
)
spy_arr = df["spy_ret"].to_numpy()
ief_arr = df["ief_ret"].to_numpy()
returns = {
    s: backtest_returns(spy_arr, ief_arr, st, tc_bps=TC_BPS)
    for s, st in state_map.items()
}

# Hard cross-check against published per-strategy Sharpe at tc_bps=10
print("\nReconstructed Sharpe vs paper metrics (tc_bps=10):")
pv2_m = pd.read_csv(PV2_METRICS_PATH)
pv2_m = pv2_m[pv2_m["tc_bps"] == TC_BPS].set_index("strategy")
ham_m = pd.read_csv(HAMILTON_METRICS).iloc[0]   # 1-row file
target = {s: float(pv2_m.loc[s, "sharpe"]) for s in STRATEGY_STATE_COLS_PV2}
target["Hamilton MS-AR(1)"] = float(ham_m["sharpe"])

ok = True
for s in ALL_STRATEGIES:
    got = annualized_sharpe(returns[s])
    diff = got - target[s]
    flag = "" if abs(diff) <= SHARPE_XCHECK_TOL else "   <-- MISMATCH"
    print(f"  {s:<22s} reconstructed={got:+.4f}  paper={target[s]:+.4f}  "
          f"diff={diff:+.4f}{flag}")
    if abs(diff) > SHARPE_XCHECK_TOL:
        ok = False
if not ok:
    raise SystemExit(f"\nABORT: at least one reconstructed Sharpe differs "
                     f"from paper by more than {SHARPE_XCHECK_TOL}. "
                     f"The bootstrap below would be on a different series "
                     f"than the paper reports.")
print("  All within tolerance; reconstruction matches paper.")

# Stronger Hamilton check: aggregate Sharpe match is necessary but not
# sufficient. If the on-disk r3b file also stored daily portfolio returns,
# compare them point-by-point to prove uniformity of the pipeline rather
# than coincidental Sharpe agreement.
if "portfolio_log_return" in ham_states.columns:
    ham_disk = ham_states["portfolio_log_return"].reindex(df.index).to_numpy()
    ham_recon = returns["Hamilton MS-AR(1)"]
    max_abs_diff = float(np.nanmax(np.abs(ham_recon - ham_disk)))
    print(f"  Hamilton daily-return max abs diff (reconstructed vs disk): "
          f"{max_abs_diff:.3e}")
    if max_abs_diff > 1e-10:
        raise SystemExit("ABORT: reconstructed Hamilton daily returns differ "
                         "from the on-disk r3b series. Pipeline is not "
                         "uniform across the six strategies.")


# ---------------------------------------------------------------------------
# P12: 15-pair Sharpe-difference inference with joint resampling
# ---------------------------------------------------------------------------
print(f"\n=== P12: 15-pair Sharpe-difference inference  "
      f"(B = {B}, block = {PAIR_BLOCK_LEN}) ===")
print(f"  Bonferroni-adjusted level for {N_PAIRS_FOR_BONFERRONI} comparisons: "
      f"{BONF_LEVEL:.5f}")

T = len(df)
idx_mat = generate_joint_index_matrix(T, PAIR_BLOCK_LEN, B, SEED)
print(f"  Pre-generated joint bootstrap index matrix: shape {idx_mat.shape}")

rows = []
for a, b in PAIR_ORDER:
    r_a, r_b = returns[a], returns[b]
    diff = annualized_sharpe(r_a) - annualized_sharpe(r_b)
    rho = float(np.corrcoef(r_a, r_b)[0, 1])
    diffs        = bootstrap_diffs_for_pair(r_a, r_b, idx_mat)
    lo95, hi95   = ci_from_diffs(diffs, 0.95)
    lo_bf, hi_bf = ci_from_diffs(diffs, BONF_LEVEL)
    rows.append({
        "candidate":         a,
        "baseline":          b,
        "delta_sharpe":      round(diff, 4),
        "return_corr":       round(rho, 4),
        "ci95_lo":           round(lo95, 4),
        "ci95_hi":           round(hi95, 4),
        "excludes_zero_95":  (lo95 > 0) or (hi95 < 0),
        "ci_bonf_level":     round(BONF_LEVEL, 5),
        "ci_bonf_lo":        round(lo_bf, 4),
        "ci_bonf_hi":        round(hi_bf, 4),
        "excludes_zero_bonf": (lo_bf > 0) or (hi_bf < 0),
    })
out15 = pd.DataFrame(rows)
out15.to_csv(OUT_15PAIRS, index=False)
print("\n" + out15[["candidate", "baseline", "delta_sharpe", "return_corr",
                    "ci95_lo", "ci95_hi", "excludes_zero_95",
                    "ci_bonf_lo", "ci_bonf_hi"]].to_string(index=False))
print(f"\nWrote {OUT_15PAIRS}")


# ---------------------------------------------------------------------------
# P11: Block-length sensitivity, CUSUM-fixed vs Static (headline pair)
# ---------------------------------------------------------------------------
print(f"\n=== P11: Block-length sensitivity, CUSUM-fixed vs Static ===")
r_a = returns["CUSUM-fixed"]
r_b = returns["Static 60/40"]
diff_point = annualized_sharpe(r_a) - annualized_sharpe(r_b)
rows = []
for L in BLOCK_LENGTHS:
    idx_L = generate_joint_index_matrix(T, L, B, SEED)
    diffs = bootstrap_diffs_for_pair(r_a, r_b, idx_L)
    lo, hi = ci_from_diffs(diffs, 0.95)
    rows.append({
        "block_length":  L,
        "delta_sharpe":  round(diff_point, 4),
        "ci95_lo":       round(lo, 4),
        "ci95_hi":       round(hi, 4),
    })
outBL = pd.DataFrame(rows)
outBL.to_csv(OUT_BLOCKLEN, index=False)
print(outBL.to_string(index=False))
print(f"\nWrote {OUT_BLOCKLEN}")