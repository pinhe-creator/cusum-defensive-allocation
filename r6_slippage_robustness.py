"""
r6_slippage_robustness.py

Slippage / crisis-execution robustness for the fixed-baseline CUSUM
overlay. Standard backtests charge a uniform transaction cost on every
rebalance. A reviewer concern is that a risk-off detector fires
precisely when markets are most illiquid, so the realized execution cost
on the *entry into risk-off* is higher than on a calm-market rebalance.

This script re-runs the portfolio backtest with an asymmetric cost
schedule: ordinary rebalances pay the baseline 10 bps per leg, but each
transition INTO the risk-off state (state 0 -> 1) pays an additional
crisis-day slippage surcharge of 0 / 25 / 50 / 100 bps per leg. If the
CUSUM-fixed Sharpe ranking against the static benchmark survives even a
100 bps entry surcharge, the slippage concern does not overturn the
result.

Design notes:
  * The detectors, hyperparameters, allocation weights, hold period,
    and no-look-ahead convention are IDENTICAL to experiments_portfolio_v2.py.
  * Detector code is loaded by file path so the machinery is bit-identical
    to the main backtest, and the modules' __main__ self-tests are NOT
    triggered (importlib sets __name__ to the module name).
  * Only the cost model changes: the surcharge is applied on the
    absolute weight change of the leg on risk-off ENTRY days.
  * The surcharge is applied to all four active strategies (the two CUSUM
    detectors, Adaptive CUSUM, and the VIX rule), since each has its own
    risk-off entries. The static benchmark never enters risk-off and is
    unaffected, so it is the natural invariant reference.

Run locally; <10 seconds.
"""

import os
import sys
import importlib.util
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load detectors by file path (bit-identical to the main backtest)
# ---------------------------------------------------------------------------
def _find_file(filename, root):
    best, best_depth = None, 10**9
    for dp, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "node_modules"]
        if filename in files:
            depth = dp.count(os.sep)
            if depth < best_depth:
                best, best_depth = os.path.join(dp, filename), depth
    return best


def _load(mod_name, path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_root = os.getcwd()
_fixed_path = _find_file("cusum_fixed.py", _root)
_abs_path = _find_file("cusum_abs.py", _root)
_adaptive_path = _find_file("cusum.py", _root)
if not (_fixed_path and _abs_path):
    sys.exit(f"Could not find cusum_fixed.py / cusum_abs.py under {_root}.")
print("Loading detectors:")
print(f"  {_fixed_path}")
print(f"  {_abs_path}")
if _adaptive_path:
    print(f"  {_adaptive_path}")
cusum_fixed = _load("cusum_fixed", _fixed_path)
cusum_abs = _load("cusum_abs", _abs_path)
cusum_adaptive = _load("cusum", _adaptive_path) if _adaptive_path else None

# ---------------------------------------------------------------------------
# Configuration --- IDENTICAL to experiments_portfolio_v2.py
# ---------------------------------------------------------------------------
W_NORMAL = np.array([0.60, 0.40])      # SPY, IEF
W_RISKOFF = np.array([0.30, 0.70])
RISK_OFF_HOLD = 20

BASE_TC_BPS = 10.0                     # ordinary per-leg cost
ENTRY_SURCHARGE_GRID = [0.0, 25.0, 50.0, 100.0]   # extra bps on risk-off entry

CUSUM_FIXED_KWARGS = dict(threshold=8.0, drift=0.50, baseline_window=252,
                          cooldown=60, side="negative")
CUSUM_ABS_KWARGS = dict(threshold=8.0, drift=0.50, baseline_window=252,
                        cooldown=60)

DATA_SIGNALS = "results/portfolio_v2_signals.csv"   # precomputed states
# Fallback: if you prefer to recompute states from spx_daily, see note below.


# ---------------------------------------------------------------------------
# Load aligned data: states + SPY/IEF returns
# ---------------------------------------------------------------------------
def load_states_and_returns():
    """Load the precomputed signal states and the SPY/IEF return series.

    States come from results/portfolio_v2_signals.csv (written by the main
    backtest). SPY/IEF daily log returns come from the data parquet files.
    """
    sig = pd.read_csv(DATA_SIGNALS, parse_dates=["date"]).sort_values("date")
    sig = sig.reset_index(drop=True)

    # SPY / IEF returns: try parquet, then csv fallbacks
    spy = ief = None
    for path in ["data/spy_daily.parquet", "data/SPY.parquet"]:
        if os.path.exists(path):
            spy = pd.read_parquet(path); break
    for path in ["data/ief_daily.parquet", "data/IEF.parquet"]:
        if os.path.exists(path):
            ief = pd.read_parquet(path); break
    if spy is None or ief is None:
        sys.exit("Could not find SPY/IEF parquet in data/. "
                 "Adjust paths in load_states_and_returns().")

    # Normalize to a date-indexed log-return series
    def to_logret(df):
        df = df.copy()
        # find a date column or index
        if "date" not in df.columns:
            df = df.reset_index()
            df.rename(columns={df.columns[0]: "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        if "log_return" in df.columns:
            r = df.set_index("date")["log_return"]
        elif "ret" in df.columns:
            r = df.set_index("date")["ret"]
        elif "close" in df.columns or "adj_close" in df.columns:
            col = "adj_close" if "adj_close" in df.columns else "close"
            r = np.log(df.set_index("date")[col]).diff()
        else:
            num = df.select_dtypes("number").columns[0]
            r = df.set_index("date")[num]
        return r

    spy_r = to_logret(spy)
    ief_r = to_logret(ief)

    # Align all three on the signal dates
    df = sig.set_index("date")
    df["spy_ret"] = spy_r.reindex(df.index)
    df["ief_ret"] = ief_r.reindex(df.index)
    df = df.dropna(subset=["spy_ret", "ief_ret"]).reset_index()
    return df


# ---------------------------------------------------------------------------
# Backtest with asymmetric (entry-surcharged) transaction costs
# ---------------------------------------------------------------------------
def backtest_surcharged(spy_returns, ief_returns, state,
                        base_tc_bps=BASE_TC_BPS, entry_surcharge_bps=0.0):
    """Net log returns with a higher cost on risk-off ENTRY days.

    Identical to the main backtest, except that on days where the state
    transitions 0 -> 1 (entry into risk-off), the per-leg cost is
    base_tc_bps + entry_surcharge_bps instead of base_tc_bps.
    """
    n = len(spy_returns)
    state = np.asarray(state)
    weights = np.zeros((n, 2))
    weights[state == 0] = W_NORMAL
    weights[state == 1] = W_RISKOFF

    base_rate = base_tc_bps / 1e4
    entry_rate = (base_tc_bps + entry_surcharge_bps) / 1e4

    # day-0 establishment + subsequent weight changes (same as main backtest)
    weight_prev = np.zeros((n, 2))
    weight_prev[1:] = weights[:-1]
    asset_returns = np.column_stack([spy_returns, ief_returns])
    gross_log_ret = np.sum(weight_prev * asset_returns, axis=1)

    dw = np.zeros((n, 2))
    dw[0] = weights[0]
    dw[1:] = weights[1:] - weights[:-1]
    abs_dw = np.sum(np.abs(dw), axis=1)

    # per-day cost rate: surcharge only on risk-off ENTRY days (0 -> 1)
    rate = np.full(n, base_rate)
    entry_days = np.zeros(n, dtype=bool)
    entry_days[1:] = (state[1:] == 1) & (state[:-1] == 0)
    rate[entry_days] = entry_rate

    tc_per_day = abs_dw * rate
    net_log_ret = gross_log_ret - tc_per_day
    return net_log_ret, int(entry_days.sum())


def sharpe(returns, ppy=252):
    mu = returns.mean() * ppy
    sd = returns.std() * np.sqrt(ppy)
    return float(mu / sd) if sd > 0 else np.nan


def max_dd(returns):
    w = np.exp(np.cumsum(returns))
    return float((w / np.maximum.accumulate(w) - 1).min())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("Slippage / crisis-execution robustness")
print(f"  base cost = {BASE_TC_BPS:.0f} bps/leg; "
      f"risk-off ENTRY surcharge grid = {ENTRY_SURCHARGE_GRID} bps/leg")
print("=" * 72)

df = load_states_and_returns()
spy_ret = df["spy_ret"].to_numpy()
ief_ret = df["ief_ret"].to_numpy()
n = len(df)
print(f"Aligned sample: {n} days, {df['date'].iloc[0].date()} "
      f"-> {df['date'].iloc[-1].date()}")

# Map strategy -> state column in the signals file
STRATEGIES = {
    "Static 60/40":   "static_state",
    "VIX threshold":  "vix_state",
    "Adaptive CUSUM": "cusum_adaptive_state",
    "CUSUM-fixed":    "cusum_fixed_state",
    "CUSUM-abs":      "cusum_abs_state",
}

# Static benchmark Sharpe (surcharge-invariant: never enters risk-off)
static_state = df["static_state"].to_numpy()
static_ret, _ = backtest_surcharged(spy_ret, ief_ret, static_state,
                                    entry_surcharge_bps=0.0)
static_sharpe = sharpe(static_ret)
print(f"\nStatic 60/40 Sharpe (invariant reference): {static_sharpe:.4f}\n")

rows = []
for name, col in STRATEGIES.items():
    if col not in df.columns:
        print(f"  [skip] {name}: column '{col}' not in signals file")
        continue
    state = df[col].to_numpy()
    for surch in ENTRY_SURCHARGE_GRID:
        ret, n_entries = backtest_surcharged(spy_ret, ief_ret, state,
                                             entry_surcharge_bps=surch)
        s = sharpe(ret)
        d = max_dd(ret)
        rows.append(dict(strategy=name, entry_surcharge_bps=surch,
                         sharpe=round(s, 4), max_dd=round(d, 4),
                         beats_static=bool(s > static_sharpe),
                         riskoff_entries=n_entries))

res = pd.DataFrame(rows)

# ---- Report: CUSUM-fixed focus ----
print("CUSUM-fixed vs static under increasing risk-off-entry slippage:")
print("-" * 72)
cf = res[res.strategy == "CUSUM-fixed"]
for _, r in cf.iterrows():
    flag = "beats static" if r.beats_static else "BELOW static"
    print(f"  +{r.entry_surcharge_bps:5.0f} bps entry surcharge: "
          f"Sharpe = {r.sharpe:.4f}  ({flag}); "
          f"MaxDD = {r.max_dd*100:6.2f}%  "
          f"[{r.riskoff_entries} risk-off entries]")

print("\nFull grid (all strategies):")
print("-" * 72)
with pd.option_context("display.width", 200):
    print(res.to_string(index=False))

os.makedirs("results", exist_ok=True)
res.to_csv("results/r6_slippage_robustness.csv", index=False)
print("\nWrote results/r6_slippage_robustness.csv")

# ---- Headline sentence for the paper ----
cf100 = cf[cf.entry_surcharge_bps == 100.0].iloc[0]
cf0 = cf[cf.entry_surcharge_bps == 0.0].iloc[0]
print("\n" + "=" * 72)
print("HEADLINE:")
print(f"  CUSUM-fixed Sharpe falls from {cf0.sharpe:.4f} (no surcharge) "
      f"to {cf100.sharpe:.4f} (+100 bps entry surcharge);")
print(f"  static benchmark = {static_sharpe:.4f}.")
if cf100.beats_static:
    print("  -> CUSUM-fixed STILL beats static even at +100 bps entry slippage.")
else:
    thresh = cf[~cf.beats_static].entry_surcharge_bps.min()
    print(f"  -> CUSUM-fixed drops below static at +{thresh:.0f} bps entry slippage.")
print("=" * 72)
print("\nDone.")
