import numpy as np
import pandas as pd

TC_BPS = 10.0
W_NORMAL = np.array([0.60, 0.40])
W_RISKOFF = np.array([0.30, 0.70])
TRADING_DAYS = 252

SPY_PARQUET = "data/spy_daily.parquet"
IEF_PARQUET = "data/ief_daily.parquet"
SIGNALS_CSV = "results/portfolio_v2_signals.csv"
PV2_METRICS_CSV = "results/portfolio_v2_metrics.csv"
HAMILTON_STATES_CSV = "results/r3b_hamilton_ms_states.csv"
HAMILTON_METRICS_CSV = "results/r3b_hamilton_ms_metrics.csv"

STRATEGY_STATE_COLS = {
    "Static 60/40": "static_state",
    "VIX threshold": "vix_state",
    "Adaptive CUSUM": "cusum_adaptive_state",
    "CUSUM-fixed": "cusum_fixed_state",
    "CUSUM-abs": "cusum_abs_state",
}


def backtest(spy_ret, ief_ret, state, tc_bps=TC_BPS):
    """Identical to experiments_portfolio_v2.py::backtest(). Log returns in, log returns out."""
    spy_ret = np.asarray(spy_ret, dtype=float)
    ief_ret = np.asarray(ief_ret, dtype=float)
    state = np.asarray(state, dtype=int)
    n = len(spy_ret)

    weights = np.zeros((n, 2))
    weights[state == 0] = W_NORMAL
    weights[state == 1] = W_RISKOFF

    weight_prev = np.zeros((n, 2))
    weight_prev[1:] = weights[:-1]

    asset_ret = np.column_stack([spy_ret, ief_ret])
    gross = np.sum(weight_prev * asset_ret, axis=1)

    dw = np.zeros((n, 2))
    dw[0] = weights[0]
    dw[1:] = weights[1:] - weights[:-1]
    tc = np.sum(np.abs(dw), axis=1) * (tc_bps / 1e4)

    return gross - tc


def sharpe(log_returns, periods=TRADING_DAYS):
    r = np.asarray(log_returns, dtype=float)
    sd = np.std(r, ddof=1)
    if sd <= 0 or not np.isfinite(sd):
        return float("nan")
    return float(np.mean(r) / sd * np.sqrt(periods))


def max_drawdown(log_returns):
    wealth = np.exp(np.cumsum(np.asarray(log_returns, dtype=float)))
    peak = np.maximum.accumulate(wealth)
    return float(np.min(wealth / peak - 1.0))


def annual_return(log_returns, periods=TRADING_DAYS):
    r = np.asarray(log_returns, dtype=float)
    if len(r) == 0:
        return float("nan")
    return float(np.exp(np.mean(r) * periods) - 1.0)


def annual_vol(log_returns, periods=TRADING_DAYS):
    return float(np.std(np.asarray(log_returns, dtype=float), ddof=1) * np.sqrt(periods))


def onset_count(state):
    s = np.asarray(state, dtype=int)
    if len(s) == 0:
        return 0
    return int(s[0] == 1) + int(np.sum(np.diff(s) > 0))


def run_lengths(state):
    s = np.asarray(state, dtype=int)
    out = []
    current = 0
    for v in s:
        if v == 1:
            current += 1
        elif current:
            out.append(current)
            current = 0
    if current:
        out.append(current)
    return out


def summarize(log_returns, state=None, label=""):
    row = {
        "strategy": label,
        "ann_return": annual_return(log_returns),
        "ann_vol": annual_vol(log_returns),
        "sharpe": sharpe(log_returns),
        "max_drawdown": max_drawdown(log_returns),
    }
    if state is not None:
        row["risk_off_share"] = float(np.mean(np.asarray(state) == 1))
        row["n_onsets"] = onset_count(state)
    return row


def alarms_to_state(alarms, duration):
    a = np.asarray(alarms, dtype=bool)
    n = len(a)
    state = np.zeros(n, dtype=int)
    remaining = 0
    for t in range(n):
        if a[t]:
            remaining = duration
        if remaining > 0:
            state[t] = 1
            remaining -= 1
    return state


def load_log_returns(path):
    x = pd.read_parquet(path)
    if "date" in x.columns:
        x["date"] = pd.to_datetime(x["date"])
        x = x.sort_values("date").set_index("date")
    else:
        x.index = pd.to_datetime(x.index)
        x = x.sort_index()
    if "log_return" in x.columns:
        s = x["log_return"]
    elif "close" in x.columns:
        s = np.log(x["close"]).diff()
    elif "return" in x.columns:
        s = np.log1p(x["return"])
    else:
        raise ValueError(f"{path}: no log_return/close/return column ({list(x.columns)})")
    return s.dropna()


def load_dated_csv(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").set_index("date")


def load_paper_panel(spy_path=SPY_PARQUET, ief_path=IEF_PARQUET,
                     signals_path=SIGNALS_CSV, hamilton_path=HAMILTON_STATES_CSV):
    """Aligned panel: SPY/IEF log returns plus every published strategy state."""
    signals = load_dated_csv(signals_path)
    spy = load_log_returns(spy_path)
    ief = load_log_returns(ief_path)

    df = signals.copy()
    df["spy_ret"] = spy.reindex(df.index)
    df["ief_ret"] = ief.reindex(df.index)

    missing = df["spy_ret"].isna().sum() + df["ief_ret"].isna().sum()
    if missing:
        raise ValueError(f"{missing} missing asset returns after alignment")

    try:
        ham = load_dated_csv(hamilton_path)
        col = "daily_state" if "daily_state" in ham.columns else ham.columns[0]
        df["hamilton_state"] = ham[col].reindex(df.index)
    except FileNotFoundError:
        df["hamilton_state"] = np.nan

    return df


def published_sharpes(pv2_path=PV2_METRICS_CSV, hamilton_path=HAMILTON_METRICS_CSV,
                      tc_bps=TC_BPS):
    out = {}
    try:
        m = pd.read_csv(pv2_path)
        if "tc_bps" in m.columns:
            m = m[m["tc_bps"] == tc_bps]
        name_col = next(c for c in m.columns if c.lower() in ("strategy", "name"))
        sh_col = next(c for c in m.columns if c.lower() in ("sharpe", "sharpe_ratio"))
        for _, r in m.iterrows():
            out[str(r[name_col]).strip()] = float(r[sh_col])
    except (FileNotFoundError, StopIteration):
        pass
    try:
        h = pd.read_csv(hamilton_path)
        sh_col = next(c for c in h.columns if c.lower() in ("sharpe", "sharpe_ratio"))
        out["Hamilton MS-AR(1)"] = float(h.iloc[0][sh_col])
    except (FileNotFoundError, StopIteration, IndexError):
        pass
    return out


def reconstruct_all(df, tc_bps=TC_BPS):
    spy = df["spy_ret"].to_numpy()
    ief = df["ief_ret"].to_numpy()
    returns, states = {}, {}
    for label, col in STRATEGY_STATE_COLS.items():
        if col not in df.columns:
            continue
        st = df[col].to_numpy(dtype=int)
        states[label] = st
        returns[label] = backtest(spy, ief, st, tc_bps)
    if "hamilton_state" in df.columns and df["hamilton_state"].notna().all():
        st = df["hamilton_state"].to_numpy(dtype=int)
        states["Hamilton MS-AR(1)"] = st
        returns["Hamilton MS-AR(1)"] = backtest(spy, ief, st, tc_bps)
    return returns, states
