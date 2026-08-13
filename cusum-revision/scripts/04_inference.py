import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from paper_backtest import TRADING_DAYS, max_drawdown, sharpe


def stationary_indices(n, block_length, rng):
    p = 1.0 / block_length
    idx = np.empty(n, dtype=np.int64)
    idx[0] = rng.integers(n)
    for t in range(1, n):
        idx[t] = rng.integers(n) if rng.random() < p else (idx[t - 1] + 1) % n
    return idx


def jk_memmel(r1, r2, periods=TRADING_DAYS):
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    n = len(r1)
    s1, s2 = np.std(r1, ddof=1), np.std(r2, ddof=1)
    sh1, sh2 = np.mean(r1) / s1, np.mean(r2) / s2
    rho = float(np.corrcoef(r1, r2)[0, 1])
    theta = (1.0 / n) * (2.0 * (1.0 - rho) + 0.5 *
                         (sh1 ** 2 + sh2 ** 2 - 2.0 * rho ** 2 * sh1 * sh2))
    if theta <= 0:
        return {"difference": (sh1 - sh2) * np.sqrt(periods), "correlation": rho,
                "z_stat": np.nan, "p_value": np.nan}
    z = (sh1 - sh2) / np.sqrt(theta)
    return {
        "difference": float((sh1 - sh2) * np.sqrt(periods)),
        "correlation": rho,
        "z_stat": float(z),
        "p_value": float(2.0 * (1.0 - stats.norm.cdf(abs(z)))),
    }


def paired_bootstrap(r1, r2, statistic, block_length, replications, seed, level):
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    n = len(r1)
    rng = np.random.default_rng(seed)
    observed = statistic(r1) - statistic(r2)
    draws = np.empty(replications)
    for b in range(replications):
        idx = stationary_indices(n, block_length, rng)
        draws[b] = statistic(r1[idx]) - statistic(r2[idx])
    alpha = 1.0 - level
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    centred = draws - observed
    return {
        "observed": float(observed),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "p_value": float(np.mean(np.abs(centred) >= abs(observed))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--returns", default="results/rev_strategy_returns.csv")
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--baseline", default="Static 60/40")
    ap.add_argument("--block-length", type=int, default=20)
    ap.add_argument("--replications", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--all-pairs", action="store_true")
    ap.add_argument("--out", default="results/rev_inference.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.returns)
    cols = [c for c in df.columns if c != args.date_col]
    if args.baseline not in cols:
        raise SystemExit(f"baseline column not found: {args.baseline}")

    pairs = (list(itertools.combinations(cols, 2)) if args.all_pairs
             else [(c, args.baseline) for c in cols if c != args.baseline])

    rows = []
    for a, b in pairs:
        r1 = df[a].to_numpy(dtype=float)
        r2 = df[b].to_numpy(dtype=float)
        jk = jk_memmel(r1, r2)
        sh = paired_bootstrap(r1, r2, sharpe, args.block_length,
                              args.replications, args.seed, args.level)
        dd = paired_bootstrap(r1, r2, max_drawdown, args.block_length,
                              args.replications, args.seed, args.level)
        rows.append({
            "strategy": a,
            "reference": b,
            "sharpe_diff": jk["difference"],
            "return_corr": jk["correlation"],
            "jk_p_value": jk["p_value"],
            "sharpe_ci_low": sh["ci_low"],
            "sharpe_ci_high": sh["ci_high"],
            "sharpe_excludes_zero": sh["excludes_zero"],
            "drawdown_diff": dd["observed"],
            "drawdown_ci_low": dd["ci_low"],
            "drawdown_ci_high": dd["ci_high"],
            "drawdown_excludes_zero": dd["excludes_zero"],
        })

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
