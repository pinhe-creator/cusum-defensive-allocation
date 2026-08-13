"""
r4_inspect_data.py

Pre-flight inspection of the 8 World-Indices-by-WRDS CSV exports before
building the cross-market validation. Reports columns, date coverage,
row counts, the magnitude of the return column (decimal vs percent),
and the minimum security-count (n) inside the first 252-observation
baseline window for each market.

Run from the project root:
    python3 r4_inspect_data.py
"""

import os
import glob
import pandas as pd
import numpy as np

# Where the CSVs live. Adjust if you put them elsewhere.
SEARCH_DIRS = [
    "data",
    "data/world_indices",
    ".",
    os.path.expanduser("~/Downloads"),
]

EXPECTED_FILES = ["UK", "Japan", "Italy", "HK", "Germany",
                  "France", "China", "Brazil"]

BASELINE_WINDOW = 252


def find_csv(name):
    for d in SEARCH_DIRS:
        p = os.path.join(d, f"{name}.csv")
        if os.path.exists(p):
            return p
    # fallback: glob anywhere under cwd
    hits = glob.glob(f"**/{name}.csv", recursive=True)
    return hits[0] if hits else None


print("=" * 70)
print("World-Indices CSV inspection")
print("=" * 70)

found = {}
for name in EXPECTED_FILES:
    path = find_csv(name)
    found[name] = path
    print(f"  {name:10s} -> {path if path else 'NOT FOUND'}")

print()
any_found = [n for n, p in found.items() if p]
if not any_found:
    print("No CSVs found. Edit SEARCH_DIRS at the top of this script to "
          "point at the folder containing UK.csv etc.")
    raise SystemExit(1)

# Inspect the first found file in detail for column structure
first = found[any_found[0]]
df0 = pd.read_csv(first)
print("=" * 70)
print(f"COLUMN STRUCTURE (from {any_found[0]}.csv):")
print("=" * 70)
print("  columns:", df0.columns.tolist())
print()
print("  first 3 rows:")
print(df0.head(3).to_string(index=False))
print()

# Try to identify the relevant columns heuristically
cols_lower = {c.lower(): c for c in df0.columns}
date_col = cols_lower.get("date")
ret_col = cols_lower.get("portret")
retx_col = cols_lower.get("portretx")
n_col = cols_lower.get("n")
print(f"  detected date column:          {date_col}")
print(f"  detected return column:        {ret_col}  (with dividends)")
print(f"  detected return-ex-div column: {retx_col}")
print(f"  detected security-count col:   {n_col}")
print()

# Per-market summary
print("=" * 70)
print("PER-MARKET COVERAGE")
print("=" * 70)
print(f"{'Market':10s} {'rows':>7s}  {'start':>12s}  {'end':>12s}  "
      f"{'ret_mean':>10s}  {'ret_std':>9s}  {'min_n_base':>10s}")
print("-" * 70)

for name in EXPECTED_FILES:
    path = found.get(name)
    if not path:
        print(f"{name:10s}  (missing)")
        continue
    df = pd.read_csv(path)
    # robust column pick
    cl = {c.lower(): c for c in df.columns}
    dcol = cl.get("date")
    rcol = cl.get("portret")
    ncol = cl.get("n")
    df[dcol] = pd.to_datetime(df[dcol])
    df = df.sort_values(dcol).reset_index(drop=True)

    ret = pd.to_numeric(df[rcol], errors="coerce")
    start = df[dcol].min().date()
    end = df[dcol].max().date()
    ret_mean = ret.mean()
    ret_std = ret.std()
    if ncol and len(df) >= BASELINE_WINDOW:
        min_n_base = int(df[ncol].iloc[:BASELINE_WINDOW].min())
    elif ncol:
        min_n_base = int(df[ncol].min())
    else:
        min_n_base = -1

    print(f"{name:10s} {len(df):>7d}  {str(start):>12s}  {str(end):>12s}  "
          f"{ret_mean:>10.5f}  {ret_std:>9.5f}  {min_n_base:>10d}")

print()
print("=" * 70)
print("MAGNITUDE CHECK")
print("=" * 70)
print("  If ret_std is ~0.01 (1%), portret is in DECIMAL form (good, "
      "use log(1+ret) directly).")
print("  If ret_std is ~1.0 (100x larger), portret is in PERCENT form "
      "(divide by 100 first).")
print()
print("Paste this entire output back and I will finalize r4_cross_market.py")
