"""
diag_dates.py  --  figure out why signals dates and SPY/IEF dates don't align.
Run locally, paste the full output back.
"""
import os, pandas as pd, numpy as np

SIGNALS = "results/portfolio_v2_signals.csv"

print("=== SIGNALS ===")
sig = pd.read_csv(SIGNALS)
print("columns:", list(sig.columns))
print("dtypes:\n", sig.dtypes)
sig["date"] = pd.to_datetime(sig["date"])
print("n rows:", len(sig))
print("date range:", sig["date"].min(), "->", sig["date"].max())
print("first 3 dates:", sig["date"].head(3).tolist())
print("dup dates:", int(sig["date"].duplicated().sum()))
print()

def show(path, label):
    if not os.path.exists(path):
        return False
    print(f"=== {label}  ({path}) ===")
    df = pd.read_parquet(path)
    print("columns:", list(df.columns))
    print("index name:", df.index.name, "| index dtype:", df.index.dtype)
    print("dtypes:\n", df.dtypes)
    print("n rows:", len(df))
    print("head:\n", df.head(3))
    # try to locate a date
    if "date" in df.columns:
        d = pd.to_datetime(df["date"])
    else:
        d = pd.to_datetime(df.reset_index().iloc[:,0], errors="coerce")
    print("inferred date range:", d.min(), "->", d.max())
    print("first 3 inferred dates:", d.head(3).tolist())
    print("dup dates:", int(d.duplicated().sum()))
    print()
    return d

spy = ief = None
for p in ["data/spy_daily.parquet","data/SPY.parquet"]:
    if os.path.exists(p): spy = show(p, "SPY"); break
for p in ["data/ief_daily.parquet","data/IEF.parquet"]:
    if os.path.exists(p): ief = show(p, "IEF"); break

if spy is None:
    print("!! no SPY parquet found. ls data/:")
    print([f for f in os.listdir("data")] if os.path.isdir("data") else "no data/ dir")

# overlap test
if spy is not None:
    s_dates = set(pd.to_datetime(sig["date"]).dt.normalize())
    p_dates = set(pd.to_datetime(spy).dt.normalize())
    print("=== OVERLAP (normalized to midnight) ===")
    print("signals dates:", len(s_dates), "| SPY dates:", len(p_dates))
    print("intersection:", len(s_dates & p_dates))
    print("signals-only (first 5):", sorted(list(s_dates - p_dates))[:5])
    print("SPY-only (first 5):", sorted(list(p_dates - s_dates))[:5])
