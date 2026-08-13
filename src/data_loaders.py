"""下载金融数据并存成 parquet。"""
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def download_spx(start="1990-01-01", end="2026-05-26", force=False):
    """下载 SPX 日度数据，计算 log return，存 parquet。"""
    out_path = DATA_DIR / "spx_daily.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        print(f"Loading cached {out_path}")
        return pd.read_parquet(out_path)

    print(f"Downloading SPX {start} to {end}...")
    df = yf.download("^GSPC", start=start, end=end,
                     auto_adjust=False, progress=False)

    # yfinance 新版本可能返回 MultiIndex columns，扁平化
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close"]].copy()
    df.columns = ["close"]
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna()

    df.to_parquet(out_path)
    print(f"Saved {len(df)} rows to {out_path}")
    return df


if __name__ == "__main__":
    df = download_spx()
    print()
    print(df.head())
    print()
    print(df.tail())
    print()
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Total observations: {len(df)}")
    print(f"Min log return: {df['log_return'].min():.4f} "
          f"on {df['log_return'].idxmin().date()}")
    print(f"Max log return: {df['log_return'].max():.4f} "
          f"on {df['log_return'].idxmax().date()}")
