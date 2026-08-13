"""跑 PELT 检测变点（RBF cost + min_size 限制）。"""
import sys
sys.path.insert(0, "src")
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt
from data_loaders import download_spx

df = download_spx()
returns = df["log_return"].values

# RBF kernel + min_size=30 大幅加速
# pen 选 1（pen=100 会让变点过少，先广撒网看看）
print(f"\nRunning PELT (RBF, min_size=30) on {len(returns)} obs...")
t0 = time.time()
algo = rpt.Pelt(model="rbf", min_size=30).fit(returns.reshape(-1, 1))
breakpoints = algo.predict(pen=10)
print(f"Done in {time.time()-t0:.1f} sec")

breakpoints = breakpoints[:-1]  # 去掉末尾哨兵
breakpoint_dates = df.index[breakpoints]

print(f"Detected {len(breakpoint_dates)} change points:")
for d in breakpoint_dates:
    print(f"  {d.date()}")

# 已知 regime shift 事件
known_events = {
    "1990 recession": "1990-07-01",
    "1997 Asian crisis": "1997-07-02",
    "1998 LTCM": "1998-08-17",
    "2000 dot-com": "2000-03-24",
    "9/11": "2001-09-11",
    "2008 Lehman": "2008-09-15",
    "2020 COVID": "2020-03-16",
    "2022 Ukraine": "2022-02-24",
    "2025 Tariff": "2025-04-02",
}

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
axes[0].plot(df.index, df["close"], lw=0.7, color="black")
axes[0].set_ylabel("SPX (log scale)")
axes[0].set_yscale("log")
axes[0].set_title(f"PELT (RBF, pen=10, min_size=30): {len(breakpoint_dates)} change points")
for d in breakpoint_dates:
    axes[0].axvline(d, color="red", alpha=0.6, lw=1.0)
for label, date in known_events.items():
    axes[0].axvline(pd.Timestamp(date), color="blue", alpha=0.4, lw=1.5, linestyle="--")

axes[1].plot(df.index, df["log_return"], lw=0.3, color="gray")
axes[1].set_ylabel("Log return")
axes[1].axhline(0, color="black", lw=0.3, alpha=0.5)
for d in breakpoint_dates:
    axes[1].axvline(d, color="red", alpha=0.6, lw=1.0)

plt.tight_layout()
import os
os.makedirs("results", exist_ok=True)
plt.savefig("results/sanity_check_pelt.png", dpi=150, bbox_inches="tight")
print(f"\nSaved to results/sanity_check_pelt.png")
