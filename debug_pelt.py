"""诊断 PELT 为什么检测不到变点。"""
import sys
sys.path.insert(0, "src")

import numpy as np
import ruptures as rpt
from data_loaders import download_spx

df = download_spx()
returns = df["log_return"].values

print("=" * 50)
print("数据基本检查")
print("=" * 50)
print(f"returns shape: {returns.shape}")
print(f"returns dtype: {returns.dtype}")
print(f"any NaN? {np.isnan(returns).any()}")
print(f"any inf? {np.isinf(returns).any()}")
print(f"mean: {returns.mean():.6f}")
print(f"std: {returns.std():.6f}")
print(f"min: {returns.min():.6f}")
print(f"max: {returns.max():.6f}")

print()
print("=" * 50)
print("PELT 测试：在已知有变点的合成数据上")
print("=" * 50)
# 先在一个绝对应该检测出来的合成信号上测试 PELT 本身
# 信号:前 500 点 N(0,1),后 500 点 N(5,1) ——均值跳变明显
synth = np.concatenate([
    np.random.RandomState(0).randn(500),
    np.random.RandomState(1).randn(500) + 5
])
print(f"合成信号 shape: {synth.shape}, 真实变点在 t=500")
algo_synth = rpt.Pelt(model="l2").fit(synth.reshape(-1, 1))
bp_synth = algo_synth.predict(pen=10)
print(f"PELT 检测到的变点 (pen=10): {bp_synth}")
print(f"(末尾应为 1000;中间应接近 500)")

print()
print("=" * 50)
print("真实 SPX returns 上扫描多个 penalty")
print("=" * 50)
returns_2d = returns.reshape(-1, 1)
print(f"reshape 后: {returns_2d.shape}")

algo_real = rpt.Pelt(model="l2").fit(returns_2d)
for pen in [0.001, 0.01, 0.1, 1, 10, 100]:
    bp = algo_real.predict(pen=pen)
    print(f"  pen={pen:>7}: 检测到 {len(bp)-1} 个变点")

print()
print("=" * 50)
print("用 RBF kernel 试一下")
print("=" * 50)
algo_rbf = rpt.Pelt(model="rbf").fit(returns_2d)
for pen in [0.001, 0.01, 0.1, 1, 10, 100]:
    bp = algo_rbf.predict(pen=pen)
    print(f"  pen={pen:>7}: 检测到 {len(bp)-1} 个变点")

print()
print("=" * 50)
print("放大 returns（乘 1000）再试 L2")
print("=" * 50)
returns_scaled = (returns * 1000).reshape(-1, 1)
print(f"放大后 std: {returns_scaled.std():.4f}")
algo_scaled = rpt.Pelt(model="l2").fit(returns_scaled)
for pen in [0.01, 1, 100, 1000]:
    bp = algo_scaled.predict(pen=pen)
    print(f"  pen={pen:>5}: 检测到 {len(bp)-1} 个变点")
