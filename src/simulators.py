"""4 类合成 DGP + 2 类 null DGP，用于 CPD 算法基准测试。

每个生成函数返回:
    series : np.ndarray, shape (n_obs,)
    true_changepoints : list[int], 真实变点位置（exclusive end）

变点约定（与 ruptures 一致）:
    - 序列被切成段 [0, tau_1), [tau_1, tau_2), ..., [tau_K, n_obs)
    - true_changepoints = [tau_1, tau_2, ..., tau_K]

DGP 一览:
    DGP-0a: Gaussian null（无变点，用于 FP 评估）
    DGP-0b: Student-t null（重尾无变点，重尾 FP 评估）
    DGP-1:  Gaussian mean shift
    DGP-2:  Gaussian variance shift（仅波动率变，均值不变）
    DGP-3:  Student-t mean shift（重尾稳健性测试，方差已标准化）
    DGP-4:  GARCH(1,1) regime switching（最接近真实金融数据）
"""
import numpy as np


# ---- 标准化设定 ----
DEFAULT_N_OBS = 1000
DEFAULT_BREAKPOINTS = [250, 500, 750]  # 3 个真实变点，分成 4 段


# ====================================================================
# Null DGPs（无变点，用于 false positive 评估）
# ====================================================================

def dgp0_gaussian_null(n_obs=DEFAULT_N_OBS, mu=0.0, sigma=1.0, seed=None,
                       breakpoints=None):
    """DGP-0a: 无变点 Gaussian null。

    用于评估算法在"应该无报警"的场景下的 false positive rate。
    breakpoints 参数被忽略（保持接口一致）。
    """
    rng = np.random.default_rng(seed)
    series = rng.normal(loc=mu, scale=sigma, size=n_obs)
    return series, []


def dgp0_student_t_null(n_obs=DEFAULT_N_OBS, df=3, scale=1.0,
                        standardize=True, seed=None, breakpoints=None):
    """DGP-0b: 无变点 Student-t null（重尾噪声）。

    用于评估算法对重尾极端值的 false positive 敏感度。
    这是金融最有意义的 null —— 真实 returns 是重尾的，
    算法不应在 isolated outlier 上误报变点。
    """
    if df <= 2 and standardize:
        raise ValueError("Cannot variance-standardize Student-t with df <= 2")
    rng = np.random.default_rng(seed)
    x = rng.standard_t(df=df, size=n_obs)
    if standardize:
        x = x / np.sqrt(df / (df - 2))
    return scale * x, []


# ====================================================================
# DGP-1: Gaussian mean shift
# ====================================================================

def dgp1_gaussian_mean(n_obs=DEFAULT_N_OBS,
                       breakpoints=None,
                       means=(0.0, 1.0, -0.5, 0.8),
                       sigma=1.0,
                       seed=None):
    """DGP-1: 高斯均值变化。每段均值不同，方差相同。

    这是最简单的 baseline，L2 cost 应该擅长检测此场景。
    """
    if breakpoints is None:
        breakpoints = DEFAULT_BREAKPOINTS
    assert len(means) == len(breakpoints) + 1, \
        f"need {len(breakpoints)+1} means for {len(breakpoints)} breakpoints"

    rng = np.random.default_rng(seed)
    segments_ends = list(breakpoints) + [n_obs]
    starts = [0] + list(breakpoints)

    series = np.empty(n_obs)
    for mu, s, e in zip(means, starts, segments_ends):
        series[s:e] = rng.normal(loc=mu, scale=sigma, size=e - s)
    return series, list(breakpoints)


# ====================================================================
# DGP-2: Gaussian variance shift
# ====================================================================

def dgp2_gaussian_variance(n_obs=DEFAULT_N_OBS,
                           breakpoints=None,
                           sigmas=(1.0, 2.0, 0.5, 1.5),
                           mu=0.0,
                           seed=None):
    """DGP-2: 高斯方差变化。每段方差不同，均值相同（0）。

    金融最常见场景:vol regime shift。
    注意:对 PELT(model="l2") 不友好，因为 L2 cost 只检测均值变化。
    适合算法:PELT(model="rbf"), PELT(model="normal"), kernel CPD,
    CUSUM on squared/absolute values。
    """
    if breakpoints is None:
        breakpoints = DEFAULT_BREAKPOINTS
    assert len(sigmas) == len(breakpoints) + 1

    rng = np.random.default_rng(seed)
    segments_ends = list(breakpoints) + [n_obs]
    starts = [0] + list(breakpoints)

    series = np.empty(n_obs)
    for sig, s, e in zip(sigmas, starts, segments_ends):
        series[s:e] = rng.normal(loc=mu, scale=sig, size=e - s)
    return series, list(breakpoints)


# ====================================================================
# DGP-3: Student-t (heavy-tailed)
# ====================================================================

def dgp3_student_t(n_obs=DEFAULT_N_OBS,
                   breakpoints=None,
                   means=(0.0, 1.0, -0.5, 0.8),
                   df=3,
                   scale=1.0,
                   standardize=True,
                   seed=None):
    """DGP-3: Student-t 重尾，均值切换。

    df=3 表示厚尾。若 standardize=True 且 df>2，
    则将 t 分布缩放到单位方差，使其与 Gaussian DGP 的
    主要差异来自尾部厚度，而不是整体方差。

    这是 RQ2（重尾稳健性）的核心 DGP。
    """
    if breakpoints is None:
        breakpoints = DEFAULT_BREAKPOINTS
    assert len(means) == len(breakpoints) + 1
    if df <= 2 and standardize:
        raise ValueError("Cannot variance-standardize Student-t with df <= 2")

    rng = np.random.default_rng(seed)
    segments_ends = list(breakpoints) + [n_obs]
    starts = [0] + list(breakpoints)

    series = np.empty(n_obs)
    for mu, s, e in zip(means, starts, segments_ends):
        x = rng.standard_t(df=df, size=e - s)
        if standardize:
            x = x / np.sqrt(df / (df - 2))
        series[s:e] = mu + scale * x
    return series, list(breakpoints)


# ====================================================================
# DGP-4: GARCH(1,1) regime switching
# ====================================================================

def dgp4_garch_switching(n_obs=DEFAULT_N_OBS,
                         breakpoints=None,
                         regimes=None,
                         seed=None,
                         reset_at_breaks=True):
    """DGP-4: GARCH(1,1) 参数切换。最接近真实金融数据的合成 DGP。

    每段用不同 (omega, alpha, beta) 模拟 GARCH(1,1):
        sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2
        eps_t = sigma_t * z_t,  z_t ~ N(0,1)

    默认 regimes 取自典型金融实证文献:
        - 平静期: omega=0.05, alpha=0.05, beta=0.90  (persistent low vol)
        - 危机期: omega=0.50, alpha=0.15, beta=0.80  (high vol, shock-reactive)
        - 恢复期: omega=0.10, alpha=0.08, beta=0.88
        - 后危机: omega=0.20, alpha=0.10, beta=0.85

    reset_at_breaks=True (默认):
        每段从该 regime 的无条件方差重新开始，产生清晰的 sharp changepoint。
        让 detection delay 完全归因于算法本身，不含 DGP 自身的平滑过渡。
    reset_at_breaks=False:
        上一段的 volatility state 传递到下一段，更接近真实金融波动惯性。
        但检测延迟会部分来自 DGP 的平滑过渡，不完全是算法的延迟。
    Robustness check 时建议两种都跑。
    """
    if breakpoints is None:
        breakpoints = DEFAULT_BREAKPOINTS
    if regimes is None:
        regimes = [
            (0.05, 0.05, 0.90),  # calm
            (0.50, 0.15, 0.80),  # crisis
            (0.10, 0.08, 0.88),  # recovery
            (0.20, 0.10, 0.85),  # post-crisis
        ]
    assert len(regimes) == len(breakpoints) + 1

    rng = np.random.default_rng(seed)
    segments_ends = list(breakpoints) + [n_obs]
    starts = [0] + list(breakpoints)
    series = np.empty(n_obs)
    sigma2_prev = None
    eps_prev = 0.0

    for (omega, alpha, beta), s, e in zip(regimes, starts, segments_ends):
        # 参数稳态性检查（保留 IGARCH 的可能性, 仅拒绝爆炸性参数）
        if omega <= 0 or alpha < 0 or beta < 0:
            raise ValueError(
                f"GARCH parameters require omega>0, alpha>=0, beta>=0, "
                f"got omega={omega}, alpha={alpha}, beta={beta}"
            )
        if alpha + beta > 1:
            raise ValueError(
                f"Non-stationary (explosive) GARCH regime: "
                f"alpha+beta={alpha + beta} > 1"
            )
        # 重置或继承 volatility state
        if reset_at_breaks or sigma2_prev is None:
            if alpha + beta < 1:
                sigma2_prev = omega / (1 - alpha - beta)
            else:
                # IGARCH 边界情形:无良定义无条件方差，用 omega 初始化
                sigma2_prev = omega
            eps_prev = 0.0

        for t in range(s, e):
            sigma2_t = omega + alpha * eps_prev**2 + beta * sigma2_prev
            sigma2_t = max(sigma2_t, 1e-12)
            eps_t = np.sqrt(sigma2_t) * rng.standard_normal()
            series[t] = eps_t
            sigma2_prev = sigma2_t
            eps_prev = eps_t

    return series, list(breakpoints)


# ====================================================================
# 批量生成函数
# ====================================================================

def generate_dgp_panel(dgp_name, n_replications=1000, n_obs=DEFAULT_N_OBS,
                       breakpoints=None, base_seed=0, **kwargs):
    """为给定 DGP 批量生成 n_replications 条独立序列。

    所有 replications 共享相同 breakpoints（用 consistency check 强制）。
    未来若加入 random-breakpoints DGP，consistency check 会立即失败而非
    悄无声息地返回错误值。

    返回:
        all_series : np.ndarray, shape (n_replications, n_obs)
        true_changepoints : list[int], 所有序列共享同一组真实变点
    """
    dgp_funcs = {
        "dgp0_gaussian": dgp0_gaussian_null,
        "dgp0_student_t": dgp0_student_t_null,
        "dgp1": dgp1_gaussian_mean,
        "dgp2": dgp2_gaussian_variance,
        "dgp3": dgp3_student_t,
        "dgp4": dgp4_garch_switching,
    }
    if dgp_name not in dgp_funcs:
        raise ValueError(
            f"Unknown DGP: {dgp_name}. "
            f"Available: {list(dgp_funcs.keys())}"
        )
    fn = dgp_funcs[dgp_name]

    all_series = np.empty((n_replications, n_obs))
    true_bps = None

    for i in range(n_replications):
        # null DGPs 接受 breakpoints 但忽略它
        series, bps = fn(n_obs=n_obs, breakpoints=breakpoints,
                         seed=base_seed + i, **kwargs)
        if true_bps is None:
            true_bps = bps
        elif bps != true_bps:
            raise ValueError(
                f"DGP {dgp_name} returned inconsistent breakpoints across "
                f"replications: {bps} vs {true_bps}"
            )
        all_series[i] = series

    return all_series, true_bps


# ====================================================================
# Self-test
# ====================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os

    print("Testing all 6 DGPs...\n")

    test_specs = [
        ("DGP-0a: Gaussian null", dgp0_gaussian_null, {}),
        ("DGP-0b: Student-t null (df=3, standardized)", dgp0_student_t_null, {}),
        ("DGP-1: Gaussian mean shift", dgp1_gaussian_mean, {}),
        ("DGP-2: Gaussian variance shift", dgp2_gaussian_variance, {}),
        ("DGP-3: Student-t (df=3) mean shift, standardized",
         dgp3_student_t, {}),
        ("DGP-4: GARCH(1,1) regime switching (reset_at_breaks=True)",
         dgp4_garch_switching, {}),
    ]

    fig, axes = plt.subplots(len(test_specs), 1, figsize=(12, 14), sharex=True)

    for ax, (name, fn, extra_kwargs) in zip(axes, test_specs):
        series, bps = fn(seed=42, **extra_kwargs)
        ax.plot(series, lw=0.5, color="black")
        for bp in bps:
            ax.axvline(bp, color="red", alpha=0.5, lw=1)
        title_bps = f"true CPs={bps}" if bps else "no true CPs"
        ax.set_title(f"{name}  ({title_bps})")
        ax.set_ylabel("Value")
        print(f"{name}")
        print(f"  shape={series.shape}, mean={series.mean():.3f}, "
              f"std={series.std():.3f}, true CPs={bps}")
        if bps == []:
            # null DGP: 报告极端值统计
            print(f"  |max|={np.abs(series).max():.3f}, "
                  f"#|x|>3 = {(np.abs(series) > 3).sum()}")
        print()

    axes[-1].set_xlabel("t")
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/dgp_self_test.png", dpi=120, bbox_inches="tight")
    print("Saved diagnostic plot to results/dgp_self_test.png\n")

    # 测试 generate_dgp_panel
    print("Testing batch generation...")
    for dgp_name in ["dgp0_gaussian", "dgp0_student_t",
                     "dgp1", "dgp2", "dgp3", "dgp4"]:
        panel, bps = generate_dgp_panel(dgp_name, n_replications=50)
        print(f"  {dgp_name:>15}: panel.shape={panel.shape}, "
              f"true_bps={bps}, "
              f"std range=[{panel.std(axis=1).min():.3f}, "
              f"{panel.std(axis=1).max():.3f}]")

    # 测试 GARCH 的 reset_at_breaks 对比
    print("\nGARCH reset_at_breaks comparison (single series, seed=42):")
    s_reset, _ = dgp4_garch_switching(seed=42, reset_at_breaks=True)
    s_noreset, _ = dgp4_garch_switching(seed=42, reset_at_breaks=False)
    # 第 2 段开始 20 天的 std (应能看出 reset=True 更快进入危机水平)
    s_reset_post = s_reset[250:270].std()
    s_noreset_post = s_noreset[250:270].std()
    print(f"  Post-break (t=250..270) std with reset=True:  {s_reset_post:.3f}")
    print(f"  Post-break (t=250..270) std with reset=False: {s_noreset_post:.3f}")
    print(f"  Expected: reset=True should show higher std faster "
          f"(crisis regime omega/(1-alpha-beta)={0.50/(1-0.15-0.80):.1f})")
