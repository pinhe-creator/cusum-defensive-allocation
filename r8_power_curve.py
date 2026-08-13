"""
r8_power_curve.py

Generate the statistical-power figure for the central Sharpe-difference
comparison (CUSUM-fixed vs static 60/40). This visualizes the MDE argument
in section 5.5: the observed Sharpe difference of +0.063 lies well below
the effect size the two-decade sample can detect at conventional power.

Inputs are the quantities already reported in the paper (no recomputation
from raw returns needed):
  - paired standard error of the annualized Sharpe difference (Memmel): 0.0576
  - observed annualized Sharpe difference:                              0.0632
  - one-sided significance level alpha = 0.05
The 80%-power MDE (0.143) is recovered from these and marked.

The power of a one-sided test at true difference delta is
    power(delta) = Phi( delta / SE  -  z_{1-alpha} ),
the standard normal-approximation power function for a mean/Sharpe
difference with known standard error SE.

Output: figures/r8_power_curve.pdf
matplotlib text uses plain ASCII (no LaTeX escaping) to avoid the
backslash artifacts seen in some earlier figures.

Run locally.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import erf, sqrt


# ---------------------------------------------------------------------------
# Inputs (from the paper / r7 output)
# ---------------------------------------------------------------------------
SE = 0.0576          # paired SE of the annualized Sharpe difference (Memmel)
OBS = 0.0632         # observed Sharpe difference, CUSUM-fixed - static
ALPHA = 0.05         # one-sided significance level
TARGET_POWER = 0.80

def Phi(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def norm_ppf(q):
    # Acklam approximation (sufficient for plotting)
    a=[-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00]
    b=[-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01]
    c=[-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00]
    d=[7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00]
    pl=0.02425
    if q<pl:
        x=sqrt(-2*np.log(q)); return (((((c[0]*x+c[1])*x+c[2])*x+c[3])*x+c[4])*x+c[5])/((((d[0]*x+d[1])*x+d[2])*x+d[3])*x+1)
    if q<=1-pl:
        x=q-0.5; r=x*x
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*x/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    x=sqrt(-2*np.log(1-q)); return -(((((c[0]*x+c[1])*x+c[2])*x+c[3])*x+c[4])*x+c[5])/((((d[0]*x+d[1])*x+d[2])*x+d[3])*x+1)


z_alpha = norm_ppf(1 - ALPHA)                 # 1.6449
z_power = norm_ppf(TARGET_POWER)              # 0.8416
MDE = (z_alpha + z_power) * SE                # ~0.143

def power(delta):
    return Phi(delta / SE - z_alpha)

# observed-effect power, for annotation
obs_power = power(OBS)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 4.3))

deltas = np.linspace(0.0, 0.30, 400)
powers = np.array([power(d) for d in deltas])

ax.plot(deltas, powers, color="#1f4e79", lw=2.0, zorder=3,
        label="Power of the one-sided test (alpha = 0.05)")

# 80% power horizontal guide
ax.axhline(TARGET_POWER, color="0.55", ls="--", lw=1.0, zorder=1)
ax.text(0.298, TARGET_POWER + 0.012, "80% power", ha="right", va="bottom",
        fontsize=9, color="0.35")

# MDE vertical line
ax.plot([MDE, MDE], [0, TARGET_POWER], color="#2e7d32", ls="-", lw=1.4, zorder=2)
ax.plot([MDE], [TARGET_POWER], marker="o", color="#2e7d32", ms=6, zorder=4)
ax.text(MDE + 0.004, 0.33,
        f"Minimum detectable\neffect = {MDE:.3f}",
        ha="left", va="center", fontsize=9, color="#2e7d32")

# observed effect vertical line
ax.plot([OBS, OBS], [0, obs_power], color="#c62828", ls="-", lw=1.4, zorder=2)
ax.plot([OBS], [obs_power], marker="o", color="#c62828", ms=6, zorder=4)
ax.text(OBS + 0.004, obs_power + 0.08,
        f"Observed difference = {OBS:.3f}\n(power = {obs_power:.2f})",
        ha="left", va="bottom", fontsize=9, color="#c62828")

# shade the "undetectable" region [0, MDE)
ax.axvspan(0, MDE, color="#f2f2f2", zorder=0)

ax.set_xlim(0, 0.30)
ax.set_ylim(0, 1.0)
ax.set_xlabel("True annualized Sharpe-ratio difference")
ax.set_ylabel("Probability of detection (power)")
ax.set_title("Statistical power for the CUSUM-fixed vs static Sharpe difference",
             fontsize=10.5)
ax.legend(loc="lower right", fontsize=8.5, frameon=True)
ax.grid(True, alpha=0.25, lw=0.5)

fig.tight_layout()
os.makedirs("figures", exist_ok=True)
out = "figures/r8_power_curve.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"Wrote {out}")
print(f"  SE = {SE:.4f}, z_alpha = {z_alpha:.4f}, z_power = {z_power:.4f}")
print(f"  MDE (80% power, 5% one-sided) = {MDE:.4f}")
print(f"  observed = {OBS:.4f}, its power = {obs_power:.3f}")
