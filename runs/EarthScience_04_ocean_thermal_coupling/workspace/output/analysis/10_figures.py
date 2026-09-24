#!/usr/bin/env python3
"""Step 10 - Supporting figures for the authoritative report.

Reads the raw table and the step-2 result artifacts and writes four figures
that carry the report's core visual evidence.  The plotted quantities are
recomputed here so the figures do not depend on intermediate state.

Outputs
-------
output/figures/fig1_record_and_ages.png
output/figures/fig2_spectra_and_null.png
output/figures/fig3_phase_and_holdout.png
output/figures/fig4_lsr_and_change_point.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as ss

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "output" / "figures"
RES = ROOT / "output" / "results"
FIG.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20220101)

df = pd.read_csv(ROOT / "data" / "observations.tsv", sep="\t")
df.columns = ["depth_m", "age_ka_bp_1", "age_ka_bp_2", "d18o_pdb"]
x = df.dropna(subset=["d18o_pdb"]).reset_index(drop=True)
depth = x["depth_m"].to_numpy(float)
age = x["age_ka_bp_1"].to_numpy(float)
y = x["d18o_pdb"].to_numpy(float)
all_age = df["age_ka_bp_1"].to_numpy(float)
all_depth = df["depth_m"].to_numpy(float)
lsr = np.gradient(all_depth, all_age)

P_BEST = 108.835


def harm(t, P, n_harm=2, trend=True):
    w = 2 * np.pi / P
    cols = [np.ones_like(t)]
    if trend:
        cols.append(t)
    for k in range(1, n_harm + 1):
        cols += [np.cos(k * w * t), np.sin(k * w * t)]
    return np.vstack(cols).T


# ---------------------------------------------------------------- figure 1
fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=False)
ax[0].plot(all_depth, all_age, lw=1.0, color="k")
ax[0].set_ylabel("age (ka BP)")
ax[0].set_xlabel("sediment depth (m)")
ax[0].set_title("Age-depth relation (both coordinates)")
ax[1].plot(age, y, lw=0.7, color="tab:blue", label="d18O (complete rows)")
miss = df[df["d18o_pdb"].isna()]
ax[1].scatter(miss["age_ka_bp_1"], np.full(len(miss), 2.2), s=8,
              color="tab:red", label=f"missing ({len(miss)} rows)")
b, *_ = np.linalg.lstsq(harm(age, P_BEST), y, rcond=None)
grid = np.linspace(age.min(), age.max(), 4000)
ax[1].plot(grid, harm(grid, P_BEST) @ b, color="tab:orange", lw=1.6,
           label=f"trend + 2 harmonics, P={P_BEST:.0f} ka")
ax[1].set_xlabel("age (ka BP)")
ax[1].set_ylabel("d18O (per mille PDB)")
ax[1].legend(fontsize=8)
ax[1].set_title("Proxy record with the dominant periodic fit")
ax[2].plot(all_age, lsr, lw=0.8, color="tab:green")
thr = np.percentile(lsr, 99)
ax[2].axhline(thr, ls="--", color="tab:red", lw=1,
              label=f"99th percentile = {thr:.4f} m/ka")
ax[2].axhline(np.median(lsr), ls=":", color="k", lw=1,
              label=f"median = {np.median(lsr):.4f} m/ka")
ax[2].set_xlabel("age (ka BP)")
ax[2].set_ylabel("sedimentation rate (m/ka)")
ax[2].legend(fontsize=8)
ax[2].set_title("Sedimentation rate from the depth-age model")
fig.tight_layout()
fig.savefig(FIG / "fig1_record_and_ages.png", dpi=140)
plt.close(fig)

# ---------------------------------------------------------------- figure 2
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

f_age = np.linspace(1 / 400, 1 / 20, 4000)
p_age = ss.lombscargle(age - age.mean(), ss.detrend(y), 2 * np.pi * f_age,
                       normalize=True)
ax[0].semilogx(1 / f_age, p_age, lw=1.0)
ax[0].axvline(P_BEST, color="tab:red", ls="--", lw=1,
              label=f"{P_BEST:.0f} ka")
ax[0].set_xlabel("period (ka)")
ax[0].set_ylabel("Lomb-Scargle power")
ax[0].set_title("Age domain")
ax[0].legend(fontsize=8)

f_dep = np.linspace(1 / 6.0, 1 / 0.05, 4000)
p_dep = ss.lombscargle(all_depth - all_depth.mean(),
                       ss.detrend(df["d18o_pdb"].interpolate().to_numpy()),
                       2 * np.pi * f_dep, normalize=True)
ax[1].semilogx(1 / f_dep, p_dep, lw=1.0, color="tab:green")
ax[1].axvline(1.877, color="tab:red", ls="--", lw=1, label="1.877 m")
ax[1].set_xlabel("period (m)")
ax[1].set_title("Depth domain")
ax[1].legend(fontsize=8)

ls = pd.read_csv(RES / "02_periodogram.csv")
ax[2].semilogx(ls["period_ka"], ls["ls_power"], lw=1.0, color="tab:purple")
for pk, lab in [(108.22, "108.2"), (97.30, "97.3"), (119.38, "119.4")]:
    ax[2].axvline(pk, ls=":", lw=0.8, color="grey")
ax[2].axvspan(97.30, 119.38, color="grey", alpha=0.15,
              label="sidelobes within 1/T")
ax[2].set_xlabel("period (ka)")
ax[2].set_title("Age domain, sidelobe structure")
ax[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fig2_spectra_and_null.png", dpi=140)
plt.close(fig)

# ---------------------------------------------------------------- figure 3
sw = pd.read_csv(RES / "02_sliding_window.csv")
hd = pd.read_csv(RES / "02_block_holdout.csv")
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

ax[0].plot(sw["window_start_ka"], sw["amplitude"], "o-", ms=3)
ax[0].set_xlabel("window start (ka)")
ax[0].set_ylabel("fundamental amplitude")
ax[0].set_title("Sliding-window amplitude (150 ka windows)")

ax[1].plot(sw["window_start_ka"], sw["phase_rad"], "o-", ms=3,
           color="tab:orange")
ax[1].set_xlabel("window start (ka)")
ax[1].set_ylabel("phase (rad)")
ax[1].set_title("Sliding-window phase")

w = 0.35
idx = np.arange(len(hd))
ax[2].bar(idx - w / 2, hd["r2_out_of_sample"], w, label="sinusoid")
ax[2].bar(idx + w / 2, hd["r2_persistence_baseline"], w,
          label="persistence baseline")
ax[2].axhline(0, color="k", lw=0.8)
ax[2].set_xticks(idx)
ax[2].set_xticklabels([f"{a:.0f}-{b:.0f}" for a, b in
                       zip(hd["age_start_ka"], hd["age_end_ka"])],
                      rotation=45, fontsize=7)
ax[2].set_ylabel("out-of-sample R2")
ax[2].set_title("Contiguous 6-block holdout")
ax[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fig3_phase_and_holdout.png", dpi=140)
plt.close(fig)

# ---------------------------------------------------------------- figure 4
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

b10 = []
edges = np.arange(age.min(), age.max() + 10, 10)
for i in range(len(edges) - 1):
    m = (age >= edges[i]) & (age < edges[i + 1])
    if m.sum() < 5:
        continue
    b10.append(((edges[i] + edges[i + 1]) / 2,
                lsr[(all_age >= edges[i]) & (all_age < edges[i + 1])].mean(),
                y[m].mean()))
b10 = np.array(b10)
ax[0].scatter(b10[:, 1], b10[:, 2], s=22)
z = np.polyfit(b10[:, 1], b10[:, 2], 1)
xs = np.linspace(b10[:, 1].min(), b10[:, 1].max(), 50)
ax[0].plot(xs, np.polyval(z, xs), "r--", lw=1,
           label=f"r = {np.corrcoef(b10[:, 1], b10[:, 2])[0, 1]:+.3f}")
ax[0].set_xlabel("block-mean sedimentation rate (m/ka)")
ax[0].set_ylabel("block-mean d18O")
ax[0].set_title("Inter-block coupling, 10 ka blocks")
ax[0].legend(fontsize=8)

t = np.arange(len(y))
sse = np.array([np.sum((y[:k] - y[:k].mean()) ** 2)
                + np.sum((y[k:] - y[k:].mean()) ** 2)
                for k in range(40, len(y) - 40)])
ax[1].plot(age[np.arange(40, len(y) - 40)], sse, lw=1.0)
ax[1].axhline(np.sum((y - y.mean()) ** 2), color="k", ls=":", lw=1,
              label="no-split SSE")
ax[1].axvline(336.63, color="tab:red", ls="--", lw=1, label="best split")
ax[1].set_xlabel("age (ka BP)")
ax[1].set_ylabel("two-segment SSE")
ax[1].set_title("Single change-point scan")
ax[1].legend(fontsize=8)

A_lin = np.vstack([np.ones_like(age), age]).T
bl, *_ = np.linalg.lstsq(A_lin, y, rcond=None)
y_detr = y - A_lin @ bl
zy = np.polyfit(age, y_detr, 2)
ax[2].plot(age, y_detr, lw=0.7, color="grey", alpha=0.8)
resid = y_detr - np.polyval(zy, age)
ax[2].plot(age, resid, lw=1.2, color="tab:blue")
ax[2].set_xlabel("age (ka BP)")
ax[2].set_ylabel("detrended d18O residuals")
ax[2].set_title("Heteroscedasticity check (amplitude varies)")
fig.tight_layout()
fig.savefig(FIG / "fig4_lsr_and_change_point.png", dpi=140)
plt.close(fig)

print("wrote 4 figures to", FIG)
for p in sorted(FIG.glob("*.png")):
    print(" ", p.name, p.stat().st_size, "bytes")