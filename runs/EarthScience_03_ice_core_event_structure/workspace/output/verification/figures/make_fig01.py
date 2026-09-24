#!/usr/bin/env python3
"""Independent production of fig01_record_overview.png.

Re-reads the raw TSV from scratch (no reuse of output/analysis code) and draws a
4-panel stacked record overview with a shared, inverted depth axis:

  panel 1  c04_sulfate_ug_kg   vs c01_depth_top_m  (log x), annotated horizons red
  panel 2  c05_density_ice_g_cm_3 vs c01_depth_top_m
  panel 3  c06_depth_w_e_m     vs c01_depth_top_m  (+ identity annotation)
  panel 4  c03_samp_thick_cm   vs c01_depth_top_m

Output: <project>/output/figures/fig01_record_overview.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ = "/data/user/conversations/conv-f07721cb-d3dc-42e9-83c7-7e986203115b/project"
DATA = os.path.join(PROJ, "data", "observations.tsv")
OUT = os.path.join(PROJ, "output", "figures", "fig01_record_overview.png")

# ---------------------------------------------------------------- load raw data
df = pd.read_csv(DATA, sep="\t")
lab = df["c07_signal"].fillna("").astype(str).to_numpy()  # labels FIRST (9 non-numeric)
for c in df.columns:
    if c != "c07_signal":
        df[c] = pd.to_numeric(df[c], errors="coerce")

depth = df["c01_depth_top_m"].to_numpy(float)
sulf = df["c04_sulfate_ug_kg"].to_numpy(float)
dens = df["c05_density_ice_g_cm_3"].to_numpy(float)
dwe = df["c06_depth_w_e_m"].to_numpy(float)
thick = df["c03_samp_thick_cm"].to_numpy(float)
ann = np.array([s.strip() != "" for s in lab])
n_ann = int(ann.sum())
print(f"rows={len(df)}  annotated_horizons={n_ann}  sulf_positive={(sulf > 0).sum()}")

# ------------------------------------------------- independent key numbers (raw)
resid_wed = dwe - depth * dens
wed_max = np.nanmax(np.abs(resid_wed))
print(f"c06 - c01*c05 : max|residual| = {wed_max:.6f} m, mean = {np.nanmean(resid_wed):.6f} m")

# depth gap / step: largest raw jump in c01 and the best-fit step from the
# c01 vs cumulative-thickness profile.  Cumulative thickness must EXCLUDE the
# current interval (top of interval i = sum of thicknesses of intervals above
# it), i.e. cum_th[i] = sum(thick[:i])/100; an exhaustive split scan then finds
# a single additive step, refined on a fine window.
gaps = np.diff(depth)
gf = int(np.argmax(gaps))
print(f"largest raw depth gap: {gaps[gf]:.4f} m between 0-based rows {gf} and {gf+1} "
      f"({depth[gf]:.2f} -> {depth[gf+1]:.2f} m)")

cum_th = np.concatenate([[0.0], np.cumsum(thick[:-1])]) / 100.0
base = depth - cum_th
best_s, best_step = 0, 0.0
for s in range(1, len(base)):
    step = base[s:].mean() - base[:s].mean()
    if abs(step) > abs(best_step):
        best_s, best_step = s, step
best_r, best_err = best_s, np.inf
for s in range(max(1, best_s - 60), min(len(base), best_s + 60)):
    r = base - (base[s:].mean() - base[:s].mean()) * (np.arange(len(base)) >= s)
    err = np.abs(r).max()
    if err < best_err:
        best_r, best_err = s, err
step_m = float(base[best_r:].mean() - base[:best_r].mean())
step_depth = float(depth[best_r - 1])
step_row = best_r  # 0-based data row that starts the stepped segment
print(f"fitted step = {step_m:.6f} m at split index {best_r} "
      f"(depth {depth[best_r-1]:.2f} -> {depth[best_r]:.2f} m), "
      f"max|residual| after removal = {best_err:.6f} m")

# ---------------------------------------------------------------- figure layout
fig, axes = plt.subplots(4, 1, figsize=(11, 12), dpi=140, sharey=True)
ax1, ax2, ax3, ax4 = axes

# ---- panel 1: sulfate (log x), annotated horizons as red points
pos = sulf > 0
ax1.plot(sulf[pos], depth[pos], "-", color="#1f4e79", lw=0.7, zorder=2,
         label="c04 sulfate")
ax1.plot(sulf[ann & pos], depth[ann & pos], "o", color="red", ms=5.5, zorder=4,
         mec="darkred", mew=0.6, label=f"c07 annotated horizons (n={n_ann})")
ax1.axhline(step_depth, color="k", ls="--", lw=1.2, zorder=1)
ax1.text(0.015, 0.035,
         f"1.008 m depth step at row 105 (depth {step_depth:.2f} m)",
         transform=ax1.transAxes, fontsize=8, va="bottom", ha="left", color="k",
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=0.85))
ax1.set_xscale("log")
ax1.set_xlabel("c04 sulfate (µg kg$^{-1}$), log scale")
ax1.legend(loc="center right", fontsize=8, framealpha=0.92)
ax1.set_ylim(depth.max() + 0.3, depth.min() - 0.3)  # inverted

# ---- panel 2: density
ax2.plot(dens, depth, "-", color="#2e7d32", lw=0.9)
ax2.set_xlabel("c05 density (g cm$^{-3}$)")
ax2.set_ylabel("")

# ---- panel 3: water-equivalent depth + identity annotation
ax3.plot(dwe, depth, "-", color="#b26a00", lw=0.9)
ax3.annotate(f"c06 = c01 x c05 to within {wed_max:.4f} m",
             xy=(0.98, 0.95), xycoords="axes fraction",
             ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.92))
ax3.set_xlabel("c06 depth water equivalent (m)")
ax3.set_ylabel("")

# ---- panel 4: sample thickness
ax4.plot(thick, depth, "-", color="#6a1b9a", lw=0.9)
ax4.set_xlabel("c03 sample thickness (cm)")
ax4.set_ylabel("")

for ax in axes:
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.45)
    ax.tick_params(labelsize=8.5)
    ax.set_ylabel("c01 depth\ntop (m)", fontsize=8.5)
axes[0].set_ylabel("c01 depth top (m)", fontsize=9)

fig.suptitle("Fig. 1  Ice-core record overview: sulfate, density, water-equivalent "
             "depth and sample thickness vs. c01 depth", fontsize=11.5)
fig.tight_layout(rect=(0, 0.005, 1, 0.975))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT)
plt.close(fig)

size = os.path.getsize(OUT)
print(f"OUTPUT_PATH={OUT}")
print(f"FILE_SIZE_BYTES={size}")