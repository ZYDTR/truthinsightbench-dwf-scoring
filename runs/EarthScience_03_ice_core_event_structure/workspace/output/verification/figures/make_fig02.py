#!/usr/bin/env python3
"""
make_fig02.py -- production figure `fig02_event_definition_sensitivity.png`.

Independent, from-scratch plotting script (does not import or re-run anything in
output/analysis/).  Inputs, read only:

  data/observations.tsv                 -- raw 1931 x 9 table
  output/results/detector_grid.csv      -- the already-computed 144 detector variants

Panels
------
(a) top        : the 15 c07-annotated horizons as sulfate value vs depth, coloured
                 by label class (event_01..05 / unidentified_event / reference_surface),
                 over the full sulfate profile drawn faintly for context.
(b) bottom-left: distribution of `n_detected` over the 144 detector variants, with the
                 median marked.  The x-axis is log-scaled because the distribution is
                 strongly right-skewed (4 .. 247 peaks).
(c) bottom-right: number of detected peaks vs number of annotated horizons recalled.
                 Horizontal line = maximum achievable recall (14/15, the local-maximum
                 ceiling of the annotated set).  Vertical line marks the local-maximum
                 ceiling point on the detection axis: the smallest n_detected for which
                 the ceiling recall 14/15 is attained.

Output: output/figures/fig02_event_definition_sensitivity.png  (figsize 12x9, dpi 140)
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = "/data/user/conversations/conv-f07721cb-d3dc-42e9-83c7-7e986203115b/project"
OBS = os.path.join(PROJECT, "data", "observations.tsv")
GRID = os.path.join(PROJECT, "output", "results", "detector_grid.csv")
OUTDIR = os.path.join(PROJECT, "output", "figures")
OUTPNG = os.path.join(OUTDIR, "fig02_event_definition_sensitivity.png")

EVENT_COLOR = "#c0392b"   # event_01 .. event_05
UNID_COLOR = "#2471a3"    # unidentified_event
REF_COLOR = "#1e8449"     # reference_surface
PROFILE_COLOR = "#b0b0b0"

# ----------------------------------------------------------------------------- data
obs = pd.read_csv(OBS, sep="\t")
ann = obs[obs["c07_signal"].notna()].copy()
ann["class"] = np.where(
    ann["c07_signal"].str.startswith("event_"),
    "event_01..05",
    ann["c07_signal"],
)
n_ann = len(ann)
n_events = int((ann["class"] == "event_01..05").sum())
n_unid = int((ann["class"] == "unidentified_event").sum())
n_ref = int((ann["class"] == "reference_surface").sum())

grid = pd.read_csv(GRID)
n_var = len(grid)
nd = grid["n_detected"].to_numpy(float)
recalled = grid["n_annotated_recalled"].to_numpy(float)
median_nd = float(np.median(nd))

# Local-maximum ceiling = max annotated horizons any local-maximum rule can return.
max_recall = int(recalled.max())          # observed maximum over the grid
ceiling_frac = max_recall / n_ann
# Smallest n_detected that attains the ceiling recall (the ceiling point on x-axis).
ceiling_nd = int(nd[recalled == max_recall].min())

# ------------------------------------------------------------------- independent raw check
# Count strict local maxima of the raw sulfate profile inside the annotated set,
# from the TSV alone (no analysis code involved).  First/last row cannot be a
# strict local maximum, which is why the ceiling is 14 of 15.
so = obs["c04_sulfate_ug_kg"].to_numpy(float)
strict = (so > np.roll(so, 1)) & (so >= np.roll(so, -1))
strict[0] = strict[-1] = False
raw_ceiling = int(strict[ann.index.to_numpy()].sum())

print(f"annotated horizons        : {n_ann} "
      f"(events={n_events}, unidentified={n_unid}, reference={n_ref})")
print(f"detector variants         : {n_var}")
print(f"n_detected  min/median/max: {nd.min():.0f} / {median_nd:.0f} / {nd.max():.0f}")
print(f"max recall over grid      : {max_recall}/{n_ann} = {ceiling_frac:.4f}")
print(f"ceiling n_detected (x)    : {ceiling_nd}")
print(f"raw strict-local-max count inside annotated set: {raw_ceiling}/{n_ann}")

# -------------------------------------------------------------------- figure layout
fig = plt.figure(figsize=(12, 9), dpi=140)
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.32, wspace=0.26)
ax_a = fig.add_subplot(gs[0, :])
ax_b = fig.add_subplot(gs[1, 0])
ax_c = fig.add_subplot(gs[1, 1])

# ------------------------------------------------------- (a) annotated horizons
ax_a.plot(obs["c01_depth_top_m"], obs["c04_sulfate_ug_kg"],
          color=PROFILE_COLOR, lw=0.7, zorder=1, label="sulfate profile (all 1931 rows)")
style = {
    "event_01..05": (EVENT_COLOR, "o", 95, f"event_01..05 (n={n_events})"),
    "unidentified_event": (UNID_COLOR, "^", 80, f"unidentified_event (n={n_unid})"),
    "reference_surface": (REF_COLOR, "s", 95, f"reference_surface (n={n_ref})"),
}
for cls, (col, mk, sz, lab) in style.items():
    sub = ann[ann["class"] == cls]
    ax_a.scatter(sub["c01_depth_top_m"], sub["c04_sulfate_ug_kg"],
                 s=sz, c=col, marker=mk, edgecolors="black", linewidths=0.6,
                 zorder=3, label=lab)
# Hand-tuned label offsets (points, dx/dy) to avoid overlap in the dense 36-41 m
# cluster; keys are (label, rounded depth).  Deterministic, no random placement.
OFFSETS = {
    ("event_01", 2.00): (8, 6), ("reference_surface", 0.00): (8, 4),
    ("event_02", 12.96): (8, 6), ("unidentified_event", 13.27): (8, 8),
    ("unidentified_event", 18.90): (-8, 8), ("unidentified_event", 19.85): (-8, -12),
    ("unidentified_event", 21.24): (8, -12), ("event_03", 23.31): (8, 8),
    ("event_04", 29.20): (8, 6), ("unidentified_event", 36.22): (-24, 18),
    ("unidentified_event", 36.59): (-20, 2), ("unidentified_event", 36.85): (22, -6),
    ("event_05", 37.28): (-46, 16), ("unidentified_event", 38.40): (18, 14),
    ("unidentified_event", 40.59): (-8, -18),
}
for _, r in ann.iterrows():
    dx, dy = OFFSETS[(r["c07_signal"], round(float(r["c01_depth_top_m"]), 2))]
    ax_a.annotate(r["c07_signal"], (r["c01_depth_top_m"], r["c04_sulfate_ug_kg"]),
                  xytext=(dx, dy), textcoords="offset points", fontsize=6.2,
                  color="#333333", ha="left" if dx >= 0 else "right",
                  arrowprops=dict(arrowstyle="-", lw=0.4, color="#999999"))
ax_a.set_xlabel("depth (m, c01_depth_top_m)")
ax_a.set_ylabel("sulfate (ug/kg, c04)")
ax_a.set_title("(a) The 15 c07-annotated horizons, coloured by label class", loc="left", fontsize=11)
ax_a.grid(alpha=0.25, lw=0.5)
ax_a.legend(loc="upper left", fontsize=8, framealpha=0.9)
ax_a.set_xlim(-0.8, obs["c01_depth_top_m"].max() + 1.5)

# ------------------------------------------- (b) distribution of n_detected
edges = np.unique(np.round(np.geomspace(max(nd.min(), 3), nd.max(), 34)))
ax_b.hist(nd, bins=edges, color="#5d6d7e", edgecolor="white", lw=0.4)
ax_b.axvline(median_nd, color="#c0392b", lw=1.8, ls="--",
             label=f"median = {median_nd:.0f} peaks")
ax_b.set_xscale("log")
ax_b.set_xlabel("detected peaks per variant (n_detected, log scale)")
ax_b.set_ylabel("number of detector variants")
ax_b.set_title(f"(b) Detected-peak counts across the {n_var} detector variants",
               loc="left", fontsize=11)
ax_b.grid(alpha=0.25, lw=0.5, axis="y")
ax_b.legend(fontsize=8, loc="upper right")
ax_b.annotate(f"min = {nd.min():.0f}", xy=(nd.min(), 0), xytext=(nd.min() * 0.9, ax_b.get_ylim()[1] * 0.55),
              fontsize=7.5, color="#444444", ha="left")
ax_b.annotate(f"max = {nd.max():.0f}", xy=(nd.max(), 0), xytext=(nd.max() * 0.72, ax_b.get_ylim()[1] * 0.75),
              fontsize=7.5, color="#444444", ha="right")

# -------------------------- (c) detected peaks vs annotated recall, with ceiling
bg_colors = {"rolling_q25": "#2471a3", "rolling_median": "#c0392b",
             "rolling_mean": "#e67e22", "linear_detrend": "#1e8449"}
for bg, col in bg_colors.items():
    m = grid["background"] == bg
    ax_c.scatter(grid.loc[m, "n_detected"], grid.loc[m, "n_annotated_recalled"],
                 s=34, c=col, alpha=0.8, edgecolors="white", linewidths=0.5,
                 label=bg, zorder=3)
ax_c.axhline(max_recall, color="black", lw=1.6, ls="--", zorder=2,
             label=f"max achievable recall = {max_recall}/{n_ann} ({ceiling_frac:.3f})")
ax_c.axvline(ceiling_nd, color="black", lw=1.2, ls=":", zorder=2,
             label=f"local-maximum ceiling reached at n={ceiling_nd}")
ax_c.annotate(
    f"ceiling attained\n(n={ceiling_nd}, recall={max_recall}/{n_ann})",
    xy=(ceiling_nd, max_recall), xytext=(max(ceiling_nd * 1.7, 24), max_recall * 0.86),
    fontsize=7.5, color="black",
    arrowprops=dict(arrowstyle="->", lw=0.8, color="black"),
)
ax_c.set_xscale("log")
ax_c.set_xlabel("detected peaks per variant (n_detected, log scale)")
ax_c.set_ylabel("annotated horizons recalled (of 15)")
ax_c.set_title("(c) Recall vs detected-peak count; structural 14/15 ceiling",
               loc="left", fontsize=11)
ax_c.set_ylim(0, 15.6)
ax_c.set_yticks(range(0, 16, 3))
ax_c.grid(alpha=0.25, lw=0.5)
ax_c.legend(fontsize=7.5, loc="lower right", framealpha=0.9, ncol=1)

fig.suptitle("Fig. 02 - Event-definition sensitivity: annotations, detector spread and the local-maximum recall ceiling",
             fontsize=12.5, y=0.995)
fig.tight_layout()

os.makedirs(OUTDIR, exist_ok=True)
fig.savefig(OUTPNG, dpi=140, bbox_inches="tight")
plt.close(fig)

size = os.path.getsize(OUTPNG)
print(f"OUTPUT PATH: {OUTPNG}")
print(f"FILE SIZE  : {size} bytes")