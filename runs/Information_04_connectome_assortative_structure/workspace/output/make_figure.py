#!/usr/bin/env python3
"""Scatter: node_strength vs mean_connection_distance, 4 color-coded systems."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = "/data/user/conversations/conv-88510242-47d4-46dc-a7c4-b833ab46f409/project/data/network_node_summary.tsv"
OUT = "/data/user/conversations/conv-88510242-47d4-46dc-a7c4-b833ab46f409/project/output/figures.png"

df = pd.read_csv(DATA, sep="\t")
df["system"] = (df["region_index"] // 200).astype(int)

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
fig, ax = plt.subplots(figsize=(8, 6))

# Overall trend line (least squares)
slope, intercept = np.polyfit(df["mean_connection_distance"], df["node_strength"], 1)
x_min, x_max = df["mean_connection_distance"].min(), df["mean_connection_distance"].max()
ax.plot([x_min, x_max],
        [slope * x_min + intercept, slope * x_max + intercept],
        color="black", linestyle="--", linewidth=1.5, alpha=0.8,
        label="Overall trend")

# Per-system clouds + Pearson r
r_values = {}
for system in range(4):
    sub = df[df["system"] == system]
    x = sub["mean_connection_distance"].to_numpy()
    y = sub["node_strength"].to_numpy()
    r = float(np.corrcoef(x, y)[0, 1])
    r_values[system] = r
    ax.scatter(x, y, s=12, alpha=0.45, label=f"System {system} (r={r:.3f})",
               color=colors[system])

# Corner annotation box (overall r + per-system r)
overall_r = float(np.corrcoef(df["mean_connection_distance"], df["node_strength"])[0, 1])
text = f"Overall Pearson r = {overall_r:+.3f}\n\n" + "\n".join(
    f"System {s}: r = {r_values[s]:+.3f}" for s in range(4)
)
ax.text(0.01, 0.99, text, transform=ax.transAxes, verticalalignment="top",
        horizontalalignment="left", fontsize=9, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="gray"))

ax.set_xlabel("mean_connection_distance")
ax.set_ylabel("node_strength")
ax.set_title("Node strength vs. mean connection distance (by network-system sub-cloud)")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"Saved {OUT}. Overall r = {overall_r:+.4f}; per-system r = "
      + ", ".join(f"sys{s}={r_values[s]:+.4f}" for s in range(4)))
