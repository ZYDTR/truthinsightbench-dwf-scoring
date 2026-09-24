#!/usr/bin/env python3
"""Figure 3 (production): chronology-coordinate comparison for the 15 dated horizons.

Panels
------
(a) age (CE) vs c01_depth_top_m  (ice-equivalent / top depth) with OLS fit, slope, R2, LOO MAE.
(b) age (CE) vs c06_depth_w_e_m  (water-equivalent depth) with OLS fit, slope, R2, LOO MAE.
(c) On both panels: leave-one-out predictions drawn as small open markers.
(d) Bottom-centre axes (via gridspec): residual (observed - fitted) age vs c01 depth for
    both coordinates, with a residual = 0 line, making the non-linearity visible.

Output: <project>/output/figures/fig03_chronology_coordinates.png
Writes nothing else. Deterministic; no randomness.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = "/data/user/conversations/conv-f07721cb-d3dc-42e9-83c7-7e986203115b/project"
DATA = os.path.join(PROJECT, "data", "observations.tsv")
OUT = os.path.join(PROJECT, "output", "figures", "fig03_chronology_coordinates.png")

DEPTH_COLOURS = {"c01_depth_top_m": "#1f4e79", "c06_depth_w_e_m": "#a3320b"}
DEPTH_LABELS = {
    "c01_depth_top_m": "c01 depth (top, m)",
    "c06_depth_w_e_m": "c06 depth (water-equivalent, m)",
}


def loo_predictions(x, y):
    """Leave-one-out OLS predictions for every point (refit n-1 points each time)."""
    n = len(x)
    pred = np.empty(n, dtype=float)
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        coef = np.polyfit(x[keep], y[keep], 1)
        pred[i] = np.polyval(coef, x[i])
    return pred


def fit_stats(x, y):
    coef = np.polyfit(x, y, 1)
    fitted = np.polyval(coef, x)
    resid = y - fitted
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    loo = loo_predictions(x, y)
    loo_resid = y - loo
    return {
        "slope": float(coef[0]),
        "intercept": float(coef[1]),
        "r2": r2,
        "fitted": fitted,
        "resid": resid,
        "loo": loo,
        "loo_mae": float(np.mean(np.abs(loo_resid))),
        "loo_rmse": float(np.sqrt(np.mean(loo_resid ** 2))),
    }


def main():
    df = pd.read_csv(DATA, sep="\t")
    dated = df[df["c08_age_a_ad_ce"].notna()].copy()
    x1 = dated["c01_depth_top_m"].to_numpy(dtype=float)
    x6 = dated["c06_depth_w_e_m"].to_numpy(dtype=float)
    age = dated["c08_age_a_ad_ce"].to_numpy(dtype=float)
    n = len(age)
    if n != 15:
        print(f"WARNING: expected 15 dated horizons, found {n}", file=sys.stderr)

    s1 = fit_stats(x1, age)
    s6 = fit_stats(x6, age)
    stats = {"c01_depth_top_m": (x1, s1), "c06_depth_w_e_m": (x6, s6)}

    fig = plt.figure(figsize=(13, 5.5), dpi=140)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.35, 1.0])
    ax1 = fig.add_subplot(gs[0, 0])
    ax6 = fig.add_subplot(gs[0, 1])
    axr = fig.add_subplot(gs[1, :])
    axes = {("c01_depth_top_m"): ax1, "c06_depth_w_e_m": ax6}

    for col in ("c01_depth_top_m", "c06_depth_w_e_m"):
        ax = axes[col]
        x, s = stats[col]
        colour = DEPTH_COLOURS[col]

        # fit line across the observed depth range
        xs = np.linspace(x.min(), x.max(), 200)
        ax.plot(xs, s["slope"] * xs + s["intercept"], color=colour, lw=1.6, zorder=2,
                label="OLS fit")
        # dated horizons
        ax.scatter(x, age, s=52, facecolor=colour, edgecolor="white", linewidth=0.7,
                   zorder=4, label=f"dated horizons (n={n})")
        # leave-one-out predictions: small open markers
        ax.scatter(x, s["loo"], s=34, marker="o", facecolor="none",
                   edgecolor="#333333", linewidth=0.9, zorder=3,
                   label="leave-one-out predictions")
        # thin connector between observed and LOO prediction
        for xi, obs, lp in zip(x, age, s["loo"]):
            ax.plot([xi, xi], [obs, lp], color="#666666", lw=0.6, alpha=0.75, zorder=1)

        ax.set_xlabel(DEPTH_LABELS[col], fontsize=10)
        ax.set_ylabel("age (a AD CE / years CE)", fontsize=10)
        ax.set_title(
            f"({'a' if col == 'c01_depth_top_m' else 'b'}) Age vs {col.split('_')[0]} depth",
            fontsize=11, loc="left")
        ax.grid(alpha=0.25, lw=0.6)
        ax.tick_params(labelsize=9)

        box = (
            f"slope = {s['slope']:.3f} yr m$^{{-1}}$\n"
            f"intercept = {s['intercept']:.2f} CE\n"
            f"$R^2$ = {s['r2']:.5f}\n"
            f"LOO MAE = {s['loo_mae']:.2f} yr\n"
            f"LOO RMSE = {s['loo_rmse']:.2f} yr"
        )
        ax.text(0.965, 0.96, box, transform=ax.transAxes, ha="right", va="top",
                fontsize=8.6, family="monospace", zorder=6,
                bbox=dict(boxstyle="round,pad=0.42", facecolor="#f7f7f2",
                          edgecolor=colour, linewidth=1.0, alpha=0.95))
        ax.legend(loc="lower left", fontsize=7.6, framealpha=0.92, borderpad=0.4)

    # (d) residual panel: observed - fitted age vs c01 depth, both coordinates
    for col in ("c01_depth_top_m", "c06_depth_w_e_m"):
        x, s = stats[col]
        colour = DEPTH_COLOURS[col]
        axr.axhline(0.0, color="black", lw=1.0, ls="--", zorder=1)
        axr.scatter(x, s["resid"], s=42, facecolor=colour, edgecolor="white",
                    linewidth=0.6, zorder=3, label=DEPTH_LABELS[col])
        axr.plot(x, s["resid"], color=colour, lw=1.2, alpha=0.85, zorder=2)
        for xi, r in zip(x, s["resid"]):
            axr.annotate("", xy=(xi, 0.0), xytext=(xi, r),
                         arrowprops=dict(arrowstyle="-", color=colour, lw=0.6, alpha=0.6))

    axr.set_xlabel("c01 depth (top, m)", fontsize=10)
    axr.set_ylabel("residual age (yr)\nobs $-$ fit", fontsize=9)
    axr.set_title(
        "(d) Residuals (observed - fitted) versus c01 depth: curvature shows the "
        "c01 linear-age assumption is only approximate",
        fontsize=10, loc="left")
    axr.grid(alpha=0.25, lw=0.6)
    axr.tick_params(labelsize=9)
    axr.legend(loc="upper left", fontsize=8.0, framealpha=0.9, ncol=2)

    fig.suptitle(
        "Chronology coordinates of the 15 dated horizons: linear age-depth fits",
        fontsize=12, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.0, w_pad=1.6)
    fig.savefig(OUT, dpi=140)

    size = os.path.getsize(OUT)
    print(f"OUTPUT_PATH={OUT}")
    print(f"FILE_SIZE_BYTES={size}")
    print(f"n_dated={n}")
    for col in ("c01_depth_top_m", "c06_depth_w_e_m"):
        s = stats[col][1]
        print(f"{col}: slope={s['slope']:.6f} intercept={s['intercept']:.4f} "
              f"R2={s['r2']:.6f} LOO_MAE={s['loo_mae']:.4f} LOO_RMSE={s['loo_rmse']:.4f}")
    print("RESID_C01_MINMAX=%.3f %.3f" % (stats['c01_depth_top_m'][1]['resid'].min(),
                                          stats['c01_depth_top_m'][1]['resid'].max()))
    print("RESID_C06_MINMAX=%.3f %.3f" % (stats['c06_depth_w_e_m'][1]['resid'].min(),
                                          stats['c06_depth_w_e_m'][1]['resid'].max()))


if __name__ == "__main__":
    main()