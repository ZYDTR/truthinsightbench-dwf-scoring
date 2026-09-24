#!/usr/bin/env python3
"""Reproducible figures for the five observable groups (TASK ID: figures).

Figures (matplotlib, Agg backend, PNG at dpi=150):
  (a) output/figures/gap_vs_u.png          - two stacked panels: dSz=1 and dSz=2
      magnon gap vs U/W, one line per Jh/U branch (8 lines), black x markers
      at the linearly-interpolated sign-change crossing where it exists.
  (b) output/figures/topological_proxy.png - two stacked panels:
      ground-state degeneracy vs U/W (step plot) and gap to the second
      distinct level vs U/W (log scale).
  (c) output/figures/local_observables.png - total magnetic moment and charge
      fluctuations vs U/W (two stacked panels).
  (d) output/figures/correlation_heatmap.png - 2D heatmap of
      inverse_correlation_length over U/W (x) x Jh/U (y), with colorbar and
      black dots marking critical cells (value < 1e-6).

Data provenance (WS/data/normalized_views/):
  control_vs_gap_case_A.tsv      (dSz=1: control_ratio, gap_like_observable)
  control_vs_gap_case_B.tsv      (dSz=2: control_ratio, gap_like_observable)
  control_vs_topological_proxy.tsv (control_ratio, spectrum_value, degeneracy)
  control_vs_local_observables.tsv (control_ratio, total_magnetic_moment,
                                     charge_fluctuation)
  control_vs_correlation.tsv     (control_ratio, secondary_interaction_ratio,
                                  inverse_correlation_length)

Branch structure: a new branch starts where control_ratio (U/W) decreases;
file order = Jh/U = 0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40.
Self-contained and re-runnable: python3 WS/output/analysis/plot_figures.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WS = "/data/user/conversations/conv-af56a8cc-3c22-4b90-8a53-8142e2d7b40e/project"
DATA = os.path.join(WS, "data", "normalized_views")
FIG_DIR = os.path.join(WS, "output", "figures")
RES_DIR = os.path.join(WS, "output", "results")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

JH_LIST = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
CRITICAL = 1e-6
DEG_EPS = 1e-9


def load(path):
    """Load a TSV and return a DataFrame with a float 'u' column."""
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    df["u"] = df["control_ratio"].astype(float)
    return df


def split_branches(u, g):
    """Split flat (u, g) arrays into contiguous branches at U/W drops."""
    u = np.asarray(u, dtype=float)
    g = np.asarray(g, dtype=float)
    starts = [0]
    for i in range(1, len(u)):
        if u[i] < u[i - 1]:
            starts.append(i)
    ends = starts[1:] + [len(u)]
    return [(u[s:e], g[s:e]) for s, e in zip(starts, ends)]


def crossing_u(u, g):
    """Linear-interpolated sign-change crossing (gap through 0).
    Returns None if the branch never changes sign."""
    for i in range(len(g) - 1):
        if g[i] == 0.0:
            return float(u[i])
        if g[i] * g[i + 1] < 0.0:
            return float(u[i] - g[i] * (u[i + 1] - u[i]) / (g[i + 1] - g[i]))
    return None


def crossing_y(u, g, c):
    """Gap value at the crossing (linear interpolation, ~0)."""
    return float(np.interp(c, u, g))


def plot_gap_figures():
    """Figure (a): gap vs U/W for dSz=1 (panel 1) and dSz=2 (panel 2)."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1]})
    panels = [
        (os.path.join(DATA, "control_vs_gap_case_A.tsv"), "dSz=1"),
        (os.path.join(DATA, "control_vs_gap_case_B.tsv"), "dSz=2"),
    ]
    crossing_report = []
    for ax, (path, dsz) in zip(axes, panels):
        df = load(path)
        branches = split_branches(df["u"], df["gap_like_observable"])
        if len(branches) != len(JH_LIST):
            raise ValueError(
                f"expected {len(JH_LIST)} branches in {path}, got {len(branches)}"
            )
        crossings = []
        for jh, (u, g) in zip(JH_LIST, branches):
            ax.plot(u, g, "-o", markersize=3, linewidth=1.2,
                    label=f"Jh/U={jh:.2f}")
            c = crossing_u(u, g)
            if c is not None:
                yv = crossing_y(u, g, c)
                ax.plot([c], [yv], "x", color="k", markersize=9,
                        markeredgewidth=2, zorder=5)
                crossings.append((jh, c, yv))
        title = f"gap (dSz={dsz}) vs U/W"
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("gap_like_observable", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Jh/U", fontsize=8, loc="best", ncol=2)
        if dsz == "2":
            ax.set_xlabel("U/W", fontsize=11)
        crossing_report.append({"dSz": dsz, "n_branches": len(branches),
                                "n_crossings": len(crossings),
                                "crossings": [
                                    {"Jh_over_U": float(jh),
                                     "u_interp": float(c),
                                     "gap_at_crossing": float(yv)}
                                    for jh, c, yv in crossings]})
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "gap_vs_u.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, {"per_dSz": crossing_report}


def topological_per_control(df):
    """Per-control ground-state degeneracy and gap to 2nd distinct level."""
    rows = []
    for u, g in df.groupby("control_ratio", sort=True):
        v = g["spectrum_value"].astype(float)
        low = float(v.min())
        low_deg = int(g.loc[(v - low).abs() <= DEG_EPS,
                            "degeneracy"].sum())
        distinct = sorted(v.unique().tolist())
        gap2 = (distinct[1] - low) if len(distinct) >= 2 else np.nan
        rows.append({"u": float(u), "low_deg": low_deg, "gap_to_2nd": gap2})
    return pd.DataFrame(rows)


def plot_topological_figure():
    """Figure (b): degeneracy step plot + gap-to-2nd-level (log)."""
    df = load(os.path.join(DATA, "control_vs_topological_proxy.tsv"))
    out = topological_per_control(df)
    u = out["u"].values
    deg = out["low_deg"].values
    gap2 = out["gap_to_2nd"].values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax1.step(u, deg, where="post", color="tab:blue", linewidth=2.5)
    for uu, dd in zip(u, deg):
        ax1.plot([uu], [dd], "o", color="tab:blue", markersize=4, zorder=5)
    ax1.set_title("Ground-state degeneracy vs U/W", fontsize=12)
    ax1.set_ylabel("degeneracy (of lowest level)", fontsize=11)
    ax1.set_yticks([1, 2])
    ax1.grid(True, alpha=0.3, which="both")

    ax2.semilogy(u, gap2, "-o", color="tab:red", markersize=3, linewidth=1.2)
    ax2.set_title("Gap to second distinct level vs U/W (log scale)", fontsize=12)
    ax2.set_ylabel("gap_to_2nd", fontsize=11)
    ax2.grid(True, alpha=0.3, which="major")
    ax2.set_xlabel("U/W", fontsize=11)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "topological_proxy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    report = {
        "n_controls": int(len(out)),
        "deg1_up_to": float(out.loc[out["low_deg"] == 1, "u"].max()),
        "deg2_from": float(out.loc[out["low_deg"] == 2, "u"].min()),
        "gap_to_2nd_min": float(gap2.min()),
        "gap_to_2nd_max": float(gap2.max()),
    }
    return path, report


def plot_local_figure():
    """Figure (c): magnetic moment and charge fluctuations vs U/W."""
    df = load(os.path.join(DATA, "control_vs_local_observables.tsv"))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(df["u"], df["total_magnetic_moment"], "-o", color="tab:green",
             markersize=3, linewidth=1.2)
    ax1.set_title("Total magnetic moment vs U/W", fontsize=12)
    ax1.set_ylabel("total_magnetic_moment", fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(df["u"], df["charge_fluctuation"], "-o", color="tab:orange",
             markersize=3, linewidth=1.2)
    ax2.set_title("Charge fluctuations vs U/W", fontsize=12)
    ax2.set_ylabel("charge_fluctuation", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("U/W", fontsize=11)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "local_observables.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    report = {
        "n_points": int(len(df)),
        "moment_min": float(df["total_magnetic_moment"].min()),
        "moment_max": float(df["total_magnetic_moment"].max()),
        "charge_min": float(df["charge_fluctuation"].min()),
        "charge_max": float(df["charge_fluctuation"].max()),
    }
    return path, report


def plot_correlation_figure():
    """Figure (d): heatmap of inverse_correlation_length (U/W x Jh/U)."""
    df = load(os.path.join(DATA, "control_vs_correlation.tsv"))
    us = np.asarray(sorted(df["control_ratio"].unique().tolist()), dtype=float)
    jhs = np.asarray(sorted(df["secondary_interaction_ratio"].unique().tolist()), dtype=float)
    n_u, n_j = len(us), len(jhs)
    M = np.zeros((n_j, n_u), dtype=float)
    for i, jh in enumerate(jhs):
        for j, u in enumerate(us):
            val = df.loc[(df["control_ratio"] == u)
                         & (df["secondary_interaction_ratio"] == jh),
                         "inverse_correlation_length"]
            if len(val) != 1:
                raise ValueError(f"expected 1 value at U/W={u}, Jh/U={jh}")
            M[i, j] = float(val.iloc[0])

    crit_mask = M < CRITICAL
    i_idx, j_idx = np.where(crit_mask)

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(M, origin="lower", cmap="viridis",
                   extent=[us[0], us[-1], jhs[0], jhs[-1]], aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("inverse_correlation_length", fontsize=10)

    if len(i_idx):
        ax.scatter(us[j_idx], jhs[i_idx], c="k", marker="x", s=70,
                   zorder=5,
                   label=f"critical (value < {CRITICAL:g}) [{len(i_idx)} cells]")
        ax.legend(loc="upper left")

    ax.set_title("Inverse correlation length vs U/W and Jh/U", fontsize=12)
    ax.set_xlabel("U/W (control_ratio)", fontsize=11)
    ax.set_ylabel("Jh/U (secondary_interaction_ratio)", fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "correlation_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    report = {
        "grid_u": [float(x) for x in us],
        "grid_jh": [float(x) for x in jhs],
        "n_cells": n_u * n_j,
        "n_critical": int(len(i_idx)),
        "value_min": float(M.min()),
        "value_max": float(M.max()),
    }
    return path, report


def main():
    results = {}
    tasks = [
        ("gap_vs_u", plot_gap_figures),
        ("topological_proxy", plot_topological_figure),
        ("local_observables", plot_local_figure),
        ("correlation_heatmap", plot_correlation_figure),
    ]
    for name, fn in tasks:
        path, report = fn()
        size = os.path.getsize(path)
        if size <= 0:
            raise RuntimeError(f"figure is empty: {path}")
        results[name] = {"path": path, "bytes": size, **report}
        print(f"Wrote {path} ({size} bytes)")

    summary = {"figures": results, "n_figures": len(results)}
    out_json = os.path.join(RES_DIR, "figures_summary.json")
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote {out_json}")

    # console report of key numbers
    print("\n--- key numbers ---")
    for name, r in results.items():
        print(f"{name}: {r['path']} = {r['bytes']} bytes")
        for k, v in r.items():
            if k in ("path", "bytes"):
                continue
            print(f"  {name}.{k} = {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
