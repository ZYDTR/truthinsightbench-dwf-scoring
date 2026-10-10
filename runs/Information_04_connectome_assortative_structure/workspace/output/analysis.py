#!/usr/bin/env python3
"""
Reproducible analysis for Task Information_04_connectome_assortative_structure.

Data (data/):
  network_node_summary.tsv : 800 nodes, region_index 0..799,
      mean_homophily, node_strength, mean_connection_distance
  network_profile_a.tsv    : 800 nodes, coords (x,y,z), annotation_1..5
  network_profile_b.tsv    : 213 nodes, coords, annotation_1
  network_profile_c.tsv    : 39  nodes, coords, annotation_1..3
  trimming_sensitivity.tsv : 4 systems x annotations x {assort_z, assort_p_fdr},
      19 trim levels (trim_percent_1..19)

Analysis unit: one network system (= one block of 200 contiguous region_index)
or a complete node profile; nodes within a system are dependent.

Produces:
  results.json   : machine-readable summary of every statistic
  figures.png    : supporting figure
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
OUT = Path(__file__).resolve().parent


def pearson(x, y):
    """Pearson r with a two-sided t-test p-value (normal approx is fine at n>=40)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    dx, dy = x - x.mean(), y - y.mean()
    r = float(dx @ dy / math.sqrt((dx @ dx) * (dy @ dy)))
    n, df = len(x), len(x) - 2
    t = r * math.sqrt(df) / math.sqrt(1.0 - r * r)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0))))
    return {"r": round(r, 6), "t": round(t, 6), "p": p, "n": int(n)}


def read_all():
    sm = pd.read_csv(DATA / "network_node_summary.tsv", sep="\t")
    pa = pd.read_csv(DATA / "network_profile_a.tsv", sep="\t")
    pb = pd.read_csv(DATA / "network_profile_b.tsv", sep="\t")
    pc = pd.read_csv(DATA / "network_profile_c.tsv", sep="\t")
    ts = pd.read_csv(DATA / "trimming_sensitivity.tsv", sep="\t")
    return sm, pa, pb, pc, ts


def main():
    sm, pa, pb, pc, ts = read_all()
    res = {"task_id": "Information_04_connectome_assortative_structure"}

    m = pd.merge(sm, pa, on="region_index")
    # systems = blocks of 200 contiguous region_index
    m["system"] = (m["region_index"] // 200).astype(int)
    sm["system"] = (sm["region_index"] // 200).astype(int)

    res["n_nodes"] = int(sm.shape[0])
    res["n_systems"] = int(sm["system"].nunique())
    res["nodes_per_system"] = int(sm.groupby("system").size().unique()[0])

    # ----------------------------------------------------------------------
    # F1: strength ~ distance (robust across all systems)
    # ----------------------------------------------------------------------
    f1 = {
        "overall": pearson(sm["node_strength"], sm["mean_connection_distance"]),
        "per_system_r": {
            str(s): round(pearson(sm[sm.system == s]["node_strength"],
                                  sm[sm.system == s]["mean_connection_distance"])["r"], 4)
            for s in sorted(sm["system"].unique())
        },
        "robust_same_direction": bool(
            all(
                pearson(sm[sm.system == s]["node_strength"],
                        sm[sm.system == s]["mean_connection_distance"])["r"] > 0
                for s in sorted(sm["system"].unique())
            )
        ),
    }
    res["finding_1_strength_distance"] = f1

    # ----------------------------------------------------------------------
    # F2: spatial y-gradient on strength is system-dependent
    # ----------------------------------------------------------------------
    f2 = {
        "overall_strength_coord_y": pearson(m["node_strength"], m["coord_y"]),
        "per_system_strength_coord_y": {
            str(s): pearson(m[m.system == s]["node_strength"],
                            m[m.system == s]["coord_y"])
            for s in sorted(m["system"].unique())
        },
        "overall_distance_coord_y": pearson(m["mean_connection_distance"], m["coord_y"]),
        "per_system_distance_coord_y": {
            str(s): pearson(m[m.system == s]["mean_connection_distance"],
                            m[m.system == s]["coord_y"])["r"]
            for s in sorted(m["system"].unique())
        },
        "quadrant_means": {
            "strength": m.groupby("qxy")["node_strength"].mean().round(4).to_dict()
            if "qxy" in m else {},
        },
    }
    m["qxy"] = (m["coord_x"] >= 0).astype(int) * 2 + (m["coord_y"] >= 0).astype(int)
    f2["quadrant_means"] = {
        "strength": m.groupby("qxy")["node_strength"].mean().round(4).to_dict(),
        "distance": m.groupby("qxy")["mean_connection_distance"].mean().round(4).to_dict(),
        "homophily": m.groupby("qxy")["mean_homophily"].mean().round(4).to_dict(),
    }
    res["finding_2_spatial_gradient"] = f2

    # ----------------------------------------------------------------------
    # F3: homophily is unassociated (null)
    # ----------------------------------------------------------------------
    f3_cols = ["node_strength", "mean_connection_distance", "coord_x",
               "coord_y", "coord_z", "annotation_1", "annotation_2",
               "annotation_3", "annotation_4", "annotation_5"]
    f3 = {
        col: pearson(m["mean_homophily"], m[col])
        for col in f3_cols
    }
    f3["max_abs_r"] = max(abs(f3[c]["r"]) for c in f3_cols)
    f3["all_null_lt_0.15"] = bool(all(abs(f3[c]["r"]) < 0.15 for c in f3_cols))
    res["finding_3_homophily_null"] = f3

    # ----------------------------------------------------------------------
    # F4: trimming reduces assort_z; FDR significance is patchy
    # ----------------------------------------------------------------------
    f4 = {"n_pairs": 0, "assort_z": {}, "assort_p_fdr": {}}
    zcols = [f"trim_percent_{i}" for i in range(1, 20)]
    z = ts[ts.result_kind == "assort_z"].copy()
    p = ts[ts.result_kind == "assort_p_fdr"].copy()
    pairs = sorted(set(zip(z.system_index, z.annotation_index)))
    f4["n_pairs"] = len(pairs)
    for sy, an in pairs:
        zzv = ts[(ts.system_index == sy) & (ts.annotation_index == an) &
                 (ts.result_kind == "assort_z")][zcols].values[0]
        ppv = ts[(ts.system_index == sy) & (ts.annotation_index == an) &
                 (ts.result_kind == "assort_p_fdr")][zcols].values[0]
        f4["assort_z"][f"{sy}/{an}"] = {
            "values": [round(v, 4) for v in zzv],
            "tp1": round(zzv[0], 4),
            "tp_peak": round(max(zzv), 4),
            "tp_last": round(zzv[-1], 4),
            "declines_with_trim": zzv[0] > zzv[-1],
        }
        f4["assort_p_fdr"][f"{sy}/{an}"] = {
            "values": [round(v, 4) for v in ppv],
            "min_p": round(min(ppv), 4),
            "n_sig": int(sum(v < 0.05 for v in ppv)),
            "ever_sig": bool(min(ppv) < 0.05),
        }
    f4["never_significant_pairs"] = [
        f"{sy}/{an}" for sy, an in pairs
        if f4["assort_p_fdr"][f"{sy}/{an}"]["n_sig"] == 0
    ]
    f4["all_assort_z_decline"] = all(
        f4["assort_z"][f"{sy}/{an}"]["declines_with_trim"] for sy, an in pairs)
    res["finding_4_trimming"] = f4

    # ----------------------------------------------------------------------
    # Profiles b & c (descriptive coverage)
    # ----------------------------------------------------------------------
    res["profile_b"] = {
        "n_nodes": int(pb.shape[0]),
        "coord_x": [round(pb.coord_x.min()), round(pb.coord_x.max())],
        "coord_y": [round(pb.coord_y.min()), round(pb.coord_y.max())],
        "coord_z": [round(pb.coord_z.min()), round(pb.coord_z.max())],
        "annotation_1": [round(pb.annotation_1.min(), 3), round(pb.annotation_1.max(), 3)],
    }
    res["profile_c"] = {
        "n_nodes": int(pc.shape[0]),
        "coord_x": [round(pc.coord_x.min(), 1), round(pc.coord_x.max(), 1)],
        "coord_y": [round(pc.coord_y.min(), 1), round(pc.coord_y.max(), 1)],
        "coord_z": [round(pc.coord_z.min(), 1), round(pc.coord_z.max(), 1)],
        "annotation_1": [round(pc.annotation_1.min(), 3), round(pc.annotation_1.max(), 3)],
        "annotation_2": [round(pc.annotation_2.min(), 3), round(pc.annotation_2.max(), 3)],
        "annotation_3": [round(pc.annotation_3.min(), 3), round(pc.annotation_3.max(), 3)],
    }
    # annotation covariance (profile a)
    anns = [f"annotation_{i}" for i in range(1, 6)]
    corr = pa[anns].corr()
    res["annotation_pairwise_correlation"] = {
        f"{i}/{j}": round(corr[f"annotation_{i}"][f"annotation_{j}"], 4)
        for i in range(1, 6) for j in range(i + 1, 6)
    }
    # summary correlations of the three node measures against each annotation
    res["summary_vs_annotations"] = {
        col: {c: round(pearson(m[col], m[c])["r"], 4) for c in anns}
        for col in ["node_strength", "mean_connection_distance", "mean_homophily"]
    }

    with open(OUT / "results.json", "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print("wrote", OUT / "results.json")

    # ----------------------------------------------------------------------
    # Figure
    # ----------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        for i, (xcol, ycol, lab) in enumerate(
            [("mean_connection_distance", "node_strength", "strength vs distance"),
             ("coord_y", "node_strength", "strength vs coord_y"),
             ("coord_y", "mean_connection_distance", "distance vs coord_y")]):
            for s, grp in m.groupby("system"):
                ax[i].scatter(grp[xcol], grp[ycol], alpha=0.35, s=8,
                              label=f"system {s}")
            ax[i].set_xlabel(xcol)
            ax[i].set_ylabel(ycol)
            ax[i].set_title(lab)
            ax[i].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / "figures.png", dpi=150)
        print("wrote figure")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
