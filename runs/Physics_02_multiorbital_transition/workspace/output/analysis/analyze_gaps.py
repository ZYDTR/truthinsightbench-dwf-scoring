#!/usr/bin/env python3
"""
Analysis of change boundaries of multiple observables under a control parameter
U/W for the anonymous two-component interacting model.

Data sources (workspace only):
  data/normalized_views/control_vs_gap_case_A.tsv   -> magnon gap Delta Sz=1, Jh/U branches
  data/normalized_views/control_vs_gap_case_B.tsv   -> magnon gap Delta Sz=2, Jh/U branches
  data/source_data/all_source_data_deidentified.zip -> table_008.dat (dSz=2), table_009.dat (dSz=1)

Task ID: Physics_02_multiorbital_transition, T0 = 2023-04-21
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "data"
OUT = BASE / "output" / "results"
FIGS = BASE / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)


def load_branches(tsv_path, gap_col="gap_like_observable"):
    """Parse a TSV long view into a list of (branch_id, array(u), array(gap))."""
    df = pd.read_csv(tsv_path, sep="\t")
    branches = []
    start = 0
    for i in range(1, len(df)):
        if df.iloc[i][df.columns[0]] < df.iloc[i - 1][df.columns[0]]:
            branches.append((start, i))
            start = i
    branches.append((start, len(df)))
    out = []
    for (s, e) in branches:
        sub = df.iloc[s:e]
        u = sub.iloc[:, 0].to_numpy()
        g = sub.iloc[:, 1].to_numpy()
        out.append((int(s), u, g))
    return out


def find_crossings(u, g):
    """Find sign-change intervals and linear-interpolated crossing points."""
    crosses = []
    for i in range(len(g) - 1):
        if g[i] * g[i + 1] < 0:
            uc = u[i] + (0.0 - g[i]) * (u[i + 1] - u[i]) / (g[i + 1] - g[i])
            crosses.append({"u_lo": u[i], "u_hi": u[i + 1], "u_cross_interp": uc,
                            "g_lo": g[i], "g_hi": g[i + 1]})
    return crosses


def main():
    # ---------- Load gap case A (dSz=1) and B (dSz=2) from normalized views ----------
    a_branches = load_branches(DATA / "normalized_views/control_vs_gap_case_A.tsv")
    b_branches = load_branches(DATA / "normalized_views/control_vs_gap_case_B.tsv")

    # ---------- Also parse raw table_009 (dSz=1) and table_008 (dSz=2) ----------
    def parse_dat(path):
        blocks, order = {}, []
        cur = None
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    cur = line
                    if cur.startswith("# Hund"):
                        order.append(cur)
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        v1, v2 = float(parts[0]), float(parts[1])
                    except ValueError:
                        continue
                    blocks.setdefault(cur, []).append((v1, v2))
        return blocks, order

    t009_blocks, t009_order = parse_dat(DATA / "source_data/all_source_data_deidentified.zip" if False else
                                        (DATA / "source_data").parent.parent / "data" / "source_data" /
                                        "all_source_data_deidentified.zip")
    # Extract into tmp for parsing
    import zipfile
    zip_path = DATA / "source_data" / "all_source_data_deidentified.zip"
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        z = zipfile.ZipFile(zip_path)
        z.extract("table_008.dat", tmp)
        z.extract("table_009.dat", tmp)
        t008_blocks, t008_order = parse_dat(Path(tmp) / "table_008.dat")
        t009_blocks, t009_order = parse_dat(Path(tmp) / "table_009.dat")

    # Build a unified branch table for both views
    def build_branch_records(br, label):
        records = []
        for bid, u, g in br:
            crosses = find_crossings(u, g)
            neg_count = int(np.sum(g < 0))
            pos_count = int(np.sum(g > 0))
            max_val = float(np.max(g))
            max_idx = int(np.argmax(g))
            min_val = float(np.min(g))
            min_idx = int(np.argmin(g))
            records.append({
                "label": label,
                "branch_index": bid,
                "n_points": len(u),
                "u_start": float(u[0]),
                "u_end": float(u[-1]),
                "u_min": float(u.min()),
                "u_max": float(u.max()),
                "gap_min": min_val,
                "gap_min_u": float(u[min_idx]),
                "gap_max": max_val,
                "gap_max_u": float(u[max_idx]),
                "n_negative": neg_count,
                "n_positive": pos_count,
                "n_crossings": len(crosses),
                "crossing_u_lo": crosses[0]["u_lo"] if crosses else None,
                "crossing_u_hi": crosses[0]["u_hi"] if crosses else None,
                "crossing_u_interp": crosses[0]["u_cross_interp"] if crosses else None,
            })
        return records

    a_records = build_branch_records(a_branches, "A_dSz1")
    b_records = build_branch_records(b_branches, "B_dSz2")

    # ---------- Find the Jh/U value for each branch ----------
    # The branch ordering in the TSVs corresponds to the order in table_009 (Jh/U = 0.05..0.40)
    # and table_008 (same).
    # In the TSV, branch 1 is the FIRST block (Jh/U=0.05), branch 2 is the SECOND (Jh/U=0.10), etc.
    jh_labels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    for i, rec in enumerate(a_records):
        rec["Jh_over_U"] = jh_labels[i]
    for i, rec in enumerate(b_records):
        rec["Jh_over_U"] = jh_labels[i]

    # ---------- Compute gap at the crossing for each branch ----------
    # For each branch, the crossing value of the gap is approximately 0.
    # The "gap magnitude at crossing" is not very meaningful; instead we look at
    # the gap value at the point where the branch is at its minimum |gap|.

    # ---------- Build a summary table ----------
    all_records = a_records + b_records
    df_all = pd.DataFrame(all_records)

    # Sort for output
    df_all = df_all.sort_values(["Jh_over_U", "label"])

    # Save results
    df_all.to_csv(OUT / "gap_branch_summary.csv", index=False)

    # ---------- Identify the "crossover" region in Jh/U ----------
    # For each Jh/U, the crossing U is approximately the same range (~0.8-2.6).
    # Let's see how the crossing U depends on Jh/U.
    for label in ["A_dSz1", "B_dSz2"]:
        sub = df_all[df_all["label"] == label]
        print(f"\n=== {label} (Jh/U -> crossing U) ===")
        print(sub[["Jh_over_U", "branch_index", "u_start", "u_end", "n_points",
                  "gap_min", "gap_min_u", "gap_max", "gap_max_u",
                  "n_negative", "n_positive", "n_crossings",
                  "crossing_u_lo", "crossing_u_hi", "crossing_u_interp"]].to_string(index=False))

    # ---------- Save per-branch detail for A and B ----------
    for label, recs in [("A_dSz1", a_records), ("B_dSz2", b_records)]:
        df_sub = pd.DataFrame(recs)
        df_sub.to_csv(OUT / f"gap_branch_detail_{label}.csv", index=False)

    # ---------- Compute the gap at each control point for each Jh/U (raw table data) ----------
    # From table_009 (dSz=1):
    t009_df = []
    for name, rows in t009_blocks.items():
        jh = float(re.search(r"Jh/U=([0-9.]+)", name).group(1)) if re.search(r"Jh/U=([0-9.]+)", name) else None
        for u, g in rows:
            t009_df.append({"Jh_over_U": jh, "U_over_W": u, "gap_dSz1": g})
    t009_df = pd.DataFrame(t009_df)
    t009_df.to_csv(OUT / "table_009_dSz1_full.csv", index=False)

    # From table_008 (dSz=2):
    t008_df = []
    for name, rows in t008_blocks.items():
        jh = float(re.search(r"Jh/U=([0-9.]+)", name).group(1)) if re.search(r"Jh/U=([0-9.]+)", name) else None
        for u, g in rows:
            t008_df.append({"Jh_over_U": jh, "U_over_W": u, "gap_dSz2": g})
    t008_df = pd.DataFrame(t008_df)
    t008_df.to_csv(OUT / "table_008_dSz2_full.csv", index=False)

    # ---------- Save a JSON summary of key findings ----------
    summary = {
        "task_id": "Physics_02_multiorbital_transition",
        "t0": "2023-04-21",
        "n_gap_branches_A": len(a_records),
        "n_gap_branches_B": len(b_records),
        "gap_case_A": {
            "Jh_over_U_values": jh_labels,
            "crossing_u_range": [
                round(float(r["crossing_u_interp"]), 5) if r["crossing_u_interp"] is not None else None
                for r in a_records
            ],
            "gap_max_values": [round(r["gap_max"], 6) for r in a_records],
            "gap_max_u_values": [round(r["gap_max_u"], 4) for r in a_records],
        },
        "gap_case_B": {
            "Jh_over_U_values": jh_labels,
            "crossing_u_range": [
                round(float(r["crossing_u_interp"]), 5) if r["crossing_u_interp"] is not None else None
                for r in b_records
            ],
            "gap_max_values": [round(r["gap_max"], 6) for r in b_records],
            "gap_max_u_values": [round(r["gap_max_u"], 4) for r in b_records],
        },
    }
    (OUT / "gap_analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSaved summary JSON")


if __name__ == "__main__":
    main()
