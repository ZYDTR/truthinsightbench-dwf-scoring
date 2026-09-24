#!/usr/bin/env python3
"""
Correlation 2D critical-cell analysis + finite-size gap-extrapolation consistency check.

Completes two analysis branches that were not finished by the delegated sub-agents.
Data sources (workspace only):
  WS/data/normalized_views/control_vs_correlation.tsv
  WS/data/source_data/all_source_data_deidentified.zip -> table_012.dat (1/xi),
      table_010.dat (dSz=2 gap vs 1/L), table_011.dat (dSz=1 gap vs 1/L),
      table_008.dat (dSz=2 extrapolated gap), table_009.dat (dSz=1 extrapolated gap).
Self-contained and re-runnable: python3 WS/output/analysis/correlation_and_extrap.py
"""
import json
import os
import re
import zipfile
import tempfile

import numpy as np
import pandas as pd

WS = "/data/user/conversations/conv-af56a8cc-3c22-4b90-8a53-8142e2d7b40e/project"
DATA = os.path.join(WS, "data")
RES_DIR = os.path.join(WS, "output", "results")
os.makedirs(RES_DIR, exist_ok=True)

CRIT = 1e-6


def load_dat_blocks(path):
    """Parse a block-structured .dat: returns list of (header, list of row-tuples).

    Row-tuples have as many floats as numeric columns are present in the file
    (2-column gap files yield 2-tuples; table_012.dat yields 3-tuples).
    """
    blocks = {}
    cur = None
    ncol = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                cur = line
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            if ncol is None:
                ncol = len(parts)
            try:
                vals = tuple(float(p) for p in parts[:ncol])
            except ValueError:
                continue
            blocks.setdefault(cur, []).append(vals)
    return [(h, v) for h, v in blocks.items() if v]


def correlation_analysis():
    df = pd.read_csv(os.path.join(DATA, "normalized_views", "control_vs_correlation.tsv"), sep="\t")
    rows = []
    crit_cells = []
    for u, g in df.groupby("control_ratio"):
        inv = g["inverse_correlation_length"]
        sec = g["secondary_interaction_ratio"]
        min_val = float(inv.min())
        argmin = sec[inv == min_val].tolist()
        rows.append({"control_ratio": float(u),
                      "inv_min": min_val, "inv_max": float(inv.max()),
                      "inv_mean": float(inv.mean()),
                      "argmin_secondary_ratio": ",".join(f"{v:g}" for v in argmin)})
        for r in g.itertuples():
            v = float(r.inverse_correlation_length)
            if abs(v) < CRIT:
                crit_cells.append({"control_ratio": float(r.control_ratio),
                                   "secondary_interaction_ratio": float(r.secondary_interaction_ratio),
                                   "inv_corr_len": v})
    per_control = pd.DataFrame(rows)
    crit = pd.DataFrame(crit_cells).sort_values(["control_ratio", "secondary_interaction_ratio"]).reset_index(drop=True)

    with tempfile.TemporaryDirectory() as tmp:
        z = zipfile.ZipFile(os.path.join(DATA, "source_data", "all_source_data_deidentified.zip"))
        z.extract("table_012.dat", tmp)
        raw_blocks = load_dat_blocks(os.path.join(tmp, "table_012.dat"))
    raw = {}
    for h, vals in raw_blocks:
        for row in vals:
            u, jh, inv = (float(row[0]), float(row[1]), float(row[2]))
            raw[(round(u, 5), round(jh, 5))] = inv
    overlap = 0
    diffs = []
    for r in df.itertuples():
        key = (round(float(r.control_ratio), 5), round(float(r.secondary_interaction_ratio), 5))
        if key in raw:
            overlap += 1
            diffs.append(abs(raw[key] - float(r.inverse_correlation_length)) / (abs(raw[key]) or 1.0))
    max_rel_diff = max(diffs) if diffs else float("nan")
    crit_controls = sorted(crit["control_ratio"].unique().tolist())
    summary = {
        "n_cells_total": int(len(df)),
        "n_cells_per_grid": "40 x 17",
        "n_critical_cells": int(len(crit)),
        "critical_controls": [round(x, 2) for x in crit_controls],
        "critical_cells": crit.to_dict(orient="records"),
        "raw_overlap_pairs": overlap,
        "raw_max_rel_diff": float(max_rel_diff),
        "raw_table": "table_012.dat (L=60)",
    }
    per_control.to_csv(os.path.join(RES_DIR, "correlation_per_control.csv"), index=False)
    crit.to_csv(os.path.join(RES_DIR, "correlation_critical_cells.csv"), index=False)
    with open(os.path.join(RES_DIR, "correlation_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def gap_extrap_analysis():
    """
    Finite-size gap extrapolation consistency check.

    table_010.dat / table_011.dat : gap vs inverse system size 1/L, one block per
    U/W (fixed Jh/U=0.25).  A linear fit gap = a + b*(1/L) gives the L->Infinity
    extrapolated gap = a.
    table_008.dat / table_009.dat : gap extrapolated to L->Infinity, one block per
    Jh/U (columns U/W, gap).  We compare a(U/W) against the raw extrapolated value
    on the Jh/U=0.25 branch at the same U/W.
    """
    with tempfile.TemporaryDirectory() as tmp:
        z = zipfile.ZipFile(os.path.join(DATA, "source_data", "all_source_data_deidentified.zip"))
        for n in ["table_010.dat", "table_011.dat", "table_008.dat", "table_009.dat"]:
            z.extract(n, tmp)
        def blocks(path):
            out = {}
            cur = None
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("#"):
                        cur = line
                        continue
                    p = line.split()
                    if len(p) >= 2:
                        try:
                            out.setdefault(cur, []).append((float(p[0]), float(p[1])))
                        except ValueError:
                            pass
            return out
        t10 = blocks(os.path.join(tmp, "table_010.dat"))
        t11 = blocks(os.path.join(tmp, "table_011.dat"))
        t08 = blocks(os.path.join(tmp, "table_008.dat"))
        t09 = blocks(os.path.join(tmp, "table_009.dat"))

    def u_from_header(h):
        m = re.search(r"U/W=([0-9.]+)", h)
        return float(m.group(1)) if m else None

    rows = []
    checks = []
    for channel, (fit_blocks, extrap_blocks) in [
        ("dSz=2", (t10, t08)), ("dSz=1", (t11, t09))]:
        raw_025 = {}
        for h, vals in extrap_blocks.items():
            if re.search(r"Jh/U=0.25", h):
                for x, y in vals:
                    raw_025[round(float(x), 5)] = float(y)
        for h, vals in fit_blocks.items():
            u = u_from_header(h)
            invL = np.array([x for x, _ in vals])
            gap = np.array([y for x, y in vals])
            a, b = np.polyfit(invL, gap, 1)
            resid = gap - (a + b * invL)
            r2 = 1 - (resid ** 2).sum() / ((gap - gap.mean()) ** 2).sum()
            rows.append({"U_over_W": float(u), "channel": channel,
                         "n_1L_points": int(len(vals)),
                         "intercept_a": float(a), "slope_b": float(b), "r2": float(r2)})
            raw_val = raw_025.get(round(u, 5), None)
            if raw_val is not None:
                checks.append({"U_over_W": float(u), "channel": channel,
                               "fitted_intercept": float(a),
                               "raw_extrapolated": float(raw_val),
                               "rel_diff": float(abs(a - raw_val) / abs(raw_val))})
    fit_df = pd.DataFrame(rows).sort_values(["channel", "U_over_W"]).reset_index(drop=True)
    check_df = pd.DataFrame(checks).sort_values(["channel", "U_over_W"]).reset_index(drop=True)
    fit_df.to_csv(os.path.join(RES_DIR, "gap_extrapolation_fit.csv"), index=False)
    check_df.to_csv(os.path.join(RES_DIR, "gap_extrapolation_check.csv"), index=False)

    summary = {
        "n_blocks_dSz2": int(len(t10)),
        "n_blocks_dSz1": int(len(t11)),
        "n_overlap_pairs": int(len(check_df)),
        "max_rel_diff_dSz2": float(check_df[check_df.channel == "dSz=2"]["rel_diff"].max())
            if len(check_df[check_df.channel == "dSz=2"]) else None,
        "max_rel_diff_dSz1": float(check_df[check_df.channel == "dSz=1"]["rel_diff"].max())
            if len(check_df[check_df.channel == "dSz=1"]) else None,
        "fit_method": "linear fit gap = a + b*(1/L); a is L->Infinity intercept",
        "raw_reference_branch": "Jh/U=0.25 in table_008.dat (dSz=2) and table_009.dat (dSz=1)",
    }
    with open(os.path.join(RES_DIR, "gap_extrapolation_check.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


if __name__ == "__main__":
    corr = correlation_analysis()
    print("CORRELATION:", json.dumps(corr, indent=2))
    ext = gap_extrap_analysis()
    print("EXTRAP:", json.dumps(ext, indent=2))
    print("DONE")
