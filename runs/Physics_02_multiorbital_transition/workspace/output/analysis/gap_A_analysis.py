#!/usr/bin/env python3
"""Gap analysis: magnon gap Delta Sz=1, Jh/U branches (case A).

Data provenance:
  - TSV: WS/data/normalized_views/control_vs_gap_case_A.tsv
        columns: control_ratio (U/W), gap_like_observable
        8 contiguous branches; a new branch starts where control_ratio decreases.
        Branch order = Jh/U = 0.05 .. 0.40 (first block = 0.05).
  - Raw: WS/data/source_data/all_source_data_deidentified.zip -> table_009.dat
        8 blocks with headers '# Hund exchenge Jh/U=<v>', columns U/W, gap.

For each branch we report n_points, u_start, u_end, n_negative, n_positive,
crossing_u_interp (linear interpolation of gap through 0 across the sign-change
interval, null if no sign change), gap_max, gap_max_u, gap_min, gap_min_u.
Cross-check: same crossing_u_interp from raw table_009.dat;
rel_diff = |raw - tsv| / |tsv| (tsv value is the reference 'b').
"""
import json
import os
import re
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # WS
TSV_PATH = os.path.join(BASE, "data", "normalized_views", "control_vs_gap_case_A.tsv")
ZIP_PATH = os.path.join(BASE, "data", "source_data", "all_source_data_deidentified.zip")
OUT_DIR = os.path.join(BASE, "output", "results")
JH_LIST = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def find_branch_starts(df):
    """Branch starts: index 0 and every row where control_ratio decreases."""
    starts = [0]
    for i in range(1, len(df)):
        if df.iloc[i, 0] < df.iloc[i - 1, 0]:
            starts.append(i)
    return starts


def branch_stats(u, g):
    """Compute per-branch statistics from U/W array u and gap array g."""
    n = len(u)
    neg = int((g < 0).sum())
    pos = int((g > 0).sum())
    crossing = None
    for i in range(n - 1):
        if g[i] == 0:
            crossing = float(u[i])
            break
        if g[i] * g[i + 1] < 0:
            crossing = float(
                u[i] - g[i] * (u[i + 1] - u[i]) / (g[i + 1] - g[i])
            )
            break
    imax = int(g.argmax())
    imin = int(g.argmin())
    return {
        "n_points": n,
        "u_start": float(u[0]),
        "u_end": float(u[-1]),
        "n_negative": neg,
        "n_positive": pos,
        "crossing_u_interp": crossing,
        "gap_max": float(g.max()),
        "gap_max_u": float(u[imax]),
        "gap_min": float(g.min()),
        "gap_min_u": float(u[imin]),
    }


def parse_raw_table(zip_path):
    """Extract table_009.dat from the zip into a temp dir and parse its 8 blocks."""
    with zipfile.ZipFile(zip_path) as z:
        with tempfile.TemporaryDirectory() as td:
            z.extractall(td)
            p = os.path.join(td, "table_009.dat")
            text = open(p, "r").read()

    blocks = {}
    cur = None
    for line in text.splitlines():
        m = re.match(r"\s*#\s*Hund\s+exchenge\s+Jh/U=([\d.]+)", line)
        if m:
            cur = float(m.group(1))
            blocks[cur] = []
        elif cur is not None and line.strip():
            parts = line.split()
            if len(parts) == 2:
                blocks[cur].append((float(parts[0]), float(parts[1])))
    return blocks


def main():
    df = pd.read_csv(TSV_PATH, sep="\t")
    starts = find_branch_starts(df)
    if len(starts) != 8:
        raise SystemExit(
            f"Expected 8 branches in TSV, found {len(starts)} (starts={starts})"
        )

    branches = []
    for i, (jh, s) in enumerate(zip(JH_LIST, starts)):
        e = starts[i + 1] if i + 1 < len(starts) else len(df)
        u = df.iloc[s:e, 0].to_numpy(dtype=float)
        g = df.iloc[s:e, 1].to_numpy(dtype=float)
        st = branch_stats(u, g)
        st["Jh_over_U"] = jh
        branches.append(st)

    # Cross-check against raw table_009.dat
    raw = parse_raw_table(ZIP_PATH)
    for st in branches:
        jh = st["Jh_over_U"]
        raw_crossing = None
        if jh in raw:
            raw_u = [p[0] for p in raw[jh]]
            raw_g = [p[1] for p in raw[jh]]
            raw_crossing = branch_stats(np.asarray(raw_u, dtype=float),
                                        np.asarray(raw_g, dtype=float))["crossing_u_interp"]
        st["raw_crossing_u_interp"] = raw_crossing
        a = raw_crossing  # raw table value
        b = st["crossing_u_interp"]  # TSV value (reference for rel_diff)
        if a is None or b is None:
            st["rel_diff_crossing"] = None
        elif b == 0:
            st["rel_diff_crossing"] = float("inf")
        else:
            st["rel_diff_crossing"] = abs(a - b) / abs(b)

    # ---- outputs ----
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "gap_A_branch_summary.csv")
    cols = [
        "Jh_over_U", "u_start", "u_end", "n_points", "n_negative", "n_positive",
        "crossing_u_interp", "gap_max", "gap_max_u", "raw_crossing_u_interp",
        "rel_diff_crossing",
    ]
    with open(csv_path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for st in branches:
            row = []
            for c in cols:
                v = st.get(c)
                if v is None:
                    row.append("")
                elif isinstance(v, float) and (v != v):
                    row.append("")
                else:
                    row.append(repr(v) if isinstance(v, float) else str(v))
            f.write("\t".join(row) + "\n")

    jh_no_change = [st["Jh_over_U"] for st in branches
                     if st["crossing_u_interp"] is None]
    summary = {
        "branches": [
            {c: st.get(c) for c in cols} for st in branches
        ],
        "n_branches": 8,
        "n_branches_with_sign_change": sum(
            1 for st in branches if st["crossing_u_interp"] is not None
        ),
        "Jh_U_no_sign_change": jh_no_change,
    }
    json_path = os.path.join(OUT_DIR, "gap_A_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    # Console report
    print("Branch summary (TSV):")
    for st in branches:
        print(
            f"  Jh/U={st['Jh_over_U']:.2f}: n={st['n_points']} "
            f"u=[{st['u_start']}, {st['u_end']}] neg={st['n_negative']} "
            f"pos={st['n_positive']} crossing={st['crossing_u_interp']} "
            f"gap_max={st['gap_max']:.6f}@{st['gap_max_u']} "
            f"gap_min={st['gap_min']:.6f}@{st['gap_min_u']}"
        )
    print("Raw cross-check (table_009.dat):")
    for st in branches:
        print(
            f"  Jh/U={st['Jh_over_U']:.2f}: raw_crossing={st['raw_crossing_u_interp']} "
            f"rel_diff={st['rel_diff_crossing']}"
        )
    print(f"\nCSV -> {csv_path}")
    print(f"JSON -> {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
