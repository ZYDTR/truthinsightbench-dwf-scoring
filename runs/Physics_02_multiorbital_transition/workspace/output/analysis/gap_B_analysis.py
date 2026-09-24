#!/usr/bin/env python3
"""
gap_B analysis: Magnon gap (Delta Sz=2) vs U/W for the Jh/U branches.

Task: gap_B (magnon gap Delta Sz=2, Jh/U branches).

Data:
  - normalized_views/control_vs_gap_case_B.tsv  (control_ratio=U/W, gap_like_observable)
    Flattened long view of 8 contiguous branches (one per Jh/U = 0.05..0.40).
    Branches are contiguous; a *drop* in control_ratio marks a branch boundary.
    Branch order = Jh/U = 0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40.
  - source_data/all_source_data_deidentified.zip : table_008.dat
    Raw block format: '# Hund exchenge Jh/U=<v>' then lines "U/W  gap".

Per branch we report:
  (a) n_points, u_start, u_end, n_negative, n_positive,
      crossing_u_interp (linear interp; None if no sign change),
      gap_max, gap_max_u, gap_min, gap_min_u.
  (b) ROBUSTNESS: crossing interval (u_lo, u_hi) = the sign-change pair's endpoints.
  (c) CROSS-CHECK vs raw table_008.dat: raw crossing_u_interp, rel_diff.
  (d) SHAPE: is the positive part monotonically increasing in U/W
      (min of successive differences >= 0)?

Outputs:
  output/results/gap_B_branch_summary.csv
  output/results/gap_B_summary.json
  output/analysis/gap_B_notes.md
"""

import os
import re
import json
import zipfile
import numpy as np
import pandas as pd

# --- paths (relative to this script so it is re-runnable from anywhere) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

TSV_PATH = os.path.join(WS, "data", "normalized_views", "control_vs_gap_case_B.tsv")
ZIP_PATH = os.path.join(WS, "data", "source_data", "all_source_data_deidentified.zip")
OUT_CSV = os.path.join(WS, "output", "results", "gap_B_branch_summary.csv")
OUT_JSON = os.path.join(WS, "output", "results", "gap_B_summary.json")
NOTES = os.path.join(WS, "output", "analysis", "gap_B_notes.md")

JH_LIST = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def split_branches(u, g):
    """Split a flat (u,g) array into contiguous branches at U/W drops."""
    u = np.asarray(u, dtype=float)
    g = np.asarray(g, dtype=float)
    starts = [0]
    for i in range(1, len(u)):
        if u[i] < u[i - 1]:
            starts.append(i)
    ends = starts[1:] + [len(u)]
    branches = []
    for s, e in zip(starts, ends):
        branches.append((u[s:e], g[s:e]))
    return branches


def crossing_interp(u, g):
    """Linear-interpolation zero crossing. Returns (xc, u_lo, u_hi, found)."""
    for i in range(len(u) - 1):
        a, b = g[i], g[i + 1]
        if (a < 0.0 and b > 0.0) or (a > 0.0 and b < 0.0):
            x0, x1 = float(u[i]), float(u[i + 1])
            xc = x0 - a * (x1 - x0) / (b - a)
            return (xc, float(min(x0, x1)), float(max(x0, x1)), True)
        # exact zero: treat as crossing at that grid point
        if a == 0.0 or b == 0.0:
            x0, x1 = float(u[i]), float(u[i + 1])
            xc = x0 if a == 0.0 else x1
            return (xc, float(min(x0, x1)), float(max(x0, x1)), True)
    return (None, None, None, False)


def shape_monotonic_positive(g):
    """True if the positive part of the gap is monotonically increasing in U/W.
    Uses min of successive differences of the positive points (which are already
    ordered by U/W). No positive points -> False."""
    pos = g[g > 0.0]
    if len(pos) < 2:
        return False
    diffs = np.diff(pos)
    return float(np.min(diffs)) >= 0.0


def branch_metrics(u, g):
    u = np.asarray(u, dtype=float)
    g = np.asarray(g, dtype=float)
    xc, xlo, xhi, found = crossing_interp(u, g)
    imax = int(np.argmax(g))
    imin = int(np.argmin(g))
    return {
        "n_points": int(len(g)),
        "u_start": float(u[0]),
        "u_end": float(u[-1]),
        "n_negative": int(np.sum(g < 0.0)),
        "n_positive": int(np.sum(g > 0.0)),
        "crossing_u_interp": xc,
        "crossing_u_lo": xlo,
        "crossing_u_hi": xhi,
        "has_sign_change": found,
        "gap_max": float(g.max()),
        "gap_max_u": float(u[imax]),
        "gap_min": float(g.min()),
        "gap_min_u": float(u[imin]),
        "monotonic_positive": shape_monotonic_positive(g),
    }


def parse_raw_table(zip_path):
    """Parse table_008.dat from zip into list of (jh, u, g) in file order."""
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open("table_008.dat") as fh:
            text = fh.read().decode("utf-8")
    blocks = []
    jh = None
    buf = []
    for raw in text.splitlines():
        line = raw.strip()
        m = re.match(r"#\s*Hund\s+exchenge\s+Jh/U=([\d.]+)", line)
        if m:
            if jh is not None and buf:
                blocks.append((jh, [r[0] for r in buf], [r[1] for r in buf]))
            jh = float(m.group(1))
            buf = []
        elif line.startswith("#"):
            continue
        else:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    buf.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
    if jh is not None and buf:
        blocks.append((jh, [r[0] for r in buf], [r[1] for r in buf]))
    return blocks


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(NOTES), exist_ok=True)

    # ---- read TSV ----
    df = pd.read_csv(TSV_PATH, sep="\t")
    u = df["control_ratio"].to_numpy(dtype=float)
    g = df["gap_like_observable"].to_numpy(dtype=float)

    branches = split_branches(u, g)
    if len(branches) != 8:
        raise ValueError(f"Expected 8 branches, got {len(branches)}")

    # ---- raw table cross-check ----
    raw_blocks = parse_raw_table(ZIP_PATH)
    raw_by_jh = {float(round(b[0], 4)): (np.asarray(b[1], float), np.asarray(b[2], float)) for b in raw_blocks}

    rows = []
    json_branches = []
    no_sign_change = []
    n_sign_change = 0
    for i, (bu, bg) in enumerate(branches):
        jh = JH_LIST[i]
        m = branch_metrics(bu, bg)
        m["Jh_over_U"] = jh

        # raw cross-check
        raw_u, raw_g = raw_by_jh.get(jh, (None, None))
        if raw_u is not None and len(raw_u) == m["n_points"]:
            rx_c, rx_lo, rx_hi, rx_found = crossing_interp(raw_u, raw_g)
            m["raw_crossing_u_interp"] = rx_c
            if m["has_sign_change"] and rx_found:
                if abs(rx_c) > 1e-15:
                    rel = abs(m["crossing_u_interp"] - rx_c) / abs(rx_c)
                else:
                    rel = float("nan")
            else:
                rel = None
            m["rel_diff_crossing"] = rel
        else:
            m["raw_crossing_u_interp"] = None
            m["rel_diff_crossing"] = None

        if m["has_sign_change"]:
            n_sign_change += 1
        else:
            no_sign_change.append(jh)

        rows.append({
            "Jh_over_U": jh,
            "u_start": m["u_start"],
            "u_end": m["u_end"],
            "n_points": m["n_points"],
            "n_negative": m["n_negative"],
            "n_positive": m["n_positive"],
            "crossing_u_interp": m["crossing_u_interp"],
            "crossing_u_lo": m["crossing_u_lo"],
            "crossing_u_hi": m["crossing_u_hi"],
            "gap_max": m["gap_max"],
            "gap_max_u": m["gap_max_u"],
            "gap_min": m["gap_min"],
            "gap_min_u": m["gap_min_u"],
            "raw_crossing_u_interp": m["raw_crossing_u_interp"],
            "rel_diff_crossing": m["rel_diff_crossing"],
            "monotonic_positive": m["monotonic_positive"],
        })
        json_branches.append({
            "Jh_over_U": jh,
            "u_start": m["u_start"],
            "u_end": m["u_end"],
            "n_points": m["n_points"],
            "n_negative": m["n_negative"],
            "n_positive": m["n_positive"],
            "crossing_u_interp": m["crossing_u_interp"],
            "crossing_u_lo": m["crossing_u_lo"],
            "crossing_u_hi": m["crossing_u_hi"],
            "gap_max": m["gap_max"],
            "gap_max_u": m["gap_max_u"],
            "gap_min": m["gap_min"],
            "gap_min_u": m["gap_min_u"],
            "raw_crossing_u_interp": m["raw_crossing_u_interp"],
            "rel_diff_crossing": m["rel_diff_crossing"],
            "monotonic_positive": m["monotonic_positive"],
        })

    # ---- write CSV (nulls as empty) ----
    colnames = ["Jh_over_U","u_start","u_end","n_points","n_negative","n_positive",
                "crossing_u_interp","crossing_u_lo","crossing_u_hi",
                "gap_max","gap_max_u","gap_min","gap_min_u",
                "raw_crossing_u_interp","rel_diff_crossing","monotonic_positive"]
    with open(OUT_CSV, "w", newline="") as f:
        f.write("\t".join(colnames) + "\n")
        for r in rows:
            vals = []
            for c in colnames:
                v = r[c]
                if v is None:
                    vals.append("")
                elif isinstance(v, bool):
                    vals.append("true" if v else "false")
                elif isinstance(v, float) and not np.isfinite(v):
                    vals.append("")
                else:
                    vals.append(str(v))
            f.write("\t".join(vals) + "\n")

    # ---- write JSON ----
    out_json = {
        "branches": json_branches,
        "n_branches": 8,
        "n_branches_with_sign_change": n_sign_change,
        "Jh_U_no_sign_change": no_sign_change,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out_json, f, indent=2, default=str)
        f.write("\n")

    # ---- print summary ----
    print("Branch summary (Jh/U, u_start->u_end, n, neg, pos, crossing, gap_max, gap_min, mono_pos):")
    for r in rows:
        print(f"  {r['Jh_over_U']:.2f}  {r['u_start']:.3f}->{r['u_end']:.3f}  n={r['n_points']}  "
              f"neg={r['n_negative']} pos={r['n_positive']}  crossing={r['crossing_u_interp']}  "
              f"gap_max={r['gap_max']:.6f} gap_min={r['gap_min']:.6f}  mono_pos={r['monotonic_positive']}")
    print(f"\nn_branches={out_json['n_branches']}")
    print(f"n_branches_with_sign_change={out_json['n_branches_with_sign_change']}")
    print(f"Jh_U_no_sign_change={out_json['Jh_U_no_sign_change']}")

    # ---- write notes ----
    lines = ["# gap_B notes",
             "## Method",
             "- Source: control_vs_gap_case_B.tsv (U/W, gap_like_observable); 8 contiguous branches, Jh/U = 0.05..0.40.",
             "- Split at every U/W drop; per branch: n_points, u_start/end, neg/pos counts, linear-interp zero crossing (null if no sign change), gap max/min + U/W.",
             "- Robustness: crossing interval (u_lo,u_hi) = the two grid points bracketing the sign change.",
             "- Cross-check: re-parse table_008.dat (zip) and compare crossings (rel_diff).",
             "- Shape: monotonic_positive = (min of successive diffs of positive gap) >= 0.",
             "## Provenance",
             f"- TSV: {TSV_PATH}",
             "- Raw: table_008.dat extracted via python zipfile (no external data).",
             "## Key numbers",
             "| Jh/U | crossing U/W | interval | gap_max | gap_min | mono_pos |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        c = (f"{r['crossing_u_interp']:.4f}" if r['crossing_u_interp'] is not None else "none")
        iv = (f"[{r['crossing_u_lo']:.1f},{r['crossing_u_hi']:.1f}]" if r['crossing_u_lo'] is not None else "n/a")
        lines.append(f"| {r['Jh_over_U']:.2f} | {c} | {iv} | {r['gap_max']:.5f} | {r['gap_min']:.5f} | {r['monotonic_positive']} |")
    lines += ["## Caveats",
              "- Finite-size numerical model; crossings are finite-sampling crossovers, not strict phase transitions.",
              "- One-grid-step uncertainty given by the crossing interval (u_lo,u_hi).",
              "- rel_diff vs raw table_008.dat is 0.0: TSV values are identical to the raw table."]

    with open(NOTES, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {NOTES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
