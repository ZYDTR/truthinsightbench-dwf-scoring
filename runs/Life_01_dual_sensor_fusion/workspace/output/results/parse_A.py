#!/usr/bin/env python3
"""GROUP A - FUSION DOSE (Finding 1 support).

Parse 4 spreadsheet-style (run-column blocks) source tables into run-level
long frames, recompute per (source, condition, metric) mean/sd with pandas
(ddof=1), cross-check source_table_01 block B against the derived run file,
and write fusion_dose_stats.json.

The independent analysis unit is ONE run. Every mean/SD is recomputed from
the run columns with pandas (ddof=1); the workbook Mean/S.D. columns are NOT
used as evidence.

Column layout is detected dynamically from the run header row (the row whose
cells contain run1..run5):
  - the first run1..run5 group  -> metric "total_fusion"
  - the second run1..run5 group -> metric "fusion_first_two_frames"
The condition column is the one whose header label contains "Included"
(one column to the left of the metric label). Missing cells ("-", blank) are
excluded from mean/sd and reported as null in runs[].

Tables (rows are 1-indexed in the raw CSV; the exact data rows are located
by the block marker + run header):
  1) source_table_01.csv  block "Source block B"
  2) source_table_03.csv  TOP PANEL
  3) source_table_04.csv  TOP PANEL
  4) source_table_08.csv  block "Source block C"
"""
import json
import math
import os

import pandas as pd

BASE = "/data/user/conversations/conv-4bd502e1-a0d9-48ae-83ce-c85e4cbc094b/project"
DATA_DIR = os.path.join(BASE, "data", "source_tables")
OUT_DIR = os.path.join(BASE, "output", "results")
DERIVED = os.path.join(
    BASE, "data", "derived_for_navigation", "fusion_dose_independent_runs.csv"
)
LONG_CSV = os.path.join(OUT_DIR, "parsed_fusion_dose_long.csv")
JSON_OUT = os.path.join(OUT_DIR, "fusion_dose_stats.json")

os.makedirs(OUT_DIR, exist_ok=True)

MISSING_TOKENS = {"", "-", "--", "n/a", "na", "none", "nan", "null", "?"}
RUNS = [1, 2, 3, 4, 5]
METRICS = ["total_fusion", "fusion_first_two_frames"]
TOLERANCE = 0.01

TABLES = [
    dict(source="source_table_01", marker="Source block B"),
    dict(source="source_table_03", marker="TOP PANEL"),
    dict(source="source_table_04", marker="TOP PANEL"),
    dict(source="source_table_08", marker="Source block C"),
]


def read_rows(path):
    """Read a CSV into a list of raw string lists (spreadsheet style)."""
    with open(path, newline="") as f:
        return [line.rstrip("\n").split(",") for line in f]


def to_num(tok):
    """Parse a cell to float; missing/blank/'-' tokens -> None."""
    if tok is None:
        return None
    t = tok.strip()
    if t.lower() in MISSING_TOKENS:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def find_block_marker(rows, marker):
    """Index of the first row whose full content equals the block marker."""
    for i, row in enumerate(rows):
        if " ".join(row).strip().lower() == marker.lower():
            return i
    return None


def find_run_header(rows, start):
    """
    First row at/after `start` that is the run-column header row
    (contains run1..run5), i.e. the row immediately above the data rows.
    """
    for i in range(start, len(rows)):
        joined = " ".join(rows[i]).lower()
        if "run1" in joined and "run5" in joined:
            return i
    return None


def run_groups(header):
    """
    Given a run header row, return a list of groups. Each group is a dict
    {run: 0-indexed column} for run1..run5, in the order the groups appear.
    Group 0 = total_fusion, group 1 = fusion_first_two_frames.
    """
    groups = []
    i = 0
    n = len(header)
    while i < n:
        group = {}
        matched_any = False
        for run in RUNS:
            tok = header[i + run - 1].strip() if i + run - 1 < n else ""
            if tok == f"run{run}":
                group[run] = i + run - 1
                matched_any = True
        if matched_any:
            groups.append(group)
            i += 5
        else:
            i += 1
    return groups


def find_condition_col(rows, run_hdr):
    """
    Find the 0-indexed condition column: the column whose label (in the row
    directly above the run header) contains the word 'Included'.
    """
    for i in range(run_hdr - 1, max(-1, run_hdr - 3), -1):
        row = rows[i]
        for c, tok in enumerate(row):
            if "included" in tok.lower():
                return c
    # fallback: first non-empty column that is not a run/mean/sd token
    for c, tok in enumerate(rows[run_hdr]):
        if tok.strip() and tok.strip().lower() not in (
            "run1", "run2", "run3", "run4", "run5", "mean", "s.d.", "s.d."
        ):
            return c
    return 0


def parse_table(spec):
    """Parse one spreadsheet block into run-level long records."""
    path = os.path.join(DATA_DIR, f"{spec['source']}.csv")
    rows = read_rows(path)
    m = find_block_marker(rows, spec["marker"])
    if m is None:
        raise RuntimeError(f"{spec['source']}: block marker {spec['marker']!r} not found")
    hdr = find_run_header(rows, m)
    if hdr is None:
        raise RuntimeError(f"{spec['source']}: run header row not found after block marker")

    groups = run_groups(rows[hdr])
    cond_col = find_condition_col(rows, hdr)

    # Build metric -> {run: col} mapping.
    metric_cols = {}
    if len(groups) >= 1:
        metric_cols["total_fusion"] = groups[0]
    if len(groups) >= 2:
        metric_cols["fusion_first_two_frames"] = groups[1]

    records = []
    for i in range(hdr + 1, len(rows)):
        row = rows[i]
        cond = row[cond_col].strip() if cond_col < len(row) else ""
        if cond == "":
            break  # end of data block (separator/blank rows)
        for metric, mcols in metric_cols.items():
            for run in RUNS:
                col = mcols.get(run)
                val = to_num(row[col]) if col is not None else None
                records.append(
                    {
                        "source": spec["source"],
                        "condition": cond,
                        "metric": metric,
                        "run": run,
                        "value_pct": val,
                    }
                )
    return records


def main():
    long_rows = []
    for spec in TABLES:
        long_rows.extend(parse_table(spec))

    df = pd.DataFrame(long_rows)
    df["value_pct"] = pd.to_numeric(df["value_pct"], errors="coerce")
    df.to_csv(LONG_CSV, index=False)

    # ---- recompute per (source, condition, metric) ----
    stats = {}
    order = []
    for (src, cond, metric), g in df.groupby(["source", "condition", "metric"], sort=False):
        vals = g["value_pct"]
        run_list = [None if math.isnan(v) else float(v) for v in vals.tolist()]
        n = int(vals.count())
        rec = {
            "source": src,
            "condition": cond,
            "metric": metric,
            "mean": round(float(vals.mean()), 4),
            "sd": round(float(vals.std(ddof=1)), 4) if n > 1 else None,
            "n_runs": n,
            "runs": run_list,
        }
        key = f"{src}__{cond}"
        if key not in stats:
            stats[key] = {
                "source": src,
                "condition": cond,
                "total_fusion": None,
                "fusion_first_two_frames": None,
            }
        stats[key][metric] = rec
        order.append((src, cond, metric))

    # ---- cross-check source_table_01 block B vs derived run file ----
    cross_check = {
        "derived_file": "data/derived_for_navigation/fusion_dose_independent_runs.csv",
        "checked_source_block": "source_table_01 / Source block B",
        "tolerance": TOLERANCE,
        "n_values_checked": 0,
        "n_values_mismatched_gt_tolerance": 0,
        "max_abs_diff": None,
        "passed": False,
    }
    try:
        # keep_default_na=False so the literal condition label "None" is not
        # coerced to NaN (which would break the join against the in-memory df).
        derived = pd.read_csv(DERIVED, keep_default_na=False)
        derived = derived.rename(columns={"value_pct": "derived_value_pct"})
        check_df = df[df["source"] == "source_table_01"].copy()
        merged = check_df.merge(derived, on=["condition", "metric", "run"], how="inner")
        merged["abs_diff"] = (merged["value_pct"] - merged["derived_value_pct"]).abs()
        cross_check["n_values_checked"] = int(len(merged))
        cross_check["n_values_mismatched_gt_tolerance"] = int(
            (merged["abs_diff"] > TOLERANCE).sum()
        )
        cross_check["max_abs_diff"] = round(float(merged["abs_diff"].max()), 6)
        cross_check["passed"] = cross_check["max_abs_diff"] <= TOLERANCE
    except Exception as exc:
        cross_check["error"] = f"{type(exc).__name__}: {exc}"

    json_out = {"stats": stats, "cross_check": cross_check}
    with open(JSON_OUT, "w") as f:
        json.dump(json_out, f, indent=2)

    # ---- stdout summary ----
    print("=" * 100)
    print("GROUP A - FUSION DOSE (Finding 1 support)")
    print("=" * 100)
    print(f"Long frame rows: {len(df)} -> {LONG_CSV}")
    print(f"Stats written:   {JSON_OUT}")
    print()
    print(f"{'source':<17} {'condition':<24} {'metric':<26} {'mean':>8} {'sd':>9} {'n':>2}")
    print("-" * 100)
    for (src, cond, metric) in order:
        rec = stats[f"{src}__{cond}"][metric]
        sd = rec["sd"] if rec["sd"] is not None else float("nan")
        print(
            f"{src:<17} {cond:<24} {metric:<26} {rec['mean']:>8.4f} "
            f"{sd:>9.4f} {rec['n_runs']:>2}"
        )
    print()
    print("Cross-check (source_table_01 block B vs derived_for_navigation/fusion_dose_independent_runs.csv):")
    if "error" in cross_check:
        print(f"  ERROR: {cross_check['error']}")
    else:
        print(
            f"  n_values_checked = {cross_check['n_values_checked']}, "
            f"n_mismatched (> {TOLERANCE}) = "
            f"{cross_check['n_values_mismatched_gt_tolerance']}, "
            f"max_abs_diff = {cross_check['max_abs_diff']}, "
            f"PASS = {cross_check['passed']}"
        )
    print()
    print("JSON keys (source__condition):")
    for k in stats:
        print(f"  {k}")
    print("Done.")


if __name__ == "__main__":
    main()
