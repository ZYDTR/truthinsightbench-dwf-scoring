#!/usr/bin/env python3
"""
local_observables_analysis.py
============================
Analysis of total magnetic moment and charge fluctuations as a function of
control ratio U/W (W=1).

Data provenance (only permitted sources for this task):
  * WS/data/normalized_views/control_vs_local_observables.tsv
  * WS/data/source_data/all_source_data_deidentified.zip  ->  table_007.dat

Analysis performed:
  (a) Monotonicity of both observables across successive U/W steps.
  (b) Linear regression (numpy.polyfit, degree 1) of each observable vs U/W.
  (c) CROSS-CHECK against raw table_007.dat:
        - report the REAL column count of the raw table (verify the header
          comment, which mentions both moment and charge fluctuations)
        - compare TSV total_magnetic_moment vs raw moment for every shared U/W
          and report the max relative difference
        - state whether the raw table contains charge fluctuations

Outputs:
  * WS/output/results/local_observables_summary.csv
  * WS/output/results/local_observables_summary.json
  * WS/output/analysis/local_observables_notes.md
"""
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import zipfile

WS = "/data/user/conversations/conv-af56a8cc-3c22-4b90-8a53-8142e2d7b40e/project"
TSV_PATH = os.path.join(WS, "data", "normalized_views", "control_vs_local_observables.tsv")
ZIP_PATH = os.path.join(WS, "data", "source_data", "all_source_data_deidentified.zip")
OUT_ANALYSIS_DIR = os.path.join(WS, "output", "analysis")
OUT_RESULTS_DIR = os.path.join(WS, "output", "results")

os.makedirs(OUT_ANALYSIS_DIR, exist_ok=True)
os.makedirs(OUT_RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# (1) Load the normalized view
# ---------------------------------------------------------------------------
df = pd.read_csv(TSV_PATH, sep="\t")
expected_cols = ["control_ratio", "total_magnetic_moment", "charge_fluctuation"]
if list(df.columns) != expected_cols:
    raise SystemExit(f"Unexpected TSV columns: {list(df.columns)}")
df = df[expected_cols].reset_index(drop=True)
n_rows = len(df)

u = df["control_ratio"].to_numpy(dtype=float)
moment = df["total_magnetic_moment"].to_numpy(dtype=float)
charge = df["charge_fluctuation"].to_numpy(dtype=float)

# ---------------------------------------------------------------------------
# (a) Monotonicity
#     moment  should be monotonically INCREASING  -> diffs >= 0
#     charge  should be monotonically DECREASING  -> diffs <= 0
# ---------------------------------------------------------------------------
d_moment = np.diff(moment)
d_charge = np.diff(charge)

moment_violations = int(np.sum(d_moment < 0))          # a decrease breaks monotonic increase
charge_violations = int(np.sum(d_charge > 0))          # an increase breaks monotonic decrease

min_d_moment = float(np.min(d_moment))
min_d_charge = float(np.min(d_charge))

# ---------------------------------------------------------------------------
# (b) Linear regression (numpy.polyfit degree 1)
# ---------------------------------------------------------------------------
def fit_r2(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")
    return float(slope), float(intercept), float(r2)

magnet_slope, magnet_intercept, magnet_r2 = fit_r2(u, moment)
charge_slope, charge_intercept, charge_r2 = fit_r2(u, charge)

# ---------------------------------------------------------------------------
# (c) CROSS-CHECK against raw table_007.dat
# ---------------------------------------------------------------------------
def parse_table007(text):
    """
    table_007.dat is NOT a single 3-column table. It contains two stacked
    2-column blocks:
        block 1: U/W , total magnetic moment T^2
        block 2: U/W , charge fluctuations delta n
    Each block is introduced by a header comment that names the 2nd column.
    Returns dict{block_name -> list of (U/W, value)}, plus the per-row
    field counts of every data line.
    """
    blocks = {"moment": [], "charge": []}
    current = None
    field_counts = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("#"):
            low = s.lower()
            if "magnetic moment" in low:
                current = "moment"
            elif "charge fluctuation" in low:
                current = "charge"
            continue
        parts = s.split()
        if len(parts) >= 2:
            field_counts.append(len(parts))
            try:
                uu = float(parts[0])
                val = float(parts[1])
                if current is not None:
                    blocks[current].append((uu, val))
            except ValueError:
                pass
    return blocks, field_counts

try:
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        txt = z.read("table_007.dat").decode("utf-8", errors="replace")
    blocks, field_counts = parse_table007(txt)
    raw_col_count = max(field_counts) if field_counts else None
    raw_moment = {uu: val for uu, val in blocks["moment"]}
    raw_charge = {uu: val for uu, val in blocks["charge"]}
    charge_available = bool(raw_charge)
except Exception as e:
    raw_col_count = None
    raw_moment = {}
    raw_charge = {}
    charge_available = None
    print(f"WARNING: table_007.dat cross-check failed: {e}", file=sys.stderr)

# Compare TSV moment vs raw moment on the intersection of U/W values.
# Match by rounded U/W key (float tolerance).
raw_check_max_rel_diff_moment = None
if raw_moment:
    tsv_moment_by_u = {float(x): float(y) for x, y in zip(u, moment)}
    diffs = []
    for raw_u, raw_val in raw_moment.items():
        key = round(raw_u, 6)
        if key in tsv_moment_by_u:
            tsv_val = tsv_moment_by_u[key]
            if raw_val != 0.0:
                diffs.append(abs(tsv_val - raw_val) / abs(raw_val))
    if diffs:
        raw_check_max_rel_diff_moment = float(max(diffs))
    else:
        print("WARNING: no U/W overlap between TSV and raw moment block", file=sys.stderr)

# ---------------------------------------------------------------------------
# Endpoint values (for key_numbers / notes)
# ---------------------------------------------------------------------------
u0 = float(u[0])
u_end = float(u[-1])
moment_at_0 = float(moment[0])
moment_at_end = float(moment[-1])
charge_at_0 = float(charge[0])
charge_at_end = float(charge[-1])

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
# CSV : verbatim copy of the 49-row view (traceability)
csv_path = os.path.join(OUT_RESULTS_DIR, "local_observables_summary.csv")
df_out = df.copy()
df_out.to_csv(csv_path, index=False)

# JSON : summary of all requested quantities
summary = {
    "mono_moment_violations": moment_violations,
    "mono_charge_violations": charge_violations,
    "magnet_slope": magnet_slope,
    "magnet_intercept": magnet_intercept,
    "magnet_R2": magnet_r2,
    "charge_slope": charge_slope,
    "charge_intercept": charge_intercept,
    "charge_R2": charge_r2,
    "raw_check_max_rel_diff_moment": raw_check_max_rel_diff_moment,
    "raw_check_charge_available": charge_available,
}
json_path = os.path.join(OUT_RESULTS_DIR, "local_observables_summary.json")
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)

# ---------------------------------------------------------------------------
# Notes file
# ---------------------------------------------------------------------------
notes_path = os.path.join(OUT_ANALYSIS_DIR, "local_observables_notes.md")
notes = f"""# local_observables — notes

Task: total magnetic moment and charge fluctuations vs control ratio U/W (W=1).

## Method
- Loaded the normalized view TSV (49 rows, U/W = 0..8).
- Monotonicity: successive differences; moment should be >= 0 (increasing),
  charge should be <= 0 (decreasing). Violations counted.
- Regression: numpy.polyfit degree 1; R^2 from (1 - SSres/SStot).
- Cross-check: extracted table_007.dat from the deidentified zip and parsed
  its real column layout.

## Data provenance
- WS/data/normalized_views/control_vs_local_observables.tsv
- WS/data/source_data/all_source_data_deidentified.zip :: table_007.dat

## Key numbers
| quantity | value |
|---|---|
| rows (U/W) | {n_rows} |
| moment @ U/W=0 | {moment_at_0} |
| moment @ U/W=8 | {moment_at_end} |
| charge @ U/W=0 | {charge_at_0} |
| charge @ U/W=8 | {charge_at_end} |
| moment min successive diff | {min_d_moment} |
| moment violations (decreases) | {moment_violations} |
| charge min successive diff | {min_d_charge} |
| charge violations (increases) | {charge_violations} |
| moment slope | {magnet_slope} |
| moment intercept | {magnet_intercept} |
| moment R^2 | {magnet_r2} |
| charge slope | {charge_slope} |
| charge intercept | {charge_intercept} |
| charge R^2 | {charge_r2} |
| raw table_007 real column count | {raw_col_count} |
| max rel diff moment (TSV vs raw) | {raw_check_max_rel_diff_moment} |
| raw charge available | {charge_available} |

## Raw-table cross-check (table_007.dat)
The header comment mentions BOTH total magnetic moment and charge fluctuations,
which could suggest a 3-column table. In reality each data row has only
{raw_col_count} column(s); the file holds TWO stacked 2-column blocks:
  block 1: U/W, total magnetic moment T^2
  block 2: U/W, charge fluctuations delta n
Charge fluctuations ARE present in the raw table (block 2), so
raw_check_charge_available = {charge_available}.
The TSV moment values match the raw moment block on the shared U/W set with
max relative difference {raw_check_max_rel_diff_moment}.

## Caveats
- U/W grid is not uniform: steps of 0.1 from 0 to 4, then steps of 0.5 from
  4 to 8 (49 points total).
- R^2 is for a single linear fit over the whole range; the moment curve is
  concave (saturating), so a straight line is only a global approximation.
- Max relative difference is computed as |TSV - raw| / |raw| over shared U/W.
"""
with open(notes_path, "w") as f:
    f.write(notes)

# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------
print("=== local_observables analysis ===")
print(f"rows: {n_rows}  (U/W range {u0}..{u_end})")
print(f"moment monotone inc: min diff={min_d_moment}, violations={moment_violations}")
print(f"charge  monotone dec: min diff={min_d_charge}, violations={charge_violations}")
print(f"moment fit: slope={magnet_slope:.6g} intercept={magnet_intercept:.6g} R2={magnet_r2:.6g}")
print(f"charge fit: slope={charge_slope:.6g} intercept={charge_intercept:.6g} R2={charge_r2:.6g}")
print(f"raw table_007 real column count: {raw_col_count}")
print(f"raw max rel diff (moment): {raw_check_max_rel_diff_moment}")
print(f"raw charge available: {charge_available}")
print(f"moment @0={moment_at_0} @8={moment_at_end}")
print(f"charge @0={charge_at_0} @8={charge_at_end}")
print("WROTE:", csv_path)
print("WROTE:", json_path)
print("WROTE:", notes_path)
print("=== done ===")
