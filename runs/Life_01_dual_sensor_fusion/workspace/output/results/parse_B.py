#!/usr/bin/env python3
"""
GROUP B - SURVIVAL & CA2+ KINETICS (Finding 3 support)

Parses two spreadsheet-style CSV tables (blocks of run columns, not tidy):
  1) source_table_02.csv  -> survival curves (block A, survival % vs bins)
  2) source_table_07.csv  -> Ca2+-triggered fusion levels (block A)

Every mean/SD is recomputed from the run columns with pandas (ddof=1); the
workbook Mean/S.D. columns are only read for cross-checking, never used as
evidence.

Outputs:
  output/results/survival_kinetics_stats.json
    {
      "survival_curves": { <condition>: [ {bin_s, survival_pct, survival_pct_sd} ... ] },
      "table07": { <condition>: {
          "no_Ca2_basal":     { "total_fusion": {mean, sd}, "fusion_first_two_frames": {mean, sd} },
          "0.5uM_Ca2_basal":  { "total_fusion": {mean, sd}, "fusion_first_two_frames": {mean, sd} }
        } }
    }
"""
import os
import json
import csv
import pandas as pd

BASE = "/data/user/conversations/conv-4bd502e1-a0d9-48ae-83ce-c85e4cbc094b/project"
DATA_DIR = os.path.join(BASE, "data", "source_tables")
OUT_DIR  = os.path.join(BASE, "output", "results")
os.makedirs(OUT_DIR, exist_ok=True)

JSON_PATH = os.path.join(OUT_DIR, "survival_kinetics_stats.json")

errors = []

def read_csv_raw(path):
    with open(path, newline="") as f:
        return [row for row in csv.reader(f)]

def to_float(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "na", "none"):
        return None
    try:
        return float(s)
    except ValueError:
        return None

def recompute(vals):
    """Recompute mean/sd (ddof=1) from a list of run values using pandas."""
    s = pd.Series([v for v in vals if v is not None])
    n = int(s.count())
    mean = float(s.mean()) if n > 0 else None
    sd = float(s.std(ddof=1)) if n >= 2 else None
    return {"mean": mean, "sd": sd, "n": n}

# ---------------------------------------------------------------------------
# Table 02: survival curves (block A)
# ---------------------------------------------------------------------------
def parse_table02():
    path = os.path.join(DATA_DIR, "source_table_02.csv")
    rows = read_csv_raw(path)
    n = len(rows)

    # Locate the 'Survival Percentage %' header line; the run header is the next row.
    surv_idx = None
    for i, r in enumerate(rows):
        if "Survival Percentage" in ",".join(r):
            surv_idx = i
            break
    if surv_idx is None:
        return None, ["table02: could not find 'Survival Percentage' row"]
    header_idx = surv_idx + 1
    header = rows[header_idx]

    run1_positions = [j for j, v in enumerate(header) if v.strip() == "run1"]
    if len(run1_positions) < 3:
        return None, [f"table02: expected >=3 run1 blocks, found {len(run1_positions)}"]

    label_row = rows[surv_idx - 1]  # 'Included in Bilayer' row
    fallback_names = ["None", "1:800 SensorB", "1:200 SensorB"]

    blocks = []
    for k, rp in enumerate(run1_positions[:3]):
        bins_col = rp - 1
        run_cols = [rp, rp + 1, rp + 2]
        mean_col = rp + 3
        sd_col = rp + 4
        cond = (label_row[bins_col].strip()
                if bins_col < len(label_row) else "")
        cond = cond if cond else fallback_names[k]

        bins, r1, r2, r3, wmean, wsd = [], [], [], [], [], []
        i = header_idx + 1
        while i < n:
            r = rows[i]
            if bins_col >= len(r):
                break
            bv = to_float(r[bins_col])
            if bv is None:          # blank / non-numeric -> end of block A data
                break
            bins.append(bv)
            r1.append(to_float(r[run_cols[0]]))
            r2.append(to_float(r[run_cols[1]]))
            r3.append(to_float(r[run_cols[2]]))
            wmean.append(to_float(r[mean_col]) if mean_col < len(r) else None)
            wsd.append(to_float(r[sd_col]) if sd_col < len(r) else None)
            i += 1
        blocks.append({
            "condition": cond, "bins": bins,
            "run1": r1, "run2": r2, "run3": r3,
            "workbook_mean": wmean, "workbook_sd": wsd,
        })
    return blocks, []

def build_survival(blocks):
    """Recompute per-bin mean/sd; emit at the 4 requested bins + the final bin."""
    target_bins = [0.147, 1.029, 3.087, 5.145]
    out = {}
    for b in blocks:
        cond = b["condition"]
        final_bin = b["bins"][-1] if b["bins"] else None
        points = []
        for tb in target_bins + ([final_bin] if final_bin is not None else []):
            idx = None
            for j, bv in enumerate(b["bins"]):
                if abs(bv - tb) < 1e-6:
                    idx = j
                    break
            if idx is None:
                errors.append(f"table02 {cond}: bin {tb} not found in data")
                continue
            stat = recompute([b["run1"][idx], b["run2"][idx], b["run3"][idx]])
            points.append({
                "bin_s": round(tb, 6),
                "survival_pct": round(stat["mean"], 4) if stat["mean"] is not None else None,
                "survival_pct_sd": round(stat["sd"], 4) if stat["sd"] is not None else None,
            })
        out[cond] = points
    return out

# ---------------------------------------------------------------------------
# Table 07: Ca2+-triggered fusion levels (block A)
# ---------------------------------------------------------------------------
def parse_table07():
    path = os.path.join(DATA_DIR, "source_table_07.csv")
    rows = read_csv_raw(path)
    n = len(rows)

    # Block A header is the row whose col1 (index 1) is 'run1' (col0 is empty
    # and holds the condition name in data rows).
    header_idx = None
    for i, r in enumerate(rows):
        if len(r) > 1 and r[1].strip() == "run1":
            header_idx = i
            break
    if header_idx is None:
        return None, ["table07: could not find block A header (col1=='run1')"]

    # 0-indexed run column lists (verified against raw data).
    colmap = {
        "no_Ca2_total":   [1, 2, 3, 5, 6],
        "no_Ca2_first":   [10, 11, 12, 13, 14],
        "0.5uM_total":    [18, 19, 20],
        "0.5uM_first":    [24, 25, 26],
    }

    records = []
    i = header_idx + 1
    while i < n and len(records) < 3:
        r = rows[i]
        if not r or not r[0].strip():
            i += 1
            continue
        name = r[0].strip()
        if name == "run1" or "Source" in name or "basal" in name:
            break
        if len(r) < 29:
            i += 1
            continue
        rec = {"condition": name}
        for key, cols in colmap.items():
            rec[key] = [to_float(r[c]) for c in cols if c < len(r)]
        records.append(rec)
        i += 1
    return records, []

def build_table07(records):
    out = {}
    for rec in records:
        cond = rec["condition"]
        out[cond] = {
            "no_Ca2_basal": {
                "total_fusion": recompute(rec["no_Ca2_total"]),
                "fusion_first_two_frames": recompute(rec["no_Ca2_first"]),
            },
            "0.5uM_Ca2_basal": {
                "total_fusion": recompute(rec["0.5uM_total"]),
                "fusion_first_two_frames": recompute(rec["0.5uM_first"]),
            },
        }
    return out

# ---------------------------------------------------------------------------
def main():
    blocks02, e02 = parse_table02()
    errors.extend(e02)
    survival = build_survival(blocks02) if blocks02 else {}

    records07, e07 = parse_table07()
    errors.extend(e07)
    table07 = build_table07(records07) if records07 else {}

    result = {
        "survival_curves": survival,
        "table07": table07,
    }
    # Round all floats for clean JSON.
    def _round(obj):
        if isinstance(obj, dict):
            return {k: _round(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_round(v) for v in obj]
        if isinstance(obj, float):
            return round(obj, 6)
        return obj
    result = _round(result)

    with open(JSON_PATH, "w") as f:
        json.dump(result, f, indent=2)

    # -----------------------------------------------------------------------
    # Stdout summary
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("GROUP B - SURVIVAL & CA2+ KINETICS")
    print("=" * 70)

    print("\n[1] SURVIVAL CURVES (source_table_02.csv, block A) "
          "recomputed mean +/- sd (ddof=1) of run1-3 per condition:")
    print(f"    n_bins per condition = {len(blocks02[0]['bins']) if blocks02 else '?'} "
          f"(final bin = {blocks02[0]['bins'][-1] if blocks02 else '?'})")
    for b in blocks02:
        cond = b["condition"]
        print(f"\n    {cond}:")
        for p in survival.get(cond, []):
            print(f"      t={p['bin_s']:.3f}s  survival={p['survival_pct']:.4f} "
                  f"+/-{p['survival_pct_sd']:.4f}  (workbook mean check "
                  f"see below)")

    print("\n    Cross-check vs workbook Mean (first 3 bins, per condition):")
    for b in blocks02:
        cond = b["condition"]
        for j in range(min(3, len(b["bins"]))):
            bv = b["bins"][j]
            stat = recompute([b["run1"][j], b["run2"][j], b["run3"][j]])
            wm = b["workbook_mean"][j] if j < len(b["workbook_mean"]) else None
            diff = (abs(stat["mean"] - wm) if (stat["mean"] is not None and wm is not None) else None)
            print(f"      {cond} t={bv:.3f}s  recomputed={stat['mean']:.4f} "
                  f"workbook={wm} diff={diff:.4f}" if wm is not None else f"      {cond} t={bv:.3f}s  recomputed={stat['mean']:.4f} (no wb)")

    print("\n[2] CA2+ TRIGGERED FUSION LEVELS (source_table_07.csv, block A) "
          "recomputed mean +/- sd (ddof=1):")
    for cond, d in table07.items():
        for basal, m in [("no_Ca2_basal", d["no_Ca2_basal"]),
                          ("0.5uM_Ca2_basal", d["0.5uM_Ca2_basal"])]:
            tf = d[basal]["total_fusion"]
            ftf = d[basal]["fusion_first_two_frames"]
            print(f"\n    {cond}  [{basal}]")
            print(f"      total_fusion            mean={tf['mean']:.4f} sd={tf['sd']:.4f} (n={tf['n']})")
            print(f"      fusion_first_two_frames mean={ftf['mean']:.4f} sd={ftf['sd']:.4f} (n={ftf['n']})")

    print("\n" + "=" * 70)
    print(f"JSON written: {JSON_PATH}")
    if errors:
        print("\nERRORS / WARNINGS:")
        for err in errors:
            print("  -", err)
    else:
        print("No errors.")

if __name__ == "__main__":
    main()
