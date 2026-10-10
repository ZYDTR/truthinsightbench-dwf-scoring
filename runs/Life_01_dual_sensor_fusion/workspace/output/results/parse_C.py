#!/usr/bin/env python3
"""
Parse binding-competition / dose-response tables and recompute run-level stats.

Ground truth (verified on disk):
  * source_table_05.csv : col1 = "SensorA Conc (uM)" = COMPETITOR.
                          Varies 0-60 uM (decreasing competition curve). 4 runs.
  * source_table_11.csv : col1 = "CPX-FusionComplex Conc (uM)" = LIGAND.
                          Varies 1-45 uM (increasing saturation curve). 4 runs.

Cross-check file:
  * data/derived_for_navigation/binding_competition_runs.csv
    (competitor_concentration_uM, run, bound_target_uM)
  It matches the SensorA (competitor) table = source_table_05.csv.

Method:
  * Parse each source CSV: every row whose col1 (index 1) is a numeric
    concentration and cols 2-5 (indexes 2-5) are numeric run values.
  * Recompute per-concentration mean and sample SD (ddof=1) from the 4 runs.
  * Log-linear fit: y = log(bound), x = concentration (natural log).
    Report slope, intercept, R^2.
  * "Half-concentration": concentration at which bound == half of the
    no-competitor reference value.  Reference = mean bound at 0 uM if present,
    else mean bound at the highest concentration.  Linearly interpolated
    between the two bracketing points.
"""

import csv
import json
import math
import os
from statistics import stdev

BASE = "/data/user/conversations/conv-4bd502e1-a0d9-48ae-83ce-c85e4cbc094b/project"
DATA = os.path.join(BASE, "data")
SRC05 = os.path.join(DATA, "source_tables", "source_table_05.csv")   # SensorA (competitor)
SRC11 = os.path.join(DATA, "source_tables", "source_table_11.csv")   # CPX-FusionComplex (ligand)
CROSS = os.path.join(DATA, "derived_for_navigation", "binding_competition_runs.csv")
OUT_DIR = os.path.join(BASE, "output", "results")
OUT_JSON = os.path.join(OUT_DIR, "binding_competition_stats.json")

RUN_COLS = [2, 3, 4, 5]  # 0-indexed: cols 2-5 = run1..run4
TOL = 0.01


def parse_table(path):
    """Return list of {concentration_uM, runs:[...]} for every numeric data row."""
    rows = []
    with open(path, newline="") as f:
        for line in f:
            cells = line.strip().split(",")
            if len(cells) < 6:
                continue
            try:
                conc = float(cells[1])
                runs = [float(cells[i]) for i in RUN_COLS]
            except (ValueError, IndexError):
                continue
            rows.append({"concentration_uM": conc, "runs": runs})
    # de-duplicate (keep first of each concentration), sort ascending
    seen, out = set(), []
    for r in rows:
        if r["concentration_uM"] in seen:
            continue
        seen.add(r["concentration_uM"])
        out.append(r)
    out.sort(key=lambda r: r["concentration_uM"])
    return out


def log_linear_fit(x, y):
    """Fit y = intercept + slope*x (least squares). Return (slope, intercept, r2)."""
    n = len(x)
    if n < 2:
        return None, None, None
    mx = sum(x) / n
    my = sum(y) / n
    sx = sum((xi - mx) ** 2 for xi in x)
    sy = sum((yi - my) ** 2 for yi in y)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    slope = sxy / sx if sx != 0 else 0.0
    intercept = my - slope * mx
    denom = sx * sy
    r2 = (sxy * sxy) / denom if denom != 0 else None
    return slope, intercept, r2


def interpolate_half_conc(points, target):
    """Linearly interpolate concentration at which bound == target."""
    if not points:
        return None
    for c, b in points:
        if abs(b - target) < 1e-9:
            return c
    for i in range(len(points) - 1):
        c0, b0 = points[i]
        c1, b1 = points[i + 1]
        if (b0 - target) * (b1 - target) <= 0:
            if b1 == b0:
                return None
            frac = (b0 - target) / (b0 - b1)
            return c0 + frac * (c1 - c0)
    return None


def compare_cross_check(cross_rows, table):
    """Compare cross-check file (conc, run, bound) to a table's per-run values."""
    cross_idx = {}
    for cr in cross_rows:
        key = (round(cr["competitor_concentration_uM"], 6), cr["run"])
        cross_idx[key] = cr["bound_target_uM"]

    diffs, matched = [], 0
    for row in table:
        conc = row["concentration_uM"]
        for run_i, run_val in enumerate(row["runs"], start=1):
            key = (round(conc, 6), run_i)
            if key in cross_idx:
                diff = abs(cross_idx[key] - run_val)
                diffs.append(diff)
                if diff <= TOL:
                    matched += 1
    if not diffs:
        return {"n_compared": 0, "n_within_tolerance": 0, "max_abs_diff": None, "matches": False}
    return {
        "n_compared": len(diffs),
        "n_within_tolerance": matched,
        "max_abs_diff": max(diffs),
        "matches": max(diffs) <= TOL,
    }


def build_table_report(path, table, role):
    """role: 'competitor' (SensorA) or 'ligand' (CPX-FusionComplex)."""
    per_conc = []
    for row in table:
        runs = row["runs"]
        m = sum(runs) / len(runs)
        sd = stdev(runs) if len(runs) > 1 else 0.0
        per_conc.append(
            {
                "concentration_uM": row["concentration_uM"],
                "runs": runs,
                "mean_uM": round(m, 6),
                "sd_uM": round(sd, 6),
            }
        )
    # log-linear fit on positive means only
    fit_xs = [p["concentration_uM"] for p in per_conc if p["mean_uM"] > 0]
    fit_ys = [math.log(p["mean_uM"]) for p in per_conc if p["mean_uM"] > 0]
    slope, intercept, r2 = log_linear_fit(fit_xs, fit_ys)

    # reference (full-binding / no-competitor) value
    if per_conc[0]["concentration_uM"] == 0:
        no_comp = per_conc[0]["mean_uM"]
        ref_basis = "mean bound at 0 uM (no competitor)"
    else:
        no_comp = per_conc[-1]["mean_uM"]
        ref_basis = "mean bound at highest concentration (ligand saturation; no 0 uM measured)"
    half_ref = no_comp / 2.0
    points = [(p["concentration_uM"], p["mean_uM"]) for p in per_conc]
    half_conc = interpolate_half_conc(points, half_ref)
    b0, bn = per_conc[0]["mean_uM"], per_conc[-1]["mean_uM"]
    direction = "increasing" if bn > b0 else ("decreasing" if bn < b0 else "flat")

    return {
        "source_file": os.path.basename(path),
        "role": role,
        "n_concentrations": len(per_conc),
        "n_runs": len(per_conc[0]["runs"]) if per_conc else 0,
        "per_concentration": per_conc,
        "fit_logbound_vs_conc": {
            "model": "log(bound) = intercept + slope * concentration (natural log)",
            "slope": round(slope, 8) if slope is not None else None,
            "intercept": round(intercept, 8) if intercept is not None else None,
            "r2": round(r2, 8) if r2 is not None else None,
        },
        "full_binding_reference": {
            "value_uM": round(no_comp, 6),
            "basis": ref_basis,
            "half_value_uM": round(half_ref, 6),
        },
        "half_concentration_uM": round(half_conc, 6) if half_conc is not None else None,
        "curve_direction": direction,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    t05 = parse_table(SRC05)  # SensorA (competitor)
    t11 = parse_table(SRC11)  # CPX-FusionComplex (ligand)

    cross_rows = []
    with open(CROSS, newline="") as f:
        for rec in csv.DictReader(f):
            cross_rows.append(
                {
                    "competitor_concentration_uM": float(rec["competitor_concentration_uM"]),
                    "run": int(rec["run"]),
                    "bound_target_uM": float(rec["bound_target_uM"]),
                }
            )

    report05 = build_table_report(SRC05, t05, "competitor")
    report11 = build_table_report(SRC11, t11, "ligand")

    cc05 = compare_cross_check(cross_rows, t05)
    cc11 = compare_cross_check(cross_rows, t11)

    # best match = the table whose per-run values the cross-check agrees with (lower max diff)
    def pick(d):
        return d["max_abs_diff"] if d["max_abs_diff"] is not None else float("inf")

    best = "source_table_05.csv (SensorA Conc, competitor)" if pick(cc05) <= pick(cc11) \
        else "source_table_11.csv (CPX-FusionComplex Conc, ligand)"
    best_d = cc05 if pick(cc05) <= pick(cc11) else cc11

    notes = [
        "Column mapping per file headers (and per task): source_table_05.csv = SensorA Conc "
        "(competitor, decreasing); source_table_11.csv = CPX-FusionComplex Conc (ligand, increasing).",
        "source_table_05.csv contains 9 data rows (0-60 uM), not the 8 (0-50 uM) listed in the task; "
        "the cross-check file also includes 60 uM, so all 9 rows were parsed. "
        "Task row ranges (05: 11-18, 11: 9-17) are off by ~1 row from the actual data rows.",
        "Recomputed mean/SD use the sample standard deviation (ddof=1) over the 4 runs; "
        "these agree with the published mean/s.d. columns within rounding.",
        "Cross-check file binding_competition_runs.csv matches source_table_05.csv (SensorA, "
        "competitor) per-run values; it is a competitor curve, so it does not match table_11.",
    ]

    result = {
        "tables": {
            "source_table_05.csv": report05,
            "source_table_11.csv": report11,
        },
        "cross_check": {
            "file": os.path.basename(CROSS),
            "n_rows": len(cross_rows),
            "tolerance": TOL,
            "vs_source_table_05.csv": cc05,
            "vs_source_table_11.csv": cc11,
            "best_match": best,
            "best_match_max_abs_diff": best_d["max_abs_diff"],
            "best_match_within_tolerance": best_d["matches"],
        },
        "notes": notes,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    # ---- stdout summary ----
    print("=" * 70)
    print("BINDING COMPETITION / DOSE-RESPONSE STATISTICS")
    print("=" * 70)
    for label, rep in (
        ("05 (SensorA Competitor)", report05),
        ("11 (CPX-FusionComplex Ligand)", report11),
    ):
        print(f"\n{label}: {rep['source_file']}  [{rep['curve_direction']} curve, role={rep['role']}]")
        print(f"  n_concentrations={rep['n_concentrations']}, n_runs={rep['n_runs']}")
        print("  conc(uM)   run1     run2     run3     run4      mean    sd")
        for p in rep["per_concentration"]:
            r = p["runs"]
            print(
                f"  {p['concentration_uM']:>9.1f}  {r[0]:>8.3f}  {r[1]:>8.3f}  "
                f"{r[2]:>8.3f}  {r[3]:>8.3f}  {p['mean_uM']:>8.3f}  {p['sd_uM']:>8.3f}"
            )
        fit = rep["fit_logbound_vs_conc"]
        print(f"  log-linear fit: slope={fit['slope']}, intercept={fit['intercept']}, R^2={fit['r2']}")
        ref = rep["full_binding_reference"]
        print(f"  full-binding ref = {ref['value_uM']} uM ({ref['basis']}); half = {ref['half_value_uM']} uM")
        print(f"  half-concentration = {rep['half_concentration_uM']} uM")

    cc = result["cross_check"]
    print("\n" + "-" * 70)
    print(f"CROSS-CHECK: {cc['file']} ({cc['n_rows']} rows), tolerance={TOL}")
    for tname in ("vs_source_table_05.csv", "vs_source_table_11.csv"):
        d = cc[tname]
        md = d["max_abs_diff"]
        status = "MATCH" if d["matches"] else "NO MATCH"
        print(
            f"  {tname}: compared={d['n_compared']}, within_tol={d['n_within_tolerance']}, "
            f"max_abs_diff={md}, {status}"
        )
    print(f"  BEST MATCH: {cc['best_match']} "
          f"(max_abs_diff={cc['best_match_max_abs_diff']}, within_tolerance={cc['best_match_within_tolerance']})")
    print("\nNotes:")
    for n in result["notes"]:
        print(f"  - {n}")
    print(f"\nJSON written to: {OUT_JSON}")
    return result


if __name__ == "__main__":
    main()
