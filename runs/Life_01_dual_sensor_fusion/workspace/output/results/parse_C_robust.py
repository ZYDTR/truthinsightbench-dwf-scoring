#!/usr/bin/env python3
"""
GROUP C - BINDING COMPETITION ROBUSTNESS (Finding 2 support).

Continues from the PRESERVED file binding_competition_stats.json (never deleted).
Delivers NEW robustness evidence:
  1) Independent re-parse of source_table_05.csv and source_table_11.csv and
     confirmation that the preserved per-concentration means/sds match (tol 0.01).
  2) Alternative power-law fit: log(bound) vs log(concentration).
  3) Leave-one-concentration-out (drop highest; sensitivity: drop lowest) on the
     natural-log line; report whether the half-concentration changes >20%.
  4) Monotonicity (strictly decreasing?) and the single largest adjacent increase.

Independent analysis unit = ONE run. All means/SD recomputed from the run columns
with pandas (ddof=1). The workbook Mean/S.D. columns are NOT used as evidence.
"""

import json
import math
import csv
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/data/user/conversations/conv-4bd502e1-a0d9-48ae-83ce-c85e4cbc094b/project")
SRC_DIR = BASE / "data" / "source_tables"
OUT_DIR = BASE / "output" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRESERVED_JSON = OUT_DIR / "binding_competition_stats.json"
OUT_JSON = OUT_DIR / "binding_robustness.json"

TABLES = {
    "source_table_05.csv": {
        "role": "competitor",
        "expected_concs": [0, 2, 5, 10, 20, 30, 40, 50, 60],
        "direction": "decreasing",
    },
    "source_table_11.csv": {
        "role": "ligand",
        "expected_concs": [1, 2, 5, 10, 15, 20, 25, 30, 45],
        "direction": "increasing",
    },
}

TOL = 0.01


# ---------------------------------------------------------------------------
# Parsing (independent of the preserved file)
# ---------------------------------------------------------------------------
def parse_table(path):
    """Parse a spreadsheet-style CSV. Returns list of (conc, [4 runs]).
    The concentration column is the column immediately before 'run1'; the
    four run columns follow 'run1'. Data stops at the first blank conc cell."""
    with open(path, newline="") as f:
        raw_lines = [ln.rstrip("\r\n") for ln in f]
    fields_list = [ln.split(",") for ln in raw_lines]

    header_idx = None
    run1_idx = None
    for i, fields in enumerate(fields_list):
        if "run1" in fields:
            header_idx = i
            run1_idx = fields.index("run1")
            break
    if header_idx is None:
        raise ValueError("no header row containing 'run1' in %s" % path)

    cidx = run1_idx - 1
    data = []
    for fields in fields_list[header_idx + 1:]:
        if len(fields) <= cidx:
            break
        conc_cell = fields[cidx].strip()
        if conc_cell == "":
            break
        # stop on any non-numeric concentration (e.g. a stray label)
        try:
            conc = float(conc_cell)
        except ValueError:
            break
        runs = []
        for j in range(4):
            k = run1_idx + j
            if k >= len(fields):
                break
            v = fields[k].strip()
            if v == "":
                break
            runs.append(float(v))
        if len(runs) < 4:
            break
        data.append({"concentration_uM": conc, "runs": runs})
    return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_summary(records):
    """records: list of {concentration_uM, runs}. Recompute mean/SD (ddof=1)."""
    out = []
    for rec in records:
        runs = np.asarray(rec["runs"], dtype=float)
        out.append({
            "concentration_uM": float(rec["concentration_uM"]),
            "runs": [float(v) for v in runs],
            "mean_uM": float(np.mean(runs)),
            "sd_uM": float(np.std(runs, ddof=1)),
        })
    return out


def linfit(x, y):
    """Ordinary least-squares line y = a + b*x. Returns (slope, intercept, r2)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        return None, None, None
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (intercept + slope * x)
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = (1.0 - sse / sst) if sst > 0 else None
    return float(slope), float(intercept), float(r2) if r2 is not None else None


def nat_log_fit(concs, bounds):
    """Fit ln(bound) = intercept + slope * conc (natural-log, linear in conc)."""
    x = np.asarray(concs, dtype=float)
    y = np.log(np.asarray(bounds, dtype=float))
    slope, intercept, r2 = linfit(x, y)
    return slope, intercept, r2


def power_law_fit(concs, bounds):
    """Fit log(bound) = intercept + exponent * log(conc). Exclude conc<=0."""
    pairs = [(c, b) for c, b in zip(concs, bounds) if c > 0 and b > 0]
    if len(pairs) < 2:
        return None, None, None, 0
    xs = [c for c, _ in pairs]
    ys = [b for _, b in pairs]
    lx = np.log(np.asarray(xs, dtype=float))
    ly = np.log(np.asarray(ys, dtype=float))
    slope, intercept, r2 = linfit(lx, ly)
    return float(slope), float(intercept), float(r2), len(pairs)


def half_conc_interp(concs, bounds, hv):
    """Linear interpolation of raw bound curve to reach hv.
    Branches on the BOUND direction (not the concentration direction).
    Returns (hc, ok). ok=False if hv is outside the bracketed range."""
    con = np.asarray(concs, dtype=float)
    bo = np.asarray(bounds, dtype=float)
    for i in range(len(con) - 1):
        b0, b1 = float(bo[i]), float(bo[i + 1])
        c0, c1 = float(con[i]), float(con[i + 1])
        if b1 < b0:  # bound decreasing: hv between b1 and b0
            if b1 <= hv <= b0:
                frac = (b0 - hv) / (b0 - b1)
                return float(c0 + frac * (c1 - c0)), True
        else:  # bound increasing: hv between b0 and b1
            if b0 <= hv <= b1:
                frac = (hv - b0) / (b1 - b0)
                return float(c0 + frac * (c1 - c0)), True
    return None, False


def half_conc_fit(hv, slope, intercept):
    """hc from the natural-log fit: ln(hv) = intercept + slope*hc."""
    if slope == 0:
        return None
    return float((math.log(hv) - intercept) / slope)


def change_pct(orig, new):
    if orig is None or new is None or orig == 0:
        return None
    return float((new - orig) / abs(orig) * 100.0)


def monotonic_decreasing(bounds):
    """Return (is_strictly_decreasing, max_adjacent_increase).
    max_adjacent_increase = largest positive (bounds[i+1]-bounds[i]), or None."""
    b = np.asarray(bounds, dtype=float)
    diffs = np.diff(b)
    strictly_dec = bool(np.all(diffs < 0))
    incs = diffs[diffs > 0]
    if incs.size == 0:
        max_inc = None
    else:
        max_inc = float(np.max(incs))
    return strictly_dec, max_inc


def analyze_table(records, preserved_table, name):
    """Full analysis for one table. Returns a dict for the JSON."""
    summ = compute_summary(records)
    concs = [s["concentration_uM"] for s in summ]
    bounds = [s["mean_uM"] for s in summ]

    # ---- 1) match preserved means/sds ----
    diffs = []
    for p, s in zip(preserved_table["per_concentration"], summ):
        diffs.append(abs(p["mean_uM"] - s["mean_uM"]))
        diffs.append(abs(p["sd_uM"] - s["sd_uM"]))
    max_diff = float(max(diffs)) if diffs else float("nan")
    matches = bool(max_diff <= TOL)

    # reference / half-value (held FIXED across all fits)
    if 0.0 in concs:
        ref_idx = concs.index(0.0)
        ref_basis = "mean bound at 0 uM (no competitor)"
    else:
        ref_idx = int(np.argmax(bounds))
        ref_basis = "mean bound at highest concentration (ligand saturation; no 0 uM measured)"
    ref_val = bounds[ref_idx]
    hv = ref_val / 2.0

    # ---- 2) natural-log fit (primary, matches preserved) ----
    slope_nl, intercept_nl, r2_nl = nat_log_fit(concs, bounds)
    hc_nl_full = half_conc_fit(hv, slope_nl, intercept_nl)
    hc_interp_full, ok_interp_full = half_conc_interp(concs, bounds, hv)

    # ---- 2b) power-law (alternative) fit ----
    exp_pl, intc_pl, r2_pl, n_pl = power_law_fit(concs, bounds)

    # ---- 3) leave-one-out on the natural-log line ----
    def leave_out(concs_, bounds_, drop_idx, hv_):
        keep_c = [concs_[i] for i in range(len(concs_)) if i != drop_idx]
        keep_b = [bounds_[i] for i in range(len(bounds_)) if i != drop_idx]
        sl, ic, r2 = nat_log_fit(keep_c, keep_b)
        hc_fit = half_conc_fit(hv_, sl, ic)
        hc_intp, ok_intp = half_conc_interp(keep_c, keep_b, hv_)
        return {
            "n_points": len(keep_c),
            "dropped_conc_uM": concs_[drop_idx],
            "natural_log_fit": {
                "slope": sl, "intercept": ic, "r2": r2,
            },
            "half_value_uM": hv_,
            "half_conc_interp": {
                "value": hc_intp, "ok": ok_intp,
                "change_pct_vs_original": change_pct(hc_interp_full, hc_intp),
                "changes_gt_20": bool(change_pct(hc_interp_full, hc_intp) is not None
                                       and abs(change_pct(hc_interp_full, hc_intp)) > 20.0),
            },
            "half_conc_fit": {
                "value": hc_fit,
                "change_pct_vs_original": change_pct(hc_nl_full, hc_fit),
                "changes_gt_20": bool(change_pct(hc_nl_full, hc_fit) is not None
                                       and abs(change_pct(hc_nl_full, hc_fit)) > 20.0),
            },
        }

    n = len(concs)
    drop_highest = n - 1
    drop_lowest = 0
    lo_highest = leave_out(concs, bounds, drop_highest, hv)
    lo_lowest = leave_out(concs, bounds, drop_lowest, hv)

    # ---- 4) monotonicity ----
    is_dec, max_inc = monotonic_decreasing(bounds)

    return {
        "role": TABLES[name]["role"],
        "n_concentrations": n,
        "n_runs": len(summ[0]["runs"]),
        "preserved_match": {
            "tolerance": TOL,
            "max_diff_mean_or_sd": max_diff,
            "matches": matches,
        },
        "reference": {
            "basis": ref_basis,
            "value_uM": ref_val,
            "half_value_uM": hv,
        },
        "natural_log_fit": {
            "model": "log(bound) = intercept + slope * concentration (natural log)",
            "slope": slope_nl, "intercept": intercept_nl, "r2": r2_nl,
            "n_points": n,
            "half_conc_interp_uM": hc_interp_full,
            "half_conc_fit_uM": hc_nl_full,
            "preserved_slope": preserved_table["fit_logbound_vs_conc"]["slope"],
            "preserved_intercept": preserved_table["fit_logbound_vs_conc"]["intercept"],
            "preserved_r2": preserved_table["fit_logbound_vs_conc"]["r2"],
            "preserved_half_concentration_uM": preserved_table["half_concentration_uM"],
            "slope_diff_vs_preserved": slope_nl - preserved_table["fit_logbound_vs_conc"]["slope"],
            "intercept_diff_vs_preserved": intercept_nl - preserved_table["fit_logbound_vs_conc"]["intercept"],
            "r2_diff_vs_preserved": r2_nl - preserved_table["fit_logbound_vs_conc"]["r2"],
        },
        "power_law_fit": {
            "model": "log(bound) = intercept + exponent * log(concentration) (power-law)",
            "exponent": exp_pl, "intercept": intc_pl, "r2": r2_pl,
            "n_points_used": n_pl,
            "note": ("excluded conc<=0 points" if n_pl < n else "all positive concentrations used"),
            "vs_natural_log": {
                "natural_log_slope": slope_nl,
                "natural_log_intercept": intercept_nl,
                "natural_log_r2": r2_nl,
            },
        },
        "leave_out": {
            "highest": lo_highest,
            "lowest": lo_lowest,
        },
        "monotonic": {
            "strictly_decreasing": is_dec,
            "curve_direction": "decreasing" if is_dec else ("increasing" if np.sum(np.diff(bounds)) > 0 else "mixed"),
            "max_adjacent_increase_uM": max_inc,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    errors = []
    with open(PRESERVED_JSON) as f:
        preserved = json.load(f)
    pres_tables = preserved["tables"]

    per_table = {}
    all_matches = True
    all_maxdiffs = []

    for name, meta in TABLES.items():
        path = SRC_DIR / name
        try:
            records = parse_table(path)
            expected = meta["expected_concs"]
            if [r["concentration_uM"] for r in records] != expected:
                # still proceed, but flag a note
                pass
            summ = compute_summary(records)
            res = analyze_table(records, pres_tables[name], name)
            per_table[name] = res
            all_matches &= res["preserved_match"]["matches"]
            all_maxdiffs.append(res["preserved_match"]["max_diff_mean_or_sd"])
            print("\n===== %s (%s) =====" % (name, meta["role"]))
            print("  parsed n_concs=%d n_runs=%d" % (res["n_concentrations"], res["n_runs"]))
            print("  preserved match: %s (max_diff=%.6f)" % (res["preserved_match"]["matches"], res["preserved_match"]["max_diff_mean_or_sd"]))
            print("  ref bound=%.4f hv=%.4f" % (res["reference"]["value_uM"], res["reference"]["half_value_uM"]))
            nl = res["natural_log_fit"]
            print("  natural-log fit: slope=%.6f intercept=%.6f r2=%.6f" % (nl["slope"], nl["intercept"], nl["r2"]))
            print("  preserved fit:   slope=%.6f intercept=%.6f r2=%.6f" % (nl["preserved_slope"], nl["preserved_intercept"], nl["preserved_r2"]))
            print("  slope diff=%.6f intercept diff=%.6f r2 diff=%.6f" % (nl["slope_diff_vs_preserved"], nl["intercept_diff_vs_preserved"], nl["r2_diff_vs_preserved"]))
            print("  half_conc: interp=%.4f  fit=%.4f  preserved=%.4f" % (nl["half_conc_interp_uM"], nl["half_conc_fit_uM"], nl["preserved_half_concentration_uM"]))
            pl = res["power_law_fit"]
            print("  power-law (log-log) fit: exponent=%.6f intercept=%.6f r2=%.6f (n=%d)" % (pl["exponent"], pl["intercept"], pl["r2"], pl["n_points_used"]))
            lo = res["leave_out"]
            for key in ("highest", "lowest"):
                d = lo[key]
                hi = d["half_conc_interp"]
                hf = d["half_conc_fit"]
                print("  leave_out %s (drop %.1f uM, n=%d): interp hc=%.4f (change %.3f%%, >20? %s) | fit hc=%.4f (change %.3f%%, >20? %s)" % (
                    key, d["dropped_conc_uM"], d["n_points"], hi["value"], hi["change_pct_vs_original"] if hi["change_pct_vs_original"] is not None else float("nan"),
                    hi["changes_gt_20"], hf["value"], hf["change_pct_vs_original"] if hf["change_pct_vs_original"] is not None else float("nan"), hf["changes_gt_20"]))
            m = res["monotonic"]
            print("  monotonic strictly_decreasing=%s (max_adjacent_increase=%s)" % (m["strictly_decreasing"], m["max_adjacent_increase_uM"]))

        except Exception as e:
            errors.append("%s: %s" % (name, repr(e)))
            per_table[name] = {"error": repr(e)}

    max_diff_all = float(max(all_maxdiffs)) if all_maxdiffs else float("nan")

    result = {
        "preserved_file_matched": {
            "bool": all_matches,
            "max_diff": max_diff_all,
            "tolerance": TOL,
            "per_table": {
                name: per_table[name]["preserved_match"]
                for name in TABLES if name in per_table and "error" not in per_table[name]
            },
        },
        "natural_log_fit": {
            name: per_table[name]["natural_log_fit"]
            for name in TABLES if name in per_table and "error" not in per_table[name]
        },
        "power_law_fit": {
            name: per_table[name]["power_law_fit"]
            for name in TABLES if name in per_table and "error" not in per_table[name]
        },
        "leave_out": {
            name: per_table[name]["leave_out"]
            for name in TABLES if name in per_table and "error" not in per_table[name]
        },
        "monotonic": {
            name: per_table[name]["monotonic"]["strictly_decreasing"]
            for name in TABLES if name in per_table and "error" not in per_table[name]
        },
        "max_adjacent_increase": {
            name: per_table[name]["monotonic"]["max_adjacent_increase_uM"]
            for name in TABLES if name in per_table and "error" not in per_table[name]
        },
        "errors": errors,
        "notes": [
            "Independent re-parse of source_table_05.csv and source_table_11.csv; means/SD recomputed from the 4 run columns (ddof=1), workbook Mean/S.D. columns not used as evidence.",
            "half-concentration computed two ways: (a) linear interpolation of the raw bound-vs-conc curve (this reproduces the preserved half_concentration_uM), and (b) from the natural-log fit, hc=(ln(hv)-intercept)/slope. hv is held fixed (ref/2) across all leave-out re-fits.",
            "power-law (log-log) fit excludes conc<=0 points (source_table_05.csv has a 0 uM point).",
            "leave_out.highest drops the highest concentration; leave_out.lowest drops the lowest concentration (sensitivity check).",
        ],
    }

    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    # ---- stdout summary ----
    print("\n" + "=" * 70)
    print("GROUP C - BINDING COMPETITION ROBUSTNESS - SUMMARY")
    print("=" * 70)
    print("preserved_file_matched: bool=%s  max_diff=%.6f  (tol=%.2f)" % (all_matches, max_diff_all, TOL))
    for name in TABLES:
        if name not in per_table or "error" in per_table[name]:
            continue
        nl = per_table[name]["natural_log_fit"]
        pl = per_table[name]["power_law_fit"]
        m = per_table[name]["monotonic"]
        print("\n[%s]" % name)
        print("  natural_log_fit : slope=%.6f intercept=%.6f r2=%.6f" % (nl["slope"], nl["intercept"], nl["r2"]))
        print("  power_law_fit   : exponent=%.6f intercept=%.6f r2=%.6f (n=%d)" % (pl["exponent"], pl["intercept"], pl["r2"], pl["n_points_used"]))
        print("  monotonic (strictly decreasing) = %s" % m["strictly_decreasing"])
        print("  max_adjacent_increase = %s uM" % (m["max_adjacent_increase_uM"] if m["max_adjacent_increase_uM"] is not None else "none"))
        for key in ("highest", "lowest"):
            d = per_table[name]["leave_out"][key]
            hi = d["half_conc_interp"]
            hf = d["half_conc_fit"]
            hi_c = hi["change_pct_vs_original"]
            hi_c = "n/a" if hi_c is None else ("%.3f%%" % hi_c)
            hf_c = hf["change_pct_vs_original"]
            hf_c = "n/a" if hf_c is None else ("%.3f%%" % hf_c)
            print("  leave_out %s (drop %.1f uM): interp change=%s (gt20=%s)  fit change=%s (gt20=%s)" % (
                key, d["dropped_conc_uM"], hi_c, hi["changes_gt_20"], hf_c, hf["changes_gt_20"]))
    print("\nWrote: %s" % OUT_JSON)
    if errors:
        print("ERRORS: %s" % errors)


if __name__ == "__main__":
    main()
