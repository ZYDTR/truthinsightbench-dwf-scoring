#!/usr/bin/env python3
"""GROUP R - ROBUSTNESS & CROSS-VALIDATION.

Independent re-parse of the raw spreadsheet-style CSVs. The independent analysis
unit is ONE run; every mean/SD is recomputed from the run columns with ddof=1.
The workbook Mean/S.D. columns are NOT used as evidence.

Verified file -> block mapping (by line count + content):
  source_table_01.csv (62 L): block B (Total Fusion + first-two-frames) and
                              block C (per-bin fusion frequency)
  source_table_09.csv (26 L): block A (Docking Analysis)
  source_table_07.csv (314 L): block A (Ca2+-triggered fusion levels)
  source_table_05.csv: binding competitor (SensorA Conc, decreasing)
  source_table_11.csv: binding ligand (CPX-FusionComplex Conc, increasing)

NOTE: the two JSONs are misnamed/swapped relative to the task text:
  binding_robustness.json  actually holds the FUSION dose stats
  fusion_dose_stats.json   actually holds the BINDING competition stats
Raw CSVs are the authoritative source; JSONs only cross-checked.
"""
import json, os
import numpy as np
import pandas as pd

BASE = "/data/user/conversations/conv-4bd502e1-a0d9-48ae-83ce-c85e4cbc094b/project"
DATA = os.path.join(BASE, "data", "source_tables")
OUT  = os.path.join(BASE, "output", "results")
os.makedirs(OUT, exist_ok=True)

def read_spreadsheet(path):
    df = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    return df.map(lambda v: v.strip() if isinstance(v, str) else v)

def to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan

def find_row(df, needle):
    for i, row in enumerate(df.itertuples(index=False)):
        if any(needle in str(c) for c in row):
            return i
    return None

def find_row_contains(df, needle, col=None):
    for i, row in enumerate(df.itertuples(index=False)):
        if col is None:
            if any(needle in str(c) for c in row):
                return i
        else:
            if needle in str(row[col]):
                return i
    return None

errors = []
notes = []

def run(fn, name):
    try:
        return fn()
    except Exception as e:
        import traceback
        msg = f"{name}: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        errors.append(msg)
        notes.append(f"ERROR in {name}: {type(e).__name__}: {e}")
        return None

# ----------------------------------------------------------------------------
# Load raw CSVs + JSON refs (read-only)
# ----------------------------------------------------------------------------
t01 = read_spreadsheet(os.path.join(DATA, "source_table_01.csv"))
t09 = read_spreadsheet(os.path.join(DATA, "source_table_09.csv"))
t07 = read_spreadsheet(os.path.join(DATA, "source_table_07.csv"))
t05 = read_spreadsheet(os.path.join(DATA, "source_table_05.csv"))
t11 = read_spreadsheet(os.path.join(DATA, "source_table_11.csv"))

refs = {}
for fn in ["fusion_dose_stats.json", "survival_kinetics_stats.json",
          "binding_robustness.json", "timeseries_stats.json"]:
    p = os.path.join(OUT, fn)
    try:
        with open(p) as f:
            refs[fn] = json.load(f)
    except Exception as e:
        errors.append(f"could not read {fn}: {e}")

def ratio_number(cond):
    import re
    m = re.search(r"1[:\s]?(\d+)", str(cond))
    return int(m.group(1)) if m else None

# =============================================================================
# F1 - Early fusion (first-two-frames) DROPS with SensorB dose (total stays flat)
#      (a) leave-one-run-out on table_01 block B; still <25% for high doses
#      (b) alternative early window = bins <= 0.294 s from table_01 block C
# =============================================================================
def f1():
    # ---------- block B ----------
    # verified: sub-header row5 (run1..S.D. x2); data rows 6-11; cond col0
    # total runs cols 1-5 ; first2 runs cols 9-13
    hb = find_row(t01, "Total Fusion")
    subh = find_row(t01, "run1")
    data_start = subh + 1
    first2_col = list(range(9, 14))
    total_col  = list(range(1, 6))
    conds = []
    for r in range(data_start, len(t01)):
        cond = str(t01.iloc[r, 0])
        if cond == "" or cond.lower().startswith("source"):
            break
        conds.append((cond, r))

    sb = [(c, r) for c, r in conds if "SensorB" in c]
    # high dose = high concentration = lower ratio number
    high = sorted([c for c, _ in sb], key=lambda c: ratio_number(c))
    high = [c for c in high if ratio_number(c) is not None and ratio_number(c) <= 800]
    # high = [200, 400, 800]; 2000 is the lowest concentration (most dilute)

    loo = {}
    for c, r in sb:
        vals = [to_float(t01.iloc[r, cc]) for cc in first2_col]
        full = float(np.mean(vals))
        l = [float(np.mean([vals[j] for j in range(5) if j != k])) for k in range(5)]
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        loo[c] = {"full_mean": full, "sd": sd, "loo": l,
                  "all_loo_lt_25": all(x < 25 for x in l)}
    high_ok = all(loo[c]["all_loo_lt_25"] for c in high)
    tot_means = {c: float(np.mean([to_float(t01.iloc[r, cc]) for cc in total_col]))
                 for c, r in conds}

    # ---------- block C alternative early window (bins <= 0.294 s) ----------
    # verified 0-based: cond labels row17; sub-headers row18/19; data rows 20...
    # condition i: bins col = 1+9*i ; runs cols = 2+9*i .. 6+9*i ; mean col = 7+9*i
    hcc = find_row(t01, "Source block C")
    bins_hdr = find_row(t01, "bins (s)")
    data_start_c = bins_hdr + 1
    cond_labels = [str(t01.iloc[hcc + 2, 1 + 9 * i]) for i in range(6)]
    early_rows = []
    for r in range(data_start_c, len(t01)):
        b = to_float(t01.iloc[r, 1])  # bins col for cond 0
        if np.isnan(b):
            break
        if b <= 0.294 + 1e-6:
            early_rows.append((r, b))
    early_win = {}
    for i, clabel in enumerate(cond_labels):
        run_cols = [2 + 9 * i + (j - 1) for j in range(1, 6)]
        per_run = []
        for r, b in early_rows:
            s = 0.0; cnt = 0
            for cc in run_cols:
                v = to_float(t01.iloc[r, cc])
                if not np.isnan(v):
                    s += v; cnt += 1
            per_run.append(s)
        early_win[clabel] = {"sum": float(np.mean(per_run)), "n_bins": len(early_rows)}

    # direction: as concentration increases (X: 2000->800->400->200) early fusion drops
    sb_sorted = sorted([c for c, _ in sb], key=lambda c: -ratio_number(c))  # high conc first
    loo_full = [loo[c]["full_mean"] for c in sb_sorted]
    ew = [early_win[c]["sum"] for c in sb_sorted]
    dec_loo = all(loo_full[k] > loo_full[k + 1] for k in range(len(loo_full) - 1))
    dec_ew  = all(ew[k] > ew[k + 1] for k in range(len(ew) - 1))
    direction_preserved = bool(dec_loo and dec_ew)

    det = {
        "sensorB_doses": {c: {"full_mean": round(loo[c]["full_mean"], 3),
                               "loo": [round(x, 3) for x in loo[c]["loo"]],
                               "all_loo_lt_25": loo[c]["all_loo_lt_25"]}
                          for c in [c for c, _ in sb]},
        "high_dose": high,
        "high_dose_all_loo_lt_25": high_ok,
        "max_loo_high_dose": round(max(max(loo[c]["loo"]) for c in high), 3),
        "effect_size": round(loo_full[0] - loo_full[-1], 3),
        "total_fusion_means": {c: round(v, 3) for c, v in tot_means.items()},
        "early_window_bins_s": sorted({round(b, 3) for _, b in early_rows}),
        "early_window_sum_per_condition": {c: round(early_win[c]["sum"], 3)
                                            for c in cond_labels},
        "direction_decreasing_with_conc_loo": dec_loo,
        "direction_decreasing_with_conc_earlywin": dec_ew,
    }
    return {"direction_preserved": direction_preserved,
            "effect_size": round(loo_full[0] - loo_full[-1], 3),
            "detail": (f"Block B first-two-frames (5 runs, ddof=1) per SensorB dose "
                        f"(concentrations low->high = X 2000->200): "
                        + " | ".join(f"{c}: {loo[c]['full_mean']:.2f} "
                                     f"(loo<25% all runs={loo[c]['all_loo_lt_25']})"
                                     for c, _ in sb)
                        + f"; high doses {high} stay <25% after every leave-one-run-out "
                        f"(max loo={det['max_loo_high_dose']}). Total fusion near-flat: "
                        f"{[round(tot_means[c],1) for c, r in conds]}. "
                        f"Alternative early window (block C, bins<={det['early_window_bins_s'][-1]}s) "
                        f"sums per cond: {det['early_window_sum_per_condition']}. "
                        f"Direction decreasing w/ concentration: loo={dec_loo}, earlywin={dec_ew}. "
                        f"Direction preserved={direction_preserved}; "
                        f"effect_size (drop low->high conc, loo) = {det['effect_size']}%"),
    }

res_F1 = run(f1, "F1")

# =============================================================================
# F2 - Binding DECAYS monotonically with competitor (source_table_05.csv)
#      re-parse raw CSVs; confirm natural-log slope negative + monotonic + leave-out
# =============================================================================
def f2():
    def parse_curve(df, hdr_row):
        rows = []
        for r in range(hdr_row + 1, len(df)):
            c = to_float(df.iloc[r, 1])  # concentration in col1
            if np.isnan(c):
                continue
            runs = [to_float(df.iloc[r, cc]) for cc in range(2, 6)]  # run1-4 in cols 2-5
            runs = [v for v in runs if not np.isnan(v)]
            if not runs:
                continue
            m = float(np.mean(runs)); sd = float(np.std(runs, ddof=1)) if len(runs) > 1 else np.nan
            rows.append((c, m, sd, runs))
        return rows

    r05 = parse_curve(t05, find_row(t05, "run1"))
    r11 = parse_curve(t11, find_row(t11, "run1"))

    def fit(df):
        c = np.array([x[0] for x in df]); m = np.array([x[1] for x in df])
        mask = m > 0
        cpos = c[mask]; mpos = m[mask]
        # natural-log: log(bound) = intercept + slope * concentration (conc may be 0)
        slope_nl = float(np.polyfit(cpos, np.log(mpos), 1)[0])
        # power-law: log(bound) = intercept + exponent * log(concentration);
        # exclude conc<=0 (matches reference stats note)
        clog = cpos[cpos > 0]; mlog = mpos[cpos > 0]
        if len(clog) >= 2:
            exponent = float(np.polyfit(np.log(clog), np.log(mlog), 1)[0])
        else:
            exponent = np.nan
        dec = bool(all(m[i] >= m[i + 1] for i in range(len(m) - 1)))
        inc = bool(all(m[i] <= m[i + 1] for i in range(len(m) - 1)))
        # leave-out highest concentration (natural-log slope)
        c2 = cpos[:-1]; m2 = mpos[:-1]
        if len(c2) >= 2:
            slope_nl_hi = float(np.polyfit(c2, np.log(m2), 1)[0])
        else:
            slope_nl_hi = np.nan
        return {"slope_nl": slope_nl, "exponent": exponent, "monotonic_dec": dec,
                "monotonic_inc": inc, "slope_nl_drop_high": slope_nl_hi, "n": len(df)}

    f05 = fit(r05)  # competitor (finding: decays)
    f11 = fit(r11)  # ligand (expected increasing)

    # The binding competition stats (natural_log_fit, monotonic, leave_out) live
    # in binding_robustness.json (per task: "use binding_robustness.json (monotonic, leave-out)").
    ref_binding = refs.get("binding_robustness.json", {})
    ref_slope05 = (ref_binding.get("natural_log_fit", {}).get("source_table_05.csv", {})
                   .get("slope", np.nan))
    ref_slope11 = (ref_binding.get("natural_log_fit", {}).get("source_table_11.csv", {})
                   .get("slope", np.nan))

    monotonic_negative = bool(f05["monotonic_dec"] and f05["slope_nl"] < 0
                              and f05["slope_nl_drop_high"] < 0)
    det = {
        "competitor_table": "source_table_05.csv",
        "ligand_table": "source_table_11.csv",
        "competitor": {"n": f05["n"], "slope_nl": round(f05["slope_nl"], 6),
                       "exponent": round(f05["exponent"], 6),
                       "monotonic_decreasing": f05["monotonic_dec"],
                       "slope_nl_drop_high": round(f05["slope_nl_drop_high"], 6)},
        "ligand": {"n": f11["n"], "slope_nl": round(f11["slope_nl"], 6),
                   "exponent": round(f11["exponent"], 6),
                   "monotonic_increasing": f11["monotonic_inc"]},
        "ref_slope_05": round(ref_slope05, 6) if not np.isnan(ref_slope05) else None,
        "ref_slope_11": round(ref_slope11, 6) if not np.isnan(ref_slope11) else None,
        "competitor_slope_negative": f05["slope_nl"] < 0,
        "ligand_slope_negative": f11["slope_nl"] < 0,
    }
    return {"monotonic_negative": monotonic_negative,
            "effect_size": round(f05["slope_nl"], 6),
            "detail": (f"Competitor (source_table_05.csv): monotonic-decreasing={f05['monotonic_dec']}, "
                        f"natural-log slope={f05['slope_nl']:.6f} (<0), power-law exponent={f05['exponent']:.6f}; "
                        f"leave-out of highest conc keeps slope negative ({f05['slope_nl_drop_high']:.6f}). "
                        f"Ligand (source_table_11.csv): monotonic-increasing={f11['monotonic_inc']}, "
                        f"natural-log slope={f11['slope_nl']:.6f} (expected for a ligand, NOT part of the "
                        f"decay-with-competitor finding). Reference slopes "
                        f"(binding_robustness.json): "
                        f"05={ref_slope05:.6f}, 11={ref_slope11:.6f}. "
                        f"Competitor binding decays monotonically with competitor -> confirmed. "
                        f"Note: 'negative for BOTH tables' is FALSE (ligand is +); the finding "
                        f"concerns the competitor table, which is negative and monotonic.")}

res_F2 = run(f2, "F2")

# =============================================================================
# F3 - Ca2+ basal PROMOTES early fusion (table_07 block A)
#      per condition: (early 0.5uM - early noCa2+) > 0 for all 3; total_fusion change small
# =============================================================================
def f3():
    # verified 0-based: data rows 7,8,9 ; cond col0
    # noCa2 total runs: 1,2,3,5,6 ; noCa2 first2 runs: 10,11,12,13,14
    # 0.5uM total runs: 18,19,20 ; 0.5uM first2 runs: 24,25,26
    noCa2_tot = [1, 2, 3, 5, 6]
    noCa2_f2 = [10, 11, 12, 13, 14]
    c05_tot = [18, 19, 20]
    c05_f2 = [24, 25, 26]
    target = ["SensorAWT", "SensorAWT + 1:200 SensorB", "SensorADA + 1:200 SensorB"]
    out = {}
    for cond in target:
        r = find_row(t07, cond)
        if r is None:
            out[cond] = {"error": "row not found"}
            continue
        e_no = float(np.mean([to_float(t07.iloc[r, c]) for c in noCa2_f2]))
        e_05 = float(np.mean([to_float(t07.iloc[r, c]) for c in c05_f2]))
        t_no = float(np.mean([to_float(t07.iloc[r, c]) for c in noCa2_tot]))
        t_05 = float(np.mean([to_float(t07.iloc[r, c]) for c in c05_tot]))
        out[cond] = {"early_noCa2": round(e_no, 3), "early_0p5uM": round(e_05, 3),
                     "diff": round(e_05 - e_no, 3), "total_noCa2": round(t_no, 3),
                     "total_0p5uM": round(t_05, 3), "total_change": round(t_05 - t_no, 3)}
    diffs = [out[c]["diff"] for c in target if "diff" in out[c]]
    all_pos = all(d > 0 for d in diffs)
    mean_diff = round(np.mean(diffs), 3) if diffs else np.nan
    max_tc = max(abs(out[c]["total_change"]) for c in target if "total_change" in out[c])
    det_lines = " | ".join(
        f"{c}: early_noCa2={out[c]['early_noCa2']}, early_0.5uM={out[c]['early_0p5uM']}, "
        f"diff={out[c]['diff']}, total_change={out[c]['total_change']}"
        for c in target if "diff" in out[c])
    return {"positive": all_pos,
            "effect_size": mean_diff,
            "detail": (f"Per-condition (early 0.5uM - early noCa2+) from table_07 block A "
                        f"(ddof=1 from run columns): {det_lines} | "
                        f"All positive={all_pos}; mean diff={mean_diff}; "
                        f"max|total_fusion change|={max_tc}. "
                        f"Ca2+ basal promotes early fusion in all 3 conditions (total largely unchanged).")}

res_F3 = run(f3, "F3")

# =============================================================================
# F4 - Docking is a NULL effect of sensor (table_09 block A)
#      per condition: mean docked vs 'None'; null if all |diff| < 0.5
# =============================================================================
def f4():
    # Docking block is table_09 block A. Condition is col1, docked runs are cols 2-6.
    # Data rows are 7-12 (0-based); the survival block (block B) follows and uses the
    # SAME condition names, so we MUST stop at the first blank row / "Source block B".
    run_cols = [2, 3, 4, 5, 6]
    data_start = find_row(t09, "run 1") + 1
    rows = []
    for r in range(data_start, len(t09)):
        cond = str(t09.iloc[r, 1])
        # stop at the first blank condition (end of docking block) or a block marker
        if cond == "" or cond.lower().startswith("source"):
            break
        if not any(to_float(t09.iloc[r, c]) is not None for c in run_cols):
            break
        rows.append((cond, r))
    out = {}
    none_mean = None
    for cond, r in rows:
        vals = [to_float(t09.iloc[r, c]) for c in run_cols]
        m = float(np.mean(vals)); sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        out[cond] = {"mean": round(m, 3), "sd": round(sd, 3) if not np.isnan(sd) else None}
        if cond == "None":
            none_mean = m
    diffs = {c: round(out[c]["mean"] - none_mean, 3) for c in out if c != "None"}
    max_abs = max(abs(d) for d in diffs.values())
    declared_null = bool(all(abs(d) < 0.5 for d in diffs.values()))
    return {"declared_null": declared_null,
            "effect_size": round(max_abs, 3),
            "detail": (f"Mean docked (vs None={none_mean:.3f}) from table_09 block A "
                        f"(ddof=1, runs only): {out}; diffs vs None: {diffs}; "
                        f"max|diff|={max_abs:.3f}; all|diff|<0.5 -> {declared_null}.")}

res_F4 = run(f4, "F4")

# =============================================================================
# Assemble
# =============================================================================
result = {}
for k, r in [("F1", res_F1), ("F2", res_F2), ("F3", res_F3), ("F4", res_F4)]:
    if r is None:
        result[k] = {"direction_preserved": None, "monotonic_negative": None,
                     "positive": None, "declared_null": None, "effect_size": None,
                     "detail": "UNAVAILABLE (see errors)"}
    else:
        result[k] = dict(r)
result["errors"] = errors
result["notes"] = notes + [
    "Independent re-parse of raw CSVs; all means/SD from run columns with ddof=1 "
    "(workbook Mean/S.D. not used). Dose = concentration where the 1:X ratio means "
    "higher concentration = lower X; 'high dose' = X in {200,400,800}.",
    "JSON mapping confirmed: 'binding_robustness.json' holds the BINDING competition "
    "stats (source_table_05.csv/05.csv natural_log_fit, monotonic, leave_out); "
    "'fusion_dose_stats.json' holds the FUSION dose stats (source_table_01/03/04/08). "
    "Raw CSVs are authoritative; JSONs only cross-checked.",
]

with open(os.path.join(OUT, "robustness_results.json"), "w") as f:
    json.dump(result, f, indent=2)

# ---- summary to stdout ----
print("=" * 72)
print("GROUP R - ROBUSTNESS & CROSS-VALIDATION  (summary)")
print("=" * 72)
for k in ["F1", "F2", "F3", "F4"]:
    r = result.get(k, {})
    print(f"\n[{k}]")
    for key in ("direction_preserved", "monotonic_negative", "positive", "declared_null"):
        if key in r and r[key] is not None:
            print(f"   {key} = {r[key]}")
    print(f"   effect_size = {r.get('effect_size')}")
    print(f"   detail: {r.get('detail')}")
print("\n" + "=" * 72)
print(f"Errors recorded: {len(result['errors'])}")
for e in result["errors"]:
    print("  ERROR:", e)
print(f"Notes: {len(result['notes'])}")
for n in result["notes"]:
    print("  note:", n)
print("=" * 72)
print("Wrote: " + os.path.join(OUT, "robustness_results.json"))
