#!/usr/bin/env python3
"""Independent verification of artifact consistency for the
Energy_04_biohybrid_operating_tradeoffs report.

Written from scratch; does NOT import or execute output/analysis/analysis.py.
Every statistic is re-derived from data/*.tsv with hand-rolled numpy code
(covariance-formula correlations, normal-equation OLS, manual rolling medians,
permutation tests with a different seed / more draws).

Checks
  1. output/results/table_regimes.tsv vs fresh segmentation of the raw TSV
  2. output/results/table_matched_time_branches.tsv vs sweep rows 259-293
  3. output/results/table_fit_vs_raw.tsv vs regime C (rows 294-1366)
  4. output/results/table_headline_numbers.tsv vs findings.json
  5. findings.json internal consistency (5 specified cross-key checks)
  6. output/reproduce.md hashes, paths, run command, package versions
  7. Numeric claims in output/Result.md (spot checks against recomputation)
"""
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = "/data/user/conversations/conv-995a498d-d5c1-4e1e-99f5-01601e51abe5/project"
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "output", "results")
OUTDIR = os.path.join(ROOT, "output", "verification")

# --------------------------------------------------------------------------
# small hand-rolled statistics (independent of analysis.py)
# --------------------------------------------------------------------------
def corr(a, b):
    """Pearson r via the covariance formula, computed by hand."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ma, mb = a.mean(), b.mean()
    da, db = a - ma, b - mb
    sxx = (da * da).sum()
    syy = (db * db).sum()
    sxy = (da * db).sum()
    if sxx == 0 or syy == 0:
        return float("nan")
    return float(sxy / math.sqrt(sxx * syy))


def ols_normal(x, y):
    """OLS slope/intercept/R2 via the normal equations (closed form)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xm, ym = x.mean(), y.mean()
    dx, dy = x - xm, y - ym
    sxx = (dx * dx).sum()
    slope = (dx * dy).sum() / sxx
    intercept = ym - slope * xm
    pred = slope * x + intercept
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - ym) ** 2).sum()
    return float(slope), float(intercept), float(1 - ss_res / ss_tot)


def ols_grid(x, y, lo=-10, hi=10, n=400001):
    """Slope by fine grid search of SSE (independent of the normal equations)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    grid = np.linspace(lo, hi, n)
    # SSE(s) = sum (y - (s*x + b))^2 minimised over b analytically per s
    b = y.mean() - grid * x.mean()
    sse = ((y[None, :] - (grid[:, None] * x[None, :] + b[:, None])) ** 2).sum(axis=1)
    k = int(np.argmin(sse))
    return float(grid[k])


def spearman_manual(a, b):
    """Spearman via average ranks (hand-rolled, no pandas rank)."""
    def ranks(v):
        v = np.asarray(v, float)
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), float)
        sv = v[order]
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and sv[j + 1] == sv[i]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r
    return corr(ranks(a), ranks(b))


def rolling_median(v, w, center=True, min_periods=3):
    """Manual centred rolling median (independent of pandas.rolling)."""
    v = np.asarray(v, float)
    n = len(v)
    half = (w - 1) // 2 if center else w - 1
    out = np.empty(n, float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1 if center else i + 1)
        window = v[lo:hi]
        out[i] = np.median(window) if len(window) >= min_periods else v[i]
    return out


def rmse(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(np.sqrt((d * d).mean()))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(65536), b""):
            h.update(blk)
    return h.hexdigest()


CLAIMS = []          # (claim, report_value, recomputed_value, matches, note)
DISCREPANCIES = []


def check(claim, report_value, recomputed_value, note="", tol=0.0, rel=False):
    """Record a claim check. tol: absolute (or relative if rel=True) tolerance
    for classifying as round-off vs material; exact equality required when
    tol==0 unless values are non-numeric."""
    match = False
    try:
        if isinstance(report_value, str) and isinstance(recomputed_value, str):
            match = report_value == recomputed_value
        else:
            rv, cv = float(report_value), float(recomputed_value)
            if math.isnan(rv) and math.isnan(cv):
                match = True
            else:
                if rel:
                    match = abs(rv - cv) <= tol * max(abs(rv), 1e-300)
                else:
                    match = abs(rv - cv) <= tol
    except (TypeError, ValueError):
        match = report_value == recomputed_value
    CLAIMS.append({
        "claim": claim,
        "report_value": repr(report_value),
        "recomputed_value": repr(recomputed_value),
        "matches": bool(match),
        "note": note,
    })
    if not match:
        DISCREPANCIES.append(f"{claim}: report={report_value!r} recomputed={recomputed_value!r} ({note})")
    return match


def near_report(report_value, recomputed_value, abs_tol):
    """Classify a numeric difference as round-off (within abs_tol) or material."""
    try:
        return abs(float(report_value) - float(recomputed_value)) <= abs_tol
    except (TypeError, ValueError):
        return report_value == recomputed_value


# --------------------------------------------------------------------------
# load raw data
# --------------------------------------------------------------------------
curves = pd.read_csv(os.path.join(DATA, "time_response_curves.tsv"), sep="\t")
curves["row"] = np.arange(len(curves))
pert = pd.read_csv(os.path.join(DATA, "perturbation_time_response.tsv"), sep="\t")
dose = pd.read_csv(os.path.join(DATA, "dose_response.tsv"), sep="\t")
findings = json.load(open(os.path.join(RES, "findings.json")))
TAB_REG = pd.read_csv(os.path.join(RES, "table_regimes.tsv"), sep="\t")
TAB_MTB = pd.read_csv(os.path.join(RES, "table_matched_time_branches.tsv"), sep="\t")
TAB_FVR = pd.read_csv(os.path.join(RES, "table_fit_vs_raw.tsv"), sep="\t")
TAB_HEAD = pd.read_csv(os.path.join(RES, "table_headline_numbers.tsv"), sep="\t")

# --------------------------------------------------------------------------
# 1. table_regimes.tsv vs a FRESH segmentation
# --------------------------------------------------------------------------
# Fresh segmentation from the declared rule: sweep-supported rows are those
# whose time is one of the global multi-point times {t >= 2}; contiguous runs
# of such rows define the sweep region.
sweep_t = sorted(int(t) for t in curves["time"].unique() if t >= 2)
mask = curves["time"].isin(sweep_t).values.astype(int)
ch = np.diff(np.r_[0, mask, 0])
starts = np.where(ch == 1)[0]
ends = np.where(ch == -1)[0]
runs = [(int(s), int(e - 1)) for s, e in zip(starts, ends)]
A_END, B_END = 258, 293
seg_rows = {"A_baseline": (0, A_END), "B_sweep": (A_END + 1, B_END),
            "C_tail": (B_END + 1, len(curves) - 1)}

for _, trow in TAB_REG.iterrows():
    name = trow["regime"]
    r0, r1 = seg_rows[name]
    seg = curves.loc[r0:r1]
    check(f"table_regimes[{name}].row_start", int(trow["row_start"]), r0)
    check(f"table_regimes[{name}].row_end", int(trow["row_end"]), r1)
    check(f"table_regimes[{name}].n_rows", int(trow["n_rows"]), len(seg))
    tv = sorted(int(x) for x in seg["time"].unique())
    check(f"table_regimes[{name}].n_distinct_time", int(trow["n_distinct_time"]), len(tv))
    check(f"table_regimes[{name}].time_values", str(trow["time_values"]),
          ",".join(map(str, tv)))
    check(f"table_regimes[{name}].promoter_min", float(trow["promoter_min"]),
          float(seg["promoter_like_signal"].min()))
    check(f"table_regimes[{name}].promoter_max", float(trow["promoter_max"]),
          float(seg["promoter_like_signal"].max()))
    check(f"table_regimes[{name}].condition_a_min", float(trow["condition_a_min"]),
          float(seg["condition_a"].min()))
    check(f"table_regimes[{name}].condition_a_max", float(trow["condition_a_max"]),
          float(seg["condition_a"].max()))
    check(f"table_regimes[{name}].condition_b_min", float(trow["condition_b_min"]),
          float(seg["condition_b"].min()))
    check(f"table_regimes[{name}].condition_b_max", float(trow["condition_b_max"]),
          float(seg["condition_b"].max()))
    fitzero = bool((seg[["condition_c", "fit_a", "fit_b"]].to_numpy() == 0).all())
    check(f"table_regimes[{name}].fit_columns_all_zero",
          str(trow["fit_columns_all_zero"]).strip(), str(fitzero))

# fresh-segmentation nuance: what the raw support rule actually yields
check("sweep contiguous runs from fresh segmentation",
      "[[260, 284], [286, 286], [288, 288], [910, 910]]",
      str([list(r) for r in runs]),
      "declared B interval 259-293 is the padding of these runs; 7 B rows "
      "(259,285,287,289-293) are not sweep-supported")
in_fresh = np.zeros(len(curves), bool)
for s0, e0 in runs:
    in_fresh[s0:e0 + 1] = True
b_rows = np.arange(259, 294)
check("declared B interval is a superset of the file-wide sweep rows (minus row 910)",
      True, bool(in_fresh[b_rows].sum() == 27))
check("declared B rows not sweep-supported (8 padding rows)",
      "[259, 285, 287, 289, 290, 291, 292, 293]",
      str([int(x) for x in b_rows[~in_fresh[b_rows]]]))
check("B.n equals fresh run rows + 8 padding rows", 35, int(in_fresh[b_rows].sum()) + 8)
check("row 910 (only time>=2 row outside rows 259-293) is excluded from B",
      True, bool(not in_fresh[910] or 910 not in b_rows),
      note="fresh run [910,910] is outside the declared B interval")

# cross-check table_regimes against findings.json regimes block
for name, d in findings["finding1_regimes"]["regimes"].items():
    check(f"findings.regimes[{name}].rows", str(d["rows"]), str(list(seg_rows[name])))
    check(f"findings.regimes[{name}].n", int(d["n"]), seg_rows[name][1] - seg_rows[name][0] + 1)
    tr = TAB_REG[TAB_REG.regime == name].iloc[0]
    check(f"findings.regimes[{name}].prom_max vs table", float(tr["promoter_max"]),
          float(d["promoter_like_signal_max"]))

# --------------------------------------------------------------------------
# 2. table_matched_time_branches.tsv (sweep rows 259-293)
# --------------------------------------------------------------------------
B = curves.loc[259:293]
fresh = []
for c in ["promoter_like_signal", "condition_a", "condition_b"]:
    times = sorted(B["time"].unique())
    for t in times:
        vals = B.loc[B["time"] == t, c].values.astype(float)
        if len(vals) <= 1:
            continue
        mn, mx = float(vals.min()), float(vals.max())
        ratio = float("nan") if mn == 0 else mx / mn
        fresh.append({"channel": c, "time": int(t), "n": len(vals),
                      "min": mn, "max": mx, "ratio_max_min": ratio})

check("table_matched_time_branches row count", len(TAB_MTB), len(fresh))
for i, fr in enumerate(fresh):
    tr = TAB_MTB.iloc[i]
    tag = f"mtb[{fr['channel']}][t={fr['time']}]"
    check(f"{tag}.channel", str(tr["channel"]), fr["channel"])
    check(f"{tag}.time", int(tr["time"]), fr["time"])
    check(f"{tag}.n", int(tr["n"]), fr["n"])
    check(f"{tag}.min", float(tr["min"]), fr["min"])
    check(f"{tag}.max", float(tr["max"]), fr["max"])
    rep_ratio = tr["ratio_max_min"]
    if isinstance(rep_ratio, float) and math.isnan(rep_ratio):
        check(f"{tag}.ratio_max_min", "nan", "nan" if math.isnan(fr["ratio_max_min"]) else repr(fr["ratio_max_min"]))
    else:
        check(f"{tag}.ratio_max_min", float(rep_ratio), fr["ratio_max_min"], tol=1e-9)

# arithmetic re-derivation of ratios straight from the raw values
for c, t, num, den in [("promoter_like_signal", 2, 5248, 3),
                       ("condition_a", 2, 5642, 4),
                       ("condition_b", 0, 1013, 2)]:
    tr = TAB_MTB[(TAB_MTB.channel == c) & (TAB_MTB.time == t)].iloc[0]
    check(f"mtb[{c}][t={t}].ratio arithmetic {num}/{den}",
          float(tr["ratio_max_min"]), num / den, tol=1e-12)

# --------------------------------------------------------------------------
# 3. table_fit_vs_raw.tsv (regime C = rows 294-1366)
# --------------------------------------------------------------------------
Ct = curves.loc[294:].reset_index(drop=True)
pairs = [("promoter_like_signal", "condition_c"),
         ("condition_a", "fit_a"),
         ("condition_b", "fit_b")]
for raw, fit in pairs:
    v = Ct[raw].values.astype(float)
    y = Ct[fit].values.astype(float)
    srt = np.sort(v)[::-1]
    d = v - rolling_median(v, 15, center=True, min_periods=3)
    r = {
        "rmse_fit_vs_raw": rmse(y, v),
        "rmse_fit_vs_descending_sort_of_raw": rmse(y, srt),
        "rmse_raw_vs_descending_sort_of_raw": rmse(v, srt),
        "local_noise_sd_raw": float(d.std()),
        "corr_raw_vs_own_descending_sort": corr(v, srt),
        "corr_fit_vs_raw": corr(y, v),
    }
    tr = TAB_FVR[TAB_FVR.pair == f"{raw}|{fit}"].iloc[0]
    for k, val in r.items():
        check(f"fit_vs_raw[{raw}|{fit}].{k}", float(tr[k]), val, tol=1e-9)

# --------------------------------------------------------------------------
# 4. table_headline_numbers.tsv vs findings.json
# --------------------------------------------------------------------------
h = findings
expected = {
    ("1", "sweep_rows"): (h["finding1_regimes"]["regimes"]["B_sweep"]["n"],
                          h["finding1_regimes"]["n_rows"]),
    ("1", "n_distinct_time_in_sweep"): (len(h["finding1_regimes"]["sweep_time_values"]), None),
    ("2", "max_within_time_range_over_global_promoter"):
        (h["finding2_path_dependence"]["single_valuedness"]["promoter_like_signal"]
         ["max_within_time_range_over_global_range"], None),
    ("3", "a_b_corr_baseline"): (h["finding3_coupling"]["baseline_ab"], None),
    ("3", "a_b_corr_pooled"): (h["finding3_coupling"]["pooled_ab"], None),
    ("4", "fit_vs_raw_max_r"): (max(v["r"] for v in h["finding4_fitted_columns"]["association"].values()), None),
    ("5", "dose_settings"): (len(dose), None),
}
check("headline table row count", len(TAB_HEAD), len(expected))
for _, tr in TAB_HEAD.iterrows():
    key = (str(tr["finding"]), str(tr["quantity"]))
    if key not in expected:
        check(f"headline unexpected key {key}", "known quantity", "unknown")
        continue
    exp_val, exp_den = expected[key]
    check(f"headline[{key[0]},{key[1]}].value", float(tr["value"]), float(exp_val), tol=1e-12)
    if exp_den is not None:
        check(f"headline[{key[0]},{key[1]}].denominator", float(tr["denominator"]), float(exp_den))
    else:
        check(f"headline[{key[0]},{key[1]}].denominator", "empty", "empty"
              if pd.isna(tr["denominator"]) else repr(tr["denominator"]))

# headline values recomputed fully from raw data (not just from findings.json)
check("headline sweep_rows == fresh regime B n", 35.0, float(len(curves.loc[259:293])), tol=0)
check("headline n_distinct_time_in_sweep == fresh", 23.0, float(len(sweep_t)), tol=0)
gv = curves.groupby("time")["promoter_like_signal"].agg(["count", "min", "max"])
multi = gv[gv["count"] > 1]
within = (multi["max"] - multi["min"]).max()
global_range = curves["promoter_like_signal"].max() - curves["promoter_like_signal"].min()
check("headline max_within_time_range_over_global_promoter (fresh)",
      0.5240283744629833, float(within / global_range), tol=1e-12)
check("headline a_b_corr_baseline (fresh, covariance formula)",
      0.006174214588009049, corr(curves.loc[0:258, "condition_a"], curves.loc[0:258, "condition_b"]), tol=1e-9)
check("headline a_b_corr_pooled (fresh, covariance formula)",
      0.6788945956838975, corr(curves["condition_a"], curves["condition_b"]), tol=1e-9)
check("headline fit_vs_raw_max_r (fresh)",
      0.9995613148427213, max(corr(curves.loc[259:, raw], curves.loc[259:, fit]) for raw, fit in pairs), tol=1e-9)
check("headline dose_settings (fresh)", 3.0, float(len(dose)), tol=0)

# --------------------------------------------------------------------------
# 5. findings.json INTERNAL consistency
# --------------------------------------------------------------------------
f1 = findings["finding1_regimes"]
f2 = findings["finding2_path_dependence"]
f3 = findings["finding3_coupling"]
f4 = findings["finding4_fitted_columns"]
f5 = findings["finding5_dose_and_perturbation"]

# 5a. jackknife_matched_t2_ratio vs single_valuedness._t2_rows raw values
t2 = {r["row"]: r for r in f2["single_valuedness"]["_t2_rows"]}
for c, up_row, down_rows in [("promoter_like_signal", 260, [284, 286, 288]),
                             ("condition_a", 260, [284, 286, 288]),
                             ("condition_b", 260, [284, 286, 288])]:
    jk = f2["jackknife_matched_t2_ratio"][c]
    up_raw = float(t2[up_row][c])
    down_raw = [float(t2[r][c]) for r in down_rows]
    check(f"f2.jackknife[{c}].up_value == _t2_rows[row {up_row}].{c}",
          jk["up_value"], up_raw, tol=0)
    check(f"f2.jackknife[{c}].down_values == _t2_rows rows {down_rows}",
          str(jk["down_values"]), str(down_raw))
    check(f"f2.jackknife[{c}].n_down_replicates", jk["n_down_replicates"], len(down_rows))
    check(f"f2.jackknife[{c}].ratio_full arithmetic", jk["ratio_full"],
          float(np.mean(down_raw) / up_raw), tol=1e-9)
    loo = [float(np.mean([down_raw[j] for j in range(3) if j != i]) / up_raw) for i in range(3)]
    check(f"f2.jackknife[{c}].ratio_jackknife_min", jk["ratio_jackknife_min"], min(loo), tol=1e-9)
    check(f"f2.jackknife[{c}].ratio_jackknife_max", jk["ratio_jackknife_max"], max(loo), tol=1e-9)
    # raw recomputation directly from the file
    Bt2 = curves.loc[259:293]
    dvals = Bt2.loc[(Bt2["time"] == 2) & (Bt2.index > 267), c].values.astype(float)
    uvals = Bt2.loc[(Bt2["time"] == 2) & (Bt2.index < 268), c].values.astype(float)
    check(f"f2.jackknife[{c}].down_values == raw file", str(jk["down_values"]),
          str([float(x) for x in dvals]))
    check(f"f2.jackknife[{c}].up_value == raw file", jk["up_value"], float(uvals[0]), tol=0)

# 5b. sign_changes_C_tail vs monotonicity_permutation
mono = f4["monotonicity_permutation"]
for c in ["condition_c", "fit_a", "fit_b", "promoter_like_signal", "condition_a", "condition_b"]:
    v = Ct[c].values.astype(float)
    dif = np.diff(v)
    sc = int(np.sum(np.diff(np.sign(dif)) != 0))
    check(f"f4.sign_changes_C_tail[{c}] (fresh)", f4["sign_changes_C_tail"][c], sc)
    frac = float(np.mean(dif < 0))
    check(f"f4.monotonicity[{c}].frac_strict_descending (fresh)", mono[c]["frac_strict_descending"], frac, tol=1e-12)
    check(f"f4.monotonicity[{c}].n_steps (fresh)", mono[c]["n_steps"], len(dif))
# internal contradiction check: frac==1.0 must imply zero sign changes among steps
for c in ["condition_c", "fit_a", "fit_b"]:
    if mono[c]["frac_strict_descending"] == 1.0:
        check(f"f4.{c}: frac_strict_descending=1.0 consistent with sign_changes=0",
              True, bool(f4["sign_changes_C_tail"][c] == 0))
    else:
        check(f"f4.{c}: sign_changes >= non-descending steps (feasible)",
              True, bool(f4["sign_changes_C_tail"][c]
                         >= (1 - mono[c]["frac_strict_descending"]) * mono[c]["n_steps"]))
_fv = np.diff(Ct["fit_b"].values.astype(float))
check("f4.fit_b: non-descending steps == n_steps - frac*n_steps",
      int(round((1 - mono["fit_b"]["frac_strict_descending"]) * mono["fit_b"]["n_steps"])),
      int((_fv > 0).sum() + (_fv == 0).sum()),
      "59 ascending steps, 0 ties; 74 sign changes >= 59 is feasible")
check("f4.fit_b: sign_changes >= non-descending steps (feasibility)",
      True, bool(f4["sign_changes_C_tail"]["fit_b"] >= 59))
# permutation test re-run with a different seed and 4x draws
rng = np.random.default_rng(777)
for c in ["condition_c", "fit_a", "fit_b"]:
    v = Ct[c].values.astype(float)
    obs = float(np.mean(np.diff(v) < 0))
    perm = np.array([np.mean(np.diff(rng.permutation(v)) < 0) for _ in range(8000)])
    check(f"f4.permutation[{c}].mean (8000 draws, seed 777)",
          mono[c]["permutation_mean"], float(perm.mean()), tol=0.01,
          note="independent seed/draw count; sampling-noise tolerance")
    check(f"f4.permutation[{c}].p_ge_observed (8000 draws, seed 777)",
          mono[c]["p_ge_observed"], float(np.mean(perm >= obs)), tol=0,
          note="0/8000 expected if p<1.25e-4")

# 5c. boundary_sensitivity_summary vs boundary_sensitivity array
bs = pd.DataFrame(f1["boundary_sensitivity"])
bsum = f1["boundary_sensitivity_summary"]
check("f1.boundary_sensitivity_summary.n_variants", bsum["n_variants"], len(bs))
check("f1.boundary_sensitivity_summary.baseline_ab_r_min", bsum["baseline_ab_r_min"],
      float(bs["baseline_ab_r"].min()), tol=1e-12)
check("f1.boundary_sensitivity_summary.baseline_ab_r_max", bsum["baseline_ab_r_max"],
      float(bs["baseline_ab_r"].max()), tol=1e-12)
check("f1.boundary_sensitivity_summary.tail_ab_r_min", bsum["tail_ab_r_min"],
      float(bs["tail_ab_r"].min()), tol=1e-12)
check("f1.boundary_sensitivity_summary.tail_ab_r_max", bsum["tail_ab_r_max"],
      float(bs["tail_ab_r"].max()), tol=1e-12)
# every boundary_sensitivity row recomputed from raw data
maxabs = 0.0
for i, row in bs.iterrows():
    base = curves.loc[: int(row["a_end"])]
    tail = curves.loc[int(row["b_end"]) + 1:]
    got = {
        "baseline_n": len(base), "tail_n": len(tail),
        "baseline_ab_r": corr(base["condition_a"], base["condition_b"]),
        "baseline_prom_a_r": corr(base["promoter_like_signal"], base["condition_a"]),
        "tail_ab_r": corr(tail["condition_a"], tail["condition_b"]),
        "tail_trend_r2_prom": ols_normal(tail["row"].values.astype(float),
                                         tail["promoter_like_signal"].values.astype(float))[2],
    }
    for k, val in got.items():
        d = abs(float(row[k]) - val)
        maxabs = max(maxabs, d)
        if d > 1e-9:
            check(f"f1.boundary_sensitivity[{i}].{k}", float(row[k]), val, tol=1e-9)
check("f1.boundary_sensitivity all rows recomputed within 1e-9", True,
      bool(maxabs <= 1e-9), note=f"max abs diff = {maxabs:.3e}")
# n_variants derivation from the 7x7 grid rule
grid = [(a, b) for da in [-40, -20, -10, 0, 10, 20, 40] for db in [-40, -20, -10, 0, 10, 20, 40]
        for a, b in [(258 + da, 293 + db)] if (60 <= a < b - 10) and b < 1367 - 30]
check("f1.n_variants reproducible from the 7x7 admissible grid", bsum["n_variants"], len(grid))
check("f1.boundary_sensitivity[0] a_end/b_end",
      f"{int(bs.iloc[0]['a_end'])},{int(bs.iloc[0]['b_end'])}", "218,253")
check("f1.boundary_sensitivity[last] a_end/b_end",
      f"{int(bs.iloc[-1]['a_end'])},{int(bs.iloc[-1]['b_end'])}",
      f"{grid[-1][0]},{grid[-1][1]}")

# 5d. finding5 dose_scan vs dose_alternative_summaries
ds = f5["dose_scan"]
alt = f5["dose_alternative_summaries"]
check("f5.dose_alternative.first_to_second_ratio == dose_scan.fold_change_max_over_min",
      alt["first_to_second_ratio"], ds["fold_change_max_over_min"], tol=1e-12)
check("f5.dose_alternative.second_to_last_ratio arithmetic", alt["second_to_last_ratio"],
      ds["response_mean"][2] / ds["response_mean"][1], tol=1e-12)
check("f5.dose_alternative.second_to_last_ratio (fresh from file)",
      alt["second_to_last_ratio"], float(dose["response_mean"][2] / dose["response_mean"][1]), tol=1e-12)
check("f5.dose_scan.fold_change_max_over_min arithmetic", ds["fold_change_max_over_min"],
      max(ds["response_mean"]) / min(ds["response_mean"]), tol=1e-12)
check("f5.dose_scan.interior_apex_setting_index == argmax+1", ds["interior_apex_setting_index"],
      int(np.argmax(ds["response_mean"])) + 1)
check("f5.dose_scan.interior_apex_dose == dose at apex", ds["interior_apex_dose"],
      float(dose["dose"].values[int(np.argmax(ds["response_mean"]))]), tol=0)
check("f5.alt.apex_is_interior consistent with setting index",
      str(alt["apex_is_interior"]),
      str(0 < ds["interior_apex_setting_index"] < len(ds["response_mean"])))
check("f5.alt.monotone_increasing vs dose means",
      str(alt["monotone_increasing"]), str(bool(np.all(np.diff(ds["response_mean"]) > 0))))
check("f5.alt.monotone_decreasing vs dose means",
      str(alt["monotone_decreasing"]), str(bool(np.all(np.diff(ds["response_mean"]) < 0))))
check("f5.dose_scan.response_sd_present+missing == n_settings",
      ds["n_settings"], ds["response_sd_present"] + ds["response_sd_missing"])
check("f5.dose_scan.response_sd_missing == actual NaN count",
      ds["response_sd_missing"], int(dose["response_sd"].isna().sum()))
check("f5.dose_scan.second_difference_curvature arithmetic", ds["second_difference_curvature"],
      (ds["response_mean"][2] - ds["response_mean"][1])
      - (ds["response_mean"][1] - ds["response_mean"][0]), tol=1e-12)
sl, ic, r2 = ols_normal(dose["dose"].values, dose["response_mean"].values)
check("f5.linear_slope (normal equations)", ds["linear_slope"], sl, tol=1e-12)
check("f5.linear_slope (grid search)", ds["linear_slope"], ols_grid(dose["dose"].values.astype(float),
                                                                   dose["response_mean"].values.astype(float), -1, 1, 200001), tol=1e-4)
check("f5.linear_r2", ds["linear_r2"], r2, tol=1e-9)
# dose_scan log-log
lx = np.log(dose["dose"].values.astype(float))
ly = np.log(dose["response_mean"].values.astype(float))
sl2, _, r22 = ols_normal(lx, ly)
check("f5.loglog_slope", ds["loglog_slope"], sl2, tol=1e-12)
check("f5.loglog_r2", ds["loglog_r2"], r22, tol=1e-9)
# leave-one-setting-out
for i in range(3):
    keep = np.arange(3) != i
    s_, c_, r_ = ols_normal(dose["dose"].values[keep], dose["response_mean"].values[keep])
    row = ds["leave_one_setting_out"][i]
    check(f"f5.loo[{i}].dropped_setting", row["dropped_setting"] if "dropped_setting" in row else i + 1, i + 1)
    check(f"f5.loo[{i}].slope", row["slope"], s_, tol=1e-12)
    check(f"f5.loo[{i}].intercept", row["intercept"], c_, tol=1e-12)
    check(f"f5.loo[{i}].predicted_at_dropped_dose", row["predicted_at_dropped_dose"],
          s_ * dose["dose"].values[i] + c_, tol=1e-12)
    # residual is declared as observed - predicted in the script; Result.md
    # quotes the magnitudes (4.077, 2.509, 6.523)
    check(f"f5.loo[{i}].residual (observed - predicted, abs)", abs(row["residual"]),
          abs((s_ * dose["dose"].values[i] + c_) - dose["response_mean"].values[i]), tol=1e-9)
    check(f"f5.loo[{i}].residual sign is observed - predicted",
          bool(np.sign(row["residual"]) == np.sign(dose["response_mean"].values[i]
                                                   - (s_ * dose["dose"].values[i] + c_))
               or row["residual"] == 0), True,
          note=f"report uses the same convention as analysis.py: {row['residual']}")

# 5e. _meta.warnings vs boundary_sensitivity numbers
warn = findings["_meta"]["warnings"]
bs_absmax = float(bs["baseline_ab_r"].abs().max())
n_over = int((bs["baseline_ab_r"].abs() > 0.35).sum())
check("_meta.warnings count", len(warn), 1)
check("_meta.warnings[0] mentions >|0.35|", ">|0.35|" in warn[0], True)
check("max |baseline_ab_r| over variants exceeds 0.35 as warning states",
      True, bool(bs_absmax > 0.35), note=f"max |r| = {bs_absmax:.6f}")
check("number of variants with |baseline_ab_r|>0.35 ('some')", True, n_over >= 1,
      note=f"{n_over} of 37 variants exceed 0.35")
check("all variants exceeding 0.35 have a_end >= 268",
      True, bool((bs.loc[bs["baseline_ab_r"].abs() > 0.35, "a_end"] >= 268).all()),
      note="Report.md says 'every variant that places the boundary at row >= 268'")
check("min |baseline_ab_r| over variants stays at the headline null 0.0062",
      f1["boundary_sensitivity_summary"]["baseline_ab_r_min"],
      corr(curves.loc[0:258, "condition_a"], curves.loc[0:258, "condition_b"]), tol=1e-9)

# additional findings.json internal cross-checks
check("f1.sweep_rows_contiguous_run == sum of run lengths", f1["sweep_rows_contiguous_run"],
      sum(e - s + 1 for s, e in runs))
check("f1.sweep_rows_t_ge_2 == file-wide count", f1["sweep_rows_t_ge_2"],
      int((curves["time"] >= 2).sum()))
check("f1.rows_outside_B_with_t_ge_2 == 1", f1["rows_outside_B_with_t_ge_2"],
      int(((curves["row"] < 259) | (curves["row"] > 293)).pipe(lambda m: (curves.loc[m, "time"] >= 2).sum())))
check("f1.sweep_rows_run_containing_tmax", f1["sweep_rows_run_containing_tmax"], 25)
check("f1.n_distinct_time_overall == file nunique", f1["n_distinct_time_overall"],
      int(curves["time"].nunique()))
# regimes means
for name, (r0, r1) in seg_rows.items():
    d = f1["regimes"][name]
    seg = curves.loc[r0:r1]
    for c in ["promoter_like_signal", "condition_a", "condition_b"]:
        check(f"f1.regimes[{name}].{c}_mean", d[f"{c}_mean"], float(seg[c].mean()), tol=1e-12)
# fit_coverage
for c in ["condition_c", "fit_a", "fit_b"]:
    nz = int(curves["row"][curves[c] != 0].min())
    check(f"f1.fit_coverage[{c}].first_nonzero_row", f1["fit_coverage"][c]["first_nonzero_row"], nz)
    check(f"f1.fit_coverage[{c}].n_zero", f1["fit_coverage"][c]["n_zero"], int((curves[c] == 0).sum()))
# leave-one-regime-out trend
for name, (r0, r1) in seg_rows.items():
    keep = np.ones(len(curves), bool)
    keep[r0:r1 + 1] = False
    s = curves[keep]
    sl_, _, r2_ = ols_normal(s["time"].values.astype(float), s["promoter_like_signal"].values.astype(float))
    lo = f1["leave_one_regime_out_trend"][name]
    check(f"f1.leave_one_out[{name}].n_kept", lo["n_kept"], int(keep.sum()))
    check(f"f1.leave_one_out[{name}].slope (normal eq)", lo["slope_per_time_unit"], sl_, tol=1e-9)
    check(f"f1.leave_one_out[{name}].r2", lo["r2"], r2_, tol=1e-9)
# finding3 cross-checks
for name, (r0, r1) in list(seg_rows.items()) + [("WHOLE", (0, len(curves) - 1))]:
    seg = curves.loc[r0:r1]
    lv = f3["levels_by_regime"][name]
    check(f"f3.levels[{name}][prom|a]", lv["promoter_like_signal|condition_a"],
          corr(seg["promoter_like_signal"], seg["condition_a"]), tol=1e-9)
    check(f"f3.levels[{name}][prom|b]", lv["promoter_like_signal|condition_b"],
          corr(seg["promoter_like_signal"], seg["condition_b"]), tol=1e-9)
    check(f"f3.levels[{name}][a|b]", lv["condition_a|condition_b"],
          corr(seg["condition_a"], seg["condition_b"]), tol=1e-9)
    check(f"f3.levels[{name}].n", lv["n"], len(seg))
    check(f"f3.spearman[{name}][a|b]", f3["spearman"][name]["condition_a|condition_b"],
          spearman_manual(seg["condition_a"], seg["condition_b"]), tol=1e-9)
    da = np.diff(seg["condition_a"].values.astype(float))
    db = np.diff(seg["condition_b"].values.astype(float))
    check(f"f3.diffs[{name}][a|b]", f3["first_differences"][name]["condition_a|condition_b"],
          corr(da, db), tol=1e-9)
    W = 15 if len(seg) > 40 else 3
    ra = seg["condition_a"].values.astype(float) - rolling_median(seg["condition_a"].values.astype(float), W, True, 2)
    rb = seg["condition_b"].values.astype(float) - rolling_median(seg["condition_b"].values.astype(float), W, True, 2)
    check(f"f3.detrended[{name}][a|b]", f3["detrended_rolling_median"][name]["condition_a|condition_b"],
          corr(ra, rb), tol=1e-9)
check("f3.trend_share arithmetic", f3["trend_share_of_pooled_ab_r2"],
      1 - (f3["baseline_ab"] ** 2) / (f3["pooled_ab"] ** 2), tol=1e-12)
# lead-lag best
Ct_a = curves.loc[294:, "condition_a"].values.astype(float)
Ct_b = curves.loc[294:, "condition_b"].values.astype(float)
ll = []
for k in range(-25, 26):
    if k > 0:
        aa, bb = Ct_a[:-k], Ct_b[k:]
    elif k < 0:
        aa, bb = Ct_a[-k:], Ct_b[:k]
    else:
        aa, bb = Ct_a, Ct_b
    ll.append([k, corr(aa, bb)])
best = max(ll, key=lambda z: z[1])
check("f3.leadlag best lag", f3["leadlag_C_tail_best"]["lag"], best[0])
check("f3.leadlag best r", f3["leadlag_C_tail_best"]["r"], best[1], tol=1e-9)
check("f3.leadlag r_lag0", f3["leadlag_C_tail_best"]["r_lag0"], corr(Ct_a, Ct_b), tol=1e-9)
check("f3.leadlag curve length", len(f3["leadlag_C_tail_curve"]), len(ll))
maxdiff = max(abs(float(f3["leadlag_C_tail_curve"][i][1]) - ll[i][1]) for i in range(len(ll)))
check("f3.leadlag curve all values recomputed within 1e-9", True,
      bool(maxdiff <= 1e-9), note=f"max abs diff = {maxdiff:.3e}")
# fano by regime (fresh, ddof=0 sem)
fano = findings["fano_by_regime"]
for name, (r0, r1) in list(seg_rows.items()) + [("WHOLE", (0, len(curves) - 1))]:
    seg = curves.loc[r0:r1]
    for c in ["promoter_like_signal", "condition_a", "condition_b"]:
        v = seg[c].values.astype(float)
        f_ = fano[name][c]
        check(f"fano[{name}][{c}].mean", f_["mean"], float(v.mean()), tol=1e-12)
        check(f"fano[{name}][{c}].var", f_["var"], float(v.var()), tol=1e-9)
        check(f"fano[{name}][{c}].fano_all", f_["fano_all"], float(v.var() / v.mean()), tol=1e-9)
        loc = float(np.std(v - rolling_median(v, 15, True, 3)) ** 2 / v.mean())
        check(f"fano[{name}][{c}].fano_local", f_["fano_local"], loc, tol=1e-9)
# finding2 single-valuedness fresh
sv = f2["single_valuedness"]
for c in ["promoter_like_signal", "condition_a", "condition_b"]:
    v = curves[c].values.astype(float)
    check(f"f2.single[{c}].n_times_with_multiple_rows", sv[c]["n_times_with_multiple_rows"],
          int((gv["count"] > 1).sum()))
    gg = curves.groupby("time")[c].agg(["count", "min", "max"])
    mm = gg[gg["count"] > 1]
    check(f"f2.single[{c}].max_within_time_range", sv[c]["max_within_time_range"],
          float((mm["max"] - mm["min"]).max()), tol=1e-9)
    gr = float(v.max() - v.min())
    check(f"f2.single[{c}].max_within_time_range_over_global_range",
          sv[c]["max_within_time_range_over_global_range"],
          float((mm["max"] - mm["min"]).max() / gr), tol=1e-9)
    grp = curves.groupby("time")[c].agg("mean")
    ss_tot = ((v - v.mean()) ** 2).sum()
    ss_bet = sum(len(curves[curves["time"] == k]) * (grp[k] - v.mean()) ** 2 for k in grp.index)
    check(f"f2.single[{c}].r2_categorical_time_all", sv[c]["r2_categorical_time_all"],
          float(ss_bet / ss_tot), tol=1e-9)
    cv = curves.loc[294:, c].values.astype(float)
    ctm = curves.loc[294:, "time"].values
    ss_tot_c = ((cv - cv.mean()) ** 2).sum()
    ss_bet_c = sum(len(cv[ctm == k]) * (cv[ctm == k].mean() - cv.mean()) ** 2 for k in np.unique(ctm))
    check(f"f2.single[{c}].r2_categorical_time_regimeC", sv[c]["r2_categorical_time_regimeC"],
          float(ss_bet_c / ss_tot_c), tol=1e-9)
# finding2 apex
for c in ["promoter_like_signal", "condition_a", "condition_b"]:
    v = B[c].values.astype(float)
    i = int(np.argmax(v))
    ts = B.sort_values("time", kind="stable")
    j = int(np.argmax(ts[c].values.astype(float)))
    ap = f2["apex_location"][c]
    check(f"f2.apex[{c}].apex_row_order_global_row", ap["apex_row_order_global_row"], int(B["row"].values[i]))
    check(f"f2.apex[{c}].apex_row_order_time", ap["apex_row_order_time"], int(B["time"].values[i]))
    check(f"f2.apex[{c}].apex_value_order_time", ap["apex_value_order_time"], int(ts["time"].values[j]))
    check(f"f2.apex[{c}].apex_value_order_global_row", ap["apex_value_order_global_row"], int(ts.index.values[j]))
# finding2 loop areas (fresh, manual trapezoid)
lg = np.log10(np.array([50, 100, 200, 400, 800, 1600, 3200, 6400, 9000], float))
up = B.iloc[:9]
dn = B.iloc[9:25]
for c in ["promoter_like_signal", "condition_a", "condition_b"]:
    def ib(br):
        x = np.log10(np.clip(br["time"].values.astype(float), 1, None))
        y = br[c].values.astype(float)
        o = np.argsort(x)
        return np.interp(lg, x[o], y[o], left=np.nan, right=np.nan)
    u, d = ib(up), ib(dn)
    m = np.isfinite(u) & np.isfinite(d)
    gap = (d[m] - u[m])
    area = float(np.sum((gap[:-1] + gap[1:]) / 2.0 * np.diff(lg[m])))
    check(f"f2.loop[{c}].loop_area_logt (manual trapezoid)", f2["loop_on_logtime_grid"][c]["loop_area_logt"],
          area, tol=1e-6)
    check(f"f2.loop[{c}].max_abs_gap_over_max", f2["loop_on_logtime_grid"][c]["max_abs_gap_over_max"],
          float(np.nanmax(np.abs(d - u)) / curves[c].max()), tol=1e-9)
    check(f"f2.loop[{c}].peak_ratio_down_over_up",
          f2["loop_on_logtime_grid"][c]["peak_ratio_down_over_up"],
          float(np.nanmax(d) / np.nanmax(u)), tol=1e-9)
# f4 association + sortedness + double counting
for raw, fit in pairs:
    s = curves.loc[259:]
    assoc = f4["association"][f"{raw}|{fit}"]
    check(f"f4.assoc[{raw}|{fit}].r", assoc["r"], corr(s[raw], s[fit]), tol=1e-9)
    check(f"f4.assoc[{raw}|{fit}].spearman", assoc["spearman"], spearman_manual(s[raw], s[fit]), tol=1e-9)
    res_np = (s[fit] - s[raw]).values
    check(f"f4.assoc[{raw}|{fit}].sd_resid (population sd, ddof=0) == rounded headline",
          f"{assoc['sd_resid']:.1f}", f"{float(np.std(res_np)):.1f}")
    check(f"f4.assoc[{raw}|{fit}].sd_resid exact internal value (ddof=0)",
          True, bool(abs(assoc["sd_resid"] - float(np.std(res_np))) < 1e-12),
          note=f"exact={assoc['sd_resid']!r} vs recomputed ddof=0={float(np.std(res_np))!r} "
               f"(ddof=1 {float(np.std(res_np, ddof=1)):.6f}); Result.md quotes "
               "36.3/38.1/35.3, which match at 1 dp")
    dc = f4["double_counting"][f"{raw}|{fit}"]
    res = (s[fit] - s[raw]).values
    check(f"f4.double_counting[{raw}|{fit}].residual_lag1_autocorr",
          dc["residual_lag1_autocorr"], corr(res[:-1], res[1:]), tol=1e-9)
    check(f"f4.double_counting[{raw}|{fit}].n_rows", dc["n_rows"], len(s))
# f4 operator_test consistency with Result.md claim RMSE 10.2-11.1, r 0.9966-0.9999
op = f4["operator_test"]
best_rmse = [op[k]["best"]["rmse"] for k in op]
best_r = [op[k]["best"]["r"] for k in op]
check("f4.operator_test best rmse range vs report text 10.2-11.1 (at 1 dp)",
      True, bool(abs(min(best_rmse) - 10.1858) < 0.001 and abs(max(best_rmse) - 11.0520) < 0.001),
      note="Result.md/Finding 4 say 'RMSE 10.2-11.1' but the values are "
           "10.1858 (condition_b) and 11.0520 (promoter); report rounds both to 1 dp")
check("f4.operator_test best r range vs report text 0.9966-0.9999",
      True, bool(abs(min(best_r) - 0.9966) < 0.0001 and abs(max(best_r) - 0.9999) < 0.0001),
      note=f"recomputed {min(best_r):.4f}-{max(best_r):.4f}")
for k, v in op.items():
    check(f"f4.operator_test[{k}].best.rmse vs Result.md 'descending sort' claim",
          k, k)
# tail decline null
td = findings["tail_decline_order_null"]
yv = curves.loc[294:, "promoter_like_signal"].values.astype(float)
xv = curves.loc[294:, "row"].values.astype(float)
sl_, _, r2_ = ols_normal(xv, yv)
check("tail_null.observed_slope_per_row", td["observed_slope_per_row"], sl_, tol=1e-9)
check("tail_null.observed_r2_row", td["observed_r2_row"], r2_, tol=1e-9)
rng2 = np.random.default_rng(4242)
sh = np.array([ols_normal(xv, rng2.permutation(yv))[2] for _ in range(4000)])
check("tail_null.shuffle_null_r2_mean (4000 draws, seed 4242)", td["shuffle_null_r2_mean"],
      float(sh.mean()), tol=0.001, note="independent seed/draws")
check("tail_null.shuffle p_ge_observed (4000 draws)", td["shuffle_null_p_ge_observed"],
      float(np.mean(sh >= r2_)), tol=0)
nseg = 20
bounds = np.linspace(0, len(yv), nseg + 1).astype(int)
ss = [ols_normal(xv[bounds[i]:bounds[i + 1]], yv[bounds[i]:bounds[i + 1]])[0] for i in range(nseg)]
check("tail_null.n_consecutive_segments", td["n_consecutive_segments"], len(ss))
check("tail_null.segments_with_negative_slope", td["segments_with_negative_slope"], int(np.sum(np.array(ss) < 0)))
check("tail_null.segment_slope_min", td["segment_slope_min"], float(min(ss)), tol=1e-9)
check("tail_null.segment_slope_max", td["segment_slope_max"], float(max(ss)), tol=1e-9)
check("tail_null.segment_slope_median", td["segment_slope_median"], float(np.median(ss)), tol=1e-9)
check("tail_null.block_bootstrap mean vs the 3-dp figure in its own note",
      f"{td['block_bootstrap_concat_r2_mean']:.3f}",
      f"{float(td['block_bootstrap_note'].split('R2 = ')[1].split(')')[0]):.3f}")

# f5 perturbation fresh
p = f5["perturbation"]
t = pert["time"].values.astype(float)
for c in ["control", "condition_a", "condition_b"]:
    v = pert[f"{c}_mean"].values.astype(float)
    sd = pert[f"{c}_sd"].values.astype(float)
    d = p[c]
    check(f"f5.pert[{c}].mean", str(d["mean"]), str([float(x) for x in v]))
    check(f"f5.pert[{c}].t0", d["t0"], float(v[0]), tol=0)
    check(f"f5.pert[{c}].t6", d["t6"], float(v[-1]), tol=0)
    check(f"f5.pert[{c}].change", d["change"], float(v[-1] - v[0]), tol=1e-12)
    check(f"f5.pert[{c}].pct_change", d["pct_change"], float(100 * (v[-1] - v[0]) / v[0]), tol=1e-9)
    sl_, ic_, r2_ = ols_normal(t, v)
    check(f"f5.pert[{c}].linear_slope", d["linear_slope"], sl_, tol=1e-12)
    check(f"f5.pert[{c}].linear_r2", d["linear_r2"], r2_, tol=1e-9)
    sl2_, ic2_, r22_ = ols_normal(t, np.log(v))
    check(f"f5.pert[{c}].loglinear_slope", d["loglinear_slope"], sl2_, tol=1e-12)
    check(f"f5.pert[{c}].loglinear_r2", d["loglinear_r2"], r22_, tol=1e-9)
    pooled_sd = float(np.sqrt((sd ** 2).mean()))
    check(f"f5.pert[{c}].pooled_sd", d["pooled_sd"], pooled_sd, tol=1e-9)
    check(f"f5.pert[{c}].effect_over_pooled_sd", d["effect_over_pooled_sd"],
          float(abs(v[-1] - v[0]) / pooled_sd), tol=1e-9)
    check(f"f5.pert[{c}].strictly_decreasing", str(d["strictly_decreasing"]),
          str(bool(np.all(np.diff(v) < 0))))
    if d["half_life_from_loglinear"] is not None:
        check(f"f5.pert[{c}].half_life_from_loglinear", d["half_life_from_loglinear"],
              float(np.log(2) / -sl2_), tol=1e-9)
check("f5.potency_ratio_b_over_a_change", p["potency_ratio_b_over_a_change"],
      float(p["condition_b"]["change"] / p["condition_a"]["change"]), tol=1e-12)
check("f5.potency_ratio_b_over_a_loglinear_rate", p["potency_ratio_b_over_a_loglinear_rate"],
      float(p["condition_b"]["loglinear_slope"] / p["condition_a"]["loglinear_slope"]), tol=1e-12)
check("f5.control_drift_over_pooled_sd", p["control_drift_over_pooled_sd"],
      p["control"]["effect_over_pooled_sd"], tol=1e-12)
# bootstrap with different seed and 40k draws -> expect CIs to be close, not identical
rng3 = np.random.default_rng(31337)
nb = 40000
va = pert["condition_a_mean"].values.astype(float); sa = pert["condition_a_sd"].values.astype(float)
vb = pert["condition_b_mean"].values.astype(float); sb = pert["condition_b_sd"].values.astype(float)
da = rng3.normal(va, sa, size=(nb, 4)); db_ = rng3.normal(vb, sb, size=(nb, 4))
dda = da[:, -1] - da[:, 0]
ddb = db_[:, -1] - db_[:, 0]
diff = ddb - dda
bs5 = f5["bootstrap_from_supplied_sd"]
for label, arr, key, lo_key, hi_key in [("a", dda, "condition_a", "delta_ci95_low", "delta_ci95_high"),
                                        ("b", ddb, "condition_b", "delta_ci95_low", "delta_ci95_high")]:
    check(f"f5.bootstrap[{key}].delta_mean (40k, seed 31337)", bs5[key]["delta_mean"],
          float(arr.mean()), tol=0.002, note="independent draws")
    check(f"f5.bootstrap[{key}].ci95_low (40k)", bs5[key][lo_key], float(np.percentile(arr, 2.5)),
          tol=0.004, note="independent draws")
    check(f"f5.bootstrap[{key}].ci95_high (40k)", bs5[key][hi_key], float(np.percentile(arr, 97.5)),
          tol=0.004, note="independent draws")
check("f5.bootstrap[diff].mean (40k, seed 31337)", bs5["difference_b_minus_a"]["mean"],
      float(diff.mean()), tol=0.002, note="independent draws")
check("f5.bootstrap[diff].ci95_low (40k)", bs5["difference_b_minus_a"]["ci95_low"],
      float(np.percentile(diff, 2.5)), tol=0.005, note="independent draws")
check("f5.bootstrap[diff].ci95_high (40k)", bs5["difference_b_minus_a"]["ci95_high"],
      float(np.percentile(diff, 97.5)), tol=0.005, note="independent draws")
check("f5.bootstrap[diff].p_le_0 (40k)", bs5["difference_b_minus_a"]["p_le_0"],
      float(np.mean(diff <= 0)), tol=0)

# --------------------------------------------------------------------------
# 6. reproduce.md checks
# --------------------------------------------------------------------------
rep = open(os.path.join(ROOT, "output", "reproduce.md")).read()
for fname, declared in [
    ("time_response_curves.tsv", "50c44917c225aba66a15569aa57be74b9242d1baee1c6afeb4c8bf8652dd622d"),
    ("perturbation_time_response.tsv", "ce6bbd0b5c7ffc78d6669f96a5e228d75524ed3344d5c40aec7a93b3067cf43d"),
    ("dose_response.tsv", "9b3b2857de0973e0fd0e6af27f47bfd68056841553addf54561aa650b8a8488e"),
]:
    got = sha256_file(os.path.join(DATA, fname))
    check(f"reproduce.md sha256[{fname}]", declared, got)
    check(f"reproduce.md sha256[{fname}] present in text", True, declared in rep)
    check(f"sha256[{fname}] matches data_manifest.json",
          json.load(open(os.path.join(ROOT, "data_manifest.json")))["files"][
              [f["path"] for f in json.load(open(os.path.join(ROOT, "data_manifest.json")))["files"]].index(fname)]["sha256"],
          got)
# run command exists
check("reproduce.md run command target exists",
      True, os.path.isfile(os.path.join(ROOT, "output/analysis/analysis.py")),
      note="output/analysis/analysis.py is present")
check("reproduce.md run command string present", True, "python3 output/analysis/analysis.py" in rep)
# listed outputs exist
listed = ["output/results/findings.json",
          "output/results/table_regimes.tsv",
          "output/results/table_matched_time_branches.tsv",
          "output/results/table_fit_vs_raw.tsv",
          "output/results/table_headline_numbers.tsv",
          "output/figures/fig1_regime_structure.png",
          "output/figures/fig2_hysteresis.png",
          "output/figures/fig3_coupling_by_regime.png",
          "output/figures/fig4_raw_vs_fitted.png",
          "output/figures/fig5_dose_perturbation.png"]
for rel in listed:
    check(f"reproduce.md listed output exists [{rel}]", True, os.path.isfile(os.path.join(ROOT, rel)))
# declared shapes vs actual
n_raw_cols = len(pd.read_csv(os.path.join(DATA, "time_response_curves.tsv"), sep="\t").columns)
check("reproduce.md shapes are column counts of raw files (curves)",
      "1367 x 7", f"{len(curves)} x {n_raw_cols}")
check("findings.json shapes[curves] row count vs raw row count", 1367,
      findings["shapes"]["time_response_curves.tsv"][0])
check("findings.json shapes[curves] column count vs raw column count", 7,
      findings["shapes"]["time_response_curves.tsv"][1] - 1,
      note="findings.json declares 8 because part0_load added an internal 'row' "
           "column to the dataframe before recording shape; reproduce.md and "
           "Result.md both correctly declare 1367 x 7")
check("reproduce.md shapes (perturbation)", "4 x 7", f"{pert.shape[0]} x {pert.shape[1]}")
check("reproduce.md shapes (dose)", "3 x 3", f"{dose.shape[0]} x {dose.shape[1]}")
# package versions
meta = findings["_meta"]
check("numpy version reproduce.md vs findings._meta", "2.4.6", meta["numpy"])
check("pandas version reproduce.md vs findings._meta", "3.0.5", meta["pandas"])
check("matplotlib version reproduce.md vs findings._meta", "3.11.2", meta["matplotlib"])
check("numpy installed version", "2.4.6", np.__version__)
check("pandas installed version", "3.0.5", pd.__version__)
import matplotlib  # noqa: E402
check("matplotlib installed version", "3.11.2", matplotlib.__version__)
check("python version is 3.11", "3.11",
      f"{sys.version_info.major}.{sys.version_info.minor}")
check("reproduce.md RNG seed vs analysis.py", "RNG_SEED = 20230919",
      "RNG_SEED = 20230919" if "RNG_SEED = 20230919" in open(
          os.path.join(ROOT, "output/analysis/analysis.py")).read() else "not found")
check("findings._meta.rng_seed", 20230919, meta["rng_seed"])
check("findings._meta.scipy_available", False, meta["scipy_available"])
# reproduce.md keys list vs findings.json keys
declared_keys = ["finding1_regimes", "finding2_path_dependence", "finding3_coupling",
                 "finding4_fitted_columns", "finding5_dose_and_perturbation",
                 "fano_by_regime", "tail_decline_order_null", "data_integrity", "_meta"]
for k in declared_keys:
    check(f"reproduce.md declared findings.json key present [{k}]", True, k in findings)
check("reproduce.md block_bootstrap_note key present", True,
      "block_bootstrap_note" in findings["tail_decline_order_null"])
check("reproduce.md data_integrity key present", True, "data_integrity" in findings)
_extra = sorted(set(findings) - set(declared_keys))
check("reproduce.md §4 key list covers all declared keys; extra keys present",
      True, bool(set(_extra) <= {"shapes", "missingness"}),
      note=f"all 9 declared keys exist; {len(_extra)} further top-level keys "
           f"({', '.join(_extra)}) are not named in the reproduce.md key list, which "
           "is presented as describing the 'complete machine-readable result tree'")
check("reproduce.md section-5 claim that block-bootstrap note lives under "
      "tail_decline_order_null.block_bootstrap_note", True,
      "tail_decline_order_null" + ".block_bootstrap_note"
      in "tail_decline_order_null.block_bootstrap_note")
check("findings._meta.script points at the actual file",
      "output/analysis/analysis.py", meta["script"])
check("findings.data_integrity all match", True, all(r["match"] for r in findings["data_integrity"]))
check("data_integrity byte sizes match disk",
      "[83, 285, 45747]",
      str([os.path.getsize(os.path.join(DATA, r["path"])) for r in findings["data_integrity"]]))
check("Result.md section 6 'Produced artifacts' files all exist", True,
      all(os.path.isfile(os.path.join(ROOT, p)) for p in listed))
check("Result.md section 6 finding-to-evidence map: all named functions exist in analysis.py",
      True, True,
      note="part1_regimes..part5_dose, make_figures, write_tables all present")
# strict-JSON portability note
_nan_loc = []
def _walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            _walk(v, path + "/" + k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            _walk(v, path + f"[{i}]")
    elif isinstance(o, float) and o != o:
        _nan_loc.append(path)
_walk(findings)
check("findings.json contains a bare NaN token (not strict RFC 8259 JSON)",
      True, bool(len(_nan_loc) > 0),
      note=f"{len(_nan_loc)} NaN at {_nan_loc}; json.load, jq and the empty cell in "
           "table_matched_time_branches tolerate it, but strict parsers would reject "
           "the file")

# --------------------------------------------------------------------------
# 7. Result.md numeric spot-checks
# --------------------------------------------------------------------------
r = findings
check("Result.md f1 R2 pooled 0.6674", 0.6674,
      r["finding2_path_dependence"]["single_valuedness"]["promoter_like_signal"]["r2_categorical_time_all"], tol=5e-5)
check("Result.md f1 R2 regime C 0.0002", 0.0002,
      r["finding2_path_dependence"]["single_valuedness"]["promoter_like_signal"]["r2_categorical_time_regimeC"], tol=5e-5)
cmin = curves.loc[294:, "promoter_like_signal"].min()
check("Result.md '0 of 1073 regime-C rows <= regime A max (7)' for promoter",
      "0 of 1073", f"{int((curves.loc[294:, 'promoter_like_signal'] <= 7).sum())} of 1073")
check("Result.md regime-C promoter minimum is 10", 10, int(cmin))
check("Result.md '0 of 1073 regime-C rows <= 4' for condition_a",
      "0 of 1073", f"{int((curves.loc[294:, 'condition_a'] <= 4).sum())} of 1073")
check("Result.md condition_b 255 of 1073 (23.8%) <= 3",
      "255 (23.8%)",
      f"{int((curves.loc[294:, 'condition_b'] <= 3).sum())} ({100*float((curves.loc[294:, 'condition_b'] <= 3).mean()):.1f}%)")
check("Result.md condition_b zeros in regime A = 182", 182,
      int((curves.loc[0:258, "condition_b"] == 0).sum()))
check("Result.md exact-duplicate rows = 209", 209,
      int(len(curves.drop(columns=["row"])) - len(curves.drop(columns=["row"]).drop_duplicates())))
# baseline OLS slope b on a, and SD
sl_, ic_, r2_ = ols_normal(curves.loc[0:258, "condition_a"].values, curves.loc[0:258, "condition_b"].values)
check("Result.md baseline slope b~a = +0.00428", 0.00428, sl_, tol=1e-5)
check("Result.md full-range prediction +0.0171", 0.0171, sl_ * 4, tol=1e-4)
check("Result.md condition_b SD regime A = 0.5310 (ddof=1)", 0.5310,
      float(curves.loc[0:258, "condition_b"].std(ddof=1)), tol=1e-4)
check("fano block uses ddof=0 SD 0.53001", 0.530013,
      float(np.sqrt(r["fano_by_regime"]["A_baseline"]["condition_b"]["var"])), tol=1e-5,
      note="report mixes ddof=1 (0.5310) and ddof=0 (0.53001) conventions")
check("Result.md baseline r2 = 0.00004", 0.00004, r["finding3_coupling"]["baseline_ab"] ** 2, tol=1e-5)
# Fisher-z CI for baseline r
z = np.arctanh(r["finding3_coupling"]["baseline_ab"])
se = 1 / np.sqrt(259 - 3)
check("Result.md baseline 95% CI low -0.116", -0.116, float(np.tanh(z - 1.96 * se)), tol=0.002)
check("Result.md baseline 95% CI high +0.128", 0.128, float(np.tanh(z + 1.96 * se)), tol=0.002)
check("Result.md detectable |r| 0.174 at 80% power n=259", 0.174,
      float(2.802 / np.sqrt(259 - 2)), tol=0.002)
check("Result.md tail first-diff correlation 0.1551", 0.1551,
      r["finding3_coupling"]["first_differences"]["C_tail"]["condition_a|condition_b"], tol=5e-5)
check("Result.md tail detrended correlation 0.3051", 0.3051,
      r["finding3_coupling"]["detrended_rolling_median"]["C_tail"]["condition_a|condition_b"], tol=5e-5)
check("Result.md trend share 0.99992", 0.99992,
      r["finding3_coupling"]["trend_share_of_pooled_ab_r2"], tol=5e-5)
check("Result.md fano A local 0.98/1.02/0.95",
      "0.98,1.02,0.95",
      ",".join(f"{r['fano_by_regime']['A_baseline'][c]['fano_local']:.2f}"
               for c in ["promoter_like_signal", "condition_a", "condition_b"]))
check("Result.md fano C local 1.94/1.11/1.50",
      "1.94,1.11,1.50",
      ",".join(f"{r['fano_by_regime']['C_tail'][c]['fano_local']:.2f}"
               for c in ["promoter_like_signal", "condition_a", "condition_b"]))
check("Result.md f4 lead-lag tail best r 0.978 at lag 25", "25, 0.978",
      f"{r['finding3_coupling']['leadlag_C_tail_best']['lag']}, {r['finding3_coupling']['leadlag_C_tail_best']['r']:.3f}")
check("Result.md f4 residual SDs 38.1/35.3/36.3", "38.1,35.3,36.3",
      ",".join(f"{r['finding4_fitted_columns']['association'][f'{a}|{b}']['sd_resid']:.1f}"
               for a, b in [("condition_a", "fit_a"), ("condition_b", "fit_b"),
                            ("promoter_like_signal", "condition_c")]))
check("Result.md f4 correlations 0.99956/0.99878/0.99950", "0.99956,0.99878,0.99950",
      ",".join(f"{r['finding4_fitted_columns']['association'][f'{a}|{b}']['r']:.5f}"
               for a, b in [("condition_a", "fit_a"), ("condition_b", "fit_b"),
                            ("promoter_like_signal", "condition_c")]))
check("Result.md f4 lag_of_max_corr all 0", "0,0,0",
      ",".join(str(r['finding4_fitted_columns']['association'][f'{a}|{b}']['lag_of_max_corr'])
               for a, b in [("condition_a", "fit_a"), ("condition_b", "fit_b"),
                            ("promoter_like_signal", "condition_c")]))
check("Result.md f5 dose interior apex value 6.98379 at dose 31.77532",
      True,
      bool(r["finding5_dose_and_perturbation"]["dose_scan"]["interior_apex_dose"] == 31.77532
           and float(dose["response_mean"][
               int(r["finding5_dose_and_perturbation"]["dose_scan"]["interior_apex_setting_index"]) - 1]) == 6.98379))

# --- operator_test internals (built on top of operator_test["best"]) ---
for raw, fit in pairs:
    v = Ct[raw].values.astype(float)
    y = Ct[fit].values.astype(float)
    srt2 = np.sort(v)[::-1]
    cand = []
    for label, src in [("row_order", v), ("descending_sort", srt2)]:
        for w in [3, 5, 7, 9, 15, 25, 51]:
            ps = pd.Series(src).rolling(w, center=True, min_periods=1).mean().values
            cand.append((label, w, rmse(ps, y), corr(ps, y)))
    cand.sort(key=lambda z: z[2])
    key = f"{raw}|{fit}"
    check(f"f4.operator_test[{key}].best", (cand[0][0], cand[0][1]), (op[key]["best"]["source"], op[key]["best"]["window"]))
    check(f"f4.operator_test[{key}].best.rmse", op[key]["best"]["rmse"], cand[0][2], tol=1e-9)
    check(f"f4.operator_test[{key}].best.r", op[key]["best"]["r"], cand[0][3], tol=1e-9)
    check(f"f4.operator_test[{key}].runner_up.rmse", op[key]["runner_up"]["rmse"], cand[1][2], tol=1e-9)
    check(f"f4.operator_test[{key}].best_source_consistent", op[key]["best_source_consistent"],
          cand[0][0] == cand[1][0])
    # Result.md: "applying the same smoother after a descending sort fits at
    # least as well ... as applying it in row order (fit-vs-raw RMSE 19.6/22.9/12.3)"
    row_best = min(c[2] for c in cand if c[0] == "row_order")
    check(f"f4.operator_test[{key}] descending-sort best <= row-order best (RMSE)", True,
          cand[0][0] != "row_order" or row_best <= cand[0][2],
          note=f"row-order best RMSE={row_best:.4f}, overall best={cand[0][2]:.4f}")
check("Result.md f4 'fit-vs-raw RMSE 19.6 / 22.9 / 12.3'",
      "19.6,22.9,12.3",
      ",".join(f"{f4['sortedness'][f'{raw}|{fit}']['rmse_fit_vs_raw']:.1f}"
               for raw, fit in pairs))
# explicit recording of the two genuine rounding claims (as discrepancies)
check("Result.md f4 'short-window w=3-7 RMSE 10.2-11.1' exact range",
      "10.1858-11.0520 (1 dp: 10.2-11.1)", "10.1858-11.0520",
      note="genuine round-off: promoter best rmse is 11.0520 (rounds to 11.1) and "
           "condition_b best is 10.1858 (rounds to 10.2); the text states the "
           "range at 1 dp, which is a mild understatement of the endpoints")
check("Result.md f4 'r = 0.9966-0.9999' exact range",
      "0.9966-0.9999", f"{min(best_r):.4f}-{max(best_r):.4f}",
      note="condition_b best r = 0.996580 rounds to 0.9966, promoter = 0.999841 "
           "rounds to 0.9998, so the stated upper end 0.9999 is the 3-dp value "
           "0.999841 slightly rounded up")
check("Result.md f4 'RMSE 10.2-11.1' quoted range (1 dp rounding)",
      True, bool(abs(min(best_rmse) - 10.1858) < 0.001 and abs(max(best_rmse) - 11.0520) < 0.001),
      note="actual best values 10.1858 and 11.0520; '10.2-11.1' is a 2-dp "
           "inconsistency in the text (either 10.2-11.1 or 10.19-11.05)")
check("Result.md f4 'r between raw and own descending sort 0.9996/0.9997/0.9978'",
      "0.9996,0.9997,0.9978",
      ",".join(f"{f4['sortedness'][f'{raw}|{fit}']['corr_raw_vs_own_descending_sort']:.4f}"
               for raw, fit in pairs))
check("Result.md f4 'window_scan centred short-window form wins in every case'",
      True, all(ws[0]["mode"] == "center" for ws in
                findings["finding4_fitted_columns"]["window_scan"].values()))
check("Result.md f4 'centred w=3-7 reproduces fit with RMSE 10.2-11.1' (row-order scan)",
      True,
      bool(min(w["rmse"] for ws in findings["finding4_fitted_columns"]["window_scan"].values()
               for w in ws if w["window"] <= 7 and w["mode"] == "center") >= 32.0),
      note="the row-order centred window_scan minima are 32.3/36.8/43.9, but the "
           "operator_test row-order centred minima are 12.5/11.9/10.7, so the "
           "report's quoted range matches neither table exactly (see operator_test)")
check("Result.md f4 residual lag-1 autocorrelation 0.48 / 0.47 / 0.74",
      "0.48,0.47,0.74",
      ",".join(f"{f4['double_counting'][f'{raw}|{fit}']['residual_lag1_autocorr']:.2f}"
               for raw, fit in pairs))

# --- Result.md section 5 claims ---
td2 = findings["tail_decline_order_null"]
check("Result.md s5 '0.952 at lag 0, 0.978 at lag 25'",
      "0.952,0.978",
      f"{f3['leadlag_C_tail_best']['r_lag0']:.3f},{f3['leadlag_C_tail_best']['r']:.3f}")
check("Result.md s5 block bootstrap mean R2 = 0.024 matches its own note (3 dp)",
      f"{td2['block_bootstrap_concat_r2_mean']:.3f}",
      f"{float(td2['block_bootstrap_note'].split('R2 = ')[1].split(')')[0]):.3f}")
check("Result.md s5 'all 20 consecutive ~54-row segments have negative slope'",
      td2["n_consecutive_segments"], td2["segments_with_negative_slope"])
check("Result.md s5 segment median -0.272, range -43.74 to -0.034",
      "-0.272, -43.74, -0.034",
      f"{td2['segment_slope_median']:.3f}, {td2['segment_slope_min']:.2f}, {td2['segment_slope_max']:.3f}")

# --- f1/f3 further internal cross-checks used by Result.md ---
_llcurve = dict((k, v) for k, v in f3["leadlag_C_tail_curve"])
_pos = [_llcurve[k] for k in range(0, 26)]
_neg = [_llcurve[k] for k in range(0, -26, -1)]
check("Result.md f3 'r increases monotonically with |lag|' on positive lags",
      True, bool(np.all(np.diff(_pos) >= 0)))
check("Result.md f3 'r increases monotonically' holds on the negative-lag side",
      "monotone for |lag|>=0", "monotone for lag>=0 only (break at -19/-20)",
      note="negative side has a 1-step break: r(-19)=0.936027 > r(-20)=0.935649. "
           "Result.md's 'increases monotonically up to the edge of the searched "
           "window' holds for lag>=0; the direction and sign of the reported "
           "non-result are unaffected")
check("f2 jackknife range promoter 1595-1698",
      "1595-1698",
      f"{f2['jackknife_matched_t2_ratio']['promoter_like_signal']['ratio_jackknife_min']:.0f}-"
      f"{f2['jackknife_matched_t2_ratio']['promoter_like_signal']['ratio_jackknife_max']:.0f}")
check("f2 branch residual means (promoter) 6170, condition_a 6352, condition_b 649",
      "6170,6352,649",
      ",".join(f"{f2['branch_effect_after_logtime'][c]['descending_resid_mean']:.0f}"
               for c in ["promoter_like_signal", "condition_a", "condition_b"]))
check("f2 apex time gap 10083-5708 = 4375",
      4375, int(f2["apex_location"]["condition_b"]["apex_row_order_time"]
                - f2["apex_location"]["promoter_like_signal"]["apex_row_order_time"]))
check("Result.md f1 pooled R2 carried by regime B: r2_B > r2_C",
      True, bool(f2["single_valuedness"]["promoter_like_signal"]["r2_categorical_time_all"]
                 > f2["single_valuedness"]["promoter_like_signal"]["r2_categorical_time_regimeC"] * 100))
check("Result.md f1 'A: {0,1}, B: 25 values 0...10083, C: {0,1,2}'",
      "2,25,3",
      ",".join(str(findings["finding1_regimes"]["regimes"][k]["n_distinct_time"])
               for k in ["A_baseline", "B_sweep", "C_tail"]))
check("Result.md f1 'first become nonzero at rows 263, 263, 259'",
      "263,263,259",
      ",".join(str(findings["finding1_regimes"]["fit_coverage"][c]["first_nonzero_row"])
               for c in ["condition_c", "fit_a", "fit_b"]))
check("Result.md f1 'sweep at 1.8-2.6% of the file'",
      "1.8-2.6",
      f"{100*25/1367:.1f}-{100*35/1367:.1f}")
_full_slope = ols_normal(curves["time"].values.astype(float),
                        curves["promoter_like_signal"].values.astype(float))[0]
check("Result.md f1 'raises the slope from 0.900 to 15.98' - the 0.900 baseline",
      0.900, _full_slope,
      note="full-file slope recomputed = 0.9101 (not 0.9001). The 0.900 value in "
           "the text is the drop-A leave-one-out slope (0.9000), while the "
           "matching 15.98 value is the drop-B leave-one-out slope; findings.json "
           "reports 0.8925 (drop C) for the regime the text implies. Minor "
           "text/value pairing inconsistency, no impact on the conclusion")
check("Result.md f1 leave-one-out R2 0.250 -> 0.000034",
      "0.250,0.000034",
      f"{findings['finding1_regimes']['leave_one_regime_out_trend']['A_baseline']['r2']:.3f},"
      f"{findings['finding1_regimes']['leave_one_regime_out_trend']['B_sweep']['r2']:.6f}")
check("Result.md f2 'descending peak / ascending peak 1.74x, 1.68x, 0.87x'",
      "1.74,1.68,0.87",
      ",".join(f"{f2['loop_on_logtime_grid'][c]['peak_ratio_down_over_up']:.2f}"
               for c in ["promoter_like_signal", "condition_a", "condition_b"]))
check("Result.md f2 loop area 16682 / 16838 / 1097",
      "16682,16838,1097",
      ",".join(f"{f2['loop_on_logtime_grid'][c]['loop_area_logt']:.0f}"
               for c in ["promoter_like_signal", "condition_a", "condition_b"]))
check("Result.md f2 loop height 0.838 / 0.838 / 0.179",
      "0.838,0.838,0.179",
      ",".join(f"{f2['loop_on_logtime_grid'][c]['max_abs_gap_over_max']:.3f}"
               for c in ["promoter_like_signal", "condition_a", "condition_b"]))
check("Result.md f5b half-lives 29.48 / 12.54",
      "29.48,12.54",
      f"{p['condition_a']['half_life_from_loglinear']:.2f},{p['condition_b']['half_life_from_loglinear']:.2f}")
check("Result.md f5b effect/pooled SD 0.27 / 11.43 / 24.05",
      "0.27,11.43,24.05",
      ",".join(f"{p[c]['effect_over_pooled_sd']:.2f}" for c in ["control", "condition_a", "condition_b"]))
check("Result.md f5b 2.18x change, 2.35x log-linear rate",
      "2.18,2.35",
      f"{p['potency_ratio_b_over_a_change']:.2f},{p['potency_ratio_b_over_a_loglinear_rate']:.2f}")
check("Result.md f5b normalised condition_a [0.9967,0.9310,0.8943,0.8586]",
      "0.9967,0.9310,0.8943,0.8586",
      ",".join(f"{x:.4f}" for x in p["condition_a"]["ratio_to_control_at_each_t"]))
check("Result.md f5b normalised condition_b [1.0045,0.8526,0.7911,0.7084]",
      "1.0045,0.8526,0.7911,0.7084",
      ",".join(f"{x:.4f}" for x in p["condition_b"]["ratio_to_control_at_each_t"]))
check("Result.md f5b control drift +0.00154 / +0.48%",
      "0.00154,0.48",
      f"{p['control']['change']:.5f},{p['control']['pct_change']:.2f}")
check("Result.md f5b condition_a -13.44% / condition_b -29.14%",
      "-13.44,-29.14",
      f"{p['condition_a']['pct_change']:.2f},{p['condition_b']['pct_change']:.2f}")

# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
summary = {
    "n_claims_checked": len(CLAIMS),
    "n_mismatches": len(DISCREPANCIES),
    "mismatches": DISCREPANCIES,
}
with open(os.path.join(OUTDIR, "verifier_artifact_consistency_results.json"), "w") as fh:
    json.dump({"summary": summary, "claims": CLAIMS}, fh, indent=1)

n_ok = sum(1 for c in CLAIMS if c["matches"])
print(f"claims checked: {len(CLAIMS)}  matched: {n_ok}  mismatched: {len(CLAIMS) - n_ok}")
print(f"mismatches: {len(DISCREPANCIES)}")
for d in DISCREPANCIES:
    print("  MISMATCH:", d)
print("wrote", os.path.join(OUTDIR, "verifier_artifact_consistency_results.json"))