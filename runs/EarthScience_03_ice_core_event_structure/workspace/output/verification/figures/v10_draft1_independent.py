#!/usr/bin/env python3
"""
INDEPENDENT verification of Draft Finding 1 (conv-f07721cb).
Written from scratch (no imports from output/analysis, no re-use of prior scripts).

Claim: of 9 columns only c04 (sulfate) and c07 (signal) carry independent info.
 (a) c02 == c01 within 0.02 m on all 1931 rows
 (b) c09 == (1950 - c08)/1000 exactly on all 15 dated rows
 (c) c01 = cumsum(c03)/100 + single +1.008 m step after row 105 (2.97 -> 4.00 m),
     residual max ~0.016 m over 41.98 m span
 (d) c06 = c01 * c05, max residual 0.0303 m over 41.98 m
 (e) c05 strictly monotone ramp, 261 distinct values over 1931 rows,
     quadratic in c01 with max residual 0.00055 g/cm3
"""
import json
import math
import numpy as np
import pandas as pd

OUT = "/data/user/conversations/conv-f07721cb-d3dc-42e9-83c7-7e986203115b/project/output/verification/figures"
DATA = "/data/user/conversations/conv-f07721cb-d3dc-42e9-83c7-7e986203115b/project/data/observations.tsv"

df = pd.read_csv(DATA, sep="\t")
n = len(df)
report = {"n_rows": n}

c01 = df["c01_depth_top_m"].to_numpy(float)
c02 = df["c02_depth_ice_snow_m"].to_numpy(float)
c03 = df["c03_samp_thick_cm"].to_numpy(float)
c04 = df["c04_sulfate_ug_kg"].to_numpy(float)
c05 = df["c05_density_ice_g_cm_3"].to_numpy(float)
c06 = df["c06_depth_w_e_m"].to_numpy(float)
c07 = df["c07_signal"].to_numpy(object)
c08 = df["c08_age_a_ad_ce"].to_numpy(float)
c09 = df["c09_age_ka_bp"].to_numpy(float)

# ------------------------------------------------------------------ (a)
d12 = np.abs(c02 - c01)
report["a"] = {
    "max_abs_c02_minus_c01": float(np.nanmax(d12)),
    "mean_abs": float(np.nanmean(d12)),
    "n_gt_0.02": int(np.sum(d12 > 0.02)),
    "n_gt_0.01": int(np.sum(d12 > 0.01)),
    "n_exactly_0.01": int(np.sum(np.isclose(d12, 0.01, atol=1e-12))),
    "n_rows": int(n),
}
# distribution of the difference rounded
vals, cnts = np.unique(np.round(d12, 6), return_counts=True)
report["a"]["diff_values"] = {float(v): int(c) for v, c in zip(vals, cnts)}

# ------------------------------------------------------------------ (b)
dated = np.where(np.isfinite(c08) & np.isfinite(c09))[0]
pred = (1950.0 - c08[dated]) / 1000.0
resid_b = c09[dated] - pred
report["b"] = {
    "n_dated": int(len(dated)),
    "max_abs_resid": float(np.max(np.abs(resid_b))),
    "rows": [
        {"i": int(i), "c08": float(c08[i]), "c09": float(c09[i]),
         "pred": float(pred[k]), "resid": float(resid_b[k])}
        for k, i in enumerate(dated)
    ],
}

# ------------------------------------------------------------------ (c)
cs = np.cumsum(c03) / 100.0
resid_c = c01 - cs
report["c"] = {
    "resid_first": float(resid_c[0]),
    "resid_last": float(resid_c[-1]),
    "span": float(c01[-1] - c01[0]),
    "n_unique_c03": int(len(np.unique(c03))),
    "c03_min": float(c03.min()), "c03_max": float(c03.max()),
}
# locate the jump: largest single-row residual increment
dres = np.diff(resid_c)
j = int(np.argmax(np.abs(dres)))
report["c"]["jump_index_0based"] = j
report["c"]["jump_size"] = float(dres[j])
report["c"]["c01_around_jump"] = [float(c01[k]) for k in range(max(0, j - 2), min(n, j + 3))]
report["c"]["resid_before_jump_last"] = float(resid_c[j])
report["c"]["resid_after_jump_first"] = float(resid_c[j + 1])
# residual after removing the single largest jump
resid_c2 = resid_c.copy()
resid_c2[j + 1:] -= dres[j]
report["c"]["resid_after_removing_biggest_jump_max"] = float(np.max(np.abs(resid_c2 - np.median(resid_c2))))
report["c"]["resid_after_removing_biggest_jump_center"] = float(np.median(resid_c2))
# also: how many residual increments are non-zero?
report["c"]["n_nonzero_resid_increments"] = int(np.sum(np.abs(dres) > 1e-9))
report["c"]["top10_abs_increments"] = sorted([float(abs(x)) for x in dres], reverse=True)[:10]
# cumulative residual trajectory summary
report["c"]["resid_min"] = float(resid_c.min())
report["c"]["resid_max"] = float(resid_c.max())

# ------------------------------------------------------------------ (d)
prod = c01 * c05
resid_d = c06 - prod
report["d"] = {
    "max_abs_resid_c06_minus_c01c05": float(np.nanmax(np.abs(resid_d))),
    "mean_abs": float(np.nanmean(np.abs(resid_d))),
    "rms": float(np.sqrt(np.nanmean(resid_d ** 2))),
    "resid_at_first": float(resid_d[0]),
    "resid_at_last": float(resid_d[-1]),
    "ratio_max_abs_over_span": float(np.nanmax(np.abs(resid_d)) / (c01[-1] - c01[0])),
}

# ------------------------------------------------------------------ (e)
u05 = np.unique(c05)
report["e"] = {
    "n_unique_c05": int(len(u05)),
    "c05_min": float(c05.min()), "c05_max": float(c05.max()),
    "step_counts_top": None,
    "strictly_monotone_overall": bool(np.all(np.diff(u05) > 0)),
}
# strictly monotone in row order?
report["e"]["monotone_nondecreasing_rows"] = bool(np.all(np.diff(c05) >= 0))
report["e"]["n_row_negative_steps"] = int(np.sum(np.diff(c05) < 0))
report["e"]["n_row_positive_steps"] = int(np.sum(np.diff(c05) > 0))
# gaps between consecutive distinct values
gaps = np.diff(u05)
report["e"]["min_gap"] = float(gaps.min())
report["e"]["max_gap"] = float(gaps.max())
report["e"]["n_distinct_gaps"] = int(len(np.unique(np.round(gaps, 9))))
# equal-c01 => equal-c05 ?
g01 = {}
eq_viol = 0
dup_c01 = 0
for i in range(n):
    k = c01[i]
    if k in g01:
        dup_c01 += 1
        if g01[k] != c05[i]:
            eq_viol += 1
    else:
        g01[k] = c05[i]
report["e"]["n_duplicate_c01"] = int(dup_c01)
report["e"]["n_equal_c01_unequal_c05"] = int(eq_viol)
report["e"]["n_unique_c01"] = int(len(g01))
# fit quadratics
coef = np.polyfit(c01, c05, 2)
quad_resid = c05 - np.polyval(coef, c01)
report["e"]["quad_coef_desc"] = [float(x) for x in coef]
report["e"]["quad_max_abs_resid"] = float(np.max(np.abs(quad_resid)))
report["e"]["quad_rms_resid"] = float(np.sqrt(np.mean(quad_resid ** 2)))
# compare vs a piecewise/staircase alternative: c05 is a ramp in index?
coef_i = np.polyfit(np.arange(n), c05, 2)
resid_i = c05 - np.polyval(coef_i, np.arange(n))
report["e"]["quad_in_index_max_abs_resid"] = float(np.max(np.abs(resid_i)))
# is c05 exactly a linear function of rounded row index / a ramp in c01?
# check c05 vs a linear ramp of c01: residual
lin = np.polyfit(c01, c05, 1)
lin_resid = c05 - np.polyval(lin, c01)
report["e"]["lin_max_abs_resid"] = float(np.max(np.abs(lin_resid)))

# ================================================================ ATTACKS
att = {}

# --- A1: any pair not covered by (a)-(e) that is algebraically dependent?
# candidates: c04 vs c06, c04 vs c01, c07 vs geometry, c03 vs diff(c01), c05 vs c06/c01.
def dep_scan(x, y, name):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return None
    return {
        "pair": name, "n": int(len(x)),
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(pd.Series(x).corr(pd.Series(y), method="spearman")),
        "max_abs_diff": float(np.max(np.abs(x - y))) if len(x) == len(y) else None,
    }

att["A1_correlations"] = [
    dep_scan(c04, c06, "c04~c06"),
    dep_scan(c04, c01, "c04~c01"),
    dep_scan(c04, c05, "c04~c05"),
    dep_scan(c04, c03, "c04~c03"),
    dep_scan(c03, np.append(np.nan, np.diff(c01) * 100), "c03~100*diff(c01)"),
    dep_scan(c04, np.append(np.nan, np.diff(c04)), "c04~diff(c04)"),
]

# c04 repeated at equal c06 / equal c01?
def repeat_test(colkey, colval, kname, vname):
    d = {}
    key = np.round(colkey, 6)
    for k, v in zip(key, colval):
        d.setdefault(k, []).append(v)
    multi = {k: vs for k, vs in d.items() if len(vs) > 1}
    n_multi = len(multi)
    n_multi_with_variation = sum(1 for vs in multi.values() if len(set(vs)) > 1)
    return {"key": kname, "value": vname, "n_keys": int(len(d)),
            "n_keys_with_dupes": int(n_multi),
            "n_dupe_keys_with_differing_value": int(n_multi_with_variation)}

att["A1_repeat_c04_at_c06"] = repeat_test(c06, c04, "c06", "c04")
att["A1_repeat_c04_at_c01"] = repeat_test(c01, c04, "c01", "c04")
att["A1_repeat_c05_at_c06"] = repeat_test(c06, c05, "c06", "c05")

# c07 vs geometry: for the 15 labelled rows, print geometry context
lab = np.where(pd.notna(c07) & (c07 != ""))[0]
att["A1_c07_rows"] = [
    {"i": int(i), "c01": float(c01[i]), "c03": float(c03[i]), "c04": float(c04[i]),
     "c05": float(c05[i]), "c06": float(c06[i]), "label": str(c07[i])}
    for i in lab
]
# do labelled rows differ systematically in residual of (c) or (d)?
att["A1_c07_label_resid_c"] = {str(l): float(resid_c2[i]) for i, l in zip(lab, c07[lab])}
att["A1_c07_label_resid_d"] = {str(l): float(resid_d[i]) for i, l in zip(lab, c07[lab])}

# exact decimal-repeat check: is c04 a low-precision re-encoding of something?
# check #decimals of c04
raw = pd.read_csv(DATA, sep="\t", dtype=str, keep_default_na=False)  # raw text
def ndec(col):
    return [len(t.split(".")[1]) if "." in t else 0 for t in raw[col].tolist()]
att["A1_c04_decimals"] = {str(k): int(v) for k, v in pd.Series(ndec("c04_sulfate_ug_kg")).value_counts().items()}
att["A1_c06_decimals"] = {str(k): int(v) for k, v in pd.Series(ndec("c06_depth_w_e_m")).value_counts().items()}
att["A1_c05_decimals"] = {str(k): int(v) for k, v in pd.Series(ndec("c05_density_ice_g_cm_3")).value_counts().items()}

# --- A2: adversarial test of the "+1.008 m step"
# H1: single step. H2: smoothly varying thickness correction (fit c03 multiplier per block).
# H3: offset drifts. H4: c01 primitive, c03 derived.
a2 = {}
# H1 residual after single step, allowing an arbitrary constant offset (median)
r1 = resid_c2 - np.median(resid_c2)
a2["H1_single_step_max_abs_resid"] = float(np.max(np.abs(r1)))
# optional: best single-step with drift removed: fit resid_c = a + b*row before/after
idx = np.arange(n)
def fit_piecewise(resid, j):
    X = np.column_stack([np.ones(n), idx, (idx > j).astype(float)])
    coef, *_ = np.linalg.lstsq(X, resid, rcond=None)
    r = resid - X @ coef
    return coef, float(np.max(np.abs(r))), float(np.sqrt(np.mean(r ** 2)))
coefpw, maxpw, rmspw = fit_piecewise(resid_c, j)
a2["H3_linear_drift_plus_step"] = {"coef_1_idx_step": [float(x) for x in coefpw],
                                   "max_abs_resid": maxpw, "rms": rmspw}
# H2: smoothly varying thickness correction -> fit resid_c with a smoothing spline-like
#     piecewise-linear with K knots (K=10) and count sign changes of its slope
from numpy.polynomial import polynomial as P
for K in (2, 4, 8, 16):
    knots = np.linspace(0, n - 1, K)
    Xk = np.column_stack([np.ones(n)] + [np.maximum(0, idx - k) for k in knots[:-1]])
    ck, *_ = np.linalg.lstsq(Xk, resid_c, rcond=None)
    rk = resid_c - Xk @ ck
    a2[f"H2_piecewise_linear_{K}seg_max_abs_resid"] = float(np.max(np.abs(rk - np.median(rk))))
    a2[f"H2_piecewise_linear_{K}seg_rms"] = float(np.sqrt(np.mean((rk - np.median(rk)) ** 2)))
# H4: is c03 derivable from diff(c01)?  c03 = 100*(c01[i+1]-c01[i]) ?
d01 = np.diff(c01) * 100.0
a2["H4_n_distinct_diff_c01_cm"] = int(len(np.unique(np.round(d01, 9))))
a2["H4_n_distinct_c03"] = int(len(np.unique(np.round(c03, 9))))
res4 = c03[:-1] - d01
m4 = np.isfinite(res4)
a2["H4_c03_minus_100diff_c01"] = {
    "max_abs": float(np.max(np.abs(res4[m4]))),
    "median": float(np.median(res4[m4])),
    "frac_within_0.005": float(np.mean(np.abs(res4[m4]) < 0.005)),
    "frac_within_0.05": float(np.mean(np.abs(res4[m4]) < 0.05)),
}
# c03 unique-value structure: values grid?
u03 = np.unique(c03)
a2["H4_c03_min"] = float(u03.min()); a2["H4_c03_max"] = float(u03.max())
a2["H4_c03_gaps_unique"] = sorted(set(np.round(np.diff(u03), 9)))
# who is coarser? c03 has how many decimals
a2["H4_c03_decimals"] = {str(k): int(v) for k, v in pd.Series(ndec("c03_samp_thick_cm")).value_counts().items()}
a2["H4_c01_decimals"] = {str(k): int(v) for k, v in pd.Series(ndec("c01_depth_top_m")).value_counts().items()}
# alternative: c01[i+1]-c01[i] == c03[i+1]/100 exactly?
res4b = c03[1:] - d01
mb = np.isfinite(res4b)
a2["H4b_c03shift_minus_100diff"] = {
    "max_abs": float(np.max(np.abs(res4b[mb]))),
    "median": float(np.median(res4b[mb])),
    "frac_within_0.005": float(np.mean(np.abs(res4b[mb]) < 0.005)),
}
att["A2"] = a2

# --- A3: direction test c06 = c01*c05  vs  c05 = round(c06/c01,3)
a3 = {}
implied05 = c06 / c01
m = np.isfinite(implied05) & np.isfinite(c05) & (c01 > 0)
r3 = np.abs(implied05[m] - c05[m])
a3["dir1_c06_eq_c01c05"] = {
    "max_abs_resid": float(np.max(np.abs(resid_d))),
    "frac_within_0.0005": float(np.mean(np.abs(resid_d) < 0.0005)),
    "frac_within_0.001": float(np.mean(np.abs(resid_d) < 0.001)),
}
a3["dir2_c05_eq_round_c06_over_c01"] = {
    "max_abs_resid": float(np.max(r3)),
    "frac_within_0.0005": float(np.mean(r3 < 0.0005)),
    "frac_within_0.001": float(np.mean(r3 < 0.001)),
    "frac_within_1e-6": float(np.mean(r3 < 1e-6)),
}
# exact-rounding test: is c05 == round(c06/c01, 3) exactly under banker's/standard rounding?
round3 = np.round(implied05[m], 3)
a3["dir2_exact_round3_matches"] = int(np.sum(np.abs(round3 - c05[m]) < 1e-12))
a3["dir2_n"] = int(np.sum(m))
# symmetric test: is c01*c05 rounded to 2 dp equal to c06?
prod_r = np.round(c01 * c05, 2)
a3["dir1_round2_product_matches_c06"] = int(np.sum(np.abs(prod_r - c06) < 1e-12))
# what precision does product need? residual quantisation
a3["dir1_resid_unique_rounded"] = {str(k): int(v) for k, v in
                                   pd.Series(np.round(resid_d, 6)).value_counts().head(10).items()}
# Density of c06 grid
a3["c06_n_unique"] = int(len(np.unique(c06)))
a3["c06_min_gap"] = float(np.min(np.diff(np.unique(c06))))
att["A3"] = a3

# --- c04 NaN position
nanpos = np.where(~np.isfinite(c04))[0]
att["c04_nan_rows"] = [int(x) for x in nanpos]

# --- c07 label counts
lab_series = pd.Series([str(x) for x in c07 if str(x) != "" and str(x) != "nan"])
att["c07_counts"] = {str(k): int(v) for k, v in lab_series.value_counts().items()}

report["attacks"] = att

# ---------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(4, 2, figsize=(14, 16))
ax[0, 0].plot(c01, c02 - c01, ".", ms=2)
ax[0, 0].set_title("c02-c01 (m)  max=%.4f" % np.max(d12))
ax[0, 1].plot(dated, resid_b, "o-")
ax[0, 1].set_title("c09 - (1950-c08)/1000  max=%.2e" % np.max(np.abs(resid_b)))
ax[1, 0].plot(c01, resid_c, ".", ms=2)
ax[1, 0].set_title("c01 - cumsum(c03)/100   jump=%.4f at i=%d" % (dres[j], j))
ax[1, 1].plot(c01, resid_c2 - np.median(resid_c2), ".", ms=2)
ax[1, 1].set_title("after removing single jump, max=%.4f" % a2["H1_single_step_max_abs_resid"])
ax[2, 0].plot(c01, resid_d, ".", ms=2)
ax[2, 0].set_title("c06 - c01*c05  max=%.4f" % report["d"]["max_abs_resid_c06_minus_c01c05"])
ax[2, 1].plot(c01, quad_resid, ".", ms=2)
ax[2, 1].set_title("c05 - quad(c01)  max=%.2e" % report["e"]["quad_max_abs_resid"])
ax[3, 0].plot(c01, c05, ".", ms=2)
ax[3, 0].set_title("c05 vs c01 (%d distinct)" % report["e"]["n_unique_c05"])
ax[3, 1].hist(r3, bins=100)
ax[3, 1].set_title("|c06/c01 - c05|  frac<0.0005=%.4f" % a3["dir2_c05_eq_round_c06_over_c01"]["frac_within_0.0005"])
plt.tight_layout()
plt.savefig(f"{OUT}/v10_draft1_independent.png", dpi=90)
plt.close()

with open(f"{OUT}/v10_draft1_independent.json", "w") as f:
    json.dump(report, f, indent=1, default=str)

# ---------------------------------------------------------------- print
def p(*a):
    print(*a)

p("rows:", n)
p("(a) max|c02-c01| = %.6f ; n>0.02 = %d ; uniq diffs = %s" %
  (report["a"]["max_abs_c02_minus_c01"], report["a"]["n_gt_0.02"], report["a"]["diff_values"]))
p("(b) max|c09-(1950-c08)/1000| = %.3e on %d rows" %
  (report["b"]["max_abs_resid"], report["b"]["n_dated"]))
p("(c) cumsum(c03)/100 residual: first=%.5f last=%.5f jump_i=%d jump=%.5f" %
  (report["c"]["resid_first"], report["c"]["resid_last"], j, dres[j]))
p("    c01 around jump:", report["c"]["c01_around_jump"])
p("    after removing jump: max=%.5f (center %.5f); n_nonzero_incr=%d" %
  (a2["H1_single_step_max_abs_resid"], report["c"]["resid_after_removing_biggest_jump_center"],
   report["c"]["n_nonzero_resid_increments"]))
p("    top10 |increments|:", report["c"]["top10_abs_increments"])
p("(d) max|c06-c01*c05| = %.6f ; span = %.2f ; ratio = %.6f" %
  (report["d"]["max_abs_resid_c06_minus_c01c05"], report["c"]["span"],
   report["d"]["ratio_max_abs_over_span"]))
p("(e) n_unique_c05 = %d ; monotone_rows=%s ; neg_steps=%d pos_steps=%d ; c05 range [%.3f,%.3f]" %
  (report["e"]["n_unique_c05"], report["e"]["monotone_nondecreasing_rows"],
   report["e"]["n_row_negative_steps"], report["e"]["n_row_positive_steps"],
   report["e"]["c05_min"], report["e"]["c05_max"]))
p("    equal-c01-unequal-c05: dup c01 = %d, violations = %d, unique c01 = %d" %
  (report["e"]["n_duplicate_c01"], report["e"]["n_equal_c01_unequal_c05"], report["e"]["n_unique_c01"]))
p("    quad(c01) max resid = %.3e ; lin max resid = %.3e ; quad(index) max = %.3e" %
  (report["e"]["quad_max_abs_resid"], report["e"]["lin_max_abs_resid"], report["e"]["quad_in_index_max_abs_resid"]))
p("    quad coef (desc):", report["e"]["quad_coef_desc"])
p("--- ATTACK A1 ---")
for r in att["A1_correlations"]:
    p("   ", r)
p("   repeat c04@c06:", att["A1_repeat_c04_at_c06"])
p("   repeat c04@c01:", att["A1_repeat_c04_at_c01"])
p("   repeat c05@c06:", att["A1_repeat_c05_at_c06"])
p("   c04 decimals:", att["A1_c04_decimals"], " c06 decimals:", att["A1_c06_decimals"])
p("   c05 decimals:", att["A1_c05_decimals"])
p("   c04 NaN rows:", att["c04_nan_rows"])
p("   c07 counts:", att["c07_counts"])
p("--- ATTACK A2 ---")
for k, v in a2.items():
    p("   ", k, "=", v)
p("--- ATTACK A3 ---")
for k, v in a3.items():
    p("   ", k, "=", v)