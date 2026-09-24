#!/usr/bin/env python3
"""Independent verification of Draft Finding 1 (claim under test).

Written from scratch; does NOT read or reuse anything in output/analysis/.
Reads only data/observations.tsv.  Writes outputs next to this script.
"""
import itertools
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

PROJ = "/data/user/conversations/conv-f07721cb-d3dc-42e9-83c7-7e986203115b/project"
OUT = os.path.join(PROJ, "output", "verification", "figures")
DATA = os.path.join(PROJ, "data", "observations.tsv")

report = {}
lines = []


def say(s=""):
    print(s)
    lines.append(str(s))


def rec(k, v):
    report[k] = v
    say(f"{k} = {v}")


raw = pd.read_csv(DATA, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
names = list(raw.columns)
c = {}
for i, nm in enumerate(names):
    key = f"c{i+1:02d}"
    c[key] = pd.to_numeric(raw[nm], errors="coerce") if key not in ("c07",) else raw[nm]
c07 = raw[names[6]].replace("", np.nan)
N = len(raw)
rec("n_rows", N)
rec("columns", names)

# ---------------------------------------------------------------- (a) c02 vs c01
d21 = (c["c02"] - c["c01"]).values
rec("a__max_abs_c02_minus_c01", float(np.abs(d21).max()))
rec("a__all_within_0.02", bool(np.all(np.abs(d21) <= 0.02)))
rec("a__offset_counter", dict(Counter(np.round(d21, 4)).most_common()))
rec("a__n_rows_where_c02_equals_c01_exactly", int(np.sum(d21 == 0)))
rec("a__n_rows_with_nonzero_offset", int(np.sum(d21 != 0)))

# ---------------------------------------------------------------- (b) c09 vs (1950-c08)/1000
m = c["c08"].notna() & c["c09"].notna()
rec("b__n_dated_rows", int(m.sum()))
pred9 = (1950.0 - c["c08"][m]) / 1000.0
res9 = (c["c09"][m] - pred9).values
rec("b__max_abs_residual", float(np.abs(res9).max()))
rec("b__all_exact_to_1e-12", bool(np.all(np.abs(res9) < 1e-12)))
rec("b__dated_rows_c01", [float(x) for x in c["c01"][m]])
rec("b__dated_rows_c08", [float(x) for x in c["c08"][m]])
rec("b__dated_rows_c09", [float(x) for x in c["c09"][m]])
rec("b__dated_rows_c07", [str(x) for x in c07[m]])
# also check alternative forms to see if any is even closer / different
alt = (1950.0 - c["c08"][m]) / 1000.0
rec("b__alt_expression_identical", bool(np.all(res9 == (c["c09"][m] - alt).values)))

# ---------------------------------------------------------------- (c) c01 vs cumsum(c03)/100
cum = np.cumsum(c["c03"].values) / 100.0
r = c["c01"].values - cum
rec("c__raw_residual_min", float(r.min()))
rec("c__raw_residual_max", float(r.max()))
step_pos = int(np.argmax(np.abs(np.diff(r))) + 0)
dstep = np.diff(r)
rec("c__largest_residual_jump", float(dstep[np.argmax(np.abs(dstep))]))
rec("c__largest_residual_jump_index", int(np.argmax(np.abs(dstep))))
# best-fit constant step: shift cumulative by best constant
r_shift = r - np.median(r)
rec("c__median_shifted_residual_absmax", float(np.abs(r_shift).max()))
# explicit step model: role +1.008 after row index 105 (0-based diff index 105 == between rows 106/107)
def step_model(k):
    v = np.where(np.arange(N) > k, 1.0 * 0 + 1.0, 0.0) * k
    return v
for k_knot in (104, 105, 106):
    pass
k = 105  # diff index 105 -> rows 0..105 have no step, rows 106.. have step
stepvec = np.zeros(N)
stepvec[k + 1:] = 1.0
base = c["c01"].values - stepvec * 1.008
r2 = c["c01"].values - (cum + stepvec * 1.008)
rec("c__step_model_knot_diff_index", k)
rec("c__step_model_residual_absmax", float(np.abs(r2).max()))
rec("c__step_model_residual_min", float(r2.min()))
rec("c__step_model_residual_max", float(r2.max()))
# refine best step size by least squares on that model
best = {}
for kk in range(95, 116):
    sv = np.zeros(N)
    sv[kk + 1:] = 1.0
    # solve min ||c01 - (cum + sv*s)||^2 over s
    t = c["c01"].values - cum
    s = np.dot(t, sv) / np.dot(sv, sv)
    res = c["c01"].values - (cum + sv * s)
    best[kk] = (float(np.abs(res).max()), float(s))
bestk = min(best, key=lambda z: best[z][0])
rec("c__best_knot_and_step", {"knot_diff_index": bestk, "step": best[bestk][1],
                              "residual_absmax": best[bestk][0]})
rec("c__c01_total_span", float(c["c01"].max() - c["c01"].min()))
rec("c__residual_absmax_over_span_frac", float(np.abs(r2).max() / (c["c01"].max() - c["c01"].min())))

# c03 quantisation grid: is c03 on a 0.01 grid?  Does c01 = round(cum,2)?
grid = np.abs(c["c03"].values * 100 - np.round(c["c03"].values * 100)).max()
rec("c__c03_on_0.01_grid_maxdev", float(grid))
rec("c__c01_equals_round_cum_2dp", int(np.sum(np.abs(c["c01"].values - np.round(cum, 2)) < 1e-9)))
rec("c__n_rows", N)
rec("c__c01_on_0.01_grid_maxdev", float(np.abs(c["c01"].values * 100 - np.round(c["c01"].values * 100)).max()))

# ---------------------------------------------------------------- (d) c06 = c01*c05
prod = c["c01"].values * c["c05"].values
rd = c["c06"].values - prod
rec("d__max_abs_residual", float(np.abs(rd).max()))
rec("d__mean_abs_residual", float(np.abs(rd).mean()))
rec("d__frac_within_0.001", float(np.mean(np.abs(rd) < 0.001)))
rec("d__frac_within_0.005", float(np.mean(np.abs(rd) < 0.005)))
rec("d__residual_sign_counter", dict(Counter(np.round(rd, 3)).most_common(8)))
# alternative: c06 = round(c01*c05, 2)
rec("d__alt_c06_round2_absmax", float(np.abs(c["c06"].values - np.round(prod, 2)).max()))
rec("d__alt_c06_round2_frac_exact", float(np.mean(np.abs(c["c06"].values - np.round(prod, 2)) < 1e-9)))
# reverse direction: c05 = round(c06/c01,3)
mask = c["c01"].values > 0
ratio = c["c06"].values[mask] / c["c01"].values[mask]
rev = c["c05"].values[mask] - np.round(ratio, 3)
rec("d__reverse_c05_from_ratio_frac_within_0.0005", float(np.mean(np.abs(rev) <= 0.0005)))
rec("d__reverse_c05_from_ratio_frac_within_0.001", float(np.mean(np.abs(rev) <= 0.001)))
rec("d__reverse_c05_from_ratio_absmax", float(np.abs(rev).max()))
fwd = np.abs(rd[mask])
rec("d__forward_frac_within_0.0005", float(np.mean(fwd <= 0.0005)))
rec("d__forward_frac_within_0.005", float(np.mean(fwd <= 0.005)))
rec("d__n_rows_c01_gt_0", int(mask.sum()))
# if c05 were round(c06/c01,3) then c06 would often not equal c01*c05 exactly for c01 large
# quantify: mismatching rows under reverse hypothesis
rec("d__reverse_n_rows_disagreeing_gt0.0005", int(np.sum(np.abs(rev) > 0.0005)))

# ---------------------------------------------------------------- (e) c05 ramp
u = np.sort(c["c05"].dropna().unique())
rec("e__n_distinct_c05_exact", int(len(u)))
rec("e__c05_min", float(u.min()))
rec("e__c05_max", float(u.max()))
rec("e__c05_min_step", float(np.diff(u).min()))
rec("e__c05_median_step", float(np.median(np.diff(u))))
rec("e__c05_on_0.001_grid_maxdev", float(np.abs(c["c05"].values * 1000 - np.round(c["c05"].values * 1000)).max()))
# monotone in c01?
order = np.argsort(c["c01"].values, kind="mergesort")
c01s = c["c01"].values[order]
c05s = c["c05"].values[order]
rec("e__c05_nonincreasing_steps_along_c01", int(np.sum(np.diff(c05s) < 0)))
rec("e__c05_decreasing_steps_absmax", float(np.abs(np.diff(c05s)).min() if len(c05s) else 0))
# equal c01 => equal c05?
agg = pd.DataFrame({"c01": c["c01"].values, "c05": c["c05"].values}).groupby("c01")["c05"].nunique()
rec("e__max_distinct_c05_per_c01", int(agg.max()))
rec("e__n_c01_with_multiple_c05", int((agg > 1).sum()))
# quadratic fit
A = np.vstack([c["c01"].values ** 2, c["c01"].values, np.ones(N)]).T
coef, *_ = np.linalg.lstsq(A, c["c05"].values, rcond=None)
qres = c["c05"].values - A @ coef
rec("e__quad_coef_a_b_c", [float(x) for x in coef])
rec("e__quad_residual_absmax", float(np.abs(qres).max()))
rec("e__quad_residual_rms", float(np.sqrt(np.mean(qres ** 2))))
# compare with linear and cubic
for deg in (1, 3, 4):
    Cf = np.polyfit(c["c01"].values, c["c05"].values, deg)
    rr = c["c05"].values - np.polyval(Cf, c["c01"].values)
    rec(f"e__poly{deg}_residual_absmax", float(np.abs(rr).max()))
# residual of quadratic vs c01 pattern
rec("e__quad_resid_sign_frac_pos", float(np.mean(qres > 0)))
# does c05 look like round(a + b*c01 + cc*c01^2, 3)?
qf = A @ coef
rec("e__quadpred_round3_exact_frac", float(np.mean(np.abs(c["c05"].values - np.round(qf, 3)) < 1e-9)))

# ---------------------------------------------------------------- ATTACK 1: pairwise deps beyond (a)-(e)
say("\n--- pairwise attack: for each ordered pair, can col Y be a function of col X alone? ---")
nums = ["c01", "c02", "c03", "c04", "c05", "c06", "c08", "c09"]
pair_res = {}
for x, y in itertools.permutations(nums, 2):
    vx = c[x].values
    vy = c[y].values
    ok = np.isfinite(vx) & np.isfinite(vy)
    if ok.sum() < 3:
        pair_res[f"{y}<-{x}"] = "insufficient"
        continue
    dfp = pd.DataFrame({"x": vx[ok], "y": vy[ok]})
    g = dfp.groupby("x")["y"].nunique()
    n_multi = int((g > 1).sum())
    pair_res[f"{y}<-{x}"] = {"n_vals_x": int(len(g)), "n_x_with_multiple_y": n_multi,
                             "max_distinct_y_per_x": int(g.max())}
report["attack1_pair_functionality"] = pair_res
for kk in sorted(pair_res):
    say(f"  {kk}: {pair_res[kk]}")

# specific: c04 repeated at equal c06 / equal c01?
say("\n--- attack1: is c04 a function of geometry? ---")
for x in ("c01", "c02", "c05", "c06", "c03"):
    ok = np.isfinite(c[x].values) & np.isfinite(c["c04"].values)
    dfp = pd.DataFrame({"x": c[x].values[ok], "y": c["c04"].values[ok]})
    g = dfp.groupby("x")["y"].nunique()
    rec(f"attack1__c04_per_{x}_x_with_multiple", int((g > 1).sum()))
    rec(f"attack1__c04_per_{x}_n_x", int(len(g)))
# c04 value repeats: if two different c01 have same c04 that is normal; check exact repeats of c04
vc = c["c04"].dropna()
cnt = vc.value_counts()
rec("attack1__c04_duplicate_values_count", int((cnt > 1).sum()))
# c05 determined by c01?
agg2 = pd.DataFrame({"c01": c["c01"].values, "c05": c["c05"].values}).groupby("c01")["c05"].nunique()
rec("attack1__c05_not_determined_by_c01", int((agg2 > 1).sum()))
# c05 determined by c06/c01 rounded?
ratio_all = np.where(c["c01"].values > 0, c["c06"].values / np.where(c["c01"].values > 0, c["c01"], np.nan), np.nan)
ok = np.isfinite(ratio_all)
dfp = pd.DataFrame({"r": np.round(ratio_all[ok], 3), "y": c["c05"].values[ok]})
g = dfp.groupby("r")["y"].nunique()
rec("attack1__c05_per_round3_ratio_x_with_multiple", int((g > 1).sum()))
# c08/c09 relationship to geometry?
for x in ("c01", "c06"):
    ok = np.isfinite(c[x].values) & c["c08"].notna().values
    g = pd.DataFrame({"x": c[x].values[ok], "y": c["c08"].values[ok]}).groupby("x")["y"].nunique()
    rec(f"attack1__c08_per_{x}_x_with_multiple", int((g > 1).sum()))
# c07 vs anything
rec("attack1__c07_labels", dict(Counter([str(v) for v in c07 if pd.notna(v)])))

# ---------------------------------------------------------------- ATTACK 2: step alternatives
say("\n--- attack2: alternative explanations for the +1.008 step ---")
dd = np.diff(c["c01"].values)
rec("attack2__diff_value_counts", dict(Counter(np.round(dd, 4)).most_common(20)))
# (i) smooth varying thickness correction: cumulative residual r vs c01 -- is it constant (+ step)?
# Fit r = a + b*c01 (linear drift) instead of piecewise constant
A2 = np.vstack([np.ones(N), c["c01"].values]).T
coef2, *_ = np.linalg.lstsq(A2, r, rcond=None)
rlin = r - A2 @ coef2
rec("attack2__linear_drift_coefs_intercept_slope", [float(x) for x in coef2])
rec("attack2__linear_drift_residual_absmax", float(np.abs(rlin).max()))
# quantile spread of r before/after the knot
pre = r[:106]
post = r[106:]
rec("attack2__r_pre_step_minmax", [float(pre.min()), float(pre.max())])
rec("attack2__r_post_step_minmax", [float(post.min()), float(post.max())])
rec("attack2__r_pre_std", float(pre.std()))
rec("attack2__r_post_std", float(post.std()))
rec("attack2__r_pre_mean", float(pre.mean()))
rec("attack2__r_post_mean", float(post.mean()))
# (iii) reverse direction: c03 derived from diff(c01)
d01 = np.diff(c["c01"].values) * 100.0
c03_head = c["c03"].values[:-1]
rdir = d01 - c03_head
rec("attack2__diffc01_vs_c03_head_absmax", float(np.abs(rdir).max()))
rec("attack2__diffc01_vs_c03_head_frac_within_0.01", float(np.mean(np.abs(rdir) <= 0.01)))
rec("attack2__diffc01_vs_c03_head_frac_exact", float(np.mean(np.abs(rdir) < 1e-9)))
# if c03 were diff-derived, c03 = round(diff(c01)*100,?) -> check c03 vs diff of the step-model base
base01 = c["c01"].values - np.where(np.arange(N) > 105, 1.008, 0.0)
d01b = np.diff(base01) * 100.0
rec("attack2__diffbase01_vs_c03head_absmax", float(np.abs(d01b - c03_head).max()))
rec("attack2__diffbase01_vs_c03head_exact_frac", float(np.mean(np.abs(d01b - c03_head) < 1e-9)))
# distribution of c03 vs diff(c01): how many diffs are not near any c03 value?
rec("attack2__c03_unique_count", int(pd.Series(c["c03"].values).nunique()))
rec("attack2__diffc01_100_unique_count", int(pd.Series(np.round(d01, 4)).nunique()))
rec("attack2__diffc01_100_value_counts_top", dict(Counter(np.round(d01, 4)).most_common(10)))
# is c03 on 0.02-exclusive grid? check pairwise differences of c03
u3 = np.sort(pd.unique(c["c03"].values))
rec("attack2__c03_unique_sorted", [float(x) for x in u3])
rec("attack2__c03_min_pair_gap", float(np.diff(u3).min()))
rec("attack2__c03_all_multiples_of_0.01", bool(np.all(np.abs(u3 * 100 - np.round(u3 * 100)) < 1e-9)))
rec("attack2__c03_all_multiples_of_0.05", bool(np.all(np.abs(u3 * 20 - np.round(u3 * 20)) < 1e-9)))

# ---------------------------------------------------------------- summary
say("\n=== SUMMARY OF KEY NUMBERS ===")
for kk in report:
    if kk.startswith(("a__", "b__", "c__", "d__", "e__")):
        say(f"{kk} = {report[kk]}")

with open(os.path.join(OUT, "verify_draft1_report.json"), "w") as f:
    json.dump(report, f, indent=2, default=str)
with open(os.path.join(OUT, "verify_draft1_log.txt"), "w") as f:
    f.write("\n".join(lines))
print("\nWROTE", os.path.join(OUT, "verify_draft1_report.json"))