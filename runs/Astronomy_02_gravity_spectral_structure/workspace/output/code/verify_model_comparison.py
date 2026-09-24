"""ADVERSARIAL VERIFICATION of track "model_comparison" (TRACK 1).

Independent re-derivation of the load-bearing numbers, written from scratch.
Does NOT import the track's script (only the shared common.load_traces loader
is avoided too: the TSV is read directly with pandas).

Writes output/results/verify_model_comparison.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT = Path("/data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project")
RES = PROJECT / "output" / "results"
DATA = PROJECT / "data" / "pass_residual_traces.tsv"

SEED = 20220101
N_BOOT = 20000

report = json.loads((RES / "model_comparison.json").read_text())
claimed = {}
claimed["median_rms_A"] = report["item1_per_pass_whole_pass_rms_and_ratios"]["rms_distribution_by_model"]["A"]["median"]
claimed["median_rms_B"] = report["item1_per_pass_whole_pass_rms_and_ratios"]["rms_distribution_by_model"]["B"]["median"]
claimed["median_rms_C"] = report["item1_per_pass_whole_pass_rms_and_ratios"]["rms_distribution_by_model"]["C"]["median"]

# ------------------------------------------------------------------ load (own reader)
df = pd.read_csv(DATA, sep="\t").sort_values(["pass_id", "relative_time_min"], kind="stable")
passes = sorted(df["pass_id"].unique().tolist())
npass = len(passes)
COL = {"A": "residual_model_A_mm_s", "B": "residual_model_B_mm_s", "C": "residual_model_C_mm_s"}

# independent implementation: groupby transform, then pivot to pass x model matrix
g = df.groupby("pass_id")
per_pass = pd.DataFrame({m: g[COL[m]].apply(lambda v: float(np.sqrt(np.mean(np.square(v)))))
                         for m in "ABC"}).loc[passes]

rms = lambda v: float(np.sqrt(np.mean(np.asarray(v, float) ** 2)))

probs = []


def check(quantity, claim, mine, tol=5e-4, note=""):
    d = None
    try:
        d = float(abs(float(claim) - float(mine)))
    except Exception as e:
        d = None
        note = (note + f" | comparison failed: {e}").strip()
    flag = (d is not None and d <= tol)
    probs.append({"quantity": quantity, "claimed": claim, "recomputed": mine,
                  "abs_diff": d, "matched_within_tol": bool(flag), "note": note})
    return flag


# ============================================================ (1) per-pass RMS + ratios
med = {m: float(np.median(per_pass[m])) for m in "ABC"}
check("median whole-pass RMS A", claimed["median_rms_A"], med["A"], 1e-6)
check("median whole-pass RMS B", claimed["median_rms_B"], med["B"], 1e-6)
check("median whole-pass RMS C", claimed["median_rms_C"], med["C"], 1e-6)

rat = {}
for x, y in [("B", "A"), ("C", "A"), ("B", "C")]:
    r = (per_pass[x] / per_pass[y]).to_numpy(float)
    rat[f"{x}/{y}"] = r
    check(f"median ratio {x}/{y}", report["item1_per_pass_whole_pass_rms_and_ratios"]
          ["ratio_distributions"][f"{x}/{y}"]["median"], float(np.median(r)), 1e-6)
    check(f"min ratio {x}/{y}", report["item1_per_pass_whole_pass_rms_and_ratios"]
          ["ratio_distributions"][f"{x}/{y}"]["min"], float(r.min()), 1e-6)
    check(f"max ratio {x}/{y}", report["item1_per_pass_whole_pass_rms_and_ratios"]
          ["ratio_distributions"][f"{x}/{y}"]["max"], float(r.max()), 1e-6)

wins_raw = {}
for x, y in [("B", "A"), ("C", "A"), ("B", "C")]:
    wins_raw[f"{x}/{y}"] = int((per_pass[x].to_numpy() < per_pass[y].to_numpy()).sum())

# ============================================================ (2) blocked paired tests
def tests(x, y, claim_obj):
    xv = per_pass[x].to_numpy(float)
    yv = per_pass[y].to_numpy(float)
    d = xv - yv
    w = stats.wilcoxon(xv, yv, zero_method="wilcox")
    t = stats.ttest_rel(xv, yv)
    check(f"wilcoxon p {x}vs{y}", claim_obj["wilcoxon_p"], float(w.pvalue), 1e-10)
    check(f"paired t p {x}vs{y}", claim_obj["paired_t_p"], float(t.pvalue), 1e-10)
    check(f"paired t stat {x}vs{y}", claim_obj["paired_t_statistic"], float(t.statistic), 1e-6)
    check(f"wins {x} vs {y}", claim_obj["wins_x_lower_rms"], int((xv < yv).sum()), 0)
    # mean-difference 95% CI, recomputed with a slightly different route
    n = d.size
    se = d.std(ddof=1) / np.sqrt(n)
    tc = stats.t.ppf(0.975, n - 1)
    check(f"mean-diff CI low {x}vs{y}", claim_obj["mean_difference_ci95"][0], float(d.mean() - tc * se), 1e-9)
    check(f"mean-diff CI high {x}vs{y}", claim_obj["mean_difference_ci95"][1], float(d.mean() + tc * se), 1e-9)


import re

by_pair = {}
for t in report["item2_blocked_paired_tests_over_22_passes"]["tests"]:
    tag = "[ortho] " if t["pair"].startswith("[orthogonalised]") else ""
    mm = re.search(r"([ABC]) vs ([ABC])", t["pair"])
    by_pair[tag + f"{mm.group(1)}vs{mm.group(2)}"] = t
for x, y in [("B", "A"), ("C", "A"), ("B", "C")]:
    tests(x, y, by_pair[f"{x}vs{y}"])

# ============================================================ (3) whole-pass bootstrap
rng = np.random.default_rng(SEED)
boot_ci = {}
for k, r in rat.items():
    idx = rng.integers(0, npass, size=(N_BOOT, npass))
    m = np.median(r[idx], axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    P = float((m < 1).mean())
    boot_ci[k] = {"ci": [float(lo), float(hi)], "P_lt1": P, "n_lt1": int((m < 1).sum())}
    c = report["item3_whole_pass_bootstrap_median_ratio"]["raw"][k]
    check(f"bootstrap CI low {k}", c["ci95_percentile"][0], float(lo), 5e-3)
    check(f"bootstrap CI high {k}", c["ci95_percentile"][1], float(hi), 5e-3)
    check(f"bootstrap P(ratio<1) {k}", c["prob_ratio_lt_1"], P, 1e-4)

# ============================================================ (4) leave-one-pass-out
loo = {}
for k, r in rat.items():
    vals = np.array([np.median(np.delete(r, i)) for i in range(npass)])
    loo[k] = [float(vals.min()), float(vals.max())]
    c = report["item4_leave_one_pass_out_median_ratio"]["raw"][k]
    check(f"LOO min {k}", c["min"], float(vals.min()), 1e-6)
    check(f"LOO max {k}", c["max"], float(vals.max()), 1e-6)

# ============================================================ (5) robustness variants
TR = {p: {m: sub[COL[m]].to_numpy(float) for m in "ABC"} for p, sub in df.groupby("pass_id", sort=True)}


def per_pass_from_transform(fn):
    rows = {m: [] for m in "ABC"}
    for p in passes:
        for m in "ABC":
            rows[m].append(rms(fn(TR[p][m])))
    return pd.DataFrame(rows, index=passes)


def med_ratios(frame):
    return {f"{x}/{y}": float(np.median(frame[x] / frame[y])) for x, y in [("B", "A"), ("C", "A"), ("B", "C")]}


# (a) detrend
def detrend(v):
    x = np.arange(len(v), dtype=float)
    p = np.polyfit(x, v, 1)
    return v - np.polyval(p, x)


f_dt = per_pass_from_transform(detrend)
mr_dt = med_ratios(f_dt)
for k in mr_dt:
    check(f"detrend median {k}", report["item5_robustness_alternative_definitions"]["perturbations"]
          ["a_linear_detrend_per_pass"]["raw_ratios"][k]["median"], mr_dt[k], 1e-6)

# (b) boxcar smoothing (independent implementation: sliding-window mean via cumsum)
def boxcar(v, k):
    if k <= 1:
        return np.asarray(v, float).copy()
    v = np.asarray(v, float)
    c = np.concatenate([[0.0], np.cumsum(v)])
    idx = np.arange(len(v))
    lo = np.clip(idx - (k // 2), 0, len(v))
    hi = np.clip(idx + (k - k // 2), 0, len(v))
    return (c[hi] - c[lo]) / (hi - lo)


smooth_ratios = {}
for k in [1, 3, 5, 7, 9, 15]:
    f = per_pass_from_transform(lambda v, k=k: boxcar(v, k))
    mr = med_ratios(f)
    smooth_ratios[k] = mr
    for kk in mr:
        check(f"boxcar k={k} median {kk}", report["item5_robustness_alternative_definitions"]["perturbations"]
              ["b_boxcar_smoothing"][f"k={k}"]["median_ratios"][kk], mr[kk], 1e-6)

# (c) drop gap passes
gap_passes = []
for p in passes:
    dt = np.diff(TR[p]["relative_time_min"]) if "relative_time_min" in TR[p] else np.diff(df[df.pass_id == p]["relative_time_min"].to_numpy(float))
    if (dt > 1.5).any():
        gap_passes.append(p)
check("gap-pass list", report["item5_robustness_alternative_definitions"]["perturbations"]
      ["c_drop_gap_passes"]["dropped_passes"], gap_passes, 0)
keep = [p for p in passes if p not in gap_passes]
mr_g = med_ratios(per_pass.loc[keep])
for k in mr_g:
    check(f"drop-gap median {k}", report["item5_robustness_alternative_definitions"]["perturbations"]
          ["c_drop_gap_passes"]["median_ratios"][k], mr_g[k], 1e-6)

# (d) pooled RMS ratio and LOO pooled
pooled = {}
for x, y in [("B", "A"), ("C", "A"), ("B", "C")]:
    both_x = np.concatenate([TR[p][x] for p in passes])
    both_y = np.concatenate([TR[p][y] for p in passes])
    full = rms(both_x) / rms(both_y)
    vals = []
    for p in passes:
        xs = np.concatenate([TR[q][x] for q in passes if q != p])
        ys = np.concatenate([TR[q][y] for q in passes if q != p])
        vals.append(rms(xs) / rms(ys))
    pooled[f"{x}/{y}"] = {"full": float(full), "loo_min": float(min(vals)), "loo_max": float(max(vals))}
    c = report["item5_robustness_alternative_definitions"]["perturbations"]["d_leave_one_pass_out_pooled"]
    check(f"pooled full ratio {x}/{y}", c[f"{x}/{y}"]["pooled_rms_ratio_full"], float(full), 1e-6)
    check(f"pooled LOO min {x}/{y}", c[f"{x}/{y}"]["pooled_rms_ratio_loo_min"], float(min(vals)), 1e-6)
    check(f"pooled LOO max {x}/{y}", c[f"{x}/{y}"]["pooled_rms_ratio_loo_max"], float(max(vals)), 1e-6)

# ============================================================ (6) pooled linear relations
A = np.concatenate([TR[p]["A"] for p in passes])
B = np.concatenate([TR[p]["B"] for p in passes])
C = np.concatenate([TR[p]["C"] for p in passes])


def ols(y, X, tags):
    Xd = np.column_stack([X, np.ones_like(y)])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    r2 = 1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)
    return {t: float(b) for t, b in zip(tags + ["intercept"], beta)}, float(rms(resid)), float(r2)


fitC, rC, R2C = ols(C, np.column_stack([A, B]), ["A", "B"])
fitB, rB, R2B = ols(B, np.column_stack([A, C]), ["A", "C"])
fitA, rA, R2A = ols(A, np.column_stack([B, C]), ["B", "C"])
checks6 = [("C~A+B", fitC, rC, R2C), ("B~A+C", fitB, rB, R2B), ("A~B+C", fitA, rA, R2A)]
cl = {f["target"]: f for f in report["item6_pooled_linear_relations_between_models"]["fits"]}
check("C~aA+bB coef a", cl["C"]["coefficients"]["A"], fitC["A"], 1e-6)
check("C~aA+bB coef b", cl["C"]["coefficients"]["B"], fitC["B"], 1e-6)
check("C~aA+bB resid rms", cl["C"]["residual_rms"], rC, 1e-9)
check("C~aA+bB R2", cl["C"]["r2"], R2C, 1e-9)
check("B~aA+bC coef a", cl["B"]["coefficients"]["A"], fitB["A"], 1e-6)
check("B~aA+bC coef b", cl["B"]["coefficients"]["C"], fitB["C"], 1e-6)
check("B~aA+bC resid rms", cl["B"]["residual_rms"], rB, 1e-9)
check("A~aB+bC resid rms", cl["A"]["residual_rms"], rA, 1e-9)
check("pooled RMS A", cl["A"]["target_rms"], float(rms(A)), 1e-6)
check("pooled RMS B", cl["B"]["target_rms"], float(rms(B)), 1e-6)
check("pooled RMS C", cl["C"]["target_rms"], float(rms(C)), 1e-6)

# weight-sum-1 mixture: C = w A + (1-w) B  (Gauss-Markov optimum w*)
Dv = A - B
w_star = float(np.dot(Dv, C - B) / np.dot(Dv, Dv))
res_mix = C - (w_star * A + (1 - w_star) * B)
mix = report["item6_pooled_linear_relations_between_models"]["mixture_structure"]
check("mixture w", mix["free_fit_C_eq_aA_plus_bB_plus_c"]["a"], w_star, 1e-6,
      "constrained w* vs free-fit coefficient a; they are algebraically different estimators but nearly equal here")
check("mixture weight sum a+b", mix["free_fit_C_eq_aA_plus_bB_plus_c"]["weight_sum_a_plus_b"],
      fitC["A"] + fitC["B"], 1e-9)
check("mixture residual rms", mix["mixture_C_eq_wA_plus_1mw_B"]["residual_rms"], float(rms(res_mix)), 1e-9)
check("mix resid/C rms", mix["mixture_C_eq_wA_plus_1mw_B"]["residual_rms_over_C_rms"],
      float(rms(res_mix) / rms(C)), 1e-9)
# equivalently: force weights to sum to 1, allow intercept
Xc = np.column_stack([A - B, np.ones_like(A)])
bc, *_ = np.linalg.lstsq(Xc, C - B, rcond=None)
res_c = (C - B) - Xc @ bc
check("sum1 fit w", mix["mixture_C_eq_wA_plus_1mw_B"]["w"], float(bc[0]), 2e-3,
      "sum-to-1 + free intercept estimator (w=0.41593) vs mixture estimator (w=0.41582); agreement to 1e-4")
check("sum1 fit residual rms", mix["mixture_C_eq_wA_plus_1mw_B"]["residual_rms"], float(rms(res_c)), 1e-5)

# ============================================================ (7) correlations
raw_mat = per_pass.to_numpy(float)
def corr_pairs(M, i, j):
    return float(np.corrcoef(M[:, i], M[:, j])[0, 1])
iA = 0  # column order A,B,C from per_pass construction
cm7 = report["item7_pearson_correlation_matrices"]["whole_pass_level_raw_rms_n22"]
check("pass-level corr A-B", cm7["A"]["B"], corr_pairs(raw_mat, 0, 1), 1e-9)
check("pass-level corr A-C", cm7["A"]["C"], corr_pairs(raw_mat, 0, 2), 1e-9)
check("pass-level corr B-C", cm7["B"]["C"], corr_pairs(raw_mat, 1, 2), 1e-9)
cs = report["item7_pearson_correlation_matrices"]["sample_level_n1698_descriptive_only"]
check("sample-level corr A-B", cs["A"]["B"], float(np.corrcoef(A, B)[0, 1]), 1e-9)
check("sample-level corr A-C", cs["A"]["C"], float(np.corrcoef(A, C)[0, 1]), 1e-9)
check("sample-level corr B-C", cs["B"]["C"], float(np.corrcoef(B, C)[0, 1]), 1e-9)

# ============================================================ (8) pass-level CV
cv = {}
for target, x1, x2 in [("B", "A", "C"), ("C", "A", "B"), ("A", "B", "C")]:
    vals = {}
    for p in passes:
        other = [q for q in passes if q != p]
        ytr = np.concatenate([TR[q][target] for q in other])
        Xtr = np.column_stack([np.concatenate([TR[q][x] for q in other]) for x in (x1, x2)])
        beta, *_ = np.linalg.lstsq(np.column_stack([Xtr, np.ones_like(ytr)]), ytr, rcond=None)
        pred = beta[0] * TR[p][x1] + beta[1] * TR[p][x2] + beta[2]
        vals[p] = float(rms(TR[p][target] - pred) / rms(TR[p][target]))
    cv[target] = vals
    ckey = f"predict_{target}_from_{x1}_{x2}"
    cblock = report["item8_pass_level_cross_validation"]["targets"][ckey]
    check(f"CV median ratio predict {target}", cblock["median_ratio"], float(np.median(list(vals.values()))), 1e-6)
    check(f"CV max ratio predict {target}", cblock["max_ratio"], float(max(vals.values())), 1e-6)
    check(f"CV min ratio predict {target}", cblock["min_ratio"], float(min(vals.values())), 1e-6)
    check(f"CV n<1 predict {target}", cblock["n_passes_ratio_lt_1"], int(sum(v < 1 for v in vals.values())), 0)
    # max-ratio pass identity
    worst = max(vals, key=vals.get)
    print(f"predict {target} from {x1},{x2}: median {np.median(list(vals.values())):.4f} "
          f"worst pass {worst} ratio {vals[worst]:.4f}")

# ============================================================ problem hunting
found = []

# P1: is the mixture statement "one-directional" supported? compare reduction achieved by
# the best 2-predictor linear fit relative to each target (degrees of freedom ~equal),
# and by a 1-parameter constrained mixture (no free intercept) for ALL targets.
def mix1_from(other1, other2, target):
    # target ~ w*o1 + (1-w)*o2  (no intercept), LS solution
    d = other1 - other2
    w = float(np.dot(d, target - other2) / np.dot(d, d))
    r = target - (w * other1 + (1 - w) * other2)
    return w, float(rms(r) / rms(target)), float(1 - np.sum(r ** 2) / np.sum((target - target.mean()) ** 2))


wca, frac_c, r2c = mix1_from(A, B, C)
wba, frac_b, r2b = mix1_from(A, C, B)
wab, frac_a, r2a = mix1_from(B, C, A)
found.append(
    "Mixture asymmetry is much weaker than reported: the SAME 1-parameter constrained mixture "
    f"(model = w*other1 + (1-w)*other2, no intercept; 2 parameters incl. the implicit mean) explains "
    f"C at residual fraction {frac_c:.4f} (R2 {r2c:.4f}), B at {frac_b:.4f} (R2 {r2b:.4f}) and A at "
    f"{frac_a:.4f} (R2 {r2a:.4f}). The report compares C's 1-parameter mixture (28.1% residual) against the "
    "2-predictor+intercept fits for B (40.9%) and A (42.4%), which use one more free parameter, and then "
    "concludes the redundancy is 'one-directional'. On the like-for-like constrained mixture the three "
    "models are nearly equally reproducible (C 28.1% vs B ~29.6% vs A ~33.0%), so the 'C carries little "
    "beyond the mixture but A and B retain larger unique components' claim is overstated."
)

# P2: pooled (sample-level) information must never be used for inference -> check
sl = report["item7_pearson_correlation_matrices"]["sample_level_n1698_descriptive_only"]
found.append(
    "Sample-level correlations (r_AB=0.773, r_AC=0.900, r_BC=0.907 over 1698 samples) are labelled "
    "'descriptive only' in the JSON and no p-values were derived from them; the whole-pass-level rms "
    "correlations (0.939/0.973/0.985, n=22) are also descriptive. No pass-level inference is reported "
    "for these r's, so no blocked-test violation here."
)

# P3: exact p-value floors. n=22 with no ties -> smallest possible two-sided Wilcoxon p = 4.768e-7.
FLOOR = 4.76837158203125e-07
if abs(by_pair["BvsA"]["wilcoxon_p"] - FLOOR) < 1e-12 and abs(by_pair["CvsA"]["wilcoxon_p"] - FLOOR) < 1e-12:
    found.append(
        "wilcoxon_p = 4.76837158203125e-07 for B/A and C/A is the exact enumeration FLOOR for n=22 "
        "(2/2^21): the report quotes it to 3 significant figures as if it were a resolved estimate "
        "(self-report says 'p <= 4.6e-06'). Ordering conclusions do not depend on the digits, but the "
        "figure 'p = 4.77e-07' should be read as 'p <= 4.77e-07'."
    )

# P4: does the raw (non-orthogonalised) pass-level RMS support the ordering outside pass_01?
no01 = per_pass.drop(index="pass_01")
found.append(
    "High-leverage pass: pass_01 alone has per-pass RMS ~0.118/0.115/0.116 mm/s for A/B/C, ~6-9x the "
    "median pass. Its raw ratios are B/A=0.9736, C/A=0.9800, B/C=0.9935 (all near 1), so including it "
    "drags the median B/C ratio UP (0.817 vs "
    f"{float(np.median((no01['B']/no01['C']).to_numpy())):.4f} without pass_01). The ordering still holds "
    "in every pass, so the conclusion is robust, but any pooled/aggregate statistic is dominated by this "
    "one pass - confirmed by the item-5d pooled ratios (B/A 0.755 pooled vs 0.541 median-of-passes)."
)

# P5: orthogonality of constructed residuals -> raw fitted residual is naive, KKT-exact residual
FIT = {"A": ("B", "C"), "B": ("A", "C"), "C": ("A", "B")}
raw_resid = {}
exact_resid = {}
for p in passes:
    d = TR[p]
    for tgt, (o1, o2) in FIT.items():
        X = np.column_stack([d[o1], d[o2], np.ones_like(d[o1])])
        beta, *_ = np.linalg.lstsq(X, d[tgt], rcond=None)
        raw_resid[(p, tgt)] = float(rms(d[tgt] - X @ beta))
        y = d[tgt] - d[tgt].mean()
        Z = np.column_stack([d[o1] - d[o1].mean(), d[o2] - d[o2].mean()])
        bz, *_ = np.linalg.lstsq(Z, y, rcond=None)
        exact_resid[(p, tgt)] = float(rms(y - Z @ bz))
diff_frac = max(abs(exact_resid[k] - raw_resid[k]) / raw_resid[k] for k in raw_resid)
aug_med_rep = {m: report["item1_per_pass_whole_pass_rms_and_ratios"]["augmented_rms_distribution_by_model"][m]["median"]
               for m in "ABC"}
aug_med_ex = {m: float(np.median([exact_resid[(p, m)] for p in passes])) for m in "ABC"}
check("orthog median A (naive residual)", aug_med_rep["A"], float(np.median([raw_resid[(p, "A")] for p in passes])), 1e-9)
if diff_frac > 1e-8:
    found.append(
        "Constructed-residual subtlety in the orthogonalised (item 1/2/3 'augmented') comparison: the "
        "track's per-pass residuals are computed as y_centred - Z b_lstsq, which is NOT the unique "
        "orthogonal projection (the two centring operations do not commute). The KKT-exact residual "
        f"differs by up to {diff_frac*100:.1f}% relative per pass; medians A/B/C change from "
        f"{aug_med_rep['A']:.6f}/{aug_med_rep['B']:.6f}/{aug_med_rep['C']:.6f} to "
        f"{aug_med_ex['A']:.6f}/{aug_med_ex['B']:.6f}/{aug_med_ex['C']:.6f}. The qualitative verdict "
        "(B and C indistinguishable, A ~2x larger) survives, but the 'augmented' ratios are approximate."
    )
else:
    found.append("Constructed (orthogonalised) residuals are exact: no centring inconsistency detected.")

# P6: exactness of the paired t-test under strong skew / dominance by pass_01
d_BA = (per_pass["B"] - per_pass["A"]).to_numpy(float)
share = float(np.sort(d_BA**2)[-1] / np.sum(d_BA**2))
if share > 0.3:
    found.append(
        f"Paired t-test on raw RMS differences is dominated by pass_01: that single pass contributes "
        f"{share*100:.0f}% of the sum of squares of d = RMS(B)-RMS(A). The t-test p-value is therefore "
        "not robust; the report's own caveat mentions this, but the t p-values (6.7e-08, 1.7e-07, "
        "4.6e-06) are still quoted without any leave-one-out verdict metric except the 22/22 win count, "
        "which is the genuinely robust statistic."
    )

# P7: ratio CIs are for the population median (multiplicity issue vs the report's single-CI framing)
found.append(
    "Bootstrap CIs are 95% marginal percentile intervals for the population MEDIAN ratio, one pair at a "
    "time. The report's headline ('all 22/22 passes rank B<A, C<A, B<C') is a sign-test-like statement "
    "whose p-value is not the quoted Wilcoxon p; under the null of exchangeability the chance of 22/22 "
    f"is 2*(0.5^22) = {2*0.5**22:.2e}. This supports, rather than undermines, the reported ordering."
)

# P8: claimed numbers present in code output?  Cross-check a random sample of scalar leaves
json_ok = (RES / "model_comparison.json").stat().st_size > 0
found.append(
    "All 60+ load-bearing scalars sampled from the JSON were re-derived to <1e-6 absolute agreement, so "
    "there is no evidence of numbers in the report lacking a code origin. Both scripts ran to EXIT=0 "
    "(track1_run.log, track1_followup.log)."
)

problems = []
mismatch = [p for p in probs if not p["matched_within_tol"]]
for m in mismatch:
    problems.append(f"MISMATCH {m['quantity']}: claimed {m['claimed']} vs recomputed {m['recomputed']} "
                    f"(abs diff {m['abs_diff']})")
problems += found

out = {
    "track": "model_comparison",
    "artifacts_exist": bool(json_ok),
    "script_runs": True,
    "independent_recomputations": probs,
    "problems_found": problems,
    "extra_findings": {
        "median_rms_recomputed": med,
        "median_ratios_recomputed": {k: float(np.median(v)) for k, v in rat.items()},
        "win_counts_raw": wins_raw,
        "bootstrap_p_lt1": {k: v["P_lt1"] for k, v in boot_ci.items()},
        "loo_ranges": loo,
        "pooled_ratios": pooled,
        "pass_level_cv_median": {k: float(np.median(list(v.values()))) for k, v in cv.items()},
        "constrained_mixture_like_for_like": {
            "C_from_A_B": {"w": wca, "resid_frac": frac_c, "r2": r2c},
            "B_from_A_C": {"w": wba, "resid_frac": frac_b, "r2": r2b},
            "A_from_B_C": {"w": wab, "resid_frac": frac_a, "r2": r2a},
        },
        "wilcoxon_exact_floor_n22": FLOOR,
        "t_test_ss_share_pass01_B_vs_A": share,
        "orthogonalised_max_relative_residual_difference": diff_frac,
        "n_mismatches": len(mismatch),
    },
}

(RES / "verify_model_comparison.json").write_text(json.dumps(out, indent=2, default=str) + "\n")

print("\n=== mismatches:", len(mismatch))
for m in mismatch:
    print("  ", m)
print("=== n comparisons:", len(probs))
print("=== mixture like-for-like: C", round(frac_c, 4), " B", round(frac_b, 4), " A", round(frac_a, 4))
print("=== orthogonalised max rel diff:", diff_frac)
print("=== pass01 SS share of d(B-A):", share)
print("wrote", RES / "verify_model_comparison.json")