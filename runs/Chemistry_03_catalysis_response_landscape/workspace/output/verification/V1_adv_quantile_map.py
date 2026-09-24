#!/usr/bin/env python3
"""
V1_adv -- the decisive adversarial test for CLAIM D1.

Alternative hypothesis to kill: "the y6-vs-y7 eta^2 asymmetry is an artefact of
the two responses having different marginal distributions (y6 is strongly
right-skewed with heavy tails; y7 is nearly Gaussian), not of different
structure."

Test: give the two responses IDENTICAL marginal distributions by quantile
mapping (monotone, distribution-annihilating), then re-run the exact same
orthogonal factorial decomposition.  Variants:
  (i)   map BOTH onto U(0,1)          -> identical uniform marginals
  (ii)  map BOTH onto N(0,1)          -> identical Gaussian marginals
  (iii) map y6 onto y7's own empirical marginal, and y7 onto y6's
        -> the two responses literally swap marginal shapes
If the eta^2 split is unchanged under (i)/(ii), the asymmetry is structural.
Under (iii) the two responses' certified ranges swap, which is the strongest
possible form of the "different distribution" attack.

Also reports the definitive reconstruction: which F statistic in the claim
(F~764 for y6, F~9.9 p~.0017 for y7) actually reproduces those numbers, and
what that F is really testing.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

BASE = "/data/user/conversations/conv-4ba01bd5-0b72-45b2-94f6-e776171d414a/project"
OUT = BASE + "/output/verification"

d = pd.read_csv(BASE + "/data/observation_set_1.tsv", sep="\t")
FA, FB, FC = "numeric_feature_2", "numeric_feature_3", "numeric_feature_4"
A = d[FA].values.astype(int); B = d[FB].values.astype(int); C = d[FC].values.astype(int)
nA, nB, nC = A.max() + 1, B.max() + 1, C.max() + 1
N = nA * nB * nC
R6, R7 = "numeric_feature_6", "numeric_feature_7"
rec = {}


def closed_form_ss(y):
    y = np.asarray(y, float)
    g = y.mean()
    total = float(((y - g) ** 2).sum())
    mA = np.array([y[A == i].mean() for i in range(nA)])
    mB = np.array([y[B == j].mean() for j in range(nB)])
    mC = np.array([y[C == k].mean() for k in range(nC)])
    mA_B = np.array([[y[(A == i) & (B == j)].mean() for j in range(nB)] for i in range(nA)])
    mA_C = np.array([[y[(A == i) & (C == k)].mean() for k in range(nC)] for i in range(nA)])
    mB_C = np.array([[y[(B == j) & (C == k)].mean() for k in range(nC)] for j in range(nB)])
    mABC = np.array([[[y[(A == i) & (B == j) & (C == k)][0] for k in range(nC)]
                      for j in range(nB)] for i in range(nA)])
    ssA = nB * nC * ((mA - g) ** 2).sum()
    ssB = nA * nC * ((mB - g) ** 2).sum()
    ssC = nA * nB * ((mC - g) ** 2).sum()
    ssAB = nC * ((mA_B - mA[:, None] - mB[None, :] + g) ** 2).sum()
    ssAC = nB * ((mA_C - mA[:, None] - mC[None, :] + g) ** 2).sum()
    ssBC = nA * ((mB_C - mB[:, None] - mC[None, :] + g) ** 2).sum()
    ssABC = ((mABC - mA_B[:, :, None] - mA_C[:, None, :] - mB_C[None, :, :]
              + mA[:, None, None] + mB[None, :, None] + mC[None, None, :] - g) ** 2).sum()
    main, two, three = ssA + ssB + ssC, ssAB + ssAC + ssBC, float(ssABC)
    return {"main": main / total, "two_way": two / total, "three_way": three / total,
            "SS": {"A": float(ssA), "B": float(ssB), "C": float(ssC),
                   "AB": float(ssAB), "AC": float(ssAC), "BC": float(ssBC),
                   "ABC": three, "total": total}}


# ------------- quantile map: put source values on target's empirical marginal
def quantile_map_to(src, target):
    """Map src's values onto target's empirical quantiles (monotone, exactly
    reproduces target's marginal distribution including its shape)."""
    n = len(src)
    r = stats.rankdata(src)
    tsorted = np.sort(target)
    # linear interpolation of the target's order statistics at ranks (1..n)
    pos = r - 1.0
    lo = np.floor(pos).astype(int); hi = np.minimum(lo + 1, n - 1)
    frac = pos - lo
    return tsorted[lo] * (1 - frac) + tsorted[hi] * frac


y6 = d[R6].values.astype(float)
y7 = d[R7].values.astype(float)
rng = np.random.default_rng(20260919)

variants = {
    "raw": {R6: y6, R7: y7},
    "rank_uniform_both": {R6: stats.rankdata(y6) / (N + 1), R7: stats.rankdata(y7) / (N + 1)},
    "normal_scores_both": {R6: stats.norm.ppf((stats.rankdata(y6) - 0.375) / (N + 0.25)),
                           R7: stats.norm.ppf((stats.rankdata(y7) - 0.375) / (N + 0.25))},
    "swapped_marginals": {R6: quantile_map_to(y6, y7), R7: quantile_map_to(y7, y6)},
    "y6_as_gaussian_y7_unchanged": {R6: stats.norm.ppf((stats.rankdata(y6) - 0.375) / (N + 0.25)),
                                    R7: y7},
}

res = {}
for vname, vv in variants.items():
    res[vname] = {}
    for r in (R6, R7):
        e = closed_form_ss(vv[r])
        res[vname][r] = {"eta2": {k: e[k] for k in ("main", "two_way", "three_way")},
                         "SS": e["SS"]}
    res[vname]["main_gap_y6_minus_y7"] = (
        res[vname][R6]["eta2"]["main"] - res[vname][R7]["eta2"]["main"])
    res[vname]["threeway_gap_y6_minus_y7"] = (
        res[vname][R6]["eta2"]["three_way"] - res[vname][R7]["eta2"]["three_way"])
rec["quantile_mapping_variants"] = res

# ------------------- what exactly IS the "Tukey F" in the claim?
def effect_codes(k):
    M = np.zeros((k, k - 1)); M[: k - 1, :] = np.eye(k - 1); M[k - 1, :] = -1.0
    return M


MA, MB, MC = effect_codes(nA), effect_codes(nB), effect_codes(nC)
blk = {"int": np.ones((N, 1)), "A": MA[A], "B": MB[B], "C": MC[C],
       "AB": (MA[A][:, :, None] * MB[B][:, None, :]).reshape(N, -1),
       "AC": (MA[A][:, :, None] * MC[C][:, None, :]).reshape(N, -1),
       "BC": (MB[B][:, :, None] * MC[C][:, None, :]).reshape(N, -1),
       "ABC": (MA[A][:, :, None, None] * MB[B][:, None, :, None]
               * MC[C][:, None, None, :]).reshape(N, -1)}


def proj(Xm, y):
    return Xm @ np.linalg.solve(Xm.T @ Xm, Xm.T @ y)


tukey_detail = {}
for r in (R6, R7):
    y = d[r].values.astype(float)
    Xa = np.hstack([blk[c] for c in ["int", "A", "B", "C"]])
    fit = proj(Xa, y)
    resid = y - fit
    z = fit ** 2
    zc = z - z.mean()
    zc = zc - proj(Xa, zc)
    bz = float((zc @ resid) / (zc @ zc))
    ss_nad = float(bz ** 2 * (zc @ zc))
    ss_err = float(((resid - bz * zc) ** 2).sum())
    df_err = N - Xa.shape[1] - 1
    F = (ss_nad / 1) / (ss_err / df_err)
    # what stratum is the error?
    ss2 = closed_form_ss(y)["SS"]
    tukey_detail[r] = {
        "F_tukey_1df": float(F), "p": float(stats.f.sf(F, 1, df_err)),
        "df_error": int(df_err),
        "SS_nonadditivity_1df": ss_nad,
        "SS_error_pooled": ss_err,
        "error_stratum_is": "pooled 2-way + 3-way interaction SS (no pure error)",
        "check_error_equals_SS2plusSS3_minus_nad":
            float(ss_err) - (ss2["AB"] + ss2["AC"] + ss2["BC"] + ss2["ABC"] - ss_nad),
        "SS_two_way_total": ss2["AB"] + ss2["AC"] + ss2["BC"],
        "SS_three_way_total": ss2["ABC"],
        "frac_of_3way_SS_captured_by_the_1df_nonadditivity":
            ss_nad / ss2["ABC"],
        "interpretation": ("This F tests ONE degree of freedom (Tukey's "
                           "non-additivity contrast on the additive-model fit), "
                           "NOT the full (a-1)(b-1)(c-1)=1080-df 3-way "
                           "interaction.  It is the fraction of the 3-way "
                           "stratum that is smooth/multiplicative."),
    }
rec["tukey_1df_forensics"] = tukey_detail

# ------------- does the 1-df nonadditivity reproduce the CLAIMED numbers?
rec["claim_reconstruction"] = {
    "claimed_eta2_y6": {"main": 0.521, "two_way": 0.325, "three_way": 0.154},
    "claimed_eta2_y7": {"main": 0.887, "two_way": 0.076, "three_way": 0.037},
    "reproduced_eta2_y6": closed_form_ss(y6),
    "reproduced_eta2_y7": closed_form_ss(y7),
    "claimed_F_y6": 764, "reproduced_F_y6": tukey_detail[R6]["F_tukey_1df"],
    "claimed_F_y7": 9.9, "claimed_p_y7": 0.0017,
    "reproduced_F_y7": tukey_detail[R7]["F_tukey_1df"],
    "reproduced_p_y7": tukey_detail[R7]["p"],
    "F_match": bool(abs(tukey_detail[R6]["F_tukey_1df"] - 764) < 5
                    and abs(tukey_detail[R7]["F_tukey_1df"] - 9.9) < 0.5),
}

# ---------------- structural-shape comparison of the 3-way stratum
shape = {}
for r in (R6, R7):
    y = d[r].values.astype(float)
    X2 = np.hstack([blk[c] for c in ["int", "A", "B", "C", "AB", "AC", "BC"]])
    resid3 = y - proj(X2, y)
    ss3 = float((resid3 ** 2).sum())
    dft = (nA - 1) * (nB - 1) * (nC - 1)
    shape[r] = {
        "SS_3way": ss3, "df_3way": dft, "MS_3way": ss3 / dft,
        "MS_3way_as_frac_of_MS_total": (ss3 / dft) / (float(((y - y.mean()) ** 2).sum()) / (N - 1)),
        "resid3_sd": float(resid3.std(ddof=1)),
        "resid3_skew": float(stats.skew(resid3)), "resid3_excess_kurt": float(stats.kurtosis(resid3)),
        "resid3_shapiro_p": float(stats.shapiro(resid3).pvalue),
        "response_sd": float(y.std(ddof=1)),
        "resid3_sd_over_response_sd": float(resid3.std(ddof=1) / y.std(ddof=1)),
    }
rec["three_way_stratum_shape"] = shape

# ---------------- leave-out / split-half stability of the eta2 split
# (a design-free sanity check that the asymmetry is not driven by a few cells)
stab = {}
for r in (R6, R7):
    y = d[r].values.astype(float)
    # split by factor A levels (5 vs 5) -- each half is itself a complete 5x13x11 factorial
    stab[r] = {}
    for name, mask in [("A_levels_0_4", A < 5), ("A_levels_5_9", A >= 5)]:
        yy = y[mask]
        # relabel each half's factor levels to contiguous 0..k-1
        a2 = np.unique(A[mask], return_inverse=True)[1]
        b2 = np.unique(B[mask], return_inverse=True)[1]
        c2 = np.unique(C[mask], return_inverse=True)[1]
        nA2, nB2, nC2 = a2.max() + 1, b2.max() + 1, c2.max() + 1
        g = yy.mean()
        tot = float(((yy - g) ** 2).sum())
        mA = np.array([yy[a2 == i].mean() for i in range(nA2)])
        mB = np.array([yy[b2 == j].mean() for j in range(nB2)])
        mC = np.array([yy[c2 == k].mean() for k in range(nC2)])
        mA_B = np.array([[yy[(a2 == i) & (b2 == j)].mean() for j in range(nB2)] for i in range(nA2)])
        mA_C = np.array([[yy[(a2 == i) & (c2 == k)].mean() for k in range(nC2)] for i in range(nA2)])
        mB_C = np.array([[yy[(b2 == j) & (c2 == k)].mean() for k in range(nC2)] for j in range(nB2)])
        mABC = np.array([[[yy[(a2 == i) & (b2 == j) & (c2 == k)][0] for k in range(nC2)]
                          for j in range(nB2)] for i in range(nA2)])
        ssA = nB2 * nC2 * ((mA - g) ** 2).sum(); ssB = nA2 * nC2 * ((mB - g) ** 2).sum()
        ssC = nA2 * nB2 * ((mC - g) ** 2).sum()
        ssAB = nC2 * ((mA_B - mA[:, None] - mB[None, :] + g) ** 2).sum()
        ssAC = nB2 * ((mA_C - mA[:, None] - mC[None, :] + g) ** 2).sum()
        ssBC = nA2 * ((mB_C - mB[:, None] - mC[None, :] + g) ** 2).sum()
        ssABC = float(((mABC - mA_B[:, :, None] - mA_C[:, None, :] - mB_C[None, :, :]
                        + mA[:, None, None] + mB[None, :, None] + mC[None, None, :] - g) ** 2).sum())
        stab[r][name] = {"main": float((ssA + ssB + ssC) / tot),
                         "two_way": float((ssAB + ssAC + ssBC) / tot),
                         "three_way": ssABC / tot, "n_cells": int(len(yy)),
                         "is_complete_factorial": bool(len(yy) == nA2 * nB2 * nC2)}
rec["half_split_by_factorA"] = stab


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        fv = float(o)
        return fv if np.isfinite(fv) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


with open(OUT + "/V1_adv_quantile_map.json", "w") as fh:
    json.dump(clean(rec), fh, indent=2)

print("=== quantile-mapping adversarial test (eta^2 main / 2way / 3way) ===")
for vname, vv in res.items():
    print(f"{vname}:")
    for r in (R6, R7):
        e = vv[r]["eta2"]
        print(f"   {r}: {e['main']:.4f} / {e['two_way']:.4f} / {e['three_way']:.4f}")
    print(f"   main-gap(y6-y7)={vv['main_gap_y6_minus_y7']:.4f}  "
          f"3way-gap={vv['threeway_gap_y6_minus_y7']:.4f}")
print("\n=== Tukey 1-df forensics ===")
print(json.dumps(clean(tukey_detail), indent=2))
print("\n=== claim reconstruction ===")
print(json.dumps(clean(rec["claim_reconstruction"]), indent=2))
print("\n=== 3-way stratum shape ===")
print(json.dumps(clean(shape), indent=2))
print("\n=== half split by factor A ===")
print(json.dumps(clean(stab), indent=2))