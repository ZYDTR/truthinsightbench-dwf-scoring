#!/usr/bin/env python3
"""
V3 (part 4) - duplicate-column verification + consolidation of all V3 results.

L. Are there duplicated / scaled-offset copies of the response among the other
   columns of each panel?  (raw string equality AND looser tolerance)
M. Why do the two residual numbers in the claim differ (0.0 vs 1.1e-16)?
   -> enumerate exactly which formula variants reproduce each number.
N. Consolidated verdict payload.
"""
import json
import os
import numpy as np
from fractions import Fraction
from scipy import stats

DATA = "/data/user/conversations/conv-4ba01bd5-0b72-45b2-94f6-e776171d414a/project/data"
OUT = "/data/user/conversations/conv-4ba01bd5-0b72-45b2-94f6-e776171d414a/project/output/verification"
PANELS = {
    "set_3": dict(file="observation_set_3.tsv", A="numeric_feature_3",
                  B="numeric_feature_4", R="numeric_feature_5"),
    "set_4": dict(file="observation_set_4.tsv", A="numeric_feature_4",
                  B="numeric_feature_5", R="numeric_feature_6"),
}


def load(cfg):
    with open(os.path.join(DATA, cfg["file"])) as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip() != ""]
    hdr = lines[0].split("\t")
    return hdr, [ln.split("\t") for ln in lines[1:]]


summary = {}
for tag, cfg in PANELS.items():
    hdr, rows = load(cfg)
    ci = {c: i for i, c in enumerate(hdr)}
    n = len(rows)
    A = np.array([float(r[ci[cfg["A"]]]) for r in rows])
    B = np.array([float(r[ci[cfg["B"]]]) for r in rows])
    R = np.array([float(r[ci[cfg["R"]]]) for r in rows])
    rawA = [r[ci[cfg["A"]]] for r in rows]
    rawB = [r[ci[cfg["B"]]] for r in rows]
    rawR = [r[ci[cfg["R"]]] for r in rows]
    S = {}
    print("\n" + "=" * 74)
    print(f"{tag}  n={n}")
    print("=" * 74)

    # ---- L. duplicate / affine-copy scan -----------------------------------
    print("[L] scanning every numeric column for copies of the response:")
    outl = []
    for c in hdr:
        if c == "observation_index":
            continue
        try:
            col = np.array([float(r[ci[c]]) for r in rows])
        except ValueError:
            continue
        if col.shape[0] != n:
            continue
        rawc = [r[ci[c]] for r in rows]
        same_str = all(x.strip() == y.strip() for x, y in zip(rawc, rawR))
        maxabs = float(np.abs(col - R).max())
        # affine fit col = s*R + t
        s, t = np.polyfit(R, col, 1)
        aff = float(np.abs(col - (s * R + t)).max())
        rel = maxabs / max(1e-300, float(np.abs(R).max()))
        rec = dict(col=c, string_identical=same_str, max_abs_diff=maxabs,
                   rel_diff=rel, affine_s=s, affine_t=t, affine_max_resid=aff)
        if same_str or maxabs == 0 or rel < 1e-9 or (aff < 1e-9 and not (abs(s) < 1e-12)):
            outl.append(rec)
            print(f"    {c:20s} string_identical={same_str}  max|diff|={maxabs:.3e}"
                  f"  rel={rel:.2e}  affine: {s:.6f}*R+{t:.3e} (resid {aff:.1e})")
    if not outl:
        print("    no copies found")
    S["response_copies"] = outl

    # exact string identity of the specific claim-relevant duplicate
    dup3 = [c for c in hdr if c != cfg["R"] and c != "observation_index"
            and all(r[ci[c]].strip() == r[ci[cfg["R"]]].strip() for r in rows)]
    print(f"    columns byte-identical to {cfg['R']}: {dup3}")
    S["byte_identical_to_response"] = dup3

    # ---- M. reconcile the claimed residual numbers -------------------------
    forms = {
        "(B-A)/(B+A)": (B - A) / (B + A),
        "np.divide(B-A, B+A)": np.divide(B - A, B + A),
        "-(A-B)/(B+A)": -(A - B) / (B + A),
        "(B-A)*np.reciprocal(B+A)": (B - A) * np.reciprocal(B + A),
        "(B-A)*(B+A)**-1": (B - A) * (B + A) ** -1,
        "B/(A+B)-A/(A+B)": B / (A + B) - A / (A + B),
        "1-2A/(A+B)": 1.0 - 2.0 * A / (A + B),
        "2B/(A+B)-1": 2.0 * B / (A + B) - 1.0,
    }
    vals = {k: float(np.abs(R - v).max()) for k, v in forms.items()}
    gives_0 = [k for k, v in vals.items() if v == 0.0]
    gives_11 = [k for k, v in vals.items() if abs(v - 2.0 ** -53) < 1e-30]
    print("[M] which expressions produce 0.0 / 1.1102230246251565e-16:")
    print(f"    -> 0.0    : {gives_0}")
    print(f"    -> 1.1e-16: {gives_11}")
    S["forms_giving_exactly_zero"] = gives_0
    S["forms_giving_1p1e-16"] = gives_11
    S["form_values"] = vals
    S["canonical_max_abs_resid"] = vals["(B-A)/(B+A)"]

    # ---- core numbers ------------------------------------------------------
    resid = R - (B - A) / (B + A)
    S.update({
        "n_rows": n,
        "max_abs_residual_canonical": float(np.abs(resid).max()),
        "n_rows_resid_lt_1e-12": int((np.abs(resid) < 1e-12).sum()),
        "n_rows_resid_eq_zero": int((resid == 0).sum()),
        "residual_variance": float(np.var(resid)),
        "response_min": float(R.min()), "response_max": float(R.max()),
        "response_span": float(R.max() - R.min()),
        "BA_min": float((B / A).min()), "BA_max": float((B / A).max()),
        "corr_R_A": float(stats.pearsonr(A, R)[0]),
        "corr_R_B": float(stats.pearsonr(B, R)[0]),
        "corr_R_AB": float(stats.pearsonr(A / B, R)[0]),
        "corr_R_BA": float(stats.pearsonr(B / A, R)[0]),
        "corr_R_sumfrac": float(stats.pearsonr(B / (A + B), R)[0]),
        "corr_A_B": float(stats.pearsonr(A, B)[0]),
        "r2_R_by_B_alone": float(stats.pearsonr(B, R)[0] ** 2),
        "r2_R_by_A_alone": float(stats.pearsonr(A, R)[0] ** 2),
        "r2_R_by_ratio_AB": float(stats.pearsonr(A / B, R)[0] ** 2),
    })

    # ---------- consolidated numbers from parts 1-3 (recomputed inline) -----
    # exact rational deviation in half-ulps
    ulps = []
    for i in range(n):
        q = (Fraction(rawB[i]) - Fraction(rawA[i])) / (Fraction(rawB[i]) + Fraction(rawA[i]))
        ul = 0.5 * float(np.spacing(abs(R[i])))
        ulps.append(float(abs(q - Fraction(rawR[i]))) / ul)
    ulps = np.array(ulps)
    S["exact_rational_max_half_ulps"] = float(ulps.max())
    S["exact_rational_median_half_ulps"] = float(np.median(ulps))

    # rounding budget
    dA = max(len(s.split(".")[1]) if "." in s else 0 for s in rawA)
    dB = max(len(s.split(".")[1]) if "." in s else 0 for s in rawB)
    hA, hB = 0.5 * 10.0 ** -dA, 0.5 * 10.0 ** -dB
    budget = np.abs(-2 * B / (A + B) ** 2) * hA + np.abs(2 * A / (A + B) ** 2) * hB
    S["printed_decimals_A_max"] = dA
    S["printed_decimals_B_max"] = dB
    S["rounding_budget_max"] = float(budget.max())
    S["observed_resid_over_budget"] = float(np.abs(resid).max() / budget.max())
    # attainable band
    Rlo = (B - hB - (A + hA)) / (B - hB + (A + hA))
    Rhi = (B + hB - (A - hA)) / (B + hB + (A - hA))
    S["frac_R_inside_attainable_band"] = float(((R >= Rlo) & (R <= Rhi)).mean())
    S["band_halfwidth_mean"] = float(np.mean((Rhi - Rlo) / 2))
    # inversion (alternative causal direction)
    S["inversion_B_max_rel_err"] = float(np.max(np.abs(A * (1 + R) / (1 - R) - B) / B))
    S["inversion_A_max_rel_err"] = float(np.max(np.abs(B * (1 - R) / (1 + R) - A) / A))

    # OLS on the sum-fraction, float
    xf = B / (A + B)
    sl, ic = np.polyfit(xf, R, 1)
    S["ols_sumfrac_slope"] = float(sl)
    S["ols_sumfrac_intercept"] = float(ic)
    S["ols_sumfrac_max_abs_resid"] = float(np.abs(R - (sl * xf + ic)).max())
    S["ols_slope_minus_2"] = float(abs(sl - 2.0))
    S["ols_intercept_plus_1"] = float(abs(ic + 1.0))
    # competing families, best SSE
    def sse_of(x):
        s, i = np.polyfit(x, R, 1)
        return float(np.sum((R - (s * x + i)) ** 2))
    S["sse_identity"] = float(np.sum(resid ** 2))
    for nm, xx in {"B/(A+B)": B / (A + B), "A/(A+B)": A / (A + B), "B/A": B / A,
                   "A/B": A / B, "B": B, "A": A, "B-A": B - A,
                   "log(B/A)": np.log(B / A)}.items():
        S[f"sse_{nm}"] = sse_of(xx)
    S["corr_B_ratioAB"] = float(stats.pearsonr(B, A / B)[0])
    summary[tag] = S

with open(os.path.join(OUT, "V3_duplicates_and_consolidate.json"), "w") as fh:
    json.dump(summary, fh, indent=2, default=str)

print("\n" + "=" * 74)
print("CONSOLIDATED")
print("=" * 74)
for tag, S in summary.items():
    print(f"\n{tag}:")
    print(f"  max|resid| canonical  = {S['max_abs_residual_canonical']:.3e}"
          f"   rows<1e-12 = {S['n_rows_resid_lt_1e-12']}/{S['n_rows']}"
          f"   rows==0 = {S['n_rows_resid_eq_zero']}/{S['n_rows']}")
    print(f"  rounding budget       = {S['rounding_budget_max']:.3e}"
          f"   observed/budget = {S['observed_resid_over_budget']:.2e}")
    print(f"  exact-rational dev    = {S['exact_rational_max_half_ulps']:.2f} half-ulps max")
    print(f"  attainable-band cover = {100*S['frac_R_inside_attainable_band']:.1f}%")
    print(f"  OLS a,b               = {S['ols_sumfrac_slope']:.15f}, {S['ols_sumfrac_intercept']:.15f}"
          f"  (|a-2|={S['ols_slope_minus_2']:.1e}, |b+1|={S['ols_intercept_plus_1']:.1e})")
    print(f"  R range [{S['response_min']:.4f},{S['response_max']:.4f}]  "
          f"B/A [{S['BA_min']:.3f},{S['BA_max']:.3f}]")
    print(f"  corr: A {S['corr_R_A']:+.3f}  B {S['corr_R_B']:+.3f}  A/B {S['corr_R_AB']:+.3f}"
          f"  corr(A,B) {S['corr_A_B']:+.3f}  corr(B,A/B) {S['corr_B_ratioAB']:+.3f}")
    print(f"  R^2: B alone {S['r2_R_by_B_alone']:.4f}  A alone {S['r2_R_by_A_alone']:.4f}"
          f"  A/B {S['r2_R_by_ratio_AB']:.4f}")
    print(f"  byte-identical columns to response: {S['byte_identical_to_response']}")
    print(f"  SSE identity {S['sse_identity']:.3e} vs best competitor "
          f"{min(v for k,v in S.items() if k.startswith('sse_') and k!='sse_identity'):.3e}")
print(f"\nwrote {os.path.join(OUT,'V3_duplicates_and_consolidate.json')}")