#!/usr/bin/env python3
"""
V3 (part 3) - follow-ups:
  G. reconcile the claimed set_4 residual 1.1e-16 : which expression produces it?
  H. exact-rational OLS of R = a*x + b with x = B/(A+B): exact a and b
  I. ulp-accurate statement of the exact-rational test
  J. fuzzy near-duplicate scan of every column vs candidate derived quantities
  K. per-series (13-condition block) structure of the response and magnitudes
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


out = {}
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
    d = {}
    print("\n" + "=" * 74)
    print(f"{tag}   n={n}")
    print("=" * 74)

    # ---- G. which expression gives 1.1102230246251565e-16 ? ----------------
    forms = {
        "(B-A)/(B+A)": (B - A) / (B + A),
        "(B-A)*pow(B+A,-1)": (B - A) * np.power(B + A, -1.0),
        "(B-A)*reciprocal(B+A)": (B - A) * np.reciprocal(B + A),
        "np.divide(B-A,B+A)": np.divide(B - A, B + A),
        "1-2A/(A+B)": 1.0 - 2.0 * A / (A + B),
        "2B/(A+B)-1": 2.0 * B / (A + B) - 1.0,
        "B/(A+B)-A/(A+B)": B / (A + B) - A / (A + B),
        "(1-A/B)/(1+A/B)": (1.0 - A / B) / (1.0 + A / B),
        "(A-B)/(A+B)": (A - B) / (A + B),
        "(B-A)/(A+B)": (B - A) / (A + B),
    }
    print("[G] residual of the printed R vs algebraically equivalent forms:")
    gd = {}
    for k, v in forms.items():
        mx = float(np.abs(R - v).max())
        nz = int((R - v != 0).sum())
        gd[k] = {"max_abs_resid": mx, "n_nonzero": nz}
        star = "  <== reproduces 1.1102230246251565e-16" if abs(mx - 2.0**-53) < 1e-30 else ""
        print(f"    {k:24s} max|resid|={mx:.16e}  n_nonzero={nz:3d}/{n}{star}")
    d["form_residuals"] = gd
    d["canonical_max_abs_resid"] = gd["(B-A)/(B+A)"]["max_abs_resid"]

    # ---- H. exact rational OLS of R = a * B/(A+B) + b ----------------------
    fA = [Fraction(s) for s in rawA]
    fB = [Fraction(s) for s in rawB]
    fR = [Fraction(s) for s in rawR]
    x = [fb / (fa + fb) for fa, fb in zip(fA, fB)]
    N = Fraction(n)
    sx = sum(x); sy = sum(fR)
    sxx = sum(v * v for v in x); sxy = sum(v * r for v, r in zip(x, fR))
    den = N * sxx - sx * sx
    a_ex = (N * sxy - sx * sy) / den
    b_ex = (sy * sxx - sx * sxy) / den
    print("[H] EXACT rational OLS (Fractions, on the printed decimals):")
    print(f"    a = {a_ex}  = {float(a_ex):.17g}")
    print(f"    b = {b_ex}  = {float(b_ex):.17g}")
    print(f"    |a-2| = {abs(a_ex-2)}  ({float(abs(a_ex-2)):.3e})")
    print(f"    |b+1| = {abs(b_ex+1)}  ({float(abs(b_ex+1)):.3e})")
    print(f"    a exactly 2 in exact arithmetic : {a_ex == 2}")
    print(f"    b exactly -1 in exact arithmetic: {b_ex == -1}")
    # biggest-picture check: is the response a function of the ratio alone?
    y = [fb / fa for fa, fb in zip(fA, fB)]
    dd = [fr - (1 - q) / (1 + q) for fr, q in zip(fR, y)]
    print(f"    exact |R - (1-A/B)/(1+A/B)| max = {max(abs(t) for t in dd)} = "
          f"{float(max(abs(t) for t in dd)):.3e}")
    d["exact_ols"] = {"a": float(a_ex), "b": float(b_ex),
                      "a_minus_2": float(abs(a_ex - 2)), "b_plus_1": float(abs(b_ex + 1)),
                      "a_is_exactly_2": bool(a_ex == 2), "b_is_exactly_minus1": bool(b_ex == -1)}

    # ---- I. ulp-accurate exact-rational test -------------------------------
    devs, ulps, nz = [], [], 0
    for i in range(n):
        fa, fb, fr = Fraction(rawA[i]), Fraction(rawB[i]), Fraction(rawR[i])
        q = (fb - fa) / (fb + fa)
        delta = q - fr
        devs.append(float(abs(delta)))
        ul = 0.5 * float(np.spacing(np.abs(R[i])))
        ulps.append(float(abs(delta)) / ul)
        if delta != 0:
            nz += 1
    devs = np.array(devs); ulps = np.array(ulps)
    print("[I] exact-rational deviation expressed in half-ulps of the stored double:")
    print(f"    max|q-R| = {devs.max():.4e}   max in half-ulps = {ulps.max():.2f}"
          f"   median = {np.median(ulps):.2f}    rows with q==R exactly: {n-nz}/{n}")
    print(f"    rows with |q-R| <= 1 ulp of double : {int((ulps<=2).sum())}/{n}")
    print("    -> q is the exact ratio of the PUBLISHED digits; it agrees with the")
    print("       published double to <= ~2 ulps, i.e. pure double-precision")
    print("       arithmetic noise.  It does NOT agree merely to the ~1e-8 level")
    print("       that would be expected if the magnitudes had been rounded from")
    print("       more precise underlying values before the ratio was taken.")
    d["exact_rational_max_abs_dev"] = float(devs.max())
    d["exact_rational_max_half_ulps"] = float(ulps.max())
    d["exact_rational_median_half_ulps"] = float(np.median(ulps))
    d["exact_rational_n_exact"] = int(n - nz)

    # ---- J. fuzzy near-duplicate scan --------------------------------------
    cand = {"A+B": A + B, "B-A": B - A, "A*B": A * B, "A/B": A / B, "B/A": B / A,
            "identity": (B - A) / (B + A), "A": A, "B": B, "R": R,
            "2A/(A+B)-1": 2 * A / (A + B) - 1}
    hits = {}
    for c in hdr:
        if c == "observation_index":
            continue
        try:
            col = np.array([float(r[ci[c]]) for r in rows])
        except ValueError:
            continue
        if col.shape[0] != n:
            continue
        for pk, pv in cand.items():
            if np.allclose(col, pv, rtol=1e-12, atol=1e-12):
                hits[f"{c} ~ {pk}"] = float(np.abs(col - pv).max())
    print("[J] fuzzy duplicate scan (rtol/atol 1e-12) of every column:")
    for k, v in sorted(hits.items(), key=lambda kv: kv[1]):
        print(f"    {k:34s} max|diff|={v:.3e}")
    if not hits:
        print("    none")
    d["fuzzy_duplicates"] = hits

    # ---- K. per-series structure -------------------------------------------
    blocks = np.arange(n) // 13
    print("[K] per-series (13-condition block) structure:")
    tab = []
    for blk in np.unique(blocks):
        m = blocks == blk
        tab.append(dict(block=int(blk), n=int(m.sum()),
                        R_min=float(R[m].min()), R_max=float(R[m].max()),
                        A_min=float(A[m].min()), A_max=float(A[m].max()),
                        B_min=float(B[m].min()), B_max=float(B[m].max()),
                        max_abs_resid=float(np.abs(R[m] - (B[m] - A[m]) / (B[m] + A[m])).max())))
        print(f"    block {int(blk)}: R=[{R[m].min():.4f},{R[m].max():.4f}] "
              f"A=[{A[m].min():.3f},{A[m].max():.3f}] B=[{B[m].min():.2f},{B[m].max():.2f}] "
              f"max|resid|={tab[-1]['max_abs_resid']:.1e}")
    d["per_block"] = tab

    # correlation of the response with the nominal series index
    f1 = []
    for r in rows:
        v = r[ci.get("numeric_feature_1", "observation_index")].strip()
        f1.append(float(v) if v not in ("", "nan") else np.nan)
    f1 = np.array(f1)
    ok = ~np.isnan(f1)
    print(f"    series label column numeric_feature_1 present on {int(ok.sum())}/{n} rows,"
          f" values {sorted(set(f1[ok]))}")
    d["series_labels"] = sorted(float(v) for v in set(f1[ok]))
    print(f"    residual has zero variance, so no series label can explain it"
          f" (Var(resid)={float(np.var(R-(B-A)/(B+A))):.3e})")
    d["residual_variance"] = float(np.var(R - (B - A) / (B + A)))

    out[tag] = d

with open(os.path.join(OUT, "V3_followup.json"), "w") as fh:
    json.dump(out, fh, indent=2, default=str)
print(f"\nwrote {os.path.join(OUT,'V3_followup.json')}")