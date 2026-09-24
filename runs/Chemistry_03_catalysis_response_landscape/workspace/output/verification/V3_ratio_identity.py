#!/usr/bin/env python3
"""
V3 - Independent falsification of CLAIM D3.

CLAIM: In observation_set_3 and observation_set_4 the response column is an EXACT
algebraic function of the two magnitude columns:
   set_3: f5 == (f4 - f3) / (f4 + f3)
   set_4: f6 == (f5 - f4) / (f5 + f4)
with max abs residual at double-precision rounding (0.0 and 1.1e-16), and the
response therefore carries NO independent measurement content.

Everything is re-derived from the raw TSV text. No claim is taken on trust.
"""
import json
import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

DATA = "/data/user/conversations/conv-4ba01bd5-0b72-45b2-94f6-e776171d414a/project/data"
OUT = "/data/user/conversations/conv-4ba01bd5-0b72-45b2-94f6-e776171d414a/project/output/verification"
os.makedirs(OUT, exist_ok=True)

RNG = np.random.default_rng(20230919)
res = {}

PANELS = {
    "set_3": {"file": "observation_set_3.tsv", "A": "numeric_feature_3",
              "B": "numeric_feature_4", "R": "numeric_feature_5"},
    "set_4": {"file": "observation_set_4.tsv", "A": "numeric_feature_4",
              "B": "numeric_feature_5", "R": "numeric_feature_6"},
}


def raw_strings(path):
    """Return header + list-of-fields from the raw TSV text (no float parsing)."""
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip() != ""]
    hdr = lines[0].split("\t")
    rows = [ln.split("\t") for ln in lines[1:]]
    return hdr, rows


def decimals(s):
    """Printed decimal places of a raw numeric string (trailing zeros stripped
    by the writer, so this is a LOWER bound on the column's print precision)."""
    s = s.strip()
    if s == "" or s.lower() == "nan":
        return None
    if "." in s:
        return len(s.split(".")[1])
    return 0


def ulp_half(s):
    """Half-ulp of the printed representation of string s (its rounding interval)."""
    d = decimals(s)
    if d is None:
        return None
    return 0.5 * 10.0 ** (-d)


print("=" * 78)
print("V3 : independent re-derivation of the ratio identity")
print("=" * 78)

for tag, cfg in PANELS.items():
    hdr, rows = raw_strings(os.path.join(DATA, cfg["file"]))
    ci = {c: i for i, c in enumerate(hdr)}
    n = len(rows)

    A = np.array([float(r[ci[cfg["A"]]]) for r in rows])
    B = np.array([float(r[ci[cfg["B"]]]) for r in rows])
    R = np.array([float(r[ci[cfg["R"]]]) for r in rows])

    rawA = [r[ci[cfg["A"]]] for r in rows]
    rawB = [r[ci[cfg["B"]]] for r in rows]
    rawR = [r[ci[cfg["R"]]] for r in rows]

    print(f"\n----- {tag}  ({cfg['file']}, {n} rows) -----")
    print(f"cols: A={cfg['A']}  B={cfg['B']}  R={cfg['R']}")

    # ---------------- 0. sanity on the mapping -------------------------------
    print(f"[0] A range [{A.min():.6f},{A.max():.6f}]  B range [{B.min():.6f},{B.max():.6f}]"
          f"  B>A everywhere: {bool(np.all(B > A))}   A>0: {bool(np.all(A>0))}"
          f"   B>0: {bool(np.all(B>0))}")
    print(f"    R range [{R.min():.16f},{R.max():.16f}]  inside (-1,1): {bool(np.all(np.abs(R)<1))}")

    # ---------------- 1. raw residual ---------------------------------------
    pred = (B - A) / (B + A)
    resid = R - pred
    absr = np.abs(resid)
    max_abs = float(absr.max())
    n_lt_1e12 = int((absr < 1e-12).sum())
    n_lt_1e16 = int((absr < 1e-16).sum())
    n_exact0 = int((absr == 0.0).sum())
    print(f"[1] max|residual| = {max_abs:.6e}   ==0.0: {n_exact0}/{n}"
          f"   <1e-16: {n_lt_1e16}/{n}   <1e-12: {n_lt_1e12}/{n}")
    print(f"    median|residual| = {np.median(absr):.3e}   "
          f"argmax row = {int(absr.argmax())}")

    # how many ulps is the max residual, relative to R's own magnitude?
    ulpR = np.abs(resid) / np.spacing(np.abs(R) + 1e-300)
    print(f"    max residual measured in ulps of R: {ulpR.max():.2f}"
          f"  (median {np.median(ulpR):.2f})")

    # ---------------- 2. adversarial rounding-budget test -------------------
    # The magnitudes are printed with at most `dmax` decimals.  If the underlying
    # full-precision magnitudes were rounded to that many decimals BEFORE the
    # response was computed, the response would inherit an error of up to
    # |dR/dA|*hA + |dR/dB|*hB  (h = half-ulp).  Compare that budget with the
    # observed residual: residual << budget  =>  the printed magnitudes are the
    # exact values used, i.e. the identity is a generating relation at full
    # double precision, not an artefact of rounding.
    dmaxA = max(d for d in (decimals(s) for s in rawA) if d is not None)
    dmaxB = max(d for d in (decimals(s) for s in rawB) if d is not None)
    dmaxR = max(d for d in (decimals(s) for s in rawR) if d is not None)
    dminA = min(d for d in (decimals(s) for s in rawA) if d is not None)
    dminB = min(d for d in (decimals(s) for s in rawB) if d is not None)
    print(f"[2] printed decimals  A: {dminA}..{dmaxA}   B: {dminB}..{dmaxB}   R: {dmaxR}"
          f"  (R is at full double repr -> its own rounding interval is ~1e-16)")

    hA = 0.5 * 10.0 ** (-dmaxA)
    hB = 0.5 * 10.0 ** (-dmaxB)
    dRdA = -2.0 * B / (A + B) ** 2
    dRdB = 2.0 * A / (A + B) ** 2
    budget = np.abs(dRdA) * hA + np.abs(dRdB) * hB
    print(f"    half-ulp of printed magnitudes: hA={hA:.1e} hB={hB:.1e}")
    print(f"    max sensitivity |dR/dA|={np.abs(dRdA).max():.3e}  |dR/dB|={np.abs(dRdB).max():.3e}")
    print(f"    rounding BUDGET on R (max over rows) = {budget.max():.3e}")
    print(f"    observed max|residual|               = {max_abs:.3e}")
    ratio = max_abs / budget.max() if budget.max() > 0 else np.nan
    print(f"    observed/budget = {ratio:.3e}   ->  "
          f"{'residual is FAR below the rounding budget' if ratio<0.01 else 'residual is comparable to rounding budget'}")

    # (a) Literal perturbation test: perturb each magnitude uniformly in +-h,
    #     recompute implied R, and ask whether the printed R is reproduced
    #     within (i) its own printed precision (half-ulp of last printed digit)
    #     and (ii) the achievable interval over the whole perturbation box.
    NPERT = 20000
    pA = A[:, None] + RNG.uniform(-hA, hA, size=(n, NPERT))
    pB = B[:, None] + RNG.uniform(-hB, hB, size=(n, NPERT))
    pR = (pB - pA) / (pB + pA)
    dev = np.abs(pR - R[:, None])
    hR_row = np.array([0.5 * 10.0 ** (-decimals(s)) for s in rawR])
    ok_exact = (dev <= hR_row[:, None])
    frac_strict = float(ok_exact.any(axis=1).mean())
    # a looser, still meaningful bar: reproduce R to <=1e-12
    frac_1e12 = float((dev <= 1e-12).any(axis=1).mean())
    # achievable interval over the box (deterministic corners are enough: R is
    # monotone in B and A on the positive orthant)
    Rmin = (B + hB - (A + hA)) / (B + hB + (A + hA))
    Rmax = (B - hB - (A - hA)) / (B - hB + (A - hA))
    in_band = (R >= Rmin - 1e-18) & (R <= Rmax + 1e-18)
    frac_band = float(in_band.mean())
    print(f"[2a] literal perturb test (uniform +-{hA:.0e}/{hB:.0e}, {NPERT} draws):")
    print(f"     fraction of rows whose printed R is reachable within R's own")
    print(f"     printed precision (~{hR_row.mean():.1e}): {frac_strict*100:.1f}%")
    print(f"     fraction reachable within 1e-12:                    {frac_1e12*100:.1f}%")
    print(f"     fraction of rows with R inside the achievable box [Rmin,Rmax]: "
          f"{frac_band*100:.1f}%")
    print(f"     worst-case deviation of implied R from printed R under the box: "
          f"{dev.max():.3e}")
    print(f"     box half-width of the achievable R band (mean): "
          f"{np.mean((Rmax-Rmin)/2):.3e}")
    # how far outside the band is the worst row (0 => fully consistent)
    outside = np.maximum(Rmin - R, R - Rmax)
    print(f"     max distance of printed R outside the achievable band: {outside.max():.3e}")

    # ---------------- 2b. inversion test ------------------------------------
    # Solve the identity for B given A and R:  B = A (1+R)/(1-R).
    # If this reproduces the printed B to ~1e-16 relative, then the triple is
    # self-consistent at full double precision: the response is a deterministic
    # recomputation of (A,B), not an independent reading.
    B_inv = A * (1.0 + R) / (1.0 - R)
    rel_inv = np.abs(B_inv - B) / B
    print(f"[2b] inversion B_implied = A(1+R)/(1-R): max rel err = {rel_inv.max():.3e}"
          f"  (median {np.median(rel_inv):.3e})")
    A_inv = B * (1.0 - R) / (1.0 + R)
    rel_invA = np.abs(A_inv - A) / A
    print(f"     inversion A_implied = B(1-R)/(1+R): max rel err = {rel_invA.max():.3e}")

    # ---------------- 3. coincidental-fit alternative ------------------------
    def fit_linear(x, y):
        X = np.column_stack([x, np.ones_like(x)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ coef
        return float(coef[0]), float(coef[1]), float(np.abs(r).max()), float(np.sum(r**2))

    fams = {}
    fams["R ~ a*B/(A+B) + b"] = fit_linear(B / (A + B), R)
    fams["R ~ a*A/(A+B) + b"] = fit_linear(A / (A + B), R)
    fams["R ~ a*(B-A)/(A+B) + b"] = fit_linear((B - A) / (B + A), R)
    fams["R ~ a*B/A + b"] = fit_linear(B / A, R)
    fams["R ~ a*A/B + b"] = fit_linear(A / B, R)
    fams["R ~ a*B + b"] = fit_linear(B, R)
    fams["R ~ a*A + b"] = fit_linear(A, R)
    fams["R ~ a*(B-A) + b"] = fit_linear(B - A, R)
    fams["R ~ a*log(B/A) + b"] = fit_linear(np.log(B / A), R)
    print("[3] two-parameter family fits  (a, b, max|resid|, SSE):")
    for k, (a, b, mx, sse) in fams.items():
        flag = ""
        if abs(mx) < 1e-12:
            flag = "  <-- FITS TO MACHINE PRECISION"
        print(f"    {k:26s} a={a: .12f}  b={b: .12f}  max|r|={mx:.3e}  SSE={sse:.3e}{flag}")

    a_bb, b_bb, mx_bb, sse_bb = fams["R ~ a*B/(A+B) + b"]
    a_ba, b_ba, _, _ = fams["R ~ a*A/(A+B) + b"]
    a_ra, b_ra, mx_ra, _ = fams["R ~ a*(B-A)/(A+B) + b"]
    print(f"    a==2 exactly? {a_bb == 2.0}   b==-1 exactly? {b_bb == -1.0}"
          f"   (|a-2|={abs(a_bb-2):.3e}, |b+1|={abs(b_bb+1):.3e})")
    print(f"    degenerate reparametrisation A/(A+B): a={a_ba:.12f} (=-2?), b={b_ba:.12f} (=+1?)")
    print(f"    canonical form (B-A)/(A+B): a={a_ra:.16f} b={b_ra:.3e}")

    # nonlinear alternative with a genuinely different shape
    try:
        popt, _ = curve_fit(lambda X, a, p: a * X[1] ** p / (X[0] ** p + X[1] ** p),
                            (A, B), R, p0=(2.0, 1.0), maxfev=20000)
        nl_max = float(np.max(np.abs(R - popt[0] * B ** popt[1] / (A ** popt[1] + B ** popt[1]))))
        print(f"    nonlinear R=a*B^p/(A^p+B^p): a={popt[0]:.6f} p={popt[1]:.6f} max|r|={nl_max:.3e}"
              f"   (Holling/Rosenzweig-type family; a=2,p=1 is the identity)")
    except Exception as e:
        nl_max = None
        print(f"    nonlinear fit failed: {e}")

    # ---------------- 4. range / constraint of the magnitudes ----------------
    print("[4] distribution of the magnitudes and the implied ratio:")
    print(f"    response R : min={R.min():.6f}  max={R.max():.6f}  span={R.max()-R.min():.6f}"
          f"  IQR={np.percentile(R,75)-np.percentile(R,25):.6f}")
    ratio_BA = B / A
    ratio_AB = A / B
    print(f"    B/A        : min={ratio_BA.min():.4f}  max={ratio_BA.max():.4f}"
          f"  span={ratio_BA.max()-ratio_BA.min():.4f}")
    print(f"    A/(A+B)    : min={np.min(A/(A+B)):.4f}  max={np.max(A/(A+B)):.4f}")
    print(f"    fraction of rows with A/B < 0.25 : {float((ratio_AB<0.25).mean())*100:.1f}%"
          f"    A/B < 0.10 : {float((ratio_AB<0.10).mean())*100:.1f}%")
    print(f"    R > 0 everywhere: {bool(np.all(R>0))}   "
          f"Coefficient of variation of R: {R.std(ddof=1)/R.mean():.4f}")

    # per-series (block) view - is the identity uniform across panels/series?
    blocks = np.arange(n) // 13
    per_block = {}
    for blk in np.unique(blocks):
        m = blocks == blk
        per_block[int(blk)] = {
            "n": int(m.sum()),
            "max_abs_resid": float(np.abs(resid[m]).max()),
            "R_min": float(R[m].min()), "R_max": float(R[m].max()),
            "BA_min": float((B[m] / A[m]).min()), "BA_max": float((B[m] / A[m]).max()),
        }
    blk_str = [format(v["max_abs_resid"], ".1e") for v in per_block.values()]
    print(f"    blocks: {len(per_block)}  max|resid| per block: {blk_str}")

    # ---------------- 5. correlation / driver artefact -----------------------
    def pearson(x, y):
        r, p = stats.pearsonr(x, y)
        return float(r), float(p)

    cA = pearson(A, R)[0]
    cB = pearson(B, R)[0]
    cAB = pearson(A, B)[0]
    cABr = pearson(A / B, R)[0]
    cBAr = pearson(B / A, R)[0]
    csum = pearson(B / (A + B), R)[0]
    print("[5] Pearson correlations:")
    print(f"    corr(R, A)      = {cA: .4f}")
    print(f"    corr(R, B)      = {cB: .4f}")
    print(f"    corr(R, A/B)    = {cABr: .4f}")
    print(f"    corr(R, B/A)    = {cBAr: .4f}")
    print(f"    corr(R, B/(A+B))= {csum: .4f}")
    print(f"    corr(A, B)      = {cAB: .4f}   <-- magnitude-magnitude collinearity")
    # partial correlation of R with B controlling for A
    def partial(x, y, z):
        rx = x - np.column_stack([z, np.ones_like(z)]) @ np.linalg.lstsq(
            np.column_stack([z, np.ones_like(z)]), x, rcond=None)[0]
        ry = y - np.column_stack([z, np.ones_like(z)]) @ np.linalg.lstsq(
            np.column_stack([z, np.ones_like(z)]), y, rcond=None)[0]
        return float(stats.pearsonr(rx, ry)[0])
    pB_A = partial(B, R, A)
    pA_B = partial(A, R, B)
    print(f"    partial corr(R, B | A) = {pB_A: .4f}")
    print(f"    partial corr(R, A | B) = {pA_B: .4f}")
    # variance explained by B alone vs by the identity
    r2_B = float(stats.pearsonr(B, R)[0] ** 2)
    print(f"    R^2 of R~B alone = {r2_B:.4f}  -> naive 'B drives the response'")
    print(f"    R^2 of the identity = 1.0 by construction (max|resid|={max_abs:.1e})")

    # verify: is B roughly a linear function of A across the panel?
    sl, ic, mxr, _ = fit_linear(A, B)
    print(f"    B ~ a*A + b : a={sl:.4f} b={ic:.4f} max|resid|={mxr:.4f} "
          f"(so B is nearly proportional to A: B~A*{sl:.2f})")

    res[tag] = {
        "file": cfg["file"], "n_rows": n,
        "columns": {"A": cfg["A"], "B": cfg["B"], "R": cfg["R"]},
        "max_abs_residual": max_abs,
        "n_rows_resid_eq_0": n_exact0,
        "n_rows_resid_lt_1e16": n_lt_1e16,
        "n_rows_resid_lt_1e12": n_lt_1e12,
        "median_abs_residual": float(np.median(absr)),
        "max_residual_in_ulps_of_R": float(ulpR.max()),
        "printed_decimals": {"A": [dminA, dmaxA], "B": [dminB, dmaxB], "R_max": dmaxR},
        "rounding_budget_on_R": float(budget.max()),
        "observed_over_budget": float(ratio),
        "perturb_frac_within_R_precision": frac_strict,
        "perturb_frac_within_1e-12": frac_1e12,
        "frac_R_in_achievable_box": frac_band,
        "max_dist_R_outside_box": float(outside.max()),
        "achievable_band_halfwidth_mean": float(np.mean((Rmax - Rmin) / 2)),
        "max_dev_implied_R_under_perturbation": float(dev.max()),
        "inversion_B_max_rel_err": float(rel_inv.max()),
        "inversion_A_max_rel_err": float(rel_invA.max()),
        "fits": {k: {"a": v[0], "b": v[1], "max_abs_resid": v[2], "sse": v[3]}
                 for k, v in fams.items()},
        "fit_main_a": a_bb, "fit_main_b": b_bb, "fit_main_max_abs_resid": mx_bb,
        "fit_main_is_exactly_2": bool(a_bb == 2.0), "fit_main_is_exactly_minus1": bool(b_bb == -1.0),
        "fit_reparam_A_over_AB": {"a": a_ba, "b": b_ba},
        "fit_canonical_BminusA_over_BplusA": {"a": a_ra, "b": b_ra, "max_abs_resid": mx_ra},
        "response_min": float(R.min()), "response_max": float(R.max()),
        "response_span": float(R.max() - R.min()),
        "response_cv": float(R.std(ddof=1) / R.mean()),
        "BA_min": float(ratio_BA.min()), "BA_max": float(ratio_BA.max()),
        "AB_max": float(ratio_AB.max()),
        "corr_R_A": cA, "corr_R_B": cB, "corr_R_AB": cABr, "corr_R_BA": cBAr,
        "corr_A_B": cAB, "corr_R_sumfrac": csum,
        "partial_corr_R_B_given_A": pB_A, "partial_corr_R_A_given_B": pA_B,
        "r2_R_vs_B_alone": r2_B,
        "A_range": [float(A.min()), float(A.max())],
        "B_range": [float(B.min()), float(B.max())],
        "per_block": per_block,
    }

# ----------------------------- summary -------------------------------------
print("\n" + "=" * 78)
print("SUMMARY vs CLAIM D3")
print("=" * 78)
for tag in PANELS:
    d = res[tag]
    print(f"{tag}: max|resid|={d['max_abs_residual']:.3e}  "
          f"rows<1e-12 = {d['n_rows_resid_lt_1e12']}/{d['n_rows']}  "
          f"budget/reality ratio={d['observed_over_budget']:.2e}")
    print(f"        main fit a={d['fit_main_a']!r} b={d['fit_main_b']!r} "
          f"(exact 2/-1: {d['fit_main_is_exactly_2']}/{d['fit_main_is_exactly_minus1']})")
    print(f"        R in [{d['response_min']:.4f},{d['response_max']:.4f}]  "
          f"B/A in [{d['BA_min']:.2f},{d['BA_max']:.2f}]  "
          f"corr(R,B)={d['corr_R_B']:.3f} corr(R,A/B)={d['corr_R_AB']:.3f} "
          f"corr(A,B)={d['corr_A_B']:.3f}")

with open(os.path.join(OUT, "V3_ratio_identity.json"), "w") as fh:
    json.dump(res, fh, indent=2, default=float)
print(f"\nwrote {os.path.join(OUT,'V3_ratio_identity.json')}")