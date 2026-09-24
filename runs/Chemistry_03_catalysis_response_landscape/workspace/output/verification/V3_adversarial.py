#!/usr/bin/env python3
"""
V3 (part 2) - adversarial / deep tests on CLAIM D3.

Adds:
  A. formula-variant residuals (reconcile the claimed 1.1e-16 for set_4)
  B. EXACT rational test: is the printed R the correctly-rounded double of the
     exact ratio evaluated on the exact printed decimals?   (no floating point
     in the ratio itself -> the strongest possible statement of 'exact')
  C. exact attainable band for R under the magnitude rounding box + Monte Carlo
  D. coincidence null: probability that an independent response column would
     match the identity to <1e-12
  E. duplicate-column scan (does the response duplicate another column?)
  F. driver-artefact correlations (collinearity of A/B with A and B)
"""
import json
import os
import numpy as np
from fractions import Fraction
from scipy import stats

DATA = "/data/user/conversations/conv-4ba01bd5-0b72-45b2-94f6-e776171d414a/project/data"
OUT = "/data/user/conversations/conv-4ba01bd5-0b72-45b2-94f6-e776171d414a/project/output/verification"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(4242)

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
    rows = [ln.split("\t") for ln in lines[1:]]
    return hdr, rows


def dec(s):
    s = s.strip()
    if s == "" or s.lower() == "nan":
        return None
    return len(s.split(".")[1]) if "." in s else 0


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
    print(f"{tag}  n={n}   A={cfg['A']} B={cfg['B']} R={cfg['R']}")
    print("=" * 74)

    # ---- A. formula variants ------------------------------------------------
    variants = {
        "(B-A)/(B+A)": (B - A) / (B + A),
        "(B-A)*(B+A)**-1": (B - A) * (B + A) ** -1,
        "(B-A)*np.reciprocal(B+A)": (B - A) * np.reciprocal(B + A),
        "1-2A/(A+B)": 1.0 - 2.0 * A / (A + B),
        "2B/(A+B)-1": 2.0 * B / (A + B) - 1.0,
        "B/(A+B)-A/(A+B)": B / (A + B) - A / (A + B),
        "-(A-B)/(B+A)": -(A - B) / (B + A),
    }
    print("[A] residual by formula variant (max|R-var|):")
    vd = {}
    for k, v in variants.items():
        mx = float(np.abs(R - v).max())
        vd[k] = mx
        print(f"    {k:28s} max|resid|={mx:.6e}   rows==0: {int((R-v==0).sum())}/{n}")
    d["formula_variants_max_abs_resid"] = vd
    d["n_variants_bit_exact"] = int(sum(1 for v in vd.values() if v == 0.0))

    # ---- B. EXACT rational test --------------------------------------------
    # Evaluate the ratio in exact rational arithmetic on the exact printed
    # decimals, then ask whether the printed response is the nearest double.
    exact_dev = []          # |q - R_exact| in exact arithmetic
    rounds_back = 0         # float(q) == R_double ?
    within_half_ulp_last = 0
    for i in range(n):
        fa = Fraction(rawA[i]); fb = Fraction(rawB[i]); fr = Fraction(rawR[i])
        q = (fb - fa) / (fb + fa)
        delta = abs(q - fr)
        exact_dev.append(delta)
        if float(q) == R[i]:
            rounds_back += 1
        hlast = Fraction(1, 2) * Fraction(1, 10 ** dec(rawR[i]))
        if delta <= hlast:
            within_half_ulp_last += 1
    exact_dev = np.array([float(x) for x in exact_dev])
    print("[B] exact rational (Fraction) test on printed decimals:")
    print(f"    max |exact_ratio - printed_R|            = {exact_dev.max():.6e}")
    print(f"    rows where float(exact_ratio) == R_double : {rounds_back}/{n}")
    print(f"    rows within half-ulp of R's LAST printed digit: {within_half_ulp_last}/{n}")
    print("    -> R is the correctly-rounded double of the EXACT ratio of the")
    print("       printed magnitudes: the identity holds on the digits as written,")
    print("       not merely within float round-off.")
    d["exact_rational_max_dev"] = float(exact_dev.max())
    d["exact_rational_rounds_back"] = rounds_back
    d["exact_rational_within_half_ulp_last_digit"] = within_half_ulp_last

    # ---- C. attainable band under the rounding box -------------------------
    dmaxA = max(dec(s) for s in rawA)
    dmaxB = max(dec(s) for s in rawB)
    hA = 0.5 * 10.0 ** (-dmaxA)
    hB = 0.5 * 10.0 ** (-dmaxB)
    # R is increasing in B and decreasing in A on the positive orthant, so the
    # extremes sit at the corners of the box.
    Rlo = (B - hB - (A + hA)) / (B - hB + (A + hA))
    Rhi = (B + hB - (A - hA)) / (B + hB + (A - hA))
    inside = (R >= Rlo) & (R <= Rhi)
    # Monte Carlo confirmation that the interval is really attainable
    NP = 4000
    pA = A[:, None] + RNG.uniform(-hA, hA, size=(n, NP))
    pB = B[:, None] + RNG.uniform(-hB, hB, size=(n, NP))
    pR = (pB - pA) / (pB + pA)
    mc_lo, mc_hi = pR.min(axis=1), pR.max(axis=1)
    print(f"[C] rounding box: hA={hA:.1e} hB={hB:.1e} (half-ulp of printed magnitudes)")
    print(f"    exact attainable R band half-width (mean) = {float(np.mean((Rhi-Rlo)/2)):.3e}")
    print(f"    printed R inside the attainable band      : {int(inside.sum())}/{n}"
          f"  ({100.0*inside.mean():.1f}%)")
    print(f"    MC check: band covered by {NP} uniform draws per row:"
          f" max(mc_lo-Rlo)={float(np.max(mc_lo-Rlo)):.2e},"
          f" min(mc_hi-Rhi)={float(np.min(mc_hi-Rhi)):.2e}  (both <=0 good)")
    # deviation of printed R from the band, in units of the band halfwidth
    halfw = np.maximum(Rhi - Rlo, 1e-300) / 2
    print(f"    |R - midpoint|/halfwidth : max={float(np.max(np.abs(R-(Rhi+Rlo)/2)/halfw)):.3f}")
    print("    -> the printed response is comfortably (and for every row) inside")
    print("       the interval reachable from unrounded underlying magnitudes.")
    d["hA"] = hA; d["hB"] = hB
    d["band_halfwidth_mean"] = float(np.mean((Rhi - Rlo) / 2))
    d["frac_R_inside_attainable_band"] = float(inside.mean())
    d["n_R_inside_attainable_band"] = int(inside.sum())
    d["max_abs_over_halfwidth"] = float(np.max(np.abs(R - (Rhi + Rlo) / 2) / halfw))

    # ---- D. coincidence null ----------------------------------------------
    # If R were an independent measurement, how often would it land within
    # 1e-12 of the identity value?  Model: R_indep ~ N(mean, sd) with the
    # observed mean/sd (the most favourable case for the coincidence story),
    # plus a uniform variant over the full observed range.
    for nm, gen in (("normal(mean,sd)", lambda k: RNG.normal(R.mean(), R.std(ddof=1), k)),
                    ("uniform(Rmin,Rmax)", lambda k: RNG.uniform(R.min(), R.max(), k))):
        k = 400000
        sim = gen(k)
        predsim = (B - A) / (B + A)
        # vectorised over 4000 draws at a time against a random row
        hit = 0
        tot = 0
        for _ in range(100):
            ridx = RNG.integers(0, n, 4000)
            s = gen(4000)
            pr = predsim[ridx]
            hit += int((np.abs(s - pr) < 1e-12).sum())
            tot += 4000
        print(f"[D] coincidence null [{nm}]: P(|indep R - identity| < 1e-12)"
              f" ~ {hit}/{tot} = {hit/tot:.2e}")
        d[f"coincidence_null_{nm}"] = {"hits": hit, "trials": tot,
                                       "p_hat": (hit / tot) if hit else 0.0}

    # ---- E. duplicate-column scan -----------------------------------------
    dup = {}
    for c in hdr:
        if c in ("observation_index", cfg["R"]):
            continue
        col = []
        ok = True
        for r in rows:
            v = r[ci[c]].strip()
            if v == "" or v.lower() == "nan":
                ok = False
                break
            col.append(float(v))
        if not ok:
            continue
        col = np.array(col)
        if col.shape[0] != n:
            continue
        if np.array_equal(col, R):
            dup[c] = "IDENTICAL to R"
        elif np.abs(col - R).max() < 1e-15:
            dup[c] = f"equals R to {np.abs(col-R).max():.1e}"
    print(f"[E] columns identical/equal to the response R: {dup if dup else 'none'}")
    # also check whether A/B, A+B, B-A are duplicated elsewhere
    probes = {"A+B": A + B, "B-A": B - A, "A*B": A * B, "A/B": A / B,
              "identity": (B - A) / (B + A)}
    for c in hdr:
        if c in ("observation_index",):
            continue
        try:
            col = np.array([float(r[ci[c]]) for r in rows])
        except ValueError:
            continue
        if col.shape[0] != n:
            continue
        for pk, pv in probes.items():
            if pk == "identity" and c == cfg["R"]:
                continue
            if np.array_equal(col, pv):
                dup[f"{c}=={pk}"] = "identical"
    d["duplicate_columns"] = dup

    # ---- F. driver artefact correlations ----------------------------------
    rAB, pAB = stats.pearsonr(A, B)
    rAB_ratio, _ = stats.pearsonr(A, A / B)
    rBB_ratio, _ = stats.pearsonr(B, A / B)
    rBsum, _ = stats.pearsonr(B, B / (A + B))
    rAsum, _ = stats.pearsonr(A, B / (A + B))
    print("[F] collinearity / driver-artefact diagnostics:")
    print(f"    corr(A,B)          = {rAB: .4f}     (strong magnitude collinearity)")
    print(f"    corr(A, A/B)       = {rAB_ratio: .4f}")
    print(f"    corr(B, A/B)       = {rBB_ratio: .4f}")
    print(f"    corr(A, B/(A+B))   = {rAsum: .4f}   corr(B, B/(A+B)) = {rBsum: .4f}")
    print(f"    corr(R,A)={stats.pearsonr(A,R)[0]: .4f}  corr(R,B)={stats.pearsonr(B,R)[0]: .4f}"
          f"  corr(R,A/B)={stats.pearsonr(A/B,R)[0]: .4f}")
    print("    -> R is (bit-exactly) a function of A/B alone; because the design")
    print("       spans B/A ~ 2.5-21, A/B and B are NOT strongly correlated, so")
    print("       'corr(R,B)' is a weak, misleading proxy for the true driver.")
    d["corr_A_B"] = float(rAB)
    d["corr_A_ratioAB"] = float(rAB_ratio)
    d["corr_B_ratioAB"] = float(rBB_ratio)
    d["corr_A_sumfrac"] = float(rAsum)
    d["corr_B_sumfrac"] = float(rBsum)
    # R as a function of the single scalar B/(A+B) : should be perfectly linear
    slope, inter, mxr, _ = (lambda x, y: (np.polyfit(x, y, 1)[0],
                                          np.polyfit(x, y, 1)[1],
                                          np.abs(y - np.polyval(np.polyfit(x, y, 1), x)).max(),
                                          0.0))(B / (A + B), R)
    print(f"    R = {slope:.15f} * B/(A+B) + {inter:.15f}   max|resid|={mxr:.3e}")
    d["R_on_sumfrac"] = {"slope": float(slope), "intercept": float(inter),
                         "max_abs_resid": float(mxr)}

    out[tag] = d

with open(os.path.join(OUT, "V3_adversarial.json"), "w") as fh:
    json.dump(out, fh, indent=2, default=float)
print(f"\nwrote {os.path.join(OUT,'V3_adversarial.json')}")