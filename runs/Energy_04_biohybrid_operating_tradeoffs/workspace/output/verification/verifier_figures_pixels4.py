#!/usr/bin/env python3
"""Fourth-pass pixel verification with explicit expected x-ranges so that the
legend (fig3) and crossing lines (fig5b) cannot contaminate the measurements.

fig3:   for each of 24 bars (4 regimes x 3 pairs x 2 panels) find the coloured
        pixels inside that bar's own x-range and only within 35 px of the zero
        line, then convert row -> value with the symmetric +-1.05 ylim.
fig5b:  for each series/timepoint take the contiguous coloured run that contains
        the marker centre, so the connecting line crossing the column is ignored.
"""
import numpy as np
import pandas as pd
from PIL import Image

BASE = "/data/user/conversations/conv-995a498d-d5c1-4e1e-99f5-01601e51abe5/project"
FIG = f"{BASE}/output/figures"
C0 = np.array([31, 119, 180]); C1 = np.array([255, 127, 14])
C2 = np.array([44, 160, 44]); C7 = np.array([127, 127, 127])


def load(n):
    return np.asarray(Image.open(f"{FIG}/{n}").convert("RGB")).astype(int)


def runs(m):
    idx = np.where(m)[0]
    if len(idx) == 0:
        return []
    out, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i == p + 1:
            p = i
        else:
            out.append((s, p)); s = p = i
    out.append((s, p))
    return out


def frame(img, x0, x1, y0, y1, frac=0.7):
    reg = img[y0:y1, x0:x1]
    dm = (reg.max(axis=2) < 110)
    h, w = reg.shape[:2]
    cols = np.where(dm.sum(axis=0) > frac * h)[0]
    rows = np.where(dm.sum(axis=1) > frac * w)[0]
    return (x0 + cols.min(), x0 + cols.max(), y0 + rows.min(), y0 + rows.max())


curves = pd.read_csv(f"{BASE}/data/time_response_curves.tsv", sep="\t")
P = curves["promoter_like_signal"].to_numpy(float)
A = curves["condition_a"].to_numpy(float)
Bv = curves["condition_b"].to_numpy(float)
REG = {"A_baseline": slice(0, 259), "B_sweep": slice(259, 294),
       "C_tail": slice(294, 1367)}
names4 = ["A_baseline", "B_sweep", "C_tail", "WHOLE"]


def r(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    return float(((x - x.mean()) * (y - y.mean())).sum() /
                 np.sqrt(((x - x.mean()) ** 2).sum() * ((y - y.mean()) ** 2).sum()))


recomp = {"levels": {}, "diffs": {}}
for nm in names4:
    s = slice(None) if nm == "WHOLE" else REG[nm]
    for pn, x, y in [("prom|a", P[s], A[s]), ("prom|b", P[s], Bv[s]),
                     ("a|b", A[s], Bv[s])]:
        recomp["levels"][f"{nm}:{pn}"] = r(x, y)
        recomp["diffs"][f"{nm}:{pn}"] = r(np.diff(x), np.diff(y))

print("=" * 78)
print("FIG 3 -- 24 bar values measured from pixels vs recomputed correlations")
print("=" * 78)
img3 = load("fig3_coupling_by_regime.png")
H3, W3 = img3.shape[:2]
allrows = []
for panel, (xa, xb) in [("levels", (0, W3 // 2)), ("diffs", (W3 // 2, W3))]:
    L3, R3, T3, B3 = frame(img3, xa + 5, xb, 5, H3 - 5)
    dm = (img3[:, L3 + 3:R3 - 3].max(axis=2) < 110)
    zr = max(range(T3 + 2, B3 - 2), key=lambda rr: dm[rr].sum())
    ppu = (B3 - T3) / 2.10
    # derive the x transform from the four tick marks just below the frame
    tickrow = min(B3 + 3, img3.shape[0] - 1)
    tickband = img3[tickrow:tickrow + 4, L3:R3]
    tdark = (tickband.max(axis=2) < 120).any(axis=0)
    ticks = [int((s + e) / 2) + L3 for (s, e) in runs(tdark) if e - s < 12]
    print(f"{panel}: frame x[{L3},{R3}] y[{T3},{B3}] zero_row={zr} ppu={ppu:.2f}")
    print(f"   tick pixel columns: {ticks}")
    padj = None
    if len(ticks) == 4:
        padj = np.polyfit(np.arange(4), ticks, 1)
        print(f"   x-transform: px = {padj[0]:.2f}*cat + {padj[1]:.2f}")
    else:
        raise SystemExit("tick detection failed")
    for i, pn in enumerate(["prom|a", "prom|b", "a|b"]):
        rgb = {"prom|a": C0, "prom|b": C1, "a|b": C2}[pn]
        m = np.abs(img3 - rgb).max(axis=2) <= 25
        m[:, :L3] = False; m[:, R3:] = False; m[:T3, :] = False; m[B3:, :] = False
        off = (i - 1) * 0.26
        for cat, nm in enumerate(names4):
            cx = padj[0] * cat + padj[1] + off * padj[0]
            halfw = 0.13 * padj[0]
            s, e = int(round(cx - halfw)), int(round(cx + halfw))
            band = m[max(T3, zr - 40):min(B3, zr + 40), s:e + 1]
            rws = np.where(band.any(axis=1))[0]
            if len(rws) == 0:
                allrows.append({"panel": panel, "pair": pn, "regime": nm,
                                "claimed_frac": None, "px_value": None,
                                "note": "no pixels (occluded?)"})
                continue
            rows_abs = rws + max(T3, zr - 40)
            v = max((zr - rows_abs.min()) / ppu, (zr - rows_abs.max()) / ppu, key=abs)
            claim = recomp[panel][f"{nm}:{pn}"]
            allrows.append({"panel": panel, "pair": pn, "regime": nm,
                            "claim_report": None, "recomputed": round(claim, 4),
                            "px_value": round(float(v), 4),
                            "diff": round(float(v) - claim, 4),
                            "x_range": [s, e],
                            "n_px": int(band.sum())})
dfb = pd.DataFrame(allrows)
print(dfb.to_string(index=False))
print("\nmax |pixel - recomputed| over bars measured:",
      dfb["diff"].abs().max())
print("bars with no pixels detected:",
      dfb[dfb.px_value.isna()][["panel", "pair", "regime"]].to_dict("records"))
dfb.to_csv(f"{BASE}/output/verification/fig3_bar_pixels.tsv", sep="\t", index=False)

print()
print("=" * 78)
print("FIG 5b -- error bars: contiguous run containing the marker centre")
print("=" * 78)
img5 = load("fig5_dose_perturbation.png")
H5, W5 = img5.shape[:2]
L5, R5, T5, B5 = frame(img5, W5 // 2, W5, 5, H5 - 5)
pert = pd.read_csv(f"{BASE}/data/perturbation_time_response.tsv", sep="\t")
t = pert["time"].to_numpy(float)
tpix = {tv: L5 + ((tv + 0.3) / 6.6) * (R5 - L5) for tv in t}
series = {"control": C7, "condition_a": C1, "condition_b": C2}

# 1) build a global pixel->value map from the 12 marker centres
marker_rows = {}
for name, rgb in series.items():
    m = np.abs(img5 - rgb).max(axis=2) <= 25
    for tv in t:
        xi = int(round(tpix[tv]))
        win = m[:, xi - 4:xi + 5]
        rr = runs(win.any(axis=1))
        # marker = the run with the largest count of coloured pixels
        best, bestn = None, -1
        for (s, e) in rr:
            n = int(win[s:e + 1].sum())
            if n > bestn:
                bestn, best = n, (s, e)
        marker_rows[(name, float(tv))] = (best[0] + best[1]) / 2, bestn
rows_all, vals_all = [], []
for name in series:
    for k, tv in enumerate(t):
        c, n = marker_rows[(name, float(tv))]
        rows_all.append(c); vals_all.append(float(pert[f"{name}_mean"][k]))
coef = np.polyfit(rows_all, vals_all, 1)
resid = np.array(vals_all) - np.polyval(coef, rows_all)
print(f"pixel->value map from 12 marker centres: slope={coef[0]:.8f} "
      f"intercept={coef[1]:.6f}")
print(f"   max |residual| of marker centre vs known mean = {np.abs(resid).max():.5f}"
      f"  (~{np.abs(resid).max()/abs(coef[0]):.1f} px)")
print(f"   implied panel y-span = {abs(coef[0])*(B5-T5):.4f}")
print("   marker centre residuals (px):",
      np.round(resid / coef[0], 2).tolist())

print()
out = []
for name, rgb in series.items():
    m = np.abs(img5 - rgb).max(axis=2) <= 25
    m[:, :L5 - 2] = False; m[:, R5 + 2:] = False
    m[:T5 - 3, :] = False; m[B5 + 4:, :] = False
    for k, tv in enumerate(t):
        xi = int(round(tpix[tv]))
        win = m[:, xi - 3:xi + 4]
        rr = runs(win.any(axis=1))
        # pick the run containing the marker centre row
        mc, _ = marker_rows[(name, float(tv))]
        pick = None
        for (s, e) in rr:
            if s - 3 <= mc <= e + 3:
                pick = (s, e)
        if pick is None:
            out.append({"series": name, "t": float(tv), "note": "no run at marker"})
            continue
        s, e = pick
        vhi = np.polyval(coef, s)   # top pixel = largest value
        vlo = np.polyval(coef, e)
        sd = float(pert[f"{name}_sd"][k]); mu = float(pert[f"{name}_mean"][k])
        out.append({"series": name, "t": float(tv), "mean": mu, "sd": sd,
                    "pix_top": s, "pix_bot": e, "n_px": int(win[s:e+1].sum()),
                    "val_top": round(float(vhi), 5), "val_bot": round(float(vlo), 5),
                    "half_extent": round((vhi - vlo) / 2, 5),
                    "sd_minus_halfextent": round(sd - (vhi - vlo) / 2, 5)})
o = pd.DataFrame(out)
print(o.to_string(index=False))
o.to_csv(f"{BASE}/output/verification/fig5b_errorbar_pixels.tsv", sep="\t", index=False)

# marker-diameter floor: markers are drawn ms=4 -> diameter in px
# fit half_extent = k*sd + d/2 using the 12 points
ok = o[o.sd.notna()]
A_ = np.column_stack([ok.sd.to_numpy(float), np.ones(len(ok))])
sol, *_ = np.linalg.lstsq(A_, ok.half_extent.to_numpy(float), rcond=None)
print(f"\nhalf_extent ~ {sol[0]:.2f}*sd + {sol[1]:.5f}  "
      f"(slope should be ~1.0 if the bars are the supplied SDs; "
      f"intercept = half the marker radius in value units)")
pred = A_ @ sol
ss = 1 - ((ok.half_extent - pred) ** 2).sum() / ((ok.half_extent - ok.half_extent.mean()) ** 2).sum()
print(f"R^2 of that fit = {ss:.5f}")
print("interpretation: slope near 1 means error bars encode the supplied SDs;")
print("the intercept absorbs the marker disc radius, which is the same for all points.")