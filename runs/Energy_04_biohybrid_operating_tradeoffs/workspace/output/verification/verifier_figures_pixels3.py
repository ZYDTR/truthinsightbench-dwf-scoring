#!/usr/bin/env python3
"""Third-pass pixel verification, fixing legend contamination and zero-line
detection.  Checks:
   fig1(a) shading-boundary pixel columns -> data rows   (3-4 px = ~7 data units)
   fig3    bar heights vs recomputed correlations         (both panels)
   fig4(a) x-axis start row (is it really the "full row range"?)
   fig5(b) error-bar vertical extents vs supplied SDs     (legend excluded)
"""
import numpy as np
import pandas as pd
from PIL import Image

BASE = "/data/user/conversations/conv-995a498d-d5c1-4e1e-99f5-01601e51abe5/project"
FIG = f"{BASE}/output/figures"


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
C = curves["condition_c"].to_numpy(float)
FA = curves["fit_a"].to_numpy(float)
T = curves["time"].to_numpy(float)
REG = {"A_baseline": slice(0, 259), "B_sweep": slice(259, 294),
       "C_tail": slice(294, 1367)}

print("=" * 78)
print("FIG 1a -- shading boundaries: pixel -> data row")
print("=" * 78)
img1 = load("fig1_regime_structure.png")
H, W = img1.shape[:2]
L, R, TOP, BOT = frame(img1, 5, W // 2, 5, H // 2)
print(f"panel (a) frame x[{L},{R}] y[{TOP},{BOT}]  (pixels per data row = "
      f"{(R-L)/1.10/1366:.4f})")
# modal colour per column, classifying by nearest of the 3 known band colours
band = {"A": 255 * 0.87 + np.array([255, 127, 14]) * 0.13,
        "B": 255 * 0.87 + np.array([214, 39, 40]) * 0.13,
        "C": 255 * 0.87 + np.array([44, 160, 44]) * 0.13}
white = np.array([255, 255, 255])
sub = img1[TOP + 5:BOT - 5, L + 1:R]
modal = []
for j in range(sub.shape[1]):
    c = sub[:, j]
    key = (c[:, 0].astype(int) << 16) + (c[:, 1].astype(int) << 8) + c[:, 2].astype(int)
    vals, counts = np.unique(key, return_counts=True)
    k = vals[np.argmax(counts)]
    modal.append(np.array([(k >> 16) & 255, (k >> 8) & 255, k & 255]))
modal = np.array(modal)
names = list(band) + ["white"]
cands = np.array([band[k] for k in band] + [white])
d = np.abs(modal[:, None, :] - cands[None, :, :]).max(axis=2)
lab = np.array([names[i] for i in d.argmin(axis=1)])

xmin, xmax = -0.05 * 1366, 1366 * 1.05
def to_row(px):
    return xmin + px / (R - L) * (xmax - xmin)

print("runs of constant band label (px -> data rows):")
for (s, e) in runs(lab != "."):
    seg = lab[s:e + 1]
    ch = [0] + [i for i in range(1, len(seg)) if seg[i] != seg[i - 1]]
    for k, i in enumerate(ch):
        j = ch[k + 1] if k + 1 < len(ch) else len(seg)
        if j - i > 4 and seg[i] != "white":
            print(f"   {seg[i]}  rows [{to_row(s+i):7.1f} .. {to_row(s+j):7.1f}]")

print("\ntail of panel (a)-style data: min/argmin of regime C")
Pc = P[REG["C_tail"]]
print(f"   regime C P: min={Pc.min()} at global row {294+int(np.argmin(Pc))}, "
      f"last={Pc[-1]} at row 1366, first={Pc[0]} at row 294")
print(f"   regime C A: min={A[REG['C_tail']].min()}, max={A[REG['C_tail']].max()}")

print()
print("=" * 78)
print("FIG 3 -- bar heights (pixels) vs recomputed correlations")
print("=" * 78)
C0 = np.array([31, 119, 180]); C1 = np.array([255, 127, 14]); C2 = np.array([44, 160, 44])
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

img3 = load("fig3_coupling_by_regime.png")
H3, W3 = img3.shape[:2]
for panel, (xa, xb) in [("levels", (0, W3 // 2)), ("diffs", (W3 // 2, W3))]:
    L3, R3, T3, B3 = frame(img3, xa + 5, xb, 5, H3 - 5)
    # zero line = row with most dark pixels, excluding the spines
    dm = (img3[:, L3 + 3:R3 - 3].max(axis=2) < 110)
    zr = max(range(T3 + 2, B3 - 2), key=lambda rr: dm[rr].sum())
    px_per_unit = (B3 - T3) / 2.10
    print(f"\n{panel}: frame x[{L3},{R3}] y[{T3},{B3}]  zero-row={zr} "
          f"({px_per_unit:.2f} px per unit; symmetric ylim +-1.05)")
    print(f"   sanity: (zero-row - top)/(B3-T3) = {(zr-T3)/(B3-T3):.4f} (expect 0.5)")
    for pn, rgb in [("prom|a", C0), ("prom|b", C1), ("a|b", C2)]:
        m = np.abs(img3 - rgb).max(axis=2) <= 25
        m[:, :L3] = False; m[:, R3:] = False
        m[:T3, :] = False; m[B3:, :] = False
        got = []
        for (s, e) in runs(m.sum(axis=0) > 0):
            if e - s < 12:
                continue
            rws = np.where(m[:, s:e + 1].any(axis=1))[0]
            v = max((zr - rws.min()) / px_per_unit, (zr - rws.max()) / px_per_unit,
                    key=abs)
            got.append(round(float(v), 3))
        want = [round(recomp[panel][f"{nm}:{pn}"], 3) for nm in names4]
        dd = [round(g - w, 3) for g, w in zip(got, want)] if len(got) == len(want) else None
        print(f"   {pn:7s} px-values={got}")
        print(f"           recomputed={want}   diff={dd}")

print()
print("=" * 78)
print("FIG 4a -- is the x-range really the 'full row range'?")
print("=" * 78)
img4 = load("fig4_raw_vs_fitted.png")
H4, W4 = img4.shape[:2]
L4, R4, T4, B4 = frame(img4, 5, W4 // 2, 5, H4 - 5)
print(f"panel (a) frame x[{L4},{R4}] y[{T4},{B4}]")
# leftmost/rightmost blue pixels (raw data) on a symlog y axis
mblu = np.abs(img4 - C0).max(axis=2) <= 40
mblu[:, :L4 + 1] = False; mblu[:, R4:] = False
mblu[:T4, :] = False; mblu[B4:, :] = False
cols = np.where(mblu.any(axis=0))[0]
print(f"   leftmost blue pixel column={cols.min()}  rightmost={cols.max()}")
# map: pandas/matplotlib default x-margin 5% of data range
xmin4, xmax4 = 259 - 0.05 * (1366 - 259), 1366 + 0.05 * (1366 - 259)
def to_row4(px):
    return xmin4 + (px - L4) / (R4 - L4) * (xmax4 - xmin4)
print(f"   -> plotted x range approx [{to_row4(cols.min()):.0f}, {to_row4(cols.max()):.0f}]"
      f"  (if xlim were 0..1366 the map differs; leftmost data row should be 259)")
# also check number of distinct x columns with data -> all rows 259..1366 present
print(f"   fraction of panel columns with data: {len(cols)/(R4-L4):.3f}")

print()
print("=" * 78)
print("FIG 5b -- error-bar extents vs supplied SDs (legend rows excluded)")
print("=" * 78)
img5 = load("fig5_dose_perturbation.png")
H5, W5 = img5.shape[:2]
L5, R5, T5, B5 = frame(img5, W5 // 2, W5, 5, H5 - 5)
pert = pd.read_csv(f"{BASE}/data/perturbation_time_response.tsv", sep="\t")
t = pert["time"].to_numpy(float)
tpix = {tv: L5 + ((tv + 0.3) / 6.6) * (R5 - L5) for tv in t}
print(f"right panel frame x[{L5},{R5}] y[{T5},{B5}]; marker x px:",
      {k: round(v) for k, v in tpix.items()})
series = {"control": np.array([127, 127, 127]), "condition_a": C1, "condition_b": C2}
records = []
for name, rgb in series.items():
    m = np.abs(img5 - rgb).max(axis=2) <= 25
    m[:, :L5 - 2] = False; m[:, R5 + 2:] = False
    m[:T5 - 3, :] = False; m[B5 + 4:, :] = False
    # legend rows: rows whose coloured run inside the panel is very wide
    wide_rows = np.zeros(img5.shape[0], bool)
    for rr in range(img5.shape[0]):
        for (s, e) in runs(m[rr]):
            if e - s > 14:
                wide_rows[rr] = True
    sd = pert[f"{name}_sd"].to_numpy(float)
    mean = pert[f"{name}_mean"].to_numpy(float)
    print(f"\n{name}: legend-sample rows = "
          f"{[(s,e) for s,e in runs(wide_rows)]}")
    for k, tv in enumerate(t):
        xi = int(round(tpix[tv]))
        win = m[:, xi - 2:xi + 3].copy()
        win[wide_rows] = False
        rws = np.where(win.any(axis=1))[0]
        if len(rws) == 0:
            print(f"   t={tv}: no pixels")
            continue
        lo, hi = int(rws.min()), int(rws.max())
        records.append({"series": name, "t": float(tv), "sd": float(sd[k]),
                        "extent_px": hi - lo + 1, "mean": float(mean[k]),
                        "y_lo": lo, "y_hi": hi})
        print(f"   t={tv}: mean={mean[k]:.5f} sd={sd[k]:.5f} extent={hi-lo+1:3d}px "
              f"(y {lo}..{hi})")

rec = pd.DataFrame(records)
# marker diameter floor: smallest extent among the smallest SDs
sub_m = rec[rec.sd < 0.0025]
floor = float(sub_m.extent_px.min()) if len(sub_m) else 0.0
print(f"\nmarker-diameter floor estimate (min extent among sd<0.0025) = {floor:.0f} px")
rec["corrected"] = rec.extent_px - floor
rec["two_sd"] = 2 * rec.sd
ok = rec[rec.two_sd > 0.004]
b = np.polyfit(ok.two_sd, ok.corrected, 1)
pred = np.polyval(b, ok.two_sd)
ss_res = ((ok.corrected - pred) ** 2).sum()
ss_tot = ((ok.corrected - ok.corrected.mean()) ** 2).sum()
print(f"regression of (extent - {floor:.0f}px) on 2*sd, sd>0.002:")
print(f"   slope={b[0]:.1f} px per unit  intercept={b[1]:.2f} px  R^2={1-ss_res/ss_tot:.4f}")
print("   (a slope near the panel's px-per-unit scale and intercept near 0 means"
      " the bars encode the supplied SDs)")
# implied panel value span if slope is the true scale
print(f"   implied y-span of panel: {(B5-T5)/b[0]:.4f}  "
      f"(data mean span = {pert[['control_mean','condition_a_mean','condition_b_mean']].values.max()-pert[['control_mean','condition_a_mean','condition_b_mean']].values.min():.4f})")
print(rec.to_string(index=False))
rec.to_csv(f"{BASE}/output/verification/fig5b_errorbar_pixels.tsv", sep="\t", index=False)