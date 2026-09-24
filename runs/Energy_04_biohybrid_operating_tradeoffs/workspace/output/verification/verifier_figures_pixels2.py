#!/usr/bin/env python3
"""Refined pixel checks: fig1(a) shading boundaries, fig3 bars vs correlations,
fig5(b) error-bar extents vs supplied SDs.  Independent of the first pixel
script and of output/analysis/analysis.py."""
import numpy as np
import pandas as pd
from PIL import Image

BASE = "/data/user/conversations/conv-995a498d-d5c1-4e1e-99f5-01601e51abe5/project"
FIG = f"{BASE}/output/figures"
C0 = np.array([31, 119, 180])
C1 = np.array([255, 127, 14])
C2 = np.array([44, 160, 44])
C3 = np.array([214, 39, 40])
C7 = np.array([127, 127, 127])


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


# ---------------------------------------------------------------------------
print("=" * 78)
print("FIG 1a -- regime shading boundaries (modal background colour per column)")
print("=" * 78)
img1 = load("fig1_regime_structure.png")
H, W = img1.shape[:2]
L, R, TOP, BOT = frame(img1, 5, W // 2, 5, H // 2)
print(f"panel (a) frame x[{L},{R}] y[{TOP},{BOT}]")
sub = img1[TOP + 3:BOT - 2, L + 1:R]
# modal colour per column
cols = []
for j in range(sub.shape[1]):
    c = sub[:, j]
    key = (c[:, 0].astype(int) << 16) + (c[:, 1].astype(int) << 8) + c[:, 2].astype(int)
    vals, counts = np.unique(key, return_counts=True)
    k = vals[np.argmax(counts)]
    cols.append(((k >> 16) & 255, (k >> 8) & 255, k & 255))
cols = np.array(cols)
uniq, cnt = np.unique(cols, axis=0, return_counts=True)
order = np.argsort(-cnt)
print("modal colours (rgb, n columns):")
for i in order[:8]:
    print("   ", tuple(uniq[i]), cnt[i])
# identify the band colours as the ones that are NOT white/black/grid
band_cols = [tuple(u) for u, c in zip(uniq, cnt) if c > 8 and not (u > 250).all()]
print("band colours chosen:", band_cols)


def label_of(c):
    for k, bc in enumerate(band_cols):
        if max(abs(c[0] - bc[0]), abs(c[1] - bc[1]), abs(c[2] - bc[2])) <= 4:
            return "ABCDE"[k]
    return "."


lab = np.array([label_of(c) for c in cols])
print("column labels runs (px -> data row):")
xmin, xmax = -0.05 * 1366, 1366 * 1.05
for (s, e) in runs(lab != "."):
    l = lab[s:e + 1]
    # split into constant-label sub-runs
    ch = [0] + [i for i in range(1, len(l)) if l[i] != l[i - 1]]
    for k, i in enumerate(ch):
        j = ch[k + 1] if k + 1 < len(ch) else len(l)
        if j - i > 4:
            x0d = xmin + (s + i) / (R - L) * (xmax - xmin)
            x1d = xmin + (s + j) / (R - L) * (xmax - xmin)
            print(f"   label={l[i]}  row [{x0d:7.1f} .. {x1d:7.1f}]  width={x1d-x0d:7.1f}")

# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("FIG 3 -- bar heights vs independently recomputed correlations")
print("=" * 78)
curves = pd.read_csv(f"{BASE}/data/time_response_curves.tsv", sep="\t")
P, A, Bv = (curves[c].to_numpy(float) for c in
            ["promoter_like_signal", "condition_a", "condition_b"])
REG = {"A_baseline": slice(0, 259), "B_sweep": slice(259, 294),
       "C_tail": slice(294, 1367)}
names = ["A_baseline", "B_sweep", "C_tail", "WHOLE"]


def r(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    return float(((x - x.mean()) * (y - y.mean())).sum() /
                 np.sqrt(((x - x.mean()) ** 2).sum() * ((y - y.mean()) ** 2).sum()))


recomputed = {"levels": {}, "diffs": {}}
for nm in names:
    s = slice(None) if nm == "WHOLE" else REG[nm]
    for pn, x, y in [("prom|a", P[s], A[s]), ("prom|b", P[s], Bv[s]),
                     ("a|b", A[s], Bv[s])]:
        recomputed["levels"][f"{nm}:{pn}"] = r(x, y)
        recomputed["diffs"][f"{nm}:{pn}"] = r(np.diff(x), np.diff(y))

img3 = load("fig3_coupling_by_regime.png")
H3, W3 = img3.shape[:2]
for panel, (xa, xb) in [("levels", (0, W3 // 2)), ("diffs", (W3 // 2, W3))]:
    L3, R3, T3, B3 = frame(img3, xa + 5, xb, 5, H3 - 5)
    # zero line row
    dm = (img3[:, L3 + 2:R3 - 2].max(axis=2) < 110)
    zr = max(range(T3, B3), key=lambda rr: dm[rr].mean())
    scale = 2.10 / (B3 - T3)      # ylim (-1.05, 1.05)
    print(f"\n{panel}: frame x[{L3},{R3}] y[{T3},{B3}] zero-row={zr} "
          f"({2.10/(B3-T3):.5f} per px)")
    for pn, rgb in [("prom|a", C0), ("prom|b", C1), ("a|b", C2)]:
        m = np.abs(img3 - rgb).max(axis=2) <= 25
        m[:, :L3] = False; m[:, R3:] = False
        m[:T3, :] = False; m[B3:, :] = False
        got = []
        for (s, e) in runs(m.sum(axis=0) > 0):
            if e - s < 12:
                continue
            rws = np.where(m[:, s:e + 1].any(axis=1))[0]
            v = max((zr - rws.min()) * scale, (zr - rws.max()) * scale,
                    key=abs)
            got.append(round(float(v), 3))
        want = [round(recomputed[panel][f"{nm}:{pn}"], 3) for nm in names]
        print(f"   {pn:7s} pixels={got}  recomputed={want}")
        if len(got) == len(want):
            print(f"           diff={[round(g-w,3) for g,w in zip(got,want)]}")

# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("FIG 5b -- error bars: pixel extents vs supplied SDs")
print("=" * 78)
img5 = load("fig5_dose_perturbation.png")
H5, W5 = img5.shape[:2]
L5, R5, T5, B5 = frame(img5, W5 // 2, W5, 5, H5 - 5)
print(f"right panel frame x[{L5},{R5}] y[{T5},{B5}]")
pert = pd.read_csv(f"{BASE}/data/perturbation_time_response.tsv", sep="\t")
t = pert["time"].to_numpy(float)
# x position of ticks: data 0..6 with 5% margins -> 0 at 4.55% and 6 at 95.45%
tpix = {}
for tv in t:
    frac = (tv - (-0.3)) / (6.6)
    tpix[tv] = L5 + frac * (R5 - L5)
print("expected marker x pixels:", {k: round(v) for k, v in tpix.items()})
rows_all = {}
for name, rgb in [("control", C7), ("condition_a", C1), ("condition_b", C2)]:
    m = np.abs(img5 - rgb).max(axis=2) <= 25
    m[:, :L5 - 2] = False; m[:, R5 + 2:] = False
    m[:T5 - 3, :] = False; m[B5 + 4:, :] = False
    ext = {}
    for tv, xp in tpix.items():
        xi = int(round(xp))
        win = m[:, max(0, xi - 2):xi + 3]
        rws = np.where(win.any(axis=1))[0]
        if len(rws):
            ext[tv] = (int(rws.min()), int(rws.max()))
    rows_all[name] = ext
    print(f"\n{name}: marker-column y extents {ext}")

# fit a linear pixel->value map from the MEAN marker positions.
# marker vertical centre = the disk; take it as the centroid of the widest row band
# Instead: use the mean value where the line passes -> use known means & solve.
# Use two anchor series: control mean at t=0 (0.32243) and condition_b mean at t=6 (0.2295)
# find marker centres as rows with max horizontal width in a narrow window
def marker_centre(name, tv):
    rgb = {"control": C7, "condition_a": C1, "condition_b": C2}[name]
    m = np.abs(img5 - rgb).max(axis=2) <= 25
    xi = int(round(tpix[tv]))
    win = m[:, xi - 4:xi + 5]
    width = win.sum(axis=1)
    # the marker row band = contiguous rows with high width
    rows = np.where(width > 0)[0]
    if len(rows) == 0:
        return None
    # centre of mass weighted by width
    return float((rows * width[rows]).sum() / width[rows].sum())


anchors = []
for name, tv, val in [("control", 0.0, pert["control_mean"][0]),
                      ("condition_b", 6.0, pert["condition_b_mean"][3]),
                      ("control", 6.0, pert["control_mean"][3]),
                      ("condition_a", 6.0, pert["condition_a_mean"][3])]:
    c = marker_centre(name, tv)
    anchors.append((c, float(val)))
print("\nanchors (row, value):", [(round(c, 1), v) for c, v in anchors])
rows_a = np.array([c for c, _ in anchors])
vals_a = np.array([v for _, v in anchors])
coef = np.polyfit(rows_a, vals_a, 1)
print(f"pixel->value map: value = {coef[0]:.8f}*row + {coef[1]:.6f}  "
      f"(=> {(abs(coef[0])*(B5-T5)):.4f} per panel height)")
for name in ["control", "condition_a", "condition_b"]:
    sd = pert[f"{name}_sd"].to_numpy(float)
    means = pert[f"{name}_mean"].to_numpy(float)
    print(f"\n{name}:")
    for k, tv in enumerate(t):
        xi = int(round(tpix[tv]))
        rgb = {"control": C7, "condition_a": C1, "condition_b": C2}[name]
        m = np.abs(img5 - rgb).max(axis=2) <= 25
        win = m[:, xi - 2:xi + 3]
        rws = np.where(win.any(axis=1))[0]
        lo, hi = rws.min(), rws.max()
        vlo, vhi = np.polyval(coef, hi), np.polyval(coef, lo)
        print(f"  t={tv}: mean={means[k]:.5f} sd={sd[k]:.5f} "
              f"| marker_centre_row={marker_centre(name, tv):.1f} "
              f"| bar_extent_val=[{vlo:.5f},{vhi:.5f}] "
              f"half-extent={(vhi-vlo)/2:.5f}")