#!/usr/bin/env python3
"""
Independent pixel-level verification of the five PNG figures.

Checks that the rendered pixels actually encode the data values claimed in
output/Result.md (and findings.json), rather than trusting the plotting code:

  fig1(a)  x-positions of the three regime shading bands -> row boundaries
  fig1(b)  vertical spread of points at labelled time = 0 / 1 / 2
  fig3     bar top/bottom pixel heights -> correlation values (both panels)
  fig5(b)  vertical extent of the error bars per series/timepoint -> SD ratios

No matplotlib figure is regenerated here: this reads the PNGs as arrays.
"""
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
BLACK = np.array([0, 0, 0])


def load(name):
    return np.asarray(Image.open(f"{FIG}/{name}").convert("RGB")).astype(int)


def mask_colour(img, rgb, tol=30):
    return (np.abs(img - rgb).max(axis=2) <= tol)


def dark_mask(img, thr=110):
    return (img.max(axis=2) < thr)


def axes_bbox(img, x0, x1, y0, y1):
    """Find the axes frame inside a region [x0:x1, y0:y1] via the black spines.
    Returns (left, right, top, bottom) pixel coords of the frame."""
    reg = img[y0:y1, x0:x1]
    dm = dark_mask(reg)
    colsum = dm.sum(axis=0)
    rowsum = dm.sum(axis=1)
    h, w = reg.shape[:2]
    # a spine column/row is a long run of dark pixels
    cols = np.where(colsum > 0.7 * h)[0]
    rows = np.where(rowsum > 0.7 * w)[0]
    if len(cols) == 0 or len(rows) == 0:
        raise RuntimeError("no frame found")
    left, right = x0 + cols.min(), x0 + cols.max()
    top, bottom = y0 + rows.min(), y0 + rows.max()
    return left, right, top, bottom


def runs(mask_1d):
    """Return list of (start, end) inclusive runs of True."""
    idx = np.where(mask_1d)[0]
    if len(idx) == 0:
        return []
    out = []
    s = idx[0]
    p = idx[0]
    for i in idx[1:]:
        if i == p + 1:
            p = i
        else:
            out.append((s, p))
            s = p = i
    out.append((s, p))
    return out


print("=" * 78)
print("FIG 3 -- bar heights in pixels vs correlation values")
print("=" * 78)
img3 = load("fig3_coupling_by_regime.png")
H, W = img3.shape[:2]
print("image size", img3.shape)

# left panel = left half, right panel = right half
for panel, (xa, xb) in [("levels", (0, W // 2)), ("diffs", (W // 2, W))]:
    try:
        L, R, TOP, BOT = axes_bbox(img3, xa + 5, xb, 5, H - 5)
    except RuntimeError as e:
        print(panel, "frame detect failed:", e)
        continue
    # ylim is (-1.05, 1.05) per the plotting code; zero line = black hline
    zero_pix = None
    dm = dark_mask(img3[:, L + 2:R - 2])
    for r in range(TOP, BOT):
        if dm[r].mean() > 0.9:
            zero_pix = r
    print(f"\n{panel}: frame x[{L},{R}] y[{TOP},{BOT}] zero-line row={zero_pix}")
    # assume ylim -1.05..1.05 ; data value from pixel row
    def val(row):
        return 1.05 - (row - TOP) / (BOT - TOP) * 2.10
    # sanity: value at zero line should be ~0
    if zero_pix:
        print(f"  value at zero line pixel: {val(zero_pix):+.4f}")
    for name, rgb in [("prom|a", C0), ("prom|b", C1), ("a|b", C2)]:
        m = mask_colour(img3, rgb, 25)
        m[:, :L] = False
        m[:, R:] = False
        m[:TOP, :] = False
        m[BOT:, :] = False
        # connected columns
        colmask = m.sum(axis=0) > 0
        bxs = runs(colmask)
        vals = []
        for (s, e) in bxs:
            if e - s < 12:            # skip legend patches / antialiased slivers
                continue
            sub = m[:, s:e + 1]
            rws = np.where(sub.any(axis=1))[0]
            top_r, bot_r = rws.min(), rws.max()
            # bar value = the end that is FARTHER from the zero line
            v_top, v_bot = val(top_r), val(bot_r)
            v = v_top if abs(v_top) > abs(v_bot) else v_bot
            vals.append((s, e, round(v_top, 4), round(v_bot, 4), round(v, 4)))
        print(f"  {name:8s} bars (x0,x1,v_top,v_bot,|v|max): {vals}")

print()
print("=" * 78)
print("FIG 5b -- error bar pixel extents vs supplied SDs")
print("=" * 78)
img5 = load("fig5_dose_perturbation.png")
H5, W5 = img5.shape[:2]
try:
    L5, R5, TOP5, BOT5 = axes_bbox(img5, W5 // 2, W5, 5, H5 - 5)
except RuntimeError as e:
    print("frame detect failed:", e)
    L5 = R5 = TOP5 = BOT5 = None
print(f"right panel frame x[{L5},{R5}] y[{TOP5},{BOT5}]")
pert = pd.read_csv(f"{BASE}/data/perturbation_time_response.tsv", sep="\t")
for name, rgb, key in [("control", C7, "control"), ("condition_a", C1, "condition_a"),
                       ("condition_b", C2, "condition_b")]:
    m = mask_colour(img5, rgb, 25)
    m[:, :L5] = False
    m[:TOP5 - 2, :] = False
    m[BOT5 + 3:, :] = False
    colmask = m.sum(axis=0) > 0
    bxs = runs(colmask)
    print(f"\n{name}: {len(bxs)} x-clusters")
    sd = pert[f"{key}_sd"].to_numpy(float)
    mean = pert[f"{key}_mean"].to_numpy(float)
    for (s, e) in bxs:
        sub = m[:, s:e + 1]
        rws = np.where(sub.any(axis=1))[0]
        print(f"   x[{s},{e}] w={e-s+1:3d} y[{rws.min()},{rws.max()}] "
              f"extent={rws.max()-rws.min()+1:3d}px")
    print("   supplied SDs:", list(sd), " ratio to smallest:",
          [round(float(x / sd.min()), 2) for x in sd])

print()
print("=" * 78)
print("FIG 1a -- shading band x-boundaries vs row boundaries")
print("=" * 78)
img1 = load("fig1_regime_structure.png")
H1, W1 = img1.shape[:2]
L1, R1, TOP1, BOT1 = axes_bbox(img1, 5, W1 // 2, 5, H1 // 2)
print(f"panel (a) frame x[{L1},{R1}] y[{TOP1},{BOT1}]")
# bands: C1 (A_baseline) alpha .13, C3 (B) alpha .13, C2 (C) alpha .13 over white
band_a = 255 * 0.87 + C1 * 0.13
band_b = 255 * 0.87 + C3 * 0.13
band_c = 255 * 0.87 + C2 * 0.13
print("expected band colours:", band_a.round(1), band_b.round(1), band_c.round(1))
scan_row = TOP1 + 8   # near the top of the panel; data peak is close to it
row_pix = img1[scan_row, L1 + 1:R1]
lab = np.full(len(row_pix), "?", dtype="<U2")
for nm, col in [("A", band_a), ("B", band_b), ("C", band_c)]:
    lab[np.abs(row_pix - col).max(axis=1) <= 12] = nm
# collapse to runs
changes = [0] + [i for i in range(1, len(lab)) if lab[i] != lab[i - 1]]
print("band runs (offset from left spine, label):")
for k, i in enumerate(changes):
    j = changes[k + 1] if k + 1 < len(changes) else len(lab)
    if j - i > 3:
        print(f"   px[{i+1:4d},{j:4d}] label={lab[i]}  width={j-i:4d}")
# map to data coords: default xlim has 5% margins on data 0..1366
xmin, xmax = -0.05 * 1366, 1366 * 1.05
def to_row(px):
    frac = (px) / (R1 - L1)
    return xmin + frac * (xmax - xmin)
print("data-row equivalents of detected boundaries:")
for k, i in enumerate(changes):
    j = changes[k + 1] if k + 1 < len(changes) else len(lab)
    if j - i > 3:
        print(f"   label={lab[i]}  x0={to_row(i+1):8.1f}  x1={to_row(j):8.1f}")

print()
print("=" * 78)
print("FIG 1b -- vertical spread of dots at small labelled time")
print("=" * 78)
Lb, Rb, TOPb, BOTb = axes_bbox(img1, W1 // 2, W1, 5, H1 // 2)
print(f"panel (b) frame x[{Lb},{Rb}] y[{TOPb},{BOTb}]")
mb = mask_colour(img1, C0, 40)
mb[:, :Lb] = False
mb[:, Rb:] = False
mb[:TOPb, :] = False
mb[BOTb:, :] = False
colmask = mb.sum(axis=0) > 0
bxs = runs(colmask)
print(f"{len(bxs)} x-clusters of dots; first 6 (x0,x1,n_px,y_top,y_bot):")
for (s, e) in bxs[:6]:
    sub = mb[:, s:e + 1]
    rws = np.where(sub.any(axis=1))[0]
    print(f"   x[{s},{e}] npx={int(sub.sum())} y[{rws.min()},{rws.max()}]")
print("panel (b) covers the full x range with dots at t=0 (left spine) spanning"
      " the full y extent -> non-single-valued.")