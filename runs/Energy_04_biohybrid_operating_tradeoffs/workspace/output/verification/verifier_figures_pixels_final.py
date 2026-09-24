#!/usr/bin/env python3
"""FINAL pixel verification pass.

Key fixes over the earlier passes:
  * fig3: horizontal grid lines are drawn OVER the bars and split a bar's
    pixel run into pieces; runs separated by <= 6 px are merged before the
    bar top is read.
  * fig1a: sub-pixel edge detection on the shading bands using the colour
    channels that actually differ between adjacent bands
    (green for A|B, red for B|C), instead of nearest-colour classification.
  * fig5b: error-bar runs measured in the marker columns; the legend is at
    rows ~300-345 and cols ~636-750 and is excluded by requiring the run to
    contain the marker centre row implied by the data mean.
"""
import numpy as np
import pandas as pd
from PIL import Image

BASE = "/data/user/conversations/conv-995a498d-d5c1-4e1e-99f5-01601e51abe5/project"
FIG = f"{BASE}/output/figures"
C0 = np.array([31, 119, 180]); C1 = np.array([255, 127, 14])
C2 = np.array([44, 160, 44]); C3 = np.array([214, 39, 40]); C7 = np.array([127, 127, 127])


def load(n):
    return np.asarray(Image.open(f"{FIG}/{n}").convert("RGB")).astype(int)


def runs(m, merge_gap=0):
    idx = np.where(m)[0]
    if len(idx) == 0:
        return []
    out, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i <= p + 1 + merge_gap:
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


def rr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    return float(((x - x.mean()) * (y - y.mean())).sum() /
                 np.sqrt(((x - x.mean()) ** 2).sum() * ((y - y.mean()) ** 2).sum()))


recomp = {"levels": {}, "diffs": {}}
for nm in names4:
    s = slice(None) if nm == "WHOLE" else REG[nm]
    for pn, x, y in [("prom|a", P[s], A[s]), ("prom|b", P[s], Bv[s]),
                     ("a|b", A[s], Bv[s])]:
        recomp["levels"][f"{nm}:{pn}"] = rr(x, y)
        recomp["diffs"][f"{nm}:{pn}"] = rr(np.diff(x), np.diff(y))

# ===========================================================================
print("=" * 78)
print("FIG 1a -- sub-pixel shading-band edges -> regime boundaries")
print("=" * 78)
img1 = load("fig1_regime_structure.png")
H1, W1 = img1.shape[:2]
L1, R1, T1, B1 = frame(img1, 5, W1 // 2, 5, H1 // 2)
lastrow = 1366
xmin, xmax = lastrow - 0.05 * lastrow, lastrow + 0.05 * lastrow


def to_row(px):
    return xmin + (px - L1) / (R1 - L1) * (xmax - xmin)


# column colour = mean over rows that are neither the data line nor a gridline
sub = img1[T1 + 4:B1 - 4, L1 + 1:R1]
prof = []
for j in range(sub.shape[1]):
    c = sub[:, j]
    keep = ~((np.abs(c - C0).max(axis=1) <= 60) | (np.abs(c - 127).max(axis=1) <= 60))
    prof.append(c[keep].mean(axis=0) if keep.sum() > 10 else c.mean(axis=0))
prof = np.array(prof)
G, R = prof[:, 1], prof[:, 0]
# A|B edge: green drops (238 -> 227);  B|C edge: red drops (250 -> 228)
gi = np.where(G > 0.5 * (G.max() + G.min()))[0]        # band A columns (high green)
ai = np.where(R > 0.5 * (R.max() + R.min()))[0]        # band A/B columns (high red)
print(f"high-green columns (band A): {gi.min() if len(gi) else None}..{gi.max() if len(gi) else None}")
print(f"high-red  columns (bands A,B): {ai.min() if len(ai) else None}..{ai.max() if len(ai) else None}")
# refine edges by linear interpolation of the transition
def edge(profile, lo, hi, rising=None):
    """first column index where profile crosses the midpoint of [lo,hi]"""
    mid = 0.5 * (lo + hi)
    for j in range(1, len(profile)):
        if (profile[j - 1] - mid) * (profile[j] - mid) < 0:
            f = (mid - profile[j - 1]) / (profile[j] - profile[j - 1])
            return j - 1 + f
    return None


e_ab = edge(G, G[:200].mean(), G[-200:].mean())
e_bc = edge(R, R[:200].mean(), R[-200:].mean())
print(f"A|B edge at pixel col {e_ab:.2f} -> data row {to_row(L1 + 1 + e_ab):.2f}"
      f"   (report claims 258/259)")
print(f"B|C edge at pixel col {e_bc:.2f} -> data row {to_row(L1 + 1 + e_bc):.2f}"
      f"   (report claims 293/294)")
print(f"pixel resolution = {(xmax-xmin)/(R1-L1):.3f} data rows per px")

# ===========================================================================
print()
print("=" * 78)
print("FIG 3 -- bar heights from merged pixel runs")
print("=" * 78)
img3 = load("fig3_coupling_by_regime.png")
H3, W3 = img3.shape[:2]
outr = []
for panel, (xa, xb) in [("levels", (0, W3 // 2)), ("diffs", (W3 // 2, W3))]:
    L3, R3, T3, B3 = frame(img3, xa + 5, xb, 5, H3 - 5)
    dm = (img3[:, L3 + 3:R3 - 3].max(axis=2) < 110)
    zr = max(range(T3 + 2, B3 - 2), key=lambda r: dm[r].sum())
    ppu = (B3 - T3) / 2.10
    tk = min(B3 + 3, H3 - 1)
    ticks = [int((s + e) / 2) + L3
             for (s, e) in runs((img3[tk:tk + 4, L3:R3].max(axis=2) < 120).any(axis=0))
             if e - s < 12]
    padj = np.polyfit(np.arange(4), ticks, 1)
    print(f"\n{panel}: frame x[{L3},{R3}] y[{T3},{B3}] zero={zr} ppu={ppu:.2f} ticks={ticks}")
    for i, pn in enumerate(["prom|a", "prom|b", "a|b"]):
        rgb = {"prom|a": C0, "prom|b": C1, "a|b": C2}[pn]
        m = np.abs(img3 - rgb).max(axis=2) <= 25
        m[:, :L3] = False; m[:, R3:] = False; m[:T3, :] = False; m[B3:, :] = False
        off = (i - 1) * 0.26
        for cat, nm in enumerate(names4):
            cx = int(round(padj[0] * cat + padj[1] + off * padj[0]))
            # grid lines split the bar: merge runs separated by <= 6 px
            rruns = runs(m[:, cx - 1:cx + 2].any(axis=1), merge_gap=6)
            cand = [q for q in rruns if q[0] - 2 <= zr <= q[1] + 2]
            if not cand:
                outr.append({"panel": panel, "pair": pn, "regime": nm, "px_value": None,
                             "recomputed": round(recomp[panel][f"{nm}:{pn}"], 4),
                             "note": f"no run at zero line; runs={rruns}"})
                continue
            s, e = max(cand, key=lambda q: q[1] - q[0])
            v = (zr - s) / ppu if abs(zr - s) > abs(zr - e) else (zr - e) / ppu
            claim = recomp[panel][f"{nm}:{pn}"]
            outr.append({"panel": panel, "pair": pn, "regime": nm,
                         "px_value": round(float(v), 4),
                         "recomputed": round(claim, 4),
                         "diff": round(float(v) - claim, 4),
                         "run": [int(s), int(e)], "col": cx})
d = pd.DataFrame(outr)
print(d.to_string(index=False))
g = d[d.px_value.notna()]
print(f"\n{len(g)}/24 bars measured; max |diff| = {g['diff'].abs().max():.4f}; "
      f"mean |diff| = {g['diff'].abs().mean():.4f}; 1 px = {2.10/(B3-T3):.5f}")
print("occluded bars:", d[d.px_value.isna()][["panel", "pair", "regime"]].to_dict("records"))
d.to_csv(f"{BASE}/output/verification/fig3_bar_pixels.tsv", sep="\t", index=False)

# ===========================================================================
print()
print("=" * 78)
print("FIG 5b -- error bars vs supplied SDs (analytic autoscale map)")
print("=" * 78)
img5 = load("fig5_dose_perturbation.png")
H5, W5 = img5.shape[:2]
L5, R5, T5, B5 = frame(img5, W5 // 2 + 5, W5, 5, H5 - 5)
pert = pd.read_csv(f"{BASE}/data/perturbation_time_response.tsv", sep="\t")
tvals = pert["time"].to_numpy(float)
series = {"control": C7, "condition_a": C1, "condition_b": C2}
tpix = {float(tv): L5 + ((tv + 0.3) / 6.6) * (R5 - L5) for tv in tvals}
allv = []
for c in series:
    allv += list(pert[f"{c}_mean"] - pert[f"{c}_sd"]) + list(pert[f"{c}_mean"] + pert[f"{c}_sd"])
lo, hi = min(allv), max(allv)
mg = 0.05 * (hi - lo)
ylo, yhi = lo - mg, hi + mg


def val_of(row):
    return yhi - (row - T5) / (B5 - T5) * (yhi - ylo)


print(f"analytic autoscale limits [{ylo:.6f}, {yhi:.6f}] span {yhi-ylo:.6f}")
print(f"panel frame y[{T5},{B5}]; marker cols "
      f"{ {k: round(v) for k, v in tpix.items()} }")
records = []
for name, rgb in series.items():
    mm = np.abs(img5 - rgb).max(axis=2) <= 25
    mm[:, :L5] = False; mm[:, R5:] = False; mm[:T5, :] = False; mm[B5:, :] = False
    for k, tv in enumerate(tvals):
        xi = int(round(tpix[float(tv)]))
        mu = float(pert[f"{name}_mean"][k]); sd = float(pert[f"{name}_sd"][k])
        mrow = (yhi - mu) / (yhi - ylo) * (B5 - T5) + T5   # predicted marker row
        win = mm[:, xi - 3:xi + 4]
        rruns = runs(win.any(axis=1), merge_gap=6)
        pick = None
        for (s, e) in rruns:
            if s - 4 <= mrow <= e + 4:
                pick = (s, e)
        if pick is None:
            records.append({"series": name, "t": float(tv), "mean": mu, "sd": sd,
                            "note": f"no run containing marker row {mrow:.0f}; {rruns}"})
            continue
        s, e = pick
        half = (val_of(s) - val_of(e)) / 2
        records.append({"series": name, "t": float(tv), "mean": mu, "sd": sd,
                        "pred_marker_row": round(mrow, 1), "run": [int(s), int(e)],
                        "half_extent": round(half, 5),
                        "sd_minus_half_extent": round(sd - half, 5)})
o = pd.DataFrame(records)
print(o.to_string(index=False))
ok = o[o.sd.notna()]
print(f"\nmax |sd - half_extent| = {ok.sd_minus_half_extent.abs().max():.5f}"
      f"  (one pixel = {(yhi-ylo)/(B5-T5):.5f})")
print("mean (sd - half_extent) = ", round(ok.sd_minus_half_extent.mean(), 5),
      " -> the small constant offset is the marker disc radius (~0.6 px)")
print("Spearman rank agreement (hand-computed):")
r_half = ok.half_extent.rank().to_numpy()
r_sd = ok.sd.rank().to_numpy()
print("   rank differences:", (r_half - r_sd).astype(int).tolist())
o.to_csv(f"{BASE}/output/verification/fig5b_errorbar_pixels.tsv", sep="\t", index=False)