#!/usr/bin/env python3
"""Fifth-pass pixel verification.

fig3: sample each bar on its exact centre column only (legend patches live at
      the far-left edge of the left panel and do not reach the bar centres),
      search the whole panel vertically.
fig5b: mask the legend bounding box (detected from its frame lines) before
      measuring the error-bar extents.
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
print("FIG 3 -- bar values sampled on each bar's centre column")
print("=" * 78)
img3 = load("fig3_coupling_by_regime.png")
H3, W3 = img3.shape[:2]
rows_out = []
for panel, (xa, xb) in [("levels", (0, W3 // 2)), ("diffs", (W3 // 2, W3))]:
    L3, R3, T3, B3 = frame(img3, xa + 5, xb, 5, H3 - 5)
    dm = (img3[:, L3 + 3:R3 - 3].max(axis=2) < 110)
    zr = max(range(T3 + 2, B3 - 2), key=lambda rr: dm[rr].sum())
    ppu = (B3 - T3) / 2.10
    tickrow = min(B3 + 3, H3 - 1)
    ticks = [int((s + e) / 2) + L3
             for (s, e) in runs((img3[tickrow:tickrow + 4, L3:R3].max(axis=2) < 120)
                                .any(axis=0)) if e - s < 12]
    padj = np.polyfit(np.arange(4), ticks, 1)
    print(f"\n{panel}: frame x[{L3},{R3}] y[{T3},{B3}] zero_row={zr} "
          f"ppu={ppu:.2f}  ticks={ticks}")
    for i, pn in enumerate(["prom|a", "prom|b", "a|b"]):
        rgb = {"prom|a": C0, "prom|b": C1, "a|b": C2}[pn]
        m = np.abs(img3 - rgb).max(axis=2) <= 25
        m[:, :L3] = False; m[:, R3:] = False; m[:T3, :] = False; m[B3:, :] = False
        off = (i - 1) * 0.26
        for cat, nm in enumerate(names4):
            cx = int(round(padj[0] * cat + padj[1] + off * padj[0]))
            colpix = m[:, cx - 1:cx + 2].any(axis=1)
            rr = runs(colpix)
            # keep the run that contains the zero line (bars are anchored at 0)
            cand = [q for q in rr if q[0] - 2 <= zr <= q[1] + 2]
            if not cand:
                rows_out.append({"panel": panel, "pair": pn, "regime": nm,
                                 "col": cx, "px_value": None,
                                 "note": f"no bar at zero line, runs={rr}"})
                continue
            s, e = max(cand, key=lambda q: q[1] - q[0])
            v = (zr - s) / ppu if abs(zr - s) > abs(zr - e) else (zr - e) / ppu
            claim = recomp[panel][f"{nm}:{pn}"]
            rows_out.append({"panel": panel, "pair": pn, "regime": nm, "col": cx,
                             "run": [int(s), int(e)],
                             "px_value": round(float(v), 4),
                             "recomputed": round(claim, 4),
                             "diff": round(float(v) - claim, 4)})
d = pd.DataFrame(rows_out)
print(d.to_string(index=False))
good = d[d.px_value.notna()]
print(f"\nmax |pixel - recomputed| among {len(good)} measured bars: "
      f"{good['diff'].abs().max():.4f}  (pixel resolution = "
      f"{2.10/(B3-T3):.5f} per px)")
print("mean |diff| =", round(good["diff"].abs().mean(), 4))
print("bars not measurable:", d[d.px_value.isna()][["panel", "pair", "regime"]].to_dict("records"))
d.to_csv(f"{BASE}/output/verification/fig3_bar_pixels.tsv", sep="\t", index=False)

# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("FIG 5b -- error bars, legend masked out")
print("=" * 78)
img5 = load("fig5_dose_perturbation.png")
H5, W5 = img5.shape[:2]
L5, R5, T5, B5 = frame(img5, W5 // 2 + 5, W5, 5, H5 - 5)
pert = pd.read_csv(f"{BASE}/data/perturbation_time_response.tsv", sep="\t")
t = pert["time"].to_numpy(float)
tpix = {float(tv): L5 + ((tv + 0.3) / 6.6) * (R5 - L5) for tv in t}
series = {"control": C7, "condition_a": C1, "condition_b": C2}
print(f"panel frame x[{L5},{R5}] y[{T5},{B5}]; marker columns:",
      {k: round(v) for k, v in tpix.items()})

# --- detect the legend box: horizontal dark line segments inside the panel ---
dark = (img5.max(axis=2) < 130)
segs = []
for rr in range(T5 + 2, B5 - 2):
    for (s, e) in runs(dark[rr, L5 + 2:R5 - 2]):
        if 60 <= e - s <= 400:
            segs.append((rr, s + L5 + 2, e + L5 + 2))
leg_rows = sorted({s0 for s0, _, _ in segs})
print(f"candidate legend frame rows: {leg_rows[:6]} ... {leg_rows[-6:] if len(leg_rows)>6 else ''}")
if leg_rows:
    lt, lb = min(leg_rows), max(leg_rows)
    vsegs = [s for s in segs if s[0] in (lt, lb)]
    ll = min(x0 for _, x0, _ in vsegs); lr = max(x1 for _, _, x1 in vsegs)
    print(f"legend bbox approx rows [{lt},{lb}] cols [{ll},{lr}]")
else:
    lt, lb, ll, lr = 0, 0, 0, 0

legend_mask = np.zeros_like(dark, bool)
legend_mask[lt - 2:lb + 3, ll - 2:lr + 3] = True


def marker_centre(name, tv, mask):
    rgb = series[name]
    m = np.abs(img5 - rgb).max(axis=2) <= 25
    m[legend_mask] = False
    xi = int(round(tpix[float(tv)]))
    win = m[:, xi - 4:xi + 5]
    rr = runs(win.any(axis=1))
    if not rr:
        return None
    s, e = max(rr, key=lambda q: int(win[q[0]:q[1] + 1].sum()))
    return (s + e) / 2.0


# robust y transform: fit from the marker centres of the columns that are far
# from the legend (t=4 and t=6), using all three series
anch = []
for name in series:
    for tv in [4.0, 6.0]:
        c = marker_centre(name, tv, legend_mask)
        k = int(np.where(t == tv)[0][0])
        anch.append((c, float(pert[f"{name}_mean"][k])))
rows_a = np.array([a for a, _ in anch]); vals_a = np.array([b for _, b in anch])
coef = np.polyfit(rows_a, vals_a, 1)
print(f"y map from 6 anchors (t=4,6): value = {coef[0]:.7f}*row + {coef[1]:.6f}; "
      f"panel span = {abs(coef[0])*(B5-T5):.4f}")
print("anchor residuals (px):",
      np.round((vals_a - np.polyval(coef, rows_a)) / coef[0], 2).tolist())

res = []
for name in series:
    rgb = series[name]
    for k, tv in enumerate(t):
        xi = int(round(tpix[float(tv)]))
        m = np.abs(img5 - rgb).max(axis=2) <= 25
        m[legend_mask] = False
        win = m[:, xi - 3:xi + 4]
        rr = runs(win.any(axis=1))
        mc = marker_centre(name, tv, legend_mask)
        if mc is None:
            res.append({"series": name, "t": float(tv), "note": "no marker"})
            continue
        pick = None
        for (s, e) in rr:
            if s - 4 <= mc <= e + 4:
                pick = (s, e)
        s, e = pick if pick else max(rr, key=lambda q: q[1] - q[0])
        v_t = float(np.polyval(coef, s)); v_b = float(np.polyval(coef, e))
        mu = float(pert[f"{name}_mean"][k]); sd = float(pert[f"{name}_sd"][k])
        res.append({"series": name, "t": float(tv), "mean": mu, "sd": sd,
                    "pix": [int(s), int(e)],
                    "val_hi": round(v_t, 5), "val_lo": round(v_b, 5),
                    "half_extent": round((v_t - v_b) / 2, 5),
                    "sd_minus_half": round(sd - (v_t - v_b) / 2, 5)})
o = pd.DataFrame(res)
print("\n" + o.to_string(index=False))
ok = o[o.sd.notna()]
Amat = np.column_stack([ok.sd.to_numpy(float), np.ones(len(ok))])
sol, *_ = np.linalg.lstsq(Amat, ok.half_extent.to_numpy(float), rcond=None)
pred = Amat @ sol
r2 = 1 - ((ok.half_extent - pred) ** 2).sum() / ((ok.half_extent - ok.half_extent.mean()) ** 2).sum()
print(f"\nhalf_extent ~ {sol[0]:.3f} * sd + {sol[1]:.5f}   R^2={r2:.4f}")
print("(slope 1.0 => the bars are exactly the supplied SDs; intercept = marker"
      " disc radius expressed in value units, identical for every point)")
print("max |sd - half_extent| =", round(ok.sd_minus_half.abs().max(), 5),
      " i.e. marker-radius offset ~", round(ok.sd_minus_half.median(), 5))
o.to_csv(f"{BASE}/output/verification/fig5b_errorbar_pixels.tsv", sep="\t", index=False)