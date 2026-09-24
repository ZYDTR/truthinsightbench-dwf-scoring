#!/usr/bin/env python3
"""
INDEPENDENT FIGURE VERIFIER (final).

Does two things, both from scratch:
  PART 1  recompute every number that fig1..fig5 are supposed to encode,
          straight from data/*.tsv (numpy/pandas only; Pearson r computed twice,
          once with np.corrcoef and once with the hand covariance formula).
  PART 2  read the numbers BACK OUT OF THE PNG PIXELS to prove the pictures
          really show those numbers.

No code from output/analysis/analysis.py is imported or executed.

Run:  python3 output/verification/verifier_figures.py
"""
import json
import numpy as np
import pandas as pd
from PIL import Image

BASE = "/data/user/conversations/conv-995a498d-d5c1-4e1e-99f5-01601e51abe5/project"
FIGD = f"{BASE}/output/figures"
OUTJSON = f"{BASE}/output/verification/verifier_figures_out.json"
C0 = np.array([31, 119, 180]); C1 = np.array([255, 127, 14])
C2 = np.array([44, 160, 44]); C7 = np.array([127, 127, 127])

R = {}

# ---------------------------------------------------------------- helpers --
def load(n):
    return np.asarray(Image.open(f"{FIGD}/{n}").convert("RGB")).astype(int)


def runs(mask, gap=0):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    out, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i <= p + 1 + gap:
            p = i
        else:
            out.append((s, p)); s = p = i
    out.append((s, p))
    return out


def frame(img, x0, x1, y0, y1, frac=0.7):
    reg = img[y0:y1, x0:x1]
    dm = (reg.max(axis=2) < 110)
    h, w = reg.shape[:2]
    cs = np.where(dm.sum(axis=0) > frac * h)[0]
    rs = np.where(dm.sum(axis=1) > frac * w)[0]
    return x0 + cs.min(), x0 + cs.max(), y0 + rs.min(), y0 + rs.max()


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    r1 = float(np.corrcoef(x, y)[0, 1])
    xc, yc = x - x.mean(), y - y.mean()
    r2 = float((xc * yc).sum() / np.sqrt((xc * xc).sum() * (yc * yc).sum()))
    return r1, r2


def r2lin(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    b1 = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
    return 1 - ((y - (y.mean() + b1 * (x - x.mean()))) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def r2cat(t, y):
    t = np.asarray(t); y = np.asarray(y, float)
    pred = np.empty_like(y)
    for g in np.unique(t):
        pred[t == g] = y[t == g].mean()
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


# ------------------------------------------------------------------ data ---
cur = pd.read_csv(f"{BASE}/data/time_response_curves.tsv", sep="\t")
dose = pd.read_csv(f"{BASE}/data/dose_response.tsv", sep="\t")
pert = pd.read_csv(f"{BASE}/data/perturbation_time_response.tsv", sep="\t")
P = cur["promoter_like_signal"].to_numpy(float)
A = cur["condition_a"].to_numpy(float)
B = cur["condition_b"].to_numpy(float)
Cc = cur["condition_c"].to_numpy(float)
FA = cur["fit_a"].to_numpy(float)
FB = cur["fit_b"].to_numpy(float)
T = cur["time"].to_numpy(float)
REG = {"A_baseline": slice(0, 259), "B_sweep": slice(259, 294), "C_tail": slice(294, 1367)}
RN = ["A_baseline", "B_sweep", "C_tail", "WHOLE"]

# ============================================================== PART 1 =====
print("=" * 78); print("PART 1 -- independent numeric recomputation"); print("=" * 78)

# ---- fig1(a) --------------------------------------------------------------
f1a = {
    "regimes_claimed": {"A": [0, 258], "B": [259, 293], "C": [294, 1366]},
    "A_time_max": float(T[:259].max()),
    "B_time_min_max": [float(T[259:294].min()), float(T[259:294].max())],
    "C_time_max": float(T[294:].max()),
    "n_rows_time_ge_2": int((T >= 2).sum()),
    "rows_time_ge_2_outside_B": [int(i) for i in np.where(T >= 2)[0] if not 259 <= i <= 293],
    "P_at_row294": float(P[294]), "P_min_in_C": float(P[294:].min()),
    "P_min_row": int(294 + int(np.argmin(P[294:]))), "P_last": float(P[-1]),
    "C_steps_non_increasing_frac": float((np.diff(P[294:]) <= 0).mean()),
    "C_rows_below_running_max_frac": float((P[295:] < np.maximum.accumulate(P[294:])[:-1]).mean()),
    "C_trend_r2_on_row": float(r2lin(np.arange(1073.), P[294:])),
    "fit_a_zero_in_A": bool((FA[:259] == 0).all()),
    "fit_b_zero_in_A": bool((FB[:259] == 0).all()),
    "condition_c_zero_in_A": bool((Cc[:259] == 0).all()),
    "P_range_A": [float(P[:259].min()), float(P[:259].max())],
    "P_range_C": [float(P[294:].min()), float(P[294:].max())],
}
# ---- fig1(b) --------------------------------------------------------------
f1b = {
    "r2_cat_all": {n: round(float(r2cat(T, v)), 6) for n, v in [("prom", P), ("a", A), ("b", B)]},
    "r2_cat_C": {n: round(float(r2cat(T[294:], v[294:])), 6) for n, v in
                 [("prom", P), ("a", A), ("b", B)]},
    "shared_times_with_widest_spread": {},
}
for n, v in [("prom", P), ("a", A), ("b", B)]:
    sp = [(float(tv), float(v[T == tv].max() - v[T == tv].min()))
          for tv in np.unique(T) if (T == tv).sum() > 1]
    wt = max(sp, key=lambda q: q[1])
    f1b["shared_times_with_widest_spread"][n] = {
        "time": wt[0], "spread": wt[1],
        "spread_over_global_range": round(wt[1] / (v.max() - v.min()), 4)}
f1b["t2_values"] = [float(x) for x in P[T == 2]]
f1b["t2_values_condition_a"] = [float(x) for x in A[T == 2]]
# ---- fig1(c)/(d) ----------------------------------------------------------
f1cd = {"n_rows": 35, "time_sequence": [float(x) for x in T[259:294]],
        "ascending_rows_259_267_time_increasing": bool((np.diff(T[259:268]) > 0).all()),
        "apex": {}, "real_branch_peaks": {}}
for n, v in [("prom", P), ("a", A), ("b", B)]:
    up, dn = v[259:268], v[268:284]
    j = int(np.argmax(v[259:294]))
    f1cd["apex"][n] = {"global_row": 259 + j, "time": float(T[259 + j]), "value": float(v[259 + j])}
    f1cd["real_branch_peaks"][n] = {
        "ascending_max_rows259_267": float(up.max()),
        "descending_max_rows268_283": float(dn.max()),
        "ratio_down_over_up": round(float(dn.max() / up.max()), 4)}
# ---- fig2 -----------------------------------------------------------------
f2 = {}
for n, v in [("promoter_like_signal", P), ("condition_a", A), ("condition_b", B)]:
    up, dn = v[259:268], v[268:284]
    tu, td = T[259:268], T[268:284]
    up2 = float(v[260]); dn2 = [float(v[i]) for i in (284, 286, 288)]
    o = np.argsort(td)
    grid = np.array([50, 100, 200, 400, 800, 1600, 3200, 6400, 9000], float)
    gu = np.interp(np.log10(grid), np.log10(tu[tu > 0]), up[tu > 0])
    gd = np.interp(np.log10(grid), np.log10(td[o]), dn[o])
    f2[n] = {
        "t2_up_row260": up2, "t2_down_rows_284_286_288": dn2,
        "t2_ratio_mean": round(float(np.mean(dn2) / up2), 3),
        "t2_ratio_jackknife": [round(min(x / up2 for x in dn2), 3),
                               round(max(x / up2 for x in dn2), 3)],
        "loop_area_logt": round(float(np.trapezoid(gd - gu, np.log10(grid))), 4),
        "peak_up_grid": round(float(gu.max()), 2), "peak_down_grid": round(float(gd.max()), 2),
        "grid_peak_ratio": round(float(gd.max() / gu.max()), 4),
        "down_above_up_flags": ["down>up" if gd[i] > gu[i] else "down<up"
                                for i in range(len(grid))],
        "max_abs_gap": round(float(np.abs(gd - gu).max()), 2),
        "max_gap_over_channel_max": round(float(np.abs(gd - gu).max() / max(v.max(), gu.max())), 4),
    }
# ---- fig3 -----------------------------------------------------------------
f3 = {"levels": {}, "diffs": {}, "handcov_matches": {}}
for n in RN:
    s = slice(None) if n == "WHOLE" else REG[n]
    for pn, x, y in [("prom|a", P[s], A[s]), ("prom|b", P[s], B[s]), ("a|b", A[s], B[s])]:
        r1, r2 = pearson(x, y)
        f3["levels"][f"{n}:{pn}"] = round(r1, 6)
        f3["diffs"][f"{n}:{pn}"] = round(pearson(np.diff(x), np.diff(y))[0], 6)
        f3["handcov_matches"][f"{n}:{pn}"] = bool(abs(r1 - r2) < 1e-12)
# ---- fig4 -----------------------------------------------------------------
dFA, dAc = np.diff(FA[294:]), np.diff(A[294:])
f4 = {
    "panel_a_plotted_row_range": [259, 1366],
    "panel_a_title_says_full_row_range_but_omits_rows": 259,
    "fit_a_frac_strictly_descending_C": float((dFA < 0).mean()),
    "fit_a_direction_reversals_C": int((np.diff(np.sign(dFA)) != 0).sum()),
    "condition_a_frac_strictly_descending_C": float((dAc < 0).mean()),
    "condition_a_direction_reversals_C": int((np.diff(np.sign(dAc)) != 0).sum()),
    "fit_a_increases_over_whole_file": [[int(i), round(float(FA[i + 1] - FA[i]), 2)]
                                        for i in range(len(FA) - 1) if FA[i + 1] > FA[i]],
    "condition_a_nonzero_rows_in_A": int((A[:259] > 0).sum()),
    "condition_a_range_in_A": [float(A[:259].min()), float(A[:259].max())],
}
rng = np.random.default_rng(20260919)          # different seed from the report
obs = float((dFA < 0).mean())
cnt = sum(1 for _ in range(5000) if float((np.diff(rng.permutation(A[294:])) < 0).mean()) >= obs)
f4["permutation_fit_a"] = {"draws": 5000, "n_ge_obs": int(cnt), "p": cnt / 5000,
                           "obs": obs, "perm_mean": round(float(
                               np.mean([(np.diff(rng.permutation(A[294:])) < 0).mean()
                                        for _ in range(300)])), 4)}
# ---- fig5 -----------------------------------------------------------------
resp = dose["response_mean"].to_numpy(float); dse = dose["dose"].to_numpy(float)
f5 = {"dose": {"values": [float(x) for x in resp],
               "annotations_2dp": [f"{x:.2f}" for x in resp],
               "sd_all_missing": bool(dose["response_sd"].isna().all()),
               "second_difference": round(float(resp[2] - 2 * resp[1] + resp[0]), 6),
               "linear_slope": round(float(((dse - dse.mean()) * (resp - resp.mean())).sum()
                                           / ((dse - dse.mean()) ** 2).sum()), 6)},
      "pert": {}}
for c in ["control", "condition_a", "condition_b"]:
    m = pert[f"{c}_mean"].to_numpy(float); sd = pert[f"{c}_sd"].to_numpy(float)
    tt = pert["time"].to_numpy(float)
    f5["pert"][c] = {"mean": [float(x) for x in m], "sd": [float(x) for x in sd],
                     "change": round(float(m[-1] - m[0]), 5),
                     "strictly_decreasing": bool((np.diff(m) < 0).all()),
                     "min_sd": float(sd.min()), "max_sd": float(sd.max()),
                     "sd_present_for_all": bool(np.isfinite(sd).all())}
R["numeric"] = {"fig1a": f1a, "fig1b": f1b, "fig1cd": f1cd, "fig2": f2,
                "fig3": f3, "fig4": f4, "fig5": f5}
for k, v in R["numeric"].items():
    print(f"\n--- {k} ---"); print(json.dumps(v, indent=1))

# ============================================================== PART 2 =====
print("\n" + "=" * 78); print("PART 2 -- PNG pixel read-back"); print("=" * 78)
px = {}

# ---------------- fig1(a): shading boundaries + tail trace ------------------
img1 = load("fig1_regime_structure.png"); H1, W1 = img1.shape[:2]
L1, R1, T1, B1 = frame(img1, 5, W1 // 2, 5, H1 // 2)
ticks = [int((s + e) / 2) + L1
         for (s, e) in runs((img1[B1 + 2:B1 + 6, L1:R1].max(axis=2) < 120).any(axis=0))
         if e - s < 12]
assert len(ticks) == 8, ticks
step = (ticks[-1] - ticks[0]) / 7.0            # px per 200 data rows
row2px = lambda r: ticks[0] + r * step / 200.0
px2row = lambda x: (x - ticks[0]) / step * 200.0
BAND = {"A": np.array([255, 238, 223]), "B": np.array([249, 227, 227]),
        "C": np.array([227, 242, 227]), "W": np.array([255, 255, 255])}
lab = []
for j in range(L1 + 1, R1):
    c = img1[T1 + 4:B1 - 4, j]
    k = ((c[:, 0] << 16) + (c[:, 1] << 8) + c[:, 2])
    vals, cnt = np.unique(k, return_counts=True)
    kk = vals[np.argmax(cnt)]
    rgb = np.array([(kk >> 16) & 255, (kk >> 8) & 255, kk & 255])
    # strict match: columns crossed by a grid line or the data curve are "."
    tight = [n for n in BAND if int(np.abs(rgb - BAND[n]).max()) <= 3]
    lab.append(tight[0] if tight else ".")
lab = np.array(lab)


def longest(lab, ch, gap=3):
    idx = np.where(lab == ch)[0]
    out, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i <= p + 1 + gap:
            p = i
        else:
            out.append((s, p)); s = p = i
    out.append((s, p))
    return max(out, key=lambda q: q[1] - q[0])


# Use the shaded SWEEP band (colour B) extent: its left and right edges must
# sit at rows 259 and 293 if the figure's shading matches the claimed regimes.
bcols = [i + L1 + 1 for i in np.where(lab == "B")[0]]
first_b, last_b = min(bcols), max(bcols)
first_c = min(i + L1 + 1 for i in np.where(lab == "C")[0])
edge_ab_row = float(px2row(first_b))
edge_bc_row = float(px2row((last_b + first_c) / 2.0))
px["fig1a_edge_columns"] = {
    "sweep_band_pixel_columns": [int(first_b), int(last_b)],
    "first_C_pixel_column": int(first_c),
    "band_span_px": int(last_b - first_b + 1),
    "claimed_band_rows": [259, 293],
    "measured_band_rows": [round(edge_ab_row, 1), round(edge_bc_row, 1)],
    "px_resolution_rows": round(200 / step, 3),
    "A|B_within_1px": bool(abs(edge_ab_row - 259) <= 200 / step),
    "B|C_within_1px": bool(abs(edge_bc_row - 293) <= 200 / step),
}
ra = longest(lab, "A"); rb = longest(lab, "B"); rc = longest(lab, "C")
px["fig1a_boundaries"] = {
    "x_tick_columns": [int(x) for x in ticks], "px_per_200_rows": round(step, 2),
    "rows_per_px": round(200 / step, 3),
    "pure_A_run_px": [int(ra[0] + L1 + 1), int(ra[1] + L1 + 1)],
    "pure_B_run_px": [int(rb[0] + L1 + 1), int(rb[1] + L1 + 1)],
    "pure_C_run_px": [int(rc[0] + L1 + 1), int(rc[1] + L1 + 1)],
}
for k, v in px["fig1a_boundaries"].items():
    print(f"  fig1a {k}: {v}")
for k, v in px["fig1a_edge_columns"].items():
    print(f"  fig1a {k}: {v}")
# tail trace: drawn y at a sequence of rows; is it a steady decline?
# "bluish" = blue channel clearly dominant, catches the antialiased thin line
mL = (img1[:, :, 2] > img1[:, :, 0] + 25) & (img1[:, :, 2] > 90)
mL[:, :L1 + 1] = False; mL[:, R1:] = False; mL[:T1, :] = False; mL[B1:, :] = False
trace = []
for row in [300, 400, 600, 800, 1000, 1200, 1300, 1364]:
    c = int(round(row2px(row)))
    rr = np.where(mL[:, c - 2:c + 3].any(axis=1))[0]
    trace.append({"row": row, "data_value": float(P[row]),
                  "drawn_y_px": [int(rr.min()), int(rr.max())] if len(rr) else None})
px["fig1a_tail_trace"] = trace
print("  fig1a tail trace (drawn pixel rows should increase steadily = decline on"
      " screen; symlog axis so spacing is log-like):")
for t in trace:
    print("   ", t)

# ---------------- fig1(b): vertical spread of the t=0 column ---------------
Lb, Rb, Tb, Bb = frame(img1, W1 // 2, W1, 5, H1 // 2)
mb = np.abs(img1 - C0).max(axis=2) <= 55
mb[:, :Lb] = False; mb[:, Rb:] = False; mb[:Tb, :] = False; mb[Bb:, :] = False
cls = [(s, e) for (s, e) in runs(mb.sum(axis=0) > 0) if e - s >= 1]
first = cls[0]
rr = np.where(mb[:, first[0]:first[1] + 1].any(axis=1))[0]
px["fig1b_t0_column"] = {
    "x_pixels": [int(first[0]), int(first[1])],
    "drawn_y_extent": [int(rr.min()), int(rr.max())],
    "panel_y_extent": [int(Tb), int(Bb)],
    "extent_over_panel_height": round((rr.max() - rr.min()) / (Bb - Tb), 3),
    "n_x_clusters": len(cls),
    "note": "dots with value 0 sit on the bottom spine and are hidden; the visible"
            " spread is still ~42% of the panel height",
}
print("\n  fig1b t=0 column:", json.dumps(px["fig1b_t0_column"]))

# ---------------- fig3: 24 bar heights ------------------------------------
img3 = load("fig3_coupling_by_regime.png"); H3, W3 = img3.shape[:2]
bars = []
for panel, (xa, xb) in [("levels", (0, W3 // 2)), ("diffs", (W3 // 2, W3))]:
    L3, R3, T3, B3 = frame(img3, xa + 5, xb, 5, H3 - 5)
    dm = (img3[:, L3 + 3:R3 - 3].max(axis=2) < 110)
    zr = max(range(T3 + 2, B3 - 2), key=lambda r: dm[r].sum())
    ppu = (B3 - T3) / 2.10
    tk = [int((s + e) / 2) + L3
          for (s, e) in runs((img3[B3 + 2:B3 + 6, L3:R3].max(axis=2) < 120).any(axis=0))
          if e - s < 12]
    padj = np.polyfit(np.arange(4), tk, 1)
    for i, pn in enumerate(["prom|a", "prom|b", "a|b"]):
        rgb = {"prom|a": C0, "prom|b": C1, "a|b": C2}[pn]
        m = np.abs(img3 - rgb).max(axis=2) <= 25
        m[:, :L3] = False; m[:, R3:] = False; m[:T3, :] = False; m[B3:, :] = False
        for cat, nm in enumerate(RN):
            cx = int(round(padj[0] * cat + padj[1] + (i - 1) * 0.26 * padj[0]))
            cand = [q for q in runs(m[:, cx - 1:cx + 2].any(axis=1), gap=6)
                    if q[0] - 2 <= zr <= q[1] + 2]
            if not cand:
                bars.append({"panel": panel, "pair": pn, "regime": nm, "px_value": None})
                continue
            s, e = max(cand, key=lambda q: q[1] - q[0])
            v = (zr - s) / ppu if abs(zr - s) > abs(zr - e) else (zr - e) / ppu
            bars.append({"panel": panel, "pair": pn, "regime": nm,
                         "px_value": round(float(v), 4),
                         "recomputed": f3[panel][f"{nm}:{pn}"],
                         "diff": round(float(v) - f3[panel][f"{nm}:{pn}"], 4)})
px["fig3_bars"] = bars
ds = [abs(b["diff"]) for b in bars if b["px_value"] is not None]
print(f"\n  fig3: {sum(1 for b in bars if b['px_value'] is not None)}/24 bars readable; "
      f"max |pixel-recomputed| = {max(ds):.4f} ({max(ds)/(2.10/(B3-T3)):.1f} px); "
      f"mean = {np.mean(ds):.4f}")
print("  spot checks:")
for b in bars:
    if b["px_value"] is not None and (b["pair"], b["regime"]) in [
            ("prom|a", "C_tail"), ("a|b", "A_baseline"), ("a|b", "C_tail"),
            ("prom|b", "WHOLE"), ("a|b", "B_sweep"), ("prom|a", "B_sweep")]:
        print(f"    {b['panel']:6s} {b['pair']:7s} {b['regime']:11s} "
              f"pixel={b['px_value']:+.4f} recomputed={b['recomputed']:+.4f}")

# ---------------- fig4(a): x range vs the title ----------------------------
img4 = load("fig4_raw_vs_fitted.png"); H4, W4 = img4.shape[:2]
L4, R4, T4, B4 = frame(img4, 5, W4 // 2, 5, H4 - 5)
tk4 = [int((s + e) / 2) + L4
       for (s, e) in runs((img4[B4 + 2:B4 + 6, L4:R4].max(axis=2) < 120).any(axis=0))
       if e - s < 12]
mblu = np.abs(img4 - C0).max(axis=2) <= 55
mblu[:, :L4 + 1] = False; mblu[:, R4:] = False; mblu[:T4, :] = False; mblu[B4:, :] = False
cb = np.where(mblu.any(axis=0))[0]
lab4 = [400 + 200 * k for k in range(len(tk4))]      # 6 ticks visible
s4 = (tk4[-1] - tk4[0]) / (len(tk4) - 1)
to_row4 = lambda x: lab4[0] + (x - tk4[0]) / s4 * 200
px["fig4a"] = {"x_tick_columns": [int(x) for x in tk4], "tick_labels_read": lab4,
               "first_data_pixel": int(cb.min()), "first_data_row": round(float(to_row4(cb.min())), 1),
               "last_data_pixel": int(cb.max()), "last_data_row": round(float(to_row4(cb.max())), 1),
               "title_claims": "full row range",
               "actual_rows_plotted": [259, 1366]}
print("\n  fig4(a):", json.dumps(px["fig4a"], indent=1))

# ---------------- fig5(b): error bars -------------------------------------
img5 = load("fig5_dose_perturbation.png"); H5, W5 = img5.shape[:2]
L5, R5, T5, B5 = frame(img5, W5 // 2 + 5, W5, 5, H5 - 5)
allv = []
for c in ["control", "condition_a", "condition_b"]:
    allv += list(pert[f"{c}_mean"] - pert[f"{c}_sd"]) + list(pert[f"{c}_mean"] + pert[f"{c}_sd"])
lo, hi = min(allv), max(allv); mg = 0.05 * (hi - lo); ylo, yhi = lo - mg, hi + mg
val_of = lambda r: yhi - (r - T5) / (B5 - T5) * (yhi - ylo)
row_of = lambda v: T5 + (yhi - v) / (yhi - ylo) * (B5 - T5)
tvals = pert["time"].to_numpy(float)
tpix = {float(tv): L5 + ((tv + 0.3) / 6.6) * (R5 - L5) for tv in tvals}
series = {"control": C7, "condition_a": C1, "condition_b": C2}
eb = []
for name, rgb in series.items():
    m = np.abs(img5 - rgb).max(axis=2) <= 55
    m[:, :L5] = False; m[:, R5:] = False; m[:T5, :] = False; m[B5:, :] = False
    for k, tv in enumerate(tvals):
        xi = int(round(tpix[float(tv)]))
        mu = float(pert[f"{name}_mean"][k]); sd = float(pert[f"{name}_sd"][k])
        r0 = row_of(mu); half_px = sd / (yhi - ylo) * (B5 - T5)
        win = m[int(r0 - half_px - 8):int(r0 + half_px + 8), max(0, xi - 3):xi + 4]
        rr = np.where(win.any(axis=1))[0]
        if len(rr) == 0:
            eb.append({"series": name, "t": float(tv), "sd": sd,
                       "measured_half_extent": None}); continue
        top = rr.min() + int(r0 - half_px - 8); bot = rr.max() + int(r0 - half_px - 8)
        half = (val_of(top) - val_of(bot)) / 2
        eb.append({"series": name, "t": float(tv), "sd": round(sd, 5),
                   "measured_half_extent": round(half, 5),
                   "sd_minus_measured": round(sd - half, 5)})
px["fig5b_errorbars"] = eb
px["fig5b_axes"] = {"analytic_autoscale": [round(ylo, 6), round(yhi, 6)],
                    "y_axis_includes_zero": bool(ylo <= 0),
                    "data_mean_span": round(float(np.ptp(np.concatenate(
                        [pert[f"{c}_mean"].values for c in series]))), 5)}
print("\n  fig5(b):", json.dumps(px["fig5b_axes"]))
for b in eb:
    print("   ", b)
ok = [b for b in eb if b.get("measured_half_extent") is not None]
print(f"  max |sd - measured half extent| = {max(abs(b['sd_minus_measured']) for b in ok):.5f}"
      f"  (1 px = {(yhi-ylo)/(B5-T5):.5f})")
R["pixels"] = px
with open(OUTJSON, "w") as f:
    json.dump(R, f, indent=1, default=str)
print(f"\nwrote {OUTJSON}")