#!/usr/bin/env python3
"""covx_s04 (dimension: coverage boundary) - decisive tests.

1. Coarse histogram Sheet_039..044: are the 18 coarse curves the SAME SHAPE
   (a common template), or do they track their fine counterpart?  Also: are the
   coarse bin values identical to the fine bin values, just coarser?
2. Image sheets: structural tests for the 4 bounds-sharing pairs and the 4
   standalone positive/signed images.
3. Definitive coverage table written to csv + json.

Outputs: covx_* csv/json + figures.
"""
from __future__ import annotations

import collections
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "results"
FIG = OUT / "figures"
DATA = ROOT / "data" / "normalized_views"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def blocks():
    blk = collections.defaultdict(list)
    order = []
    for r in csv.DictReader(open(OUT / "covx_absent_blocks.csv")):
        if r["kind"] not in ("point", "label"):
            continue
        if r["x"] in ("", "None") or r["y"] in ("", "None"):
            continue
        k = (r["sheet"], r["label"])
        if k not in blk:
            order.append(k)
        blk[k].append((float(r["x"]), float(r["y"])))
    return {k: np.asarray(v, float) for k, v in blk.items()}, order


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    blk, order = blocks()
    rep = {}
    figlist = []

    # ------------------------------------------------- 1 coarse template test
    print("=" * 78)
    print("1  COARSE HISTOGRAMS: common template or fine-tracked?")
    coarse_sheet = {39: ("Sheet_025", "Sheet_024"), 40: ("Sheet_027", "Sheet_026"),
                    41: ("Sheet_029", "Sheet_028"), 42: ("Sheet_031", "Sheet_030"),
                    43: ("Sheet_033", "Sheet_032"), 44: ("Sheet_035", "Sheet_034")}
    curves = {}
    pairs = []
    for sh, (fast_sh, slow_sh) in coarse_sheet.items():
        cs = f"Sheet_{sh:03d}"
        for cl in [k for k in order if k[0] == cs]:
            curves[("coarse", cs, cl[1])] = blk[cl]
            cand = [k for k in order if k[0] in (fast_sh, slow_sh)
                    and k[1].replace(" ", "") == cl[1].replace(" ", "")]
            if cand:
                curves[("fine", cand[0][0], cl[1])] = blk[cand[0]]
                pairs.append((("coarse", cs, cl[1]), ("fine", cand[0][0], cl[1])))

    # all coarse curves share the same 20-point x grid -> shape matrix
    ckeys = [k for k in curves if k[0] == "coarse"]
    # the coarse sheets are mostly on a 20-point 1.75 MHz/s grid but Sheet_040's
    # '0<s<20' block has 30 points; reduce every curve onto one common 0.875 MHz/s
    # grid by nearest-bin accumulation so the shapes are comparable.
    xgrid = np.arange(0.0, 35.0, 0.875)
    print(f"  coarse curve bin counts: {sorted(set(len(curves[k]) for k in ckeys))}"
          f"   x-ranges: {sorted(set((float(curves[k][0,0]), float(curves[k][-1,0])) for k in ckeys))}")
    def reduce_to_grid(a):
        red = np.zeros(len(xgrid))
        for (xx, yy) in a:
            red[int(np.argmin(np.abs(xgrid - xx)))] += yy
        return red
    M = np.array([reduce_to_grid(curves[k]) for k in ckeys])
    M = M / M.sum(axis=1, keepdims=True)
    print(f"  {len(ckeys)} coarse curves on a common {len(xgrid)}-point grid")
    # pairwise similarity
    sim = []
    for i in range(len(ckeys)):
        for j in range(i + 1, len(ckeys)):
            sim.append(dict(a=str(ckeys[i]), b=str(ckeys[j]),
                            pearson=corr(M[i], M[j]),
                            l1=float(np.abs(M[i] - M[j]).sum()),
                            same_substride=ckeys[i][2] == ckeys[j][2]))
    norml = lambda t: t.replace(" ", "")
    sim_same = [d["pearson"] for d in sim if norml(eval(d["a"])[2]) == norml(eval(d["b"])[2])]
    sim_diff = [d["pearson"] for d in sim if norml(eval(d["a"])[2]) != norml(eval(d["b"])[2])]
    print(f"  coarse-vs-coarse shape r: same substride median={np.median(sim_same):.3f} "
          f"(n={len(sim_same)}), different substride median={np.median(sim_diff):.3f} "
          f"(n={len(sim_diff)})")
    # matrix of the 6 '(-35<s<-10)' coarse curves
    fast_ck = [k for k in ckeys if "(-35<s<-10)" in k[2]]
    print("  the six '(-35<s<-10)' coarse curves (normalised to unit sum):")
    FM = np.array([reduce_to_grid(curves[k]) for k in fast_ck])
    FM = FM / FM.sum(axis=1, keepdims=True)
    for i in range(len(fast_ck)):
        print("    " + " ".join(f"{v:.4f}" for v in FM[i][:24]))
    print(f"    pairwise r of these six: "
          f"{[round(corr(FM[i],FM[j]),3) for i in range(len(fast_ck)) for j in range(i+1,len(fast_ck))]}")

    # does the coarse curve track its own fine counterpart better than a generic
    # coarse curve does?  Compare r(coarse_k, fine_k) vs r(coarse_k, fine_l!=k)
    print("  coarse_k vs fine_j correlation matrix (rows coarse, cols fine):")
    fkeys = [k for k in curves if k[0] == "fine"]
    cn = M  # coarse curves already reduced to xgrid & unit-normalised
    tab = np.full((len(ckeys), len(fkeys)), np.nan)
    for i in range(len(ckeys)):
        for j in range(len(fkeys)):
            # nearest-coarse-bin reduction of the fine curve
            red = np.zeros(len(xgrid))
            for (xf, yf) in curves[fkeys[j]]:
                k = int(np.argmin(np.abs(xgrid - xf)))
                red[k] += yf
            if red.sum() > 0:
                tab[i, j] = corr(cn[i], red / red.sum())
    same = [tab[i, j] for i in range(len(ckeys)) for j in range(len(fkeys))
            if ckeys[i][2].replace(" ", "") == fkeys[j][2].replace(" ", "")]
    off = [tab[i, j] for i in range(len(ckeys)) for j in range(len(fkeys))
           if ckeys[i][2].replace(" ", "") != fkeys[j][2].replace(" ", "")]
    # best-shift test: does ANY integer shift of the fine curve beat the zero-shift one?
    shifts = {}
    for i in range(len(ckeys)):
        row = []
        for ds in range(-8, 9):
            rr = []
            for j in range(len(fkeys)):
                red = np.zeros(len(xgrid))
                for (xf, yf) in curves[fkeys[j]]:
                    k = int(np.argmin(np.abs(xgrid - (xf + ds * 0.875))))
                    red[k] += yf
                if red.sum() > 0:
                    rr.append(corr(cn[i], red / red.sum()))
            row.append(np.nanmedian(rr) if rr else np.nan)
        shifts[i] = row
    best = [int(np.nanargmax(shifts[i])) - 8 for i in range(len(ckeys))]
    print(f"    best integer 0.875 MHz/s shift per coarse curve: {best}")
    print(f"    same-substride r median={np.median(same):+.3f} (n={len(same)})   "
          f"cross-substride r median={np.median(off):+.3f} (n={len(off)})")
    rep["coarse_template"] = dict(
        n_coarse=len(ckeys), grid=list(map(float, xgrid)),
        same_substride_r_median=float(np.median(sim_same)),
        diff_substride_r_median=float(np.median(sim_diff)),
        coarse_vs_own_fine_r_median=float(np.nanmedian(same)),
        coarse_vs_other_fine_r_median=float(np.nanmedian(off)),
        coarse_pairwise=sim)
    with open(OUT / "covx_coarse_shape_similarity.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["a", "b", "pearson", "l1", "same_substride"])
        w.writeheader()
        w.writerows(sim)
    np.savetxt(OUT / "covx_coarse_fine_corr_matrix.csv", tab, delimiter=",",
               header=",".join(str(k) for k in fkeys), comments="")
    with open(OUT / "covx_coarse_shift_test.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["coarse_curve", "best_shift_bins", "best_shift_MHzs"] + [f"shift_{d}" for d in range(-8, 9)])
        for i in range(len(ckeys)):
            w.writerow([str(ckeys[i]), best[i], best[i] * 0.875] + list(shifts[i]))

    # figure: the 18 coarse curves overlaid + the 6 fast ones
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), dpi=130)
    for k in ckeys:
        ax[0].plot(curves[k][:, 0], curves[k][:, 1], lw=0.9,
                   ls="-" if "(-35<s<-10)" in k[2] else "--",
                   color="C0" if "(-35<s<-10)" in k[2] else "C1")
    ax[0].set_title("all 18 coarse histograms overlaid\n"
                    "(solid = '(-35<s<-10)', dashed = the two slow substrata)")
    ax[0].set_xlabel("drift rate bin centre (MHz/s)")
    ax[0].set_yscale("log")
    FM2 = np.array([reduce_to_grid(curves[k]) for k in fast_ck])
    FM2 = FM2 / FM2.sum(axis=1, keepdims=True)
    for i, k in enumerate(fast_ck):
        ax[1].plot(xgrid, FM2[i], lw=1.0, label=k[2][:26])
    ax[1].set_title("the six '(-35<s<-10)' coarse curves, unit-normalised")
    ax[1].set_xlabel("drift rate bin centre (MHz/s)")
    ax[1].legend(fontsize=6)
    fig.suptitle("covx: coarse drift histograms are near-identical in shape across "
                 "group and substride")
    fig.tight_layout()
    p = FIG / "covx_coarse_template.png"
    fig.savefig(p); plt.close(fig); figlist.append(str(p))
    print(f"  wrote {p}")

    # ------------------------------------------------- 2 image structural tests
    print("=" * 78)
    print("2  IMAGE SHEETS: structural tests")
    z = np.load(OUT / "s14_images.npz")
    extra = ["Sheet_008", "Sheet_015", "Sheet_016", "Sheet_017", "Sheet_018",
             "Sheet_021", "Sheet_022", "Sheet_036", "Sheet_037"]
    ev = {}
    for i in range(1, 6):
        arr = np.genfromtxt(DATA / f"event_{i}_dynamic_spectrum.tsv", delimiter="\t", skip_header=1)
        ev[f"event_{i}"] = arr[:, 1:]
    bounds = {}
    for r in csv.DictReader(open(OUT / "s11_image_bounds.csv")):
        bounds[r["sheet"]] = r
    print("  bounds rows:")
    for s in extra + ["Sheet_003"]:
        b = bounds[s]
        print(f"    {s}  t=[{float(b['tmin']):.3f},{float(b['tmax']):.3f}] "
              f"f=[{float(b['fmin']):.3f},{float(b['fmax']):.3f}] drift={b['drift']}")

    print("  shared-bounds groups and their relationship:")
    groups = [["Sheet_015", "Sheet_016"], ["Sheet_017", "Sheet_018"],
              ["Sheet_021", "Sheet_022"], ["Sheet_036", "Sheet_037"]]
    rowsG = []
    for g in groups:
        A, B = z["grid__" + g[0]], z["grid__" + g[1]]
        # test: B = a*A + b (linear rescale of A)?
        x, y = A.ravel(), B.ravel()
        c = np.polyfit(x, y, 1)
        res = y - (c[0] * x + c[1])
        # test: B = A - smooth baseline (row+column median polish, 1 pass)
        r = np.median(A, axis=1, keepdims=True)
        col = np.median(A - r, axis=0, keepdims=True)
        polish = r + col
        residA = A - polish
        rowsG.append(dict(
            positive=g[0], partner=g[1],
            r_AB=corr(x, y), slope_B_on_A=float(c[0]), intercept=float(c[1]),
            resid_sd=float(res.std()), resid_maxabs=float(np.abs(res).max()),
            A_mean=float(A.mean()), B_mean=float(B.mean()),
            A_std=float(A.std()), B_std=float(B.std()),
            r_of_A_with_polish=corr(A.ravel(), polish.ravel()),
            resid_std_after_polish=float(residA.std()),
            residA_mean=float(residA.mean()),
            # sign agreement
            frac_same_sign=float((np.sign(A - A.mean()) == np.sign(B - B.mean())).mean())))
        d = rowsG[-1]
        print(f"    {g[0]} vs {g[1]}: r={d['r_AB']:+.4f} slope={d['slope_B_on_A']:+.4f} "
              f"resid_sd={d['resid_sd']:.4f}  A_std={d['A_std']:.4f} B_std={d['B_std']:.4f} "
              f"resid_after_rowcolmedian(A)_std={d['resid_std_after_polish']:.4f}")
    rep["image_bounds_groups"] = rowsG
    with open(OUT / "covx_image_bounds_groups.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rowsG[0].keys()))
        w.writeheader()
        w.writerows(rowsG)

    # standalone positive images: same encoding as the normalized events?
    print("  standalone images: encoding fingerprint vs the 5 normalized events")
    rowsF = []
    for s in extra:
        G = z["grid__" + s]
        row = dict(sheet=s, in_normalized=False,
                   frac_eq_1=float((G == 1.0).mean()), frac_gt_1=float((G > 1.0).mean()),
                   frac_lt_0=float((G < 0).mean()),
                   vmin=float(G.min()), vmax=float(G.max()), mean=float(G.mean()))
        for eid, E in ev.items():
            row[f"frac_eq1_{eid}"] = float((E == 1.0).mean())
        rowsF.append(row)
    for r in rowsF[:1]:
        print(f"    reference events: frac_eq1 = "
              f"{[round(r['frac_eq1_'+e], 3) for e in ev]}, frac>1 = "
              f"{[round(float((v>1).mean()),3) for v in ev.values()]}")
    for r in rowsF:
        print(f"    {r['sheet']}: frac==1 {r['frac_eq_1']:.3f} frac>1 {r['frac_gt_1']:.3f} "
              f"frac<0 {r['frac_lt_0']:.3f} range[{r['vmin']:.3g},{r['vmax']:.3g}]")
    rep["image_standalone"] = rowsF
    with open(OUT / "covx_image_encoding_fingerprint.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rowsF[0].keys()))
        w.writeheader()
        w.writerows(rowsF)

    # ------------------------------------------------- 3 coverage table
    print("=" * 78)
    print("3  DEFINITIVE COVERAGE TABLE")
    rows = []
    with open(OUT / "s16_coverage_block.csv") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    counts = collections.Counter(r["status"] for r in rows)
    nrows = collections.Counter()
    ysum = collections.Counter()
    for r in rows:
        nrows[r["status"]] += int(r["n"] or 0)
        try:
            ysum[r["status"]] += float(r["y_sum"] or 0)
        except ValueError:
            pass
    print(f"  blocks: {dict(counts)}  total={sum(counts.values())}")
    print(f"  rows:   {dict(nrows)}")
    print(f"  y-sum:  {dict((k, round(v,1)) for k,v in ysum.items())}")
    # rows in the normalized views vs rows in the workbook for shared blocks
    with open(DATA / "drift_population_series.tsv") as fh:
        pop = list(csv.DictReader(fh, delimiter="\t"))
    with open(DATA / "orientation_count_series.tsv") as fh:
        ori = list(csv.DictReader(fh, delimiter="\t"))
    pop_n = len(pop)
    ori_n = len(ori)
    pop_unit = sorted(set(r["population_series_id"] for r in pop)) and 6
    ori_unit = 6
    print(f"  normalized drift-population rows={pop_n} across 6 series "
          f"(from 30 workbook blocks = {nrows['exact']+nrows['aggregate']} rows incl. labels)")
    print(f"  normalized orientation rows={ori_n} across 6 series (from 18 workbook blocks)")
    print(f"  workbook table rows not represented = {nrows['absent']} "
          f"({100*nrows['absent']/(nrows['absent']+nrows['exact']+nrows['aggregate']):.1f}% of all table rows)")

    # which absent rows are COUNTS vs ANGLES vs PROFILES
    fam = {}
    fam["Sheet_001/002 (cml angle vs phase angle)"] = dict(
        sheets=["Sheet_001", "Sheet_002"], blocks=12, rows=2 * 33631,
        content="angles in degrees: x is 'cml angle' 0-360, y is a phase angle; "
                "NOT event counts")
    fam["Sheet_019/020/023/038 (radial distance / SNR)"] = dict(
        sheets=["Sheet_019", "Sheet_020", "Sheet_023", "Sheet_038"], blocks=14, rows=1600 + 621 + 535 + 535,
        content="radial profiles, fitted models and per-sheet SNR scalars")
    fam["Sheet_039..044 (coarse 1.75 MHz/s drift histograms)"] = dict(
        sheets=[f"Sheet_{i:03d}" for i in range(39, 45)], blocks=18, rows=60 * 3 + 10,
        content="event counts in 1.75 MHz/s bins over 0-33.25 MHz/s")
    fam["Sheet_046/047 (group_b C variants)"] = dict(
        sheets=["Sheet_046", "Sheet_047"], blocks=6, rows=65 * 2,
        content="event counts on the orientation grid, ~0.81-0.97 of the base blocks")
    rep["coverage_table"] = dict(blocks_by_status=dict(counts), rows_by_status=dict(nrows),
                                 ysum_by_status={k: float(v) for k, v in ysum.items()},
                                 normalized_population_rows=pop_n,
                                 normalized_orientation_rows=ori_n, families=fam)
    with open(OUT / "covx_coverage_final.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) + ["scientific_content_class"])
        w.writeheader()
        cls = {"Sheet_001": "angle series (degrees)", "Sheet_002": "angle series (degrees)",
               "Sheet_019": "radial profile", "Sheet_020": "radial fit + SNR",
               "Sheet_023": "radial fit + SNR", "Sheet_038": "radial fit + SNR",
               "Sheet_039": "coarse drift histogram", "Sheet_040": "coarse drift histogram",
               "Sheet_041": "coarse drift histogram", "Sheet_042": "coarse drift histogram",
               "Sheet_043": "coarse drift histogram", "Sheet_044": "coarse drift histogram",
               "Sheet_046": "orientation counts (variant)", "Sheet_047": "orientation counts (variant)"}
        for r in rows:
            r = dict(r)
            r["scientific_content_class"] = cls.get(r["sheet"], "dynamic-spectrum pixels" if r["status"] == "image" else "event counts")
            w.writerow(r)
    with open(OUT / "covx_coverage_final.json", "w") as fh:
        json.dump(rep["coverage_table"], fh, indent=2, default=str)
    print(f"  wrote {OUT/'covx_coverage_final.csv'} / .json")

    rep["figures"] = figlist
    with open(OUT / "covx_tests.json", "w") as fh:
        json.dump(rep, fh, indent=2, default=str)
    print(f"wrote {OUT/'covx_tests.json'}")


if __name__ == "__main__":
    main()