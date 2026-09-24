#!/usr/bin/env python3
"""covx_s03 (dimension: coverage boundary) - mechanism tests for the absent families.

Follows covx_s02 with sharper numerical tests:

  A  Sheet_001/002: is the x grid *identical* point-for-point, and is the y2
     column a deterministic function of y1 (or of y1 modulo 360)?
  C  Sheet_039..044: how far is the coarse histogram from the fine one after a
     proper density rebinning?  Mean, median, Wasserstein-1, total variation,
     peak position, and the coarse block's internal bin-width heterogeneity.
  D  Sheet_046/047: per-bin ratio structure vs base; is the deficit uniform
     (a scale factor) or concentrated in the high-count bins (a subsample)?
  E  the 9 unrepresented images: full 14x14 linear-fit matrix (R^2 and slope),
     plus explicit tests for 'sum', 'difference' and 'partner' relationships.

Outputs: covx_* csv/json + one figure.
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


def hist_stats(x, y):
    """Distributional summaries from a histogram given bin centres x and counts y."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    t = y.sum()
    if t <= 0:
        return dict(total=0.0, mean=float("nan"), sd=float("nan"), peak_x=float("nan"),
                    peak_frac=float("nan"), n_bins=int(len(x)), p5=float("nan"),
                    p50=float("nan"), p95=float("nan"))
    w = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    # treat each bin as a point mass at its centre for the moments
    m = float((x * y).sum() / t)
    sd = float(np.sqrt(((x - m) ** 2 * y).sum() / t))
    c = np.cumsum(y) / t

    def q(p):
        i = int(np.searchsorted(c, p))
        return float(x[min(i, len(x) - 1)])
    return dict(total=float(t), mean=m, sd=sd, peak_x=float(x[int(np.argmax(y))]),
                peak_frac=float(y.max() / t), n_bins=int(len(x)), width=w,
                p5=q(0.05), p50=q(0.50), p95=q(0.95))


def w1_from_hist(xa, ya, xb, yb):
    """Wasserstein-1 between two 1-D distributions given as bin-centre masses."""
    xa, ya = np.asarray(xa, float), np.asarray(ya, float)
    xb, yb = np.asarray(xb, float), np.asarray(yb, float)
    if ya.sum() <= 0 or yb.sum() <= 0:
        return float("nan")
    pa = ya / ya.sum()
    pb = yb / yb.sum()
    grid = np.unique(np.concatenate([xa, xb]))
    ca = np.array([pa[xa <= g].sum() for g in grid])
    cb = np.array([pb[xb <= g].sum() for g in grid])
    return float(np.trapezoid(np.abs(ca - cb), grid))


def main():
    blk, order = blocks()
    rep = {}
    figlist = []

    # ---------------------------------------------------------------- A
    print("=" * 78)
    print("A  Sheet_001 / Sheet_002 - are the y columns related?")
    lab1 = [k for k in order if k[0] == "Sheet_001"]
    lab2 = [k for k in order if k[0] == "Sheet_002"]
    rowsA = []
    for k1, k2 in zip(lab1, lab2):
        a, b = blk[k1], blk[k2]
        y1, y2 = a[:, 1], b[:, 1]
        # is y2 a deterministic linear function of y1?
        A = np.vstack([y1, np.ones_like(y1)]).T
        coef, res, *_ = np.linalg.lstsq(A, y2, rcond=None)
        pred = A @ coef
        ss = ((y2 - y2.mean()) ** 2).sum()
        r2 = 1 - ((y2 - pred) ** 2).sum() / ss if ss > 0 else float("nan")
        # modulo-360 view: is y2 ~ y1 mod 360, or 180-y1?
        d1 = np.abs(((y2 - y1) + 180) % 360 - 180)
        d2 = np.abs(((y2 + y1) + 180) % 360 - 180)
        rowsA.append(dict(label=k1[1], n=len(y1),
                          r=float(np.corrcoef(y1, y2)[0, 1]),
                          slope=float(coef[0]), intercept=float(coef[1]), r2_linear=float(r2),
                          max_abs_dev_linear=float(np.max(np.abs(y2 - pred))),
                          med_abs_mod360_y2_minus_y1=float(np.median(d1)),
                          med_abs_mod360_y2_plus_y1=float(np.median(d2))))
    for d in rowsA:
        print(f"  {d['label']:22s} n={d['n']:6d} r={d['r']:+.4f} slope={d['slope']:+.5f} "
              f"R2={d['r2_linear']:.5f} max|dev|={d['max_abs_dev_linear']:.4g} "
              f"med|dY mod360|={d['med_abs_mod360_y2_minus_y1']:.3f} "
              f"med|sum mod360|={d['med_abs_mod360_y2_plus_y1']:.3f}")
    rep["A_s001_s002"] = rowsA
    with open(OUT / "covx_s001_s002_relations.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rowsA[0].keys()))
        w.writeheader()
        w.writerows(rowsA)

    # ---------------------------------------------------------------- C
    print("=" * 78)
    print("C  coarse (Sheet_039..044) vs fine blocks: distributional distance")
    coarse_sheet = {39: ("Sheet_025", "Sheet_024"), 40: ("Sheet_027", "Sheet_026"),
                    41: ("Sheet_029", "Sheet_028"), 42: ("Sheet_031", "Sheet_030"),
                    43: ("Sheet_033", "Sheet_032"), 44: ("Sheet_035", "Sheet_034")}
    rowsC = []
    for sh, (fast_sh, slow_sh) in coarse_sheet.items():
        csheet = f"Sheet_{sh:03d}"
        for cl in [k for k in order if k[0] == csheet]:
            norm = lambda s: s.replace(" ", "")
            cand = [k for k in order if k[0] in (fast_sh, slow_sh)
                    and norm(k[1]) == norm(cl[1])]
            if not cand:
                continue
            fa = blk[cand[0]]
            ca = blk[cl]
            sa, sf = hist_stats(ca[:, 0], ca[:, 1]), hist_stats(fa[:, 0], fa[:, 1])
            # density rebinning of the FINE blocks onto the coarse x grid
            cw = sa["width"]
            # coarse x grid may be non-uniform: use midpoint boundaries
            xs = ca[:, 0]
            edges = np.concatenate([[xs[0] - cw / 2],
                                    (xs[:-1] + xs[1:]) / 2, [xs[-1] + cw / 2]])
            fw = sf["width"]
            redge = fa[:, 0] + fw / 2
            reb = np.zeros(len(xs))
            for j in range(len(fa)):
                lo_e = fa[j, 0] - fw / 2
                hi_e = fa[j, 0] + fw / 2
                for kk in range(len(xs)):
                    ov = max(0.0, min(hi_e, edges[kk + 1]) - max(lo_e, edges[kk]))
                    if ov > 0:
                        reb[kk] += fa[j, 1] * ov / fw
            tv = 0.5 * float(np.abs(reb / reb.sum() - ca[:, 1] / ca[:, 1].sum()).sum())
            w1 = w1_from_hist(ca[:, 0], ca[:, 1], fa[:, 0], fa[:, 1])
            rowsC.append(dict(
                coarse_sheet=csheet, label=cl[1], fine_sheet=cand[0][0],
                coarse_total=sa["total"], fine_total=sf["total"],
                coarse_bins=sa["n_bins"], fine_bins=sf["n_bins"],
                coarse_width=cw, fine_width=fw,
                coarse_mean=sa["mean"], fine_mean=sf["mean"],
                mean_shift=sa["mean"] - sf["mean"],
                coarse_median=sa["p50"], fine_median=sf["p50"],
                coarse_sd=sa["sd"], fine_sd=sf["sd"],
                coarse_peak_x=sa["peak_x"], fine_peak_x=sf["peak_x"],
                coarse_peak_frac=sa["peak_frac"], fine_peak_frac=sf["peak_frac"],
                wasserstein1=w1, total_variation=tv,
                max_abs_bin_ratio=float(np.max(np.maximum(np.abs(reb / np.where(ca[:,1]>0, ca[:,1], np.nan)), 1e-9))),
                n_bins_coarse_zero=int((ca[:, 1] == 0).sum()),
                n_bins_fine_zero=int((fa[:, 1] == 0).sum()),
            ))
    for d in rowsC:
        print(f"  {d['coarse_sheet']} {d['label']:32s} ncoarse={d['coarse_bins']:2d} nfine={d['fine_bins']:3d} "
              f"tot {d['coarse_total']:8.0f} vs {d['fine_total']:8.0f}  "
              f"mean {d['coarse_mean']:6.3f} vs {d['fine_mean']:6.3f} (shift {d['mean_shift']:+.3f})  "
              f"peak@{d['coarse_peak_x']:.2f} vs {d['fine_peak_x']:.2f}  "
              f"W1={d['wasserstein1']:.3f} TV={d['total_variation']:.3f}")
    rep["C_coarse_vs_fine"] = rowsC
    with open(OUT / "covx_coarse_vs_fine_distances.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rowsC[0].keys()))
        w.writeheader()
        w.writerows(rowsC)

    # ---------------------------------------------------------------- D
    print("=" * 78)
    print("D  Sheet_046/047 variants - per-bin ratio structure")
    base = {}
    for r in csv.DictReader(open(OUT / "s11_blocks.csv")):
        if r["sheet"] in ("Sheet_011", "Sheet_012"):
            base.setdefault((r["sheet"], r["block_label"]), []).append(
                (float(r["x"]), float(r["y"])))
    base = {k: np.asarray(v, float) for k, v in base.items()}
    b11 = sorted([k for k in base if k[0] == "Sheet_011"], key=lambda k: base[k][0, 0])
    b12 = sorted([k for k in base if k[0] == "Sheet_012"], key=lambda k: base[k][0, 0])
    rowsD = []
    for vsheet in ("Sheet_046", "Sheet_047"):
        vl = [k for k in order if k[0] == vsheet]
        for j, vk in enumerate(vl):
            va = blk[vk]
            for name, blist in (("channel_l", b11), ("channel_r", b12)):
                ba = base[blist[j]]
                # x grids: variant block j may be finer (0.5) than base (1.0)
                wv = float(np.median(np.diff(va[:, 0]))) if len(va) > 1 else 1.0
                wb = float(np.median(np.diff(ba[:, 0]))) if len(ba) > 1 else 1.0
                f = min(wv, wb)

                def dens(a, w):
                    c = collections.Counter()
                    for (xx, yy) in a:
                        c[round(xx / w)] += yy
                    return c
                ca_, cb_ = dens(va, f), dens(ba, f)
                ks = sorted(set(ca_) | set(cb_))
                pa = np.array([ca_.get(k, 0.0) for k in ks])
                pb = np.array([cb_.get(k, 0.0) for k in ks])
                m = pb > 0
                ratio = pa[m] / pb[m]
                fr = pa.sum() / pb.sum()
                rowsD.append(dict(variant_sheet=vsheet, block_index=j,
                                  variant_label=vk[1], compared_to=name,
                                  n_bins=int(m.sum()),
                                  variant_total=float(pa.sum()), base_total=float(pb.sum()),
                                  overall_ratio=float(fr),
                                  ratio_mean=float(ratio.mean()), ratio_sd=float(ratio.std()),
                                  ratio_cv=float(ratio.std() / ratio.mean()) if ratio.mean() else float("nan"),
                                  ratio_min=float(ratio.min()), ratio_max=float(ratio.max()),
                                  # correlation of ratio with base count (subsample -> negative)
                                  corr_ratio_vs_base=float(np.corrcoef(ratio, pb[m])[0, 1]) if m.sum() > 2 else float("nan"),
                                  corr_ratio_vs_x=float(np.corrcoef(ratio, np.array(ks, float)[m])[0, 1]) if m.sum() > 2 else float("nan"),
                                  l1_rel=float(np.abs(pa - pb).sum() / pb.sum()),
                                  pearson=float(np.corrcoef(pa, pb)[0, 1]) if pa.std() > 0 and pb.std() > 0 else float("nan"),
                                  # binomial-subsample chi2 with f fixed at the overall ratio
                                  chi2_bin_per_dof=float(np.sum((pa[m] - fr * pb[m]) ** 2 /
                                                                np.maximum(fr * (1 - fr) * pb[m], 1e-12)) /
                                                         max(int(m.sum()) - 1, 1)),
                                  # pure-scale chi2 (var = f^2 b)
                                  chi2_scale_per_dof=float(np.sum((pa[m] - fr * pb[m]) ** 2 /
                                                                  np.maximum(fr ** 2 * pb[m], 1e-12)) /
                                                           max(int(m.sum()) - 1, 1))))
    for d in rowsD:
        print(f"  {d['variant_sheet']} b{d['block_index']} vs {d['compared_to']}: "
              f"n={d['n_bins']:2d} R={d['overall_ratio']:.4f} "
              f"ratio mean={d['ratio_mean']:.4f} sd={d['ratio_sd']:.4f} "
              f"cv={d['ratio_cv']:.3f} min={d['ratio_min']:.3f} max={d['ratio_max']:.3f} "
              f"r(ratio,base)={d['corr_ratio_vs_base']:+.3f} r={d['pearson']:.5f} "
              f"chi2_bin/dof={d['chi2_bin_per_dof']:.2f}")
    rep["D_variant_ratios"] = rowsD
    with open(OUT / "covx_group_b_C_ratio_structure.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rowsD[0].keys()))
        w.writeheader()
        w.writerows(rowsD)

    # ---------------------------------------------------------------- E
    print("=" * 78)
    print("E  linear-fit matrix over the 14 image sheets + 5 events")
    z = np.load(OUT / "s14_images.npz")
    names = ["Sheet_003", "Sheet_004", "Sheet_005", "Sheet_006", "Sheet_007",
             "Sheet_008", "Sheet_015", "Sheet_016", "Sheet_017", "Sheet_018",
             "Sheet_021", "Sheet_022", "Sheet_036", "Sheet_037"]
    ev = {}
    for i in range(1, 6):
        arr = np.genfromtxt(DATA / f"event_{i}_dynamic_spectrum.tsv", delimiter="\t", skip_header=1)
        ev[f"event_{i}"] = arr[:, 1:]
    mats = {s: z["grid__" + s] for s in names}
    mats.update(ev)
    keys = list(mats)
    # subsample for speed: every 7th pixel each way (chi2 unaffected in ranking)
    sub = {k: v[::7, ::7].ravel() for k, v in mats.items()}
    rowsE = []
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if i >= j:
                continue
            x, y = sub[a], sub[b]
            sx, sy = x.std(), y.std()
            if sx == 0 or sy == 0:
                continue
            r = float(np.corrcoef(x, y)[0, 1])
            slope = float(np.polyfit(x, y, 1)[0])
            rowsE.append(dict(a=a, b=b, pearson=r, r2=r * r, slope_b_on_a=slope,
                              mean_abs_diff=float(np.abs(x - y).mean()),
                              frac_identical=float((x == y).mean())))
    rowsE.sort(key=lambda d: -abs(d["pearson"]))
    print("  top 18 pairs by |r|:")
    for d in rowsE[:18]:
        print(f"    {d['a']:10s} {d['b']:10s} r={d['pearson']:+.4f} R2={d['r2']:.4f} "
              f"slope={d['slope_b_on_a']:+.4f} mae={d['mean_abs_diff']:.4f}")
    rep["E_linear_fits"] = rowsE
    with open(OUT / "covx_image_linear_fits.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rowsE[0].keys()))
        w.writeheader()
        w.writerows(rowsE)

    # explicit structural tests on the 4 bounds-sharing pairs
    print("  bounds-sharing pair tests (positive partner vs signed partner):")
    pairs = [("Sheet_015", "Sheet_016"), ("Sheet_017", "Sheet_018"),
             ("Sheet_021", "Sheet_022"), ("Sheet_036", "Sheet_037")]
    rowsP = []
    for a, b in pairs:
        A, B = mats[a], mats[b]
        s = A + B
        d = A - B
        rowsP.append(dict(a=a, b=b,
                          r=float(np.corrcoef(A.ravel(), B.ravel())[0, 1]),
                          sum_std=float(s.std()), diff_std=float(d.std()),
                          sum_mean=float(s.mean()), sum_range=float(s.max() - s.min()),
                          diff_mean=float(d.mean()),
                          r_sum_vs_unit=float(np.corrcoef(s.ravel(), (A * 0 + 1.0).ravel())[0, 1]) if s.std() > 0 else float("nan"),
                          b_over_a_slope=float(np.polyfit(A.ravel(), B.ravel(), 1)[0])))
        print(f"    {a}+{b}: r(A,B)={rowsP[-1]['r']:+.4f} std(A+B)={s.std():.4f} "
              f"std(A-B)={d.std():.4f} mean(A+B)={s.mean():.4f} slope(B on A)={rowsP[-1]['b_over_a_slope']:+.4f}")
    rep["E_pair_tests"] = rowsP
    with open(OUT / "covx_image_pair_tests.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rowsP[0].keys()))
        w.writeheader()
        w.writerows(rowsP)

    # row/column structure of the signed images: are they row-baseline subtracted?
    print("  row/column profile structure of the 9 unrepresented images:")
    rowsS = []
    for s in ["Sheet_008", "Sheet_015", "Sheet_016", "Sheet_017", "Sheet_018",
              "Sheet_021", "Sheet_022", "Sheet_036", "Sheet_037"]:
        G = mats[s]
        rmean = G.mean(axis=1)
        cmean = G.mean(axis=0)
        rowsS.append(dict(sheet=s,
                          row_mean_sd=float(rmean.std()), col_mean_sd=float(cmean.std()),
                          grand_mean=float(G.mean()),
                          row0_mean=float(G[0].mean()), row_last_mean=float(G[-1].mean()),
                          corr_rowmean_with_event3=float(np.corrcoef(rmean, mats["Sheet_005"].mean(axis=1))[0, 1]),
                          frac_gt_zero=float((G > 0).mean())))
        print(f"    {s}: grand_mean={rowsS[-1]['grand_mean']:+.4f} "
              f"sd(row means)={rowsS[-1]['row_mean_sd']:.4f} sd(col means)={rowsS[-1]['col_mean_sd']:.4f} "
              f"first-row mean={rowsS[-1]['row0_mean']:+.4f} last-row mean={rowsS[-1]['row_last_mean']:+.4f}")
    rep["E_profile_structure"] = rowsS
    with open(OUT / "covx_image_profile_structure.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rowsS[0].keys()))
        w.writeheader()
        w.writerows(rowsS)

    # figure: the bounds-sharing quartet, downsampled
    fig, ax = plt.subplots(2, 4, figsize=(16, 8), dpi=130)
    for col, (a, b) in enumerate(pairs):
        for row, s in enumerate((a, b)):
            G = mats[s]
            f = max(1, int(math.ceil(G.shape[0] / 64)))
            Gs = G[:(G.shape[0] // f) * f, :(G.shape[1] // f) * f].reshape(
                G.shape[0] // f, f, G.shape[1] // f, f).mean(axis=(1, 3))
            im = ax[row, col].imshow(Gs, aspect="auto", interpolation="nearest")
            ax[row, col].set_title(f"{s}  [{np.nanmin(G):.3g},{np.nanmax(G):.3g}]", fontsize=8)
            ax[row, col].set_xticks([]); ax[row, col].set_yticks([])
            fig.colorbar(im, ax=ax[row, col], fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
    fig.suptitle("covx: the four bounds-sharing image pairs - top row 'positive' partner, "
                 "bottom row signed residual partner")
    fig.tight_layout()
    p = FIG / "covx_image_pairs.png"
    fig.savefig(p)
    plt.close(fig)
    figlist.append(str(p))
    print(f"  wrote {p}")

    rep["figures"] = figlist
    with open(OUT / "covx_mechanisms.json", "w") as fh:
        json.dump(rep, fh, indent=2, default=str)
    print(f"wrote {OUT/'covx_mechanisms.json'}")


if __name__ == "__main__":
    main()