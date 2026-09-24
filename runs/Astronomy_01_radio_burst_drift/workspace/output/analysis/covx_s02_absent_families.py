#!/usr/bin/env python3
"""covx_s02 (dimension: coverage boundary) - characterise the 50 absent blocks.

Inputs (all already on disk, no workbook access):
  output/results/covx_absent_blocks.csv   raw (sheet,row,label,x,y) of the absent
                                          families + fine slow/fast block copies
  output/results/s11_block_index.csv      block sizes
  output/results/s16_coverage_block.csv   coverage classification
  output/results/s14_images.npz           14 image grids
  output/results/s14_image_stats.csv
  data/normalized_views/*.tsv             the normalized views

Outputs: covx_* csv/json tables + figures.
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

FIG.mkdir(parents=True, exist_ok=True)


def load_blocks():
    """{(sheet,label): [(x,y), ...]} from the covx tidy dump.

    A block's LABEL row also carries its first data point (cols C/D), exactly as
    the s11 parser assumed; numeric x AND numeric y are therefore taken from both
    'label' and 'point' rows.  The 'params' pseudo-block has y=None and stays out.
    """
    blk = collections.defaultdict(list)
    order = []
    with open(OUT / "covx_absent_blocks.csv") as fh:
        for r in csv.DictReader(fh):
            if r["kind"] not in ("point", "label"):
                continue
            if r["x"] in ("", "None") or r["y"] in ("", "None"):
                continue
            k = (r["sheet"], r["label"])
            if k not in blk:
                order.append(k)
            blk[k].append((float(r["x"]), float(r["y"])))
    return blk, order


def corr(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    fig_manifest = []
    blk, order = load_blocks()
    idx = {}
    with open(OUT / "s11_block_index.csv") as fh:
        for r in csv.DictReader(fh):
            idx[(r["sheet"], r["block_label"])] = r
    cov = {}
    with open(OUT / "s16_coverage_block.csv") as fh:
        for r in csv.DictReader(fh):
            cov[(r["sheet"], r["block_label"])] = r

    report = {}

    # ---------------------------------------------------------------- 1. S001/S002
    print("=" * 78)
    print("FAMILY A  Sheet_001 / Sheet_002  'cml angle' vs phase-angle series")
    rows = []
    lab1 = [k for k in order if k[0] == "Sheet_001"]
    lab2 = [k for k in order if k[0] == "Sheet_002"]
    print(f"  blocks: Sheet_001={len(lab1)}  Sheet_002={len(lab2)}")
    print(f"  labels S001: {[k[1] for k in lab1]}")
    print(f"  labels S002: {[k[1] for k in lab2]}")
    # pair by position
    for (k1, k2) in zip(lab1, lab2):
        a = np.asarray(blk[k1], float)
        b = np.asarray(blk[k2], float)
        same_x = (a.shape == b.shape) and np.allclose(a[:, 0], b[:, 0], rtol=0, atol=0)
        same_x_near = (a.shape == b.shape) and np.allclose(a[:, 0], b[:, 0], rtol=0, atol=1e-9)
        d = dict(
            sheet_001=k1[0], label_001=k1[1], sheet_002=k2[0], label_002=k2[1],
            n_001=len(a), n_002=len(b),
            x_min_001=float(a[:, 0].min()), x_max_001=float(a[:, 0].max()),
            y_min_001=float(a[:, 1].min()), y_max_001=float(a[:, 1].max()),
            x_min_002=float(b[:, 0].min()), x_max_002=float(b[:, 0].max()),
            y_min_002=float(b[:, 1].min()), y_max_002=float(b[:, 1].max()),
            x_grid_identical=bool(same_x), x_grid_identical_1e9=bool(same_x_near),
            max_abs_dx=float(np.max(np.abs(a[:, 0] - b[:, 0]))) if a.shape == b.shape else float("nan"),
            dx_median_001=float(np.median(np.diff(a[:, 0]))) if len(a) > 1 else float("nan"),
            pearson_y=corr(a[:, 1], b[:, 1]) if a.shape == b.shape else float("nan"),
            spearman_y=float("nan"),
            y_ratio_median=float(np.median(b[:, 1] / a[:, 1])) if a.shape == b.shape else float("nan"),
        )
        if a.shape == b.shape and len(a) > 2:
            ra = np.argsort(np.argsort(a[:, 1]))
            rb = np.argsort(np.argsort(b[:, 1]))
            d["spearman_y"] = corr(ra, rb)
        rows.append(d)
    for d in rows:
        print(f"  {d['label_001']:22s} n={d['n_001']:6d} x=[{d['x_min_001']:.3f},{d['x_max_001']:.3f}]"
              f"  y1=[{d['y_min_001']:.4f},{d['y_max_001']:.4f}]"
              f"  y2=[{d['y_min_002']:.4f},{d['y_max_002']:.4f}]"
              f"  sameX={d['x_grid_identical']}  r={d['pearson_y']:.4f}  rho={d['spearman_y']:.4f}")
    tot = dict(
        blocks=int(sum(d["n_001"] for d in rows)),
        rows_001=int(sum(d["n_001"] for d in rows)),
        rows_002=int(sum(d["n_002"] for d in rows)),
        x_all_min=float(min(d["x_min_001"] for d in rows)),
        x_all_max=float(max(d["x_max_001"] for d in rows)),
        y_all_min_001=float(min(d["y_min_001"] for d in rows)),
        y_all_max_001=float(max(d["y_max_001"] for d in rows)),
        y_all_min_002=float(min(d["y_min_002"] for d in rows)),
        y_all_max_002=float(max(d["y_max_002"] for d in rows)),
        n_blocks_same_x=int(sum(d["x_grid_identical"] for d in rows)),
    )
    print(f"  TOTAL rows S001={tot['rows_001']} S002={tot['rows_002']}  "
          f"x in [{tot['x_all_min']:.3f},{tot['x_all_max']:.3f}] "
          f"y1 in [{tot['y_all_min_001']:.4f},{tot['y_all_max_001']:.4f}] "
          f"y2 in [{tot['y_all_min_002']:.4f},{tot['y_all_max_002']:.4f}]")
    report["family_A_cml_angle"] = dict(pairs=rows, totals=tot)

    with open(OUT / "covx_s001_s002_pairs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # figure: S001 vs S002 for the biggest block + scatter of y1 vs y2
    k1 = max(lab1, key=lambda k: len(blk[k]))
    k2 = [k for k in lab2 if k[1].replace(" ", "") == k1[1].replace(" ", "")] or [k for k in lab2]
    k2 = k2[0]
    a = np.asarray(blk[k1], float)
    b = np.asarray(blk[k2], float)
    n = min(len(a), len(b))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), dpi=130)
    ax[0].plot(a[:n, 0], a[:n, 1], lw=0.4, color="C0")
    ax[0].set_title(f"{k1[0]} {k1[1]}\n'cml angle' vs 'io phase angle'  (n={n})")
    ax[0].set_xlabel("cml angle (col C)")
    ax[0].set_ylabel("io phase angle (col D)")
    ax[1].plot(b[:n, 0], b[:n, 1], lw=0.4, color="C3")
    ax[1].set_title(f"{k2[0]} {k2[1]}\n'cml angle' vs 'group_b phase angle'  (n={n})")
    ax[1].set_xlabel("cml angle (col C)")
    ax[1].set_ylabel("group_b phase angle (col D)")
    if n > 2:
        ax[2].scatter(a[:n, 1], b[:n, 1], s=1, alpha=0.3, color="k")
        ax[2].set_xlabel("Sheet_001 y (io phase angle)")
        ax[2].set_ylabel("Sheet_002 y (group_b phase angle)")
        ax[2].set_title(f"y-vs-y  pearson r={corr(a[:n,1],b[:n,1]):.4f}")
    fig.suptitle("covx: Sheet_001 vs Sheet_002 (absent from normalized views) - same x grid, different phase-angle y")
    fig.tight_layout()
    p = FIG / "covx_s001_vs_s002.png"
    fig.savefig(p)
    plt.close(fig)
    fig_manifest.append(str(p))
    print(f"  wrote {p}")

    # ---------------------------------------------------------- 2. RD / SNR family
    print("=" * 78)
    print("FAMILY B  Sheet_019/020/023/038  radial-distance / SNR blocks")
    rd_rows = []
    for (sh, lb) in order:
        if sh not in ("Sheet_019", "Sheet_020", "Sheet_023", "Sheet_038"):
            continue
        a = np.asarray(blk[(sh, lb)], float)
        i = idx.get((sh, lb))
        rd_rows.append(dict(
            sheet=sh, block_label=lb, n=len(a),
            x_min=float(a[:, 0].min()), x_max=float(a[:, 0].max()),
            x_step_median=float(np.median(np.diff(a[:, 0]))) if len(a) > 1 else float("nan"),
            y_min=float(a[:, 1].min()), y_max=float(a[:, 1].max()),
            y_mean=float(a[:, 1].mean()), y_sum=float(a[:, 1].sum()),
            y_first=float(a[0, 1]), y_last=float(a[-1, 1]),
            x_at_ymax=float(a[int(np.argmax(a[:, 1])), 0]),
        ))
    # SNR: each of these sheets carries the literal header  SNR | value  in the
    # row whose col C is the string 'SNR' (row 3 of the sheet).  Sheet_019 has no
    # such row.  'params' holds the per-sheet fitted scale.
    snr = {}
    params = {}
    with open(OUT / "covx_absent_blocks.csv") as fh:
        for r in csv.DictReader(fh):
            if r["sheet"] not in ("Sheet_019", "Sheet_020", "Sheet_023", "Sheet_038"):
                continue
            if r["raw_c"] == "SNR" and r["raw_d"] not in ("", "None"):
                snr[r["sheet"]] = float(r["raw_d"])
            if r["raw_b"] == "params" and r["raw_c"] not in ("", "None"):
                params[r["sheet"]] = float(r["raw_c"])
    print(f"  SNR header (row 3 'SNR' label -> col D): {snr}")
    print(f"  'params' block value:                   {params}")
    print(f"  SNR row (row 3, col C): {snr}")
    for d in rd_rows:
        d["SNR_header"] = snr.get(d["sheet"], float("nan"))
        d["params_value"] = params.get(d["sheet"], float("nan"))
        print(f"  {d['sheet']} {d['block_label']:20s} n={d['n']:5d} x=[{d['x_min']:.3f},{d['x_max']:.3f}]"
              f" y=[{d['y_min']:.6g},{d['y_max']:.6g}] sum={d['y_sum']:.6g}"
              f" SNR_header={d['SNR_header']} params={d['params_value']:.6g}")
    report["family_B_radial_snr"] = dict(blocks=rd_rows, snr_header=snr, params=params)
    with open(OUT / "covx_radial_snr_blocks.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rd_rows[0].keys()))
        w.writeheader()
        w.writerows(rd_rows)

    # pairwise: do the RD_corr blocks of different sheets describe the same curve?
    print("  RD_corr cross-sheet comparison (common x grid?):")
    rd_corr = {sh: (lb, np.asarray(blk[(sh, lb)], float)) for (sh, lb) in order
               if sh in ("Sheet_019", "Sheet_020", "Sheet_023", "Sheet_038") and lb in ("RD_corr", "RD_raw")}
    rd_pairs = []
    for s1, (l1, a1) in rd_corr.items():
        for s2, (l2, a2) in rd_corr.items():
            if s1 >= s2:
                continue
            n = min(len(a1), len(a2))
            same = n > 2 and np.allclose(a1[:n, 0], a2[:n, 0], atol=1e-12)
            rd_pairs.append(dict(a=f"{s1}:{l1}", b=f"{s2}:{l2}", n=n,
                                 same_x_grid=bool(same), pearson=corr(a1[:n, 1], a2[:n, 1]),
                                 max_abs_dy=float(np.max(np.abs(a1[:n, 1] - a2[:n, 1]))),
                                 rel_scale=float(np.nanmedian(np.abs(a2[:n, 1]) / np.where(np.abs(a1[:n, 1]) > 0, np.abs(a1[:n, 1]), np.nan)))))
    for d in rd_pairs:
        print(f"    {d['a']:22s} vs {d['b']:22s} n={d['n']:4d} sameX={d['same_x_grid']} "
              f"r={d['pearson']:.4f} max|dy|={d['max_abs_dy']:.4g} scale={d['rel_scale']:.3f}")
    report["family_B_pairs"] = rd_pairs
    with open(OUT / "covx_radial_snr_pairs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rd_pairs[0].keys()))
        w.writeheader()
        w.writerows(rd_pairs)

    # ------------------------------------------------- 3. coarse drift histograms
    print("=" * 78)
    print("FAMILY C  Sheet_039..044  1.75 MHz/s coarse drift histograms")
    coarse_sheet = {39: ("Sheet_025", "Sheet_024"), 40: ("Sheet_027", "Sheet_026"),
                    41: ("Sheet_029", "Sheet_028"), 42: ("Sheet_031", "Sheet_030"),
                    43: ("Sheet_033", "Sheet_032"), 44: ("Sheet_035", "Sheet_034")}
    crows = []
    coarse_curves = {}
    fine_curves = {}
    for sh, (fast_sh, slow_sh) in coarse_sheet.items():
        csheet = f"Sheet_{sh:03d}"
        clabels = [k for k in order if k[0] == csheet]
        # fine blocks: fast sheet has 1 block, slow sheet 2 blocks (union = slow)
        flabels = [k for k in order if k[0] == fast_sh]
        slabels = [k for k in order if k[0] == slow_sh]
        for cl in clabels:
            ca = np.asarray(blk[cl], float)
            # the coarse sheet carries the SAME three substride labels as the fine
            # pair, so pair strictly by normalised label text.
            norm = lambda s: s.replace(" ", "")
            cand = [k for k in (flabels + slabels) if norm(k[1]) == norm(cl[1])]
            if not cand:
                print(f"  !! no fine counterpart for {cl}")
                continue
            fa = np.asarray(blk[cand[0]], float)
            fl = cand[0][1]
            coarse_curves[(sh, cl[1])] = ca
            fine_curves[(sh, cl[1])] = fa
            ctot = float(ca[:, 1].sum())
            ftot = float(fa[:, 1].sum())
            # --- correct rebinning: the coarse histogram is indexed k=0..N-1 with
            # x_k = k*1.75, so coarse bin k covers drift in [k*1.75-0.875, k*1.75+0.875).
            # The fine bins have their own width w=median(dx).  Integrate the fine
            # DISTRIBUTION over each coarse window using the exact 1-D mass:
            # each fine bin contributes its full count to the coarse bin of its centre
            # only if fine bins are nested inside coarse bins; otherwise use overlap.
            ck = np.rint(ca[:, 0] / 1.75).astype(int)
            cmap = {int(kk): float(yy) for kk, yy in zip(ck, ca[:, 1])}
            w = float(np.median(np.diff(fa[:, 0]))) if len(fa) > 1 else 0.0
            rb = collections.Counter()
            for (xf, yf) in fa:
                lo, hi = xf - w / 2.0, xf + w / 2.0
                for kk in range(int(np.floor(lo / 1.75)) - 1, int(np.ceil(hi / 1.75)) + 2):
                    a, b = kk * 1.75 - 0.875, kk * 1.75 + 0.875
                    ov = max(0.0, min(hi, b) - max(lo, a))
                    if ov > 0:
                        rb[kk] += yf * ov / (hi - lo)
            common = sorted(set(rb) | set(cmap))
            matched = [kk for kk in common
                       if rb.get(kk, 0.0) > 0 and cmap.get(kk, 0.0) > 0]
            num = sum(abs(rb.get(kk, 0.0) - cmap.get(kk, 0.0)) for kk in common)
            den = sum(cmap.get(kk, 0.0) for kk in common)
            # distribution-shape agreement on bins where BOTH are non-zero
            pr = np.array([rb.get(kk, 0.0) for kk in common], float)
            pc = np.array([cmap.get(kk, 0.0) for kk in common], float)
            shape_r = corr(pr, pc)
            crows.append(dict(coarse_sheet=csheet, coarse_label=cl[1], fine_source=fl,
                              coarse_n=len(ca), fine_n=len(fa),
                              coarse_total=ctot, fine_total=ftot,
                              total_diff=ctot - ftot,
                              total_rel_diff=(ctot - ftot) / ftot if ftot else float("nan"),
                              n_coarse_bins=len(common), n_bins_both_nonzero=len(matched),
                              fine_bin_width=w,
                              abs_dev_rebinned=num, rel_dev_rebinned=num / den if den else float("nan"),
                              shape_pearson_all_bins=shape_r,
                              # identical values in the raw tables?
                              exact_multiset_match=bool(sorted(set(ca[:, 1])) == sorted(set(fa[:, 1])))))
    for d in crows:
        print(f"  {d['coarse_sheet']} {d['coarse_label']:32s} coarse_total={d['coarse_total']:9.1f} "
              f"fine_total={d['fine_total']:9.1f} diff={d['total_diff']:+8.1f} "
              f"rel={d['total_rel_diff']:+.4f} L1rebin/fine={d['rel_dev_rebinned']:.3f} "
              f"shape_r={d['shape_pearson_all_bins']:.3f} "
              f"both_nonzero={d['n_bins_both_nonzero']}/{d['n_coarse_bins']} "
              f"exactmultiset={d['exact_multiset_match']}")
    report["family_C_coarse_vs_fine"] = crows
    with open(OUT / "covx_coarse_vs_fine.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(crows[0].keys()))
        w.writeheader()
        w.writerows(crows)

    # figure: coarse vs fine rebinned, one panel per group (channel_l, fast+slow)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), dpi=130)
    for ax, (sh, (fast_sh, slow_sh)) in zip(axes.ravel(), coarse_sheet.items()):
        csheet = f"Sheet_{sh:03d}"
        clabels = [k for k in order if k[0] == csheet]
        for cl in clabels:
            ca = coarse_curves[(sh, cl[1])]
            fa = fine_curves[(sh, cl[1])]
            w = float(np.median(np.diff(fa[:, 0]))) if len(fa) > 1 else 0.0
            rb = collections.Counter()
            for (xf, yf) in fa:
                lo, hi = xf - w / 2.0, xf + w / 2.0
                for kk in range(int(np.floor(lo / 1.75)) - 1, int(np.ceil(hi / 1.75)) + 2):
                    a, b = kk * 1.75 - 0.875, kk * 1.75 + 0.875
                    ov = max(0.0, min(hi, b) - max(lo, a))
                    if ov > 0:
                        rb[kk] += yf * ov / (hi - lo)
            xs = sorted(set(rb) | set(np.rint(ca[:, 0] / 1.75).astype(int)))
            ax.step([x * 1.75 for x in xs], [rb.get(x, 0.0) for x in xs],
                    where="mid", lw=1.0, color="C0",
                    label="fine blocks rebinned" if cl is clabels[0] else None)
            ax.step(ca[:, 0], ca[:, 1], where="mid", lw=1.0, color="C3", ls="--",
                    label="coarse Sheet_%03d" % sh if cl is clabels[0] else None)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_title(f"Sheet_{sh:03d} (group {(sh-38)} channel_l)")
        ax.set_xlabel("drift rate (MHz/s)")
        ax.legend(fontsize=7)
    fig.suptitle("covx: coarse 1.75 MHz/s histograms (red dashed) vs fine blocks rebinned (blue)")
    fig.tight_layout()
    p = FIG / "covx_coarse_vs_fine.png"
    fig.savefig(p)
    plt.close(fig)
    fig_manifest.append(str(p))
    print(f"  wrote {p}")

    # ---------------------------------------------------- 4. group_b C variants
    print("=" * 78)
    print("FAMILY D  Sheet_046/047  'group_b C' variants vs base group_b blocks")
    # NOTE: the variant sheets label ALL THREE blocks '... channel_l', but the
    # base counterpart of the 2nd and 3rd block is the channel_r family
    # (Sheet_012), whose labels the s11/s16 index wrote with a different spacing
    # variant.  We match by BLOCK POSITION within the sheet, taking the base
    # blocks from the same positional slot of Sheet_011 (channel_l) because the
    # variant is labelled channel_l throughout.
    # base blocks live in Sheet_011 (group_b channel_l) / Sheet_012 (channel_r);
    # they are not in the covx dump, so read them from s11_blocks.csv.
    base = {}
    for r in csv.DictReader(open(OUT / "s11_blocks.csv")):
        if r["sheet"] in ("Sheet_011", "Sheet_012"):
            base.setdefault((r["sheet"], r["block_label"]), []).append(
                (float(r["x"]), float(r["y"])))
    base = {k: np.asarray(v, float) for k, v in base.items()}
    b11 = [k for k in base if k[0] == "Sheet_011"]
    b12 = [k for k in base if k[0] == "Sheet_012"]
    b11.sort(key=lambda k: base[k][0, 0])
    b12.sort(key=lambda k: base[k][0, 0])
    print(f"  base Sheet_011 blocks: {[(k[1], len(base[k]), float(base[k][:,1].sum())) for k in b11]}")
    print(f"  base Sheet_012 blocks: {[(k[1], len(base[k]), float(base[k][:,1].sum())) for k in b12]}")

    vrows = []
    for vsheet in ("Sheet_046", "Sheet_047"):
        vlabels = [k for k in order if k[0] == vsheet]
        for j, vl in enumerate(vlabels):
            va = np.asarray(blk[vl], float)
            for chname, blist in (("channel_l(base Sheet_011)", b11),
                                  ("channel_r(base Sheet_012)", b12)):
                ba = base[blist[j]]
                # common grid: integrate both onto the union of the two x grids
                xs = np.unique(np.concatenate([va[:, 0], ba[:, 0]]))
                # step = the finer of the two grids
                wv = float(np.median(np.diff(va[:, 0]))) if len(va) > 1 else 1.0
                wb = float(np.median(np.diff(ba[:, 0]))) if len(ba) > 1 else 1.0
                def dens(a, w):
                    c = collections.Counter()
                    for (xx, yy) in a:
                        c[round(xx / w)] += yy
                    return c, w
                f = min(wv, wb)
                ca_, _ = dens(va, f)
                cb_, _ = dens(ba, f)
                ks = sorted(set(ca_) | set(cb_))
                pa = np.array([ca_.get(k, 0.0) for k in ks], float)
                pb = np.array([cb_.get(k, 0.0) for k in ks], float)
                ta, tb = pa.sum(), pb.sum()
                r_ = corr(pa, pb)
                # binomial-thinning test: if variant = binomial(f, base),
                #   E[v]=f*b, Var[v]=f(1-f)*b ; chi2 with dof = #bins(b>0)
                fr = ta / tb if tb else float("nan")
                m = pb > 0
                chi2 = float(np.sum((pa[m] - fr * pb[m]) ** 2 / np.maximum(fr * (1 - fr) * pb[m], 1e-12)))
                # constant-fraction (multinomial) test: Var[v] = f^2 * b (pure scaling)
                chi2s = float(np.sum((pa[m] - fr * pb[m]) ** 2 / np.maximum(fr ** 2 * pb[m], 1e-12)))
                vrows.append(dict(
                    variant_sheet=vsheet, variant_label=vl[1], compared_to=chname,
                    n_variant=len(va), n_base=len(ba),
                    x_variant=f"[{va[0,0]:.2f},{va[-1,0]:.2f}] step {wv:.3f}",
                    x_base=f"[{ba[0,0]:.2f},{ba[-1,0]:.2f}] step {wb:.3f}",
                    x_grids_identical=bool(len(va) == len(ba) and np.allclose(va[:, 0], ba[:, 0])),
                    variant_total=ta, base_total=tb, ratio=fr,
                    pearson_on_common_grid=r_,
                    l1_rel_on_common_grid=float(np.abs(pa - pb).sum() / tb),
                    n_bins=len(ks), n_bins_base_nonzero=int(m.sum()),
                    chi2_binomial_per_dof=chi2 / max(int(m.sum()) - 1, 1),
                    chi2_scaled_per_dof=chi2s / max(int(m.sum()) - 1, 1),
                    dof=int(m.sum()) - 1,
                ))
    for d in vrows:
        print(f"  {d['variant_sheet']} blk {d['variant_label'][:26]:26s} vs {d['compared_to']:24s}"
              f" ratio={d['ratio']:.4f} r={d['pearson_on_common_grid']:.5f}"
              f" L1rel={d['l1_rel_on_common_grid']:.3f} sameX={d['x_grids_identical']}"
              f" chi2_bin/dof={d['chi2_binomial_per_dof']:.2f} chi2_scaled/dof={d['chi2_scaled_per_dof']:.2f}")
    # also: whole-family totals per candidate channel
    fam = []
    for vsheet in ("Sheet_046", "Sheet_047"):
        vt = sum(float(np.asarray(blk[k], float)[:, 1].sum()) for k in order if k[0] == vsheet)
        l = sum(float(base[k][:, 1].sum()) for k in b11)
        r = sum(float(base[k][:, 1].sum()) for k in b12)
        fam.append(dict(variant_sheet=vsheet, variant_total=vt, channel_l_total=l,
                        channel_r_total=r, ratio_vs_l=vt / l, ratio_vs_r=vt / r,
                        closer="channel_l" if abs(vt / l - 1) < abs(vt / r - 1) else "channel_r"))
        print(f"  {vsheet}: total={vt:.0f}  channel_l={l:.0f} (r={vt/l:.4f})  "
              f"channel_r={r:.0f} (r={vt/r:.4f})  -> closer to "
              f"{'channel_l' if abs(vt/l-1)<abs(vt/r-1) else 'channel_r'}")
    report["family_D_variants"] = vrows
    report["family_D_family_totals"] = fam
    with open(OUT / "covx_group_b_C_variants.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(vrows[0].keys()))
        w.writeheader()
        w.writerows(vrows)
    with open(OUT / "covx_group_b_C_family_totals.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fam[0].keys()))
        w.writeheader()
        w.writerows(fam)

    # -------------------------------------------------- 5. unrepresented images
    print("=" * 78)
    print("FAMILY E  Sheet_008/015/016/017/018/021/022/036/037  unrepresented images")
    imgs = list(np.load(OUT / "s14_images.npz").files)
    extra = [s for s in ["Sheet_008", "Sheet_015", "Sheet_016", "Sheet_017", "Sheet_018",
                         "Sheet_021", "Sheet_022", "Sheet_036", "Sheet_037"]]
    known = ["Sheet_003", "Sheet_004", "Sheet_005", "Sheet_006", "Sheet_007"]
    ev = {}
    for i in range(1, 6):
        arr = np.genfromtxt(DATA / f"event_{i}_dynamic_spectrum.tsv", delimiter="\t", skip_header=1)
        ev[f"event_{i}"] = arr[:, 1:]
    z = np.load(OUT / "s14_images.npz")
    irows = []
    for s in extra + known:
        g = z["grid__" + s]
        r = dict(sheet=s, in_normalized_views=s in known, rows=g.shape[0], cols=g.shape[1],
                 vmin=float(np.nanmin(g)), vmax=float(np.nanmax(g)),
                 mean=float(np.nanmean(g)), sum=float(np.nansum(g)),
                 n_gt1=int((g > 1).sum()), n_eq1=int((g == 1).sum()),
                 n_lt0=int((g < 0).sum()), n_nan=int(np.isnan(g).sum()),
                 frac_eq1=float((g == 1).mean()), frac_gt1=float((g > 1).mean()),
                 frac_lt0=float((g < 0).mean()))
        # best correlation with the five known events
        best, bestr = None, -2
        for eid, G in ev.items():
            rr = float(np.corrcoef(g.ravel(), G.ravel())[0, 1])
            if rr > bestr:
                best, bestr = eid, rr
        r["best_known_event"] = best
        r["best_known_event_r"] = bestr
        # is it a difference of two known images?
        bestdiff, bestdiffv = None, 1e18
        for i in range(5):
            for j in range(5):
                if i == j:
                    continue
                D = ev[f"event_{i+1}"] - ev[f"event_{j+1}"]
                v = float(np.nanmax(np.abs(g - D)))
                if v < bestdiffv:
                    bestdiff, bestdiffv = f"event_{i+1}-event_{j+1}", v
        r["closest_known_difference"] = bestdiff
        r["closest_known_difference_maxabs"] = bestdiffv
        # is it a mean/median of known events?
        M = np.mean([ev[f"event_{i+1}"] for i in range(5)], axis=0)
        r["mean_of_5_maxabs"] = float(np.nanmax(np.abs(g - M)))
        r["sum_of_5_maxabs"] = float(np.nanmax(np.abs(g - np.sum([ev[f"event_{i+1}"] for i in range(5)], axis=0))))
        irows.append(r)
    for r in irows:
        print(f"  {r['sheet']}  in_norm={r['in_normalized_views']}  range=[{r['vmin']:.4g},{r['vmax']:.4g}]"
              f" mean={r['mean']:.4f} sum={r['sum']:.1f} #eq1={r['n_eq1']} #gt1={r['n_gt1']} #lt0={r['n_lt0']}"
              f"  best_known={r['best_known_event']}(r={r['best_known_event_r']:.4f})"
              f"  closest_diff={r['closest_known_difference']}({r['closest_known_difference_maxabs']:.4g})")
    report["family_E_images"] = irows
    with open(OUT / "covx_unrepresented_images.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(irows[0].keys()))
        w.writeheader()
        w.writerows(irows)

    # classification
    cls = []
    for r in irows:
        s = r["sheet"]
        if r["vmax"] <= 1.5000001 and r["vmin"] >= 0.9999999:
            kind = "positive-unit-capped spectrum (same encoding as normalized events)"
        elif r["vmax"] > 4.0 and r["vmin"] > 1.0:
            kind = "positive uncapped image, >4x the unit ceiling"
        elif r["vmin"] < -1.0:
            kind = "signed residual / difference image"
        else:
            kind = "signed near-zero residual image"
        cls.append((s, kind, r["best_known_event_r"], r["closest_known_difference_maxabs"]))
    for s, kind, rr, dd in cls:
        print(f"    classify {s}: {kind}   best-event r={rr:.4f}  best diff maxabs={dd:.4g}")
    report["family_E_classification"] = [dict(sheet=s, kind=k, best_event_r=rr, best_diff_maxabs=dd)
                                         for s, k, rr, dd in cls]

    # montage of the 9 unrepresented images, each downsampled to <= 64x64
    fig, axes = plt.subplots(3, 3, figsize=(11, 11), dpi=130)
    for ax, s in zip(axes.ravel(), extra):
        g = z["grid__" + s]
        f = max(1, int(math.ceil(g.shape[0] / 64)))
        gs = g[: (g.shape[0] // f) * f, : (g.shape[1] // f) * f].reshape(
            g.shape[0] // f, f, g.shape[1] // f, f).mean(axis=(1, 3))
        im = ax.imshow(gs, aspect="auto", origin="upper", interpolation="nearest")
        ax.set_title(f"{s}  n={gs.shape[0]}x{gs.shape[1]}  range[{float(np.nanmin(g)):.3g},{float(np.nanmax(g)):.3g}]",
                     fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
    fig.suptitle("covx: the 9 workbook image sheets with no normalized view (downsampled to <=64x64)",
                 fontsize=11)
    fig.tight_layout()
    p = FIG / "covx_unrepresented_images_montage.png"
    fig.savefig(p)
    plt.close(fig)
    fig_manifest.append(str(p))
    print(f"  wrote {p}")

    # -------------------------------------------------------- 6. coverage table
    print("=" * 78)
    print("COVERAGE TABLE")
    counts = collections.Counter()
    rows_n = collections.Counter()
    ytot = collections.Counter()
    with open(OUT / "s16_coverage_block.csv") as fh:
        for r in csv.DictReader(fh):
            counts[r["status"]] += 1
            rows_n[r["status"]] += int(r["n"] or 0)
            try:
                ytot[r["status"]] += float(r["y_sum"] or 0.0)
            except ValueError:
                pass
    print("  blocks by status:", dict(counts))
    print("  rows   by status:", dict(rows_n))
    print("  y_sum  by status:", {k: round(v, 1) for k, v in ytot.items()})
    # normalized-view side
    with open(DATA / "drift_population_series.tsv") as fh:
        pop = list(csv.DictReader(fh, delimiter="\t"))
    with open(DATA / "orientation_count_series.tsv") as fh:
        ori = list(csv.DictReader(fh, delimiter="\t"))
    norm_stats = dict(
        population_rows=len(pop),
        population_events=float(sum(float(r["event_count"]) for r in pop)),
        population_series=sorted(set(r["population_series_id"] for r in pop)),
        orientation_rows=len(ori),
        orientation_events=float(sum(float(r["event_count"]) for r in ori)),
        orientation_series=sorted(set(r["orientation_series_id"] for r in ori)),
    )
    print("  normalized views:", norm_stats)
    report["coverage"] = dict(
        blocks_by_status=dict(counts), rows_by_status=dict(rows_n),
        ysum_by_status={k: round(v, 3) for k, v in ytot.items()},
        normalized=norm_stats)

    # coverage bar chart
    labels = ["exact", "aggregate", "image", "empty", "absent"]
    nb = [counts.get(l, 0) for l in labels]
    nr = [rows_n.get(l, 0) for l in labels]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), dpi=130)
    b = ax[0].bar(labels, nb, color=["C2", "C0", "C1", "C7", "C3"])
    ax[0].bar_label(b, fontsize=9)
    ax[0].set_title("workbook blocks (n=101) by representation in the normalized views")
    ax[0].set_ylabel("blocks")
    b = ax[1].bar(labels, nr, color=["C2", "C0", "C1", "C7", "C3"])
    ax[1].bar_label(b, fontsize=9)
    ax[1].set_yscale("symlog", linthresh=100)
    ax[1].set_title("table rows carried by each class (symlog)")
    ax[1].set_ylabel("rows")
    fig.suptitle("covx: authoritative coverage boundary of the bundled workbook vs the normalized views")
    fig.tight_layout()
    p = FIG / "covx_coverage_summary.png"
    fig.savefig(p)
    plt.close(fig)
    fig_manifest.append(str(p))
    print(f"  wrote {p}")

    report["figures"] = fig_manifest
    with open(OUT / "covx_summary.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"wrote {OUT/'covx_summary.json'}")


if __name__ == "__main__":
    main()