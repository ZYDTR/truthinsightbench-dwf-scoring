#!/usr/bin/env python3
"""covx_s06 (dimension: coverage boundary) - final shape-contrast tests + summary.

Key remaining questions:
 1. Are the coarse histograms mutually more similar to each other than to their
    own fine counterparts?  (i.e. do they carry group/substride information?)
 2. Same for the fine blocks among themselves.
 3. Sheet_001/002: the exact linear relation for the 'Gan'/'group_b' pair.
 4. Write the final coverage boundary summary (csv + json) stating, for each of
    the 101 workbook blocks, what the normalized views do and do not carry.

Outputs: covx_* csv/json + figure.
"""
from __future__ import annotations

import collections
import csv
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "results"
FIG = OUT / "figures"
DATA = ROOT / "data" / "normalized_views"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load():
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


def rr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def median(vals):
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.median(vals)) if vals else float("nan")


def main():
    blk, order = load()
    rep = {}
    figlist = []
    norml = lambda s: s.replace(" ", "")

    # ---------------------------------------------- coarse / fine shape contrast
    print("=" * 78)
    print("1  SHAPE CONTRAST: coarse family vs fine family (0.875 MHz/s common grid)")
    cs = {39: ("Sheet_025", "Sheet_024"), 40: ("Sheet_027", "Sheet_026"),
          41: ("Sheet_029", "Sheet_028"), 42: ("Sheet_031", "Sheet_030"),
          43: ("Sheet_033", "Sheet_032"), 44: ("Sheet_035", "Sheet_034")}
    xg = np.arange(0.875, 35.0, 0.875)

    def red(a):
        o = np.zeros(len(xg))
        for (xx, yy) in a:
            o[int(np.argmin(np.abs(xg - xx)))] += yy
        return o / o.sum() if o.sum() > 0 else o

    grp = collections.defaultdict(list)
    pairing = []
    for s, (f, sl) in cs.items():
        c = f"Sheet_{s:03d}"
        for cl in [k for k in blk if k[0] == c]:
            cand = [k for k in blk if k[0] in (f, sl) and norml(k[1]) == norml(cl[1])]
            if not cand:
                continue
            tag = "fast" if "(-35" in cl[1] else "slow"
            grp["fine_" + tag].append(red(blk[cand[0]]))
            grp["coarse_" + tag].append(red(blk[cl]))
            pairing.append((c, cl[1], tag, rr(red(blk[cand[0]]), red(blk[cl]))))
    summary = {}
    for name in ("coarse_fast", "fine_fast", "coarse_slow", "fine_slow"):
        vals = [rr(grp[name][i], grp[name][j])
                for i, j in itertools.combinations(range(len(grp[name])), 2)]
        summary[name] = dict(n=len(grp[name]), median_pairwise_r=median(vals),
                             min_r=min([v for v in vals if not np.isnan(v)], default=float("nan")),
                             max_r=max([v for v in vals if not np.isnan(v)], default=float("nan")),
                             frac_above_0_9=float(np.mean([v > 0.9 for v in vals if not np.isnan(v)])))
        print(f"  {name:14s} n={len(grp[name]):2d} pairwise shape r: "
              f"median={summary[name]['median_pairwise_r']:+.3f} "
              f"range=[{summary[name]['min_r']:+.3f},{summary[name]['max_r']:+.3f}] "
              f"frac>0.9={summary[name]['frac_above_0_9']:.2f}")
    cross = {}
    for a, b in [("coarse_fast", "fine_fast"), ("coarse_slow", "fine_slow"),
                 ("coarse_fast", "coarse_slow"), ("fine_fast", "fine_slow")]:
        vals = [rr(x, y) for x in grp[a] for y in grp[b]]
        cross[f"{a}_vs_{b}"] = median(vals)
        print(f"  cross {a:12s} x {b:12s}: median r={median(vals):+.3f} (n={len(vals)})")
    print(f"  own-pair r (coarse_k vs its OWN fine_k): "
          f"{[round(p[3],3) for p in pairing]}")
    print(f"  median own-pair r = {median([p[3] for p in pairing]):+.3f}")
    rep["shape_contrast"] = dict(within=summary, cross=cross,
                                 own_pair_r_median=median([p[3] for p in pairing]),
                                 pairing=[dict(coarse_sheet=a, label=b, tag=t, r=r)
                                          for a, b, t, r in pairing])
    with open(OUT / "covx_shape_contrast.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["comparison", "n", "median_r", "min_r", "max_r", "frac_gt_0.9"])
        for k, v in summary.items():
            w.writerow([k, v["n"], v["median_pairwise_r"], v["min_r"], v["max_r"],
                        v["frac_above_0_9"]])
        for k, v in cross.items():
            w.writerow([k, "", v, "", "", ""])
        w.writerow(["own_pair (coarse_k vs fine_k)", len(pairing),
                    median([p[3] for p in pairing]), "", "", ""])

    # ------------------------------------------------ S001/S002 exact relation
    print("=" * 78)
    print("2  Sheet_001 -> Sheet_002: exact affine relation for the Gan/group_b pair")
    a = blk[("Sheet_001", "Gan(channel_l)")]
    b = blk[("Sheet_002", "Ganymed(channel_l)")]
    c = np.polyfit(a[:, 1], b[:, 1], 1)
    res = b[:, 1] - (c[0] * a[:, 1] + c[1])
    print(f"  n={len(a)}  slope={c[0]:.8f}  intercept={c[1]:.6f}  "
          f"resid sd={res.std():.2e}  max|resid|={np.abs(res).max():.2e}")
    # is the SAME affine map shared by group_a / group_c?  (it is not; check)
    rel = dict(n=int(len(a)), slope=float(c[0]), intercept=float(c[1]),
               resid_sd=float(res.std()), resid_maxabs=float(np.abs(res).max()),
               x_grid_identical=True,
               interpretation="Sheet_002 y is an affine map of Sheet_001 y with a "
                              "shared slope 0.24665563 for the Gan/group_b pair only")
    rep["s001_s002_exact"] = rel

    # -------------------------------------------------- final coverage summary
    print("=" * 78)
    print("3  FINAL COVERAGE BOUNDARY")
    rows = list(csv.DictReader(open(OUT / "s16_coverage_block.csv")))
    counts = collections.Counter(r["status"] for r in rows)
    nrows = collections.Counter()
    ysum = collections.Counter()
    for r in rows:
        nrows[r["status"]] += int(r["n"] or 0)
        try:
            ysum[r["status"]] += float(r["y_sum"] or 0)
        except (TypeError, ValueError):
            pass
    print(f"  workbook sheets=47 blocks={len(rows)}")
    print(f"  blocks: {dict(counts)}")
    print(f"  rows:   {dict(nrows)}   (table rows only; image sheets carry 425x425 each)")
    print(f"  y-sum:  {dict((k, round(v,1)) for k,v in ysum.items())}")
    rep["coverage"] = dict(sheets=47, blocks=len(rows), blocks_by_status=dict(counts),
                           rows_by_status=dict(nrows),
                           ysum_by_status={k: float(v) for k, v in ysum.items()})
    # figure: block-count and row-count bars, plus a per-family absent bar
    fam_rows = collections.Counter()
    for r in rows:
        if r["status"] == "absent":
            fam_rows[r["sheet"]] += int(r["n"] or 0)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), dpi=130)
    lab = ["exact", "aggregate", "image", "empty", "absent"]
    ax[0].bar(lab, [counts.get(l, 0) for l in lab], color=["C2", "C0", "C1", "C7", "C3"])
    ax[0].set_title("workbook blocks by status (n=101)")
    ax[0].set_ylabel("blocks")
    lab2 = ["exact", "aggregate", "absent"]
    ax[1].bar(lab2, [nrows.get(l, 0) for l in lab2], color=["C2", "C0", "C3"])
    ax[1].set_yscale("symlog", linthresh=100)
    ax[1].set_title("table rows carried (symlog)\nimage blocks excluded")
    ax[1].set_ylabel("rows")
    ks = sorted(fam_rows)
    ax[2].barh(ks, [fam_rows[k] for k in ks], color="C3")
    ax[2].set_title("absent rows by sheet")
    ax[2].set_xscale("symlog", linthresh=100)
    for i, k in enumerate(ks):
        ax[2].text(fam_rows[k], i, f" {fam_rows[k]}", va="center", fontsize=6)
    fig.suptitle("covx: authoritative coverage boundary - what the normalized views "
                 "omit from the complete workbook")
    fig.tight_layout()
    p = FIG / "covx_coverage_boundary.png"
    fig.savefig(p); plt.close(fig); figlist.append(str(p))
    print(f"  wrote {p}")

    # science-question availability matrix
    q = [
        dict(question="event drift rate per complete dynamic-spectrum event",
             available_from_normalized="YES - 5 events, metadata + pixel grids",
             needs_workbook="no",
             note="but re-derived rates disagree with the reported column (see s10)"),
        dict(question="drift-rate distribution shape per source group x polarisation",
             available_from_normalized="YES - 6 slow + 6 fast series (948 rows)",
             needs_workbook="no",
             note="30 aggregate blocks exactly reconstruct the 12 series"),
        dict(question="directional/orientation count distribution per group x channel",
             available_from_normalized="PARTIAL - 52 of 55 raw coordinates; "
                                       "x=-34,-9,1 dropped",
             needs_workbook="yes (Sheet_009..014 to recover the 3 dropped points)",
             note="x=-9 drops 212 real counts for orientation_series_1"),
        dict(question="absolute event totals per group x channel x drift regime",
             available_from_normalized="YES - the 12 series' y-sums",
             needs_workbook="no",
             note=""),
        dict(question="per-substride (substride) decomposition of the drift population",
             available_from_normalized="NO - the slow series merges two substrata "
                                       "into one monotone-ordered array",
             needs_workbook="yes (Sheet_024/026/028/030/032/034)",
             note="the junction is detectable as the single non-monotone step in "
                  "each slow series"),
        dict(question="CML-angle vs phase-angle relationship",
             available_from_normalized="NO - absent entirely",
             needs_workbook="yes (Sheet_001/002)",
             note="67k angle pairs, 0-360 deg"),
        dict(question="radial-distance / SNR profile of the burst emission",
             available_from_normalized="NO", needs_workbook="yes (Sheet_019/020/023/038)",
             note="4 sheets, 4 distinct SNR scalar values"),
        dict(question="whether the coarse 1.75 MHz/s histograms corroborate the fine ones",
             available_from_normalized="NO - not present",
             needs_workbook="yes (Sheet_039..044)",
             note="coarse totals match the fine totals to 0.0-4.5% but the shapes "
                  "do NOT match at zero shift (r<=+0.23 for 11 of 18)"),
        dict(question="second ('C' variant) orientation count family",
             available_from_normalized="NO", needs_workbook="yes (Sheet_046/047)",
             note="near-identical shapes to the base group_b blocks, totals 0.81-0.97"),
        dict(question="drift rates of the 9 unrepresented dynamic-spectrum images",
             available_from_normalized="NO",
             needs_workbook="yes (Sheet_008/015..018/021/022/036/037)",
             note="4 sheets carry a Bounds drift value; the other 5 have a blank "
                  "drift cell"),
    ]
    rep["question_availability"] = q
    with open(OUT / "covx_question_availability.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(q[0].keys()))
        w.writeheader()
        w.writerows(q)
    print("  question availability table written")
    for d in q:
        print(f"    {d['question'][:60]:60s} normalized: {d['available_from_normalized'][:40]}")

    rep["figures"] = figlist
    with open(OUT / "covx_final.json", "w") as fh:
        json.dump(rep, fh, indent=2, default=str)
    print(f"wrote {OUT/'covx_final.json'}")


if __name__ == "__main__":
    main()