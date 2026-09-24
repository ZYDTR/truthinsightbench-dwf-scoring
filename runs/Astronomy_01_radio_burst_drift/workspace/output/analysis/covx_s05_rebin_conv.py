#!/usr/bin/env python3
"""covx_s05 (dimension: coverage boundary) - can ANY rebinning convention make
the coarse histograms (Sheet_039..044) reproduce the fine blocks?

Sweeps, for all 18 coarse curves:
  - 5 bin-edge conventions (centre +/-w/2, left edge, right edge, left edge with
    the data shifted by +/-w)
  - a shift of the fine curve by -20..+20 coarse bins (in units of 0.875 MHz/s)
  - a multiplicative rescale to a common total (shape-only comparison)
and reports the best achievable correlation and L1 distance.

Also tests the image sheets against explicit model families:
  - Sheet_x == event_i - event_j  (correlation of the signed image with every
    one of the 20 pairwise differences of the 5 normalized events)
  - the same for the positive images
  - Sheet_xy == |event_i - event_j|

Outputs: covx_* csv/json.
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
DATA = ROOT / "data" / "normalized_views"


def blocks():
    blk = collections.defaultdict(list)
    for r in csv.DictReader(open(OUT / "covx_absent_blocks.csv")):
        if r["kind"] not in ("point", "label"):
            continue
        if r["x"] in ("", "None") or r["y"] in ("", "None"):
            continue
        blk[(r["sheet"], r["label"])].append((float(r["x"]), float(r["y"])))
    return {k: np.asarray(v, float) for k, v in blk.items()}


def main():
    blk = blocks()
    rep = {}
    coarse_sheet = {39: ("Sheet_025", "Sheet_024"), 40: ("Sheet_027", "Sheet_026"),
                    41: ("Sheet_029", "Sheet_028"), 42: ("Sheet_031", "Sheet_030"),
                    43: ("Sheet_033", "Sheet_032"), 44: ("Sheet_035", "Sheet_034")}
    norm = lambda s: s.replace(" ", "")

    # ----------------------------------------------------------- rebin sweep
    print("=" * 78)
    print("COARSE vs FINE rebinning sweep (18 pairs, shifts in 0.875 MHz/s units)")
    rows = []
    for sh, (fast_sh, slow_sh) in coarse_sheet.items():
        cs = f"Sheet_{sh:03d}"
        for cl in [k for k in blk if k[0] == cs]:
            cand = [k for k in blk if k[0] in (fast_sh, slow_sh) and norm(k[1]) == norm(cl[1])]
            if not cand:
                continue
            ca, fa = blk[cl], blk[cand[0]]
            cw = 1.75
            best = dict(corr=-2.0, conv=None, shift=None)
            for conv, offset in (("centre", 0.0), ("left", -cw / 2), ("right", +cw / 2)):
                for ds in range(-20, 21):
                    s = ds * 0.875
                    red = np.zeros(len(ca))
                    for (xf, yf) in fa:
                        xx = xf + s
                        k = int(round((xx - offset) / cw))
                        if 0 <= k < len(ca):
                            red[k] += yf
                    if red.sum() <= 0:
                        continue
                    pr, pc = red / red.sum(), ca[:, 1] / ca[:, 1].sum()
                    if pr.std() == 0 or pc.std() == 0:
                        continue
                    r = float(np.corrcoef(pr, pc)[0, 1])
                    if r > best["corr"]:
                        best = dict(corr=r, conv=conv, shift=s,
                                    l1=float(np.abs(pr - pc).sum()),
                                    total_fine_in_range=float(red.sum()),
                                    frac_fine_in_range=float(red.sum() / fa[:, 1].sum()))
            # zero-shift numbers for reference
            red0 = np.zeros(len(ca))
            for (xf, yf) in fa:
                k = int(round(xf / cw))
                if 0 <= k < len(ca):
                    red0[k] += yf
            p0r, p0c = red0 / max(red0.sum(), 1e-9), ca[:, 1] / ca[:, 1].sum()
            r0 = float(np.corrcoef(p0r, p0c)[0, 1]) if p0r.std() and p0c.std() else float("nan")
            rows.append(dict(coarse_sheet=cs, label=cl[1], fine_sheet=cand[0][0],
                             coarse_total=float(ca[:, 1].sum()), fine_total=float(fa[:, 1].sum()),
                             total_ratio=float(ca[:, 1].sum() / fa[:, 1].sum()),
                             corr_zero_shift=r0,
                             best_corr=best["corr"], best_convention=best["conv"],
                             best_shift_MHzs=best["shift"], best_l1=best.get("l1"),
                             frac_fine_inside_window=best.get("frac_fine_in_range")))
    for d in rows:
        print(f"  {d['coarse_sheet']} {d['label']:32s} ratio={d['total_ratio']:.4f} "
              f"r(0-shift)={d['corr_zero_shift']:+.3f} -> best r={d['best_corr']:+.3f} "
              f"at {d['best_convention']} shift={d['best_shift_MHzs']:+.3f} MHz/s "
              f"(fine frac inside window {d['frac_fine_inside_window']:.3f})")
    rep["rebin_sweep"] = rows
    with open(OUT / "covx_coarse_rebin_sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ------------------------------------------------- image model tests
    print("=" * 78)
    print("IMAGE SHEETS vs explicit model families")
    z = np.load(OUT / "s14_images.npz")
    extra = ["Sheet_008", "Sheet_015", "Sheet_016", "Sheet_017", "Sheet_018",
             "Sheet_021", "Sheet_022", "Sheet_036", "Sheet_037"]
    ev = {}
    for i in range(1, 6):
        arr = np.genfromtxt(DATA / f"event_{i}_dynamic_spectrum.tsv", delimiter="\t", skip_header=1)
        ev[f"event_{i}"] = arr[:, 1:]
    diffs, sums, absd = {}, {}, {}
    for a, b in itertools.permutations(ev, 2):
        diffs[f"{a}-{b}"] = ev[a] - ev[b]
        absd[f"|{a}-{b}|"] = np.abs(ev[a] - ev[b])
    sums["sum_of_5"] = np.sum(list(ev.values()), axis=0)
    rowsI = []
    for s in extra:
        G = z["grid__" + s]
        g = G[::5, ::5].ravel()
        best = ("none", 0.0)
        for name, D in list(diffs.items()) + list(sums.items()) + list(absd.items()):
            d = D[::5, ::5].ravel()
            if d.std() == 0:
                continue
            r = float(np.corrcoef(g, d)[0, 1])
            if abs(r) > abs(best[1]) or best[0] == "none":
                best = (name, r)
        # best against a single event
        beste = ("none", 0.0)
        for name, E in ev.items():
            e = E[::5, ::5].ravel()
            r = float(np.corrcoef(g, e)[0, 1])
            if abs(r) > abs(beste[1]) or beste[0] == "none":
                beste = (name, r)
        rowsI.append(dict(sheet=s, best_event=beste[0], r_best_event=beste[1],
                          best_combination=best[0], r_best_combination=best[1],
                          vmin=float(G.min()), vmax=float(G.max()), mean=float(G.mean()),
                          std=float(G.std())))
        print(f"  {s}: best single event {beste[0]} r={beste[1]:+.4f} | "
              f"best combination {best[0]} r={best[1]:+.4f} | "
              f"range[{rowsI[-1]['vmin']:.3g},{rowsI[-1]['vmax']:.3g}] mean={rowsI[-1]['mean']:+.4f}")
    rep["image_model_tests"] = rowsI
    with open(OUT / "covx_image_model_tests.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rowsI[0].keys()))
        w.writeheader()
        w.writerows(rowsI)

    # ------------------------------ are the 9 unrepresented images independent
    #  of each other?  full pairwise |r| table, subsampled
    print("  pairwise |r| among the 9 unrepresented image sheets (subsampled 5x5):")
    subs = {s: z["grid__" + s][::5, ::5].ravel() for s in extra}
    rowsP = []
    for a, b in itertools.combinations(extra, 2):
        r = float(np.corrcoef(subs[a], subs[b])[0, 1])
        same_bounds = None
        rowsP.append(dict(a=a, b=b, pearson=r, mean_abs=float(np.abs(subs[a] - subs[b]).mean())))
    for d in sorted(rowsP, key=lambda d: -abs(d["pearson"]))[:8]:
        print(f"    {d['a']} {d['b']} r={d['pearson']:+.4f} mae={d['mean_abs']:.4f}")
    rep["image_pairwise_extra"] = rowsP
    with open(OUT / "covx_image_pairwise_extra.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rowsP[0].keys()))
        w.writeheader()
        w.writerows(rowsP)

    with open(OUT / "covx_rebin_and_models.json", "w") as fh:
        json.dump(rep, fh, indent=2, default=str)
    print(f"wrote {OUT/'covx_rebin_and_models.json'}")


if __name__ == "__main__":
    main()