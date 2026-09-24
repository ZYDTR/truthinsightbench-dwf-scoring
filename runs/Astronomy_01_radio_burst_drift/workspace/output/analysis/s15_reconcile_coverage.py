#!/usr/bin/env python3
"""S15 (dimension: provenance & coverage) - reconciliation + coverage boundary.

Uses the tidy workbook parse from s11 (s11_blocks.csv / s11_block_index.csv) and
the normalized views to answer, for every workbook block, whether it is

  exact       -> reproduced identically by one normalized series,
  aggregate    -> reproduced by concatenating several workbook blocks,
  derivative   -> an image sheet that duplicates another image's bounds,
  absent       -> no normalized counterpart,

and produces the machine-readable coverage table.  Every comparison is
elementwise on (x, y); the tolerance is 0 (exact) with a printed residual.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "output" / "results"

IMAGE_SHEETS = {f"Sheet_{i:03d}" for i in
                [3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 21, 22, 36, 37]}
FINE = {"Sheet_024", "Sheet_026", "Sheet_028", "Sheet_030", "Sheet_032", "Sheet_034"}
FAST = {"Sheet_025", "Sheet_027", "Sheet_029", "Sheet_031", "Sheet_033", "Sheet_035"}
ORIENT = {f"Sheet_{i:03d}" for i in range(9, 15)}
RDP = {"Sheet_019", "Sheet_020", "Sheet_023", "Sheet_038"}
COARSE = {f"Sheet_{i:03d}" for i in range(39, 45)}
SHEET46 = {"Sheet_046", "Sheet_047"}
ANGLE = {"Sheet_001", "Sheet_002"}


def load_blocks():
    df = pd.read_csv(OUT / "s11_blocks.csv")
    blocks = {}
    for (sh, lab), g in df.groupby(["sheet", "block_label"], sort=False):
        blocks.setdefault(sh, []).append((lab, g["x"].to_numpy(float), g["y"].to_numpy(float)))
    return blocks


def best_match(x, y, refs, xtol=1e-9):
    """Closest elementwise match of (x,y) against a dict of reference series."""
    best = None
    for key, (rx, ry) in refs.items():
        if len(rx) != len(x):
            continue
        if np.max(np.abs(rx - x)) > xtol:
            continue
        d = float(np.max(np.abs(ry - y)))
        if best is None or d < best[1]:
            best = (key, d)
    return best


def main() -> None:
    blocks = load_blocks()
    pop = pd.read_csv(DATA / "normalized_views" / "drift_population_series.tsv", sep="\t")
    ori = pd.read_csv(DATA / "normalized_views" / "orientation_count_series.tsv", sep="\t")

    pop_refs = {}
    for (pid, reg), g in pop.groupby(["population_series_id", "drift_regime"]):
        g = g.sort_values("distribution_bin_center")
        pop_refs[f"{pid}|{reg}"] = (g["distribution_bin_center"].to_numpy(float),
                                    g["event_count"].to_numpy(float))
    ori_refs = {}
    for oid, g in ori.groupby("orientation_series_id"):
        g = g.sort_values("relative_coordinate")
        ori_refs[oid] = (g["relative_coordinate"].to_numpy(float),
                         g["event_count"].to_numpy(float))

    # ------------- 1. per-block exact match against every normalized series -------------
    recs = []
    exact_refs = dict(pop_refs)
    exact_refs.update({("ori|" + k): v for k, v in ori_refs.items()})
    for sh in sorted(blocks):
        for lab, x, y in blocks[sh]:
            m = best_match(x, y, exact_refs)
            recs.append(dict(sheet=sh, block_label=lab, n=len(x),
                             x0=float(x[0]), x1=float(x[-1]),
                             y_sum=float(np.nansum(y)),
                             exact_match=(m[0] if m and m[1] == 0.0 else ""),
                             closest_match=(m[0] if m else ""),
                             closest_maxdiff=(m[1] if m else np.nan)))
    rec = pd.DataFrame(recs)
    print("=== exact block -> normalized-series matches ===")
    print(rec[rec.exact_match != ""][["sheet", "block_label", "n", "y_sum", "exact_match"]]
          .to_string(index=False))

    # ------------- 2. AGGREGATE reconstructions -------------
    print("\n=== aggregate reconstruction: 'slow' = union of two fine blocks ===")
    slow_map = {"population_series_1": ("Sheet_024", 0, 1),
                "population_series_2": ("Sheet_026", 0, 1),
                "population_series_3": ("Sheet_028", 0, 1),
                "population_series_4": ("Sheet_030", 0, 1),
                "population_series_5": ("Sheet_032", 0, 1),
                "population_series_6": ("Sheet_034", 0, 1)}
    recon_rows = []
    for pid, (sh, i, j) in slow_map.items():
        x1, y1 = blocks[sh][i][1], blocks[sh][i][2]
        x2, y2 = blocks[sh][j][1], blocks[sh][j][2]
        X = np.concatenate([x1, x2]); Y = np.concatenate([y1, y2])
        o = np.argsort(X); X, Y = X[o], Y[o]
        rx, ry = pop_refs[f"{pid}|slow"]
        dx = float(np.max(np.abs(X - rx))); dy = float(np.max(np.abs(Y - ry)))
        recon_rows.append(dict(population_series_id=pid, workbook_blocks=2, sheet=sh,
                               labels=" + ".join([blocks[sh][i][0], blocks[sh][j][0]]),
                               n_recon=len(X), n_norm=len(rx),
                               sum_recon=float(Y.sum()), sum_norm=float(ry.sum()),
                               max_abs_dx=dx, max_abs_dy=dy,
                               exact=bool(len(X) == len(rx) and dx < 1e-9 and dy < 1e-9)))
        print(rec_rows := recon_rows[-1])
    pd.DataFrame(recon_rows).to_csv(OUT / "s15_slow_reconstruction.csv", index=False)

    print("\n=== aggregate reconstruction: orientation = 3 blocks, first point of each dropped ===")
    ori_map = {"orientation_series_1": "Sheet_009", "orientation_series_2": "Sheet_010",
               "orientation_series_3": "Sheet_011", "orientation_series_4": "Sheet_012",
               "orientation_series_5": "Sheet_013", "orientation_series_6": "Sheet_014"}
    ori_rows = []
    for oid, sh in ori_map.items():
        parts = [(b[1][1:], b[2][1:]) for b in blocks[sh]]
        X = np.concatenate([p[0] for p in parts]); Y = np.concatenate([p[1] for p in parts])
        o = np.argsort(X); X, Y = X[o], Y[o]
        rx, ry = ori_refs[oid]
        dropped = [(b[0], float(b[1][0]), float(b[2][0])) for b in blocks[sh]]
        dx = float(np.max(np.abs(X - rx))); dy = float(np.max(np.abs(Y - ry)))
        naive = np.concatenate([b[1] for b in blocks[sh]])
        ori_rows.append(dict(orientation_series_id=oid, sheet=sh,
                             n_concat_kept=len(X), n_norm=len(rx),
                             sum_recon=float(Y.sum()), sum_norm=float(ry.sum()),
                             sum_all_55=float(np.nansum(naive)),
                             max_abs_dx=dx, max_abs_dy=dy,
                             exact=bool(len(X) == len(rx) and dx < 1e-9 and dy < 1e-9),
                             dropped_points=str(dropped)))
        print(ori_rows[-1])
    pd.DataFrame(ori_rows).to_csv(OUT / "s15_orientation_reconstruction.csv", index=False)

    # ------------- 3. coverage classification -------------
    slow_sheets = {sh for sh, _, _ in slow_map.values()}
    cov = []
    for r in recs:
        sh, lab = r["sheet"], r["block_label"]
        if r["exact_match"]:
            status = "exact"
            prov = r["exact_match"]
        elif sh in slow_sheets:
            status = "aggregate"
            prov = [k for k, v in slow_map.items() if v[0] == sh][0] + "|slow (union with the other block of the same sheet)"
        elif sh in IMAGE_SHEETS:
            status = "image"
            prov = ""
        else:
            status = "absent"
            prov = ""
        cov.append(dict(sheet=sh, block_label=lab, n=r["n"], x0=r["x0"], x1=r["x1"],
                        y_sum=r["y_sum"], status=status, normalized_counterpart=prov,
                        closest_normalized_series=r["closest_match"],
                        closest_maxdiff=r["closest_maxdiff"]))
    # image sheets + empty sheet are not (x,y) blocks
    idx = pd.read_csv(OUT / "s11_block_index.csv")
    for _, row in idx.iterrows():
        if row["class_"] == "image":
            cov.append(dict(sheet=row["sheet"], block_label="Image",
                            n=int(row["n"]), x0=np.nan, x1=np.nan, y_sum=row["y_sum"],
                            status="image", normalized_counterpart="",
                            closest_normalized_series="", closest_maxdiff=np.nan))
        elif row["class_"] == "empty":
            cov.append(dict(sheet=row["sheet"], block_label="<none>", n=0, x0=np.nan,
                            x1=np.nan, y_sum=0.0, status="empty",
                            normalized_counterpart="", closest_normalized_series="",
                            closest_maxdiff=np.nan))
    covdf = pd.DataFrame(cov).sort_values(["sheet", "block_label"]).reset_index(drop=True)
    covdf.to_csv(OUT / "s15_coverage_table.csv", index=False)
    print("\n=== coverage status counts ===")
    print(covdf["status"].value_counts().to_string())
    print("\n=== ABSENT blocks (no normalized counterpart) ===")
    print(covdf[covdf.status == "absent"]
          [["sheet", "block_label", "n", "y_sum", "closest_normalized_series", "closest_maxdiff"]]
          .to_string(index=False))
    print(f"\nabsent blocks: {int((covdf.status == 'absent').sum())}, "
          f"total events in absent blocks: {covdf[covdf.status == 'absent']['y_sum'].sum():.0f}")

    # ------------- 4. event-total accounting -------------
    tot = {}
    tot["normalized_population_events"] = float(pop["event_count"].sum())
    tot["normalized_orientation_events"] = float(ori["event_count"].sum())
    tot["workbook_events_fast_blocks"] = float(rec[rec.sheet.isin(FAST)]["y_sum"].sum())
    tot["workbook_events_fine_blocks"] = float(rec[rec.sheet.isin(FINE)]["y_sum"].sum())
    tot["workbook_events_orient_blocks_all55"] = float(rec[rec.sheet.isin(ORIENT)]["y_sum"].sum())
    tot["workbook_events_coarse_blocks"] = float(rec[rec.sheet.isin(COARSE)]["y_sum"].sum())
    tot["workbook_events_RD_blocks"] = float(rec[rec.sheet.isin(RDP)]["y_sum"].sum())
    tot["workbook_events_46_47"] = float(rec[rec.sheet.isin(SHEET46)]["y_sum"].sum())
    tot["workbook_events_angle_series_001_002"] = float(rec[rec.sheet.isin(ANGLE)]["y_sum"].sum())
    tot["workbook_image_sheets"] = 14
    print("\n=== event-total accounting ===")
    for k, v in tot.items():
        print(f"  {k}: {v:.0f}")
    (OUT / "s15_totals.json").write_text(json.dumps(tot, indent=2) + "\n")

    # per-sheet totals of the fine blocks vs slow series
    print("\n=== fine-block totals vs slow series totals ===")
    for pid, (sh, i, j) in slow_map.items():
        tot_sh = sum(float(b[2].sum()) for b in blocks[sh])
        rx, ry = pop_refs[f"{pid}|slow"]
        print(f"  {sh}: all-block total={tot_sh:.0f} slow series total={ry.sum():.0f} "
              f"equal={tot_sh == ry.sum()}")
    print("\nwrote s15_coverage_table.csv / s15_slow_reconstruction.csv / "
          "s15_orientation_reconstruction.csv / s15_totals.json")


if __name__ == "__main__":
    main()