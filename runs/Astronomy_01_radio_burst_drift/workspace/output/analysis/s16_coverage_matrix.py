#!/usr/bin/env python3
"""S16 (dimension: provenance & coverage) - per-BLOCK coverage matrix + boundary.

Refines s15 in three ways:
  1. classifies every workbook block by the NORMALIZED SERIES it feeds, not only
     by 1:1 reproduction - the six orientation sheets (Sheet_009..014) are three
     blocks each that together form one normalized series, so their blocks are
     'aggregate' (52 of the 55 points are kept; the first point of each block is
     dropped), not 'absent';
  2. tests every normalized series against (a) its own reconstructed workbook
     union and (b) the coarse 1.75 MHz/s sheets Sheet_039..044, reporting totals;
  3. writes the coverage boundary: which workbook blocks feed no normalized view
     at all, with size and total counts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "output" / "results"

# (sheet -> (series_id, status, note))
ORIENT = {"Sheet_009": "orientation_series_1", "Sheet_010": "orientation_series_2",
          "Sheet_011": "orientation_series_3", "Sheet_012": "orientation_series_4",
          "Sheet_013": "orientation_series_5", "Sheet_014": "orientation_series_6"}
FINE = {"Sheet_024": "population_series_1", "Sheet_026": "population_series_2",
        "Sheet_028": "population_series_3", "Sheet_030": "population_series_4",
        "Sheet_032": "population_series_5", "Sheet_034": "population_series_6"}
FAST = {"Sheet_025": "population_series_1", "Sheet_027": "population_series_2",
        "Sheet_029": "population_series_3", "Sheet_031": "population_series_4",
        "Sheet_033": "population_series_5", "Sheet_035": "population_series_6"}
COARSE = [f"Sheet_{i:03d}" for i in range(39, 45)]
IMAGES = [f"Sheet_{i:03d}" for i in
          [3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 21, 22, 36, 37]]


def main() -> None:
    idx = pd.read_csv(OUT / "s11_block_index.csv")
    blocks = pd.read_csv(OUT / "s11_blocks.csv")
    pop = pd.read_csv(DATA / "normalized_views" / "drift_population_series.tsv", sep="\t")
    ori = pd.read_csv(DATA / "normalized_views" / "orientation_count_series.tsv", sep="\t")

    pop_tot = pop.groupby(["population_series_id", "drift_regime"])["event_count"].sum()
    ori_tot = ori.groupby("orientation_series_id")["event_count"].sum()

    rows = []
    for _, r in idx.iterrows():
        sh, lab, cls = r["sheet"], r["block_label"], r["class_"]
        if cls == "image":
            rows.append(dict(sheet=sh, block_label="Image", block_class="image",
                             n=int(r["n"]), y_sum=r["y_sum"], status="image",
                             normalized_counterpart="", note="dynamic-spectrum pixel grid"))
            continue
        if cls == "empty":
            rows.append(dict(sheet=sh, block_label="<none>", block_class="empty", n=0,
                             y_sum=0.0, status="empty", normalized_counterpart="",
                             note="sheet has a single blank cell"))
            continue
        ysum = r["y_sum"]
        series = ""
        status = "absent"
        note = ""
        if sh in FAST:
            pid = FAST[sh]
            series = f"{pid}|fast"
            status = "exact"
            note = "identical (x,y) to the normalized fast series"
        elif sh in FINE:
            pid = FINE[sh]
            series = f"{pid}|slow"
            status = "aggregate"
            note = ("52/55 points kept; the other block of the same sheet supplies "
                    "the other substride; union == normalized slow series exactly")
        elif sh in ORIENT:
            oid = ORIENT[sh]
            series = oid
            status = "aggregate"
            note = ("3 blocks of 25/10/20 x 55 points; normalized series = the three "
                    "blocks concatenated with the FIRST point of each block dropped (52 points)")
        elif sh in ("Sheet_046", "Sheet_047"):
            status = "absent"
            note = "third orientation block family, no normalized counterpart"
        elif sh in COARSE:
            status = "absent"
            note = "1.75 MHz/s coarse re-binning; totals differ slightly from every normalized series"
        elif sh in ("Sheet_019", "Sheet_020", "Sheet_023", "Sheet_038"):
            status = "absent"
            note = "radial-distance / fit / SNR block"
        elif sh in ("Sheet_001", "Sheet_002"):
            status = "absent"
            note = "CML-angle vs phase-angle path samples"
        rows.append(dict(sheet=sh, block_label=lab, block_class=cls, n=int(r["n"]),
                         y_sum=float(ysum), status=status,
                         normalized_counterpart=series, note=note))

    cov = pd.DataFrame(rows).sort_values(["sheet", "block_label"]).reset_index(drop=True)
    cov.to_csv(OUT / "s16_coverage_block.csv", index=False)
    print("=== coverage status counts (per workbook block) ===")
    print(cov["status"].value_counts().to_string())
    print("\n=== status x sheet ===")
    print(pd.crosstab(cov.sheet, cov.status).to_string())

    # ---- normalized series -> workbook provenance, with totals ----
    prov = []
    for (pid, reg), g in pop.groupby(["population_series_id", "drift_regime"]):
        if reg == "fast":
            sh = [k for k, v in FAST.items() if v == pid][0]
            src = f"{sh} (all {len(pop[(pop.population_series_id == pid) & (pop.drift_regime == 'fast')])} points)"
        else:
            sh = [k for k, v in FINE.items() if v == pid][0]
            src = f"{sh} (union of its two blocks)"
        prov.append(dict(series=f"{pid}|{reg}", workbook_source=src,
                         n=len(g), events=float(g.event_count.sum())))
    for oid, g in ori.groupby("orientation_series_id"):
        sh = [k for k, v in ORIENT.items() if v == oid][0]
        prov.append(dict(series=oid, workbook_source=f"{sh} (3 blocks, first point dropped)",
                         n=len(g), events=float(g.event_count.sum())))
    provdf = pd.DataFrame(prov)
    provdf.to_csv(OUT / "s16_normalized_provenance.csv", index=False)
    print("\n=== normalized series provenance ===")
    print(provdf.to_string(index=False))

    # ---- coverage boundary: blocks with no normalized counterpart ----
    absent = cov[cov.status == "absent"]
    print(f"\n=== COVERAGE BOUNDARY: {len(absent)} blocks with no normalized counterpart ===")
    for sh, g in absent.groupby("sheet"):
        print(f"  {sh}: blocks={len(g)} rows={int(g['n'].sum())} "
              f"total_counts={g['y_sum'].sum():.0f}  [{g['block_class'].iloc[0]}]")
    print(f"\n  TOTAL absent: blocks={len(absent)} rows={int(absent['n'].sum())} "
          f"counts={absent['y_sum'].sum():.0f}")

    summary = dict(
        workbook_sheets=47,
        workbook_blocks_total=len(cov),
        blocks_exact=int((cov.status == "exact").sum()),
        blocks_aggregate=int((cov.status == "aggregate").sum()),
        blocks_image=int((cov.status == "image").sum()),
        blocks_empty=int((cov.status == "empty").sum()),
        blocks_absent=int((cov.status == "absent").sum()),
        absent_rows=int(absent["n"].sum()),
        absent_counts=float(absent["y_sum"].sum()),
        normalized_population_events=float(pop.event_count.sum()),
        normalized_orientation_events=float(ori.event_count.sum()),
        absent_by_family={k: dict(blocks=int(len(g)), rows=int(g["n"].sum()),
                                  counts=float(g["y_sum"].sum()))
                          for k, g in absent.groupby("sheet")},
    )
    (OUT / "s16_coverage_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nwrote s16_coverage_block.csv / s16_normalized_provenance.csv / s16_coverage_summary.json")


if __name__ == "__main__":
    main()