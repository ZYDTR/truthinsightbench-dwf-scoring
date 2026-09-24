#!/usr/bin/env python3
"""S03 - Exact mapping between the 47 workbook sheets and the normalized views.

Recovers every labelled (block, x, y) triple from the deidentified workbook,
then joins the workbook blocks against `drift_population_series.tsv` and
`orientation_count_series.tsv` to establish which source block feeds which
normalized series, and which workbook blocks are NOT represented in any
normalized view (coverage boundary).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import openpyxl

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "output" / "results"
XLSX = DATA / "source_data" / "all_source_data_deidentified.xlsx"


def extract_blocks(sheet: str, ws) -> list[dict]:
    """Recover [(label, x_array, y_array)] blocks from one workbook sheet."""
    blocks = []
    cur = None
    for row in ws.iter_rows(min_col=2, max_col=4, values_only=True):
        b, x, y = row[0], row[1], row[2]
        if isinstance(b, str) and b.strip() not in ("", "Bounds"):
            cur = {"label": b.strip(), "x": [], "y": []}
            blocks.append(cur)
        if isinstance(b, str) and b.strip() == "Bounds":
            cur = None
            continue
        if cur is not None and x is not None and isinstance(x, (int, float)):
            try:
                cur["x"].append(float(x))
                cur["y"].append(float(y) if y is not None else np.nan)
            except (TypeError, ValueError):
                pass
    for blk in blocks:
        blk["x"] = np.array(blk["x"], float)
        blk["y"] = np.array(blk["y"], float)
    return blocks


def main() -> None:
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    all_blocks = []
    sizes = {}
    for name in wb.sheetnames:
        ws = wb[name]
        sizes[name] = (ws.max_row, ws.max_column)
    # re-open without read_only to get reliable max_column / positions
    wb.close()
    wb = openpyxl.load_workbook(XLSX, read_only=False, data_only=True)
    for name in wb.sheetnames:
        ws = wb[name]
        if ws.max_column <= 100:
            for blk in extract_blocks(name, ws):
                if len(blk["x"]) == 0:
                    continue
                all_blocks.append(
                    {
                        "sheet": name,
                        "label": blk["label"],
                        "n": len(blk["x"]),
                        "x0": float(blk["x"][0]),
                        "x1": float(blk["x"][-1]),
                        "dx": float(np.median(np.diff(blk["x"]))) if len(blk["x"]) > 1 else np.nan,
                        "y_sum": float(np.nansum(blk["y"])),
                        "x": blk["x"],
                        "y": blk["y"],
                    }
                )
    wb.close()

    print(f"non-image sheets recovered blocks: {len(all_blocks)}")
    rows = []
    for i, b in enumerate(all_blocks):
        rows.append({k: v for k, v in b.items() if k not in ("x", "y")})
    bdf = pd.DataFrame(rows)
    print(bdf.to_string())
    bdf.to_csv(OUT / "s03_workbook_blocks.csv", index=False)

    # ---- normalized views ----
    pop = pd.read_csv(DATA / "normalized_views" / "drift_population_series.tsv", sep="\t")
    ori = pd.read_csv(DATA / "normalized_views" / "orientation_count_series.tsv", sep="\t")

    # Try to match each workbook block against each normalized population series
    matches = []
    for pid, g in pop.groupby("population_series_id"):
        for regime, gg in g.groupby("drift_regime"):
            xs = np.sort(gg["distribution_bin_center"].to_numpy(float))
            ys = gg.sort_values("distribution_bin_center")["event_count"].to_numpy(float)
            best = None
            for b in all_blocks:
                if len(b["x"]) != len(xs):
                    continue
                if np.allclose(b["x"], xs, atol=1e-6) and np.allclose(b["y"], ys, atol=1e-6):
                    best = (b["sheet"], b["label"], "exact")
                    break
            if best is None:
                for b in all_blocks:
                    if len(b["x"]) == len(xs) and np.allclose(b["x"], xs, atol=1e-6):
                        best = (b["sheet"], b["label"], "x_only")
                        break
            matches.append(
                {
                    "population_series_id": pid,
                    "drift_regime": regime,
                    "group": g["anonymous_source_group"].iloc[0],
                    "channel": g["polarization_channel"].iloc[0],
                    "n_bins": len(xs),
                    "events": float(gg["event_count"].sum()),
                    "workbook_sheet": best[0] if best else None,
                    "workbook_label": best[1] if best else None,
                    "match": best[2] if best else "UNMATCHED",
                }
            )
    mdf = pd.DataFrame(matches)
    print("\n=== population series -> workbook block mapping ===")
    print(mdf.to_string())
    mdf.to_csv(OUT / "s03_population_mapping.csv", index=False)

    # orientation series vs orientation blocks
    omatches = []
    for oid, g in ori.groupby("orientation_series_id"):
        xs = np.sort(g["relative_coordinate"].to_numpy(float))
        ys = g.sort_values("relative_coordinate")["event_count"].to_numpy(float)
        cands = [
            b for b in all_blocks
            if len(b["x"]) == len(xs) and np.allclose(b["x"], xs, atol=1e-6)
            and np.allclose(b["y"], ys, atol=1e-6)
        ]
        omatches.append(
            {
                "orientation_series_id": oid,
                "n": len(xs),
                "x0": float(xs[0]),
                "x1": float(xs[-1]),
                "events": float(g["event_count"].sum()),
                "candidates": [(c["sheet"], c["label"]) for c in cands],
            }
        )
    print("\n=== orientation series -> workbook block mapping ===")
    for m in omatches:
        print(m)
    pd.DataFrame(omatches).to_csv(OUT / "s03_orientation_mapping.csv", index=False)

    # union of orientation blocks in workbook: how do they compose into a 52-pt series?
    ori_blocks = [b for b in all_blocks if b["x0"] >= -34.5 and b["x1"] <= 20.5 and b["dx"] == 1.0]
    union = {}
    for b in ori_blocks:
        key = b["label"]
        union.setdefault(key, []).extend(b["x"].tolist())
    print("\n=== orientation-type workbook labels -> x extent ===")
    for k, v in sorted(union.items()):
        v = sorted(v)
        print(f"  {k!r}: n={len(v)} x=[{v[0]},{v[-1]}]")

    (OUT / "s03_orientation_labels.json").write_text(
        json.dumps({k: sorted(v) for k, v in union.items()}, indent=2) + "\n"
    )
    np.savez_compressed(
        OUT / "s03_workbook_blocks.npz",
        **{f"{b['sheet']}__{i}": np.vstack([b["x"], b["y"]]) for i, b in enumerate(all_blocks)},
    )
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()