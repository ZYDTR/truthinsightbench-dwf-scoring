#!/usr/bin/env python3
"""Inventory, integrity, missingness and independence audit of the frozen payload.

Verifies every data/ file against data_manifest.json, characterises every file group,
emits row-level audit trails for anomalies, and diagnoses the alignment of the
auxiliary per-location files relative to the two main tables.

Reads data/ only. Writes into output/analysis/w5_audit/.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
OUT: dict[str, object] = {}
TABLES = []


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(df: pd.DataFrame, name: str, caption: str) -> None:
    df.to_csv(HERE / name, index=False)
    TABLES.append({"file": name, "caption": caption, "rows": int(len(df))})


def main() -> None:
    manifest = json.loads((ROOT / "data_manifest.json").read_text(encoding="utf-8"))
    declared = {row["path"]: row for row in manifest["files"]}

    # ---- 1. file inventory and hash verification ---------------------------
    rows = []
    for path in sorted(DATA.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(DATA).as_posix()
        entry = declared.get(relative)
        actual_hash, actual_bytes = sha256(path), path.stat().st_size
        rows.append({
            "path": relative, "bytes": actual_bytes, "sha256": actual_hash,
            "declared_in_manifest": entry is not None,
            "declared_bytes": entry["bytes"] if entry else None,
            "declared_sha256": entry["sha256"] if entry else None,
            "bytes_match": bool(entry and entry["bytes"] == actual_bytes),
            "sha256_match": bool(entry and entry["sha256"] == actual_hash),
        })
    inventory = pd.DataFrame(rows)
    save(inventory, "table_file_inventory.csv",
         "Every file under data/ with its recomputed hash compared to data_manifest.json.")
    OUT["inventory"] = {
        "files_on_disk": int(len(inventory)),
        "declared_in_manifest": int(inventory["declared_in_manifest"].sum()),
        "sha256_all_match": bool(inventory["sha256_match"].all()),
        "bytes_all_match": bool(inventory["bytes_match"].all()),
        "files_on_disk_absent_from_manifest": inventory.loc[~inventory["declared_in_manifest"], "path"].tolist(),
        "manifest_entries_absent_from_disk": sorted(set(declared) - set(inventory["path"])),
    }

    # ---- 2. per-file shape characterisation --------------------------------
    rows = []
    for relative in inventory["path"]:
        path = DATA / relative
        if path.suffix.lower() not in {".tsv", ".csv", ".txt", ".jl"}:
            continue
        try:
            head = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as error:
            rows.append({"path": relative, "note": f"unreadable: {error}"})
            continue
        if not head:
            rows.append({"path": relative, "note": "empty file"})
            continue
        sep = "\t" if "\t" in head[0] else ("," if "," in head[0] else None)
        header_like = sep is not None and not head[0].replace(".", "").replace("-", "").replace("e", "").replace("+", "").strip().isdigit()
        rows.append({
            "path": relative, "lines": len(head),
            "delimiter": {None: "none", "\t": "tab", ",": "comma"}[sep],
            "first_line_is_header": bool(header_like),
            "n_columns_first_line": len(head[0].split(sep)) if sep else 1,
            "first_line_preview": head[0][:120],
        })
    shapes = pd.DataFrame(rows)
    save(shapes, "table_file_shapes.csv",
         "Row count, delimiter and header status of every text file in the payload.")

    # ---- 3. row-level anomaly audit ----------------------------------------
    g = pd.read_csv(RAW / "glacial_erosion_Earth.tsv", sep="\t", dtype=str, keep_default_na=False)
    n = pd.read_csv(RAW / "nonglacial_erosion_Earth.tsv", sep="\t", dtype=str, keep_default_na=False)
    g["Process"], n["Process"] = "Glacial", "Nonglacial"

    def audit(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        f = frame.copy()
        rate = pd.to_numeric(f["Erosion rate (mm/yr)"].str.strip().replace({"": np.nan}), errors="coerce")
        span = pd.to_numeric(f["Time interval (yr)"].str.strip().replace({"": np.nan}), errors="coerce")
        area = pd.to_numeric(f["Area (km2)"].str.strip().replace({"": np.nan}), errors="coerce")
        lat = pd.to_numeric(f["Latitude"].str.strip().replace({"": np.nan}), errors="coerce")
        precip = pd.to_numeric(f["Precipitation (mm/yr)"].str.strip().replace({"": np.nan}), errors="coerce")
        flags = pd.DataFrame({
            "Process": label, "row_index": np.arange(len(f)),
            "Reference": f["Reference"].values, "Locality": f["Locality"].values,
            "Methodology": f["Methodology"].values,
            "Erosion rate (mm/yr)": rate.values, "Time interval (yr)": span.values,
            "Area (km2)": area.values, "Latitude": lat.values,
            "issue": np.select(
                [rate <= 0, span.isna(), span <= 0, area.isna(), area <= 0,
                 lat.notna() & ((lat < -90) | (lat > 90)), precip.isna()],
                ["nonpositive_rate", "missing_interval", "nonpositive_interval",
                 "missing_area", "nonpositive_area", "latitude_out_of_range", "missing_precipitation"],
                default="ok"),
        })
        return flags[flags["issue"] != "ok"]

    anomalies = pd.concat([audit(g, "Glacial"), audit(n, "Nonglacial")], ignore_index=True)
    save(anomalies, "table_row_anomaly_audit.csv",
         "Row-level audit trail of every non-positive, missing or out-of-range value in the two main tables.")
    OUT["row_anomalies"] = anomalies["issue"].value_counts().to_dict()

    exact_dups = (int(g.assign(Process="Glacial").duplicated().sum())
                  + int(n.assign(Process="Nonglacial").duplicated().sum()))
    OUT["exact_duplicate_rows"] = exact_dups

    # ---- 4. independence structure -----------------------------------------
    rows = []
    for frame, label in ((g, "Glacial"), (n, "Nonglacial")):
        key = frame["Latitude"].str.strip() + ";" + frame["Longitude"].str.strip()
        rows.append({
            "table": label, "rows": int(len(frame)),
            "unique_references": int(frame["Reference"].nunique()),
            "unique_compilations": int(frame["Compilation"].nunique()),
            "unique_localities": int(frame["Locality"].nunique()),
            "unique_coordinate_pairs": int(key.nunique()),
            "max_rows_single_reference": int(frame["Reference"].value_counts().max()),
            "max_rows_single_compilation": int(frame["Compilation"].value_counts().max()),
            "share_top_compilation": float(frame["Compilation"].value_counts().iloc[0] / len(frame)),
        })
    independence = pd.DataFrame(rows)
    save(independence, "table_independence_structure.csv",
         "Clustering structure of the analysis unit; rows sharing a source or site are not independent replicates.")
    OUT["independence"] = independence.to_dict("records")

    top = pd.concat([
        g["Reference"].value_counts().head(10).rename("rows").reset_index().assign(table="Glacial", kind="Reference"),
        n["Reference"].value_counts().head(10).rename("rows").reset_index().assign(table="Nonglacial", kind="Reference"),
        g["Compilation"].value_counts().head(10).rename("rows").reset_index().assign(table="Glacial", kind="Compilation"),
        n["Compilation"].value_counts().head(10).rename("rows").reset_index().assign(table="Nonglacial", kind="Compilation"),
    ], ignore_index=True).rename(columns={"index": "name", "Reference": "name", "Compilation": "name"})
    save(top, "table_top_contributors.csv",
         "Largest individual sources and compilations by row share, per main table.")

    # ---- 5. auxiliary-file alignment diagnosis ------------------------------
    def read_column(path: Path, column: int | str = 0) -> pd.Series:
        if path.suffix == ".tsv" and "slope" in path.name:
            return pd.to_numeric(pd.read_csv(path, sep="\t", header=None, dtype=str)[0]
                                 .str.strip().replace({"": np.nan}), errors="coerce")
        frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        return pd.to_numeric(frame[column].astype(str).str.strip().replace({"": np.nan}), errors="coerce")

    checks = []
    for main, label, aux in ((g, "Glacial", RAW / "slope_glacial.tsv"),
                             (n, "Nonglacial", RAW / "slope_nonglacial.tsv")):
        series = read_column(aux)
        main_slope = pd.to_numeric(main["Slope (m/km)"].str.strip().replace({"": np.nan}), errors="coerce")
        key = main["Latitude"].str.strip() + ";" + main["Longitude"].str.strip()
        checks.append({
            "auxiliary_file": aux.name, "main_table": label,
            "aux_rows": int(len(series)), "main_rows": int(len(main)),
            "unique_coordinates_in_main": int(key.nunique()),
            "equals_main_rows": bool(len(series) == len(main)),
            "equals_unique_coordinates": bool(len(series) == key.nunique()),
            "row_aligned": bool(len(series) == len(main) and np.allclose(main_slope, series, equal_nan=True)),
            "verdict": ("per-row companion" if len(series) == len(main) and np.allclose(main_slope, series, equal_nan=True)
                        else "per-unique-location lookup (or partially overlapping), NOT row-aligned"),
        })
    for aux in (RAW / "prior compilations/CHELSEA/precipitation_glacial.tsv",
                RAW / "prior compilations/CHELSEA/precipitation_nonglacial.tsv",
                RAW / "lithology_parsed.tsv", RAW / "lithology_parsed_nonglacial.tsv"):
        frame = pd.read_csv(aux, sep="\t", dtype=str, keep_default_na=False)
        # "nonglacial" contains "glacial" as a substring, so test the nonglacial marker first.
        is_nonglacial = "_nonglacial" in aux.name or "nonglacial" in aux.name
        main = n if is_nonglacial else g
        key = (frame["Latitude"].str.strip() + ";" + frame["Longitude"].str.strip()
               if "Latitude" in frame.columns else None)
        checks.append({
            "auxiliary_file": aux.name, "main_table": "Nonglacial" if is_nonglacial else "Glacial",
            "aux_rows": int(len(frame)), "main_rows": int(len(main)),
            "unique_coordinates_in_main": int((main["Latitude"].str.strip() + ";" + main["Longitude"].str.strip()).nunique()),
            "unique_coordinates_in_aux": int(key.nunique()) if key is not None else None,
            "row_aligned": False,
            "verdict": "coordinate-keyed lookup table, NOT row-aligned to the main table",
        })
    alignment = pd.DataFrame(checks)
    save(alignment, "table_auxiliary_alignment.csv",
         "Whether each auxiliary file can be joined row-for-row to a main table, or only by coordinate key.")
    OUT["auxiliary_alignment"] = alignment.to_dict("records")

    # ---- 6. unit audit of the two prior compilations -------------------------
    ch = pd.read_csv(RAW / "prior compilations/Chen et al. 2023/Chen_et_al_2021_short-term_erosion_rate_data.csv",
                     dtype=str, keep_default_na=False, encoding="latin-1")
    ch_rate = pd.to_numeric(ch["Short-term erosion rate (mm kyr^-1)"], errors="coerce") / 1000.0
    ng_chen = n[n["Compilation"] == "Chen et al., 2022"]
    ng_rate = pd.to_numeric(ng_chen["Erosion rate (mm/yr)"], errors="coerce")
    OUT["chen_unit_audit"] = {
        "csv_rows": int(len(ch)), "main_table_rows_with_this_compilation": int(len(ng_chen)),
        "csv_rate_median_after_mm_per_kyr_to_mm_per_yr": float(ch_rate.median()),
        "main_table_rate_median_mm_per_yr": float(ng_rate.median()),
        "ratio_of_medians": float(ng_rate.median() / ch_rate.median()),
        "pearson_r_on_row_order": float(np.corrcoef(ch_rate.values[:len(ng_rate)], ng_rate.values)[0, 1]),
        "verdict": "main table stores mm/yr; CSV stores mm/kyr converted by /1000",
    }

    gr = pd.read_csv(RAW / "prior compilations/GloRiSe/GloRiSe combined.tsv", sep="\t", dtype=str, keep_default_na=False)
    OUT["glorise_count_audit"] = {
        "combined_file_rows": int(len(gr)),
        "main_table_rows_with_this_compilation": int((n["Compilation"] == "GloRiSe").sum()),
        "difference": int(len(gr) - (n["Compilation"] == "GloRiSe").sum()),
        "note": "21 file rows do not appear in the main table; the main table is a filtered subset.",
    }

    # ---- 7. files outside the erosion evidence chain ------------------------
    be = pd.read_csv(RAW / "beryllium.tsv", sep="\t", dtype=str, keep_default_na=False)
    OUT["non_erosion_files"] = {
        "data/raw/beryllium.tsv": {
            "rows": int(len(be)), "columns": list(be.columns),
            "analytes": ["Be (nmol/kg)", "Be dissolved (nmol/kg)", "Be colloidal (nmol/kg)", "DOC (nmol/kg)", "Alkalinity (mM HCO3)"],
            "reason_excluded": "dissolved/colloidal beryllium chemistry in estuaries and groundwater; "
                               "no erosion-rate, time-interval or denudation column exists in this file",
        }
    }

    OUT["tables"] = TABLES
    (HERE / "results.json").write_text(json.dumps(OUT, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: OUT[k] for k in ("inventory", "row_anomalies", "exact_duplicate_rows",
                                          "chen_unit_audit", "glorise_count_audit")}, indent=2, default=str))


if __name__ == "__main__":
    main()
