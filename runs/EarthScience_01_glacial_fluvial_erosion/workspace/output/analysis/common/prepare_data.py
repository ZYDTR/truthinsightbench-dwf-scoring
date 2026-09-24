#!/usr/bin/env python3
"""Consolidate the frozen glacial / nonglacial erosion-rate tables into one analysis frame.

This module is the shared foundation for every downstream analysis in output/analysis/.
It performs no filtering decisions of its own beyond parsing: all rows are retained and
all exclusion rules are exposed as explicit, auditable columns so that downstream
analyses can report their own filter counts.

Only workspace data under data/ is read. No external data is used.

Outputs
-------
output/analysis/common/erosion_consolidated.csv   one row per published estimate
output/analysis/common/parse_audit.json           row-level parsing and missingness audit
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

G_FILE = RAW / "glacial_erosion_Earth.tsv"
N_FILE = RAW / "nonglacial_erosion_Earth.tsv"

NUMERIC = [
    "Area (km2)",
    "Time interval (yr)",
    "Erosion rate (mm/yr)",
    "Erosion rate min",
    "Erosion rate max",
    "Slope (m/km)",
    "Precipitation (mm/yr)",
    "Latitude",
    "Longitude",
    "Elevation (m)",
    "Exhumation (m)",
    "Sediment flux (km3/yr)",
]

SHARED = [
    "Compilation",
    "Reference",
    "Reference DOI",
    "Locality",
    "Region",
    "Methodology",
    "Type",
    "Area (km2)",
    "Time interval (yr)",
    "Erosion rate (mm/yr)",
    "Slope (m/km)",
    "Precipitation (mm/yr)",
    "Latitude",
    "Longitude",
    "Exhumation (m)",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path, process: str) -> pd.DataFrame:
    """Read one frozen table, coercing numerics and retaining raw string values."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df = df.replace({"": np.nan})
    for column in NUMERIC:
        if column in df.columns:
            df[column + " (raw)"] = df[column]
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["Process"] = process
    df["source_row"] = np.arange(len(df))
    df["source_file"] = path.name
    return df


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add the log-space and product columns every downstream analysis shares."""
    rate = df["Erosion rate (mm/yr)"]
    span = df["Time interval (yr)"]
    area = df["Area (km2)"]
    slope = df["Slope (m/km)"]
    precip = df["Precipitation (mm/yr)"]

    df["logER"] = np.where(rate > 0, np.log10(rate.where(rate > 0)), np.nan)
    df["logTI"] = np.where(span > 0, np.log10(span.where(span > 0)), np.nan)
    df["logA"] = np.where(area > 0, np.log10(area.where(area > 0)), np.nan)
    df["logS"] = np.where(slope > 0, np.log10(slope.where(slope > 0)), np.nan)
    df["logP"] = np.where(precip > 0, np.log10(precip.where(precip > 0)), np.nan)

    # ET is the eroded thickness implied by the reported rate acting over the reported span.
    df["ET_mm"] = rate * span
    df["logET"] = np.where(df["ET_mm"] > 0, np.log10(df["ET_mm"].where(df["ET_mm"] > 0)), np.nan)

    # Exact unit identity: rate[mm/yr] * span[yr] == exhumation[m] * 1000.
    exh = df["Exhumation (m)"]
    df["ET_vs_exhumation_ratio"] = df["ET_mm"] / (exh * 1000.0)
    identity = pd.Series(
        np.isclose(df["ET_mm"], exh * 1000.0, rtol=1e-3, atol=0.0),
        index=df.index,
    )
    df["ET_equals_exhumation"] = identity.where(df["ET_mm"].notna() & exh.notna())

    df["has_logER"] = df["logER"].notna()
    df["has_logTI"] = df["logTI"].notna()

    # Independence key: rows sharing a source, site or compilation are not independent replicates.
    df["site_key"] = df["Latitude"].astype(str).str.strip() + ";" + df["Longitude"].astype(str).str.strip()
    return df


def main() -> None:
    audit: dict[str, object] = {
        "generated_by": "output/analysis/common/prepare_data.py",
        "inputs": {},
        "row_counts": {},
        "missingness": {},
        "parse_anomalies": {},
    }

    frames = []
    for path, process in ((G_FILE, "Glacial"), (N_FILE, "Nonglacial")):
        audit["inputs"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        raw = load(path, process)
        audit["row_counts"][process] = int(len(raw))
        audit["missingness"][process] = {
            column: int(raw[column].isna().sum())
            for column in SHARED
            if column in raw.columns
        }
        frames.append(raw[SHARED + ["Process", "source_row", "source_file"]])

    combined = pd.concat(frames, ignore_index=True)
    combined = add_derived(combined)

    audit["parse_anomalies"] = {
        "nonpositive_erosion_rate_rows": int((combined["Erosion rate (mm/yr)"] <= 0).sum()),
        "nonpositive_or_missing_time_interval_rows": int(
            (combined["Time interval (yr)"].isna() | (combined["Time interval (yr)"] <= 0)).sum()
        ),
        "rows_with_exhumation_and_identity_checked": int(
            combined["ET_equals_exhumation"].notna().sum()
        ),
        "rows_satisfying_E_times_T_equals_exhumation": int(
            combined["ET_equals_exhumation"].fillna(False).sum()
        ),
        "unique_site_keys": int(combined["site_key"].nunique()),
        "unique_references": int(combined["Reference"].nunique()),
        "unique_compilations": int(combined["Compilation"].nunique()),
    }

    combined.to_csv(OUT / "erosion_consolidated.csv", index=False)
    (OUT / "parse_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
