#!/usr/bin/env python3
"""Integrity checks: (a) SHA-256/size of every data file vs data_manifest.json,
(b) lossless-deidentification cross-check: normalized_views TSVs vs the
corresponding source workbook sheets (openpyxl, read-only)."""
import hashlib
import json
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent          # output/
PROJECT = ROOT.parent                            # project root
DATA = PROJECT / "data"
MANIFEST = PROJECT / "data_manifest.json"

TOL = 1e-9


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def tsv_numeric(path: Path) -> list[list[float]]:
    rows = []
    with path.open() as fh:
        header = fh.readline().strip().split("\t")
        for line in fh:
            if not line.strip():
                continue
            rows.append([float(x) for x in line.strip().split("\t")])
    return header, rows


def sheet_numeric(wb: openpyxl.Workbook, idx: int) -> list[list[float]]:
    ws = wb.worksheets[idx]
    rows = []
    for row in ws.iter_rows(values_only=True):
        try:
            converted = [None if v is None else float(v) for v in row]
        except ValueError:
            continue  # header row (text)
        rows.append(converted)
    return rows


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())

    # (a) hashes and sizes
    hash_rows = []
    for f in manifest["files"]:
        p = DATA / f["path"]
        computed = sha256(p)
        size = p.stat().st_size
        hash_rows.append(
            {
                "path": f["path"],
                "manifest_sha256": f["sha256"],
                "computed_sha256": computed,
                "sha256_match": computed == f["sha256"],
                "bytes_expected": f["bytes"],
                "bytes_actual": size,
                "size_match": size == f["bytes"],
            }
        )

    # (b) normalized views vs source workbook sheets
    # Sheet indices (0-based) inferred from header inspection:
    #  Sheet_002 (idx 1)  -> control_a (Ku, Frequency_solver_a/b)
    #  Sheet_003 (idx 2)  -> control_b (DMI, Frequency_solver_a/b)
    #  Sheet_061 (idx 60) -> signed_control (mu0H, frequency_solver_a/b)
    #  Sheet_026 (idx 25) -> trajectory_condition_01 (Input, Delta_mz_a/b)
    #  Sheet_028 (idx 27) -> trajectory_condition_02
    #  Sheet_029 (idx 28) -> trajectory_condition_03
    # Workbook also holds (a, b-variant) trajectory sheets (Sheet_027, _029, _030)
    # that are NOT in the normalized views — cross-checked separately.
    cross = []
    wb = openpyxl.load_workbook(DATA / "source_data" / "all_source_data_deidentified.xlsx", read_only=True)
    variant_cross = []
    pairs = [
        ("control_a_frequency_response.tsv", 1, 3),   # cols: control, ref, fc
        ("control_b_frequency_response.tsv", 2, 3),
        ("signed_control_frequency_response.tsv", 60, 3),
        ("trajectory_condition_01.tsv", 25, 4),       # cols: time, drive, ref, fc
        ("trajectory_condition_02.tsv", 27, 4),
        ("trajectory_condition_03.tsv", 29, 4),
    ]
    # The workbook additionally carries (a, b-variant) trajectory sheets that are
    # NOT part of the normalized views; cross-check those against the same TSVs
    # to document the de-identification mapping.
    variant_pairs = [
        ("trajectory_condition_01.tsv", 26, 4),
        ("trajectory_condition_02.tsv", 28, 4),
        ("trajectory_condition_03.tsv", 30, 4),
    ]
    def compare(tsv_name, sheet_idx, n_cols, cross_list):
        header, tsv_rows = tsv_numeric(DATA / "normalized_views" / tsv_name)
        sh = sheet_numeric(wb, sheet_idx)
        if len(tsv_rows) != len(sh):
            cross_list.append({"file": tsv_name, "sheet_index": sheet_idx,
                               "row_count_match": False, "tsv_rows": len(tsv_rows),
                               "sheet_rows": len(sh)})
            return
        mismatches = 0
        # TSV columns map to workbook columns 1..n_cols (workbook col 0 is row index)
        for i, (t, s) in enumerate(zip(tsv_rows, sh)):
            for j in range(n_cols):
                if abs(t[j] - s[j + 1]) > TOL * max(1.0, abs(s[j + 1])):
                    mismatches += 1
                    break
        cross_list.append(
            {
                "file": tsv_name,
                "sheet_index": sheet_idx,
                "sheet_title": wb.worksheets[sheet_idx].title,
                "row_count_match": True,
                "mismatched_cells": mismatches,
                "lossless_match": mismatches == 0,
            }
        )

    for tsv_name, sheet_idx, n_cols in pairs:
        compare(tsv_name, sheet_idx, n_cols, cross)
    for tsv_name, sheet_idx, n_cols in variant_pairs:
        compare(tsv_name, sheet_idx, n_cols, variant_cross)
    wb.close()

    report = {
        "files_checked": len(hash_rows),
        "all_sha256_match": all(r["sha256_match"] for r in hash_rows),
        "all_size_match": all(r["size_match"] for r in hash_rows),
        "hash_rows": hash_rows,
        "cross_check": cross,
        "all_cross_match": all(c["lossless_match"] for c in cross),
    }
    out = json.dumps(report, indent=2)
    (ROOT / "integrity_check.json").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
