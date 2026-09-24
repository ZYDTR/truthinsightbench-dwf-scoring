#!/usr/bin/env python3
"""covx_s01 (dimension: coverage boundary / absent families).

Extracts, from the workbook, ONLY the raw material needed to characterise the
50 blocks the normalized views omit.  Uses the s14 two-pass streaming strategy
so that the 1.44M-entry shared-string table is resolved once and only the
wanted rows are kept.

Sheets touched: 001, 002 (cml-angle vs phase-angle, 33638 rows),
019, 020, 023, 038 (radial-distance / SNR), 039..044 (coarse drift histograms),
046, 047 (group_b C variants), 024,026,028,030,032,034 (fine slow blocks, for
the coarse-vs-fine rebinning check) and 025,027,029,031,033,035 (fast).

Output: output/results/covx_absent_blocks.csv  (tidy sheet,label,x,y)
"""
from __future__ import annotations

import csv
import io
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
XLSX = ROOT / "data" / "source_data" / "all_source_data_deidentified.xlsx"
OUT = ROOT / "output" / "results"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
RNS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

TARGET = [f"Sheet_{i:03d}" for i in
          [1, 2, 19, 20, 23, 38, 39, 40, 41, 42, 43, 44, 46, 47,
           24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]]
COLS = (1, 2, 3)  # 0-based: col B (label), col C (x), col D (y)


def parse_ref(ref: str):
    m = re.match(r"([A-Z]+)(\d+)", ref)
    col, row = m.group(1), int(m.group(2))
    n = 0
    for ch in col:
        n = n * 26 + (ord(ch) - 64)
    return n, row  # 1-based


def sheet_paths(zf):
    wb = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rmap = {r.get("Id"): r.get("Target") for r in rels}
    out = {}
    for sh in wb.iter(NS + "sheet"):
        t = rmap[sh.get(RNS + "id")]
        out[sh.get("name")] = t if t.startswith("xl/") else "xl/" + t.lstrip("/")
    return out


def pass1(zf, path):
    """{(row,col): ('s',idx)|('n',text)|('t',str)} for cols B..D of every row."""
    needed: set[int] = set()
    cells: dict[tuple[int, int], tuple[str, object]] = {}
    with zf.open(path) as raw:
        buf = io.BufferedReader(raw, buffer_size=1 << 20)
        for _, el in ET.iterparse(buf, events=("end",)):
            if el.tag == NS + "c":
                ref = el.get("r")
                if ref:
                    ci, ri = parse_ref(ref)
                    if ci in (c + 1 for c in COLS):
                        t = el.get("t")
                        v = el.find(NS + "v")
                        if t == "s" and v is not None:
                            idx = int(v.text)
                            needed.add(idx)
                            cells[(ri, ci - 1)] = ("s", idx)
                        elif t == "inlineStr":
                            isv = el.find(NS + "is")
                            cells[(ri, ci - 1)] = ("t", "".join(
                                x.text or "" for x in isv.iter(NS + "t")) if isv is not None else None)
                        elif v is not None:
                            cells[(ri, ci - 1)] = ("n", v.text)
                el.clear()
    return needed, cells


def resolve_sst(zf, needed):
    out = {}
    if not needed:
        return out
    hi = max(needed)
    with zf.open("xl/sharedStrings.xml") as raw:
        buf = io.BufferedReader(raw, buffer_size=1 << 20)
        i = -1
        for _, el in ET.iterparse(buf, events=("end",)):
            if el.tag == NS + "si":
                i += 1
                if i in needed:
                    out[i] = "".join(t.text or "" for t in el.iter(NS + "t"))
                el.clear()
                if i >= hi:
                    break
    return out


def main():
    with zipfile.ZipFile(XLSX) as zf:
        paths = sheet_paths(zf)
        raw = {}
        allneeded: set[int] = set()
        for sh in TARGET:
            need, cells = pass1(zf, paths[sh])
            raw[sh] = cells
            allneeded |= need
            print(f"pass1 {sh}: cells={len(cells)} distinct_sst={len(need)}")
        sst = resolve_sst(zf, allneeded)
        print(f"pass2 resolved {len(sst)} shared strings")

        rows_out = []
        for sh in TARGET:
            cells = raw[sh]
            maxrow = max((r for r, _ in cells), default=0)
            grid = {}
            for (r, c), (kind, payload) in cells.items():
                if kind == "s":
                    tok = sst.get(payload)
                elif kind == "t":
                    tok = payload
                else:
                    tok = payload
                if tok is None or tok == "":
                    continue
                try:
                    grid[(r, c)] = float(tok)
                except (TypeError, ValueError):
                    grid[(r, c)] = str(tok)
            # emit in row order.  A 'label' row is one whose col B is a *string*
            # (a shared-string token); numeric col B is only a block's first point.
            cur_label = None
            nrow = 0
            for r in range(1, maxrow + 1):
                b = grid.get((r, 1))
                x = grid.get((r, 2))
                y = grid.get((r, 3))
                if isinstance(b, str) and b.strip():
                    cur_label = b.strip()
                    rows_out.append(dict(sheet=sh, row=r, label=cur_label,
                                         x=x if isinstance(x, (int, float)) else "",
                                         y=y if isinstance(y, (int, float)) else "",
                                         kind="label", raw_b=b,
                                         raw_c=str(x), raw_d=str(y)))
                    nrow += 1
                    continue
                if isinstance(x, (int, float)):
                    rows_out.append(dict(sheet=sh, row=r, label=cur_label or "",
                                         x=x,
                                         y=y if isinstance(y, (int, float)) else "",
                                         kind="point", raw_b="", raw_c=str(x),
                                         raw_d=str(y)))
                    nrow += 1
                elif isinstance(b, str) or isinstance(y, str):
                    rows_out.append(dict(sheet=sh, row=r, label=cur_label or "",
                                         x="", y="", kind="note",
                                         raw_b=str(b), raw_c=str(x), raw_d=str(y)))
                    nrow += 1
            print(f"{sh}: emitted {nrow} rows, maxrow={maxrow}")

    # also keep a raw header dump for every target sheet (first 12 rows)
    with open(OUT / "covx_absent_blocks.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sheet", "row", "label", "x", "y",
                                           "kind", "raw_b", "raw_c", "raw_d"])
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote covx_absent_blocks.csv rows={len(rows_out)}")


if __name__ == "__main__":
    main()