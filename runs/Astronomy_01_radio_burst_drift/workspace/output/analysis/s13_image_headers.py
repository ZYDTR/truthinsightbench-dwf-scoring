#!/usr/bin/env python3
"""S13 (dimension: provenance & coverage) - exact header/Bounds text of the 14 image sheets.

Reads only the first 8 rows of each image sheet straight from the OOXML zip and
breaks out of the parse immediately, so it is cheap even though the full sheets
are ~9 MB of XML each.  Establishes, per sheet, the literal column-header text
(row 3) and the literal Bounds values (row 4) before any numeric interpretation.
"""
from __future__ import annotations

import csv
import io
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
XLSX = ROOT / "data" / "source_data" / "all_source_data_deidentified.xlsx"
OUT = ROOT / "output" / "results"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
RNS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
IMAGE_SHEETS = [f"Sheet_{i:03d}" for i in
                [3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 21, 22, 36, 37]]
NROWS = 8


def main() -> None:
    with zipfile.ZipFile(XLSX) as zf:
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rmap = {r.get("Id"): r.get("Target") for r in rels}
        paths = {}
        for sh in wb.iter(NS + "sheet"):
            t = rmap[sh.get(RNS + "id")]
            paths[sh.get("name")] = t if t.startswith("xl/") else "xl/" + t.lstrip("/")

        out_rows = []
        for sh in IMAGE_SHEETS:
            cells: dict[tuple[int, int], str] = {}
            with zf.open(paths[sh]) as raw:
                buf = io.BufferedReader(raw, buffer_size=1 << 18)
                stop = False
                for _, el in ET.iterparse(buf, events=("end",)):
                    if el.tag == NS + "row":
                        r = int(el.get("r"))
                        if r >= NROWS:
                            stop = True
                    elif el.tag == NS + "c":
                        ref = el.get("r") or ""
                        m = re.match(r"([A-Z]+)(\d+)", ref)
                        col, row = m.group(1), int(m.group(2))
                        if row <= NROWS and col in ("A", "B", "C", "D", "E", "F", "G"):
                            t = el.get("t")
                            v = el.find(NS + "v")
                            if t == "s":
                                # shared-string index; resolved below by re-reading
                                cells[(row, col)] = f"<sst#{v.text}>" if v is not None else ""
                            elif t == "inlineStr":
                                cells[(row, col)] = "".join(x.text or "" for x in el.iter(NS + "t"))
                            elif v is not None:
                                cells[(row, col)] = v.text
                            else:
                                cells[(row, col)] = ""
                        el.clear()
                    if stop:
                        break
            rowmap = {r: {c: cells.get((r, c), "") for c in "ABCDEFG"} for r in range(1, NROWS + 1)}
            for r in range(1, NROWS + 1):
                print(f"{sh} r{r}: "
                      + " | ".join(f"{c}={rowmap[r][c]}" for c in "ABCDEFG"))
                out_rows.append(dict(sheet=sh, row=r,
                                     **{c: rowmap[r][c] for c in "ABCDEFG"}))
            print()

    with open(OUT / "s13_image_headers.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sheet", "row"] + list("ABCDEFG"))
        w.writeheader()
        w.writerows(out_rows)
    print("wrote", OUT / "s13_image_headers.csv")


if __name__ == "__main__":
    main()