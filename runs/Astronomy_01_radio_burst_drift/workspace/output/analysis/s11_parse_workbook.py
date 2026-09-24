#!/usr/bin/env python3
"""S11 (dimension: provenance & coverage) - full tidy parse of the 47-sheet workbook.

A direct OOXML (zip + iterparse) reader is used instead of openpyxl because the
workbook holds ~2.6M populated cells across 14 image sheets; the streaming XML
reader makes a complete, deterministic parse cheap enough to re-run.

Outputs (under output/results/):
  s11_blocks.csv        tidy (sheet, block_label, block_kind, x, y) rows for every
                        NON-image sheet block, including RD/Fit/params blocks.
  s11_block_index.csv   one row per (sheet, block_label): size, x-range, y-sum.
  s11_image_bounds.csv  Bounds row of each of the 14 image sheets.
  s11_images.npz        float64 425x425 pixel grid per image sheet + bounds arrays.
"""
from __future__ import annotations

import csv
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


def col_index(ref: str) -> int:
    """'BC12' -> 1-based column index of BC."""
    letters = re.match(r"([A-Z]+)", ref).group(1)
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n


def shared_strings(zf: zipfile.ZipFile) -> list[str]:
    try:
        data = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    out = []
    for _, el in ET.iterparse(__import__("io").BytesIO(data), events=("end",)):
        if el.tag == NS + "si":
            out.append("".join(t.text or "" for t in el.iter(NS + "t")))
            el.clear()
    return out


def sheet_paths(zf: zipfile.ZipFile) -> list[tuple[str, str]]:
    wb = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rmap = {r.get("Id"): r.get("Target") for r in rels}
    pairs = []
    for sh in wb.iter(NS + "sheet"):
        tgt = rmap[sh.get(RNS + "id")]
        if not tgt.startswith("xl/"):
            tgt = "xl/" + tgt.lstrip("/")
        pairs.append((sh.get("name"), tgt))
    return pairs


def read_sheet(zf: zipfile.ZipFile, path: str, sst: list[str]) -> list[list]:
    """Return the sheet as a list of rows; each row is a list of cell values."""
    rows: list[list] = []
    cur: list = []
    maxc = 0
    with zf.open(path) as fh:
        for _, el in ET.iterparse(fh, events=("end",)):
            if el.tag == NS + "row":
                for c in el.findall(NS + "c"):
                    ci = col_index(c.get("r")) - 1
                    t = c.get("t")
                    v = c.find(NS + "v")
                    if t == "s":
                        val = sst[int(v.text)] if v is not None else None
                    elif t == "inlineStr":
                        isv = c.find(NS + "is")
                        val = "".join(x.text or "" for x in isv.iter(NS + "t")) if isv is not None else None
                    elif v is None:
                        val = None
                    else:
                        try:
                            val = float(v.text)
                        except (TypeError, ValueError):
                            val = v.text
                    while len(cur) <= ci:
                        cur.append(None)
                    cur[ci] = val
                    maxc = max(maxc, ci + 1)
                rows.append(cur)
                cur = []
                el.clear()
    for r in rows:
        while len(r) < maxc:
            r.append(None)
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(XLSX) as zf:
        sst = shared_strings(zf)
        pairs = sheet_paths(zf)
        print(f"sheets: {len(pairs)}")

        bounds_rows: list[dict] = []
        image_grids: dict[str, np.ndarray] = {}
        block_rows: list[dict] = []
        index_rows: list[dict] = []

        for name, path in pairs:
            rows = read_sheet(zf, path, sst)
            nrow = len(rows)
            ncol = max((len(r) for r in rows), default=0)
            cls = ("image" if (nrow >= 420 and ncol >= 400) else
                   "empty" if nrow == 1 and ncol <= 1 else "table")
            if cls == "empty":
                index_rows.append(dict(sheet=name, class_="empty", nrow=nrow, ncol=ncol,
                                       block_label="<none>", n=0, x0=np.nan, x1=np.nan,
                                       dx=np.nan, y_sum=0.0))
                print(f"{name}: empty(sheet) {nrow}x{ncol}")
                continue
            if cls == "image":
                bounds, grid, hdr = parse_image(name, rows)
                bounds_rows.append(bounds)
                if grid is not None:
                    image_grids[name] = grid
                index_rows.append(dict(sheet=name, class_="image", nrow=nrow, ncol=ncol,
                                       block_label="Image", n=grid.shape[0] * grid.shape[1] if grid is not None else 0,
                                       x0=np.nan, x1=np.nan, dx=np.nan,
                                       y_sum=float(np.nansum(grid)) if grid is not None else np.nan))
                print(f"{name}: image grid={None if grid is None else grid.shape} bounds={bounds} hdr={hdr}")
                continue

            blocks = parse_table(name, rows)
            for b in blocks:
                xs = np.asarray(b["x"], float)
                ys = np.asarray(b["y"], float)
                for xx, yy in zip(b["x"], b["y"]):
                    block_rows.append(dict(sheet=name, block_label=b["label"],
                                           block_kind=b["kind"], x=xx, y=yy))
                dx = float(np.median(np.diff(xs))) if len(xs) > 1 else np.nan
                index_rows.append(dict(sheet=name, class_="table", nrow=nrow, ncol=ncol,
                                       block_label=b["label"], n=len(xs),
                                       x0=float(xs[0]) if len(xs) else np.nan,
                                       x1=float(xs[-1]) if len(xs) else np.nan,
                                       dx=dx, y_sum=float(np.nansum(ys))))
            print(f"{name}: table {nrow}x{ncol} blocks={[ (b['label'], len(b['x'])) for b in blocks ]}")

    with open(OUT / "s11_blocks.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sheet", "block_label", "block_kind", "x", "y"])
        w.writeheader()
        w.writerows(block_rows)
    with open(OUT / "s11_block_index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sheet", "class_", "nrow", "ncol", "block_label",
                                           "n", "x0", "x1", "dx", "y_sum"])
        w.writeheader()
        w.writerows(index_rows)
    with open(OUT / "s11_image_bounds.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sheet", "header_tmin", "header_tmax", "header_fmin",
                                           "header_fmax", "header_drift", "tmin", "tmax",
                                           "fmin", "fmax", "drift", "grid_rows", "grid_cols"])
        w.writeheader()
        w.writerows(bounds_rows)

    payload = {}
    for k, g in image_grids.items():
        payload["grid__" + k] = g
    for b in bounds_rows:
        payload["bounds__" + b["sheet"]] = np.array(
            [np.nan if b[k] is None else b[k] for k in
             ("tmin", "tmax", "fmin", "fmax", "drift")], float)
    np.savez_compressed(OUT / "s11_images.npz", **payload)
    print(f"\nwrote s11_blocks.csv rows={len(block_rows)} images={len(image_grids)} "
          f"bounds={len(bounds_rows)} index={len(index_rows)}")


def parse_image(name: str, rows: list[list]) -> tuple[dict, np.ndarray | None, list]:
    """Bounds row (label 'Bounds' in col B) + the numeric 425x425 block."""
    hdr = [v for v in rows[2][1:6]] if len(rows) > 2 else []
    bnd: dict = dict(sheet=name, header_tmin=hdr[0] if len(hdr) > 0 else None,
                     header_tmax=hdr[1] if len(hdr) > 1 else None,
                     header_fmin=hdr[2] if len(hdr) > 2 else None,
                     header_fmax=hdr[3] if len(hdr) > 3 else None,
                     header_drift=hdr[4] if len(hdr) > 4 else None,
                     tmin=None, tmax=None, fmin=None, fmax=None, drift=None,
                     grid_rows=0, grid_cols=0)
    for r in rows[:8]:
        if len(r) > 1 and isinstance(r[1], str) and r[1].strip() == "Bounds":
            vals = r[2:7]
            bnd.update(tmin=vals[0], tmax=vals[1], fmin=vals[2], fmax=vals[3],
                       drift=vals[4] if len(vals) > 4 else None)
            break
    # numeric grid: contiguous rows whose col1..colN are numeric and col0/col1 not str
    grid = []
    for r in rows:
        if len(r) > 1 and isinstance(r[1], str):
            continue
        vals = r[1:426]
        num = [v for v in vals if isinstance(v, (int, float))]
        if len(num) >= 400:
            grid.append([float(v) if isinstance(v, (int, float)) else np.nan for v in vals])
    g = np.asarray(grid, float) if grid else None
    if g is not None:
        bnd["grid_rows"], bnd["grid_cols"] = g.shape
    return bnd, g, hdr


def parse_table(name: str, rows: list[list]) -> list[dict]:
    """Recover labelled (x,y) blocks separated by blank rows / new col-B strings."""
    blocks: list[dict] = []
    cur = None
    for r in rows:
        b = r[1] if len(r) > 1 else None
        x = r[2] if len(r) > 2 else None
        y = r[3] if len(r) > 3 else None
        if isinstance(b, str) and b.strip():
            cur = {"label": b.strip(), "x": [], "y": [], "kind": "distribution"}
            blocks.append(cur)
            # a label row may itself carry numbers in col C/D (Sheet_020 'params')
            if isinstance(x, (int, float)) and not isinstance(y, (int, float)):
                cur["kind"] = "params"
                cur["x"].append(float(x))
                cur["y"].append(0.0)
            elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
                cur["x"].append(float(x))
                cur["y"].append(float(y))
            continue
        if cur is None:
            continue
        if isinstance(x, (int, float)):
            cur["x"].append(float(x))
            cur["y"].append(float(y) if isinstance(y, (int, float)) else np.nan)
    # drop label placeholders with no rows
    return [b for b in blocks if len(b["x"]) > 0]


if __name__ == "__main__":
    main()