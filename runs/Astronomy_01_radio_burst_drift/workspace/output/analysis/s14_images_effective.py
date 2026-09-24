#!/usr/bin/env python3
"""S14 (dimension: provenance & coverage) - pixel grids of the 14 image sheets.

The workbook stores each dynamic-spectrum pixel as a *text* cell backed by a
shared string (1,442,476 shared strings total).  Parsing the whole shared-string
table repeatedly is what made the naive readers slow, so this script uses a
two-pass strategy:

  pass 1  stream each of the 14 image sheets, keep the needed cells (rows 6..430,
          columns 3..427) and, for text cells, only the shared-string *index*.
  pass 2  stream xl/sharedStrings.xml once and resolve ONLY the indices collected
          in pass 1 (a few thousand distinct values) into the value cache.

Outputs:
  s14_image_bounds.csv      bounds + numeric range per image sheet
  s14_images.npz            float64 425x425 grid per image sheet
  s14_event_match.csv       event_<i> vs image sheet: exact / transform match
  s14_pairwise.csv          image-vs-image agreement (identity, correlation, diffs)
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
DATA = ROOT / "data"
XLSX = DATA / "source_data" / "all_source_data_deidentified.xlsx"
OUT = ROOT / "output" / "results"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
RNS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
IMAGE_SHEETS = [f"Sheet_{i:03d}" for i in
                [3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 21, 22, 36, 37]]
R0, R1, C0, C1 = 6, 430, 3, 427  # inclusive row / column band of the 425x425 grid


def sheet_paths(zf: zipfile.ZipFile) -> dict[str, str]:
    wb = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rmap = {r.get("Id"): r.get("Target") for r in rels}
    out = {}
    for sh in wb.iter(NS + "sheet"):
        t = rmap[sh.get(RNS + "id")]
        out[sh.get("name")] = t if t.startswith("xl/") else "xl/" + t.lstrip("/")
    return out


def parse_abc(ref: str):
    m = re.match(r"([A-Z]+)(\d+)", ref)
    col, row = m.group(1), int(m.group(2))
    n = 0
    for ch in col:
        n = n * 26 + (ord(ch) - 64)
    return n, row


def col_letter(n: int) -> str:
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def pass1(zf: zipfile.ZipFile, path: str):
    """Return (needed_sst_indices, cellmap {(row,col): ('s'|'v', payload)})."""
    needed: set[int] = set()
    cells: dict[tuple[int, int], tuple[str, object]] = {}
    with zf.open(path) as raw:
        buf = io.BufferedReader(raw, buffer_size=1 << 20)
        for _, el in ET.iterparse(buf, events=("end",)):
            if el.tag == NS + "c":
                ref = el.get("r")
                if ref:
                    ci, ri = parse_abc(ref)
                    if R0 <= ri <= R1 and C0 <= ci <= C1:
                        t = el.get("t")
                        v = el.find(NS + "v")
                        if t == "s" and v is not None:
                            idx = int(v.text)
                            needed.add(idx)
                            cells[(ri, ci)] = ("s", idx)
                        elif t == "inlineStr":
                            isv = el.find(NS + "is")
                            cells[(ri, ci)] = ("t", "".join(
                                x.text or "" for x in isv.iter(NS + "t")) if isv is not None else None)
                        elif v is not None:
                            cells[(ri, ci)] = ("n", v.text)
                el.clear()
    return needed, cells


def resolve_sst(zf: zipfile.ZipFile, needed: set[int]) -> dict[int, str]:
    out: dict[int, str] = {}
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


def main() -> None:
    with zipfile.ZipFile(XLSX) as zf:
        paths = sheet_paths(zf)
        raw_cells = {}
        all_needed: set[int] = set()
        for sh in IMAGE_SHEETS:
            needed, cells = pass1(zf, paths[sh])
            raw_cells[sh] = cells
            all_needed |= needed
            print(f"pass1 {sh}: cells={len(cells)} distinct_sst={len(needed)}")
        sst = resolve_sst(zf, all_needed)
        print(f"pass2 resolved {len(sst)} shared strings")

        grids = {}
        for sh in IMAGE_SHEETS:
            g = np.full((R1 - R0 + 1, C1 - C0 + 1), np.nan)
            for (ri, ci), (kind, payload) in raw_cells[sh].items():
                if kind == "s":
                    tok = sst.get(payload)
                elif kind == "t":
                    tok = payload
                else:
                    tok = payload
                if tok is None or tok == "":
                    continue
                try:
                    g[ri - R0, ci - C0] = float(tok)
                except (TypeError, ValueError):
                    pass
            grids[sh] = g
            print(f"{sh}: grid={g.shape} n_nan={int(np.isnan(g).sum())} "
                  f"range=[{np.nanmin(g):.6f}, {np.nanmax(g):.6f}] "
                  f"n_gt1={int((g > 1.0).sum())} n_eq1={int((g == 1.0).sum())}")

    # ---- bounds from the earlier header dump is re-derived here for completeness ----
    with open(OUT / "s14_images.npz", "wb") as fh:
        np.savez_compressed(fh, **{"grid__" + k: v for k, v in grids.items()})

    rows = []
    for sh, g in grids.items():
        rows.append(dict(sheet=sh, rows=g.shape[0], cols=g.shape[1],
                         n_nan=int(np.isnan(g).sum()),
                         vmin=float(np.nanmin(g)), vmax=float(np.nanmax(g)),
                         mean=float(np.nanmean(g)), sum=float(np.nansum(g)),
                         n_gt1=int((g > 1.0).sum()), n_eq1=int((g == 1.0).sum())))
    with open(OUT / "s14_image_stats.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- normalized event grids ----
    ev = {}
    for i in range(1, 6):
        arr = np.genfromtxt(DATA / "normalized_views" / f"event_{i}_dynamic_spectrum.tsv",
                            delimiter="\t", skip_header=1)
        ev[f"event_{i}"] = arr[:, 1:]
        print(f"event_{i}: {ev[f'event_{i}'].shape} "
              f"range=[{np.nanmin(ev[f'event_{i}']):.6f},{np.nanmax(ev[f'event_{i}']):.6f}]")

    # ---- identity / transform matching ----
    mr = []
    used = {}
    for eid, G in ev.items():
        found = None
        for sh, H in grids.items():
            if H.shape != G.shape:
                continue
            for name, V in (("direct", H), ("transpose", H.T), ("flipud", H[::-1]),
                            ("fliplr", H[:, ::-1]), ("rot180", H[::-1, ::-1]),
                            ("T_flipud", H.T[::-1]), ("T_fliplr", H.T[:, ::-1])):
                if np.array_equal(V, G):
                    found = (sh, name)
                    break
            if found:
                break
        if found is None:
            best = min(((sh, float(np.nanmax(np.abs(H - G))))
                        for sh, H in grids.items() if H.shape == G.shape), key=lambda t: t[1])
            mr.append(dict(event=eid, sheet=best[0], transform="closest", max_abs_diff=best[1]))
            print(f"{eid}: NO exact match, closest {best[0]} maxdiff={best[1]:.6g}")
        else:
            used[found[0]] = eid
            mr.append(dict(event=eid, sheet=found[0], transform=found[1], max_abs_diff=0.0))
            print(f"{eid}: EXACT {found}")
    with open(OUT / "s14_event_match.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["event", "sheet", "transform", "max_abs_diff"])
        w.writeheader()
        w.writerows(mr)

    # ---- pairwise image comparison ----
    pr = []
    keys = sorted(grids)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            A, B = grids[a], grids[b]
            d = A - B
            finite = np.isfinite(d)
            pr.append(dict(a=a, b=b, mean_abs=float(np.nanmean(np.abs(d))),
                           max_abs=float(np.nanmax(np.abs(d))),
                           n_identical=int(np.sum(d[finite] == 0)),
                           frac_identical=float(np.sum(d[finite] == 0) / finite.sum()),
                           pearson=float(np.corrcoef(A[finite], B[finite])[0, 1]),
                           sum_a=float(np.nansum(A)), sum_b=float(np.nansum(B))))
    pr.sort(key=lambda r: r["mean_abs"])
    with open(OUT / "s14_pairwise.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pr[0].keys()))
        w.writeheader()
        w.writerows(pr)
    print("\nclosest image pairs:")
    for r in pr[:12]:
        print(" ", r)
    print("\nwrote s14_images.npz / s14_image_stats.csv / s14_event_match.csv / s14_pairwise.csv")


if __name__ == "__main__":
    main()