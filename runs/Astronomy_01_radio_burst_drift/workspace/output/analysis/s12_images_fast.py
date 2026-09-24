#!/usr/bin/env python3
"""S12 (dimension: provenance & coverage) - fast image-sheet extractor + event identity test.

Reads the 14 dynamic-spectrum sheets (Sheet_003..008, 015..018, 021, 022, 036,
037) straight from the OOXML zip with a streaming iterparse, recovering the
Bounds row and the numeric 425x425 pixel grid of each.  The workbook stores the
pixel values as *text* (t="s" shared strings, or plain text cells), so a value
cache keyed on the raw XML token is used to avoid float() on ~2.6M cells.

Outputs:
  s12_image_bounds.csv    Bounds + value-range of each of the 14 image sheets
  s12_images.npz          float64 grids + bounds vectors
  s12_event_match.csv     normalized event_<i> grid vs image-sheet grid identity
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


def shared_strings(zf: zipfile.ZipFile) -> list[str]:
    """Only strings that look numeric are needed; the rest stay as raw text."""
    out: list[str] = []
    seen = 0
    with zf.open("xl/sharedStrings.xml") as fh:
        for _, el in ET.iterparse(fh, events=("end",)):
            if el.tag == NS + "si":
                s = "".join(t.text or "" for t in el.iter(NS + "t"))
                out.append(s)
                el.clear()
                seen += 1
    print(f"sharedStrings: {seen}")
    return out


def sheet_paths(zf: zipfile.ZipFile) -> dict[str, str]:
    wb = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rmap = {r.get("Id"): r.get("Target") for r in rels}
    out = {}
    for sh in wb.iter(NS + "sheet"):
        tgt = rmap[sh.get(RNS + "id")]
        if not tgt.startswith("xl/"):
            tgt = "xl/" + tgt.lstrip("/")
        out[sh.get("name")] = tgt
    return out


def read_image(zf: zipfile.ZipFile, path: str, sst: list[str], sheet: str):
    """Return (bounds_dict, grid 425x425) for one image sheet."""
    vcache: dict[str, float] = {}
    bounds: dict = {}
    grid: list[list[float]] = []
    ncache_flush = 0
    with zf.open(path) as raw:
        buf = io.BufferedReader(raw, buffer_size=1 << 20)
        for _, el in ET.iterparse(buf, events=("end",)):
            if el.tag == NS + "c":
                ref = el.get("r") or ""
                t = el.get("t")
                row_i = int(re.search(r"(\d+)", ref).group(1))
                col_i = re.match(r"([A-Z]+)", ref).group(1)
                if sheet in IMAGE_SHEETS and row_i <= 6 and col_i in ("A", "B", "C", "D", "E", "F", "G"):
                    if t == "s":
                        v = el.find(NS + "v")
                        txt = sst[int(v.text)] if v is not None else None
                    else:
                        txt = None
                        for tag in (NS + "v", NS + "is"):
                            node = el.find(tag)
                            if node is not None:
                                txt = "".join(x.text or "" for x in node.iter(NS + "t")) or node.text
                                break
                    bounds.setdefault("hdr", {})[f"{col_i}{row_i}"] = txt
                el.clear()
            elif el.tag == NS + "row":
                r = int(el.get("r"))
                if 5 <= r <= 429:
                    vals = []
                    for c in el.findall(NS + "c"):
                        t = c.get("t")
                        v = c.find(NS + "v")
                        token = None
                        if t == "s" and v is not None:
                            token = sst[int(v.text)]
                        elif v is not None:
                            token = v.text
                        if token is None:
                            vals.append(np.nan)
                            continue
                        fv = vcache.get(token)
                        if fv is None:
                            try:
                                fv = float(token)
                            except (TypeError, ValueError):
                                fv = np.nan
                            if len(vcache) > 4_000_000:
                                vcache.clear()
                                ncache_flush += 1
                            vcache[token] = fv
                        vals.append(fv)
                    if sum(1 for v in vals if not np.isnan(v)) >= 400:
                        grid.append(vals)
                el.clear()
    return bounds, np.asarray(grid, float) if grid else None


def main() -> None:
    with zipfile.ZipFile(XLSX) as zf:
        sst = shared_strings(zf)
        paths = sheet_paths(zf)
        rows = []
        store = {}
        for sh in IMAGE_SHEETS:
            b, g = read_image(zf, paths[sh], sst, sh)
            hdr = b.get("hdr", {})
            bd = dict(sheet=sh,
                      header_tmin=hdr.get("C3"), header_tmax=hdr.get("D3"),
                      header_fmin=hdr.get("E3"), header_fmax=hdr.get("F3"),
                      header_drift=hdr.get("G3"),
                      tmin=hdr.get("C4"), tmax=hdr.get("D4"), fmin=hdr.get("E4"),
                      fmax=hdr.get("F4"), drift=hdr.get("G4"),
                      grid_rows=0 if g is None else g.shape[0],
                      grid_cols=0 if g is None else g.shape[1],
                      vmin=float(np.nanmin(g)) if g is not None else np.nan,
                      vmax=float(np.nanmax(g)) if g is not None else np.nan,
                      n_one=int(np.nansum(g == 1.0)) if g is not None else 0,
                      n_gt1=int(np.nansum(g > 1.0)) if g is not None else 0)
            rows.append(bd)
            store[sh] = g
            print(f"{sh}: grid={None if g is None else g.shape} "
                  f"tmin={bd['tmin']} drift={bd['drift']} "
                  f"range=[{bd['vmin']:.6f},{bd['vmax']:.6f}] n>1={bd['n_gt1']}")

    with open(OUT / "s12_image_bounds.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    payload = {}
    for sh, g in store.items():
        payload["grid__" + sh] = g
    for b in rows:
        payload["bounds__" + b["sheet"]] = np.array(
            [np.nan if b[k] is None else b[k] for k in ("tmin", "tmax", "fmin", "fmax", "drift")], float)
    np.savez_compressed(OUT / "s12_images.npz", **payload)

    # ---- normalized event grids ----
    meta = DATA / "normalized_views" / "dynamic_spectrum_event_metadata.tsv"
    ev_meta = {}
    with open(meta) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            ev_meta[r["event_id"]] = r
    ev = {}
    for eid in sorted(ev_meta):
        df = np.genfromtxt(DATA / "normalized_views" / f"{eid}_dynamic_spectrum.tsv",
                           delimiter="\t", skip_header=1)
        ev[eid] = df[:, 1:]  # drop frequency column
        print(f"{eid}: normalized grid {ev[eid].shape} range "
              f"[{np.nanmin(ev[eid]):.6f},{np.nanmax(ev[eid]):.6f}]")

    # ---- identity test: exact / transpose / shifted ----
    match_rows = []
    for eid, G in ev.items():
        best = None
        for sh, H in store.items():
            if H is None:
                continue
            variants = {
                "direct": H,
                "transpose": H.T,
                "flipud": H[::-1, :],
                "fliplr": H[:, ::-1],
                "rot180": H[::-1, ::-1],
                "transpose_flipud": H.T[::-1, :],
                "transpose_fliplr": H.T[:, ::-1],
            }
            for name, V in variants.items():
                if V.shape != G.shape:
                    continue
                d = np.nanmax(np.abs(V - G))
                if d == 0.0:
                    match_rows.append(dict(event=eid, sheet=sh, transform=name,
                                           max_abs_diff=0.0, exact=True))
                    best = (sh, name, 0.0)
                    break
                if best is None or d < best[2]:
                    best = (sh, name, float(d))
            if any(m["event"] == eid and m["exact"] for m in match_rows):
                break
        if not any(m["event"] == eid for m in match_rows):
            match_rows.append(dict(event=eid, sheet=best[0], transform=best[1] + " (closest)",
                                   max_abs_diff=best[2], exact=False))
        print(f"{eid}: {match_rows[-1]}")

    with open(OUT / "s12_event_match.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["event", "sheet", "transform", "max_abs_diff", "exact"])
        w.writeheader()
        w.writerows(match_rows)
    print("\nwrote s12_image_bounds.csv / s12_images.npz / s12_event_match.csv")


if __name__ == "__main__":
    main()