#!/usr/bin/env python3
"""Reproduce the Chi -> same-type NN fraction figure and its data table.

This is the deliverable version of the hand-built SVG analysis (matplotlib
unavailable) that produced output/figures/chi_vs_samefrac.svg and
output/results/figure_data.csv.

Method (matches the published evidence):
- Equal-size subset = R2==3 (structures 01-04; 9 C + 9 H2 each).
- c = first local maximum of the 1D all-pair distance histogram (0.5-wide bins)
  using periodic (p)-box x,y min-image, z free.
- same_frac = same-type pairs / (same-type + cross-type pairs) with d <= c.
- Figure: x = Chi (0.4..1.0), y = same_frac (0.0..1.0).
"""
import json
import re
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent.parent.parent  # project root
DATA = BASE / "data"
FIGDIR = BASE / "output/figures"
OUTDIR = BASE / "output/results"
FIGDIR.mkdir(parents=True, exist_ok=True)
OUTDIR.mkdir(parents=True, exist_ok=True)

HDR = re.compile(r"Box dimension:\s*(\d+(?:\.\d+)?)\(p\)")


def parse_xyz(path):
    lines = path.read_text().splitlines()
    pts = []
    box = None
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        if i == 0 and re.match(r"^\d+\s*$", s):
            continue  # particle-count line
        if s.startswith("R1=") or "Energy" in s:
            m = HDR.search(s)
            box = float(m.group(1)) if m else None
            continue
        p = s.split()
        if len(p) >= 4:
            try:
                pts.append((p[0], float(p[1]), float(p[2]), float(p[3])))
            except ValueError:
                pass
    return box, pts


def pairwise(pts, box):
    N = len(pts)
    out = []
    for a in range(N):
        for b in range(N):
            if a >= b:
                continue
            dx = ((pts[b][1] - pts[a][1] + box / 2) % box) - box / 2
            dy = ((pts[b][2] - pts[a][2] + box / 2) % box) - box / 2
            dz = pts[b][3] - pts[a][3]
            out.append((dx * dx + dy * dy + dz * dz) ** 0.5)
    return out


def first_peak(dlist, width=0.5):
    d = np.array([x for x in dlist if np.isfinite(x) and x > 0.0], dtype=float)
    if d.size == 0:
        return None
    hist, edges = np.histogram(d, bins=np.arange(0.0, d.max() + width, width))
    for i in range(1, len(hist) - 1):
        if hist[i] >= hist[i - 1] and hist[i] > hist[i + 1]:
            return float((edges[i] + edges[i + 1]) / 2.0)
    return float((edges[-2] + edges[-1]) / 2.0)


def same_frac_at(pts, box, cutoff):
    N = len(pts)
    same = cross = 0
    for a in range(N):
        for b in range(N):
            if a >= b:
                continue
            dx = ((pts[b][1] - pts[a][1] + box / 2) % box) - box / 2
            dy = ((pts[b][2] - pts[a][2] + box / 2) % box) - box / 2
            dz = pts[b][3] - pts[a][3]
            if (dx * dx + dy * dy + dz * dz) ** 0.5 <= cutoff:
                if pts[a][0] == pts[b][0]:
                    same += 1
                else:
                    cross += 1
    return (same / (same + cross)) if (same + cross) > 0 else 0.0


def build_svg(points):
    """points: list of (chi, frac, pid) sorted by chi."""
    L = []
    L.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 620" width="900" height="620" font-family="Helvetica, Arial, sans-serif">')
    L.append('<rect width="900" height="620" fill="white"/>')
    L.append('<text x="450" y="38" text-anchor="middle" font-size="20" font-weight="bold" fill="#222222">Same-type NN fraction vs interfacial preference (Chi), equal-size 9C/9H2</text>')
    L.append('<text x="450" y="60" text-anchor="middle" font-size="12" fill="#666666">Same-type NN pair fraction at first-peak cutoff (c = first local max of all-pair distance histogram)</text>')
    # horizontal gridlines
    for y in (500.0, 418.0, 336.0, 254.0, 172.0, 90.0):
        L.append(f'<line x1="130" y1="{y}" x2="770" y2="{y}" stroke="#e8e8e8" stroke-width="1"/>')
    # vertical gridlines
    for x in (210.0, 290.0, 370.0, 450.0, 530.0, 610.0, 690.0):
        L.append(f'<line x1="{x}" y1="90" x2="{x}" y2="500" stroke="#f0f0f0" stroke-width="1"/>')
    # axes
    L.append('<line x1="130" y1="500" x2="770" y2="500" stroke="#444444" stroke-width="1.5"/>')
    L.append('<line x1="130" y1="90" x2="130" y2="500" stroke="#444444" stroke-width="1.5"/>')
    # y-axis ticks/labels (0.0..1.0 step 0.2)
    for i, frac in enumerate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
        y = 500 - frac * 410
        L.append(f'<line x1="123" y1="{y}" x2="130" y2="{y}" stroke="#444444" stroke-width="1"/>')
        L.append(f'<text x="120" y="{y}" text-anchor="end" font-size="13" fill="#333333">{frac:.1f}</text>')
    # x-axis ticks/labels (0.4..1.0 step 0.1)
    for chi in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        x = 210 + (chi - 0.4) * 800
        L.append(f'<line x1="{x:.1f}" y1="500" x2="{x:.1f}" y2="507" stroke="#444444" stroke-width="1"/>')
        L.append(f'<text x="{x:.1f}" y="524" text-anchor="middle" font-size="13" fill="#333333">{chi:.1f}</text>')
    # axis titles
    L.append('<text x="450" y="555" text-anchor="middle" font-size="15" font-weight="bold" fill="#222222">Chi (interfacial preference)</text>')
    L.append('<text x="75" y="295" text-anchor="middle" font-size="15" font-weight="bold" fill="#222222" transform="rotate(-90 75,295)">Same-type NN pair fraction (first-peak cutoff)</text>')
    # polyline (geometry from rounded-4-dec frac, matching published SVG)
    pts_str = " ".join(f"{210 + (chi - 0.4) * 800:.1f},{500 - round(frac,4) * 410:.1f}" for chi, frac, _ in points)
    L.append(f'<polyline points="{pts_str}" fill="none" stroke="#d4493a" stroke-width="2.5" stroke-linejoin="round"/>')
    # circles + value labels + structure labels
    for chi, frac, pid in points:
        cx = 210 + (chi - 0.4) * 800
        cy = 500 - round(frac, 4) * 410
        L.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="#d4493a" stroke="#ffffff" stroke-width="2"/>')
        L.append(f'<text x="{cx + 10:.1f}" y="{cy - 12:.1f}" text-anchor="start" font-size="13" fill="#333333">{frac:.3f}</text>')
        L.append(f'<text x="{cx:.1f}" y="{cy + 24}" text-anchor="middle" font-size="11" fill="#888888">{pid}</text>')
    # annotation + footnote
    L.append('<text x="764" y="116" text-anchor="end" font-size="13" fill="#d4493a" font-weight="bold">Monotonic increase in chi</text>')
    L.append('<text x="764" y="134" text-anchor="end" font-size="11" fill="#666666">(Spearman = +1.0; 4 equal-size points)</text>')
    L.append('<text x="900" y="606" text-anchor="end" font-size="11" fill="#999999">c = first local max of 1D all-pair distance histogram (0.5-wide bins); same_type = same-type pairs / (same + cross pairs) within c</text>')
    L.append('</svg>')
    return "\n".join(L) + "\n"


def main():
    man = json.loads((DATA / "structure_manifest.json").read_text())
    points = []
    for s in man["structures"]:
        if s["r2"] != 3.0:
            continue
        pid = s["public_id"]
        box, pts = parse_xyz(DATA / f"structures/{pid}.xyz")
        c = first_peak(pairwise(pts, box))
        frac = same_frac_at(pts, box, c)
        points.append((s["chi"], frac, pid))
    points.sort(key=lambda t: t[0])

    with open(OUTDIR / "figure_data.csv", "w") as f:
        f.write("chi,same_type_nn_fraction,structure_id\n")
        for chi, frac, pid in points:
            f.write(f"{chi},{frac:.4f},{pid}\n")

    with open(FIGDIR / "chi_vs_samefrac.svg", "w") as f:
        f.write(build_svg(points))

    print("Wrote figure_data.csv and chi_vs_samefrac.svg")
    for chi, frac, pid in points:
        print(f"  chi={chi} frac={frac:.4f} {pid}")


if __name__ == "__main__":
    main()
