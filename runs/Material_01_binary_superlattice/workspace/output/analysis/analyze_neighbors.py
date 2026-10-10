"""Neighbor / composition analysis with physically-motivated cutoffs.

For each structure we compute the full periodic pairwise distance matrix,
locate the first distance peak (the characteristic "bond/nearest-neighbor"
distance), and count same-type (C-C or H2-H2) vs cross-type (C-H2) pairs at
that distance and at +/-20% perturbation. This tests whether interfacial
preference (Chi) promotes same-type clustering.
"""
import json
import re
import math
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parent.parent.parent / "data"
OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
HEADER_BOX_RE = re.compile(r"Box dimension:\s*(\d+(?:\.\d+)?)\(p\)")


def parse_xyz(path):
    lines = path.read_text().splitlines()
    particles, header, box = [], None, None
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if i == 0 and re.match(r"^\d+\s*$", line):
            continue
        if line.startswith("R1=") or line.startswith("Box") or "Energy" in line:
            header = line
            m = HEADER_BOX_RE.search(line)
            if m:
                box = float(m.group(1))
            continue
        parts = line.split()
        if len(parts) >= 4 and re.match(r"^[A-Za-z0-9]+", parts[0]):
            try:
                particles.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
            except ValueError:
                pass
    return header, particles, box


def all_pair_distances(pos, box):
    n = len(pos)
    d = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                d[i, j] = np.inf
            else:
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dz = pos[j, 2] - pos[i, 2]
                dx = ((dx + box / 2) % box) - box / 2
                dy = ((dy + box / 2) % box) - box / 2
                dz = ((dz + box / 2) % box) - box / 2
                d[i, j] = math.sqrt(dx * dx + dy * dy + dz * dz)
    return d


def first_peak(distances):
    """Return the first local peak in the 1D distance distribution."""
    finite = np.array([x for x in distances if np.isfinite(x)])
    finite = finite[finite > 0.001]
    if finite.size == 0:
        return None
    # histogram, 0.5 A bins
    bins = np.arange(0, finite.max() + 1, 0.5)
    counts, edges = np.histogram(finite, bins=bins)
    peak = int(np.argmax(counts))
    while peak + 1 < len(counts) and counts[peak + 1] >= counts[peak]:
        peak += 1
    return float((edges[peak] + edges[peak + 1]) / 2)


def pair_types(types, i, j):
    if types[i] == types[j]:
        return "same"
    return "cross"


def main():
    man = json.loads((DATA / "structure_manifest.json").read_text())
    rows = []
    for s in man["structures"]:
        pid = s["public_id"]
        _, particles, box = parse_xyz(DATA / "structures" / (pid + ".xyz"))
        box = box if box is not None else float(s["box"])
        pos = np.asarray([[p[1], p[2], p[3]] for p in particles], dtype=float)
        types = [p[0] for p in particles]
        nd = all_pair_distances(pos, box)
        # upper triangle distances
        dists = []
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                dists.append(nd[i, j])
        peak = first_peak(dists)
        if peak is None:
            rows.append({"public_id": pid, "first_peak": None, "n_particles": len(pos)})
            continue
        # evaluate at peak and +/-20%
        out = []
        for frac, name in [(0.8, "peak_-20%"), (1.0, "peak"), (1.2, "peak_+20%")]:
            c = peak * frac
            same = cross = 0
            for i in range(len(pos)):
                for j in range(i + 1, len(pos)):
                    if nd[i, j] < c:
                        t = pair_types(types, i, j)
                        if t == "same":
                            same += 1
                        else:
                            cross += 1
            total = same + cross
            frac_same = same / total if total > 0 else float("nan")
            out.append({"cutoff": round(c, 3), "mean_neighbors": 2 * same / len(pos) + 2 * cross / len(pos),
                        "same_pairs": same, "cross_pairs": cross, "same_frac": frac_same})
        flat = {"public_id": pid, "first_peak": round(peak, 3), "n_particles": len(pos),
                "n_C": types.count("C"), "n_H2": types.count("H2"),
                "R2": s["r2"], "chi": s["chi"], "epsilon": s["epsilon"],
                "R2_over_R1": round(s["r2"] / s["r1"], 4),
                "same_frac_-20": out[0]["same_frac"], "same_frac_0": out[1]["same_frac"],
                "same_frac_+20": out[2]["same_frac"],
                "cross_frac_0": (1.0 - out[1]["same_frac"]) if np.isfinite(out[1]["same_frac"]) else float("nan"),
                "same_pairs_0": out[1]["same_pairs"], "cross_pairs_0": out[1]["cross_pairs"],
                "cutoff_peak": round(peak, 3)}
        rows.append(flat)
    rows = [r for r in rows if "first_peak" in r]
    rows = sorted(rows, key=lambda r: r["public_id"])
    keys = sorted({k for r in rows for k in r})
    with open(OUT / "neighbor_analysis.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    print("Wrote neighbor_analysis.csv:", len(rows), "rows")
    for r in rows:
        print(f"{r['public_id']}: peak={r['first_peak']} R2={r['R2']} chi={r['chi']} "
              f"same_frac_-20={r['same_frac_-20']} same_frac_0={r['same_frac_0']} "
              f"same_frac_+20={r['same_frac_+20']}")

if __name__ == "__main__":
    main()
