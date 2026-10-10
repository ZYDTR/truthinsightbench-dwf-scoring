"""Core structural analysis of the 14 low-energy binary nanoparticle configurations.

Analysis unit: one global low-energy configuration per parameter combination.
Outputs:
  output/results/structure_features.csv      - one row per structure
  output/results/structure_features.json     - machine-readable summary
  output/results/neighbor_analysis.csv       - neighbor statistics + cutoff perturbation
  output/results/trajectory_summary.csv      - trajectory frames/energies
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
    """Parse a simple .xyz file.
    Returns (header_line, particles, box).
    particles = list of (type, x, y, z). box from header (p) value."""
    lines = path.read_text().splitlines()
    particles = []
    header = None
    box = None
    for i, line in enumerate(lines):
        raw = line
        line = line.strip()
        if not line:
            continue
        if i == 0 and re.match(r"^\d+\s*$", line):
            continue  # particle-count line
        if line.startswith("R1=") or line.startswith("Box") or "Energy" in line:
            header = raw
            m = HEADER_BOX_RE.search(raw)
            if m:
                box = float(m.group(1))
            continue
        parts = line.split()
        if len(parts) >= 4 and re.match(r"^[A-Za-z0-9]+", parts[0]):
            try:
                particles.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
            except ValueError:
                pass
    return (header, particles, box)


def parse_energy_file(path):
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                rows.append((int(parts[0]), float(parts[1])))
            except ValueError:
                pass
    return rows


def parse_movie(path):
    frames = []
    cur = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"^\d+\s*$", line):
            if cur is not None:
                frames.append(cur)
            cur = {"count": int(line), "particles": [], "header": None, "energy": None}
        elif cur is not None and line.startswith("R1="):
            cur["header"] = line
            m = re.search(r"Energy\s*=\s*([-\d.]+)", line)
            if m:
                cur["energy"] = float(m.group(1))
        elif cur is not None:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    cur["particles"].append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
                except ValueError:
                    pass
    if cur is not None:
        frames.append(cur)
    return frames


def periodic_distance(a, b, box):
    d = np.asarray([b - a])
    d = np.array([np.minimum(np.abs(c), box - np.abs(c)) for c in d]).flatten()
    return math.sqrt(float(np.sum(d**2)))


def all_pair_distances(pos, box):
    n = len(pos)
    d = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            d[i, j] = periodic_distance(pos[i], pos[j], box) if i != j else np.inf
    return d


def layering_z(pos, box):
    """Count distinct z-shells via periodic gap clustering.
    Returns (n_layers, z_shell_centers, intra_spread)."""
    z = np.asarray([p[2] for p in pos], dtype=float)
    z = (z + box / 2) % box - box / 2
    z_sorted = np.sort(z)
    gaps = np.diff(z_sorted)
    gap_wrap = box - (z_sorted[-1] - z_sorted[0])
    gap_list = np.concatenate([gaps, [gap_wrap]])
    med_gap = float(np.median(gap_list))
    thresh = 1.5 * med_gap
    n_layers = int(np.sum(gap_list > thresh))
    if n_layers == 0:
        n_layers = 1
    # shell centers: cluster by gap
    centers = []
    cur = [z_sorted[0]]
    for g, zval in zip(gap_list, z_sorted[1:]):
        if g > thresh:
            centers.append(float(np.mean(cur)))
            cur = [zval]
        else:
            cur.append(zval)
    centers.append(float(np.mean(cur)))
    return n_layers, centers, float(np.std(z))


def inplane_metrics(pos, box):
    p = np.asarray([p[:3] for p in pos], dtype=float)
    xy = p[:, :2]
    n = len(xy)
    cx, cy = float(xy.mean(axis=0)[0]), float(xy.mean(axis=0)[1])
    r = np.sqrt(((xy - [cx, cy]) ** 2).sum(axis=1))
    d = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                d[i, j] = np.inf
            else:
                dx = xy[j, 0] - xy[i, 0]
                dy = xy[j, 1] - xy[i, 1]
                dx = ((dx + box / 2) % box) - box / 2
                dy = ((dy + box / 2) % box) - box / 2
                d[i, j] = math.sqrt(dx * dx + dy * dy)
    nn_d = np.min(d, axis=1)
    angles = np.zeros(n)
    for i in range(n):
        j = int(np.argmin(d[i]))
        dx = ((xy[j, 0] - xy[i, 0] + box / 2) % box) - box / 2
        dy = ((xy[j, 1] - xy[i, 1] + box / 2) % box) - box / 2
        angles[i] = np.arctan2(dy, dx)
    if n:
        cv = float(np.abs(np.mean(np.exp(2j * np.pi * (angles / (2 * np.pi) - 0)))))
    else:
        cv = float("nan")
    return {
        "nn_dist_mean": float(np.mean(nn_d)),
        "nn_dist_std": float(np.std(nn_d)),
        "nn_angle_cv": cv,
        "xy_rms_from_centroid": float(np.std(r)),
    }


def main():
    man = json.loads((DATA / "structure_manifest.json").read_text())
    rows = []
    for s in man["structures"]:
        pid = s["public_id"]
        header, particles, box = parse_xyz(DATA / "structures" / (pid + ".xyz"))
        n = len(particles)
        box = box if box is not None else float(s["box"])
        types = [p[0] for p in particles]
        counts = {}
        for t in types:
            counts[t] = counts.get(t, 0) + 1
        pos = np.asarray([[p[1], p[2], p[3]] for p in particles], dtype=float)
        n_layers, centers, z_spread = layering_z(pos, box)
        inplane = inplane_metrics(pos, box)
        nd = all_pair_distances(pos, box)
        nn_d = np.sort(nd, axis=1)[:, 0]
        rows.append({
            "public_id": pid,
            "n_particles": n,
            "n_C": counts.get("C", 0),
            "n_H2": counts.get("H2", 0),
            "R1": s["r1"], "R2": s["r2"], "chi": s["chi"], "epsilon": s["epsilon"],
            "energy": s["energy"],
            "energy_per_particle": s["energy"] / n if n else float("nan"),
            "box": box,
            "n_layers_z": n_layers,
            "z_spread": z_spread,
            "nn_dist_mean": float(np.mean(nn_d)),
            "nn_dist_std": float(np.std(nn_d)),
            "xy_rms": inplane["xy_rms_from_centroid"],
            "nn_angle_cv": inplane["nn_angle_cv"],
            "z_shells": json.dumps(centers),
        })
    rows = sorted(rows, key=lambda r: r["public_id"])
    keys = sorted({k for r in rows for k in r})
    with open(OUT / "structure_features.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    with open(OUT / "structure_features.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("Wrote structure_features.csv/json:", len(rows), "rows")

    # neighbor analysis with cutoff perturbation
    cutoffs = [3.0, 3.5, 4.0]
    cut_rows = []
    for s in man["structures"]:
        pid = s["public_id"]
        _, particles, box = parse_xyz(DATA / "structures" / (pid + ".xyz"))
        box = box if box is not None else float(s["box"])
        pos = np.asarray([[p[1], p[2], p[3]] for p in particles], dtype=float)
        types = [p[0] for p in particles]
        nd = all_pair_distances(pos, box)
        n = len(pos)
        for c in cutoffs:
            nn_per = float(np.mean(np.sum(nd < c, axis=1)))
            same = cross = 0
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if nd[i, j] < c:
                        if types[i] == types[j]:
                            same += 1
                        else:
                            cross += 1
            total = same + cross
            same_frac = same / total if total > 0 else float("nan")
            cut_rows.append({
                "public_id": pid, "cutoff": c,
                "mean_neighbors_per_particle": nn_per,
                "same_type_pair_frac": same_frac,
                "total_pairs": total,
            })
    keys = sorted({k for r in cut_rows for k in r})
    with open(OUT / "neighbor_analysis.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in cut_rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    print("Wrote neighbor_analysis.csv:", len(cut_rows), "rows")

    # trajectory
    energies = parse_energy_file(DATA / "optimization_trajectory" / "LM_ENERGIES_32.txt")
    frames = parse_movie(DATA / "optimization_trajectory" / "LM_MOVIE_32.xyz")
    with open(OUT / "trajectory_summary.csv", "w") as f:
        f.write("frame,accepted_energy,movie_count,movie_header\n")
        for i, (frame_idx, e) in enumerate(energies):
            frow = frames[i] if i < len(frames) else {}
            f.write(f"{frame_idx},{e},{frow.get('count','')},{frow.get('header','')}\n")
    min_e = min(e for _, e in energies)
    final_e = energies[-1][1]
    plateau = [e for _, e in energies if e > min_e - 0.05]
    print(f"Trajectory: min={min_e:.5f} final={final_e:.5f} frames_within_0.05={len(plateau)}")

if __name__ == "__main__":
    main()
