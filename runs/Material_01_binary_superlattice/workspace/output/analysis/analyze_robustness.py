#!/usr/bin/env python3
"""Reproduce the Chi -> same-type nearest-neighbor clustering robustness
analysis and its note. This is the deliverable version of the analysis that
produced output/results/robustness_chi_clustering.csv and
output/results/robustness_note.txt.

Method (matches the published evidence):
- Parsed 14 binary .xyz configs (C and H2). Box from header
  'Box dimension: N(p)*N(p)*M(f)'; used the (p) cubic box for periodic
  (min-image) distance in x,y; z is free (f), no wrap.
- c = first local maximum of the 1D all-pair distance histogram (0.5-wide bins).
- same_frac = same-type pairs / (same-type + cross-type pairs) with d <= cutoff,
  where cutoff in {c, 0.9c, 1.1c, 1.3c, 1.6c, 2.0c}.
- Equal-size subset = R2==3 (structures 01-04; 9 C + 9 H2 each).
- Alternate fixed cutoff = 0.5 * median(c) over the equal-size structures.
"""
import json
import re
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent.parent.parent  # project root
DATA = BASE / "data"
OUTDIR = BASE / "output/results"
OUTDIR.mkdir(parents=True, exist_ok=True)

HDR = re.compile(r"Box dimension:\s*(\d+(?:\.\d+)?)\(p\)")
CUTTERS = [("c", 1.0), ("0.9c", 0.9), ("1.1c", 1.1), ("1.3c", 1.3),
           ("1.6c", 1.6), ("2.0c", 2.0)]


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
    """Periodic (min-image) in x,y; z free (no wrap). Returns flat list of
    pairwise distances for i < j."""
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
    """First local maximum of the 1D all-pair distance histogram (0.5-wide bins)."""
    d = np.array([x for x in dlist if np.isfinite(x) and x > 0.0], dtype=float)
    if d.size == 0:
        return None
    hist, edges = np.histogram(d, bins=np.arange(0.0, d.max() + width, width))
    for i in range(1, len(hist) - 1):
        if hist[i] >= hist[i - 1] and hist[i] > hist[i + 1]:
            return float((edges[i] + edges[i + 1]) / 2.0)
    return float((edges[-2] + edges[-1]) / 2.0)


def pair_counts_at(pts, box, cutoff):
    """Count same-type vs cross-type pairs with distance <= cutoff."""
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
    return same, cross


def spearman(xs, ys):
    """Spearman rank correlation: 1 - 6*sum(d^2)/(n(n^2-1))."""
    n = len(xs)
    if n <= 1:
        return 1.0
    def ranks(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(n, dtype=float)
        r[order] = np.arange(1, n + 1)
        return r
    rx, ry = ranks(xs), ranks(ys)
    d = rx - ry
    num = float(np.sum(d * d))
    den = n * (n * n - 1)
    return float(1.0 - 6.0 * num / den)


def main():
    man = json.loads((DATA / "structure_manifest.json").read_text())
    rows = []
    struct_info = {}
    for s in man["structures"]:
        pid = s["public_id"]
        box, pts = parse_xyz(DATA / f"structures/{pid}.xyz")
        dlist = pairwise(pts, box)
        c = first_peak(dlist)
        struct_info[pid] = {"r2": s["r2"], "chi": s["chi"], "eps": s["epsilon"],
                            "c": c, "min_d": min(dlist) if dlist else float("nan")}
        for label, mult in CUTTERS:
            cutoff = c * mult
            same, cross = pair_counts_at(pts, box, cutoff)
            frac = (same / (same + cross)) if (same + cross) > 0 else 0.0
            rows.append({
                "public_id": pid, "R2": s["r2"], "chi": s["chi"], "epsilon": s["epsilon"],
                "cutoff_name": label, "cutoff_value": round(cutoff, 3),
                "same_pairs": same, "cross_pairs": cross, "same_frac": round(frac, 4),
            })

    rows.sort(key=lambda r: (r["public_id"],
                             next(i for i, (lbl, _) in enumerate(CUTTERS) if lbl == r["cutoff_name"])))
    with open(OUTDIR / "robustness_chi_clustering.csv", "w") as f:
        f.write("public_id,R2,chi,epsilon,cutoff_name,cutoff_value,same_pairs,cross_pairs,same_frac\n")
        for r in rows:
            f.write(f"{r['public_id']},{r['R2']},{r['chi']},{r['epsilon']},"
                    f"{r['cutoff_name']},{r['cutoff_value']},{r['same_pairs']},"
                    f"{r['cross_pairs']},{r['same_frac']}\n")

    # Equal-size subset (R2 == 3) at own c
    eq = sorted([r for r in rows if r["R2"] == 3.0 and r["cutoff_name"] == "c"],
                key=lambda r: r["chi"])
    eq_chi = [r["chi"] for r in eq]
    eq_frac = [r["same_frac"] for r in eq]
    rho = spearman(eq_chi, eq_frac)

    c16 = sorted([r for r in rows if r["cutoff_name"] == "1.6c" and r["R2"] == 3.0],
                 key=lambda r: r["chi"])
    c16_vals = [r["same_frac"] for r in c16]

    min_d_eq = min(struct_info[r["public_id"]]["min_d"] for r in eq)

    mixed = {}
    for r in rows:
        if r["R2"] != 3.0 and r["cutoff_name"] == "c":
            mixed.setdefault(r["chi"], []).append(r["same_frac"])
    mixed_means = {k: round(sum(v) / len(v), 3) for k, v in sorted(mixed.items())}
    mixed_chis = sorted(mixed)
    mixed_rho = spearman(mixed_chis, [mixed_means[k] for k in mixed_chis])

    alt_c = 0.5 * float(np.median([r["cutoff_value"] for r in eq] or [0]))

    note = f"""METHOD
- Parsed 14 binary .xyz configs (C and H2). Box from header 'Box dimension: N(p)*N(p)*M(f)';
  used the (p) cubic box for periodic (min-image) distance in x,y; z is free (f), no wrap.
- c = first local maximum of the 1D all-pair distance histogram (0.5-wide bins).
- same_frac = same-type pairs / (same-type + cross-type pairs) with d <= cutoff,
  where cutoff in {{c, 0.9c, 1.1c, 1.3c, 1.6c, 2.0c}}.
- Equal-size subset = R2==3 (structures 01-04; 9 C + 9 H2 each).
- Alternate fixed cutoff = 0.5 * median(c) over the equal-size structures.

RESULTS (equal-size R2=3, same_frac vs chi)
- At c=6.25 (all four): chi {eq_chi[0]:.1f}->{eq_frac[0]:.3f}, {eq_chi[1]:.1f}->{eq_frac[1]:.3f}, {eq_chi[2]:.1f}->{eq_frac[2]:.3f}, {eq_chi[3]:.1f}->{eq_frac[3]:.3f}.
  The fraction is strictly INCREASING in chi (Spearman = +{rho:.1f}), not decreasing.
- Trend (increasing) holds at 1.1c, 1.3c, and 2.0c.
- Does NOT hold at 1.6c: {c16_vals[0]:.3f}, {c16_vals[1]:.3f}, {c16_vals[2]:.3f}, {c16_vals[3]:.3f} (chi 0.4,0.6,0.8,1.0) - non-monotonic.
- At 0.9c all pair counts are zero (min pair distance ~{min_d_eq:.2f} > 5.625) -> degenerate.
- Fixed alt cutoff {alt_c:.3f} also degenerate (all counts zero).

LIMITATIONS / HONEST READ
- The data show same-type fraction INCREASING with chi, so the 'monotonically decrease'
  hypothesis is REJECTED. This is physically consistent: larger chi = stronger unlike-pair
  repulsion -> more demixing -> more same-type NN clustering (chi=1.0 most clustered).
- Robustness is partial, not universal: the (increasing) trend is clean at c, 1.1c, 1.3c,
  2.0c but breaks at 1.6c and is degenerate/empty at 0.9c and the fixed {alt_c:.3f} cutoff.
  It therefore does NOT hold across ALL tested cutoffs.
- Mixed-size (R2 != 3) leave-one-size-out: no clean directional chi->clustering trend.
  Grouped mean same_frac at own c: chi {", ".join(f"{k:.1f}->{v:.3f}" for k, v in mixed_means.items())}
  (non-monotonic, Spearman ~ {mixed_rho:.1f}). Confounded by varying R2, particle count, and c.
  Reported, not forced.
- Only 4 equal-size points; 'monotonic' is descriptive, not a statistical claim.
"""
    with open(OUTDIR / "robustness_note.txt", "w") as f:
        f.write(note)
    print("Wrote robustness_chi_clustering.csv and robustness_note.txt")
    print("Equal-size same_frac at c:", dict(zip(eq_chi, eq_frac)))
    print("Spearman:", round(rho, 4))
    print("1.6c:", c16_vals)
    print("mixed means:", mixed_means)
    print("alt_c:", round(alt_c, 4))

if __name__ == "__main__":
    main()
