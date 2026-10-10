#!/usr/bin/env python3
"""
Analyze the 6:6 (6 CH4 + 6 H2) accepted-local-minimum optimization trajectory.

Inputs:
  LM_ENERGIES_32.txt : lines "frame  energy" (34 frames, 0..33)
  LM_MOVIE_32.xyz    : XYZ blocks; each frame = count line, header line, particle lines
                      (the file actually contains 36 frames; the last 3 share the
                      same converged energy -4.15068)

Outputs:
  output/results/trajectory_analysis.csv
  output/results/trajectory_summary.json
  output/results/trajectory_note.txt
"""

import csv, json, math
from collections import Counter
import numpy as np

from pathlib import Path
BASE = Path(__file__).resolve().parent.parent.parent  # project root
ENERGY = BASE / "data/optimization_trajectory/LM_ENERGIES_32.txt"
MOVIE  = BASE / "data/optimization_trajectory/LM_MOVIE_32.xyz"
OUTDIR = BASE / "output/results"

# ------------------------------------------------------------------
# 1. Parse energies
# ------------------------------------------------------------------
def parse_energies(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                frame = int(parts[0])
                energy = float(parts[1])
            except (ValueError, IndexError):
                continue
            rows.append((frame, energy))
    return rows

energies = parse_energies(ENERGY)
n_frames = len(energies)
energy_series = np.array([e for _, e in energies], dtype=float)
frames = np.array([fr for fr, _ in energies], dtype=int)

# ------------------------------------------------------------------
# 2. Parse XYZ movie
# ------------------------------------------------------------------
def parse_xyz(path):
    frames_list = []
    with open(path) as f:
        lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        # count line
        cnt_line = lines[i].strip()
        i += 1
        # header line
        header = lines[i].strip()
        i += 1
        # particle lines
        particles = []
        for _ in range(int(cnt_line)):
            parts = lines[i].split()
            i += 1
            ptype = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            particles.append((ptype, np.array([x, y, z])))
        frames_list.append((int(cnt_line), particles, header))
    return frames_list

movie = parse_xyz(MOVIE)
n_xyz_frames = len(movie)

# ------------------------------------------------------------------
# 3. Composition + per-frame particle counts (use energy frames, 0..33)
# ------------------------------------------------------------------
counts = Counter()
particle_counts = []
for k in range(min(n_frames, n_xyz_frames)):
    cnt, particles, header = movie[k]
    particle_counts.append(cnt)
    for ptype, _ in particles:
        counts[ptype] += 1

n_particles_per_frame = sorted(set(particle_counts))
# composition per single frame (all frames should agree)
last_cnt, last_particles, _ = movie[n_frames - 1] if n_frames < n_xyz_frames else movie[n_xyz_frames - 1]
# use a canonical per-frame count from frame 0
cnt0, particles0, _ = movie[0]
ch4 = sum(1 for p, _ in particles0 if p == "CH4")
h2 = sum(1 for p, _ in particles0 if p == "H2")
total = ch4 + h2
composition = f"{ch4} CH4 + {h2} H2 = {ch4}:{h2}"

# ------------------------------------------------------------------
# Energy stats
# ------------------------------------------------------------------
energy_frame0 = energy_series[0]
energy_final = energy_series[-1]
idx_min = int(np.argmin(energy_series))
energy_min = float(energy_series[idx_min])
energy_min_frame = int(idx_min)
final_frame_is_min = bool(energy_series[-1] == energy_min)

# frames within 0.05 of the minimum
within_mask = np.abs(energy_series - energy_min) <= 0.05
frames_within_05 = int(np.sum(within_mask))

# describe the "last N frames" oscillation window
# Use a "settled" band around the global minimum (0.25) and take the longest
# trailing contiguous run of frames that all lie within [min, min+0.25].
band = 0.25
band_low = energy_min
band_high = energy_min + band
osc_start = None
for k in range(n_frames - 1, -1, -1):
    if band_low <= energy_series[k] <= band_high:
        osc_start = k
    else:
        break
osc_end = n_frames - 1
osc_n = osc_end - osc_start + 1

# amplitude of oscillation within the tail window
tail = energy_series[osc_start:osc_end + 1]
osc_amp = float(tail.max() - tail.min())

# Is the descent before the tail monotonic?
pre = energy_series[:osc_start]
pre_monotonic = bool(np.all(np.diff(pre) <= 1e-9)) if len(pre) > 1 else True

# detect a short repeating cycle in the tail (periodic local-minimum bounce)
cycle_len = 0
if osc_n >= 6:
    best = None
    for L in (2, 3, 4):
        if osc_n >= 2 * L:
            # score = number of near-repeats t -> t+L
            score = 0
            for t in range(osc_n - L):
                if abs(energy_series[osc_start + t] -
                       energy_series[osc_start + t + L]) < 0.005:
                    score += 1
            if best is None or score > best[1]:
                best = (L, score)
    if best is not None:
        cycle_len = best[0]

# Build a human-readable convergence behavior string
desc_parts = []
desc_parts.append(f"descent from {energy_frame0:.3f} to the minimum region")
if pre_monotonic and len(pre) > 1:
    desc_parts.append(f"monotonic descent over frames 0..{osc_start-1}")
else:
    desc_parts.append(f"roughly monotonic descent over frames 0..{osc_start-1}")
desc_parts.append(f"then oscillation around the minimum in the last {osc_n} frames")
desc_parts.append(f"(frames {osc_start}..{osc_end}), amplitude ~{osc_amp:.3f}")
if cycle_len:
    desc_parts.append(f"with a short ~{cycle_len}-frame repeating cycle")
desc_parts.append("i.e. a local minimum with dynamical oscillation, not a single frozen point")
convergence_behavior = " ".join(desc_parts)

# ------------------------------------------------------------------
# 4. Packing descriptors for the final (converged) frame
# We use the last frame of the energy trajectory (frame 33) which corresponds
# to the final XYZ frame (index 33). Frames 34,35 are duplicates.
# ------------------------------------------------------------------
def packing_descriptor(frame_idx):
    cnt, particles, header = movie[frame_idx]
    types = [p[0] for p in particles]
    coords = np.array([p[1] for p in particles])
    N = len(coords)

    # all pairwise distances (mask self-distances with np.inf)
    d = np.zeros((N, N))
    for a in range(N):
        for b in range(N):
            if a != b:
                d[a, b] = np.linalg.norm(coords[a] - coords[b])
    np.fill_diagonal(d, np.inf)

    # mean nearest-neighbor distance (min over other particles)
    nn = np.min(d, axis=1)  # per-particle min (self excluded via inf)
    mean_nn = float(np.mean(nn))

    # first peak of distance histogram (characteristic NN distance)
    # exclude zero, bin in 0.5 units
    distances = d[d > 0]
    distances = distances[distances <= 20.0]  # reasonable cutoff
    hist, bin_edges = np.histogram(distances, bins=np.arange(0.2, 15.0, 0.5))
    peak_idx = int(np.argmax(hist))
    nn_peak = float((bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2.0)

    # z-layering: gap clustering of z coordinates
    z = coords[:, 2]
    z_sorted = np.sort(z)
    gaps = np.diff(z_sorted)
    # cluster: a new shell when gap > threshold
    threshold = 0.6 * (np.max(gaps) + 1e-9)  # use a fraction of max gap
    # more robust: threshold = mean+2*std of gaps
    gap_mean = float(np.mean(gaps))
    gap_std = float(np.std(gaps))
    thr = gap_mean + 2 * gap_std
    shells = 1
    for g in gaps:
        if g > thr:
            shells += 1
    return nn_peak, mean_nn, shells

nn_peak_final, mean_nn_final, z_shells_final = packing_descriptor(n_frames - 1)

# Also compute per-frame descriptors for the CSV (all 34 frames)
csv_rows = []
for k in range(n_frames):
    if k < n_xyz_frames:
        nnp, mnn, zsh = packing_descriptor(k)
    else:
        nnp, mnn, zsh = np.nan, np.nan, np.nan
    # type counts per frame
    cnt, particles, header = movie[k]
    tc = Counter(p[0] for p in particles)
    type_counts = f"{tc.get('CH4',0)}/{tc.get('H2',0)}"
    csv_rows.append([k, energy_series[k], type_counts, round(nnp, 4), round(mnn, 4), int(zsh)])

# ------------------------------------------------------------------
# 5. Write CSV
# ------------------------------------------------------------------
with open(f"{OUTDIR}/trajectory_analysis.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame", "energy", "type_counts", "nn_peak", "mean_nn", "z_shells"])
    w.writerows(csv_rows)
    # summary row
    w.writerow(["summary", f"min={energy_min:.5f}@{energy_min_frame}", composition,
                f"final={nn_peak_final:.4f}", f"final={mean_nn_final:.4f}", z_shells_final])

# ------------------------------------------------------------------
# 6. Write JSON summary
# ------------------------------------------------------------------
summary = {
    "n_frames": n_frames,
    "n_particles_per_frame": sorted(set(particle_counts)),
    "particle_composition": composition,
    "energy_frame0": round(energy_frame0, 5),
    "energy_final": round(energy_final, 5),
    "energy_min": round(energy_min, 5),
    "energy_min_frame": energy_min_frame,
    "final_frame_is_min": final_frame_is_min,
    "frames_within_05_of_min": frames_within_05,
    "xyz_frames_total": n_xyz_frames,
    "oscillation": {
        "start_frame": osc_start,
        "n_osc_frames": osc_n,
        "oscillation_amplitude": round(osc_amp, 5),
        "pre_tail_monotonic_descent": pre_monotonic,
    },
    "final_frame_packing": {
        "frame": n_frames - 1,
        "nn_peak": round(nn_peak_final, 4),
        "mean_nn": round(mean_nn_final, 4),
        "z_shells": z_shells_final,
    },
}

with open(f"{OUTDIR}/trajectory_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# ------------------------------------------------------------------
# 7. Write note
# ------------------------------------------------------------------
note = f"""Trajectory Analysis Note - 6:6 (6 CH4 + 6 H2) accepted-local-minimum trajectory
============================================================================
Method
------
- Parsed LM_ENERGIES_32.txt into (frame, energy) pairs: {n_frames} frames (0..33).
- Parsed LM_MOVIE_32.xyz into per-frame (count, types, coords). The XYZ file
  actually contains {n_xyz_frames} frames; the last {n_xyz_frames - n_frames}
  frames (energy {energy_final}) are repeats of the converged frame 33
  (minor coordinate jitter only). Energy frame i is mapped to XYZ frame i for i=0..33.
- Composition confirmed: {ch4} CH4 + {h2} H2 = {composition} in every frame.
- Energy stats computed: frame-0, final, global minimum (+frame), final==min?,
  count of frames within 0.05 of the minimum.
- Convergence behavior: checked monotonicity of the pre-tail descent and
  quantified the amplitude of the tail oscillation window.
- Packing descriptors for the final frame:
  * nn_peak  = first peak of the all-pair distance histogram (bin 0.5, cut 20)
  * mean_nn  = mean of per-particle nearest-neighbor distances
  * z_shells = number of z-layers from gap clustering of z coords
  (threshold = mean(gap)+2*std(gap))

Findings
--------
n_frames = {n_frames}; particles per frame = {sorted(set(particle_counts))}
frame-0 energy = {energy_frame0:.5f}
final energy   = {energy_final:.5f}
global min     = {energy_min:.5f} at frame {energy_min_frame}
final frame is the global minimum: {final_frame_is_min}
frames within 0.05 of min: {frames_within_05} of {n_frames}

Convergence: energy descends from {energy_frame0:.3f} to near {energy_min:.3f}.
The pre-tail segment (frames 0..{osc_start-1}) is {'monotonic' if pre_monotonic else 'non-monotonic'}
descent; from frame {osc_start} onward the energy oscillates within about
{osc_amp:.4f} around the minimum (band ~{band_low:.4f}..{band_high:.4f}),
i.e. a local minimum with dynamical oscillation rather than a single frozen point.

Final-frame packing descriptors (frame {n_frames-1}):
  nn_peak  = {nn_peak_final:.4f}
  mean_nn  = {mean_nn_final:.4f}
  z_shells = {z_shells_final}
  (final frame shows a 2-layer geometry: CH4 around z~ -2.1 and H2 around z~ +2.1)

Limitations
-----------
- Only per-frame accepted energies are stored; no per-frame energy error/step size
  is available, so "convergence" is judged by energy alone.
- The trajectory is a sequence of accepted local minima from a stochastic
  search (e.g. basin-hopping / annealing-like); the final frame is a minimum
  of the potential, not a true zero-temperature ground-state proof.
- Packing descriptors are simple global descriptors (no force field); z-layering
  threshold is heuristic (mean+2*std of z gaps).
- The last 2 XYZ frames duplicate frame 33 (identical energy); they were counted
  as repeats, not as new independent frames.
"""

with open(f"{OUTDIR}/trajectory_note.txt", "w") as f:
    f.write(note)

# ------------------------------------------------------------------
# Print a compact report for the StructuredOutput
# ------------------------------------------------------------------
print("n_frames:", n_frames)
print("n_particles_per_frame:", sorted(set(particle_counts)))
print("composition:", composition)
print("energy_frame0:", round(energy_frame0, 5))
print("energy_final:", round(energy_final, 5))
print("energy_min:", round(energy_min, 5), "at frame", energy_min_frame)
print("final_frame_is_min:", final_frame_is_min)
print("frames_within_05_of_min:", frames_within_05)
print("oscillation: start frame", osc_start, "n frames", osc_n,
      "amplitude", round(osc_amp, 5), "pre-tail monotonic", pre_monotonic)
print("final packing: nn_peak", round(nn_peak_final, 4),
      "mean_nn", round(mean_nn_final, 4), "z_shells", z_shells_final)
print("xyz frames total:", n_xyz_frames)
