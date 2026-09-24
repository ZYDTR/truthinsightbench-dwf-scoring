#!/usr/bin/env python3
"""Self-contained per-file DNP processor.

Usage: python3 process_one.py <input_mat> <output_json>

Processes a single raw .mat file (EPR echo, NMR DNP-contact, or NMR-only
reference) and writes a per-file metrics JSON to <output_json>. This is the
unit of parallelism: each file's processing is independent, so N such calls
can be run concurrently.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import scipy.io as sio
from scipy.signal import hilbert

DET_PHASES = np.array([0.0, 0.0, 90.0, 90.0, 180.0, 180.0, 270.0, 270.0]) * np.pi / 180.0


def rfft_safe(x):
    """rfft via fft (numpy's rfft_n_even is broken in this environment)."""
    f = np.fft.fft(x.astype(np.complex128))
    return f[: len(f) // 2 + 1]


def log_decay_tau(env, min_frac=0.02):
    """Exponential-decay time constant (points) from a log-linear fit."""
    env = np.asarray(env, dtype=float)
    m = env > env.max() * min_frac
    if m.sum() < 4:
        return float("nan")
    tt = np.arange(len(env))[m].astype(float)
    yy = np.log(env[m])
    A = np.vstack([tt, np.ones(len(tt))]).T
    coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
    if coef[0] >= 0 or not np.isfinite(coef[0]):
        return float("nan")
    return 1.0 / (-coef[0])


def process_epr(path):
    d = sio.loadmat(path, squeeze_me=True)
    x = d["dta_001"].astype(np.float64)          # (1024, 150)
    trace = x.mean(axis=1)                       # 1024 points
    n = len(trace)
    f = rfft_safe(trace)
    freqs = np.fft.rfftfreq(n, d=1.0)
    mag = np.abs(f)
    # Strongest non-DC line: skip DC (index 0) and take the strongest remaining.
    # Use descending magnitude order, skipping index 0 (DC) if present.
    top = np.argsort(mag)[::-1]
    f0_idx = top[0] if top[0] != 0 else top[1]
    f0 = freqs[f0_idx]          # carrier (Larmor) frequency, pts/sample
    t = np.arange(n)
    baseband = trace * np.exp(-2j * np.pi * f0 * t)
    env = np.abs(baseband)
    thr = env.max() * 0.10
    idx = np.where(env > thr)[0]
    ws, we = int(idx[0]), int(idx[-1])
    return {
        "f0_pts_per_sample": float(f0),
        "echo_start": ws,
        "echo_end": we,
        "echo_len": we - ws + 1,
        "echo_area_env": float(env[ws : we + 1].sum()),
        "echo_area_signed": float(baseband.real[ws : we + 1].sum()),
        "echo_peak": float(env.max()),
        "echo_tau_pts": log_decay_tau(env[ws : we + 1]),
        "n_avgs": int(x.shape[1]),
        "points": int(n),
    }


def process_nmr(path, is_ref):
    d = sio.loadmat(path, squeeze_me=True)
    n_ind = sum(1 for k in d if k.startswith("nmrdta_"))
    rows = np.stack([d[f"nmrdta_{i}"] for i in range(1, n_ind + 1)], axis=0)
    rows = rows.astype(np.complex128)             # (n_ind, 8, 256)
    comb = (rows * np.exp(-1j * DET_PHASES[None, :, None])).sum(axis=1)  # (n_ind, 256)
    comb_dc = comb - comb.mean(axis=1, keepdims=True)
    env = np.abs(comb_dc)                        # (n_ind, 256)
    total = float(env.sum())
    early = float(env[:, :10].sum())
    late = float(env[:, -10:].sum())
    mid = float(env[:, 10:-10].sum()) if env.shape[1] > 20 else float("nan")
    taus = []
    for i in range(n_ind):
        tau = log_decay_tau(env[i], min_frac=0.05)
        if np.isfinite(tau):
            taus.append(tau)
    return {
        "n_indirect": n_ind,
        "points_direct": 256,
        "total_area": total,
        "early_area": early,
        "late_area": late,
        "mid_area": mid,
        "early_fraction": early / total if total else float("nan"),
        "late_fraction": late / total if total else float("nan"),
        "fid_tau_pts_mean": float(np.mean(taus)) if taus else float("nan"),
        "per_indirect_peak": [float(v) for v in env.max(axis=1)],
        "is_reference": is_ref,
    }


def main():
    path, out = sys.argv[1], sys.argv[2]
    name = path.split("/")[-1]
    if "dnp_echo" in name:
        res = {"file": name, "type": "epr", **process_epr(path)}
    elif "nmr_only" in name:
        res = {"file": name, "type": "nmr", **process_nmr(path, True)}
    else:
        res = {"file": name, "type": "nmr", **process_nmr(path, False)}
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print("wrote", out, "| type=", res["type"], "| keys=", list(res.keys()))


if __name__ == "__main__":
    main()
