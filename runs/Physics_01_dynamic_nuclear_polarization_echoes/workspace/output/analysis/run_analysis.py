#!/usr/bin/env python3
"""
Reproducible analysis for Physics_01_dynamic_nuclear_polarization_echoes.

Task: "Electron and Nuclear Spin Dynamics under Phase Inversion".
Phase-inversion times (ns): 0 (control), 132, 220, 440, 880, 2200.

Two readouts are processed independently:
  * EPR  (data/raw/*_dnp_echo_pp_length_2.mat)  -> echo integration + signed-signal reconstruction.
  * NMR  (data/raw/*_dnpdata_dnp_contact_swp.mat, *_nmr_only.mat)
         -> 8-step phase cycling + reference phase + normalization.

For each phase-inversion time we estimate the temporal structure of the
electron- and nuclear-spin readout, then compare whether their relationship
is stable, offset, condition-dependent, or indeterminate.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parent.parent          # .../project/output
PROJECT = ROOT.parent                                   # .../project
DATA = PROJECT                                           # data paths are relative to project
OUT = ROOT / "results"
FIG = ROOT / "figures"
for d in (OUT, FIG):
    d.mkdir(parents=True, exist_ok=True)

INV_NS = [0, 132, 220, 440, 880, 2200]

# ---------------------------------------------------------------------------
# File maps (from comparison_manifest.json; control = 0 ns)
# ---------------------------------------------------------------------------
EPR = {
    0:    "data/raw/20240522_125527_dnp_echo_pp_length_2.mat",
    132:  "data/raw/20240522_130805_dnp_echo_pp_length_2.mat",
    220:  "data/raw/20240522_130914_dnp_echo_pp_length_2.mat",
    440:  "data/raw/20240522_131018_dnp_echo_pp_length_2.mat",
    880:  "data/raw/20240522_131142_dnp_echo_pp_length_2.mat",
    2200: "data/raw/20240522_131404_dnp_echo_pp_length_2.mat",
}
NMR = {
    0:    "data/raw/20240523_1308_dnpdata_dnp_contact_swp.mat",
    132:  "data/raw/20240522_1404_dnpdata_dnp_contact_swp.mat",
    220:  "data/raw/20240522_1434_dnpdata_dnp_contact_swp.mat",
    440:  "data/raw/20240522_1505_dnpdata_dnp_contact_swp.mat",
    880:  "data/raw/20240522_1535_dnpdata_dnp_contact_swp.mat",
    2200: "data/raw/20240523_0954_dnpdata_dnp_contact_swp.mat",
}
NMR_REF = "data/raw/20240424_1547_dnpdata_nmr_only.mat"   # phase/normalization reference

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def rfft_safe(x):
    """rfft with a working numpy backend (rfft_n_even is broken here)."""
    f = np.fft.fft(x.astype(np.complex128))
    return f[: len(f) // 2 + 1]

def envelope(x):
    """Analytic-signal envelope of a real trace."""
    return np.abs(hilbert(np.asarray(x, dtype=float)))


def log_decay_tau(env, min_frac=0.02):
    """Exponential-decay time constant (in points) from a log-linear fit
    of the envelope over the region above min_frac of its max."""
    env = np.asarray(env, dtype=float)
    m = env > env.max() * min_frac
    if m.sum() < 4:
        return np.nan
    tt = np.arange(len(env))[m].astype(float)
    yy = np.log(env[m])
    A = np.vstack([tt, np.ones(len(tt))]).T
    coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
    if coef[0] >= 0 or not np.isfinite(coef[0]):
        return np.nan
    return 1.0 / (-coef[0])


# ---------------------------------------------------------------------------
# EPR processing
# ---------------------------------------------------------------------------
def process_epr(path):
    d = sio.loadmat(path, squeeze_me=True)
    x = d["dta_001"].astype(np.float64)          # (1024, 150) = points x avgs
    trace = x.mean(axis=1)                       # 1024-point averaged trace
    n = len(trace)

    # carrier (Larmor) frequency from the trace FFT (skip DC)
    f = rfft_safe(trace)
    freqs = np.fft.rfftfreq(n, d=1.0)
    mag = np.abs(f)
    top = np.argsort(mag)[::-1]
    f0 = freqs[top[1]]                          # strongest non-DC line

    t = np.arange(n)
    baseband = trace * np.exp(-2j * np.pi * f0 * t)
    env = np.abs(baseband)
    real_sig = baseband.real                     # signed baseband signal

    # echo window: where envelope exceeds 10% of its max
    thr = env.max() * 0.10
    idx = np.where(env > thr)[0]
    ws, we = int(idx[0]), int(idx[-1])

    area_env = float(env[ws : we + 1].sum())
    area_signed = float(real_sig[ws : we + 1].sum())
    peak = float(env.max())
    tau = log_decay_tau(env[ws : we + 1])

    return {
        "f0_pts_per_sample": float(f0),
        "echo_start": ws,
        "echo_end": we,
        "echo_len": we - ws + 1,
        "echo_area_env": area_env,
        "echo_area_signed": area_signed,
        "echo_peak": peak,
        "echo_tau_pts": tau,
        "n_avgs": int(x.shape[1]),
        "points": int(n),
        "trace": trace,
        "env": env,
        "ws": ws,
        "we": we,
    }


# ---------------------------------------------------------------------------
# NMR processing
# ---------------------------------------------------------------------------
DET_PHASES = np.array([0.0, 0.0, 90.0, 90.0, 180.0, 180.0, 270.0, 270.0]) * np.pi / 180.0

def process_nmr(path, sign=-1.0):
    """Combine the 8-phase steps. sign=-1 -> exp(-i phi), sign=+1 -> exp(+i phi)."""
    d = sio.loadmat(path, squeeze_me=True)
    n_ind = sum(1 for k in d if k.startswith("nmrdta_"))
    rows = np.stack([d[f"nmrdta_{i}"] for i in range(1, n_ind + 1)], axis=0)
    rows = rows.astype(np.complex128)           # (n_ind, 8, 256)

    # phase-cycle recombination -> (n_ind, 256)
    comb = (rows * np.exp(sign * 1j * DET_PHASES[None, :, None])).sum(axis=1)
    # remove per-indirect-point DC / baseline offset
    comb_dc = comb - comb.mean(axis=1, keepdims=True)

    env = np.abs(comb_dc)                       # (n_ind, 256)
    t_dir = np.arange(256)

    total = float(env.sum())
    per_indirect = env.max(axis=1)
    early = float(env[:, :10].sum())
    late = float(env[:, -10:].sum())
    mid = float(env[:, 10:-10].sum()) if env.shape[1] > 20 else float("nan")

    # FID decay over the direct axis (per indirect point, then average)
    taus = []
    for i in range(n_ind):
        tau = log_decay_tau(env[i], min_frac=0.05)
        if np.isfinite(tau):
            taus.append(tau)
    tau_mean = float(np.mean(taus)) if taus else np.nan

    return {
        "n_indirect": n_ind,
        "points_direct": 256,
        "total_area": total,
        "early_area": early,
        "late_area": late,
        "mid_area": mid,
        "early_fraction": early / total if total else np.nan,
        "late_fraction": late / total if total else np.nan,
        "per_indirect_peak": per_indirect,
        "fid_tau_pts_mean": tau_mean,
        "env": env,
        "comb_dc": comb_dc,
    }


def process_nmr_ref(path):
    """Reference scan: used for phase reference and normalization scale."""
    d = sio.loadmat(path, squeeze_me=True)
    n_ind = sum(1 for k in d if k.startswith("nmrdta_"))
    rows = np.stack([d[f"nmrdta_{i}"] for i in range(1, n_ind + 1)], axis=0).astype(np.complex128)
    comb = (rows * np.exp(-1j * DET_PHASES[None, :, None])).sum(axis=1)
    comb_dc = comb - comb.mean(axis=1, keepdims=True)
    env = np.abs(comb_dc)
    return {
        "n_indirect": n_ind,
        "total_area": float(env.sum()),
        "env": env,
        "first_indirect": comb_dc[0],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epr_rows, nmr_rows = [], []
    epr_full = {}
    nmr_full = {}

    print("=== EPR ===")
    for ns in INV_NS:
        r = process_epr(DATA / EPR[ns])
        r["inv_ns"] = ns
        epr_rows.append({k: v for k, v in r.items() if k not in ("trace", "env")})
        epr_full[ns] = r
        print(f" inv={ns:5d} f0={r['f0_pts_per_sample']:.5f} ws={r['ws']} we={r['we']} "
              f"area_env={r['echo_area_env']:.0f} area_signed={r['echo_area_signed']:.0f} "
              f"peak={r['echo_peak']:.0f} tau={r['echo_tau_pts']:.1f}")

    print("\n=== NMR ===")
    ref = process_nmr_ref(DATA / NMR_REF)
    print(f" reference (nmr_only): n_ind={ref['n_indirect']} total={ref['total_area']:.0f}")
    for ns in INV_NS:
        r = process_nmr(DATA / NMR[ns], sign=-1.0)
        r["inv_ns"] = ns
        nmr_rows.append({k: v for k, v in r.items()
                     if k not in ("env", "comb_dc", "per_indirect_peak")})
        # also keep per-indirect peak (as a list for JSON)
        nmr_rows[-1]["per_indirect_peak"] = r["per_indirect_peak"].tolist()
        nmr_full[ns] = r
        print(f" inv={ns:5d} n_ind={r['n_indirect']} total={r['total_area']:.0f} "
              f"early_frac={r['early_fraction']:.3f} late_frac={r['late_fraction']:.3f} "
              f"fid_tau={r['fid_tau_pts_mean']:.1f}")

    # robustness: alternative NMR phase sign
    nmr_alt = []
    for ns in INV_NS:
        r = process_nmr(DATA / NMR[ns], sign=+1.0)
        nmr_alt.append({"inv_ns": ns, "total_area": float(r["env"].sum()),
                        "early_frac": float(r["env"][:, :10].sum() / r["env"].sum())})
    print("\n=== NMR robustness (exp(+i phi)) ===")
    for a in nmr_alt:
        print(f" inv={a['inv_ns']:5d} total={a['total_area']:.0f} early_frac={a['early_frac']:.3f}")

    # correlation of EPR echo-area and NMR total-area vs inversion time
    inv = np.array(INV_NS, dtype=float)
    epr_areas = np.array([epr_full[ns]["echo_area_env"] for ns in INV_NS])
    nmr_areas = np.array([nmr_full[ns]["total_area"] for ns in INV_NS])
    corr_epr = float(np.corrcoef(epr_areas, inv)[0, 1])
    corr_nmr = float(np.corrcoef(nmr_areas, inv)[0, 1])
    corr_epr_nmr = float(np.corrcoef(epr_areas, nmr_areas)[0, 1])
    print(f"\n corr(EPR_area, inv)={corr_epr:.4f}  corr(NMR_area, inv)={corr_nmr:.4f}")
    print(f" corr(EPR_area, NMR_area)={corr_epr_nmr:.4f}")

    # Robustness: fixed inner echo window [190,790] for EPR
    ws2, we2 = 190, 790
    epr_alt = []
    for ns in INV_NS:
        r = epr_full[ns]
        epr_alt.append({"inv_ns": ns,
                        "fixed_window_area_env": float(r["env"][ws2:we2 + 1].sum())})
    print("\n=== EPR robustness (fixed inner window [190,790]) ===")
    for a in epr_alt:
        print(f" inv={a['inv_ns']:5d} area={a['fixed_window_area_env']:.0f}")

    # save CSV / JSON
    import csv
    with open(OUT / "epr_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(epr_rows[0].keys()))
        w.writeheader()
        w.writerows(epr_rows)
    with open(OUT / "nmr_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(nmr_rows[0].keys()))
        w.writeheader()
        w.writerows(nmr_rows)

    summary = {
        "task_id": "Physics_01_dynamic_nuclear_polarization_echoes",
        "t0": "2024-06-26",
        "inversion_times_ns": INV_NS,
        "epr": epr_rows,
        "nmr": nmr_rows,
        "nmr_robustness_exp_posi": nmr_alt,
        "correlations": {
            "epr_area_vs_inv": corr_epr,
            "nmr_area_vs_inv": corr_nmr,
            "epr_area_vs_nmr_area": corr_epr_nmr,
        },
        "nmr_reference_total_area": float(ref["total_area"]),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\nWrote epr_metrics.csv, nmr_metrics.csv, summary.json")

    # ---- figures ----
    inv = np.array(INV_NS, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(inv, epr_areas, "o-", color="C0", label="integrated echo envelope area")
    ax.plot(inv, [epr_full[ns]["echo_peak"] for ns in INV_NS], "s-", color="C1",
            label="echo peak envelope")
    ax.set_xlabel("phase-inversion time (ns)")
    ax.set_ylabel("EPR echo signal")
    ax.set_title("EPR echo vs phase-inversion time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(FIG / "epr_echo.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(inv, nmr_areas, "o-", color="C2")
    ax2 = ax.twinx()
    ax2.plot(inv, [nmr_full[ns]["early_fraction"] for ns in INV_NS], "s-", color="C3")
    ax.set_xlabel("phase-inversion time (ns)")
    ax.set_ylabel("NMR integrated area", color="C2")
    ax2.set_ylabel("early-indirect fraction", color="C3")
    ax2.legend(loc="upper right")
    ax.set_title("NMR integrated area + early-indirect fraction vs phase-inversion time")
    ax.grid(True, alpha=0.3)
    fig.savefig(FIG / "nmr_structure.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(inv, epr_areas / epr_areas[0], "o-", color="C0", label="EPR echo (norm/0)")
    ax.plot(inv, nmr_areas / nmr_areas[0], "s-", color="C2", label="NMR area (norm/0)")
    ax.set_xlabel("phase-inversion time (ns)")
    ax.set_ylabel("normalized signal")
    ax.axhline(1, color="k", lw=0.5, ls="--")
    ax.set_title("Normalized EPR vs NMR signal vs phase-inversion time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(FIG / "comparison.png", dpi=150)
    plt.close(fig)
    print("Wrote epr_echo.png, nmr_structure.png, comparison.png")
    return summary


if __name__ == "__main__":
    main()
