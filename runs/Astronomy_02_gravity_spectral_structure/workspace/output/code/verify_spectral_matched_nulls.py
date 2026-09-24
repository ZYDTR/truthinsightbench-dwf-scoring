#!/usr/bin/env python3
"""Matched-statistic nulls for the TRACK 2 stacked-peak claims.

All nulls here evaluate the SAME statistic as the observed quantity:
    stacked(f) = (1/22) * sum_p P_p(f),   obs = max_f stacked(f)
so mean-of-maxima vs max-of-means mismatch is removed.

Also:
 - circular-shift null computed with the matched statistic (using a random subset
   of shifts, which for a Gaussian-ish construction samples the same distribution
   as the full orbit; track used 300 draws from an orbit of <=80 per pass).
 - a valid red-noise null for the pass-peak support statistic (searching globally).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

PROJ = Path("/data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project")
MODELS = {"A": "residual_model_A_mm_s", "B": "residual_model_B_mm_s", "C": "residual_model_C_mm_s"}
RES_ELEM = 210.9704641350211


def my_ls(t_sec, Y, f_hz):
    t_sec = np.asarray(t_sec, float); Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    n = Y.shape[0]
    Yc = Y - Y.mean(axis=0, keepdims=True); var = np.mean(Yc * Yc, axis=0)
    F = np.asarray(f_hz, float).size
    out = np.empty((Y.shape[1], F))
    for lo in range(0, F, 700):
        hi = min(lo + 700, F)
        w = 2.0 * np.pi * f_hz[lo:hi]; wt = np.outer(w, t_sec)
        c, s = np.cos(wt), np.sin(wt)
        CC = (c * c).mean(1); SS = (s * s).mean(1); CS = (c * s).mean(1)
        tau = 0.5 * np.arctan2(2 * CS, CC - SS)
        c2 = np.cos(wt - tau[:, None]); s2 = np.sin(wt - tau[:, None])
        cc = (c2 * c2).mean(1); ss = (s2 * s2).mean(1)
        YC = (Yc.T @ c2.T) / n; YS = (Yc.T @ s2.T) / n
        out[:, lo:hi] = (YC * YC / cc[None, :] + YS * YS / ss[None, :]) / var[:, None]
    return out


def ar1_fit(y):
    y = np.asarray(y, float); y = y - y.mean()
    return float(np.clip(np.dot(y[:-1], y[1:]) / np.dot(y[:-1], y[:-1]), -0.99, 0.99)), float(np.std(y))


def ar1_gen(n, a, s, rng):
    e = rng.normal(0.0, s * np.sqrt(max(1 - a * a, 1e-12)), n)
    return lfilter([1.0], [1.0, -a], e)


def main():
    t0 = time.time()
    tr = pd.read_csv(PROJ / "data/pass_residual_traces.tsv", sep="\t")
    passes = sorted(tr.pass_id.unique())
    TS = {p: tr.loc[tr.pass_id == p, "relative_time_min"].to_numpy() for p in passes}
    YS = {p: {k: tr.loc[tr.pass_id == p, c].to_numpy() for k, c in MODELS.items()} for p in passes}
    f_uhz = np.arange(200.0, 3000.0, 1.0)
    f_hz = f_uhz * 1e-6
    R = 300
    rng = np.random.default_rng(4242)
    out = {}

    for k in MODELS:
        obs_stack = None
        for p in passes:
            P = my_ls(TS[p] * 60.0, YS[p][k], f_hz)[0]
            obs_stack = P if obs_stack is None else obs_stack + P
        obs_stack /= len(passes)
        obs_peak = float(obs_stack.max())

        ar1_peak = np.empty(R)
        circ_peak = np.empty(R)
        for r in range(R):
            acc = np.zeros(f_hz.size)
            accc = np.zeros(f_hz.size)
            for p in passes:
                y = YS[p][k]; n = len(y)
                a, s = ar1_fit(y)
                ys = ar1_gen(n, a, s, rng)
                acc += my_ls(TS[p] * 60.0, ys, f_hz)[0]
                sh = np.roll(y, int(rng.integers(1, n)))
                accc += my_ls(TS[p] * 60.0, sh, f_hz)[0]
            acc /= len(passes); accc /= len(passes)
            ar1_peak[r] = acc.max()
            circ_peak[r] = accc.max()
            if r % 50 == 0:
                print(f"  {k} {r}/{R} obs={obs_peak:.4f} ar1_p95={np.percentile(ar1_peak[:r+1],95):.4f}", flush=True)
        out[k] = {
            "observed_stacked_peak": obs_peak,
            "matched_AR1_null_q50_q95_q99_max": [float(np.percentile(ar1_peak, q)) for q in (50, 95, 99, 100)],
            "p_matched_AR1": float((ar1_peak >= obs_peak).mean()),
            "matched_circshift_null_q50_q95_q99_max": [float(np.percentile(circ_peak, q)) for q in (50, 95, 99, 100)],
            "p_matched_circshift": float((circ_peak >= obs_peak).mean()),
            "n_draws": R,
        }
        print(k, json.dumps(out[k], indent=1), flush=True)

    # ---- proper red-noise null for the pass-peak support statistic
    supp = {}
    for k in MODELS:
        obs_stack = None
        peaks = []
        for p in passes:
            P = my_ls(TS[p] * 60.0, YS[p][k], f_hz)[0]
            obs_stack = P if obs_stack is None else obs_stack + P
            peaks.append(float(f_uhz[int(np.argmax(P))]))
        obs_stack /= len(passes)
        fpk = float(f_uhz[int(np.argmax(obs_stack))])
        obs_count = int(np.sum(np.abs(np.array(peaks) - fpk) <= RES_ELEM))
        counts = []
        ND = 40
        for r in range(60):
            pk = []
            for p in passes:
                y = YS[p][k]; a, s = ar1_fit(y)
                S = np.column_stack([ar1_gen(len(y), a, s, rng) for _ in range(ND)])
                P = my_ls(TS[p] * 60.0, S, f_hz)
                pk.append(f_uhz[np.argmax(P, axis=1)])
            pk = np.concatenate(pk)
            counts.append(int(np.sum(np.abs(pk - fpk) <= RES_ELEM)) / (len(passes) * ND) * len(passes))
        counts = np.array(counts)
        supp[k] = {
            "observed_count_of_22": obs_count,
            "ar1_global_search_null_mean_count_per_22": float(counts.mean()),
            "ar1_global_search_null_q2.5_q50_q97.5": [float(np.percentile(counts, q)) for q in (2.5, 50, 97.5)],
            "frac_null_ge_observed": float((counts >= obs_count).mean()),
            "naive_uniform_count": 22 * min(1.0, 2 * RES_ELEM / 2800.0),
            "kludge_track_used_count": 22 * min(1.0, 2 * RES_ELEM / (3000.0 - 200.0)),
        }
        print("support", k, json.dumps(supp[k]), flush=True)
    out["pass_peak_support"] = supp
    out["runtime_sec"] = round(time.time() - t0, 1)
    (PROJ / "output/results/_verify_spectral_matched_nulls.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()