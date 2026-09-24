#!/usr/bin/env python3
"""Proper chance null for TRACK 2's "pass-peak support" statistic.

TRACK 2 compares n_passes_peaking_within_1_res_elem_of_stacked_peak against
22 * 2*R_res/(3000-200) = 3.32 (a UNIFORM-frequency expectation).  Under red
noise the per-pass global-max frequency is NOT uniform, so we compute the chance
expectation under AR(1) draws with each pass's own fitted (a, s), using the
track's own stacking/support recipe.
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
    fu = np.arange(200.0, 3000.0, 1.0); f_hz = fu * 1e-6
    rng = np.random.default_rng(54321)
    ND = 100
    out = {}
    for k in MODELS:
        obs_stack = None; peaks = []
        for p in passes:
            P = my_ls(TS[p] * 60.0, YS[p][k], f_hz)[0]
            obs_stack = P if obs_stack is None else obs_stack + P
            peaks.append(float(fu[int(np.argmax(P))]))
        obs_stack /= len(passes)
        fpk = float(fu[int(np.argmax(obs_stack))])
        obs_near = int(np.sum(np.abs(np.array(peaks) - fpk) <= RES_ELEM))
        # AR(1) full-pipeline null: for each surrogate stacked spectrum, find its
        # peak and count how many of that set's 22 pass peaks lie within 1 res.
        counts = []
        for j in range(ND):
            St = None; pk = []
            for p in passes:
                y = YS[p][k]
                a, s = ar1_fit(y)
                ys = ar1_gen(len(y), a, s, rng)
                P = my_ls(TS[p] * 60.0, ys, f_hz)[0]
                St = P if St is None else St + P
                pk.append(float(fu[int(np.argmax(P))]))
            St /= len(passes)
            fj = float(fu[int(np.argmax(St))])
            counts.append(int(np.sum(np.abs(np.array(pk) - fj) <= RES_ELEM)))
        counts = np.array(counts)
        out[k] = {
            "observed_near_count_of_22": obs_near,
            "observed_stacked_peak_uhz": fpk,
            "ar1_null_mean_count_of_22": float(counts.mean()),
            "ar1_null_q2.5_q50_q97.5": [float(np.percentile(counts, q)) for q in (2.5, 50, 97.5)],
            "ar1_null_frac_ge_observed": float((counts >= obs_near).mean()),
            "track_uniform_chance_expectation": 22 * 2 * RES_ELEM / 2800.0,
            "n_null_stacked_realisations": ND,
        }
        print(k, json.dumps(out[k]), flush=True)
    out["runtime_sec"] = round(time.time() - t0, 1)
    (PROJ / "output/results/_verify_spectral_support_null.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()