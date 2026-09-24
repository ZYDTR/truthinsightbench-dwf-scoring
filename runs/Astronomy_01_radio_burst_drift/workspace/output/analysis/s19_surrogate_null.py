#!/usr/bin/env python3
"""
Dimension tag: s19  (ridge-free surrogate null for the decisive slope test)

Task: TruthInsightBench Astronomy_01_radio_burst_drift (T0 = 2022-10-13)
Analysis unit: ONE COMPLETE DYNAMIC-SPECTRUM EVENT (n_events = 5).

The slope-scan null of s18 is an internal (image-as-its-own-null) comparison: it
asks whether the reported slope is extreme AMONG slopes in the same image, not
whether the image contains any coherent straight drift at all.  s19 supplies the
missing external null.

Surrogate construction: each frequency row of the real 425x425 image is rolled
along the time axis by an INDEPENDENT uniform random lag.  This preserves exactly
the marginal distribution of every row (and of the whole grid), preserves the
row-to-row nearest-neighbour texture in frequency, and destroys any time-coherent
straight-line structure.  The surrogate therefore answers "what would the scan
statistic be if there were no coherent drift ridge in this image, with all other
gross properties held fixed?".

For each event and 25 surrogate draws we compute the same coverage-matched
corner-anchored statistic L(m) (s18 definition, band +/- 0.35 MHz, threshold
1.05) and report:
  * p_slope : fraction of surrogate L(m_reported) >= real L(m_reported)
  * p_any   : fraction of surrogates whose max over ALL slopes >= the real max
              (is there any coherent straight ridge at all?)
  * p_len   : same, restricted to slopes whose in-band time coverage is within
              +-30% of the reported slope's coverage (a ridge of the
              reported length)
"""
import gc
import json
import os
import traceback

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

ROOT = "/data/user/conversations/conv-02a7c52b-6f08-461b-b10b-65392ab66dba/project"
NV = os.path.join(ROOT, "data", "normalized_views")
RES = os.path.join(ROOT, "output", "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

THRESH = 1.05
BAND_HW = 0.35
SLOPES = np.linspace(-40.0, 40.0, 401)
N_SURR = 25
SEED = 20221013
MATCH_TOL = 0.30
ERRORS = []


def log(m=""):
    print(m, flush=True)


def corner_curve(S_asc, f_asc, t, slopes, band):
    """Coverage-matched corner-anchored line statistic L(m); fixed denominator."""
    nF, nT = S_asc.shape
    fmin, fmax = float(f_asc[0]), float(f_asc[-1])
    dfq = float(f_asc[1] - f_asc[0])
    x0 = float(t[0])
    num = np.vstack([np.zeros((1, nT)), np.cumsum(S_asc, axis=0)])
    j = np.arange(nT)
    L = np.full(slopes.size, np.nan)
    cov = np.full(slopes.size, np.nan)
    for k, m in enumerate(slopes):
        a = fmax if m < 0 else fmin
        fl = a + m * (t - x0)
        ok = (fl >= fmin) & (fl <= fmax)
        if not ok.any():
            continue
        ia = np.clip(np.searchsorted(f_asc, fl - band, "left"), 0, nF)
        ib = np.clip(np.searchsorted(f_asc, fl + band, "right"), 0, nF)
        nb = np.maximum(ib - ia, 1)
        v = (num[ib, j] - num[ia, j]) / nb
        L[k] = float(v[ok].mean())
        cov[k] = float(ok.mean())
    return L, cov


def main():
    meta = pd.read_csv(os.path.join(NV, "dynamic_spectrum_event_metadata.tsv"), sep="\t")
    log("=" * 90)
    log("s19  RIDGE-FREE SURROGATE NULL (row-rolled in time), threshold 1.05")
    log(f"ANALYSIS UNIT = one complete dynamic-spectrum event (n=5); {N_SURR} surrogates/event")
    log("=" * 90)

    rows = []
    fig_curves = {}
    for _, mr in meta.iterrows():
        eid = mr["event_id"]
        i = int(eid.split("_")[1])
        rep = float(mr["reported_event_drift_rate_MHz_s"])
        df = pd.read_csv(os.path.join(NV, f"event_{i}_dynamic_spectrum.tsv"), sep="\t")
        f = df.iloc[:, 0].to_numpy(float)
        t = np.array([float(s.replace("time_s_", "")) for s in df.columns[1:]], float)
        Z = df.iloc[:, 1:].to_numpy(float)
        del df
        f_asc = f[::-1].copy()
        fmin, fmax = float(f.min()), float(f.max())
        Df, Dt, t0 = fmax - fmin, float(t[-1] - t[0]), float(t[0])
        nT = t.size

        S = np.where(Z > THRESH, Z - 1.0, 0.0)
        L_real, cov = corner_curve(S[::-1, :], f_asc, t, SLOPES, BAND_HW)
        krep = int(np.argmin(np.abs(SLOPES - rep)))
        Lr = float(L_real[krep])
        cov_rep = float(cov[krep])
        length = np.minimum(1.0, Df / (np.abs(SLOPES) * Dt + 1e-12))
        sel_len = np.abs(length - cov_rep) <= MATCH_TOL * cov_rep
        real_max_all = float(np.nanmax(L_real))
        real_max_len = float(np.nanmax(np.where(sel_len, L_real, -np.inf)))
        real_argmax_all = float(SLOPES[int(np.nanargmax(L_real))])
        real_argmax_len = float(SLOPES[int(np.argmax(np.where(sel_len, L_real, -np.inf)))])

        rng = np.random.default_rng(SEED + i)
        Ls_rep = np.empty(N_SURR)
        Ls_allmax = np.empty(N_SURR)
        Ls_lenmax = np.empty(N_SURR)
        Ls_curves = np.empty((N_SURR, SLOPES.size))
        for s_i in range(N_SURR):
            Zs = np.empty_like(Z)
            for r in range(Z.shape[0]):
                Zs[r] = np.roll(Z[r], int(rng.integers(1, nT)))
            Ss = np.where(Zs > THRESH, Zs - 1.0, 0.0)
            Ls, _ = corner_curve(Ss[::-1, :], f_asc, t, SLOPES, BAND_HW)
            Ls_curves[s_i] = Ls
            Ls_rep[s_i] = Ls[krep]
            Ls_allmax[s_i] = np.nanmax(Ls)
            Ls_lenmax[s_i] = np.nanmax(np.where(sel_len, Ls, -np.inf))
            del Zs, Ss
        gc.collect()

        p_slope = float((1 + np.sum(Ls_rep >= Lr)) / (1 + N_SURR))
        p_any = float((1 + np.sum(Ls_allmax >= real_max_all)) / (1 + N_SURR))
        p_len = float((1 + np.sum(Ls_lenmax >= real_max_len)) / (1 + N_SURR))
        rows.append(dict(
            event_id=eid, reported_drift=rep, threshold=THRESH,
            L_real_reported=Lr, L_surrogate_reported_mean=float(Ls_rep.mean()),
            L_surrogate_reported_p95=float(np.percentile(Ls_rep, 95)),
            p_slope=p_slope,
            real_max_all=real_max_all, surrog_allmax_mean=float(Ls_allmax.mean()),
            surrog_allmax_p95=float(np.percentile(Ls_allmax, 95)),
            real_argmax_all=real_argmax_all, p_any=p_any,
            real_max_lenmatched=real_max_len, real_argmax_lenmatched=real_argmax_len,
            surrog_lenmax_mean=float(Ls_lenmax.mean()),
            surrog_lenmax_p95=float(np.percentile(Ls_lenmax, 95)),
            p_len=p_len, n_surrogates=N_SURR,
            cov_reported=cov_rep))
        log(f"### {eid}: reported {rep:+.4f} MHz/s")
        log(f"   real L(reported) = {Lr:.5f}; surrogate mean {Ls_rep.mean():.5f},"
            f" p95 {np.percentile(Ls_rep,95):.5f} -> p_slope = {p_slope:.3f}")
        log(f"   real max_m L = {real_max_all:.5f} at {real_argmax_all:+.2f} MHz/s;"
            f" surrogate max mean {Ls_allmax.mean():.5f}, p95"
            f" {np.percentile(Ls_allmax,95):.5f} -> p_any = {p_any:.3f}")
        log(f"   length-matched (cov {100*cov_rep:.1f}%): real max {real_max_len:.5f} at"
            f" {real_argmax_len:+.2f} MHz/s; surrogate mean {Ls_lenmax.mean():.5f},"
            f" p95 {np.percentile(Ls_lenmax,95):.5f} -> p_len = {p_len:.3f}")
        fig_curves[eid] = (L_real, Ls_curves, rep, cov_rep)
        del Z, S, L_real, Ls_curves
        gc.collect()

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(RES, "s19_surrogate_null.csv"), index=False)
    log()
    log("summary:")
    log(R[["event_id", "reported_drift", "p_slope", "p_any", "p_len"]].to_string(index=False))
    with open(os.path.join(RES, "s19_surrogate_summary.json"), "w") as fh:
        json.dump(dict(
            analysis_unit="one complete dynamic-spectrum event (n=5)",
            surrogate="independent random time-roll of every frequency row; preserves all marginals",
            n_surrogates=N_SURR, seed=SEED, threshold=THRESH,
            band_half_width_MHz=BAND_HW, slopes=[float(SLOPES[0]), float(SLOPES[-1]), int(SLOPES.size)],
            results=json.loads(R.to_json(orient="records")),
            n_events_rejected_p_any_0p05=int((R.p_any <= 0.05).sum()),
            n_events_rejected_p_len_0p05=int((R.p_len <= 0.05).sum()),
            n_events_rejected_p_slope_0p05=int((R.p_slope <= 0.05).sum()),
            errors=ERRORS), fh, indent=1)

    # figure: real L(m) vs surrogate envelope
    fig, axes = plt.subplots(1, 5, figsize=(27, 5.6), dpi=125, sharex=True)
    for ax, e in zip(axes, list(fig_curves.keys())):
        L_real, Ls, rep, cov_rep = fig_curves[e]
        lo = np.nanpercentile(Ls, 5, axis=0)
        hi = np.nanpercentile(Ls, 95, axis=0)
        ax.fill_between(SLOPES, lo, hi, color="gray", alpha=0.35,
                        label="surrogate 5-95% band")
        ax.plot(SLOPES, np.nanmean(Ls, axis=0), color="gray", lw=1.0,
                label="surrogate mean")
        ax.plot(SLOPES, L_real, color="C3", lw=1.8, label="REAL image")
        ax.axvline(rep, color="k", ls="--", lw=2, label=f"reported {rep:+.2f}")
        row = R[R.event_id == e].iloc[0]
        ax.set_title(f"{e}\np_slope={row.p_slope:.2f}  p_any={row.p_any:.2f}"
                     f"  p_len={row.p_len:.2f}", fontsize=10)
        ax.set_xlabel("slope (MHz/s)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("L(m) band-averaged line intensity")
    fig.suptitle("s19 real image vs ridge-free (row-rolled) surrogate null, "
                 "threshold 1.05; unit = one event (n=5)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG, "s19_surrogate_null.png"))
    plt.close(fig)
    log("s19 saved s19_surrogate_null.csv/.json + s19_surrogate_null.png")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        ERRORS.append(traceback.format_exc())
        log("FATAL:\n" + traceback.format_exc())
        raise