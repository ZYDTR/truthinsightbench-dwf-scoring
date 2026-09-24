#!/usr/bin/env python3
"""S04 / dimension "robustness+falsification": shared helpers.

Defines
  * the exact dynamic-spectrum loader (grid constants re-derived from the files)
  * three INDEPENDENT event drift-rate estimators, each validated on synthetic data
  * the source-block decomposition of the six `drift_regime` series

Every constant here is re-derived from the data files, never hard-coded from the
prompt, so the harness re-verifies what it relies on.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "normalized_views"
RESULTS = ROOT / "output" / "results"
FIGDIR = RESULTS / "figures"
LOGDIR = ROOT / "output" / "logs"

EVENT_IDS = ["event_1", "event_2", "event_3", "event_4", "event_5"]


# ----------------------------------------------------------------------------
# dynamic spectra
# ----------------------------------------------------------------------------
def load_event(i: int):
    """Return (freq_MHz descending, time_s ascending, value[425,425])."""
    df = pd.read_csv(DATA / f"event_{i}_dynamic_spectrum.tsv", sep="\t")
    freq = df.iloc[:, 0].to_numpy(float)
    tcol = np.array([float(c.split("_")[-1]) for c in df.columns[1:]])
    V = df.iloc[:, 1:].to_numpy(float)
    assert np.all(np.diff(freq) < 0), "frequency axis not descending"
    assert np.all(np.diff(tcol) > 0), "time axis not ascending"
    return freq, tcol, V


def grid_constants(i: int):
    freq, tcol, _ = load_event(i)
    dfreq = float(np.abs(np.diff(freq)).mean())
    dt = float(np.diff(tcol).mean())
    return freq, tcol, dfreq, dt


# ----------------------------------------------------------------------------
# estimator A: 2-D FFT striation-orientation ridge
# ----------------------------------------------------------------------------
def fft_slope(V, freq, band, dfreq, dt, th=None, dgrid=None):
    """Mean power per FFT coefficient whose (ky,kx) implies a given drift slope.

    A plane wave exp(2*pi*i*(ky*r + kx*c)) lies along ridges r = m*c + b with
    m = -kx/ky (rows per column).  Converting: d[MHz/s] = m*dfreq/dt.

    `th` implements the signal threshold: pixels <= th are zeroed first, i.e. the
    transform is computed on the thresholded, background-subtracted image.
    """
    rows = np.where((freq >= band[0]) & (freq <= band[1]))[0]
    A = V[rows].astype(float)
    if th is not None:
        A = np.where(A > th, A - th, 0.0)
    A = A - A.mean(axis=1, keepdims=True)
    A = A - A.mean(axis=0, keepdims=True)
    A = A * np.hanning(A.shape[0])[:, None] * np.hanning(A.shape[1])[None, :]
    nr, nc = A.shape
    P = np.abs(np.fft.fft2(A)) ** 2
    ky = np.fft.fftfreq(nr)
    kx = np.fft.fftfreq(nc)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    with np.errstate(divide="ignore", invalid="ignore"):
        m = -KX / KY
        d = m * dfreq / dt
    good = (
        (np.abs(KY) > 1e-9)
        & (np.abs(KX) > 1e-9)
        & (np.abs(m) <= 8.0)
        & (np.abs(m) >= 0.02)
        & (np.abs(KY) >= 4.0 / nr)
        & (np.abs(KX) >= 4.0 / nc)
    )
    out = np.full(len(dgrid), np.nan)
    for k, dd in enumerate(dgrid):
        mm = dd * dt / dfreq
        sel = good & (np.abs(m - mm) < 0.05)
        n = int(sel.sum())
        if n:
            out[k] = P[sel].mean()
    return out, good, P, d


def fft_ridge_peaks(prof, dgrid, rel=0.55, nmax=5):
    """Local maxima of the slope profile, strongest first (relative to its max)."""
    pk = [j for j in range(1, len(prof) - 1) if prof[j] > prof[j - 1] and prof[j] >= prof[j + 1]]
    pk = [j for j in pk if np.isfinite(prof[j]) and prof[j] >= rel * np.nanmax(prof)]
    pk.sort(key=lambda j: -prof[j])
    return [(float(dgrid[j]), float(prof[j])) for j in pk[:nmax]]


# ----------------------------------------------------------------------------
# estimator B: brightness-weighted band-centroid slope (OLS on time)
# ----------------------------------------------------------------------------
def centroid_slope(V, freq, band, tcol, th):
    """OLS slope of the weighted frequency centroid of everything above `th`."""
    rows = np.where((freq >= band[0]) & (freq <= band[1]))[0]
    W = V[np.ix_(rows, np.arange(len(tcol)))] - th
    W = np.where(W > 0, W, 0.0)
    s = W.sum(axis=0)
    ok = s > 0
    if ok.sum() < 10:
        return np.nan, 0
    cen = (W[:, ok] * freq[rows][:, None]).sum(axis=0) / s[ok]
    return float(np.polyfit(tcol[ok], cen, 1)[0]), int(ok.sum())


# ----------------------------------------------------------------------------
# estimator C: de-dispersion (coherent-sum) peak
# ----------------------------------------------------------------------------
def dedisp_peak(V, freq, band, dfreq, dt, dgrid):
    """Sum of background-subtracted power after shearing by candidate drift d."""
    S = V - 1.0
    nrow, ncol = V.shape
    bandrows = np.where((freq >= band[0]) & (freq <= band[1]))[0]
    jj = np.broadcast_to(np.arange(ncol)[None, :], (len(bandrows), ncol))
    out = np.zeros(len(dgrid))
    for k, d in enumerate(dgrid):
        rr = bandrows[:, None] + (-d * (jj * dt) / dfreq)
        ok = (rr >= 0) & (rr < nrow)
        r0 = np.clip(np.round(rr).astype(int), 0, nrow - 1)
        out[k] = S[r0, jj][ok].sum()
    return out


# ----------------------------------------------------------------------------
# source-block decomposition of the six drift_regime series
# ----------------------------------------------------------------------------
def series_blocks():
    """Split each (series, regime) into contiguous source blocks.

    Within a block the bin centre spacing is constant; a NEGATIVE jump in the
    centre sequence marks a new source block (the normalised view concatenated
    two workbook blocks of different width, of which six sheets are known:
    Sheet_024/026/028/030/032/034 hold '(0<s<20)' plus '(-10<s<0)').
    """
    d = pd.read_csv(DATA / "drift_population_series.tsv", sep="\t")
    recs = []
    for (pid, reg), g in d.groupby(["population_series_id", "drift_regime"], sort=True):
        g = g.reset_index(drop=True)
        c = g["distribution_bin_center"].to_numpy(float)
        y = g["event_count"].to_numpy(float)
        cut = np.where(np.diff(c) < 0)[0] + 1
        starts = np.concatenate([[0], cut])
        ends = np.concatenate([cut, [len(c)]])
        for bi, (s, e) in enumerate(zip(starts, ends)):
            cc, yy = c[s:e], y[s:e]
            dx = float(np.median(np.diff(cc))) if len(cc) > 1 else np.nan
            recs.append(
                dict(
                    population_series_id=pid,
                    anonymous_source_group=g["anonymous_source_group"].iloc[0],
                    polarization_channel=g["polarization_channel"].iloc[0],
                    drift_regime=reg,
                    block_index=int(bi),
                    n_bins=len(cc),
                    dx=dx,
                    lo=float(cc[0]),
                    hi=float(cc[-1]),
                    events=float(yy.sum()),
                    counts=yy,
                    centers=cc,
                )
            )
    return recs


def rebin(blocks, edges):
    """Overlap-proportional re-bin of source bins onto a common edge grid.

    Each source bin is treated as the interval [c-dx/2, c+dx/2] and its count is
    split across target bins in proportion to the overlap length.  No count is
    invented and none is lost.
    """
    edges = np.asarray(edges, float)
    out = np.zeros(len(edges) - 1)
    for b in blocks:
        c, y, dx = b["centers"], b["counts"], b["dx"]
        if not np.isfinite(dx) or dx <= 0:
            continue
        lo = c - dx / 2.0
        hi = c + dx / 2.0
        for l, h, k in zip(lo, hi, y):
            if k == 0:
                continue
            tot = h - l
            i0 = np.searchsorted(edges, l, "right") - 1
            i1 = np.searchsorted(edges, h, "right") - 1
            for t in range(max(i0, 0), min(i1, len(out) - 1) + 1):
                ov = min(h, edges[t + 1]) - max(l, edges[t])
                if ov > 0:
                    out[t] += k * ov / tot
    return out


# ----------------------------------------------------------------------------
# 1- vs 2-component lognormal mixture on a binned histogram (Poisson likelihood)
# ----------------------------------------------------------------------------
def _lognorm_bin_probs(edges, mus, sigmas, weights):
    """P(bin i) for a mixture of lognormals with parameters in log10(d)."""
    from scipy.stats import norm

    lo = np.log10(np.maximum(edges[:-1], 1e-12))
    hi = np.log10(np.maximum(edges[1:], 1e-12))
    p = np.zeros(len(lo))
    for w, m, s in zip(weights, mus, sigmas):
        s = max(s, 1e-3)
        p += w * (norm.cdf((hi - m) / s) - norm.cdf((lo - m) / s))
    return np.maximum(p, 1e-300)


def mix_fit(edges, counts, ncomp, n_restart=12, seed=0):
    """Poisson-likelihood EM + Nelder-Mead polish for a 1- or 2-lognormal histogram."""
    from scipy.optimize import minimize

    edges = np.asarray(edges, float)
    counts = np.asarray(counts, float)
    N = counts.sum()
    if N <= 0:
        return None
    rng = np.random.default_rng(seed)
    centers = np.sqrt(np.maximum(edges[:-1], 1e-12) * np.maximum(edges[1:], 1e-12))
    cen_log = np.log10(np.maximum(centers, 1e-12))
    ok = counts > 0
    if ok.sum() < 2:
        return None

    def negll(params):
        if ncomp == 1:
            m, ls = params
            p = _lognorm_bin_probs(edges, [m], [np.exp(ls)], [1.0])
        else:
            a, m1, ls1, m2, ls2 = params
            w1 = 1.0 / (1.0 + np.exp(-a))
            p = _lognorm_bin_probs(edges, [m1, m2], [np.exp(ls1), np.exp(ls2)], [w1, 1 - w1])
        mu = N * p
        return -np.sum(counts * np.log(np.maximum(mu, 1e-12)) - mu)

    best = None
    npar = 2 if ncomp == 1 else 5
    for r in range(n_restart):
        if ncomp == 1:
            x0 = np.array([np.average(cen_log, weights=counts), np.log(0.4 + 0.3 * rng.random())])
        else:
            q1, q2 = np.quantile(np.repeat(cen_log, counts.astype(int).clip(0, 10**6)), [0.33, 0.66])
            x0 = np.array(
                [rng.normal(0, 1.0), q1, np.log(0.25 + 0.2 * rng.random()), q2, np.log(0.25 + 0.2 * rng.random())]
            )
        try:
            res = minimize(negll, x0, method="Nelder-Mead",
                           options=dict(maxiter=8000, xatol=1e-6, fatol=1e-8))
        except Exception:
            continue
        if best is None or res.fun < best.fun:
            best = res
    if best is None:
        return None
    ll = -best.fun
    k = npar
    n = int((counts > 0).sum())
    return dict(
        loglik=float(ll),
        npar=k,
        aic=float(2 * k - 2 * ll),
        bic=float(k * np.log(max(n, 1)) - 2 * ll),
        params=[float(v) for v in best.x],
        n_occupied_bins=n,
        n_events=float(N),
    )


def mode_of(edges, counts):
    if counts.sum() <= 0:
        return np.nan
    i = int(np.argmax(counts))
    return float(np.sqrt(max(edges[i], 1e-12) * max(edges[i + 1], 1e-12)))


def concentration(counts, topk=3):
    c = np.sort(np.asarray(counts, float))[::-1]
    tot = c.sum()
    if tot <= 0:
        return dict(top1=np.nan, top3=np.nan, n_for_50pct=np.nan)
    return dict(
        top1=float(c[0] / tot),
        top3=float(c[:topk].sum() / tot),
        n_for_50pct=int(np.searchsorted(np.cumsum(c), 0.5 * tot) + 1),
    )


def write_pair(name, df_or_recs, index=False):
    """Write the same table as CSV and JSON (records orientation)."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    if isinstance(df_or_recs, pd.DataFrame):
        df = df_or_recs
    else:
        df = pd.DataFrame(df_or_recs)
    csv = RESULTS / f"{name}.csv"
    js = RESULTS / f"{name}.json"
    df.to_csv(csv, index=index)
    df.to_json(js, orient="records", indent=2)
    return csv, js