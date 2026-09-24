#!/usr/bin/env python3
"""
Dimension tag: s18  (DECISIVE, coverage-matched slope recovery test)

Task: TruthInsightBench Astronomy_01_radio_burst_drift (T0 = 2022-10-13)
Analysis unit: ONE COMPLETE DYNAMIC-SPECTRUM EVENT (n_events = 5).

Why a second scan: the s17-v2 Hough statistic H(m) = max over phase of the mean
line intensity is contaminated by phase-coverage, which varies strongly with |m|
because the 8-17 MHz band is finite and the reported slopes are steep (-3.4 to
-16.9 MHz/s).  s18 therefore uses a scan whose coverage is IDENTICAL for the
reported slope and for its mirror image +|m|, plus a same-length null subset:

  L(m) = mean over the time columns where the line is inside the band of the
         band-averaged (or single-bin) background-subtracted intensity along
         f(t) = a + m*(t - t0),  a = f_max if m<0 else f_min
         (the line enters the band at the corner and sweeps across it).

  * percentile of L(m_reported) in {L(m)} over 401 slopes in [-40,40] MHz/s
  * percentile of L(m_reported) in the length-matched subset
    (same |m|*Dt / Df within +-30%)
  * the geometric control: L(m_reported) vs L(-m_reported), which has EXACTLY the
    same in-band time coverage by construction
  * a local-maximum test: a genuine ridge must make L(m) peak near m_reported
  * a coverage-restricted fit: Theil-Sen slope of the centroid track using only
    the time columns where the reported-slope line is in band
  * injection control: a synthetic ridge with the reported slope is added to the
    REAL image at amplitude A, and the whole pipeline is re-run, which measures
    the power of the estimator to recover the reported rate if it were present
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

EPS = 1.0e-9
THRESHOLDS = [("1.01", 1.01), ("1.05", 1.05), ("1.20", 1.20)]
SLOPES = np.linspace(-40.0, 40.0, 401)
BAND_HW = 0.35
MATCH_TOL = 0.30
INJ_AMPS = [0.02, 0.05, 0.10, 0.25]
INJ_SIGMA = 0.40
ERRORS = []


def log(m=""):
    print(m, flush=True)


def theil_sen(x, y):
    r = stats.theilslopes(y, x, 0.95)
    se = (r[3] - r[2]) / (2 * 1.959964)
    return float(r[0]), float(r[1]), float(se)


def coverage(S_asc, f_asc, t, m, anchor_corner=True, x0=None, band=None):
    """Per-column matched-filter values along f(t)=a+m*(t-x0) plus in-band mask.

    a = f_max if m<0 else f_min (corner entry).  band=None -> the single nearest
    frequency bin; otherwise the mean of S over ALL grid bins within +/- band MHz
    of the line, with EMPTY bins counted as 0.

    The denominator is the FIXED number of grid bins inside the band, not the
    number of non-empty ones.  This matters: a data-dependent denominator makes
    the statistic non-monotone in the image (adding signal can add low-value bins
    and *lower* the mean), which invalidates percentile and injection arguments.
    """
    nF, nT = S_asc.shape
    fmin, fmax = float(f_asc[0]), float(f_asc[-1])
    x0 = float(t[0]) if x0 is None else x0
    a = fmax if m < 0 else fmin
    fl = a + m * (t - x0)
    ok = (fl >= fmin) & (fl <= fmax)
    if band is None:
        idx = np.clip(np.rint((fl - fmin) / (f_asc[1] - f_asc[0])).astype(int), 0, nF - 1)
        v = S_asc[idx, np.arange(nT)]
    else:
        num = np.vstack([np.zeros((1, nT)), np.cumsum(S_asc, axis=0)])
        ia = np.clip(np.searchsorted(f_asc, fl - band, "left"), 0, nF)
        ib = np.clip(np.searchsorted(f_asc, fl + band, "right"), 0, nF)
        j = np.arange(nT)
        nb = np.maximum(ib - ia, 1)             # fixed geometric window width
        nk = num[ib, j] - num[ia, j]
        v = nk / nb
    return v, ok


def corner_scan(S_asc, f_asc, t, slopes, band):
    """L(m): mean line intensity over the in-band columns, and the coverage."""
    L = np.full(slopes.size, np.nan)
    cov = np.full(slopes.size, np.nan)
    vals = []
    for k, m in enumerate(slopes):
        v, ok = coverage(S_asc, f_asc, t, m, band=band)
        if ok.sum() == 0:
            continue
        L[k] = float(np.nanmean(v[ok]))
        cov[k] = float(ok.mean())
        vals.append(v)
    return L, cov


def pct_of(null, value):
    fin = np.isfinite(null)
    return float(stats.percentileofscore(null[fin], value, kind="mean"))


def local_max_near(slopes, curve, target, window):
    """Distance from target to the nearest local maximum of curve, and whether
    the maximum of the curve inside +-window is at/near the target."""
    fin = np.isfinite(curve)
    if fin.sum() < 5:
        return np.nan, False
    y = curve.copy()
    y[~fin] = -np.inf
    near = np.abs(slopes - target) <= window
    if not near.any():
        return np.nan, False
    i_loc = int(np.argmax(np.where(near, y, -np.inf)))
    # local maxima
    idx = [i for i in range(1, len(y) - 1)
           if y[i] >= y[i - 1] and y[i] > y[i + 1] and np.isfinite(curve[i])]
    if not idx:
        return np.nan, False
    d = min(abs(slopes[i] - target) for i in idx)
    is_peak = bool(abs(slopes[i_loc] - target) <= 2 * (slopes[1] - slopes[0]))
    return float(d), is_peak


def main():
    meta = pd.read_csv(os.path.join(NV, "dynamic_spectrum_event_metadata.tsv"), sep="\t")
    log("=" * 90)
    log("s18  DECISIVE COVERAGE-MATCHED SLOPE RECOVERY TEST")
    log("ANALYSIS UNIT = one complete dynamic-spectrum event (n_events = 5)")
    log("=" * 90)

    rows, inj_rows = [], []
    store = {}

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
        Df, Dt = fmax - fmin, float(t[-1] - t[0])
        t0 = float(t[0])
        nT = t.size
        cov_rep_geom = min(1.0, Df / (abs(rep) * Dt))
        log("#" * 90)
        log(f"### {eid}: reported {rep:+.4f} MHz/s; |m|*Dt = {abs(rep)*Dt:.3f} MHz"
            f" = {100*abs(rep)*Dt/Df:.1f}% of the {Df:.3f} MHz band; geometric in-band"
            f" time coverage of a line at the reported slope = {100*cov_rep_geom:.1f}%"
            f" of the {Dt:.4f} s window")

        st = dict(f=f, t=t, rep=rep, Df=Df, Dt=Dt, thr={}, Z=Z.astype(np.float32).copy())
        for lab, thr in THRESHOLDS:
            mask = Z > thr
            S = np.where(mask, Z - 1.0, 0.0)
            S_asc = S[::-1, :]
            # ---------------- corner-anchored, coverage-matched scan ----------
            Lb, cov = corner_scan(S_asc, f_asc, t, SLOPES, BAND_HW)
            Ls, covs = corner_scan(S_asc, f_asc, t, SLOPES, None)
            krep = int(np.argmin(np.abs(SLOPES - rep)))
            mirror = -rep
            kmir = int(np.argmin(np.abs(SLOPES - mirror)))
            length = np.minimum(1.0, Df / (np.abs(SLOPES) * Dt + 1e-12))
            sel = np.abs(length - cov_rep_geom) <= MATCH_TOL * cov_rep_geom
            res = dict(
                event_id=eid, threshold=lab, reported_drift=rep,
                cov_geom_reported=cov_rep_geom, cov_scan_at_reported=float(cov[krep]),
                L_band_reported=float(Lb[krep]),
                L_band_mirror=float(Lb[kmir]),
                mirror_slope=float(SLOPES[kmir]),
                ratio_reported_over_mirror=float(Lb[krep] / Lb[kmir]),
                pct_all_band=pct_of(Lb, Lb[krep]),
                pct_matched_band=pct_of(Lb[sel], Lb[krep]) if sel.sum() >= 5 else np.nan,
                pct_mirror_all=pct_of(Lb, Lb[kmir]),
                n_matched=int(sel.sum()),
                L_single_reported=float(Ls[krep]),
                pct_all_single=pct_of(Ls, Ls[krep]),
                pct_matched_single=pct_of(Ls[sel], Ls[krep]) if sel.sum() >= 5 else np.nan,
                argmax_slope_band=float(SLOPES[int(np.nanargmax(Lb))]),
                Lmax_band=float(np.nanmax(Lb)),
                pct_of_max_band=float(100.0 * np.nanmax(Lb) / max(Lb[krep], 1e-12)),
                frac_grid_mask=float(mask.mean()), n_px_mask=int(mask.sum()))
            d_band, peak_band = local_max_near(SLOPES, Lb, rep, 5.0)
            d_sing, peak_sing = local_max_near(SLOPES, Ls, rep, 5.0)
            res["nearest_localmax_dist_band"] = d_band
            res["argmax_near_reported_band"] = peak_band
            res["nearest_localmax_dist_single"] = d_sing
            res["argmax_near_reported_single"] = peak_sing

            # -------- coverage-restricted centroid fit (only in-band columns) --
            colw = S.sum(axis=0)
            good = colw > 0
            cen = np.full(nT, np.nan)
            cen[good] = (S[:, good] * f[:, None]).sum(axis=0) / colw[good]
            _, inband = coverage(S_asc, f_asc, t, rep, band=None)
            use = good & inband
            if use.sum() >= 3:
                sl_cov, _, se_cov = theil_sen(t[use], cen[use])
            else:
                sl_cov, se_cov = np.nan, np.nan
            res["covrestricted_rate"] = sl_cov
            res["covrestricted_se"] = se_cov
            res["covrestricted_ncols"] = int(use.sum())
            res["covrestricted_diff"] = float(sl_cov - rep)

            rows.append(res)
            log("-" * 90)
            log(f"[{eid} thr>{thr}] mask {100*res['frac_grid_mask']:.1f}% grid;"
                f" scan coverage at reported slope {100*res['cov_scan_at_reported']:.1f}%")
            log(f"   L_band(reported) = {res['L_band_reported']:.5f}"
                f"   L_band(mirror {res['mirror_slope']:+.1f}) = {res['L_band_mirror']:.5f}"
                f"   ratio = {res['ratio_reported_over_mirror']:.4f}")
            log(f"   percentile in null: all slopes {res['pct_all_band']:.2f},"
                f" length-matched {res['pct_matched_band']:.2f} (n={res['n_matched']});"
                f"   percentile of the MIRROR slope {res['pct_mirror_all']:.2f}")
            log(f"   single-bin variant: L={res['L_single_reported']:.5f},"
                f" pct all {res['pct_all_single']:.2f}, matched"
                f" {res['pct_matched_single']:.2f}")
            log(f"   argmax of L_band at {res['argmax_slope_band']:+.2f} MHz/s"
                f" (Lmax={res['Lmax_band']:.5f} = {res['pct_of_max_band']:.0f}% of L(reported))")
            log(f"   nearest local max to the reported slope: {d_band:.2f} MHz/s (band),"
                f" {d_sing:.2f} MHz/s (single); argmax within +-5 MHz/s at the reported"
                f" slope? band={peak_band} single={peak_sing}")
            log(f"   coverage-restricted centroid fit ({res['covrestricted_ncols']} cols"
                f" in band) = {sl_cov:+.4f} +- {se_cov:.4f} MHz/s vs reported {rep:+.4f}"
                f" (diff {res['covrestricted_diff']:+.4f})")

            # ---------------------- injection control -------------------------
            for amp in INJ_AMPS:
                Zs = 1.0 + (Z - 1.0) + amp * np.exp(
                    -0.5 * ((f[:, None] - (fmax + rep * (t[None, :] - t0))) / INJ_SIGMA) ** 2)
                Ss = np.where(Zs > thr, Zs - 1.0, 0.0)
                Lb_i, _ = corner_scan(Ss[::-1, :], f_asc, t, SLOPES, BAND_HW)
                kbest = int(np.nanargmax(Lb_i))
                inj_rows.append(dict(
                    event_id=eid, threshold=lab, injected_amplitude=amp,
                    injected_slope=rep, sigma_MHz=INJ_SIGMA,
                    L_reported=float(Lb_i[krep]), L_mirror=float(Lb_i[kmir]),
                    pct_all=pct_of(Lb_i, Lb_i[krep]),
                    pct_matched=(pct_of(Lb_i[sel], Lb_i[krep]) if sel.sum() >= 5 else np.nan),
                    argmax_slope=float(SLOPES[kbest]),
                    ratio_over_mirror=float(Lb_i[krep] / Lb_i[kmir])))
                log(f"   INJECT amp {amp:.2f}: L(reported) {Lb_i[krep]:.5f} vs mirror"
                    f" {Lb_i[kmir]:.5f}; ratio {Lb_i[krep]/Lb_i[kmir]:.3f};"
                    f" pct all {pct_of(Lb_i, Lb_i[krep]):.2f};"
                    f" argmax {SLOPES[kbest]:+.2f}")

            st["thr"][lab] = dict(S=S, Lb=Lb, Ls=Ls, cov=cov, krep=krep, kmir=kmir,
                                  cen=cen, good=good, inband=inband)
        store[eid] = st
        del Z, S, S_asc, Zs, Ss
        gc.collect()

    R = pd.DataFrame(rows)
    INJ = pd.DataFrame(inj_rows)
    R.to_csv(os.path.join(RES, "s18_decisive_scan.csv"), index=False)
    INJ.to_csv(os.path.join(RES, "s18_injection_control.csv"), index=False)

    # ---- LOO of the decisive statistics over the 5 events -------------------
    LOG = []
    for lab, _ in THRESHOLDS:
        sub = R[R.threshold == lab].set_index("event_id")
        for name, vals, bound in [
                ("pct_all_band", sub["pct_all_band"], 50.0),
                ("pct_matched_band", sub["pct_matched_band"], 50.0),
                ("ratio_reported_over_mirror", sub["ratio_reported_over_mirror"], 1.0),
                ("n_events_same_sign_as_reported",
                 np.sign(sub["covrestricted_rate"] - sub["reported_drift"]), 0.0)]:
            v = np.asarray(vals, float)
            full = float(np.nanmedian(v))
            folds = np.array([np.nanmedian(np.delete(v, k)) for k in range(len(v))], float)
            LOG.append(dict(threshold=lab, statistic=name, full_sample=full,
                            loo_min=float(np.nanmin(folds)), loo_max=float(np.nanmax(folds)),
                            loo_range=float(np.nanmax(folds) - np.nanmin(folds)),
                            crosses_boundary=bool(np.any((folds - bound) * (full - bound) < 0))))
    LOO = pd.DataFrame(LOG)
    LOO.to_csv(os.path.join(RES, "s18_leave_one_out.csv"), index=False)
    log()
    log("s18 LEAVE-ONE-OUT of the decisive statistics")
    log(LOO.to_string(index=False))

    # ---- aggregated headline numbers ---------------------------------------
    headline = {}
    for lab, _ in THRESHOLDS:
        sub = R[R.threshold == lab]
        headline[lab] = dict(
            n_events_extreme_p05=int(((sub.pct_all_band <= 5) | (sub.pct_all_band >= 95)).sum()),
            n_events_extreme_p10=int(((sub.pct_all_band <= 10) | (sub.pct_all_band >= 90)).sum()),
            n_events_reported_above_mirror=int((sub.ratio_reported_over_mirror > 1).sum()),
            median_ratio_reported_over_mirror=float(sub.ratio_reported_over_mirror.median()),
            median_pct_all=float(sub.pct_all_band.median()),
            median_pct_matched=float(sub.pct_matched_band.median()),
            median_Lmax_over_Lreported=float(sub.pct_of_max_band.median()),
            n_events_argmax_near_reported=int(sub.argmax_near_reported_band.sum()),
            n_events_same_sign_covrestricted=int(
                (np.sign(sub.covrestricted_rate) == np.sign(sub.reported_drift)).sum()))
    log()
    log("s18 HEADLINE per threshold:")
    log(json.dumps(headline, indent=1))
    with open(os.path.join(RES, "s18_decisive_summary.json"), "w") as fh:
        json.dump(dict(
            analysis_unit="one complete dynamic-spectrum event (n=5)",
            slopes=[float(SLOPES[0]), float(SLOPES[-1]), int(SLOPES.size)],
            band_half_width_MHz=BAND_HW, length_match_tol=MATCH_TOL,
            injection_amplitudes=INJ_AMPS, injection_sigma_MHz=INJ_SIGMA,
            headline=headline,
            decisive_scan=json.loads(R.to_json(orient="records")),
            injection_control=json.loads(INJ.to_json(orient="records")),
            leave_one_out=json.loads(LOO.to_json(orient="records")),
            errors=ERRORS), fh, indent=1)

    # ----------------------------- figures -----------------------------------
    ord_ids = list(store.keys())
    fig, axes = plt.subplots(1, 5, figsize=(27, 5.6), dpi=125, sharex=True)
    for ax, e in zip(axes, ord_ids):
        thr = "1.05"
        s = store[e]["thr"][thr]
        ax.plot(SLOPES, s["Lb"], lw=1.6, color="C0", label="L(m), band +-0.35 MHz")
        ax.plot(SLOPES, s["Ls"], lw=1.0, color="C1", alpha=0.8, label="L(m), single bin")
        ax.axvline(store[e]["rep"], color="k", ls="--", lw=2,
                   label=f"reported {store[e]['rep']:+.2f}")
        ax.axvline(-store[e]["rep"], color="b", ls=":", lw=1.6,
                   label=f"mirror {-store[e]['rep']:+.2f}")
        ax.set_title(f"{e}  (thr {thr})\nL(rep)/L(mirror)="
                     f"{s['Lb'][s['krep']]/s['Lb'][s['kmir']]:.3f}, pct="
                     f"{pct_of(s['Lb'], s['Lb'][s['krep']]):.1f}", fontsize=10)
        ax.set_xlabel("slope (MHz/s)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("mean line intensity")
    fig.suptitle("s18(a) DECISIVE coverage-matched scan: corner-anchored line "
                 "intensity vs slope, threshold 1.05; mirror slope is the exact "
                 "coverage control", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(FIG, "s18_decisive_scan.png"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(24, 5.6), dpi=125)
    x = range(len(THRESHOLDS)); xl = [l for l, _ in THRESHOLDS]
    ax = axes[0]
    for e in ord_ids:
        sub = R[R.event_id == e].set_index("threshold")
        ax.plot(x, [sub.loc[l, "ratio_reported_over_mirror"] for l, _ in THRESHOLDS],
                "o-", label=e)
    ax.axhline(1.0, color="k", lw=1.2)
    ax.set_xticks(list(x)); ax.set_xticklabels(xl)
    ax.set_xlabel("threshold"); ax.set_ylabel("L(reported)/L(mirror)")
    ax.set_title("(b1) coverage-controlled contrast\n(>1 supports the reported sign)")
    ax.legend(fontsize=8)
    ax = axes[1]
    for e in ord_ids:
        sub = R[R.event_id == e].set_index("threshold")
        ax.plot(x, [sub.loc[l, "pct_all_band"] for l, _ in THRESHOLDS], "o-", label=e)
        ax.plot(x, [sub.loc[l, "pct_matched_band"] for l, _ in THRESHOLDS], "s--",
                alpha=0.5, color="gray")
    ax.axhspan(90, 100, color="green", alpha=0.12)
    ax.set_ylim(0, 100); ax.set_xticks(list(x)); ax.set_xticklabels(xl)
    ax.set_xlabel("threshold"); ax.set_ylabel("percentile of reported slope")
    ax.set_title("(b2) percentile vs threshold (grey = length-matched)")
    ax.legend(fontsize=8)
    ax = axes[2]
    for e in ord_ids:
        sub = R[R.event_id == e].set_index("threshold")
        ax.plot(x, [sub.loc[l, "Lmax_band"] / sub.loc[l, "L_band_reported"]
                    for l, _ in THRESHOLDS], "o-", label=e)
    ax.axhline(1.0, color="k", lw=1.0)
    ax.set_xticks(list(x)); ax.set_xticklabels(xl)
    ax.set_xlabel("threshold"); ax.set_ylabel("Lmax / L(reported)")
    ax.set_title("(b3) how far the reported slope is from\nthe best slope")
    ax.legend(fontsize=8)
    ax = axes[3]
    sub = INJ[INJ.threshold == "1.05"]
    for e in ord_ids:
        s = sub[sub.event_id == e].sort_values("injected_amplitude")
        ax.plot(s.injected_amplitude, s.pct_all, "o-", label=e)
    ax.axhspan(90, 100, color="green", alpha=0.12)
    ax.set_ylim(0, 100)
    ax.set_xlabel("injected ridge amplitude at the reported slope")
    ax.set_ylabel("percentile of the reported slope")
    ax.set_title("(b4) INJECTION CONTROL: power of the\nestimator to see the reported rate")
    ax.legend(fontsize=8)
    fig.suptitle("s18(b) decisive-test sensitivity; unit = one event (n=5)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(FIG, "s18_decisive_sensitivity.png"))
    plt.close(fig)

    # (c) images with reported + mirror chords
    fig, axes = plt.subplots(1, 5, figsize=(27, 6.2), dpi=125)
    for ax, e in zip(axes, ord_ids):
        st = store[e]
        f_, t_ = st["f"], st["t"]
        ax.imshow(np.clip(st["Z"] - 1, 0, None), origin="lower", aspect="auto",
                  extent=[t_[0], t_[-1], f_.min(), f_.max()], cmap="magma",
                  norm=PowerNorm(0.4, vmin=0, vmax=0.5))
        rep = st["rep"]
        fmax, fmin = f_.max(), f_.min()
        tl = np.array([t_[0], t_[-1]])
        ax.plot(tl, fmax + rep * (tl - t_[0]), "c--", lw=2.2,
                label=f"reported {rep:+.2f}")
        ax.plot(tl, fmin - rep * (tl - t_[0]), "b:", lw=1.8,
                label=f"mirror {-rep:+.2f}")
        ax.set_title(f"{e}: reported implies {abs(rep)*st['Dt']:.1f} MHz of motion"
                     f" ({100*abs(rep)*st['Dt']/st['Df']:.0f}% of band)", fontsize=10)
        ax.set_xlabel("time (s)"); ax.set_ylabel("frequency (MHz)")
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("s18(c) the reported-slope chord (cyan) and its coverage-matched "
                 "mirror (blue) on the real images; unit = one event (n=5)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG, "s18_chords.png"))
    plt.close(fig)

    store.clear(); gc.collect()
    log("s18 saved: s18_decisive_scan.csv, s18_injection_control.csv, "
        "s18_leave_one_out.csv, s18_decisive_summary.json + 3 figures")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        ERRORS.append(traceback.format_exc())
        log("FATAL:\n" + traceback.format_exc())
        raise