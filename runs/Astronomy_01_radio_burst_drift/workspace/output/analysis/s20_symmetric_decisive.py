#!/usr/bin/env python3
"""
Dimension tag: s20  (SYMMETRIC decisive test: centroid-anchored scan)

Task: TruthInsightBench Astronomy_01_radio_burst_drift (T0 = 2022-10-13)
Analysis unit: ONE COMPLETE DYNAMIC-SPECTRUM EVENT (n_events = 5).

Motivation / fix of a confound in s18 and s19.  The s18/s19 "corner anchor" puts
f(t) = f_max + m*(t-t0) for m<0 (line starts at the TOP of the band) and
f(t) = f_min + m*(t-t0) for m>0 (line starts at the BOTTOM).  A slope of +2 MHz/s
therefore samples only 8.0-10.2 MHz, while a slope of -2 MHz/s samples only
14.8-17.0 MHz.  Any frequency-dependent brightness in the images (and the images
do have a non-flat time-mean spectrum) is thus aliased into the slope axis, and
the reported slopes (-3.4 to -16.9 MHz/s) are compared against an asymmetric
alternative set.

s20 removes the confound.  Every line passes through the intensity-weighted
centroid (t_c, f_c) of the thresholded mask:

    f(t) = f_c + m*(t - t_c)

so +m and -m traverse exactly mirrored frequency ranges, the in-band time
coverage is IDENTICAL for +m and -m, and the sampled frequency intervals are
mirror images.  The decisive statistics are:

  * L(c m_rep) and L(-m_rep): the reported slope and its exact geometric mirror
  * ratio = L(m_rep) / L(-m_rep)
  * percentile of m_rep within {L(m)} and within the length-matched subset
  * local-maximum test: is there a local maximum of L(m) near m_rep?
  * injection control with the same anchor, to measure the estimator's power to
    see a ridge at the reported slope if one were present.

Statistic: mean background-subtracted intensity along the line, averaged over the
in-band columns, with the band average taken over a FIXED number of grid bins
(empty bins count as 0).  This is a strictly monotone functional of the image.
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
INJ_AMPS = [0.01, 0.02, 0.05, 0.10, 0.25]
INJ_SIGMA = 0.40
ERRORS = []


def log(m=""):
    print(m, flush=True)


def centroid_curve(S_asc, f_asc, t, slopes, band, anchor=None):
    """L(m) with every line passing through the mask centroid (t_c, f_c).

    anchor=(fc, tc) overrides the centroid: needed for the injection control,
    where adding a ridge moves the centroid and would otherwise move the line
    away from the very ridge being tested.
    """
    nF, nT = S_asc.shape
    fmin, fmax = float(f_asc[0]), float(f_asc[-1])
    if anchor is None:
        rr, cc = np.nonzero(S_asc > 0)
        w = S_asc[rr, cc]
        if w.sum() <= 0:
            return (np.full(slopes.size, np.nan), np.full(slopes.size, np.nan),
                    np.nan, np.nan)
        W = w / w.sum()
        fc = float((W * f_asc[rr]).sum())
        tc = float((W * t[cc]).sum())
    else:
        fc, tc = float(anchor[0]), float(anchor[1])
    num = np.vstack([np.zeros((1, nT)), np.cumsum(S_asc, axis=0)])
    j = np.arange(nT)
    L = np.full(slopes.size, np.nan)
    cov = np.full(slopes.size, np.nan)
    for k, m in enumerate(slopes):
        fl = fc + m * (t - tc)
        ok = (fl >= fmin) & (fl <= fmax)
        if not ok.any():
            continue
        ia = np.clip(np.searchsorted(f_asc, fl - band, "left"), 0, nF)
        ib = np.clip(np.searchsorted(f_asc, fl + band, "right"), 0, nF)
        nb = np.maximum(ib - ia, 1)
        v = (num[ib, j] - num[ia, j]) / nb
        L[k] = float(v[ok].mean())
        cov[k] = float(ok.mean())
    return L, cov, fc, tc


def pct_of(null, v):
    fin = np.isfinite(null)
    return float(stats.percentileofscore(null[fin], v, kind="mean"))


def main():
    meta = pd.read_csv(os.path.join(NV, "dynamic_spectrum_event_metadata.tsv"), sep="\t")
    log("=" * 92)
    log("s20 SYMMETRIC DECISIVE TEST (centroid anchor): is the reported slope present?")
    log("ANALYSIS UNIT = one complete dynamic-spectrum event (n_events = 5)")
    log("=" * 92)

    rows, inj_rows = [], []
    curves = {}
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
        log("#" * 92)
        log(f"### {eid}: reported {rep:+.4f} MHz/s; a line at that slope crosses the"
            f" whole {Df:.3f} MHz band in {Df/abs(rep):.3f} s"
            f" ({100*Df/abs(rep)/Dt:.0f}% of the {Dt:.4f} s window)")

        for lab, thr in THRESHOLDS:
            S = np.where(Z > thr, Z - 1.0, 0.0)
            L, cov, fc, tc = centroid_curve(S[::-1, :], f_asc, t, SLOPES, BAND_HW)
            krep = int(np.argmin(np.abs(SLOPES - rep)))
            kmir = int(np.argmin(np.abs(SLOPES + rep)))
            Lr, Lm = float(L[krep]), float(L[kmir])
            cov_rep = float(cov[krep])
            length = np.minimum(1.0, Df / (np.abs(SLOPES) * Dt + 1e-12))
            sel = np.abs(length - cov_rep) <= MATCH_TOL * cov_rep
            # local maxima of L
            y = np.where(np.isfinite(L), L, -np.inf)
            idx = [k for k in range(1, len(y) - 1)
                   if y[k] >= y[k - 1] and y[k] > y[k + 1]]
            d_lm = min(abs(SLOPES[k] - rep) for k in idx) if idx else np.nan
            near = np.abs(SLOPES - rep) <= 5.0
            argmax_near = float(SLOPES[int(np.argmax(np.where(near, y, -np.inf)))])
            res = dict(
                event_id=eid, threshold=lab, reported_drift=rep,
                f_centroid=fc, t_centroid=tc, cov_reported=cov_rep,
                L_reported=Lr, L_mirror=Lm, ratio_reported_over_mirror=Lr / Lm,
                pct_all=pct_of(L, Lr),
                pct_matched=pct_of(L[sel], Lr) if sel.sum() >= 5 else np.nan,
                pct_mirror=pct_of(L, Lm),
                argmax_slope=float(SLOPES[int(np.nanargmax(L))]),
                Lmax=float(np.nanmax(L)),
                Lmax_over_Lreported=float(np.nanmax(L) / max(Lr, 1e-12)),
                nearest_localmax_dist=d_lm,
                argmax_within5MHz=argmax_near,
                n_matched=int(sel.sum()))
            rows.append(res)
            log("-" * 92)
            log(f"[{eid} thr>{thr}] centroid ({tc:.3f} s, {fc:.3f} MHz); in-band time"
                f" coverage at the reported slope {100*cov_rep:.1f}%")
            log(f"   L(reported)={Lr:.5f}  L(mirror {-rep:+.2f})={Lm:.5f}"
                f"  ratio={res['ratio_reported_over_mirror']:.4f}"
                f"   (exact geometric control: identical coverage)")
            log(f"   percentile of m_rep: all slopes {res['pct_all']:.2f},"
                f" length-matched {res['pct_matched']:.2f} (n={res['n_matched']});"
                f" percentile of the mirror {res['pct_mirror']:.2f}")
            log(f"   argmax of L at {res['argmax_slope']:+.2f} MHz/s; Lmax/L(rep) ="
                f" {res['Lmax_over_Lreported']:.3f}; argmax within +-5 MHz/s of the"
                f" reported slope: {res['argmax_within5MHz']:+.2f} MHz/s; nearest local"
                f" max {d_lm:.2f} MHz/s away")

            for amp in INJ_AMPS:
                # the injected ridge must lie ON the scanned chord, i.e. through
                # the (pre-injection) centroid with the reported slope; otherwise
                # the test line and the injected ridge are different lines that
                # merely share a slope
                f_inj = fc + rep * (t - tc)
                Zs = 1.0 + (Z - 1.0) + amp * np.exp(
                    -0.5 * ((f[:, None] - f_inj[None, :]) / INJ_SIGMA) ** 2)
                Ss = np.where(Zs > thr, Zs - 1.0, 0.0)
                # anchor on the PRE-injection centroid so the tested line does not
                # drift away from the ridge as the ridge is added
                Li, _, _, _ = centroid_curve(Ss[::-1, :], f_asc, t, SLOPES, BAND_HW,
                                             anchor=(fc, tc))
                inj_rows.append(dict(
                    event_id=eid, threshold=lab, injected_amplitude=amp,
                    injected_slope=rep, sigma_MHz=INJ_SIGMA,
                    L_reported=float(Li[krep]), L_mirror=float(Li[kmir]),
                    ratio_over_mirror=float(Li[krep] / Li[kmir]),
                    pct_all=pct_of(Li, Li[krep]),
                    pct_matched=(pct_of(Li[sel], Li[krep]) if sel.sum() >= 5 else np.nan),
                    argmax_slope=float(SLOPES[int(np.nanargmax(Li))]),
                    anchor_fc=float(fc), anchor_tc=float(tc)))
            curves[eid] = (L.copy(), rep, fc, tc)
            del S, Zs, Ss, Li
            gc.collect()
        del Z
        gc.collect()

    R = pd.DataFrame(rows)
    INJ = pd.DataFrame(inj_rows)
    R.to_csv(os.path.join(RES, "s20_symmetric_decisive.csv"), index=False)
    INJ.to_csv(os.path.join(RES, "s20_symmetric_injection.csv"), index=False)

    # ------- LOO of the decisive numbers over the 5 events -------------------
    log()
    log("s20 LEAVE-ONE-OUT over the 5 events")
    loo = []
    for lab, _ in THRESHOLDS:
        sub = R[R.threshold == lab].set_index("event_id")
        for name, vals, bound, mode in [
                ("pct_all", sub["pct_all"], 50.0, "median"),
                ("pct_matched", sub["pct_matched"], 50.0, "median"),
                ("ratio_reported_over_mirror", sub["ratio_reported_over_mirror"], 1.0, "median"),
                ("n_events_ratio_gt1", (sub["ratio_reported_over_mirror"] > 1),
                 2.5, "count"),
                ("n_events_pct_extreme_p10",
                 ((sub["pct_all"] <= 10) | (sub["pct_all"] >= 90)), 2.5, "count"),
                ("Lmax_over_Lreported", sub["Lmax_over_Lreported"], 1.0, "median")]:
            v = np.asarray(vals, float)
            red = {"median": np.nanmedian, "count": np.nansum}[mode]
            full = float(red(v))
            folds = np.array([red(np.delete(v, k)) for k in range(len(v))], float)
            loo.append(dict(threshold=lab, statistic=name, aggregation=mode,
                            full_sample=full, loo_min=float(folds.min()),
                            loo_max=float(folds.max()),
                            loo_range=float(folds.max() - folds.min()),
                            crosses_boundary=bool(np.any(
                                (folds - bound) * (full - bound) < 0))))
    LOO = pd.DataFrame(loo)
    LOO.to_csv(os.path.join(RES, "s20_symmetric_leave_one_out.csv"), index=False)
    log(LOO.to_string(index=False))

    log()
    log("s20 injection control (threshold 1.05), percentile of the reported slope:")
    sub = INJ[INJ.threshold == 1.05]
    piv = sub.pivot_table(index="injected_amplitude", columns="event_id",
                          values="pct_all")
    log(piv.to_string())
    med = piv.median(axis=1)
    log("median over events: " + ", ".join(f"{a}:{b:.1f}" for a, b in med.items()))
    min_amp = float(med[med >= 90].index.min()) if (med >= 90).any() else None
    log(f"smallest injected amplitude whose median percentile reaches 90: {min_amp}")
    log("real-image percentiles (thr 1.05): "
        + ", ".join(f"{e}:{v:.1f}" for e, v in
                    R[R.threshold == "1.05"].set_index("event_id")["pct_all"].items()))

    headline = {}
    for lab, _ in THRESHOLDS:
        s = R[R.threshold == lab]
        headline[lab] = dict(
            median_ratio_reported_over_mirror=float(s.ratio_reported_over_mirror.median()),
            n_events_ratio_gt1=int((s.ratio_reported_over_mirror > 1).sum()),
            median_pct_all=float(s.pct_all.median()),
            median_pct_matched=float(s.pct_matched.median()),
            n_events_extreme_p10=int(((s.pct_all <= 10) | (s.pct_all >= 90)).sum()),
            n_events_extreme_p05=int(((s.pct_all <= 5) | (s.pct_all >= 95)).sum()),
            median_Lmax_over_Lreported=float(s.Lmax_over_Lreported.median()),
            n_events_argmax_within_1MHz=int(
                (np.abs(s.argmax_within5MHz - s.reported_drift) <= 1.0).sum()),
            median_nearest_localmax_dist=float(s.nearest_localmax_dist.median()))
    log()
    log("s20 HEADLINE: " + json.dumps(headline, indent=1))
    with open(os.path.join(RES, "s20_symmetric_summary.json"), "w") as fh:
        json.dump(dict(
            analysis_unit="one complete dynamic-spectrum event (n=5)",
            anchor="line passes through the intensity-weighted mask centroid, so +m and -m have identical coverage and mirrored frequency sampling",
            slopes=[float(SLOPES[0]), float(SLOPES[-1]), int(SLOPES.size)],
            band_half_width_MHz=BAND_HW, length_match_tol=MATCH_TOL,
            injection_amplitudes=INJ_AMPS, injection_sigma_MHz=INJ_SIGMA,
            headline=headline,
            min_amp_median_pct_ge90=min_amp,
            real_pct_thr105={e: float(v) for e, v in
                             R[R.threshold == "1.05"].set_index("event_id")["pct_all"].items()},
            decisive=json.loads(R.to_json(orient="records")),
            injection=json.loads(INJ.to_json(orient="records")),
            leave_one_out=json.loads(LOO.to_json(orient="records")),
            errors=ERRORS), fh, indent=1)

    # ------------------------------ figures ----------------------------------
    fig, axes = plt.subplots(1, 5, figsize=(27, 5.8), dpi=125, sharex=True)
    for ax, e in zip(axes, list(curves.keys())):
        L, rep, fc, tc = curves[e]
        ax.plot(SLOPES, L, lw=1.8, color="C0", label="L(m), centroid anchored")
        ax.axvline(rep, color="k", ls="--", lw=2, label=f"reported {rep:+.2f}")
        ax.axvline(-rep, color="b", ls=":", lw=1.5, label=f"mirror {-rep:+.2f}")
        s = R[(R.event_id == e) & (R.threshold == "1.05")].iloc[0]
        ax.set_title(f"{e}\nratio={s.ratio_reported_over_mirror:.3f}  "
                     f"pct={s.pct_all:.1f}  argmax={s.argmax_slope:+.1f}", fontsize=10)
        ax.set_xlabel("slope (MHz/s)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("mean line intensity")
    fig.suptitle("s20 symmetric (centroid-anchored) decisive scan, threshold 1.05; "
                 "+m and -m have identical coverage and mirrored sampling; "
                 "unit = one event (n=5)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG, "s20_symmetric_decisive.png"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), dpi=125)
    ax = axes[0]
    amps = sorted(INJ.injected_amplitude.unique())
    s105 = INJ[INJ.threshold == "1.05"]
    log(f"power-figure amplitudes plotted: {amps}")
    for e in list(curves.keys()):
        ss = s105[s105.event_id == e].sort_values("injected_amplitude")
        log(f"  {e}: pct_all = {[round(float(x), 2) for x in ss.pct_all]}")
        ax.plot(ss.injected_amplitude, ss.pct_all, "o-", label=e)
    for e in list(curves.keys()):
        rv = float(R[(R.event_id == e) & (R.threshold == "1.05")].pct_all.iloc[0])
        ax.axhline(rv, color="gray", ls=":", lw=0.8)
    ax.axhspan(90, 100, color="green", alpha=0.12)
    ax.set_ylim(0, 100)
    ax.set_xticks(amps)
    ax.set_xlabel("injected ridge amplitude (uniform grid, no log scale)")
    ax.set_ylabel("percentile of the reported slope")
    ax.set_title("(a) injection control: power of the symmetric test\n"
                 "(grey dotted = the real image's own percentile)")
    ax.legend(fontsize=8)
    ax = axes[1]
    for e in list(curves.keys()):
        rv = float(R[(R.event_id == e) & (R.threshold == "1.05")].pct_all.iloc[0])
        rm = float(R[(R.event_id == e) & (R.threshold == "1.05")].pct_matched.iloc[0]
                   if np.isfinite(R[(R.event_id == e) &
                                    (R.threshold == "1.05")].pct_matched.iloc[0]) else np.nan)
        ax.plot([0, 1], [rv, rm], "o-", label=e)
    ax.axhspan(90, 100, color="green", alpha=0.12)
    ax.set_xlim(-0.3, 1.3); ax.set_ylim(0, 100)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["all slopes", "length-matched"])
    ax.set_ylabel("percentile of the reported slope")
    ax.set_title("(b) real-image percentile of the reported slope\n"
                 "(shaded = 'recovered' region)")
    ax.legend(fontsize=8)
    fig.suptitle("s20 symmetric decisive test: real data vs injected control",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(FIG, "s20_symmetric_power.png"))
    plt.close(fig)
    log("s20 saved s20_symmetric_decisive.csv, s20_symmetric_injection.csv, "
        "s20_symmetric_leave_one_out.csv, s20_symmetric_summary.json + 2 figures")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        ERRORS.append(traceback.format_exc())
        log("FATAL:\n" + traceback.format_exc())
        raise