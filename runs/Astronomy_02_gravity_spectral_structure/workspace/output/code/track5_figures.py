#!/usr/bin/env python3
"""TRACK 5 -- Evidence figures and the reproduction harness (figures half).

Generates the five required publication-quality PNGs into output/figures/,
writes output/figures/figures_manifest.json (file name, byte size, exact
command) and dumps every number quoted in the figure annotations to
output/results/track5_figure_annotations.json.

Independent analysis unit: one complete time-resolved pass (n = 22) for all
trace-derived quantities, one complete candidate spectral parameter
combination for all IC-derived quantities.  No p-value or standard error in
this script treats the 1698 time samples or the 250/507 grid rows as
independent replicates: every trace statistic is a whole-pass statistic and
its uncertainty is a bootstrap over the 22 passes.

Run:  python3 output/code/track5_figures.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, required
import matplotlib.pyplot as plt
import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from common import (  # noqa: E402
    CAND_BAND,
    MODELS,
    OUT,
    PROJECT,
    freq_grid,
    load_candidates,
    load_modes,
    load_ranked_grid,
    load_traces,
    load_uniform,
    ls_power,
    per_pass_rms,
    write_json,
)

FIGDIR = OUT / "figures"
RESDIR = OUT / "results"
FIGDIR.mkdir(parents=True, exist_ok=True)
RESDIR.mkdir(parents=True, exist_ok=True)

COLORS = {"A": "#0072B2", "B": "#D55E00", "C": "#009E73"}
STYLES = {"A": "-", "B": "--", "C": (0, (1, 1))}
BOOT_SEED = 20220101
BOOT_B = 10000
COMB = np.arange(CAND_BAND[0], CAND_BAND[1] + 1e-9, 50.0)  # 860..1310 step 50 uHz

PRODUCED: list[Path] = []


# --------------------------------------------------------------------------
# shared statistics (whole-pass unit only)
# --------------------------------------------------------------------------
def boot_mean_ci(x, b=BOOT_B, seed=BOOT_SEED, lo=2.5, hi=97.5):
    """Bootstrap CI of the mean over the declared independent unit."""
    x = np.asarray(x, float)
    n = len(x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(b, n))
    d = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(d, lo)), float(np.percentile(d, hi))


def build_stats():
    t0 = time.time()
    traces = load_traces()
    rms = per_pass_rms(traces)
    f_uhz, f_hz = freq_grid(200.0, 3000.0, 1.0)
    band = (f_uhz >= CAND_BAND[0]) & (f_uhz <= CAND_BAND[1])

    spectra = {}
    for tag, col in MODELS.items():
        per_pass = []
        for _pid, g in traces.groupby("pass_id"):
            per_pass.append(ls_power(g["relative_time_min"].values, g[col].values, f_hz))
        spectra[tag] = np.asarray(per_pass)

    st = {"f_uhz": f_uhz, "band_mask": band, "rms": rms, "spectra": spectra}

    st["rms_stats"] = {}
    for tag in ("A", "B", "C"):
        m, lo, hi = boot_mean_ci(rms[tag].values)
        st["rms_stats"][tag] = {
            "mean": m, "ci_lo": lo, "ci_hi": hi,
            "median": float(rms[tag].median()),
            "min": float(rms[tag].min()), "max": float(rms[tag].max()),
        }
    st["rms_pair"] = {}
    for a, b in (("A", "B"), ("A", "C"), ("C", "B")):
        d = rms[a].values - rms[b].values
        m, lo, hi = boot_mean_ci(d)
        st["rms_pair"][f"{a}-{b}"] = {
            "mean": m, "ci_lo": lo, "ci_hi": hi,
            "n_pass_gt0": int((d > 0).sum()), "n_pass": int(d.size),
        }
    st["rms_loso"] = {
        tag: [float((rms[tag].values.sum() - v) / (len(rms) - 1)) for v in rms[tag].values]
        for tag in ("A", "B", "C")
    }

    st["spec_mean"] = {tag: spectra[tag].mean(axis=0) for tag in ("A", "B", "C")}
    st["spec_p16"] = {tag: np.percentile(spectra[tag], 16, axis=0) for tag in ("A", "B", "C")}
    st["spec_p84"] = {tag: np.percentile(spectra[tag], 84, axis=0) for tag in ("A", "B", "C")}
    st["spec_peak"] = {tag: float(f_uhz[np.argmax(st["spec_mean"][tag])]) for tag in ("A", "B", "C")}

    st["band_power"] = {}
    for tag in ("A", "B", "C"):
        bp = spectra[tag][:, band].mean(axis=1)
        m, lo, hi = boot_mean_ci(bp)
        st["band_power"][tag] = {
            "mean": m, "ci_lo": lo, "ci_hi": hi,
            "min": float(bp.min()), "max": float(bp.max()),
        }
    st["band_power_pair"] = {}
    for a, b in (("A", "B"), ("A", "C"), ("C", "B")):
        d = spectra[a][:, band].mean(axis=1) - spectra[b][:, band].mean(axis=1)
        m, lo, hi = boot_mean_ci(d)
        st["band_power_pair"][f"{a}-{b}"] = {
            "mean": m, "ci_lo": lo, "ci_hi": hi,
            "n_pass_gt0": int((d > 0).sum()), "n_pass": int(d.size),
        }
    st["comb"] = COMB
    st["comb_idx"] = np.array([int(np.argmin(np.abs(f_uhz - c))) for c in COMB])
    st["elapsed_build_s"] = time.time() - t0
    return st


# --------------------------------------------------------------------------
# fig 1 -- residual traces, 22 passes x 3 models
# --------------------------------------------------------------------------
def fig1(traces, st):
    pass_ids = sorted(traces["pass_id"].unique())
    ncol, nrow = 6, 4  # 24 panels: 22 passes + legend + zoom, no empty axes
    fig, axes = plt.subplots(nrow, ncol, figsize=(20.0, 12.0), constrained_layout=True)
    flat = axes.ravel()
    for k, pid in enumerate(pass_ids):
        ax = flat[k]
        g = traces[traces["pass_id"] == pid]
        t = g["relative_time_min"].values
        ymax = 1e-9
        for tag, col in MODELS.items():
            y = g[col].values
            ymax = max(ymax, float(np.max(np.abs(y))))
            ax.plot(t, y, color=COLORS[tag], ls=STYLES[tag], lw=0.95, label=f"model {tag}")
        ax.set_xlim(-42.5, 42.5)
        ax.set_ylim(-1.06 * ymax, 1.06 * ymax)
        ax.axhline(0.0, color="0.75", lw=0.5, zorder=0)
        r = st["rms"].loc[pid]
        ax.set_title(f"{pid}   n={len(g)}", fontsize=9.0, pad=2.5)
        # RMS values inside the axes so that neighbouring titles cannot collide
        ax.text(0.015, 0.035,
                "RMS A/B/C = %.4f/%.4f/%.4f mm/s" % (r.A, r.B, r.C),
                transform=ax.transAxes, fontsize=6.2, color="0.15", family="monospace",
                bbox=dict(fc="white", ec="none", alpha=0.72, pad=0.8), zorder=6)
        ax.tick_params(labelsize=6.8)
        ax.grid(alpha=0.22, lw=0.4)
        if k % ncol == 0:
            ax.set_ylabel("residual (mm/s)", fontsize=8.0)
        if k >= len(pass_ids) - ncol:
            ax.set_xlabel("relative time (min)", fontsize=8.0)

    # panel 23: legend
    axl = flat[22]
    axl.axis("off")
    handles = [
        plt.Line2D([], [], color=COLORS[t], ls=STYLES[t], lw=2.2, label=f"model {t}") for t in ("A", "B", "C")
    ]
    handles.append(plt.Line2D([], [], color="0.75", lw=0.8, label="zero"))
    axl.legend(handles=handles, loc="center", fontsize=11, frameon=True, title="22 passes, 1698 samples",
               title_fontsize=10)
    axl.text(0.5, 0.16,
             "panels share x range; every panel uses its own\n"
             "symmetric y scale (max |residual| of the pass)",
             ha="center", va="top", fontsize=8, transform=axl.transAxes)

    # panel 24: zoom on pass_14 (largest relative A-B separation)
    axz = flat[23]
    g = traces[traces["pass_id"] == "pass_14"]
    t = g["relative_time_min"].values
    zm = (t >= -10.0) & (t <= 10.0)
    for tag, col in MODELS.items():
        axz.plot(t[zm], g[col].values[zm], color=COLORS[tag], ls=STYLES[tag], lw=1.6,
                 marker="o", ms=2.6, label=f"model {tag}")
    r = st["rms"].loc["pass_14"]
    axz.set_title("zoom: pass_14, |t| <= 10 min", fontsize=9.0)
    axz.text(0.5, 0.035,
             "whole-pass RMS A/B/C = %.4f/%.4f/%.4f mm/s" % (r.A, r.B, r.C),
             transform=axz.transAxes, ha="center", fontsize=6.6, family="monospace",
             bbox=dict(fc="white", ec="none", alpha=0.72, pad=1.0), zorder=6)
    axz.set_xlabel("relative time (min)", fontsize=8.0)
    axz.set_ylabel("residual (mm/s)", fontsize=8.0)
    axz.tick_params(labelsize=6.8)
    axz.legend(fontsize=7.0, loc="upper left")
    axz.grid(alpha=0.25, lw=0.4)

    fig.suptitle(
        "Fig 1 - time-resolved gravity residuals, 22 passes, models A/B/C "
        "(unit of analysis = one complete pass)",
        fontsize=13,
    )
    out = FIGDIR / "fig1_residual_traces.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    PRODUCED.append(out)
    return out


# --------------------------------------------------------------------------
# fig 2 -- whole-pass RMS per model, pass as unit
# --------------------------------------------------------------------------
def fig2(st):
    rms = st["rms"]
    order = rms.mean(axis=1).sort_values().index.tolist()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 7.4), constrained_layout=True,
                                   gridspec_kw={"width_ratios": [1.65, 1.0]})

    x = np.arange(len(order))
    for i, pid in enumerate(order):
        vals = {t: float(rms.loc[pid, t]) for t in ("A", "B", "C")}
        ax1.plot([i, i], [min(vals.values()), max(vals.values())], color="0.7", lw=0.9, zorder=1)
        for tag in ("A", "B", "C"):
            ax1.plot(i, vals[tag], marker="o", ms=4.6, color=COLORS[tag], zorder=3,
                     mec="white", mew=0.5)
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels([p.replace("pass_", "") for p in order], fontsize=8)
    ax1.set_xlabel("pass (ordered by mean RMS of the three models)", fontsize=10)
    ax1.set_ylabel("whole-pass RMS residual (mm/s, log scale)", fontsize=10)
    ax1.set_title("paired per-pass RMS: A, B, C evaluated on the same pass", fontsize=10.5)
    ax1.grid(alpha=0.25, lw=0.5, which="both")
    ax1.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=COLORS[t], label=f"model {t}")
                        for t in ("A", "B", "C")], fontsize=9, loc="upper left")

    labs = ["A", "B", "C"]
    for j, tag in enumerate(labs):
        s = st["rms_stats"][tag]
        ax2.errorbar(j, s["mean"], yerr=[[s["mean"] - s["ci_lo"]], [s["ci_hi"] - s["mean"]]],
                     fmt="s", ms=9, color=COLORS[tag], capsize=6, lw=2.0, zorder=3)
        ax2.scatter(np.full(len(rms), j) + np.random.default_rng(7).uniform(-0.13, 0.13, len(rms)),
                    rms[tag].values, s=16, color=COLORS[tag], alpha=0.45, zorder=2)
        ax2.text(j, s["ci_hi"] * 1.12, f"mean {s['mean']:.5f}\n[{s['ci_lo']:.5f}, {s['ci_hi']:.5f}]",
                 ha="center", fontsize=8.2)
    ax2.set_yscale("log")
    ax2.set_xticks(range(3))
    ax2.set_xticklabels([f"model {t}" for t in labs], fontsize=10)
    ax2.set_ylabel("whole-pass RMS residual (mm/s, log scale)", fontsize=10)
    ax2.set_title("mean whole-pass RMS, 95% bootstrap CI\nover the 22 passes (B = 10000)", fontsize=10.5)
    ax2.grid(alpha=0.25, lw=0.5, which="both")
    ax2.set_xlim(-0.6, 2.6)
    ax2.set_ylim(0.005, 0.42)
    txt = "\n".join(
        f"{k}: {v['mean']:+.5f} mm/s [{v['ci_lo']:+.5f}, {v['ci_hi']:+.5f}], "
        f"{v['n_pass_gt0']}/{v['n_pass']} passes > 0"
        for k, v in st["rms_pair"].items()
    )
    if min(st["rms_stats"][t]["mean"] for t in ("A", "B", "C")) < 0:
        print("[track5] WARNING negative RMS mean -- log scale invalid")
    ax2.text(0.0, -0.235,
             "paired whole-pass differences (95% bootstrap CI over the 22 passes):\n" + txt,
             transform=ax2.transAxes, ha="left", va="top", fontsize=8.0, family="monospace")

    fig.suptitle("Fig 2 - whole-pass RMS residual per model (independent unit: one complete pass, n = 22)",
                 fontsize=12.5)
    out = FIGDIR / "fig2_per_pass_rms.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    PRODUCED.append(out)
    return out


# --------------------------------------------------------------------------
# fig 3 -- pass-averaged normalized Lomb-Scargle spectra
# --------------------------------------------------------------------------
def fig3(st):
    f = st["f_uhz"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14.0, 9.2), constrained_layout=True,
                                   gridspec_kw={"height_ratios": [1.35, 1.0]})
    for ax in (ax1, ax2):
        ax.axvspan(CAND_BAND[0], CAND_BAND[1], color="gold", alpha=0.22, lw=0, zorder=0)
        for c in COMB:
            ax.axvline(c, color="0.45", ls=":", lw=0.7, zorder=0)

    for tag in ("A", "B", "C"):
        ax1.fill_between(f, np.maximum(st["spec_p16"][tag], 1e-5), st["spec_p84"][tag],
                         color=COLORS[tag], alpha=0.14, lw=0)
        ax1.plot(f, st["spec_mean"][tag], color=COLORS[tag], ls=STYLES[tag], lw=1.2,
                 label=f"model {tag}  (pass-average)")
        ax1.plot([st["spec_peak"][tag]], [st["spec_mean"][tag].max()], marker="v", ms=7,
                 color=COLORS[tag], mec="k", mew=0.5)
    ax1.set_yscale("log")
    ax1.set_ylim(1e-4, 0.6)
    ax1.set_xlim(200, 3000)
    ax1.set_xlabel("frequency (microHz)", fontsize=10)
    ax1.set_ylabel("normalized Lomb-Scargle power", fontsize=10)
    ax1.set_title("pass-averaged normalized LS spectra (mean over the 22 passes; shaded = 16-84% pass envelope)",
                  fontsize=10.5)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(alpha=0.22, lw=0.5, which="both")
    ax1.text(0.012, 0.05,
             "mean-spectrum peaks:  A %.0f uHz,  B %.0f uHz,  C %.0f uHz" % (
                 st["spec_peak"]["A"], st["spec_peak"]["B"], st["spec_peak"]["C"]),
             transform=ax1.transAxes, fontsize=8.5)
    ax1.text(0.012, 0.115, "gold band = candidate band 860-1310 uHz;  dotted = 50-uHz candidate comb",
             transform=ax1.transAxes, fontsize=8.5)

    for tag in ("A", "B", "C"):
        ax2.fill_between(f, np.maximum(st["spec_p16"][tag], 1e-5), st["spec_p84"][tag],
                         color=COLORS[tag], alpha=0.16, lw=0)
        ax2.plot(f, st["spec_mean"][tag], color=COLORS[tag], ls=STYLES[tag], lw=1.4,
                 label=f"model {tag}")
    for c in COMB:
        ax2.text(c, 0.30, f"{c:.0f}", ha="center", va="top", fontsize=7.2, color="0.25",
                 rotation=90, zorder=4)
    ax2.set_yscale("log")
    ax2.set_xlim(800, 1400)
    ax2.set_ylim(3e-4, 0.45)
    ax2.set_xlabel("frequency (microHz)", fontsize=10)
    ax2.set_ylabel("power", fontsize=10)
    ax2.set_title("zoom 800-1400 uHz: candidate comb lines and the ranked-grid peak frequencies "
                  "(1010-1160 uHz ridge)", fontsize=10.5)
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(alpha=0.22, lw=0.5, which="both")
    bp = st["band_power"]
    ax2.text(0.985, 0.06,
             "band-mean power (whole-pass bootstrap):\n" + "\n".join(
                 f"  {t}: {bp[t]['mean']:.4f} [{bp[t]['ci_lo']:.4f}, {bp[t]['ci_hi']:.4f}]"
                 for t in ("A", "B", "C")),
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=7.6, family="monospace",
             bbox=dict(fc="white", ec="0.8", alpha=0.85, pad=1.6), zorder=6)

    fig.suptitle("Fig 3 - pass-averaged Lomb-Scargle spectra, 200-3000 uHz, candidate band 860-1310 uHz",
                 fontsize=12.5)
    out = FIGDIR / "fig3_periodograms.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    PRODUCED.append(out)
    return out


# --------------------------------------------------------------------------
# fig 4 -- information-criterion surfaces
# --------------------------------------------------------------------------
def fig4():
    cand = load_candidates()
    fa = cand[cand["candidate_family"] == "frequency_amplitude_scan"]
    mv = cand[cand["candidate_family"] == "minimum_velocity_scan"]
    uni = load_uniform()

    vmax = np.array(sorted(fa["max_velocity_cm_s"].unique()))
    fs = np.array(sorted(fa["peak_frequency_microhz"].unique()))
    piv = fa.pivot_table(index="max_velocity_cm_s", columns="peak_frequency_microhz",
                         values="information_criterion")
    Z = piv.loc[vmax, fs].values

    def log_edges(v):
        e = np.sqrt(v[:-1] * v[1:])
        return np.concatenate([[v[0] ** 2 / e[0]], e, [v[-1] ** 2 / e[-1]]])

    xe = np.concatenate([[fs[0] - 25.0], fs[:-1] + 25.0, [fs[-1] + 25.0]])
    ye = log_edges(vmax)

    fig = plt.figure(figsize=(16.5, 6.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0])
    ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(3))

    pcm = ax1.pcolormesh(xe, ye, Z, shading="flat", norm=matplotlib.colors.LogNorm(),
                         cmap=matplotlib.colormaps["viridis"])
    ax1.set_yscale("log")
    ax1.set_xlabel("peak frequency (microHz)", fontsize=10)
    ax1.set_ylabel("max velocity (cm/s, log scale)", fontsize=10)
    ax1.set_title("frequency_amplitude_scan: IC surface\n(304/304 cells, min velocity = 0, width = 300 uHz)",
                  fontsize=10)
    cb = fig.colorbar(pcm, ax=ax1, pad=0.02)
    cb.set_label("information criterion", fontsize=9)
    k = np.unravel_index(np.nanargmin(Z), Z.shape)
    ax1.plot(fs[k[1]], vmax[k[0]], marker="*", ms=17, color="white", mec="k", mew=0.8, zorder=5)
    ax1.annotate(f"min IC = {Z[k]:.1f}\nf = {fs[k[1]]:.0f} uHz, vmax = {vmax[k[0]]:.0f} cm/s",
                 xy=(fs[k[1]], vmax[k[0]]), xytext=(fs[k[1]] + 60, vmax[k[0]] * 3.4),
                 fontsize=8.5, color="white", arrowprops=dict(arrowstyle="->", color="white", lw=1.1))
    ax1.axvspan(CAND_BAND[0], CAND_BAND[1], color="gold", alpha=0.10, lw=0)

    ax2.plot(uni["uniform_velocity_cm_s"], uni["information_criterion"], "o-", color="#CC79A7",
             lw=1.6, ms=5)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    iu = int(np.argmin(uni["information_criterion"].values))
    umin = float(uni["information_criterion"].iloc[iu])
    uv = float(uni["uniform_velocity_cm_s"].iloc[iu])
    ax2.plot(uv, umin, marker="*", ms=16, color="k", zorder=5)
    ax2.axhline(Z[k], color=COLORS["B"], ls="--", lw=1.2)
    ax2.text(0.03, Z[k] * 1.12, f"best candidate IC = {Z[k]:.1f} (FA family)", color=COLORS["B"],
             fontsize=8.5)
    ax2.annotate(f"uniform min IC = {umin:.1f}\nat {uv:.2f} cm/s", xy=(uv, umin),
                 xytext=(uv * 0.12, umin * 0.35), fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", lw=1.0))
    ax2.set_xlabel("uniform velocity (cm/s, log scale)", fontsize=10)
    ax2.set_ylabel("information criterion (log scale)", fontsize=10)
    ax2.set_title("uniform-alternative IC vs velocity\n(24 cells; gap to best candidate is large)",
                  fontsize=10)
    ax2.grid(alpha=0.25, lw=0.5, which="both")

    fa_min_per_v = piv.min(axis=1).loc[vmax].values
    mv_min_per_v = mv.groupby("max_velocity_cm_s")["information_criterion"].min()
    mv_v = np.array(sorted(mv_min_per_v.index))
    ax3.plot(vmax, fa_min_per_v, "o-", color=COLORS["A"], lw=1.6, ms=5, label="FA family (min over f)")
    ax3.plot(mv_v, mv_min_per_v.loc[mv_v].values, "s--", color=COLORS["C"], lw=1.6, ms=5,
             label="MV family (min over min-velocity)")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.plot(vmax[np.argmin(fa_min_per_v)], fa_min_per_v.min(), marker="*", ms=16, color="k", zorder=5)
    ax3.annotate(f"{fa_min_per_v.min():.1f} @ {vmax[np.argmin(fa_min_per_v)]:.0f} cm/s",
                 xy=(vmax[np.argmin(fa_min_per_v)], fa_min_per_v.min()),
                 xytext=(vmax[np.argmin(fa_min_per_v)] * 0.16, fa_min_per_v.min() * 1.6),
                 fontsize=8.5, arrowprops=dict(arrowstyle="->", lw=1.0))
    ax3.set_xlabel("max velocity (cm/s, log scale)", fontsize=10)
    ax3.set_ylabel("best information criterion (log scale)", fontsize=10)
    ax3.set_title("best IC vs max velocity, both candidate families\nridge minimum at 20-30 cm/s",
                  fontsize=10)
    ax3.legend(fontsize=8.5, loc="upper right")
    ax3.grid(alpha=0.25, lw=0.5, which="both")

    fig.suptitle("Fig 4 - information-criterion surfaces (unit of analysis = one candidate parameter combination)",
                 fontsize=12.5)
    out = FIGDIR / "fig4_ic_surfaces.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    PRODUCED.append(out)
    return {
        "fa_min_ic": float(Z[k]), "fa_min_f": float(fs[k[1]]), "fa_min_vmax": float(vmax[k[0]]),
        "uniform_min_ic": umin, "uniform_min_v": uv,
        "mv_min_ic": float(mv["information_criterion"].min()),
        "fa_second_min_ic": float(np.sort(Z, axis=None)[1]), "out": out,
    }


# --------------------------------------------------------------------------
# fig 5 -- ranked-grid coverage / delta-IC vs rank
# --------------------------------------------------------------------------
def fig5():
    g = load_ranked_grid()
    ic = g["ic"].values
    rank = g["candidate_rank"].values
    levels = {
        "f (uHz)": (np.round(g["f"].values, 3), sorted(g["f"].unique().tolist())),
        "w (uHz)": (np.round(g["w"].values, 3), sorted(g["w"].unique().tolist())),
        "vmax (cm/s)": (g["vmax"].values, sorted(g["vmax"].unique().tolist())),
        "vmin (cm/s)": (np.round(g["vmin"].values, 3), sorted(set(np.round(g["vmin"], 3).tolist()))),
    }
    full_factorial = 7 * 8 * 10 * 4  # vmax x vmin x f x w levels

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.0), constrained_layout=True)
    axA, axB, axC, axD = axes.ravel()

    # (a) delta-IC vs rank, full truncated grid
    axA.plot(rank, ic, "-", color="0.35", lw=1.0)
    axA.plot(rank, ic, "o", ms=2.4, color="#0072B2")
    axA.axvspan(0, 20, color="gold", alpha=0.28, lw=0)
    axA.set_yscale("symlog", linthresh=1.0)
    axA.set_xlabel("candidate rank", fontsize=10)
    axA.set_ylabel(r"$\Delta$IC relative to rank 1 (symlog)", fontsize=10)
    axA.set_title("(a) ranked grid is truncated: 250 of %d factorial cells kept\n"
                  "max $\\Delta$IC = %.1f at rank %d; gold = top 20" % (full_factorial, ic[-1], rank[-1]),
                  fontsize=10)
    axA.grid(alpha=0.25, lw=0.5, which="both")
    for r_, i_ in zip(rank[:10], ic[:10]):
        axA.annotate(f"{i_:.1f}", xy=(r_, i_), xytext=(3, 4), textcoords="offset points", fontsize=7.0)
    axA.axvline(20, color="k", ls=":", lw=0.8)
    axA.text(20, ic[-1] * 0.75, " top-20 cut", fontsize=8.5)

    # (b) zoom of the top 20 with consecutive margins
    n = 20
    axB.plot(rank[:n], ic[:n], "o-", color="#0072B2", lw=1.3, ms=5.5)
    axB.set_xlabel("candidate rank", fontsize=10)
    axB.set_ylabel(r"$\Delta$IC relative to rank 1", fontsize=10)
    axB.set_title("(b) near-tie margins among the top cells:\nrank 6-7 margin = %.1f, rank 10-11 margin = %.1f"
                  % (ic[6] - ic[5], ic[10] - ic[9]), fontsize=10)
    axB.grid(alpha=0.25, lw=0.5)
    margins = np.diff(ic[:n])
    for i_, m_ in enumerate(margins):
        axB.annotate(f"+{m_:.1f}", xy=((rank[i_] + rank[i_ + 1]) / 2.0, (ic[i_] + ic[i_ + 1]) / 2.0),
                     xytext=(0, -13), textcoords="offset points", ha="center", fontsize=7.2)
    smallest = int(np.argmin(margins))
    axB.plot([rank[smallest], rank[smallest + 1]], [ic[smallest], ic[smallest + 1]], color="#D55E00",
             lw=2.4, zorder=4)
    axB.annotate("smallest margin\n$\\Delta$IC = %.1f (ranks %d, %d)" % (
        margins[smallest], rank[smallest], rank[smallest + 1]),
        xy=(rank[smallest], ic[smallest]), xytext=(14.0, 6.0),  # data coordinates, inside the axes
        fontsize=8.4, color="#D55E00", ha="center", va="center", zorder=6,
        arrowprops=dict(arrowstyle="->", lw=1.0, color="#D55E00"),
        bbox=dict(fc="white", ec="#D55E00", alpha=0.9, pad=1.6, lw=0.6))
    axB.set_ylim(-1.5, 34.0)
    axB.text(0.50, 0.975,
             "margins are $\\Delta$IC differences (evidence ratios on a log scale);\n"
             "they carry NO per-row standard error -- rows are not replicates",
             transform=axB.transAxes, fontsize=8.0, ha="center", va="top", zorder=6,
             bbox=dict(fc="white", ec="0.8", alpha=0.9, pad=2.0))

    # (c) first-appearance rank of every axis level (coverage vs truncation)
    ytick, ylab = [], []
    axC.set_xscale("log")
    for row, (name, (vals, lv)) in enumerate(levels.items()):
        for j, lv_val in enumerate(lv):
            hit = np.where(vals == lv_val)[0]
            first = int(rank[hit[0]]) if len(hit) else np.nan
            y = row + (j - (len(lv) - 1) / 2.0) / max(len(lv), 1) * 0.62
            axC.plot([first], [y], marker="o", ms=6.5,
                     color=matplotlib.colormaps["plasma"](j / max(len(lv) - 1, 1)))
            right = (j % 2 == 0)
            axC.annotate(f"{lv_val:g}", xy=(first, y), xytext=(5 if right else -5, 0),
                         textcoords="offset points", fontsize=6.8, va="center",
                         ha="left" if right else "right")
            ytick.append(y)
            ylab.append("")
    axC.axvspan(250, 3000, color="0.85", alpha=0.6, lw=0, zorder=0)
    axC.text(360, len(levels) - 0.55, "outside the ranked\n(truncated) grid", fontsize=8.0, color="0.3")
    axC.set_xlim(0.85, 800)
    axC.set_ylim(-0.6, len(levels) - 0.15)
    axC.set_yticks(range(len(levels)))
    axC.set_yticklabels(list(levels.keys()), fontsize=10)
    axC.set_xlabel("first rank at which the axis level appears", fontsize=10)
    axC.set_title("(c) axis coverage of the truncated ranking\n"
                  "f = 860 uHz only enters at rank %d" % int(rank[np.where(np.round(g["f"].values, 3) == 860)[0][0]]),
                  fontsize=10)
    axC.grid(alpha=0.25, lw=0.5, axis="x")

    # (d) coverage counts of each axis level in top-N windows
    Ns = [5, 10, 20, 31, 50, 100, 250]
    cats, rows = [], []
    for name, (vals, lv) in levels.items():
        for lv_val in lv:
            cats.append(f"{name.split(' ')[0]}={lv_val:g}")
            rows.append([int((vals[:N] == lv_val).sum()) for N in Ns])
    M = np.asarray(rows)
    im = axD.imshow(M, aspect="auto", cmap=matplotlib.colormaps["Blues"], origin="upper")
    axD.set_xticks(range(len(Ns)))
    axD.set_xticklabels([str(N) for N in Ns], fontsize=8.5)
    axD.set_yticks(range(len(cats)))
    axD.set_yticklabels(cats, fontsize=6.4)
    axD.set_xlabel("top-N window of the ranked grid", fontsize=10)
    axD.set_title("(d) level counts in each top-N window\n(0 = level absent from the window)", fontsize=10)
    fig.colorbar(im, ax=axD, pad=0.02, label="cells")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] == 0:
                axD.plot(j, i, marker="x", ms=4, color="#D55E00")

    fig.suptitle("Fig 5 - ranked full parameter grid: truncation, near-tie margins and axis coverage",
                 fontsize=12.5)
    out = FIGDIR / "fig5_grid_coverage.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    PRODUCED.append(out)
    return {
        "out": out, "n_ranked": int(len(g)), "n_factorial": int(full_factorial),
        "max_delta_ic": float(ic[-1]),
        "margins_top20": [float(x) for x in np.diff(ic[:20])],
        "min_margin_rank": int(rank[int(np.argmin(np.diff(ic[:20])))]),
        "min_margin_value": float(np.min(np.diff(ic[:20]))),
    }


# --------------------------------------------------------------------------
def main():
    t0 = time.time()
    traces = load_traces()
    st = build_stats()
    print(f"[track5] stats (incl. 66 LS transforms) in {time.time() - t0:.1f} s", flush=True)

    marks, returns = {}, {}
    for name, fn in (("fig1", lambda: fig1(traces, st)), ("fig2", lambda: fig2(st)),
                     ("fig3", lambda: fig3(st)), ("fig4", fig4), ("fig5", fig5)):
        t1 = time.time()
        returns[name] = fn()
        marks[name] = round(time.time() - t1, 2)
        print(f"[track5] {name} in {marks[name]:.1f} s", flush=True)
    f4, f5 = returns["fig4"], returns["fig5"]

    modes = load_modes()
    in_band = modes[(modes["mode_frequency_microhz"] >= CAND_BAND[0]) &
                    (modes["mode_frequency_microhz"] <= CAND_BAND[1])]

    annotations = {
        "generated_by": str(Path(__file__).resolve()),
        "n_passes": int(traces["pass_id"].nunique()),
        "n_samples": int(len(traces)),
        "rms_stats": st["rms_stats"],
        "rms_pair": st["rms_pair"],
        "rms_loso_range": {t: [min(v), max(v)] for t, v in st["rms_loso"].items()},
        "pass01_share_of_total_ms_model_A": float(
            st["rms"].loc["pass_01", "A"] ** 2 / (st["rms"]["A"] ** 2).sum()),
        "spectral_mean_peak_uhz": st["spec_peak"],
        "band_power": st["band_power"],
        "band_power_pair": st["band_power_pair"],
        "candidate_band_uhz": list(CAND_BAND),
        "candidate_comb_uhz": [float(c) for c in COMB],
        "modes_in_band": int(len(in_band)),
        "fig4": {k: v for k, v in f4.items() if k != "out"},
        "fig5": {k: v for k, v in f5.items() if k != "out"},
        "ls_spectra_elapsed_s": round(st["elapsed_build_s"], 2),
    }
    write_json(RESDIR / "track5_figure_annotations.json", annotations)

    cmd = f"{sys.executable} {Path(__file__).resolve()}"
    manifest = {
        "generated_by": str(Path(__file__).resolve()),
        "exact_command": cmd,
        "python": sys.executable,
        "matplotlib_backend": matplotlib.get_backend(),
        "matplotlib_version": matplotlib.__version__,
        "numpy_version": np.__version__,
        "figure_dir": str(FIGDIR),
        "total_elapsed_s": round(time.time() - t0, 2),
        "figures": [],
    }
    for p in PRODUCED:
        manifest["figures"].append({
            "file": p.name,
            "path": str(p),
            "bytes": int(p.stat().st_size),
            "command": cmd,
        })
    write_json(FIGDIR / "figures_manifest.json", manifest)

    print(f"[track5] backend={matplotlib.get_backend()} matplotlib={matplotlib.__version__}")
    for e in manifest["figures"]:
        print(f"[track5] wrote {e['path']} ({e['bytes']} bytes)")
    print(f"[track5] wrote {FIGDIR / 'figures_manifest.json'}")
    print(f"[track5] wrote {RESDIR / 'track5_figure_annotations.json'}")
    print(f"[track5] done in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())