"""Step 4 - inspectable figures for the report.

TruthInsightBench task: Information_03_signal_surface_invariants
Project: Robust Regions and Failure Boundaries in Replicated Input-Output Surfaces
T0 = 2022-01-01

Five figures are produced from data/ only.  Every annotated number is recomputed
here (never copied from the prompt) and echoed to stdout as a ``CHECK`` line so
the run is auditable.  Each figure is wrapped in its own try/except: a failing
figure keeps its traceback and the remaining figures are still delivered.

Run: python3 output/analysis/step4_figures.py
"""

from __future__ import annotations

import sys
import traceback

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

from common import (  # noqa: E402
    FIGURES,
    GRID,
    KEY4,
    KEY6,
    LOGS,
    cell_means,
    flush,
    load_panels,
    pearson,
    r2,
    rank1_profile,
    spearman,
)

LOG: list[str] = []
CHECKS = 0
FAILURES: list[str] = []

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 165,
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.fontsize": 8,
        "figure.constrained_layout.use": True,
    }
)

CHAN_COLORS = plt.get_cmap("viridis")(np.linspace(0.0, 0.92, 7))


def check(name: str, value, fmt: str = "{:.4f}") -> None:
    """Record one recomputed number in the audit log and on stdout."""
    global CHECKS
    CHECKS += 1
    if isinstance(value, (int, np.integer)) and fmt == "{:.4f}":
        text = str(int(value))
    elif isinstance(value, float):
        text = fmt.format(value)
    else:
        text = str(value)
    LOG.append(f"CHECK {name} = {text}")
    print(f"CHECK {name} = {text}", flush=True)


# ---------------------------------------------------------------- load data
frames = load_panels(LOG)
rep, ind = frames["rep"], frames["ind"]
rc = cell_means(rep, "ly")  # replicate-averaged replicated panel (cell level)
ic = cell_means(ind, "ly")  # replicate-averaged independent panel

chan_levels = sorted(rep["output_channel_index"].unique())
bin_levels = sorted(rep["signal_bin_index"].unique())

# ------------------------------------------------------------ shared pieces
M = (
    rc.pivot_table(
        index=["input_axis_1_index", "input_axis_2_index"],
        columns="output_channel_index",
        values="ly",
    )
    .sort_index()
    .values
)
prof = rank1_profile(M)  # 49 x 7 log surface, channel-centred SVD
LOADINGS = np.array(prof["loadings_comp1"])
GRID_SD = M.std(axis=0)  # ddof=0, matches t04 grid_sd
SING = np.array(prof["singular_values"])
EXPVAR = np.array(prof["explained_variance"])
CUMVAR = np.cumsum(EXPVAR)
N_CELLS_A1A2 = M.shape[0]

# per-channel mean OLS slope of log-response on input_axis_1_index (t04 definition)
x = np.arange(7)
SLOPES = np.empty(7)
for ch in chan_levels:
    sub = rc[rc["output_channel_index"] == ch]
    pa = sub.pivot_table(index="input_axis_2_index", columns="input_axis_1_index", values="ly")
    SLOPES[ch] = float(np.mean([np.polyfit(x, pa.loc[a].sort_index().values, 1)[0] for a in pa.index]))

# normalisation perturbations of the same rank-1 question (t05 definitions)
lin_cell = rep.groupby(GRID + ["output_channel_index"], as_index=False)["response_value"].mean()
Mlin = lin_cell.pivot_table(
    index=["input_axis_1_index", "input_axis_2_index"],
    columns="output_channel_index",
    values="response_value",
).sort_index().values
Mnorm = Mlin / Mlin.mean(axis=0, keepdims=True)
Mz = (M - M.mean(axis=0, keepdims=True)) / M.std(axis=0, keepdims=True)
NORM = {
    "linear": rank1_profile(Mlin)["rank1_r2"],
    "log": rank1_profile(M)["rank1_r2"],
    "linear_per_channel_mean": rank1_profile(Mnorm)["rank1_r2"],
    "log_z_scored": rank1_profile(Mz)["rank1_r2"],
}

# bin-0 boundary: quadratic trend fitted on bins >= 1 of the bin means
x_all = np.arange(len(bin_levels))
rep_bin = rc.groupby("signal_bin_index")["ly"].mean().reindex(bin_levels)
co_rep = np.polyfit(x_all[1:], rep_bin.values[1:], 2)
REP_BIN0_OBS = float(rep_bin.values[0])
REP_BIN0_TREND = float(np.polyval(co_rep, 0))
REP_BIN0_EXCESS = REP_BIN0_OBS - REP_BIN0_TREND

x_ind = np.arange(len(sorted(ind["signal_bin_index"].unique())))
ind_bin = ic.groupby("signal_bin_index")["ly"].mean().sort_index()
co_ind = np.polyfit(x_ind[1:], ind_bin.values[1:], 2)
IND_BIN0_OBS = float(ind_bin.values[0])
IND_BIN0_TREND = float(np.polyval(co_ind, 0))
IND_BIN0_EXCESS = IND_BIN0_OBS - IND_BIN0_TREND

# cross-panel matched cells: replicated bins 0-6 x channels 0-3 == independent grid
common = rep[(rep["signal_bin_index"] < 7) & (rep["output_channel_index"] < 4)]
matched = common.merge(ind, on=KEY6, suffixes=("_rep", "_ind"))


def cell_cv(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(KEY4)["response_value"]
    return pd.DataFrame({"cv": g.std() / g.mean()}).reset_index()


cell_pair = matched.groupby(KEY4, as_index=False).agg(ly_rep=("ly_rep", "mean"), ly_ind=("ly_ind", "mean"))
cv_pair = cell_cv(common).merge(cell_cv(ind), on=KEY4, suffixes=("_rep", "_ind"))
CELL_R = pearson(cell_pair["ly_rep"], cell_pair["ly_ind"])
CELL_RHO = spearman(cell_pair["ly_rep"], cell_pair["ly_ind"])
CV_R = pearson(cv_pair["cv_rep"], cv_pair["cv_ind"])
_rng = np.random.default_rng(5)
_null = np.array([pearson(cv_pair["cv_rep"], _rng.permutation(cv_pair["cv_ind"].values)) for _ in range(5000)])
CV_P = float((np.abs(_null) >= abs(CV_R)).mean())

check("fig1.n_channels", int(len(chan_levels)))
check("fig1.slope_ch0", float(SLOPES[0]))
check("fig1.slope_ch6", float(SLOPES[6]))
check("fig1.slope_ratio_ch6_over_ch0", float(SLOPES[6] / SLOPES[0]), "{:.2f}")
check("fig2.rank1_r2_log", float(prof["rank1_r2"]))
check("fig2.rank2_r2_log", float(prof["rank2_r2"]))
check("fig2.norm_r2_linear", float(NORM["linear"]))
check("fig2.norm_r2_log", float(NORM["log"]))
check("fig2.norm_r2_per_channel_mean", float(NORM["linear_per_channel_mean"]))
check("fig2.norm_r2_z_scored", float(NORM["log_z_scored"]))
check("fig3.rep_bin0_observed", REP_BIN0_OBS)
check("fig3.rep_bin0_quadratic_trend", REP_BIN0_TREND)
check("fig3.rep_bin0_excess", REP_BIN0_EXCESS, "{:+.4f}")
check("fig3.rep_bin0_fold", float(np.exp(REP_BIN0_EXCESS)), "{:.2f}")
check("fig3.ind_bin0_observed", IND_BIN0_OBS)
check("fig3.ind_bin0_quadratic_trend", IND_BIN0_TREND)
check("fig3.ind_bin0_excess", IND_BIN0_EXCESS, "{:+.4f}")
check("fig3.ind_bin0_fold", float(np.exp(IND_BIN0_EXCESS)), "{:.2f}")
check("fig4.n_cells", int(len(cell_pair)))
check("fig4.cell_pearson", CELL_R, "{:+.4f}")
check("fig4.cell_spearman", CELL_RHO, "{:+.4f}")
check("fig4.cv_pearson", CV_R, "{:+.4f}")
check("fig4.cv_perm_p", CV_P, "{:.5f}")
check("fig5.loadings", np.array2string(LOADINGS, precision=4))
check("fig5.grid_sd", np.array2string(GRID_SD, precision=4))
check("fig5.corr_abs_loadings_vs_grid_sd", pearson(np.abs(LOADINGS), GRID_SD), "{:+.4f}")


# ---------------------------------------------------------------- figure 1
def fig1_channel_amplification() -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.4))

    for ch in chan_levels:
        p = rc.groupby(["output_channel_index", "input_axis_1_index"])["ly"].mean().loc[ch]
        ax.plot(
            p.index.values,
            p.values,
            "-o",
            ms=4,
            lw=1.5,
            color=CHAN_COLORS[ch],
            label=f"output_channel_index = {ch}",
        )
    ax.set_xlabel("input_axis_1_index")
    ax.set_ylabel("replicate-averaged log(response_value)")
    ax.set_title("Log-response vs input axis 1, one line per output channel (replicated panel)")
    ax.set_xticks(x)
    ax.legend(loc="upper left", framealpha=0.9)

    ax2.plot(chan_levels, SLOPES, "-o", color="#1f4e79", ms=6, lw=1.8)
    ax2.set_yscale("log")
    for ch, s in zip(chan_levels, SLOPES):
        ax2.annotate(f"{s:.4f}", (ch, s), textcoords="offset points", xytext=(6, 4), fontsize=7.5)
    ax2.set_xlabel("output_channel_index")
    ax2.set_ylabel("mean OLS slope of log-response on input_axis_1_index (log scale)")
    ax2.set_title("Per-channel gain to input axis 1 (log y-axis)")
    ax2.set_xticks(list(chan_levels))
    ax2.annotate(
        f"ch6 / ch0 = {SLOPES[6] / SLOPES[0]:.1f}$\\times$\n({SLOPES[0]:.4f} $\\rightarrow$ {SLOPES[6]:.4f})",
        xy=(0.5, 0.16),
        xycoords="axes fraction",
        ha="center",
        fontsize=8.5,
        bbox=dict(boxstyle="round", fc="#fff6d5", ec="#c8a63a"),
    )

    path = FIGURES / "fig1_channel_amplification.png"
    fig.savefig(path)
    plt.close(fig)
    LOG.append(f"wrote {path}")


# ---------------------------------------------------------------- figure 2
def fig2_rank1_spectrum() -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.4))

    ks = np.arange(1, len(SING) + 1)
    ax.bar(ks, SING, color="#4878a8", width=0.62)
    ax.set_yscale("log")
    for k, s, ev in zip(ks, SING, EXPVAR):
        ax.annotate(f"{100 * ev:.2f}%", (k, s), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=7.5)
    ax.set_xlabel("singular value index of the 49$\\times$7 (a1,a2)-by-channel log surface")
    ax.set_ylabel("singular value (log scale)")
    ax.set_title("Singular-value spectrum of the replicate-averaged log surface")
    ax.set_xticks(ks)
    ax.annotate(
        f"leading singular value {SING[0]:.2f}\n"
        f"rank-1 $R^2$ = {prof['rank1_r2']:.4f}\nrank-2 $R^2$ = {prof['rank2_r2']:.4f}",
        xy=(0.97, 0.93),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round", fc="#eaf1f8", ec="#4878a8"),
    )

    ax2.plot(ks, CUMVAR, "-o", color="#1f4e79", ms=6, lw=1.8)
    for k, c in zip(ks, CUMVAR):
        if k <= 3:
            ax2.annotate(f"{c:.4f}", (k, c), textcoords="offset points", xytext=(8, -2), fontsize=8)
    ax2.set_ylim(0.5, 1.06)
    ax2.set_xlabel("number of retained singular components")
    ax2.set_ylabel("cumulative explained variance")
    ax2.set_title("Cumulative explained variance and normalisation perturbation")
    ax2.set_xticks(ks)
    ax2.axhline(prof["rank1_r2"], color="#1f4e79", ls="--", lw=1.0, alpha=0.7)

    marks = [
        ("linear", NORM["linear"], "#c0392b"),
        ("log", NORM["log"], "#1f4e79"),
        ("per-channel-mean", NORM["linear_per_channel_mean"], "#e08a1e"),
        ("z-scored", NORM["log_z_scored"], "#6b3fa0"),
    ]
    for label, value, color in marks:
        ax2.plot([0.75, 2.4], [value, value], color=color, lw=1.1, alpha=0.55)
        ax2.annotate(
            f"{label} {value:.4f}",
            xy=(2.45, value),
            fontsize=8,
            va="center",
            color=color,
        )
    ax2.annotate(
        f"log rank-1 $R^2$ = {prof['rank1_r2']:.4f}",
        xy=(4.6, prof["rank1_r2"] + 0.012),
        fontsize=8.5,
        color="#1f4e79",
    )

    path = FIGURES / "fig2_rank1_spectrum.png"
    fig.savefig(path)
    plt.close(fig)
    LOG.append(f"wrote {path}")


# ---------------------------------------------------------------- figure 3
def fig3_bin0_notch() -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.8))

    ax.plot(bin_levels, rep_bin.values, "-o", color="#1f4e79", lw=1.9, ms=6,
            label="replicated panel, mean log-response")
    ax.plot(x_all, np.polyval(co_rep, x_all), "--", color="#c0392b", lw=1.4,
            label="quadratic trend fitted to replicated bins 1-7")
    ax.annotate(
        f"bin 0 excess = {REP_BIN0_EXCESS:+.4f}\n(observed {REP_BIN0_OBS:.4f} vs trend {REP_BIN0_TREND:.4f};\n"
        f"$\\times${np.exp(REP_BIN0_EXCESS):.2f} on the raw scale)",
        xy=(0, REP_BIN0_OBS),
        xytext=(1.15, 11.05),
        fontsize=8.5,
        color="#c0392b",
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
        bbox=dict(boxstyle="round", fc="#fdecea", ec="#c0392b"),
    )

    ind_bins = ind_bin.index.values
    ax.plot(ind_bins, ind_bin.values, "-s", color="#e08a1e", lw=1.9, ms=5.5,
            label="independent panel, mean log-response")
    ax.plot(x_ind, np.polyval(co_ind, x_ind), ":", color="#7d5109", lw=1.6,
            label="quadratic trend fitted to independent bins 1-6")
    ax.annotate(
        f"independent bin 0 excess = {IND_BIN0_EXCESS:+.4f}\n($\\times${np.exp(IND_BIN0_EXCESS):.2f}) - opposite sign,\n"
        "the notch does not transfer",
        xy=(0, IND_BIN0_OBS),
        xytext=(0.7, 8.75),
        fontsize=8.5,
        color="#7d5109",
        arrowprops=dict(arrowstyle="->", color="#7d5109", lw=1.2),
        bbox=dict(boxstyle="round", fc="#fdf3e3", ec="#7d5109"),
    )

    ax.set_xlabel("signal_bin_index")
    ax.set_ylabel("replicate-averaged log(response_value)")
    ax.set_title("Lowest signal bin carries a positive notch in the replicated panel only")
    ax.set_xticks(sorted(set(list(bin_levels) + list(ind_bins))))
    ax.legend(loc="lower right", framealpha=0.92)

    path = FIGURES / "fig3_bin0_notch.png"
    fig.savefig(path)
    plt.close(fig)
    LOG.append(f"wrote {path}")


# ---------------------------------------------------------------- figure 4
def fig4_cross_panel_noise() -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.6))

    ax.scatter(cell_pair["ly_rep"], cell_pair["ly_ind"], s=6, alpha=0.35, color="#4878a8",
               edgecolors="none")
    lo = float(min(cell_pair["ly_rep"].min(), cell_pair["ly_ind"].min()))
    hi = float(max(cell_pair["ly_rep"].max(), cell_pair["ly_ind"].max()))
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.1, color="#888888", label="identity")
    b, a0 = np.polyfit(cell_pair["ly_rep"], cell_pair["ly_ind"], 1)
    xs = np.linspace(lo, hi, 20)
    ax.plot(xs, b * xs + a0, "-", lw=1.6, color="#c0392b",
            label=f"OLS fit (slope {b:.3f})")
    ax.set_xlabel("input_axis_1_index x input_axis_2_index x signal_bin_index x output_channel_index\n"
                  "cell mean log-response, replicated panel")
    ax.set_ylabel("cell mean log-response,\nindependent panel")
    ax.set_title(f"Cell-level correspondence is absent (n = {len(cell_pair)} common cells)")
    ax.annotate(
        f"Pearson r = {CELL_R:+.4f}\nSpearman rho = {CELL_RHO:+.4f}\n"
        f"OLS $R^2$ = {r2(cell_pair['ly_ind'], b * cell_pair['ly_rep'] + a0):.5f}",
        xy=(0.03, 0.96),
        xycoords="axes fraction",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round", fc="#fdecea", ec="#c0392b"),
    )
    ax.legend(loc="lower right", framealpha=0.92)

    ax2.scatter(cv_pair["cv_rep"], cv_pair["cv_ind"], s=6, alpha=0.35, color="#2e7d5b",
                edgecolors="none")
    c = float(max(cv_pair["cv_rep"].max(), cv_pair["cv_ind"].max()))
    ax2.plot([0, c], [0, c], ls="--", lw=1.1, color="#888888", label="identity")
    b2, a2 = np.polyfit(cv_pair["cv_rep"], cv_pair["cv_ind"], 1)
    xs2 = np.linspace(0, c, 20)
    ax2.plot(xs2, b2 * xs2 + a2, "-", lw=1.6, color="#1f4e79", label=f"OLS fit (slope {b2:.3f})")
    ax2.set_xlabel("within-cell coefficient of variation,\nreplicated panel")
    ax2.set_ylabel("within-cell coefficient of variation,\nindependent panel")
    ax2.set_title("Within-cell dispersion IS shared (n = {} cells)".format(len(cv_pair)))
    ax2.annotate(
        f"Pearson r = {CV_R:+.4f}\npermutation p < {max(CV_P, 1e-5):.5f} (5000 perms)\n"
        "same cells as the left panel",
        xy=(0.03, 0.96),
        xycoords="axes fraction",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round", fc="#e8f4ee", ec="#2e7d5b"),
    )
    ax2.legend(loc="lower right", framealpha=0.92)

    path = FIGURES / "fig4_cross_panel_noise.png"
    fig.savefig(path)
    plt.close(fig)
    LOG.append(f"wrote {path}")


# ---------------------------------------------------------------- figure 5
def fig5_gain_axis_loadings() -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.5))

    pos = np.arange(len(chan_levels))
    ax.bar(pos - 0.2, LOADINGS, width=0.4, color="#1f4e79",
           label="leading singular vector loading (left axis)")
    ax.set_ylabel("leading singular vector loading\n(49$\\times$7 channel-centred log surface)")
    ax.set_xlabel("output_channel_index")
    ax.set_xticks(pos)
    ax.set_xticklabels([str(c) for c in chan_levels])
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_title("Single gain axis: component-1 channel loadings")

    axb = ax.twinx()
    axb.bar(pos + 0.2, GRID_SD, width=0.4, color="#e08a1e",
            label="grid SD of log-response (right axis)")
    axb.set_ylabel("grid SD of log-response across the 49 (a1,a2) cells")
    axb.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axb.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", framealpha=0.92)

    ax2.scatter(np.abs(LOADINGS), GRID_SD, s=70, color="#6b3fa0", zorder=3)
    for ch in chan_levels:
        ax2.annotate(f"ch{ch}", (abs(LOADINGS[ch]), GRID_SD[ch]), textcoords="offset points",
                     xytext=(7, -3), fontsize=8)
    lim = float(max(np.abs(LOADINGS).max(), GRID_SD.max())) * 1.15
    ax2.plot([0, lim], [0, lim], ls="--", lw=1.1, color="#888888", label="identity")
    ax2.set_xlim(-0.02, lim)
    ax2.set_ylim(-0.02, lim)
    ax2.set_xlabel("|leading singular vector loading| per output_channel_index")
    ax2.set_ylabel("grid SD of log-response per channel\n(49 (a1,a2) cells)")
    ax2.set_title("Loadings track channel grid spread")
    ax2.annotate(
        f"Pearson r(|loading|, grid SD) = {pearson(np.abs(LOADINGS), GRID_SD):+.4f}\n"
        f"grid SD ch0 {GRID_SD[0]:.4f} $\\rightarrow$ ch6 {GRID_SD[6]:.4f} ($\\times${GRID_SD[6] / GRID_SD[0]:.1f})",
        xy=(0.03, 0.96),
        xycoords="axes fraction",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round", fc="#f0eaf8", ec="#6b3fa0"),
    )
    ax2.legend(loc="lower right", framealpha=0.92)

    path = FIGURES / "fig5_gain_axis_loadings.png"
    fig.savefig(path)
    plt.close(fig)
    LOG.append(f"wrote {path}")


# ---------------------------------------------------------------- run them
FIGURES.mkdir(parents=True, exist_ok=True)
for name, fn in [
    ("fig1_channel_amplification", fig1_channel_amplification),
    ("fig2_rank1_spectrum", fig2_rank1_spectrum),
    ("fig3_bin0_notch", fig3_bin0_notch),
    ("fig4_cross_panel_noise", fig4_cross_panel_noise),
    ("fig5_gain_axis_loadings", fig5_gain_axis_loadings),
]:
    try:
        fn()
        print(f"OK {name}", flush=True)
    except Exception:
        FAILURES.append(name)
        tb = traceback.format_exc()
        LOG.append(f"FAILED {name}\n{tb}")
        print(f"FAILED {name}\n{tb}", flush=True)

LOG.append(f"checks emitted: {CHECKS}")
LOG.append(f"figures ok: {5 - len(FAILURES)}/5; failures: {FAILURES}")
flush(LOG, LOGS / "step4.log")
print(f"checks emitted: {CHECKS}", flush=True)
print(f"figures ok: {5 - len(FAILURES)}/5; failures: {FAILURES}", flush=True)
sys.exit(len(FAILURES))