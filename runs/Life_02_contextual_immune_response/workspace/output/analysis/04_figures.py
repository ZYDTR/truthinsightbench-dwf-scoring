#!/usr/bin/env python3
"""Publication-quality figures for the Life_02 contextual-immune-response task.

Every plotted quantity is read from the already-computed CSVs in
``output/results/`` (produced by scripts 01/02/03).  Nothing is re-derived from
the raw data except the time-resolved point-wise 95% CI of the condition means,
which is *not* stored in any result CSV (``time_resolved_contrasts.csv`` only
stores the paired *difference* CI); that one quantity is computed here from
``time_resolved_paired_wide.csv`` and is flagged in the printed captions.

Backend is Agg (no display).  Every figure is closed after saving and each file
is reported with its path and byte size.

Run from any directory:  python3 output/analysis/04_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent
RESULTS = OUTPUT / "results"
FIGURES = OUTPUT / "figures"

DPI = 200
CAPTIONS: list[tuple[str, str]] = []

# Consistent, colour-blind-safe palette.
COMPARTMENT_COLORS = {
    "compartment_1": "#1b6ca8",
    "compartment_2": "#e07b39",
    "compartment_3": "#2e8b57",
}
ENDPOINT_MARKERS = {"endpoint_count": "o", "endpoint_fraction": "^"}
ASSAY_COLORS = {"assay_1": "#1b6ca8", "assay_2": "#c0392b", "assay_3": "#2e8b57"}
ROLE_COLORS = {"baseline": "#1b6ca8", "perturbation": "#c0392b"}

plt.rcParams.update(
    {
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        "axes.axisbelow": True,
        "legend.frameon": False,
        "figure.autolayout": False,
    }
)


def read_results(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"missing result CSV: {path}")
    return pd.read_csv(path)


def save(fig, filename: str, caption: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    size = path.stat().st_size
    print(f"[figure] {path}  {size} bytes  ({size / 1024:.1f} KiB)")
    CAPTIONS.append((filename, caption))


# --------------------------------------------------------------------------
# Figure 1: compartment fold change, grouped bars, two endpoint panels
# --------------------------------------------------------------------------
def figure1() -> None:
    cells = read_results("compartment_cell_effects.csv")
    assays = sorted(cells["assay_id"].unique())
    compartments = sorted(cells["compartment_id"].unique())
    endpoints = ["endpoint_count", "endpoint_fraction"]

    n_per_cell = int(cells["n_replicates"].max())

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    width = 0.8 / len(compartments)
    x = np.arange(len(assays), dtype=float)

    for ax, endpoint in zip(axes, endpoints):
        block = cells[cells["endpoint_type"] == endpoint]
        for j, comp in enumerate(compartments):
            sub = block[block["compartment_id"] == comp].set_index("assay_id")
            values = np.array(
                [float(sub.loc[a, "fc_ratio_of_means"]) if a in sub.index else np.nan
                 for a in assays]
            )
            offsets = x - 0.4 + width * (j + 0.5)
            bars = ax.bar(
                offsets,
                values,
                width=width * 0.9,
                color=COMPARTMENT_COLORS.get(comp, None),
                edgecolor="black",
                linewidth=0.5,
                label=comp,
            )
            for bar, value in zip(bars, values):
                if np.isfinite(value):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        value + 0.02,
                        f"{value:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                    )
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [a.replace("assay_", "assay ") for a in assays]
        )
        ax.set_title(endpoint.replace("endpoint_", "endpoint: "))
        ax.set_ylim(0, float(cells["fc_ratio_of_means"].max()) * 1.22)
        ax.set_xlabel("assay")

    axes[0].set_ylabel("fold change (perturbation / baseline)")
    axes[0].text(
        0.02, 0.96,
        f"n = {n_per_cell} replicates per assay x endpoint x compartment cell\n"
        "dashed line: no change (fold change = 1)",
        transform=axes[0].transAxes, va="top", ha="left", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85,
                  edgecolor="0.7", linewidth=0.6),
    )
    handles = [
        Patch(facecolor=COMPARTMENT_COLORS[c], edgecolor="black", label=c)
        for c in compartments
    ]
    handles.append(
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.2,
               label="fold change = 1")
    )
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        "Compartment fold change by assay and endpoint type", fontsize=12, y=1.02
    )
    fig.tight_layout()

    values_all = cells["fc_ratio_of_means"]
    caption = (
        "Figure 1. Fold change (perturbation/baseline) for all 18 assay x endpoint x "
        f"compartment cells (n = {n_per_cell} replicates per cell; bars are the ratio of "
        "condition means). Left panel: endpoint_count; right panel: endpoint_fraction; "
        f"the dashed line marks no change (fold change = 1). Every one of the 18 cells lies "
        f"above 1 (range {values_all.min():.2f}-{values_all.max():.2f}, median "
        f"{values_all.median():.2f})."
    )
    save(fig, "fig1_compartment_fold_change.png", caption)


# --------------------------------------------------------------------------
# Figure 2: baseline level vs fold change, all 18 cells
# --------------------------------------------------------------------------
def figure2() -> None:
    cells = read_results("compartment_cell_effects.csv")
    endpoints = sorted(cells["endpoint_type"].unique())
    compartments = sorted(cells["compartment_id"].unique())

    fig, ax = plt.subplots(figsize=(7.6, 5.4))

    for endpoint in endpoints:
        for comp in compartments:
            sub = cells[
                (cells["endpoint_type"] == endpoint)
                & (cells["compartment_id"] == comp)
            ]
            ax.scatter(
                sub["baseline"],
                sub["fc_ratio_of_means"],
                marker=ENDPOINT_MARKERS[endpoint],
                s=95,
                facecolor=COMPARTMENT_COLORS[comp],
                edgecolor="black",
                linewidth=0.7,
                alpha=0.9,
                zorder=3,
            )
            for _, row in sub.iterrows():
                ax.annotate(
                    row["assay_id"].replace("assay_", "a"),
                    (row["baseline"], row["fc_ratio_of_means"]),
                    textcoords="offset points",
                    xytext=(5, 4),
                    fontsize=7,
                    color="0.35",
                )

    ax.set_xscale("log")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, zorder=2)
    ax.set_xlabel("baseline level (condition-mean readout, log10 scale)")
    ax.set_ylabel("fold change (perturbation / baseline)")
    ax.set_title("Baseline level vs fold change, all 18 assay x endpoint x compartment cells")

    spearman_r = cells[["baseline", "fc_ratio_of_means"]].corr(
        method="spearman"
    ).iloc[0, 1]
    pearson_log_r = np.corrcoef(
        np.log10(cells["baseline"].to_numpy(dtype=float)),
        cells["fc_ratio_of_means"].to_numpy(dtype=float),
    )[0, 1]
    ax.text(
        0.02, 0.97,
        f"Spearman(baseline, fold change) = {spearman_r:.3f}\n"
        f"Pearson(log10 baseline, fold change) = {pearson_log_r:.3f}\n"
        "dashed line: fold change = 1",
        transform=ax.transAxes, va="top", ha="left", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85,
                  edgecolor="0.7", linewidth=0.6),
    )
    comp_handles = [
        Line2D([0], [0], marker="s", linestyle="none", markersize=9,
               markerfacecolor=COMPARTMENT_COLORS[c], markeredgecolor="black",
               label=c)
        for c in compartments
    ]
    ep_handles = [
        Line2D([0], [0], marker=ENDPOINT_MARKERS[e], linestyle="none", markersize=9,
               markerfacecolor="0.85", markeredgecolor="black", label=e)
        for e in endpoints
    ]
    first = ax.legend(handles=comp_handles, title="compartment_id", loc="lower right")
    ax.add_artist(first)
    ax.legend(handles=ep_handles, title="endpoint_type", loc="upper right")
    fig.tight_layout()

    caption = (
        "Figure 2. Baseline level (log10 x-axis) versus fold change (y-axis) for all 18 "
        "assay x endpoint x compartment cells; marker shape encodes endpoint_type, colour "
        "encodes compartment_id and the dashed line marks fold change = 1. Baseline spans "
        f"{cells['baseline'].min():.1f} to {cells['baseline'].max():.1f} (about "
        f"{np.log10(cells['baseline'].max() / cells['baseline'].min()):.1f} log10 units), yet "
        f"fold change spans only {cells['fc_ratio_of_means'].min():.2f}-"
        f"{cells['fc_ratio_of_means'].max():.2f}: high-baseline compartments are not the "
        f"high-fold-change ones (Spearman rho = {spearman_r:.3f}; Pearson on log10 baseline "
        f"r = {pearson_log_r:.3f})."
    )
    save(fig, "fig2_baseline_vs_foldchange.png", caption)


# --------------------------------------------------------------------------
# Figure 3: intervention profiles, log y, replicate SD error bars
# --------------------------------------------------------------------------
def figure3() -> None:
    means = read_results("intervention_setting_means.csv")
    assays = sorted(means["assay_id"].unique())

    means = means.copy()
    means["setting_index"] = (
        means["intervention_setting_id"].str.extract(r"(\d+)").astype(int)
    )

    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    for assay in assays:
        sub = means[means["assay_id"] == assay].sort_values("setting_index")
        ax.errorbar(
            sub["setting_index"],
            sub["response_mean"],
            yerr=sub["response_sd"],
            marker="o",
            markersize=6,
            linewidth=1.8,
            capsize=4,
            elinewidth=1.2,
            color=ASSAY_COLORS[assay],
            label=assay,
            zorder=3,
        )
        low = sub.iloc[sub["response_mean"].argmin()]
        ax.annotate(
            f"lowest: setting {int(low['setting_index'])}",
            (low["setting_index"], low["response_mean"]),
            textcoords="offset points",
            xytext=(6, -14),
            fontsize=7.5,
            color=ASSAY_COLORS[assay],
        )

    ax.set_yscale("log")
    # Headroom above the assay_2 curve keeps the annotation box clear of the data.
    ax.set_ylim(0.6, 9000.0)
    ax.set_xticks(sorted(means["setting_index"].unique()))
    ax.set_xticklabels(
        [f"setting_{i}" for i in sorted(means["setting_index"].unique())]
    )
    ax.set_xlabel("intervention setting index")
    ax.set_ylabel("mean response (log10 scale)")
    ax.set_title(
        "Intervention response profiles per assay (log scale; error bars = replicate SD, n = 3)"
    )

    med = means.groupby("assay_id")["response_mean"].median()
    # label the pair that the ratio actually describes, rather than asserting a pair
    hi_assay = str(med.idxmax())
    lo_assay = str(med.idxmin())
    sep = float(med.max() / med.min())
    ax.text(
        0.02, 0.97,
        f"median response per assay: "
        + ", ".join(f"{a} = {med[a]:.3g}" for a in assays)
        + f"\nlargest pairwise median separation {hi_assay} vs {lo_assay} = {sep:.0f}x, "
        + "hence the log y-axis",
        transform=ax.transAxes, va="top", ha="left", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85,
                  edgecolor="0.7", linewidth=0.6),
    )
    ax.legend(title="assay", loc="lower right", ncol=3, fontsize=8)
    fig.tight_layout()

    lowest = means.loc[means.groupby("assay_id")["response_mean"].idxmin()]
    lowest_txt = ", ".join(
        f"{row['assay_id']} -> setting {int(row['setting_index'])}"
        for _, row in lowest.iterrows()
    )
    # describe the peak setting per assay from the data instead of asserting each differs
    highest = means.loc[means.groupby("assay_id")["response_mean"].idxmax()]
    peak_txt = ", ".join(
        f"{row['assay_id']} -> setting {int(row['setting_index'])}"
        for _, row in highest.iterrows()
    )
    order_ratio = float(med.max() / med.min())
    caption = (
        "Figure 3. Mean intervention response versus intervention setting index (1-5), one "
        "line per assay (n = 3 replicates per assay x setting; error bars are replicate SD). "
        "The y-axis is logarithmic because the assay medians span "
        f"{order_ratio:.0f}x, i.e. about {np.log10(order_ratio):.1f} orders of magnitude "
        f"(median assay_1 = {med['assay_1']:.3g}, assay_2 = {med['assay_2']:.3g}, "
        f"assay_3 = {med['assay_3']:.3g}). All three assays reach their minimum at the same "
        f"setting ({lowest_txt}); the peak setting is {peak_txt}."
    )
    save(fig, "fig3_intervention_profiles.png", caption)


# --------------------------------------------------------------------------
# Figure 4: time-resolved means with CI, and paired difference with OLS slope
# --------------------------------------------------------------------------
def figure4() -> None:
    contrasts = read_results("time_resolved_contrasts.csv").sort_values("time_units")
    paired = read_results("time_resolved_paired_wide.csv")
    roles = ["baseline", "perturbation"]

    # The point-wise 95% CI of each condition mean is NOT present in any result
    # CSV (contrasts hold only the paired-difference CI), so it is computed here
    # from the paired wide table.  Mean/sd/n reproduce the stored *_mean columns.
    pointwise = {}
    for role in roles:
        stats = []
        for time, grp in paired.groupby("time_units"):
            values = grp[role].to_numpy(dtype=float)
            n = values.size
            mean = float(values.mean())
            sd = float(values.std(ddof=1))
            # t(0.975, df=n-1) for n=4 is 3.18245; use scipy when available.
            try:
                from scipy import stats as _scipy_stats

                crit = float(_scipy_stats.t.ppf(0.975, n - 1))
            except Exception:
                crit = 3.18245
            half = crit * sd / np.sqrt(n)
            stats.append(
                {"time_units": float(time), "mean": mean, "ci": half, "sd": sd, "n": n}
            )
        pointwise[role] = pd.DataFrame(stats).sort_values("time_units")

    # Sanity: computed means must match the stored contrast means.  The pairing
    # table is written with 6 significant digits, so allow ~1e-2 absolute
    # tolerance (observed max drift is ~6e-4 percentage points).
    stored = contrasts.set_index("time_units")
    for role in roles:
        frame = pointwise[role].set_index("time_units")
        delta = float(np.abs(frame["mean"] - stored[f"{role}_mean"]).max())
        print(
            f"[check] recomputed {role} means vs stored contrasts: "
            f"max abs diff = {delta:.2e} pp (paired wide table is 6-sig-fig rounded)"
        )
        if not delta < 1e-2:
            raise RuntimeError(f"{role} mean mismatch vs stored contrasts: {delta}")

    difference = paired.groupby("time_units")["difference"].mean().reset_index()
    slope = float(np.polyfit(difference["time_units"], difference["difference"], 1)[0])

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))

    ax = axes[0]
    for role in roles:
        frame = pointwise[role]
        ax.errorbar(
            frame["time_units"],
            frame["mean"],
            yerr=frame["ci"],
            marker="o",
            markersize=6,
            linewidth=1.8,
            capsize=4,
            elinewidth=1.2,
            color=ROLE_COLORS[role],
            label=role,
            zorder=3,
        )
    ax.set_xlabel("time (time_units)")
    ax.set_ylabel("normalized response (%)")
    ax.set_title("Condition means with 95% CI (left: means)")
    ax.set_xticks(sorted(contrasts["time_units"].unique()))
    # Range taken from the plotted quantities so the widest CI bar is not clipped.
    roles_lo = min(
        float((pointwise[r]["mean"] - pointwise[r]["ci"]).min()) for r in roles
    )
    roles_hi = max(
        float((pointwise[r]["mean"] + pointwise[r]["ci"]).max()) for r in roles
    )
    pad = 3.0
    ax.set_ylim(np.floor(roles_lo - pad), np.ceil(roles_hi + pad))
    ax.legend(title="condition_role", loc="upper left")

    ax = axes[1]
    ax.errorbar(
        contrasts["time_units"],
        contrasts["difference_mean"],
        yerr=[
            contrasts["difference_mean"] - contrasts["difference_ci_low"],
            contrasts["difference_ci_high"] - contrasts["difference_mean"],
        ],
        marker="o",
        markersize=6,
        linewidth=1.8,
        capsize=4,
        elinewidth=1.2,
        color="#444444",
        label="paired difference (perturbation - baseline), 95% CI",
        zorder=3,
    )
    ax.axhline(0.0, color="#c0392b", linestyle="--", linewidth=1.3, zorder=2)
    times = difference["time_units"].to_numpy(dtype=float)
    ax.plot(
        times,
        np.polyval(np.polyfit(times, difference["difference"].to_numpy(dtype=float), 1), times),
        color="#1b6ca8",
        linewidth=1.8,
        linestyle="-",
        label=f"OLS slope = {slope:.3f} %/time unit",
        zorder=4,
    )
    ax.set_xlabel("time (time_units)")
    ax.set_ylabel("difference (percentage points)")
    ax.set_title("Paired difference with fitted slope (right: difference)")
    ax.set_xticks(sorted(contrasts["time_units"].unique()))
    ax.legend(loc="lower left", fontsize=8)
    ax.text(
        0.98, 0.96,
        f"slope = {slope:.3f} %/time unit\nno time point significantly differs\n"
        "from 0 (all paired p > 0.05)",
        transform=ax.transAxes, va="top", ha="right", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85,
                  edgecolor="0.7", linewidth=0.6),
    )

    fig.suptitle(
        "Time-resolved response: parallel rise of both conditions, flat difference",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    max_p = float(contrasts["paired_p"].max())
    caption = (
        "Figure 4. Left: mean normalized_response_percent versus time_units with one line "
        "per condition_role and error bars = 95% CI (n = 4 replicates per role per time "
        "point; the point-wise CI of each condition mean is computed in this script because "
        "the result CSVs store only the paired-difference CI). Right: paired difference "
        "(perturbation - baseline) versus time with a dashed line at 0 and the OLS "
        f"difference-series slope = {slope:.3f} %/time unit (drawn). Both roles rise "
        f"together from ~{pointwise['baseline']['mean'].iloc[0]:.0f}-"
        f"{pointwise['perturbation']['mean'].iloc[0]:.0f}% at time 3 to "
        f"~{pointwise['baseline']['mean'].iloc[-1]:.0f}-"
        f"{pointwise['perturbation']['mean'].iloc[-1]:.0f}% at time 21, while the paired "
        f"difference stays near zero and is non-significant at every time point (largest "
        f"paired p = {max_p:.3f})."
    )
    save(fig, "fig4_time_resolved.png", caption)


def main() -> int:
    print(f"matplotlib {matplotlib.__version__} backend={matplotlib.get_backend()}")
    print(f"results dir : {RESULTS}")
    print(f"figures dir : {FIGURES}")
    figure1()
    figure2()
    figure3()
    figure4()

    print()
    print("=" * 72)
    print("CAPTIONS")
    print("=" * 72)
    for filename, caption in CAPTIONS:
        path = FIGURES / filename
        print()
        print(f"-- {filename}  ({path.stat().st_size} bytes)")
        print(caption)
    return 0


if __name__ == "__main__":
    sys.exit(main())