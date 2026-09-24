"""Figures for the interface response-boundary task.

Every figure is written as PNG plus a CSV of the exact plotted numbers so the
figures can be checked without re-running matplotlib.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tib_stats as ts  # noqa: E402
from run_stage_e import fit_logistic, logistic  # noqa: E402

PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "data"
FIG = Path(__file__).resolve().parents[1] / "figures"
TAB = Path(__file__).resolve().parents[1] / "results"
FIG.mkdir(parents=True, exist_ok=True)


def load():
    m = pd.read_csv(DATA / "anonymous_condition_response_curves.tsv", sep="\t")
    c = pd.read_csv(DATA / "independent_condition_check_curves.tsv", sep="\t")
    return (m.pivot(index="condition_id", columns="temperature_like_setting",
                    values="rate_like_response"),
            c.pivot(index="condition_id", columns="temperature_like_setting",
                    values="rate_like_response"))


def fig1_curves(mp, cp):
    T = mp.columns.values.astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    rows = []
    for ax, (lab, piv) in zip(axes, (("main", mp), ("check", cp))):
        for cid in piv.index:
            y = piv.loc[cid].values.astype(float)
            ax.plot(T, y, marker="o", ms=4, lw=1.4, label=cid)
            for t, v in zip(T, y):
                rows.append({"panel": lab, "condition_id": cid,
                             "temperature_like_setting": float(t),
                             "rate_like_response": float(v)})
        ax.set_xlabel("temperature-like setting (K)")
        ax.set_ylabel("rate-like response")
        ax.set_title(f"{lab} panel: raw condition-response curves")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Saturating rise of every condition across the setting range", y=1.00)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_response_curves.png", dpi=150)
    plt.close(fig)
    with (TAB / "fig1_data.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("[figure] figures/fig1_response_curves.png")


def fig2_ordering(mp):
    T = mp.columns.values.astype(float)
    ranks = mp.rank(axis=0, method="average")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for cid in ranks.index:
        axes[0].plot(T, ranks.loc[cid].values, marker="o", ms=4, lw=1.4, label=cid)
    axes[0].set_xlabel("temperature-like setting (K)")
    axes[0].set_ylabel("rank among 8 curves (1 = lowest response)")
    axes[0].set_title("Rank trajectories cross repeatedly")
    axes[0].invert_yaxis()
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)

    r1 = mp[T[0]].rank().values
    r2 = mp[T[-1]].rank().values
    for cid, a, b in zip(mp.index, r1, r2):
        axes[1].plot([0, 1], [a, b], marker="o", ms=5)
        axes[1].annotate(cid.replace("condition_", ""), (0, a), fontsize=7,
                         xytext=(-4, 0), textcoords="offset points", ha="right")
        axes[1].annotate(cid.replace("condition_", ""), (1, b), fontsize=7,
                         xytext=(4, 0), textcoords="offset points", ha="left")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels([f"{int(T[0])} K", f"{int(T[-1])} K"])
    axes[1].set_ylabel("rank")
    axes[1].set_xlim(-0.45, 1.45)
    axes[1].invert_yaxis()
    axes[1].set_title("Ordering is not preserved end to end\n"
                      f"(Spearman rho = {ts.spearman(r1, r2)[0]:+.2f})")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_rank_ordering.png", dpi=150)
    plt.close(fig)
    with (TAB / "fig2_data.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["condition_id"] + [f"rank@{int(t)}" for t in T])
        for cid in ranks.index:
            w.writerow([cid] + [f"{v:.1f}" for v in ranks.loc[cid].values])
    print("[figure] figures/fig2_rank_ordering.png")


def fig3_model(mp, cp):
    T = mp.columns.values.astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    rows = []
    for ax, (lab, piv) in zip(axes, (("main", mp), ("check", cp))):
        for cid in piv.index:
            y = piv.loc[cid].values.astype(float)
            m = y > 0
            pred = logistic(T[m], 100.0, fit_logistic(T, y)["T50"] if False else 100.0, 30.0)
            ax.scatter(T[m], np.log(y[m]), s=18, label=cid)
        # overlay fitted quadratic-in-T log-response for each curve
        for cid in piv.index:
            y = piv.loc[cid].values.astype(float)
            m = (T >= 140) & (y > 0)
            f = ts.ols(np.column_stack([np.ones(m.sum()), T[m], T[m] ** 2]),
                       np.log(y[m]))
            grid = np.linspace(T[m].min(), T[m].max(), 100)
            ax.plot(grid, f["beta"][0] + f["beta"][1] * grid + f["beta"][2] * grid ** 2,
                    lw=1.2)
        ax.set_xlabel("temperature-like setting (K)")
        ax.set_ylabel("log(rate-like response)")
        ax.set_title(f"{lab} panel: log-response is concave (quadratic fits)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_log_response_concavity.png", dpi=150)
    plt.close(fig)
    for lab, piv in (("main", mp), ("check", cp)):
        for cid in piv.index:
            y = piv.loc[cid].values.astype(float)
            m = (T >= 140) & (y > 0)
            f = ts.ols(np.column_stack([np.ones(m.sum()), T[m], T[m] ** 2]),
                       np.log(y[m]))
            rows.append({"panel": lab, "condition_id": cid,
                         "quadratic_c": float(f["beta"][2]),
                         "c_se": float(f["se"][2]),
                         "p_one_sided_c_less_0": float(f["p"][2] / 2),
                         "r2": float(f["r2"])})
    with (TAB / "fig3_data.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("[figure] figures/fig3_log_response_concavity.png")


def fig4_forward_holdout(mp):
    """Fit on T<=280, predict 300 and 320: logistic vs exponential."""
    T = mp.columns.values.astype(float)
    train, test = T <= 280, T > 280
    fig, ax = plt.subplots(figsize=(9, 5.6))
    rows = []
    for cid in mp.index:
        y = mp.loc[cid].values.astype(float)
        fl = fit_logistic(T[train], y[train])
        fe = ts.ols(np.column_stack([np.ones((train & (y > 0)).sum()),
                                     T[train & (y > 0)]]),
                    np.log(y[train & (y > 0)]))
        lg = logistic(T, fl["ymax"], fl["T50"], fl["w"])
        ex = np.exp(fe["beta"][0] + fe["beta"][1] * T)
        ax.plot(T, y, "o", ms=4, color="black", zorder=5)
        ax.plot(T, lg, "-", lw=1.3)
        ax.plot(T[train], ex[train], ":", lw=1.0)
        ax.plot(T[test], ex[test], ":", lw=1.0, color="red")
        rows.append({"condition_id": cid,
                     "pred_logistic_320": float(lg[-1]),
                     "pred_exponential_320": float(ex[-1]),
                     "observed_320": float(y[-1]),
                     "logistic_abs_err_320": abs(float(lg[-1]) - float(y[-1])),
                     "exponential_abs_err_320": abs(float(ex[-1]) - float(y[-1]))})
    ax.set_yscale("log")
    ax.set_xlabel("temperature-like setting (K)")
    ax.set_ylabel("rate-like response (log scale)")
    ax.set_title("Forward holdout: fit T<=280, extrapolate to 300/320 K\n"
                 "solid = saturating fit, dotted = unsaturated exponential")
    ax.axvspan(280, 325, color="grey", alpha=0.12)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_forward_holdout.png", dpi=150)
    plt.close(fig)
    with (TAB / "fig4_data.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("[figure] figures/fig4_forward_holdout.png")


def main():
    mp, cp = load()
    fig1_curves(mp, cp)
    fig2_ordering(mp)
    fig3_model(mp, cp)
    fig4_forward_holdout(mp)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()