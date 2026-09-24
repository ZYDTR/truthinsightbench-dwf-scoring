"""Summary figures for the registered findings.

Produces output/figures/summary_findings.png with four panels, one per
registered finding, plus a standalone error-vs-magnitude panel.

matplotlib has no display in this environment, so the Agg backend is forced.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "font.size": 8.5, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True})

C_Q, C_L, C_R = "#1f77b4", "#d95f02", "#7570b3"


def panel_coverage(ax, t):
    """F1: database domain boundary."""
    db = common.load_db()
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    db_hac = np.array([Chem.MolFromSmiles(m).GetNumHeavyAtoms() for m in db.molecule.unique()])
    bench_hac = t.hac_parent.to_numpy(int)

    bins = np.arange(0.5, 39.5, 1)
    ax.hist(db_hac, bins=bins, color="#bbbbbb", label=f"database molecules (n={len(db_hac)})")
    ax.hist(bench_hac, bins=bins, color=C_Q, alpha=0.75,
            label=f"benchmark observations (n={len(bench_hac)})")
    ax.axvline(10.5, color="crimson", ls="--", lw=1.4)
    ax.annotate("database hard ceiling\nHAC = 10", xy=(10.5, ax.get_ylim()[1] * 0.62),
                xytext=(15, ax.get_ylim()[1] * 0.72), color="crimson", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="crimson"))
    n_out = int((bench_hac > 10).sum())
    ax.text(0.97, 0.55, f"{n_out}/543 benchmark rows\n({100*n_out/543:.1f}%) lie beyond\nthe database domain",
            transform=ax.transAxes, ha="right", va="center", fontsize=8,
            bbox=dict(fc="white", ec="crimson", alpha=0.9))
    ax.set_xlabel("parent heavy-atom count (HAC)")
    ax.set_ylabel("count")
    ax.set_title("F1  Database coverage boundary", loc="left", fontweight="bold")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_xlim(0, 39)


def panel_floor(ax, t, floor):
    """F3: each channel's MAE against the experimental noise floor."""
    names = ["quantum", "learned", "reference"]
    colours = [C_Q, C_L, C_R]
    maes = [t[f"{c}_ae"].mean() for c in names]
    oracle = floor["noise_floor"]["oracle_mae"]
    lo, hi = floor["noise_floor"]["oracle_mae_ci95"]

    y = np.arange(len(names))
    ax.barh(y, maes, color=colours, height=0.55)
    ax.axvspan(lo, hi, color="crimson", alpha=0.15,
               label=f"reference noise floor\n(oracle MAE {oracle:.2f}, CI {lo:.2f}-{hi:.2f})")
    ax.axvline(oracle, color="crimson", ls="--", lw=1.3)
    for yi, (m, n) in enumerate(zip(maes, names)):
        ratio = m / oracle
        ax.text(m + 0.06, yi, f"{m:.2f}  ({ratio:.2f}x floor)", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("MAE vs experimental reference (kcal/mol)")
    ax.set_title("F3  Error relative to the reference's own heterogeneity",
                 loc="left", fontweight="bold")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.set_xlim(0, max(maes) * 1.45)


def panel_size_interaction(ax, t, inter):
    """F2: the quantum-vs-learned reversal with molecule size."""
    bands = [("HAC<=5", t[t.hac_parent <= 5]),
             ("HAC 6-9", t[(t.hac_parent >= 6) & (t.hac_parent <= 9)]),
             ("HAC>=10", t[t.hac_parent >= 10])]
    labels = [b[0] for b in bands]
    q = [b[1].quantum_ae.mean() for b in bands]
    l = [b[1].learned_ae.mean() for b in bands]
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, q, w, color=C_Q, label="quantum")
    ax.bar(x + w / 2, l, w, color=C_L, label="learned")
    for xi, (qv, lv, (_, f)) in enumerate(zip(q, l, bands)):
        ax.text(xi - w / 2, qv + 0.05, f"{qv:.2f}", ha="center", fontsize=7.5)
        ax.text(xi + w / 2, lv + 0.05, f"{lv:.2f}", ha="center", fontsize=7.5)
        ax.text(xi, -0.42, f"n={len(f)}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MAE (kcal/mol)")
    ax.set_ylim(0, max(max(q), max(l)) * 1.22)
    ax.set_title("F2  Channel ranking reverses with size", loc="left", fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.text(0.5, 0.90, f"interaction Welch p={inter['interaction_test']['welch_p']:.1e}",
            transform=ax.transAxes, ha="center", fontsize=7.5,
            bbox=dict(fc="white", ec="grey", alpha=0.9))


def panel_error_vs_magnitude(ax, t):
    """F4: signed error against the experimental value, per channel."""
    E = t[common.EXP_COL].to_numpy(float)
    for c, col, lab in (("quantum", C_Q, "quantum"), ("learned", C_L, "learned"),
                        ("reference", C_R, "reference")):
        ax.scatter(E, t[f"{c}_err"], s=9, color=col, alpha=0.42, label=lab, edgecolors="none")
    ax.axhline(0, color="black", lw=1)
    # bin means
    edges = np.arange(10, 171, 15)
    for c, col in (("quantum", C_Q), ("learned", C_L), ("reference", C_R)):
        mids, means = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (E >= lo) & (E < hi)
            if m.sum() >= 6:
                mids.append((lo + hi) / 2)
                means.append(t[f"{c}_err"].to_numpy()[m].mean())
        ax.plot(mids, means, color=col, lw=2, marker="o", ms=4, mec="white")
    ax.set_xlabel("experimental BDE (kcal/mol)")
    ax.set_ylabel("estimate - experiment (kcal/mol)")
    ax.set_title("F4  Signed error vs bond strength\n(lines = per-bin means)",
                 loc="left", fontweight="bold")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_ylim(-36, 28)


def main() -> int:
    t = common.build_analysis_table(verbose=False)
    floor = json.loads((common.RESULTS / "headline_noise_floor.json").read_text())
    inter = json.loads((common.RESULTS / "size_interaction_test.json").read_text())

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
    panel_coverage(axes[0, 0], t)
    panel_size_interaction(axes[0, 1], t, inter)
    panel_floor(axes[1, 0], t, floor)
    panel_error_vs_magnitude(axes[1, 1], t)
    fig.suptitle("Bond dissociation enthalpy: database domain and estimation-channel boundaries",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = common.FIGURES / "summary_findings.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[figures] wrote {out}")

    # standalone F4 at larger size
    fig2, ax2 = plt.subplots(figsize=(7.2, 4.6))
    panel_error_vs_magnitude(ax2, t)
    fig2.tight_layout()
    out2 = common.FIGURES / "error_vs_bond_strength.png"
    fig2.savefig(out2)
    plt.close(fig2)
    print(f"[figures] wrote {out2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())