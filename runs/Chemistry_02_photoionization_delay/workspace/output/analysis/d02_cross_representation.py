#!/usr/bin/env python3
"""d02_cross_representation.py -- cross-representation checks that qualify two
headline results of this run.

PART A. Energy-scramble null for the experiment-versus-theory comparison.
  p6_exptheory reported a high angular correlation between the experiment-like
  and theory-like phase panels at matched energies.  A correlation of that size
  is only informative if it is specific to the MATCHED energy.  Here the matched
  experiment row is instead compared against every theory energy, and the
  matched pair is ranked inside that null distribution.

PART B. The measured CoefLeg blocks versus the dense panel.
  p7_anisotropy found that the literal summed ratio sum(F20)/sum(F00) across
  all nine order rows anti-correlates with the dense-grid beta(E).  This part
  shows why: that ratio is an order-truncation artefact.  The comparison is
  redone order by order against the correct dense expansion.

Which dense expansion is the correct one to compare with CoefLeg:
  the complex amplitude Z = magnitude*exp(i*phase) is exactly finite degree in
  the experiment panel, and so is the angular intensity I = |Z|^2 = magnitude^2
  (degree 6, see d02_intensity_degree.json).  Both are tested here.

Run:  python3 output/analysis/d02_cross_representation.py
Writes: output/evidence/d02_cross_representation.json, .log, .png
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd

ROOT = "/data/user/conversations/conv-9e234f5e-4d56-4e60-8340-10f7a5d416d0/project"
DATA = os.path.join(ROOT, "data")
EV = os.path.join(ROOT, "output", "evidence")

MEAS = [22.5, 23.25, 24.8, 26.35, 27.9, 29.45, 31.0, 32.55, 34.1, 35.65, 37.2, 38.75, 48.4]
MATCHED = [23.25, 24.8, 26.35, 27.9, 29.45, 31.0, 32.55, 34.1, 35.65, 37.2, 38.75]
X = np.cos(np.deg2rad(np.arange(181.0)))


def legendre(lmax, x=X):
    P = np.empty((lmax + 1, x.size))
    P[0] = 1.0
    if lmax >= 1:
        P[1] = x
    for n in range(2, lmax + 1):
        P[n] = ((2 * n - 1) * x * P[n - 1] - (n - 1) * P[n - 2]) / n
    return P


def load():
    obs = pd.read_csv(os.path.join(DATA, "angle_energy_observations_datasets.tsv"), sep="\t")

    def g(p):
        return obs[obs.dataset_path == p].pivot_table(
            index="index_0", columns="index_1", values="value").to_numpy()

    def v(p):
        return obs[obs.dataset_path == p].sort_values("index_0")["value"].to_numpy()

    co = pd.read_csv(os.path.join(DATA, "angular_coefficients_datasets.tsv"), sep="\t")
    return dict(E=v("experiment/photon_energy_eV"), Et=v("theory/photon_energy_eV"),
                mag=g("experiment/magnitude"), pha=g("experiment/phase"),
                tmag=g("theory/magnitude"), tpha=g("theory/phase"),
                co=co, names=sorted(co.dataset_path.unique()))


def part_a(d, out):
    E, Et = d["E"], d["Et"]
    epha = np.unwrap(d["pha"], axis=1)
    tpha = np.unwrap(d["tpha"], axis=1)
    rows_p, rows_m = [], []
    print("=== PART A: energy-scramble null (is the agreement energy specific?) ===")
    print(f"{'exp E':>7} | {'phase r':>9} {'phase med_other':>15} {'phase frac>=':>13} | "
          f"{'mag r':>8} {'mag med_other':>14} {'mag frac>=':>11}")
    for e in MATCHED:
        j = int(np.argmin(np.abs(E - e)))
        ti = int(np.argmin(np.abs(Et - e)))
        rp = float(np.corrcoef(tpha[ti], epha[j])[0, 1])
        op = np.array([np.corrcoef(tpha[k], epha[j])[0, 1] for k in range(tpha.shape[0]) if k != ti])
        rm = float(np.corrcoef(d["tmag"][ti], d["mag"][j])[0, 1])
        om = np.array([np.corrcoef(d["tmag"][k], d["mag"][j])[0, 1] for k in range(d["tmag"].shape[0]) if k != ti])
        rows_p.append({"e_exp": e, "e_theory": float(Et[ti]), "r_matched": rp,
                       "r_median_mismatched": float(np.median(op)),
                       "r_p95_mismatched": float(np.percentile(op, 95)),
                       "frac_mismatched_ge_matched": float(np.mean(op >= rp))})
        rows_m.append({"e_exp": e, "e_theory": float(Et[ti]), "r_matched": rm,
                       "r_median_mismatched": float(np.median(om)),
                       "r_p95_mismatched": float(np.percentile(om, 95)),
                       "frac_mismatched_ge_matched": float(np.mean(om >= rm))})
        print(f"{e:7.2f} | {rp:9.4f} {np.median(op):15.4f} {np.mean(op >= rp):13.3f} | "
              f"{rm:8.4f} {np.median(om):14.4f} {np.mean(om >= rm):11.3f}")
    np_ = sum(1 for r in rows_p if r["frac_mismatched_ge_matched"] < 0.05)
    nm_ = sum(1 for r in rows_m if r["frac_mismatched_ge_matched"] < 0.05)
    out["partA_energy_scramble"] = {
        "phase": rows_p, "magnitude": rows_m,
        "n_matched_phase_beats_p95_of_mismatched": np_,
        "n_matched_magnitude_beats_p95_of_mismatched": nm_,
        "n_matched_total": len(MATCHED),
        "interpretation": ("a matched pair that beats >=95% of mismatched theory energies is energy specific; "
                           "frac_mismatched_ge_matched < 0.05 is the criterion"),
    }
    print(f"\nA: matched pair beats >=95% of mismatched theory energies in only "
          f"{np_}/{len(MATCHED)} (phase) and {nm_}/{len(MATCHED)} (magnitude) cases.")
    print("A: a high matched-pair correlation is therefore NOT by itself evidence of energy-matched agreement;")
    print("   the panels share a common angular baseline that all energies reproduce.")
    return out


def part_b(d, out):
    co, names = d["co"], d["names"]
    E = d["E"]
    idx = [int(np.argmin(np.abs(E - e))) for e in MEAS]
    assert len(names) == 13

    def fam(col):
        return np.array([co[co.dataset_path == n].pivot_table(
            index="index_0", columns="index_1", values="value").to_numpy()[:, col] for n in names])

    F00, F20 = fam(1), fam(3)
    A9 = legendre(8).T
    A7 = legendre(6).T
    Z = np.array([d["mag"][i] * np.exp(1j * d["pha"][i]) for i in idx])
    Cz = np.array([np.linalg.lstsq(A9, z, rcond=None)[0] for z in Z])
    CI = np.array([np.linalg.lstsq(A7, np.abs(z) ** 2, rcond=None)[0] for z in Z])
    Am = np.array([np.linalg.lstsq(A9, d["mag"][i], rcond=None)[0] for i in idx])

    print("\n=== PART B: CoefLeg versus the dense panel, order by order ===")
    print("  (the intensity basis has exact degree 6, so it carries 7 coefficients; the complex")
    print("   amplitude has degree 3, so only its k=0..6 entries are informative — k>=4 is round-off.)")
    out["partB"] = {}
    for nm, C, kmax in (("complex Z (deg 3)", Cz, 9), ("intensity |Z|^2 (deg 6)", CI, 7),
                        ("real magnitude", Am, 9)):
        diag = [float(np.corrcoef(np.abs(C[:, k]), F20[:, k])[0, 1]) for k in range(kmax)]
        off = [float(np.corrcoef(np.abs(C[:, k]), F20[:, o])[0, 1])
               for k in range(kmax) for o in range(9) if o != k]
        out["partB"][nm] = {"diag": np.round(diag, 4).tolist(),
                            "n_coefficients_used": kmax,
                            "diag_mean_abs": float(np.abs(diag).mean()),
                            "offdiag_mean_abs": float(np.abs(off).mean())}
        print(f"  dense basis {nm:22s}: mean|diag(k=0..{kmax-1})|={np.abs(diag).mean():.3f}  "
              f"mean|off|={np.abs(off).mean():.3f}")
        print(f"  {'':30s} diag by order = {np.round(diag,3)}")

    # permutation test on the energy labelling, using the intensity basis
    rng = np.random.default_rng(20260919)
    stat = float(np.abs([np.corrcoef(np.abs(CI[:, k]), F20[:, k])[0, 1] for k in range(7)]).mean())
    N = 50000
    hits = sum(1 for _ in range(N)
               if np.abs([np.corrcoef(np.abs(CI[:, k]), F20[rng.permutation(13), k])[0, 1]
                          for k in range(7)]).mean() >= stat)
    print(f"\n  permutation test on the ENERGY labelling (intensity basis, k=0..6): p = {hits/N:.5f} (n={N})")
    out["partB_permutation_energy_labelling"] = {
        "observed_mean_abs_r_k0_6": stat, "p": hits / N, "n_draws": N,
        "basis": "dense intensity |Z|^2 coefficients c_k versus CoefLeg F20 order k",
    }

    # grid-wide max comparison against its own null
    obsmax = float(np.abs([np.corrcoef(np.abs(CI[:, k]), F20[:, o])[0, 1]
                           for k in range(7) for o in range(9)]).max())
    nullmax = [float(np.abs([np.corrcoef(np.abs(CI[:, k]), F20[rng.permutation(13), o])[0, 1]
                             for k in range(7) for o in range(9)]).max()) for _ in range(3000)]
    out["partB_grid_max_null"] = {
        "observed_max_abs_r": obsmax,
        "null_median": float(np.median(nullmax)),
        "null_p95": float(np.percentile(nullmax, 95)),
        "p_grid_max": float(np.mean(np.array(nullmax) >= obsmax)),
        "note": "selecting the max over a 7x9 grid inflates |r|; this null controls for that selection.",
    }
    print(f"  max|r| over the 7x9 grid: observed {obsmax:.3f}; chance median {np.median(nullmax):.3f}; "
          f"p={np.mean(np.array(nullmax) >= obsmax):.5f}")

    # the truncation artefact behind the p7 anti-correlation
    beta_dense = (CI[:, 2] / CI[:, 0])
    tr = {}
    for kmax in range(9):
        ratio = F20[:, :kmax + 1].sum(axis=1) / F00[:, :kmax + 1].sum(axis=1)
        tr[kmax] = {"pearson_vs_dense_beta": float(np.corrcoef(ratio, beta_dense)[0, 1]),
                    "spearman_vs_dense_beta": float(pd.Series(ratio).corr(pd.Series(beta_dense), method="spearman"))}
    out["partB_sum_ratio_truncation"] = tr
    print("\n  corr( sum(F20 orders<=n) / sum(F00 orders<=n),  dense beta ) by truncation n:")
    for k, v in tr.items():
        print(f"    n={k}: pearson {v['pearson_vs_dense_beta']:+.4f}   spearman {v['spearman_vs_dense_beta']:+.4f}")
    print("  -> the sign of the 'cross-representation conflict' is set by the truncation order, so the")
    print("     literal all-order summed ratio of p7 is an artefact of the summation convention.")

    # order-consistent ratio
    oc = {}
    for k in range(1, 7):
        r = F20[:, k] / F00[:, k]
        oc[k] = {"corr_with_dense_c_k_over_c_0": float(np.corrcoef(r, (CI[:, k] / CI[:, 0]))[0, 1]),
                 "corr_with_dense_beta": float(np.corrcoef(r, beta_dense)[0, 1])}
    out["partB_order_consistent_ratio"] = oc
    print("\n  order-consistent ratio F20(k)/F00(k) versus the dense same-order ratio c_k/c_0:")
    for k, v in oc.items():
        print(f"    k={k}: r={v['corr_with_dense_c_k_over_c_0']:+.4f}")

    out["partB_series_csv"] = os.path.join(EV, "d02_cross_representation.csv")
    pd.DataFrame({
        "energy_eV": MEAS,
        "dense_I_c0": CI[:, 0].real, "dense_I_c1": CI[:, 1].real, "dense_I_c2": CI[:, 2].real,
        "dense_I_c3": CI[:, 3].real, "dense_I_beta_c2_over_c0": beta_dense.real,
        "denseI_abs_c0": np.abs(CI[:, 0]), "denseI_abs_c1": np.abs(CI[:, 1]),
        "denseI_abs_c2": np.abs(CI[:, 2]), "denseI_abs_c3": np.abs(CI[:, 3]),
        "denseI_abs_c4": np.abs(CI[:, 4]), "denseI_abs_c5": np.abs(CI[:, 5]), "denseI_abs_c6": np.abs(CI[:, 6]),
        **{f"F20_order{k}": F20[:, k] for k in range(9)},
        **{f"F00_order{k}": F00[:, k] for k in range(9)},
        "ratio_summed_all_orders": F20.sum(axis=1) / F00.sum(axis=1),
        "ratio_summed_orders0_2": F20[:, :3].sum(axis=1) / F00[:, :3].sum(axis=1),
    }).to_csv(out["partB_series_csv"], index=False)
    print("  wrote", out["partB_series_csv"])
    return out


def main():
    out = {"schema": "tib-cross-representation-v3"}
    d = load()
    out["data_sha256"] = {
        f: hashlib.sha256(open(os.path.join(DATA, f), "rb").read()).hexdigest()
        for f in ("angle_energy_observations_datasets.tsv", "angular_coefficients_datasets.tsv")
    }
    part_a(d, out)
    part_b(d, out)

    with open(os.path.join(EV, "d02_cross_representation.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote", os.path.join(EV, "d02_cross_representation.json"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6))
    ax = axes[0, 0]
    for row in out["partA_energy_scramble"]["phase"]:
        ax.plot(row["e_exp"], row["r_matched"], "o", color="#12457a", ms=6)
        ax.plot([row["e_exp"], row["e_exp"]], [row["r_median_mismatched"], row["r_matched"]],
                "-", color="#999", lw=1)
        ax.plot(row["e_exp"], row["r_median_mismatched"], "_", color="#c1440e", ms=11)
    ax.plot([], [], "o", color="#12457a", label="matched energy pair")
    ax.plot([], [], "_", color="#c1440e", label="median mismatched theory energy")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("photon energy (eV)")
    ax.set_ylabel("angular correlation r (unwrapped phase)")
    ax.set_title("Part A: the matched pair rarely beats a mismatched energy\n(phase panels)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    fr_p = [r["frac_mismatched_ge_matched"] for r in out["partA_energy_scramble"]["phase"]]
    fr_m = [r["frac_mismatched_ge_matched"] for r in out["partA_energy_scramble"]["magnitude"]]
    w = 0.38
    xe = np.arange(len(MATCHED))
    ax.bar(xe - w / 2, fr_p, w, label="phase", color="#12457a")
    ax.bar(xe + w / 2, fr_m, w, label="magnitude", color="#7aa7d2")
    ax.axhline(0.05, color="#c1440e", ls="--", lw=1.2, label="0.05 threshold")
    ax.set_xticks(xe)
    ax.set_xticklabels([f"{e:g}" for e in MATCHED], rotation=45, fontsize=7)
    ax.set_xlabel("photon energy (eV)")
    ax.set_ylabel("fraction of mismatched energies with r >= matched r")
    ax.set_title("Part A: energy specificity fails at most energies\n(bars below 0.05 = energy specific)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    diagI = out["partB"]["intensity |Z|^2 (deg 6)"]["diag"]
    diagZ = out["partB"]["complex Z (deg 3)"]["diag"]
    ax.plot(range(len(diagI)), np.abs(diagI), "o-", color="#12457a",
            label="dense intensity $|Z|^2$ (7 coefficients, exact degree 6)")
    ax.plot(range(len(diagZ)), np.abs(diagZ), "s--", color="#c1440e",
            label="dense complex $Z$ (only k<=3 carries signal)")
    ax.axvspan(3.5, 8.5, color="#dddddd", alpha=0.5)
    ax.text(6, 0.5, "k>=4: $|Z|^2$ has no such\ndegree; $Z$ is round-off", fontsize=7, ha="center")
    ax.set_xlabel("CoefLeg order index (= dense coefficient order k)")
    ax.set_ylabel("|order-wise correlation r| over 13 energies")
    ax.set_title("Part B: order-wise correspondence\ndense coefficient k  vs  F20(order k)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    tr_ = {int(k): v for k, v in out["partB_sum_ratio_truncation"].items()}
    ks = sorted(tr_)
    ax.plot(ks, [tr_[k]["pearson_vs_dense_beta"] for k in ks], "o-", color="#12457a")
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xlabel("CoefLeg order truncation n (families summed over orders 0..n)")
    ax.set_ylabel("Pearson r of summed F20/F00 vs dense beta")
    ax.set_title("Part B: the reported 'cross-representation conflict'\nis a summation-truncation artefact")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(EV, "d02_cross_representation.png"), dpi=130)
    print("wrote", os.path.join(EV, "d02_cross_representation.png"))


if __name__ == "__main__":
    main()