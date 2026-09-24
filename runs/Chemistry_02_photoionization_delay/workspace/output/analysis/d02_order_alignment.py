#!/usr/bin/env python3
"""d02_order_alignment.py -- Cross-representation order alignment between the
dense experiment-like panel and the measured CoefLeg angular-coefficient blocks.

Motivation: p7_anisotropy reported that the summary ratio F20_sum/F00_sum
anti-correlates with the dense-grid beta(E) across the 13 measured energies.
This script tests whether that summary ratio is the right object to compare, by
testing the ORDER-WISE correspondence instead: the dense complex amplitude
Z = magnitude*exp(i*phase) has an exact finite angular degree (3, established by
d01/p1), so its complex Legendre coefficients c_k(E) can be compared, order by
order, against the CoefLeg F20 family (order index k = 0..8).

Tests
  T1  order-wise correlation matrix between |c_k(E)| and F20(order k', E)
  T2  permutation test on the energy labelling of the F20 rows
  T3  a diagonal shift scan (is the order alignment the correct one?)
  T4  proportionality test F20(E,k) = s_k * c_k(E) with one complex scale per order
  T5  control pairings (random and shift-by-one) for T4
  T6  the summary ratio F20_sum/F00_sum as a function of the truncation order

Run:  python3 output/analysis/d02_order_alignment.py
Writes: output/evidence/d02_order_alignment.json, .log, .png
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = "/data/user/conversations/conv-9e234f5e-4d56-4e60-8340-10f7a5d416d0/project"
DATA = os.path.join(ROOT, "data")
EV = os.path.join(ROOT, "output", "evidence")

MEAS = [22.5, 23.25, 24.8, 26.35, 27.9, 29.45, 31.0, 32.55, 34.1, 35.65, 37.2, 38.75, 48.4]
ANG = np.arange(181.0)
X = np.cos(np.deg2rad(ANG))


def legendre_table(x, lmax):
    P = np.empty((lmax + 1, x.size))
    P[0] = 1.0
    if lmax >= 1:
        P[1] = x
    for n in range(2, lmax + 1):
        P[n] = ((2 * n - 1) * x * P[n - 1] - (n - 1) * P[n - 2]) / n
    return P


def main():
    out = {"schema": "tib-order-alignment-v1"}
    obs = pd.read_csv(os.path.join(DATA, "angle_energy_observations_datasets.tsv"), sep="\t")
    co = pd.read_csv(os.path.join(DATA, "angular_coefficients_datasets.tsv"), sep="\t")
    E = obs[obs.dataset_path == "experiment/photon_energy_eV"].sort_values("index_0")["value"].to_numpy()
    mag = obs[obs.dataset_path == "experiment/magnitude"].pivot_table(
        index="index_0", columns="index_1", values="value").to_numpy()
    pha = obs[obs.dataset_path == "experiment/phase"].pivot_table(
        index="index_0", columns="index_1", values="value").to_numpy()

    names = sorted(co.dataset_path.unique())
    assert len(names) == 13
    idx = [int(np.argmin(np.abs(E - e))) for e in MEAS]
    print("measured energies aligned to dense rows:", [round(E[i], 2) for i in idx])

    def fam(name, col):
        return co[co.dataset_path == name].pivot_table(
            index="index_0", columns="index_1", values="value").to_numpy()[:, col]

    F20 = np.array([fam(n, 3) for n in names])          # 13 x 9
    F00 = np.array([fam(n, 1) for n in names])
    F21 = np.array([fam(n, 5) for n in names])
    F22 = np.array([fam(n, 7) for n in names])
    F11 = np.array([fam(n, 9) for n in names])

    P9 = legendre_table(X, 8)
    C = np.array([np.linalg.lstsq(P9.T, (mag[i] * np.exp(1j * pha[i])), rcond=None)[0] for i in idx])
    print("dense complex Legendre coefficients c_k: shape", C.shape)

    # ---- T1 order-wise correlation matrix ------------------------------------
    M = np.array([[np.corrcoef(np.abs(C[:, k]), F20[:, o])[0, 1] for o in range(9)] for k in range(9)])
    diag = np.array([M[k, k] for k in range(9)])
    off = M[np.triu_indices(9, 1)]
    out["T1_order_correlation"] = {
        "matrix_rows_dense_ck_cols_F20_order": np.round(M, 4).tolist(),
        "diagonal_k0_3": np.round(diag[:4], 4).tolist(),
        "diagonal_all_9_mean_abs": float(np.abs(diag).mean()),
        "offdiagonal_mean_abs": float(np.abs(off).mean()),
        "offdiagonal_max_abs": float(np.abs(off).max()),
        "offdiagonal_max_location_k_o": [int(v) for v in np.unravel_index(np.abs(off).argmax(), off.shape)],
    }
    print("\nT1: |c_k| vs F20(order k) diagonal k=0..3 =", np.round(diag[:4], 4))
    print("T1: mean |r| diagonal (all 9) = %.4f ; off-diagonal mean |r| = %.4f ; off-diagonal max |r| = %.4f"
          % (np.abs(diag).mean(), np.abs(off).mean(), np.abs(off).max()))

    # ---- T2 permutation test on the energy labelling --------------------------
    rng = np.random.default_rng(20260919)
    obs_stat = float(np.abs(diag[:4]).mean())
    N = 20000
    hits = 0
    for _ in range(N):
        p = rng.permutation(13)
        d = np.array([np.corrcoef(np.abs(C[:, k]), F20[p, k])[0, 1] for k in range(4)])
        if np.abs(d).mean() >= obs_stat:
            hits += 1
    out["T2_permutation_energy_labelling"] = {
        "n_draws": N,
        "observed_mean_abs_r_diag_k0_3": obs_stat,
        "p_value": hits / N,
        "hypothesis_tested": "the alignment of dense coefficient order k with CoefLeg order k is an artifact of the energy ordering of the 13 blocks",
        "limit_note": "with 13 blocks only 13! energy relabellings exist; the Monte Carlo floor of this p-value is 1/N and the test is coarse.",
    }
    print(f"T2: permutation p = {hits/N:.5f} (observed mean|r| over k=0..3 = {obs_stat:.4f})")

    # ---- T3 diagonal shift scan ---------------------------------------------
    shift = {}
    for s in range(-3, 4):
        vals = [float(np.corrcoef(np.abs(C[:, k]), F20[:, k + s])[0, 1]) for k in range(4) if 0 <= k + s < 9]
        shift[s] = {"corrs": np.round(vals, 4).tolist(), "mean_abs": float(np.abs(vals).mean())}
    out["T3_diagonal_shift_scan"] = shift
    best_s = max(shift, key=lambda s: shift[s]["mean_abs"])
    out["T3_best_shift"] = best_s
    print("T3: best order shift =", best_s, "mean|r| =", round(shift[best_s]["mean_abs"], 4),
          "vs shift 0 mean|r| =", round(shift[0]["mean_abs"], 4))

    # ---- T4 proportionality --------------------------------------------------
    prop = {}
    for k in range(4):
        ck = C[:, k]
        f = F20[:, k]
        s = np.vdot(ck, f) / np.vdot(ck, ck)
        pred = s * ck
        rel = np.abs(f - pred) / np.abs(f)
        r2 = 1 - np.sum(np.abs(f - pred) ** 2) / np.sum(np.abs(f - f.mean()) ** 2)
        prop[k] = {"scale_real": float(s.real), "scale_imag": float(s.imag), "scale_abs": float(abs(s)),
                   "max_rel_resid": float(rel.max()), "var_explained": float(r2)}
    out["T4_proportionality_one_scale_per_order"] = prop
    print("\nT4: one complex scale per order, F20(E,k) = s_k * c_k(E):")
    for k, v in prop.items():
        print("   k=%d |s_k|=%.4f (imag %.2e)  var_explained=%.5f  max_rel_resid=%.3f"
              % (k, v["scale_abs"], v["scale_imag"], v["var_explained"], v["max_rel_resid"]))

    # ---- T5 controls ---------------------------------------------------------
    ctr = {"random_pairings": [], "shift_by_one": []}
    for _ in range(200):
        ck = C[:, rng.integers(0, 9)]
        f = F20[:, rng.integers(0, 9)]
        s = np.vdot(ck, f) / np.vdot(ck, ck)
        ctr["random_pairings"].append(
            float(1 - np.sum(np.abs(f - s * ck) ** 2) / np.sum(np.abs(f - f.mean()) ** 2)))
    for k in range(3):
        ck = C[:, k]
        f = F20[:, k + 1]
        s = np.vdot(ck, f) / np.vdot(ck, ck)
        ctr["shift_by_one"].append(
            {"k": k, "to_order": k + 1,
             "var_explained": float(1 - np.sum(np.abs(f - s * ck) ** 2) / np.sum(np.abs(f - f.mean()) ** 2))})
    rp = np.array(ctr["random_pairings"])
    ctr["random_pairing_summary"] = {
        "n": int(rp.size), "median": float(np.median(rp)),
        "p95": float(np.percentile(rp, 95)), "max": float(rp.max()),
    }
    out["T5_controls"] = ctr
    print("\nT5 controls: random pairing var-explained median %.4f p95 %.4f max %.4f"
          % (np.median(rp), np.percentile(rp, 95), rp.max()))
    print("T5 controls: shift-by-one var-explained =",
          [round(v["var_explained"], 4) for v in ctr["shift_by_one"]])

    # ---- T6 summary ratio vs truncation order --------------------------------
    t6 = {}
    for kmax in range(0, 9):
        ssum = F20[:, :kmax + 1].sum(axis=1)
        fsum = F00[:, :kmax + 1].sum(axis=1)
        ratio = ssum / fsum
        beta = (C[:, 2] / C[:, 0]).real
        t6[kmax] = {"pearson_vs_dense_beta": float(np.corrcoef(ratio, beta)[0, 1]),
                    "spearman_vs_dense_beta": float(pd.Series(ratio).corr(pd.Series(beta), method="spearman"))}
    out["T6_summary_ratio_by_truncation"] = t6
    print("\nT6: corr(F20_sum/F00_sum truncated at order n, dense beta) by n:")
    for k, v in t6.items():
        print("   n=%d  pearson=%+.4f  spearman=%+.4f" % (k, v["pearson_vs_dense_beta"], v["spearman_vs_dense_beta"]))

    # ---- T7 the other families, same order-wise test --------------------------
    t7 = {}
    for fname, arr in (("F00", F00), ("F20", F20), ("F21", F21), ("F22", F22), ("F11", F11)):
        d = [float(np.corrcoef(np.abs(C[:, k]), arr[:, k])[0, 1]) for k in range(4)]
        t7[fname] = {
            "diag_k0_3": np.round(d, 4).tolist(),
            "mean_abs": float(np.abs(d).mean()),
            "note": "F00 order 0 is pinned to 1.0 for all 13 energies, so its k=0 correlation is driven by |c_0| alone.",
        }
    out["T7_all_families_order_wise"] = t7
    print("\nT7: order-wise diagonal correlations by family:")
    for k, v in t7.items():
        print("   %-4s %s  mean|r|=%.3f" % (k, v["diag_k0_3"], v["mean_abs"]))

    out["data_sha256"] = {}
    import hashlib
    for f in ("angle_energy_observations_datasets.tsv", "angular_coefficients_datasets.tsv"):
        h = hashlib.sha256(open(os.path.join(DATA, f), "rb").read()).hexdigest()
        out["data_sha256"][f] = h
        print("sha256", f, h)
    out["artifacts"] = {"csv": os.path.join(EV, "d02_order_alignment.csv")}

    # csv of the compared series
    pd.DataFrame({
        "energy_eV": MEAS,
        "dense_abs_c0": np.abs(C[:, 0]), "dense_abs_c1": np.abs(C[:, 1]),
        "dense_abs_c2": np.abs(C[:, 2]), "dense_abs_c3": np.abs(C[:, 3]),
        "dense_beta_c1_over_c0": (C[:, 1] / C[:, 0]).real,
        "dense_beta_c2_over_c0": (C[:, 2] / C[:, 0]).real,
        "F20_order0": F20[:, 0], "F20_order1": F20[:, 1], "F20_order2": F20[:, 2], "F20_order3": F20[:, 3],
        "F00_order0": F00[:, 0], "F00_sum_all": F00.sum(axis=1), "F20_sum_all": F20.sum(axis=1),
        "ratio_F20sum_over_F00sum_all": F20.sum(axis=1) / F00.sum(axis=1),
    }).to_csv(out["artifacts"]["csv"], index=False)
    print("wrote", out["artifacts"]["csv"])

    with open(os.path.join(EV, "d02_order_alignment.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", os.path.join(EV, "d02_order_alignment.json"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4))
    ax = axes[0, 0]
    im = ax.imshow(np.abs(M), cmap="magma", vmin=0, vmax=1)
    ax.set_xlabel("CoefLeg F20 order index")
    ax.set_ylabel("dense $|c_k|$ order k")
    ax.set_title("T1: |correlation| across the 13 measured energies\n(diagonal = order-wise correspondence)")
    for k in range(9):
        ax.plot(k, k, "o", ms=11, mfc="none", mec="#6fe3ff", mew=1.6)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[0, 1]
    n = np.arange(9)
    ax.plot(n, np.abs(diag), "o-", label="diagonal (order-wise)", color="#12457a")
    ax.plot(n, np.abs(off).reshape(9, 8).mean(axis=1) if off.size == 72 else np.zeros(9),
            "s--", label="off-diagonal mean", color="#c1440e")
    ax.set_xlabel("order index")
    ax.set_ylabel("mean |r|")
    ax.set_title("T1: order-wise vs off-diagonal correlation strength")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    for k in range(4):
        ax.plot(MEAS, np.abs(C[:, k]) / np.abs(C[:, k]).max(), "o-", ms=3, label=f"|c_{k}|/max")
        ax.plot(MEAS, F20[:, k] / np.abs(F20[:, k]).max(), "s--", ms=3, alpha=0.65)
    ax.set_yscale("symlog", linthresh=0.02)
    ax.set_xlabel("photon energy (eV)")
    ax.set_ylabel("normalized magnitude (symlog)")
    ax.set_title("T4: dense $|c_k|$ (circles) vs F20(order k) (squares)\neach series normalized to its own maximum")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    ax = axes[1, 1]
    ks = sorted(t6)
    ax.plot(ks, [t6[k]["pearson_vs_dense_beta"] for k in ks], "o-", color="#12457a")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("CoefLeg order truncation n (families summed over orders 0..n)")
    ax.set_ylabel("Pearson r vs dense beta(E)")
    ax.set_title("T6: the summary ratio F20/F00 is only a fair\nanisotropy proxy if the sum is truncated at low order")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(EV, "d02_order_alignment.png"), dpi=130)
    print("wrote", os.path.join(EV, "d02_order_alignment.png"))


if __name__ == "__main__":
    main()