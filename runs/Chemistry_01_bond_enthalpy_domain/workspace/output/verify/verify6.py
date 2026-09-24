"""Independent adversarial verification of CLAIM 6 (oracle-MAE-floor ratios).

Recomputed from the raw TSV with no import of the original workstream code.
"""
import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

P = Path("/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project")
TSV = P / "data/normalized_views/experimental_benchmark.tsv"
OUT = P / "output/verify/verify6_raw.json"

bench = pd.read_csv(TSV, sep="\t")            # RAW read, own code
print("shape", bench.shape, "unique obs id", bench.observation_id.nunique())

E = bench.experimental_bde_kcal_mol.to_numpy(float)
Q = bench.quantum_estimate_bde_kcal_mol.to_numpy(float)
L = bench.learned_estimate_bde_kcal_mol.to_numpy(float)
R = bench.reference_estimate_bde_kcal_mol.to_numpy(float)
n = len(E)

# ---------- 1. headline channel MAEs ---------------------------------------
res = {}
for name, x in (("quantum", Q), ("learned", L), ("reference", R)):
    err = x - E
    res[name] = {"mae": float(np.mean(np.abs(err))),
                 "rmse": float(np.sqrt(np.mean(err**2))),
                 "bias": float(err.mean())}
print("MAEs:", {k: round(v["mae"], 6) for k, v in res.items()})

# ---------- 2. analysis-unit forensics -------------------------------------
keys = list(zip(bench.parent_structure_smiles, bench.fragment_a_smiles, bench.fragment_b_smiles))
res["unit_forensics"] = {
    "n_rows": int(n),
    "n_unique_observation_id": int(bench.observation_id.nunique()),
    "n_unique_molecule": int(bench.parent_structure_smiles.nunique()),
    "n_unique_fragment_pair": int(len(set(map(tuple, zip(bench.fragment_a_smiles, bench.fragment_b_smiles))))),
    "n_unique_mol_fragpair_triple": int(len(set(keys))),
    "n_unique_quantum_value": int(pd.Series(Q).nunique()),
    "n_unique_experimental_value": int(pd.Series(E).nunique()),
    "n_unique_learned_value": int(pd.Series(L).nunique()),
    "n_unique_reference_value": int(pd.Series(R).nunique()),
    # are duplicate triples present (repeat measurements of the same bond)?
    "dup_triples": int(n - len(set(keys))),
    "dup_molecules": int(n - bench.parent_structure_smiles.nunique()),
}
print("unit forensics:", json.dumps(res["unit_forensics"], indent=1))

# do repeated quantum values correspond to repeated experimentals/l learned?
qv_counts = pd.Series(Q).value_counts()
res["quantum_value_repeats"] = {
    "n_rows_with_quantum_value_seen_more_than_once":
        int((pd.Series(Q).map(qv_counts) > 1).sum()),
    "max_multiple": int(qv_counts.max()),
}

# ---------- 3. METHOD A (matched near-duplicate quantum pairs) -------------
def pair_masks(tol):
    dQf = Q[:, None] - Q[None, :]
    iu = np.triu_indices(n, k=1)
    dQ = dQf[iu]
    m = np.abs(dQ) <= tol
    dE = (E[:, None] - E[None, :])[iu][m]
    dL = (L[:, None] - L[None, :])[iu][m]
    dR = (R[:, None] - R[None, :])[iu][m]
    idx = np.array(iu).T[m]          # row index pairs, for chemical forensics
    dQk = dQ[m]
    return dE, dL, dR, dQk, idx

MAE_FROM_SIGMA = float(np.sqrt(2.0 / np.pi))
methodA = []
pair_cache = {}
for tol in (0.0, 0.05, 0.1, 0.2, 0.5, 1.0):
    dE, dL, dR, dQk, idx = pair_masks(tol)
    if len(dE) < 20:
        continue
    pair_cache[tol] = (dE, dL, dR, dQk, idx)
    sig = float(np.std(dE, ddof=1) / np.sqrt(2))
    methodA.append({
        "tol": tol, "n_pairs": int(len(dE)),
        "sd_delta_E": float(np.std(dE, ddof=1)),
        "sigma_ref": sig,
        "oracle_mae": sig * MAE_FROM_SIGMA,
        "sigma_delta_Q_selfcheck_sqrt2": float(np.std(dQk, ddof=1) / np.sqrt(2)),
        "sigma_delta_L_selfcheck_sqrt2": float(np.std(dL, ddof=1) / np.sqrt(2)),
        "sigma_delta_R_selfcheck_sqrt2": float(np.std(dR, ddof=1) / np.sqrt(2)),
        "frac_pairs_exact_zero_dQ": float(np.mean(np.abs(dQk) < 1e-9)),
    })
res["method_A"] = methodA
for r in methodA:
    print("A tol=%-4s n=%5d sigma_ref=%.4f oracle_MAE=%.4f  zero_dQ_frac=%.3f"
          % (r["tol"], r["n_pairs"], r["sigma_ref"], r["oracle_mae"], r["frac_pairs_exact_zero_dQ"]))

# ---------- 4. chemical forensics on the tol=0.1 pairs ---------------------
tol = 0.1
dE, dL, dR, dQk, idx = pair_cache[tol]
mol = bench.parent_structure_smiles.to_numpy()
fa = bench.fragment_a_smiles.to_numpy()
fb = bench.fragment_b_smiles.to_numpy()
i, j = idx[:, 0], idx[:, 1]
same_mol = mol[i] == mol[j]
same_pair = (fa[i] == fa[j]) & (fb[i] == fb[j])
same_set = (np.sort(np.stack([fa[i], fb[i]]), axis=0) ==
            np.sort(np.stack([fa[j], fb[j]]), axis=0)).all(axis=0)
group = np.where(same_mol, np.where(same_set, "same_mol_same_fragset", "same_mol_diff_fragset"),
                 "diff_mol")
res["method_A_tol0.1_chemical_breakdown"] = {}
for g in ("same_mol_same_fragset", "same_mol_diff_fragset", "diff_mol"):
    m = group == g
    if m.sum() < 3:
        res["method_A_tol0.1_chemical_breakdown"][g] = {"n_pairs": int(m.sum())}
        continue
    res["method_A_tol0.1_chemical_breakdown"][g] = {
        "n_pairs": int(m.sum()),
        "sd_delta_E": float(np.std(dE[m], ddof=1)),
        "sigma_ref": float(np.std(dE[m], ddof=1) / np.sqrt(2)),
        "oracle_mae": float(np.std(dE[m], ddof=1) / np.sqrt(2) * MAE_FROM_SIGMA),
        "median_sigma_delta_E": float(np.median(np.abs(dE[m] - np.median(dE[m]))) * 1.4826),
        "sd_delta_Q": float(np.std(dQk[m], ddof=1)),
    }
print("tol=0.1 chemical breakdown:", json.dumps(res["method_A_tol0.1_chemical_breakdown"], indent=1))

# is same_mol/ same_molecule a bond duplicate or chemical diversity? show examples
ex = pd.DataFrame({"i": i[same_mol][:5], "j": j[same_mol][:5], "dE": dE[same_mol][:5]})
print("examples of same-molecule pairs:\n", ex)

# filtered A: only pairs from the SAME molecule and SAME fragment set (true replicates)
m_rep = same_mol & same_set
if m_rep.sum() >= 3:
    res["method_A_tol0.1_replicates_only"] = {
        "n_pairs": int(m_rep.sum()),
        "sigma_ref": float(np.std(dE[m_rep], ddof=1) / np.sqrt(2)),
        "oracle_mae": float(np.std(dE[m_rep], ddof=1) / np.sqrt(2) * MAE_FROM_SIGMA),
    }
    print("replicate-only pairs:", res["method_A_tol0.1_replicates_only"])

# ---------- 5. cluster structure: group rows by identical quantum value ----
clusters = {}
for k, g in pd.Series(range(n)).groupby(pd.Series(Q)):
    if len(g) > 1:
        clusters[k] = list(g)
within = []
for k, gidx in clusters.items():
    vals = E[gidx]
    if len(vals) > 1:
        within.append({"quantum": float(k), "n": len(gidx), "sd_E": float(np.std(vals, ddof=1)),
                       "range_E": float(vals.max() - vals.min())})
within_sd = np.array([w["sd_E"] for w in within])
res["clusters_identical_quantum"] = {
    "n_clusters_gt1": len(clusters),
    "n_rows_in_clusters": int(sum(len(v) for v in clusters.values())),
    "pooled_within_sd_E": float(np.sqrt(np.average(within_sd**2, weights=[w["n"] - 1 for w in within]))) if len(within) else None,
    "median_within_sd_E": float(np.median(within_sd)) if len(within) else None,
    "examples": sorted(within, key=lambda w: -w["n"])[:8],
}
print("identical-quantum clusters:", res["clusters_identical_quantum"]["n_clusters_gt1"],
      "rows:", res["clusters_identical_quantum"]["n_rows_in_clusters"],
      "pooled within sd:", res["clusters_identical_quantum"]["pooled_within_sd_E"])

# ---------- 6. METHOD B (moment decomposition) ----------------------------
v_eq = float(np.var(E - Q, ddof=1)); v_el = float(np.var(E - L, ddof=1)); v_ql = float(np.var(Q - L, ddof=1))
sigB_sq = (v_eq + v_el - v_ql) / 2.0
sigB = float(np.sqrt(sigB_sq))
res["method_B"] = {"var_EQ": v_eq, "var_EL": v_el, "var_QL": v_ql,
                   "sigma_ref_sq": sigB_sq, "sigma_ref": sigB,
                   "oracle_mae": sigB * MAE_FROM_SIGMA,
                   "implied_sigma_Q": float(np.sqrt(max(v_eq - sigB_sq, 0))),
                   "implied_sigma_L": float(np.sqrt(max(v_el - sigB_sq, 0)))}
print("method B:", res["method_B"])

# method B using only pairwise covariances (algebraically identical check)
cov_eq = float(np.cov(E, Q, ddof=1)[0, 1]); cov_el = float(np.cov(E, L, ddof=1)[0, 1])
cov_ql = float(np.cov(Q, L, ddof=1)[0, 1])
v_e = float(np.var(E, ddof=1)); v_q = float(np.var(Q, ddof=1)); v_l = float(np.var(L, ddof=1))
sigB2_sq = v_e + (cov_ql - cov_eq - cov_el)
res["method_B_covariance_form"] = {"sigma_ref_sq": float(sigB2_sq), "sigma_ref": float(np.sqrt(max(sigB2_sq, 0)))}

# ---------- 7. sensitivity of the floor to the MAE-from-sigma assumption ---
# if reference error is Gaussian zero-mean: MAE = sigma*sqrt(2/pi)
# if reference error is heavy-tailed / has per-bond bias, floor is larger.
res["floor_assumption_probe"] = {
    "oracle_mae_gaussian_assumption_tol0.1": float(methodA[1]["oracle_mae"] if methodA[1]["tol"] == 0.1 else
        next(r["oracle_mae"] for r in methodA if r["tol"] == 0.1)),
    "oracle_mae_empirical_same_bond_pairs": None,
}
print("floor from tol=0.1:", res["floor_assumption_probe"]["oracle_mae_gaussian_assumption_tol0.1"])

# ---------- 8. ratios against candidate floors ----------------------------
def ratios(floor):
    return {c: float(res[c]["mae"] / floor) for c in ("quantum", "learned", "reference")}

floors = {"A_tol0.1": next(r["oracle_mae"] for r in methodA if r["tol"] == 0.1),
          "A_tol0.05": next(r["oracle_mae"] for r in methodA if r["tol"] == 0.05),
          "A_tol0.2": next(r["oracle_mae"] for r in methodA if r["tol"] == 0.2),
          "A_tol0.5": next(r["oracle_mae"] for r in methodA if r["tol"] == 0.5),
          "A_tol1.0": next(r["oracle_mae"] for r in methodA if r["tol"] == 1.0),
          "B_moment": res["method_B"]["oracle_mae"],
          "claim_2.26": 2.26}
if "method_A_tol0.1_replicates_only" in res:
    floors["A_replicates_only"] = res["method_A_tol0.1_replicates_only"]["oracle_mae"]
res["ratios_by_floor"] = {k: ratios(v) for k, v in floors.items()}
for k, v in res["ratios_by_floor"].items():
    print("ratio vs %-18s q=%.3f l=%.3f r=%.3f" % (k, v["quantum"], v["learned"], v["reference"]))

# ---------- 9. bootstrap CI on the floor (own implementation) -------------
rng = np.random.default_rng(7)
boot = []
for _ in range(2000):
    s = rng.integers(0, n, n)
    Es, Qs = E[s], Q[s]
    dQf = Qs[:, None] - Qs[None, :]
    iu = np.triu_indices(n, k=1)
    m = np.abs(dQf[iu]) <= 0.1
    dEs = (Es[:, None] - Es[None, :])[iu][m]
    if len(dEs) > 20:
        boot.append(np.std(dEs, ddof=1) / np.sqrt(2))
boot = np.array(boot)
res["bootstrap_floor"] = {"n_ok": int(len(boot)), "mean_sigma": float(boot.mean()),
                          "ci95_sigma": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                          "oracle_mae_ci95": [float(np.quantile(boot, .025) * MAE_FROM_SIGMA),
                                              float(np.quantile(boot, .975) * MAE_FROM_SIGMA)]}
# p-value style: fraction of bootstrap floors below the quantum MAE
res["bootstrap_floor"]["frac_floors_below_quantum_mae"] = float(np.mean(boot * MAE_FROM_SIGMA < res["quantum"]["mae"]))
res["bootstrap_floor"]["frac_floors_below_learned_mae"] = float(np.mean(boot * MAE_FROM_SIGMA < res["learned"]["mae"]))
res["bootstrap_floor"]["frac_floors_below_reference_mae"] = float(np.mean(boot * MAE_FROM_SIGMA < res["reference"]["mae"]))
print("bootstrap floor:", res["bootstrap_floor"])

OUT.write_text(json.dumps(res, indent=2, default=str))
print("wrote", OUT)
