"""Robustness: outlier influence, trimmed/median errors, and a clean size-vs-domain test."""
import json
import numpy as np
import pandas as pd
from scipy import stats

P = "/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project"
b = pd.read_csv(P + "/output/results/_verify9_table.csv")
db = pd.read_csv(P + "/data/source_data/bde_db_full.csv.gz")
db_mols = set(db.molecule)
b["parent_in_db"] = b.parent_structure_smiles.isin(db_mols)
ch = b[b.bond_class == "C-H"].copy()
g = ch[ch.hac >= 10]; s = ch[ch.hac <= 5]
out = {}

def robust(df):
    dl = df.learned_ae.to_numpy(float); dq = df.quantum_ae.to_numpy(float)
    return {"n": len(df),
            "learned_median_ae": float(np.median(dl)), "quantum_median_ae": float(np.median(dq)),
            "learned_trim10": float(stats.trim_mean(dl, 0.1)), "quantum_trim10": float(stats.trim_mean(dq, 0.1)),
            "learned_max_ae": float(dl.max()), "quantum_max_ae": float(dq.max()),
            "learned_p90": float(np.quantile(dl, 0.9)), "quantum_p90": float(np.quantile(dq, 0.9))}
out["robust_ge10"] = robust(g)
out["robust_le5"] = robust(s)

# leave-one-molecule-out influence on the HAC>=10 C-H learned MAE
mols = g.parent_structure_smiles.unique()
loo = []
for m in mols:
    d = g[g.parent_structure_smiles != m]
    loo.append(float(d.learned_ae.mean()))
loo = np.array(loo)
out["loo_learned_mae_ge10"] = {"min": float(loo.min()), "max": float(loo.max()),
                               "argmin_mol": mols[int(loo.argmin())], "argmax_mol": mols[int(loo.argmax())]}
# worst single observation influence
worst_idx = g.learned_ae.idxmax()
d = g.drop(worst_idx)
out["worst_obs_removed"] = {"n": int(len(d)), "learned": float(d.learned_ae.mean()),
                            "quantum": float(d.quantum_ae.mean())}
# top-5 largest learned errors removed
d5 = g.drop(g.learned_ae.nlargest(5).index)
out["top5_removed"] = {"n": int(len(d5)), "learned": float(d5.learned_ae.mean()),
                       "quantum": float(d5.quantum_ae.mean())}

# size-vs-domain: is HAC>=10 separable from out-of-DB?
out["collinearity"] = {
    "hac_gt_10_all_out_of_db": bool((~b[b.hac > 10].parent_in_db).all()),
    "hac_eq_10_all_in_db": bool(b[b.hac == 10].parent_in_db.all()),
    "hac_ge_10_in_db_n": int(b[(b.hac >= 10) & b.parent_in_db].shape[0]),
    "hac_ge_10_out_db_n": int(b[(b.hac >= 10) & ~b.parent_in_db].shape[0]),
}
# within out-of-DB stratum, does edge grow with HAC?
odb = ch[~ch.parent_in_db]
size_edge = {}
for lo, hi in [(0, 5), (6, 9), (10, 13), (14, 100)]:
    d = odb[(odb.hac >= lo) & (odb.hac <= hi)]
    if len(d):
        size_edge[f"{lo}-{hi}"] = {"n": int(len(d)), "learned": float(d.learned_ae.mean()),
                                   "quantum": float(d.quantum_ae.mean()),
                                   "edge": float(d.learned_ae.mean() - d.quantum_ae.mean())}
out["out_of_db_size_gradient_CH"] = size_edge
# within in-DB stratum
idb = ch[ch.parent_in_db]
size_edge2 = {}
for lo, hi in [(0, 5), (6, 9), (10, 10)]:
    d = idb[(idb.hac >= lo) & (idb.hac <= hi)]
    if len(d):
        size_edge2[f"{lo}-{hi}"] = {"n": int(len(d)), "learned": float(d.learned_ae.mean()),
                                    "quantum": float(d.quantum_ae.mean()),
                                    "edge": float(d.learned_ae.mean() - d.quantum_ae.mean())}
out["in_db_size_gradient_CH"] = size_edge2

# experimental value distribution behind each slice (confound: reference hardness)
out["exp_stats"] = {
    "ge10": {"mean": float(g.experimental_bde_kcal_mol.mean()), "sd": float(g.experimental_bde_kcal_mol.std()),
             "min": float(g.experimental_bde_kcal_mol.min()), "max": float(g.experimental_bde_kcal_mol.max()),
             "n_distinct": int(g.experimental_bde_kcal_mol.nunique())},
    "le5": {"mean": float(s.experimental_bde_kcal_mol.mean()), "sd": float(s.experimental_bde_kcal_mol.std()),
            "min": float(s.experimental_bde_kcal_mol.min()), "max": float(s.experimental_bde_kcal_mol.max()),
            "n_distinct": int(s.experimental_bde_kcal_mol.nunique())},
}
# quantum estimate distribution
out["quant_stats"] = {
    "ge10": {"mean": float(g.quantum_estimate_bde_kcal_mol.mean()), "sd": float(g.quantum_estimate_bde_kcal_mol.std())},
    "le5": {"mean": float(s.quantum_estimate_bde_kcal_mol.mean()), "sd": float(s.quantum_estimate_bde_kcal_mol.std())},
}
# signed bias
out["bias"] = {
    "ge10_learned": float((g.learned_estimate_bde_kcal_mol - g.experimental_bde_kcal_mol).mean()),
    "ge10_quantum": float((g.quantum_estimate_bde_kcal_mol - g.experimental_bde_kcal_mol).mean()),
    "le5_learned": float((s.learned_estimate_bde_kcal_mol - s.experimental_bde_kcal_mol).mean()),
    "le5_quantum": float((s.quantum_estimate_bde_kcal_mol - s.experimental_bde_kcal_mol).mean()),
}
print(json.dumps(out, indent=2, default=str))
json.dump(out, open(P + "/output/results/_verify9_robust.json", "w"), indent=2, default=str)