"""Boilerplate checks appended to every claim prompt:
claim-5 (two independent noise-floor methods) and claim-11 (pooled within-molecule SD)."""
import json
import numpy as np
import pandas as pd

P = "/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project"
b = pd.read_csv(P + "/data/normalized_views/experimental_benchmark.tsv", sep="\t")
E = b.experimental_bde_kcal_mol.to_numpy(float)
Q = b.quantum_estimate_bde_kcal_mol.to_numpy(float)
L = b.learned_estimate_bde_kcal_mol.to_numpy(float)
n = len(b)
out = {}

# ---- claim 5, method A: near-duplicate pairs matched on quantum estimate ----
dQ = np.abs(Q[:, None] - Q[None, :])
iu = np.triu_indices(n, k=1)
mask = dQ[iu] <= 0.1
dE = E[:, None] - E[None, :]
diff = dE[iu][mask]
out["methodA"] = {
    "n_pairs": int(mask.sum()),
    "sigma_ref": float(np.sqrt(np.mean(diff ** 2) / 2.0)),
    "note": "pairs with |Q_i-Q_j|<=0.1; Var(E_i-E_j)=2 sigma^2",
}

# ---- claim 5, method B: moment decomposition ----
# Var(E-Q) = sig_ref^2 + sig_q^2 ; Var(E-L) = sig_ref^2 + sig_l^2 ;
# Var(Q-L) = sig_q^2 + sig_l^2 - 2cov(q,l).  Assuming cov(q,l)~0:
# sig_ref^2 = [V(E-Q)+V(E-L)-V(Q-L)]/2
V = lambda x: float(np.var(x, ddof=1))
v1, v2, v3 = V(E - Q), V(E - L), V(Q - L)
out["methodB"] = {
    "V_E_minus_Q": v1, "V_E_minus_L": v2, "V_Q_minus_L": v3,
    "sigma_ref_sq": (v1 + v2 - v3) / 2.0,
    "sigma_ref": float(np.sqrt(max((v1 + v2 - v3) / 2.0, 0.0))),
}
oracle_mae = float(np.sqrt(2 / np.pi) * np.sqrt((v1 + v2 - v3) / 2.0))
out["oracle_mae_from_B"] = oracle_mae
out["agreement_AB"] = abs(out["methodA"]["sigma_ref"] - out["methodB"]["sigma_ref"])

# ---- claim 11: pooled within-molecule SD ----
g = b.groupby("parent_structure_smiles").experimental_bde_kcal_mol
sizes = g.size()
multi = sizes[sizes > 1]
n_g = multi.to_numpy(float)
s_g = g.std(ddof=1).loc[multi.index].to_numpy(float)
mean_g = g.mean().loc[multi.index].to_numpy(float)
spread = (g.max() - g.min()).loc[multi.index].to_numpy(float)
pooled_var = float(np.sum((n_g - 1) * s_g ** 2) / np.sum(n_g - 1))
out["claim11"] = {
    "n_unique_molecules": int(sizes.size),
    "n_multi_row_molecules": int(len(multi)),
    "mean_within_spread": float(spread.mean()),
    "median_within_spread": float(np.median(spread)),
    "max_within_spread": float(spread.max()),
    "pooled_within_sd_correct": float(np.sqrt(pooled_var)),
    # naive-variant candidates for the ~165 figure
    "naive_var_squared_weighted": float(np.sqrt(np.average(s_g ** 2, weights=n_g - 1))) if False else None,
    "naive_mean_of_variances_ddof0": float(np.sqrt(np.average(s_g ** 2))),
    "naive_n_weighted_ddof0": float(np.sqrt(np.average(s_g ** 2, weights=n_g))),
    "naive_sqrt_sum_variances": float(np.sqrt(np.sum(s_g ** 2))),
    "naive_squared_var_weighted": float(np.sqrt(np.average((s_g ** 2) ** 2, weights=n_g - 1))),
    "naive_pooled_over_unique_mol": float(np.sqrt(np.sum((n_g - 1) * s_g ** 2) / len(sizes))),
}
print(json.dumps(out, indent=2))
json.dump(out, open(P + "/output/results/_verify9_boiler.json", "w"), indent=2)