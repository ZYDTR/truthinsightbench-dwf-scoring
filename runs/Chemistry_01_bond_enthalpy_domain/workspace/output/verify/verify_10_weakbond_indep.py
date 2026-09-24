"""Independent adversarial verification of claim 10 (weak-bond overshoot / conditioning).

Reads ONLY data/normalized_views/experimental_benchmark.tsv with plain pandas.
No import of the workstream common.py or its scripts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJ = Path("/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project")
TSV = PROJ / "data" / "normalized_views" / "experimental_benchmark.tsv"
OUT = PROJ / "output" / "results" / "verify_10_weakbond.json"

df = pd.read_csv(TSV, sep="\t")
E = df["experimental_bde_kcal_mol"].to_numpy(float)
Q = df["quantum_estimate_bde_kcal_mol"].to_numpy(float)
L = df["learned_estimate_bde_kcal_mol"].to_numpy(float)
R = df["reference_estimate_bde_kcal_mol"].to_numpy(float)

res = {}

# ---- Part A: signed error binned by EXPERIMENT (bias = pred - exp) --------
for name, P in (("learned", L), ("quantum", Q)):
    m = (E >= 70) & (E < 80)
    d = P[m] - E[m]
    res[f"exp_{name}"] = {
        "n": int(m.sum()),
        "mean_signed_error_pred_minus_exp": float(d.mean()),
        "n_overshoot": int((d > 0).sum()),
        "n_undershoot": int((d < 0).sum()),
        "n_exact_zero": int((d == 0).sum()),
        "median": float(np.median(d)),
        "sd": float(d.std(ddof=1)),
        "se": float(d.std(ddof=1) / np.sqrt(len(d))),
        "t_p": [float(x) for x in stats.ttest_1samp(d, 0.0)],
    }

# ---- Part B: bias conditioned on the ESTIMATE (exp - pred) ----------------
for name, P in (("learned", L), ("quantum", Q)):
    m = (P >= 70) & (P < 80)
    d = E[m] - P[m]
    res[f"pred_{name}"] = {
        "n": int(m.sum()),
        "mean_exp_minus_pred": float(d.mean()),
        "sd": float(d.std(ddof=1)),
        "se": float(d.std(ddof=1) / np.sqrt(len(d))),
        "t_p": [float(x) for x in stats.ttest_1samp(d, 0.0)],
    }

# ---- Part C: sanity / boundary sensitivity --------------------------------
res["boundary_counts"] = {
    f"n_exp_in_[{a},{b})": int(((E >= a) & (E < b)).sum())
    for a, b in [(60, 70), (70, 80), (80, 90), (70, 80.0000001)]
}
res["n_exp_exactly_70_or_80"] = int(((E == 70) | (E == 80)).sum())
res["n_unique_rows"] = int(len(df))
res["n_unique_(parent,fa,fb)"] = int(
    df.groupby(["parent_structure_smiles", "fragment_a_smiles", "fragment_b_smiles"]).ngroups
)

# ---- Part D: same numbers on the workstream's own cached analysis table ---
cache = PROJ / "output" / "results" / "analysis_table.csv"
if cache.exists():
    t = pd.read_csv(cache)
    res["cache_rows"] = int(len(t))
    Em = t["experimental_bde_kcal_mol"].to_numpy(float)
    Lm = t["learned_estimate_bde_kcal_mol"].to_numpy(float)
    Qm = t["quantum_estimate_bde_kcal_mol"].to_numpy(float)
    m1 = (Em >= 70) & (Em < 80)
    m2 = (Lm >= 70) & (Lm < 80)
    m3 = (Qm >= 70) & (Qm < 80)
    res["cache_check"] = {
        "exp_learned_bias": float((Lm[m1] - Em[m1]).mean()),
        "exp_learned_n": int(m1.sum()),
        "pred_learned_bias": float((Em[m2] - Lm[m2]).mean()),
        "pred_learned_n": int(m2.sum()),
        "pred_quantum_bias": float((Em[m3] - Qm[m3]).mean()),
        "pred_quantum_n": int(m3.sum()),
        "tsv_vs_cache_exp_max_abs_diff": float(np.abs(np.sort(E) - np.sort(Em)).max()),
    }

# ---- Part E: noise floor cross-check -------------------------------------
flo = PROJ / "output" / "results" / "headline_noise_floor.json"
if flo.exists():
    res["sigma_ref_from_floor_json"] = json.loads(flo.read_text())["noise_floor"]["sigma_ref"]

print(json.dumps(res, indent=2))
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
print("wrote", OUT)