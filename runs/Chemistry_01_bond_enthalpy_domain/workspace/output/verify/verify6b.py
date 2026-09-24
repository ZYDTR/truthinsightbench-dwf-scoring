"""Adversarial forensics part 2: floor validity, training flag, separability."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis")
import common   # only for load_xlsx (pandas.read_excel is broken); data read raw

P = Path("/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project")
bench = pd.read_csv(P/"data/normalized_views/experimental_benchmark.tsv", sep="\t")
E = bench.experimental_bde_kcal_mol.to_numpy(float)
Q = bench.quantum_estimate_bde_kcal_mol.to_numpy(float)
L = bench.learned_estimate_bde_kcal_mol.to_numpy(float)
R = bench.reference_estimate_bde_kcal_mol.to_numpy(float)
n = len(E)
MAE_FROM_SIGMA = float(np.sqrt(2/np.pi))
out = {}

# ---- 1. quantum value granularity / is Q a lookup table? ------------------
dec = [len(str(v).split('.')[-1]) for v in bench.quantum_estimate_bde_kcal_mol]
out["Q_granularity"] = {"decimal_digit_counts": pd.Series(dec).value_counts().to_dict(),
                        "n_unique": int(pd.Series(Q).nunique()),
                        "top_repeated_Q": {str(k): int(v) for k, v in pd.Series(Q).value_counts().head(5).to_dict().items()}}

# ---- 2. training flag from workbook --------------------------------------
x = common.load_xlsx()
flag = x.get("in_reference_training_set")
out["xlsx_columns"] = list(x.columns)
if flag is not None:
    out["training_flag_counts"] = pd.Series(flag).value_counts(dropna=False).to_dict()

# ---- 3. duplicate fragment pairs across different molecules ---------------
fp = bench.groupby(["fragment_a_smiles","fragment_b_smiles"]).size()
dups = fp[fp>1]
out["fragment_pairs_repeated"] = {"n_pairs": int(len(dups)), "detail": {str(k): int(v) for k, v in dups.to_dict().items()}}
# examine those rows: same bond in different molecules -> true near-replicates
for (a,b), cnt in dups.items():
    sub = bench[(bench.fragment_a_smiles==a)&(bench.fragment_b_smiles==b)]
    out.setdefault("repeated_frag_pair_rows", []).append({
        "frag": [a,b], "n": int(cnt),
        "molecules": list(sub.parent_structure_smiles),
        "E": list(sub.experimental_bde_kcal_mol),
        "Q": list(sub.quantum_estimate_bde_kcal_mol),
        "L": list(sub.learned_estimate_bde_kcal_mol),
        "R": list(sub.reference_estimate_bde_kcal_mol),
        "spread_E": float(sub.experimental_bde_kcal_mol.max()-sub.experimental_bde_kcal_mol.min()),
    })
print(json.dumps(out["repeated_frag_pair_rows"], indent=1, default=str))

# ---- 4. floor sensitivity: exact-match (tol=0) pairs ----------------------
iu = np.triu_indices(n, k=1)
dQf = Q[:,None]-Q[None,:]; dE = (E[:,None]-E[None,:])[iu]
for tol in (0.0, 0.02, 0.1, 0.2, 0.5, 1.0):
    m = np.abs(dQf[iu]) <= tol
    s = np.std(dE[m], ddof=1)/np.sqrt(2)
    out.setdefault("floor_scan", {})[str(tol)] = {"n": int(m.sum()), "sigma": float(s),
        "oracle_mae": float(s*MAE_FROM_SIGMA),
        "ratio_q": float(np.mean(np.abs(Q-E))/(s*MAE_FROM_SIGMA)),
        "ratio_l": float(np.mean(np.abs(L-E))/(s*MAE_FROM_SIGMA)),
        "ratio_r": float(np.mean(np.abs(R-E))/(s*MAE_FROM_SIGMA))}
print("floor scan:", json.dumps(out["floor_scan"], indent=1))

# ---- 5. separability of Q vs L: paired tests ------------------------------
ae_q = np.abs(Q-E); ae_l = np.abs(L-E); ae_r = np.abs(R-E)
from scipy import stats
diff = ae_q - ae_l
t, p = stats.ttest_rel(ae_q, ae_l)
w = stats.wilcoxon(ae_q, ae_l)
rng = np.random.default_rng(11)
bs = np.array([np.mean((ae_q-ae_l)[rng.integers(0,n,n)]) for _ in range(4000)])
out["Q_vs_L_paired"] = {"mean_abs_err_Q": float(ae_q.mean()), "mean_abs_err_L": float(ae_l.mean()),
    "diff": float(diff.mean()), "t": float(t), "p_t": float(p),
    "wilcoxon_stat": float(w.statistic), "wilcoxon_p": float(w.pvalue),
    "bootstrap_ci95_diff": [float(np.quantile(bs,.025)), float(np.quantile(bs,.975))],
    "frac_boot_diff_lt_0": float(np.mean(bs<0))}
out["Q_vs_L_winrate"] = {"frac_bonds_Q_abs_err_lt_L": float(np.mean(ae_q<ae_l)),
    "frac_equal": float(np.mean(ae_q==ae_l)),
    "frac_Q_worse": float(np.mean(ae_q>ae_l))}

# ---- 6. implied channel-vs-truth sigma from method B, and separability of
#         channel error from zero under that decomposition ------------------
v_eq=float(np.var(E-Q,ddof=1)); v_el=float(np.var(E-L,ddof=1)); v_ql=float(np.var(Q-L,ddof=1))
sref2=(v_eq+v_el-v_ql)/2; sref=float(np.sqrt(sref2))
sq=float(np.sqrt(max(v_eq-sref2,0))); sl=float(np.sqrt(max(v_el-sref2,0)))
out["methodB_implied"] = {"sigma_ref": sref, "sigma_Q_vs_truth": sq, "sigma_L_vs_truth": sl,
    "oracle_mae": sref*MAE_FROM_SIGMA,
    "implied_MAE_Q_vs_truth": sq*MAE_FROM_SIGMA, "implied_MAE_L_vs_truth": sl*MAE_FROM_SIGMA,
    "ratio_implied_Q_over_L": sq/sl}

# ---- 7. does the floor apply per-bond? check reference bias vs structure ---
out["bias"] = {c: float(np.mean(v-E)) for c,v in (("quantum",Q),("learned",L),("reference",R))}
# reference error correlation with Q-L disagreement (are they independent?)
out["corr_ref_err_with_Q_err"] = float(np.corrcoef(E-R, E-Q)[0,1])
out["corr_ref_err_with_L_err"] = float(np.corrcoef(E-R, E-L)[0,1])
out["corr_Q_err_L_err"] = float(np.corrcoef(E-Q, E-L)[0,1])

# ---- 8. floor under a stricter "same bond class" filter -------------------
import rdkit.Chem as Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
def rc(s):
    m=Chem.MolFromSmiles(s)
    if m is None: return None
    rad=[a for a in m.GetAtoms() if a.GetNumRadicalElectrons()>0]
    a=rad[0] if len(rad)==1 else m.GetAtomWithIdx(0)
    return a.GetSymbol()
def bc(a,b):
    ra,rb=rc(a),rc(b)
    if ra is None or rb is None: return None
    return "-".join(sorted([ra,rb]))
order=["C","H","N","O","S","P","F","Cl","Br","I"]
def bc2(a,b):
    ra,rb=rc(a),rc(b)
    return "-".join(sorted([ra,rb], key=lambda e: order.index(e) if e in order else 99))
cls=[bc2(a,b) for a,b in zip(bench.fragment_a_smiles, bench.fragment_b_smiles)]
bench["bclass"]=cls
out["bond_class_counts"]=pd.Series(cls).value_counts().to_dict()
m01 = np.abs(dQf[iu])<=0.1
i,j=iu[0][m01], iu[1][m01]
same_cls = np.array(cls)[i]==np.array(cls)[j]
for tag, mask in (("all", np.ones(len(i),bool)), ("same_bond_class", same_cls)):
    s=np.std(dE[m01][mask],ddof=1)/np.sqrt(2)
    out.setdefault("floor_tol01_by_class",{})[tag]={"n":int(mask.sum()),"sigma":float(s),
        "oracle_mae":float(s*MAE_FROM_SIGMA),
        "ratio_q":float(ae_q.mean()/(s*MAE_FROM_SIGMA))}
print("class-filtered floor:", json.dumps(out["floor_tol01_by_class"], indent=1))
print("bias:", out["bias"]); print("corrs:", {k:round(v,3) for k,v in out.items() if k.startswith("corr")})

(P/"output/verify/verify6b_raw.json").write_text(json.dumps(out, indent=2, default=str))
print("wrote verify6b_raw.json")
