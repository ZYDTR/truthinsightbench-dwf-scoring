"""Separability tests + pooled-variance (claim-11 style) check."""
import json
from pathlib import Path
import numpy as np, pandas as pd

P = Path("/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project")
b = pd.read_csv(P/"data/normalized_views/experimental_benchmark.tsv", sep="\t")
E=b.experimental_bde_kcal_mol.to_numpy(float); Q=b.quantum_estimate_bde_kcal_mol.to_numpy(float)
L=b.learned_estimate_bde_kcal_mol.to_numpy(float); R=b.reference_estimate_bde_kcal_mol.to_numpy(float)
n=len(E); MF=float(np.sqrt(2/np.pi)); rng=np.random.default_rng(2026)
out={}

# ---- A) proper joint bootstrap: learned MAE minus floor, same resample -----
diffs=[]; qm=[]; lm=[]; fl=[]
for _ in range(4000):
    s=rng.integers(0,n,n); Es,Qs,Ls=E[s],Q[s],L[s]
    iu=np.triu_indices(n,k=1)
    dQ=(Qs[:,None]-Qs[None,:])[iu]
    m=np.abs(dQ)<=0.1
    if m.sum()<50: continue
    dE=(Es[:,None]-Es[None,:])[iu][m]
    f=float(np.std(dE,ddof=1)/np.sqrt(2)*MF)
    q=float(np.mean(np.abs(Qs-Es))); l=float(np.mean(np.abs(Ls-Es)))
    fl.append(f); qm.append(q); lm.append(l); diffs.append(l-f)
diffs=np.array(diffs); fl=np.array(fl); qm=np.array(qm); lm=np.array(lm)
out["joint_bootstrap"]={
 "floor_mean":float(fl.mean()),"floor_ci95":[float(np.quantile(fl,.025)),float(np.quantile(fl,.975))],
 "P_floor_below_Q_MAE":float(np.mean(fl<qm)),"P_floor_below_L_MAE":float(np.mean(fl<lm)),
 "learned_minus_floor_mean":float(diffs.mean()),
 "learned_minus_floor_ci95":[float(np.quantile(diffs,.025)),float(np.quantile(diffs,.975))],
 "P_learned_above_floor":float(np.mean(diffs>0)),
 "quantum_minus_floor_mean":float(np.mean(qm-fl)),
 "quantum_minus_floor_ci95":[float(np.quantile(qm-fl,.025)),float(np.quantile(qm-fl,.975))],
 "P_quantum_above_floor":float(np.mean((qm-fl)>0))}
print("joint bootstrap:", json.dumps(out["joint_bootstrap"],indent=1))

# ---- B) method-B implied channel-vs-truth sigma: bootstrap CI -------------
bs=[]
for _ in range(4000):
    s=rng.integers(0,n,n); Es,Qs,Ls=E[s],Q[s],L[s]
    v_eq=np.var(Es-Qs,ddof=1); v_el=np.var(Es-Ls,ddof=1); v_ql=np.var(Qs-Ls,ddof=1)
    sr2=(v_eq+v_el-v_ql)/2
    bs.append((np.sqrt(max(sr2,0)), np.sqrt(max(v_eq-sr2,0)), np.sqrt(max(v_el-sr2,0))))
bs=np.array(bs); bs=bs[np.isfinite(bs).all(axis=1)]
out["methodB_bootstrap"]={
 "sigma_ref_ci95":[float(np.quantile(bs[:,0],.025)),float(np.quantile(bs[:,0],.975))],
 "sigma_Q_vs_truth_ci95":[float(np.quantile(bs[:,1],.025)),float(np.quantile(bs[:,1],.975))],
 "sigma_L_vs_truth_ci95":[float(np.quantile(bs[:,2],.025)),float(np.quantile(bs[:,2],.975))],
 "P_sigma_Q_vs_truth_gt_0":float(np.mean(bs[:,1]>0)),
 "P_sigma_L_vs_truth_gt_0":float(np.mean(bs[:,2]>0)),
 "P_sigma_L_gt_sigma_Q":float(np.mean(bs[:,2]>bs[:,1])),
 "implied_MAE_Q_ci95":[float(np.quantile(bs[:,1],.025)*MF),float(np.quantile(bs[:,1],.975)*MF)],
 "implied_MAE_L_ci95":[float(np.quantile(bs[:,2],.025)*MF),float(np.quantile(bs[:,2],.975)*MF)]}
print("methodB bootstrap:", json.dumps(out["methodB_bootstrap"],indent=1))

# ---- C) direct observed MAE separability Q vs L vs floor CI ---------------
out["mae_ci95_observed"]={}
for nm,x in (("quantum",Q),("learned",L),("reference",R)):
    ae=np.abs(x-E); bsamp=np.array([ae[rng.integers(0,n,n)].mean() for _ in range(4000)])
    out["mae_ci95_observed"][nm]={"mae":float(ae.mean()),
        "ci95":[float(np.quantile(bsamp,.025)),float(np.quantile(bsamp,.975))]}
print("observed MAE CIs:", json.dumps(out["mae_ci95_observed"],indent=1))

# ---- D) pooled-variance / 165 check (claim-11 instruction, boilerplate) ---
grp=b.groupby("parent_structure_smiles")["experimental_bde_kcal_mol"]
within=grp.agg(["count","var","mean","min","max","std"])
multi=within[within["count"]>1]
num=(multi["count"]-1)*multi["var"]
correct=float(np.sqrt(num.sum()/ (multi["count"]-1).sum()))
naive_ddof0=float(np.sqrt(np.average(multi["var"],weights=multi["count"])))
unweighted=float(np.sqrt(multi["var"].mean()))
sumsq=float(np.sqrt((multi["var"]**2).sum()))
var_squared_weighted=float(np.sqrt(np.average(multi["var"].to_numpy(float)**2,weights=(multi["count"]-1))))
out["pooled_variance_check"]={
 "n_multi_row_molecules":int(len(multi)),
 "correct_pooled_within_sd_ddof1":correct,
 "naive_ddof0_nweighted":naive_ddof0,
 "unweighted_mean_of_var":unweighted,
 "sqrt_sum_of_variances":sumsq,
 "var_squared_weighted_artifact":var_squared_weighted,
 "mean_within_spread":float((multi["max"]-multi["min"]).mean()),
 "median_within_spread":float((multi["max"]-multi["min"]).median()),
 "max_within_spread":float((multi["max"]-multi["min"]).max())}
print("pooled check:", json.dumps(out["pooled_variance_check"],indent=1))

(P/"output/verify/verify6d_raw.json").write_text(json.dumps(out,indent=2,default=str))
print("wrote verify6d_raw.json")
