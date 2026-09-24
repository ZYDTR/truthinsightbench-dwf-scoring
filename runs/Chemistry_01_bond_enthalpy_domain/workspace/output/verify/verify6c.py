"""Robustness of the floor: sd-based vs direct moment (mean |dE|) estimator,
plus tail/outlier diagnostics."""
import json
import numpy as np, pandas as pd
from pathlib import Path

P = Path("/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project")
b = pd.read_csv(P/"data/normalized_views/experimental_benchmark.tsv", sep="\t")
E=b.experimental_bde_kcal_mol.to_numpy(float)
Q=b.quantum_estimate_bde_kcal_mol.to_numpy(float)
L=b.learned_estimate_bde_kcal_mol.to_numpy(float)
R=b.reference_estimate_bde_kcal_mol.to_numpy(float)
n=len(E); iu=np.triu_indices(n,k=1)
dQf=Q[:,None]-Q[None,:]
out={}
mae={"quantum":np.mean(np.abs(Q-E)),"learned":np.mean(np.abs(L-E)),"reference":np.mean(np.abs(R-E))}

for tol in (0.0,0.05,0.1,0.2,0.5,1.0):
    m=np.abs(dQf[iu])<=tol
    dE=(E[:,None]-E[None,:])[iu][m]
    sd_floor=float(np.std(dE,ddof=1)/np.sqrt(2)*np.sqrt(2/np.pi))
    meanabs_floor=float(np.mean(np.abs(dE))/np.sqrt(2))
    med_floor=float(np.median(np.abs(dE))/np.sqrt(2))
    # trimmed
    cut=np.quantile(np.abs(dE),0.90)
    trim_floor=float(np.mean(np.abs(dE)[np.abs(dE)<=cut])/np.sqrt(2))
    out[str(tol)]={"n":int(m.sum()),"sd_floor":sd_floor,"meanabs_floor":meanabs_floor,
        "median_floor":med_floor,"trim90_floor":trim_floor,
        "ratio_q_sd":float(mae["quantum"]/sd_floor),"ratio_q_meanabs":float(mae["quantum"]/meanabs_floor),
        "ratio_l_meanabs":float(mae["learned"]/meanabs_floor),"ratio_r_meanabs":float(mae["reference"]/meanabs_floor),
        "ratio_q_med":float(mae["quantum"]/med_floor),"ratio_l_med":float(mae["learned"]/med_floor),
        "ratio_r_med":float(mae["reference"]/med_floor)}
print(json.dumps(out,indent=1))

# tail diagnostics of the matched-pair experimental spread
m=np.abs(dQf[iu])<=0.1
dE=(E[:,None]-E[None,:])[iu][m]
out["dE_quantiles_tol01"]= {str(q): float(np.quantile(np.abs(dE),q)) for q in (0.5,0.75,0.9,0.95,0.99,1.0)}
out["frac_abs_dE_gt_8"]=float(np.mean(np.abs(dE)>8))
print("dE quantiles:", out["dE_quantiles_tol01"], "frac>8:", out["frac_abs_dE_gt_8"])

# Split-half stability of the sd-floor
rng=np.random.default_rng(3)
sh=[]
for _ in range(400):
    idx=rng.permutation(n); h1,h2=idx[:n//2],idx[n//2:]
    vals=[]
    for h in (h1,h2):
        dd=(E[h][:,None]-E[h][None,:])[np.triu_indices(len(h),k=1)]
        qq=(Q[h][:,None]-Q[h][None,:])[np.triu_indices(len(h),k=1)]
        mk=np.abs(qq)<=0.1
        if mk.sum()>20: vals.append(np.std(dd[mk],ddof=1)/np.sqrt(2)*np.sqrt(2/np.pi))
    if len(vals)==2: sh.append(vals)
sh=np.array(sh)
out["split_half_floor"]={"mean_abs_gap":float(np.mean(np.abs(sh[:,0]-sh[:,1]))),
    "corr":float(np.corrcoef(sh[:,0],sh[:,1])[0,1]),
    "p05":float(np.quantile(sh,0.05)),"p95":float(np.quantile(sh,0.95)),
    "frac_both_below_quantum_mae":float(np.mean((sh[:,0]<mae["quantum"])&(sh[:,1]<mae["quantum"])))}
print("split-half floor:", out["split_half_floor"])

# 165 / pooled-variance check for the claim-11 instruction
(P/"output/verify/verify6c_raw.json").write_text(json.dumps(out,indent=2,default=str))
print("wrote verify6c_raw.json")
