"""Robust-scale floor estimates vs the Gaussian sd conversion."""
import json
import numpy as np, pandas as pd
from scipy import stats
from pathlib import Path
P=Path("/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project")
b=pd.read_csv(P/"data/normalized_views/experimental_benchmark.tsv",sep="\t")
E=b.experimental_bde_kcal_mol.to_numpy(float);Q=b.quantum_estimate_bde_kcal_mol.to_numpy(float)
L=b.learned_estimate_bde_kcal_mol.to_numpy(float);R=b.reference_estimate_bde_kcal_mol.to_numpy(float)
n=len(E);iu=np.triu_indices(n,k=1);MF=float(np.sqrt(2/np.pi))
MAE={"quantum":float(np.mean(np.abs(Q-E))),"learned":float(np.mean(np.abs(L-E))),"reference":float(np.mean(np.abs(R-E)))}
out={"MAE":MAE}
m=np.abs((Q[:,None]-Q[None,:])[iu])<=0.1
dE=(E[:,None]-E[None,:])[iu][m]
sd=float(np.std(dE,ddof=1)); mad=float(stats.median_abs_deviation(dE,scale=1.0))
q1,q3=np.quantile(dE,[.25,.75]); iqr=float(q3-q1)
rob_sd=mad*1.4826                     # normal-consistent robust sigma of dE
rob_sd_iqr=iqr/1.349
def conv(sig_dE, mode):
    s=sig_dE/np.sqrt(2)
    return s*MF if mode=="gauss" else float(np.mean(np.abs(dE)))/2
ests={
 "sd_gauss": sd/np.sqrt(2)*MF,
 "mad_gauss": rob_sd/np.sqrt(2)*MF,
 "iqr_gauss": rob_sd_iqr/np.sqrt(2)*MF,
 "meanabs_model_free": float(np.mean(np.abs(dE)))/2,
 "trimmed_meanabs_90": float(np.mean(np.abs(dE)[np.abs(dE)<=np.quantile(np.abs(dE),.90)]))/2,
 "median_abs": float(np.median(np.abs(dE)))/2,
}
out["floor_estimators_tol0.1"]={k:{"floor":v,"ratios":{c:MAE[c]/v for c in MAE}} for k,v in ests.items()}
out["outlier_share_of_variance"]={
 "frac_var_from_top1pct_abs_pairs": float(np.sum(dE[np.abs(dE)>=np.quantile(np.abs(dE),.99)]**2)/np.sum(dE**2)),
 "frac_var_from_top5pct_abs_pairs": float(np.sum(dE[np.abs(dE)>=np.quantile(np.abs(dE),.95)]**2)/np.sum(dE**2)),
 "n_pairs": int(len(dE)), "n_abs_gt_8": int(np.sum(np.abs(dE)>8))}
print(json.dumps(out,indent=1))
(P/"output/verify/verify6h_raw.json").write_text(json.dumps(out,indent=2,default=str))
