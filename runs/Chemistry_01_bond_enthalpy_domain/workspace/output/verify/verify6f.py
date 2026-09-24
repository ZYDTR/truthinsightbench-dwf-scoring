"""Estimator-dependence of the oracle floor + distribution diagnostics."""
import json
import numpy as np, pandas as pd
from scipy import stats
from pathlib import Path
P = Path("/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project")
b = pd.read_csv(P/"data/normalized_views/experimental_benchmark.tsv", sep="\t")
E=b.experimental_bde_kcal_mol.to_numpy(float); Q=b.quantum_estimate_bde_kcal_mol.to_numpy(float)
L=b.learned_estimate_bde_kcal_mol.to_numpy(float); R=b.reference_estimate_bde_kcal_mol.to_numpy(float)
n=len(E); iu=np.triu_indices(n,k=1); dQf=Q[:,None]-Q[None,:]
MAE={"quantum":float(np.mean(np.abs(Q-E))),"learned":float(np.mean(np.abs(L-E))),"reference":float(np.mean(np.abs(R-E)))}
MF=float(np.sqrt(2/np.pi)); out={}

for tol in (0.0,0.05,0.1,0.2,0.5,1.0):
    m=np.abs(dQf[iu])<=tol
    dE=(E[:,None]-E[None,:])[iu][m]
    sd=float(np.std(dE,ddof=1))
    gauss=sd/np.sqrt(2)*MF                 # workstream's estimator  (0.5642*sd)
    gini=float(np.mean(np.abs(dE)))/2.0    # model-free: E|dE|=2E|eps| for iid sym errors
    out[str(tol)]={"n":int(m.sum()),"sd_dE":sd,"mean_abs_dE":float(np.mean(np.abs(dE))),
        "kurtosis":float(stats.kurtosis(dE,fisher=True,bias=False)),
        "skew":float(stats.skew(dE,bias=False)),
        "shapiro_p":float(stats.shapiro(dE[:min(len(dE),5000)]).pvalue),
        "floor_gauss":gauss,"floor_gini":gini,
        "ratios_gauss":{k:MAE[k]/gauss for k in MAE},
        "ratios_gini":{k:MAE[k]/gini for k in MAE},
        "skew_of_E_minus_Q_pairs":float(stats.skew(dE,bias=False))}
print(json.dumps(out,indent=1))
out["MAE_observed"]=MAE

# bootstrap CI of the Gini floor at tol=0.1 (row bootstrap)
rng=np.random.default_rng(5); g=[]
for _ in range(3000):
    s=rng.integers(0,n,n); Es,Qs=E[s],Q[s]
    mk=np.abs((Qs[:,None]-Qs[None,:])[iu])<=0.1
    if mk.sum()<50: continue
    g.append(np.mean(np.abs((Es[:,None]-Es[None,:])[iu][mk]))/2.0)
g=np.array(g)
out["gini_floor_ci95"]=[float(np.quantile(g,.025)),float(np.quantile(g,.975))]
out["gini_ratio_ci95"]={k:[float(np.quantile(MAE[k]/g,.025)),float(np.quantile(MAE[k]/g,.975))] for k in MAE}
print("gini floor ci:", out["gini_floor_ci95"]); print("gini ratios:", json.dumps(out["gini_ratio_ci95"],indent=1))

# internal-consistency tension: observed quantum MAE vs floor
out["tension"]={"quantum_mae_below_floor_by":float(2.268915657606384-MAE["quantum"]),
 "learned_minus_floor":float(MAE["learned"]-2.268915657606384),
 "reference_minus_floor":float(MAE["reference"]-2.268915657606384),
 "note":"under independence observed MAE should be sqrt(floor^2 + channel^2) > floor; quantum violates this"}
print("tension:", out["tension"])
(P/"output/verify/verify6f_raw.json").write_text(json.dumps(out,indent=2,default=str))
print("wrote verify6f_raw.json")
