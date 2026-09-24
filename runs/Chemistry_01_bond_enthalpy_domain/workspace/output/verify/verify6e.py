"""Unit stress test: molecule-clustered bootstrap vs row bootstrap."""
import json
import numpy as np, pandas as pd
from pathlib import Path
P = Path("/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project")
b = pd.read_csv(P/"data/normalized_views/experimental_benchmark.tsv", sep="\t")
E=b.experimental_bde_kcal_mol.to_numpy(float); Q=b.quantum_estimate_bde_kcal_mol.to_numpy(float)
L=b.learned_estimate_bde_kcal_mol.to_numpy(float); R=b.reference_estimate_bde_kcal_mol.to_numpy(float)
mol=b.parent_structure_smiles.to_numpy(); n=len(E); MF=float(np.sqrt(2/np.pi))
mols=pd.unique(mol); idx_by_mol=[np.where(mol==m)[0] for m in mols]
rng=np.random.default_rng(99)

def mae_stats(sub):
    return (np.mean(np.abs(Q[sub]-E[sub])), np.mean(np.abs(L[sub]-E[sub])), np.mean(np.abs(R[sub]-E[sub])))

# molecule-cluster bootstrap of full-table MAEs
qs=[];ls=[];rs=[]
for _ in range(4000):
    picks=rng.integers(0,len(idx_by_mol),len(idx_by_mol))
    sub=np.concatenate([idx_by_mol[k] for k in picks])
    q,l,r=mae_stats(sub); qs.append(q);ls.append(l);rs.append(r)
out={"cluster_bootstrap_n_molecules":int(len(mols)),
     "quantum_mae_ci95":[float(np.quantile(qs,.025)),float(np.quantile(qs,.975))],
     "learned_mae_ci95":[float(np.quantile(ls,.025)),float(np.quantile(ls,.975))],
     "reference_mae_ci95":[float(np.quantile(rs,.025)),float(np.quantile(rs,.975))]}

# molecule-cluster bootstrap of the tol=0.1 floor: resample MOLECULES, rebuild pair set
fl=[]
for _ in range(1500):
    picks=rng.integers(0,len(idx_by_mol),len(idx_by_mol))
    sub=np.concatenate([idx_by_mol[k] for k in picks])
    Es,Qs=E[sub],Q[sub]; m=len(sub)
    if m<50: continue
    iu=np.triu_indices(m,k=1)
    dQ=(Qs[:,None]-Qs[None,:])[iu]
    mk=np.abs(dQ)<=0.1
    if mk.sum()<30: continue
    dE=(Es[:,None]-Es[None,:])[iu][mk]
    fl.append(np.std(dE,ddof=1)/np.sqrt(2)*MF)
fl=np.array(fl)
out["cluster_floor"]={"mean":float(fl.mean()),"ci95":[float(np.quantile(fl,.025)),float(np.quantile(fl,.975))],
  "P_floor_below_quantum_mae":float(np.mean(fl<2.0978084714548806)),
  "P_floor_below_learned_mae":float(np.mean(fl<2.410699815837938)),
  "P_floor_below_reference_mae":float(np.mean(fl<5.48596685082873))}
out["ratio_ci_cluster_floor"]={"quantum":[float(np.quantile(2.0978084714548806/fl,.025)),float(np.quantile(2.0978084714548806/fl,.975))],
  "learned":[float(np.quantile(2.410699815837938/fl,.025)),float(np.quantile(2.410699815837938/fl,.975))],
  "reference":[float(np.quantile(5.48596685082873/fl,.025)),float(np.quantile(5.48596685082873/fl,.975))]}

# leave-one-molecule-out sensitivity of floor and ratios (influence)
loo=[]
for k in range(len(mols)):
    mask=np.ones(n,bool); mask[idx_by_mol[k]]=False
    Es,Qs=E[mask],Q[mask]; m=mask.sum(); iu=np.triu_indices(m,k=1)
    mk=np.abs((Qs[:,None]-Qs[None,:])[iu])<=0.1
    if mk.sum()<50: continue
    dE=(Es[:,None]-Es[None,:])[iu][mk]
    loo.append(np.std(dE,ddof=1)/np.sqrt(2)*MF)
loo=np.array(loo)
out["loo_floor"]={"min":float(loo.min()),"max":float(loo.max()),"median":float(np.median(loo)),
  "max_abs_shift":float(max(abs(loo.min()-2.2689),abs(loo.max()-2.2689)))}
print(json.dumps(out,indent=1))
(P/"output/verify/verify6e_raw.json").write_text(json.dumps(out,indent=2,default=str))
