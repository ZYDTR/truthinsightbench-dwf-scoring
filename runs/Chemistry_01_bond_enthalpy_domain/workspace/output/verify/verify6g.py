"""Confound: in_reference_training_set split + subsample stability of ratios."""
import json, sys
import numpy as np, pandas as pd
from pathlib import Path
P=Path("/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project")
sys.path.insert(0,str(P/"output/analysis")); import common
bench=pd.read_csv(P/"data/normalized_views/experimental_benchmark.tsv",sep="\t")
x=common.load_xlsx()
x=x.rename(columns={"Molecule":"parent_structure_smiles","Radical 1":"fragment_a_smiles","Radical 2":"fragment_b_smiles"})
m=bench.merge(x[["parent_structure_smiles","fragment_a_smiles","fragment_b_smiles","in_reference_training_set"]],
              on=["parent_structure_smiles","fragment_a_smiles","fragment_b_smiles"],how="left",validate="one_to_one")
print("merge rows",len(m),"flag null",int(m.in_reference_training_set.isna().sum()))
MF=float(np.sqrt(2/np.pi))
out={}
for tag,sub in (("train_true",m[m.in_reference_training_set==True]),("train_false",m[m.in_reference_training_set==False]),("all",m)):
    E=sub.experimental_bde_kcal_mol.to_numpy(float);Q=sub.quantum_estimate_bde_kcal_mol.to_numpy(float)
    L=sub.learned_estimate_bde_kcal_mol.to_numpy(float);R=sub.reference_estimate_bde_kcal_mol.to_numpy(float)
    n=len(E);iu=np.triu_indices(n,k=1)
    mk=np.abs((Q[:,None]-Q[None,:])[iu])<=0.1
    dE=(E[:,None]-E[None,:])[iu][mk]
    s=np.std(dE,ddof=1)/np.sqrt(2); floor=s*MF
    out[tag]={"n":n,"n_pairs":int(mk.sum()),"sigma_ref":float(s),"oracle_mae":float(floor),
      "mae":{"quantum":float(np.mean(np.abs(Q-E))),"learned":float(np.mean(np.abs(L-E))),"reference":float(np.mean(np.abs(R-E)))},
      "ratios":{"quantum":float(np.mean(np.abs(Q-E))/floor),"learned":float(np.mean(np.abs(L-E))/floor),"reference":float(np.mean(np.abs(R-E))/floor)}}
print(json.dumps(out,indent=1))
(P/"output/verify/verify6g_raw.json").write_text(json.dumps(out,indent=2,default=str))
