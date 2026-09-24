import sys; sys.path.insert(0,'/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis')
import numpy as np, pandas as pd, common, json
j = pd.read_pickle('/tmp/bench_joined.pkl')
q = j.quantum_estimate_bde_kcal_mol.to_numpy(float)
d = j.db_bde.to_numpy(float)
diff = d - q
inb = j.in_db_row.to_numpy()
bad = inb & (np.abs(diff) > 0.5)
print(f"matched rows with |db-quantum|>0.5: {int(bad.sum())} of {int(inb.sum())}")
print("\nlargest 20 discrepancies:")
idx = np.argsort(-np.abs(np.where(inb, diff, 0)))[:20]
cols=['observation_id','parent_structure_smiles','fragment_a_smiles','fragment_b_smiles','experimental_bde_kcal_mol','quantum_estimate_bde_kcal_mol','learned_estimate_bde_kcal_mol','db_bde','db_n','db_bt']
t = j.iloc[idx][cols].copy(); t['db_minus_q']=diff[idx]
print(t.to_string())
print("\n--- decimal places of db vs quantum on matched rows ---")
def dp(x):
    s=repr(float(x)); return len(s.split('.')[-1]) if '.' in s else 0
sub=j[inb]
print("db decimals:", pd.Series([min(dp(x),4) for x in sub.db_bde]).value_counts().sort_index().to_dict())
print("quantum decimals:", pd.Series([min(dp(x),4) for x in sub.quantum_estimate_bde_kcal_mol]).value_counts().sort_index().to_dict())
# agreement at 2-dp rounding
q2 = np.round(sub.quantum_estimate_bde_kcal_mol.to_numpy(float),2)
d2 = np.round(sub.db_bde.to_numpy(float),2)
print(f"\nquantum == round(db,2): {int((q2==d2).sum())}/{len(sub)}")
print(f"quantum == round(db,1): {int((np.round(sub.quantum_estimate_bde_kcal_mol.to_numpy(float),1)==np.round(sub.db_bde.to_numpy(float),1)).sum())}/{len(sub)}")
# distribution of |diff| excluding the tail
ad=np.abs(diff[inb])
print("\n|db-quantum| quantiles among matched:", {f"q{int(p*100)}":round(float(v),4) for p,v in zip([.1,.25,.5,.75,.9,.95,.99,1.0], np.quantile(ad,[.1,.25,.5,.75,.9,.95,.99,1.0]))})
print("n |diff|<=0.005:", int((ad<=0.005).sum()), " <=0.001:", int((ad<=0.001).sum()), " ==0:", int((ad==0).sum()))
print("\nfrac of matched with |diff|>0.5:", float((ad>0.5).mean()))
