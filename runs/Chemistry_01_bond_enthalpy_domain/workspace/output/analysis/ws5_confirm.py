import sys; sys.path.insert(0,'/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis')
import numpy as np, pandas as pd, common, json
from scipy import stats
j = pd.read_pickle('/tmp/bench_joined.pkl')
exp=j.experimental_bde_kcal_mol.to_numpy(float); q=j.quantum_estimate_bde_kcal_mol.to_numpy(float)
l=j.learned_estimate_bde_kcal_mol.to_numpy(float); r=j.reference_estimate_bde_kcal_mol.to_numpy(float)
db=j.db_bde.to_numpy(float); inb=j.in_db_row.to_numpy()
out={}
print("=== reference error: matched(331) vs unmatched(212) ===")
ae_m=np.abs(r[inb]-exp[inb]); ae_u=np.abs(r[~inb]-exp[~inb])
t,p=stats.ttest_ind(ae_m,ae_u,equal_var=False)
u,pu=stats.mannwhitneyu(ae_m,ae_u)
print(f"  matched  MAE={ae_m.mean():.4f} medAE={np.median(ae_m):.4f} n={len(ae_m)}")
print(f"  unmatched MAE={ae_u.mean():.4f} medAE={np.median(ae_u):.4f} n={len(ae_u)}")
print(f"  Welch t={t:.4f} p={p:.3g}   Mann-Whitney p={pu:.3g}   cliffs_delta={common.cliffs_delta(ae_m,ae_u):.4f}")
out["reference_matched_vs_unmatched"]={"mae_matched":float(ae_m.mean()),"mae_unmatched":float(ae_u.mean()),
  "median_matched":float(np.median(ae_m)),"median_unmatched":float(np.median(ae_u)),
  "welch_t":float(t),"p_welch":float(p),"p_mannwhitney":float(pu),
  "cliffs_delta":float(common.cliffs_delta(ae_m,ae_u)),"n_matched":int(inb.sum()),"n_unmatched":int((~inb).sum())}

print("\n=== how close is quantum to round(db,2)? ===")
ad=np.abs(db[inb]-q[inb]); 
for thr in [0.001,0.005,0.01,0.02,0.05]:
    print(f"  |db-quantum| <= {thr}: {int((ad<=thr).sum())}/{int(inb.sum())} = {(ad<=thr).mean():.4f}")
out["quantum_db_close"]={f"n_le_{t}":int((ad<=t).sum()) for t in [0.001,0.005,0.01,0.02,0.05,0.1,0.5]}
out["quantum_db_close"]["n"]=int(inb.sum())
# permutation test: is the quantum-vs-db closeness better than learned-vs-db by chance?
adl=np.abs(db[inb]-l[inb])
print(f"\n  median |db-quantum|={np.median(ad):.5f}  median |db-learned|={np.median(adl):.5f}  ratio={np.median(ad)/np.median(adl):.4f}")
w,pw=common.wilcoxon(ad-adl)
print(f"  paired Wilcoxon (quantum closer) W={w:.1f} p={pw:.3g}")
out["quantum_vs_learned_closeness_to_db"]={"median_abs_db_minus_quantum":float(np.median(ad)),
  "median_abs_db_minus_learned":float(np.median(adl)),"ratio":float(np.median(ad)/np.median(adl)),
  "wilcoxon_p":float(pw),"n_quantum_closer":int((ad<adl).sum()),"n_learned_closer":int((ad>adl).sum())}
print(f"  n quantum closer: {int((ad<adl).sum())}, learned closer: {int((ad>adl).sum())}")

print("\n=== experimental value quantization / ties ===")
ec=pd.Series(exp).value_counts()
out["experimental_value_reuse"]={"n_distinct":int(pd.Series(exp).nunique()),"n_rows":int(len(exp)),
   "n_values_used_once":int((ec==1).sum()),"n_values_used_multiple":int((ec>1).sum()),"max_reuse":int(ec.max())}
print(f"  distinct exp values={out['experimental_value_reuse']['n_distinct']} of {len(exp)} rows; "
      f"max reuse={ec.max()}")
json.dump(out, open('/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/results/ws5_confirm.json','w'), indent=2, default=float)
print("DONE")
