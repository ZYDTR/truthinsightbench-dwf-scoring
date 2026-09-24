import sys; sys.path.insert(0,'/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis')
import numpy as np, pandas as pd, common, json
j = pd.read_pickle('/tmp/bench_joined.pkl')
exp=j.experimental_bde_kcal_mol.to_numpy(float); q=j.quantum_estimate_bde_kcal_mol.to_numpy(float)
l=j.learned_estimate_bde_kcal_mol.to_numpy(float)
Vf=(np.var(exp-q,ddof=1)+np.var(exp-l,ddof=1)-np.var(q-l,ddof=1))/2
sq=float(np.var(exp-q,ddof=1)-Vf); sl=float(np.var(exp-l,ddof=1)-Vf)
print(f"sigma_ref^2={Vf:.4f}  sigma_q^2={sq:.4f}  sigma_l^2={sl:.4f}")
print(f"\nshare of QUANTUM's error variance attributable to reference values: {Vf/(Vf+sq):.4f}")
print(f"share of LEARNED's error variance attributable to reference values: {Vf/(Vf+sl):.4f}")
print(f"\nshare of quantum's MAE (linear) if errors were Gaussian: {np.sqrt(Vf)/np.sqrt(Vf+sq):.4f}")
out={"sigma_ref_sq":float(Vf),"sigma_quantum_sq":float(sq),"sigma_learned_sq":float(sl),
 "var_share_ref_in_quantum":float(Vf/(Vf+sq)),"var_share_ref_in_learned":float(Vf/(Vf+sl)),
 "mae_share_ref_in_quantum_gaussian":float(np.sqrt(Vf)/np.sqrt(Vf+sq)),
 "mae_share_ref_in_learned_gaussian":float(np.sqrt(Vf)/np.sqrt(Vf+sl)),
 "sigma_ref":float(np.sqrt(Vf)),"sigma_quantum_vs_truth":float(np.sqrt(max(sq,0))),"sigma_learned_vs_truth":float(np.sqrt(max(sl,0)))}
json.dump(out, open('/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/results/ws5_variance_share.json','w'), indent=2, default=float)
print("DONE")
