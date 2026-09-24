import sys; sys.path.insert(0,'/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis')
import numpy as np, pandas as pd, common, json
j = pd.read_pickle('/tmp/bench_joined.pkl')
exp=j.experimental_bde_kcal_mol.to_numpy(float); q=j.quantum_estimate_bde_kcal_mol.to_numpy(float)
l=j.learned_estimate_bde_kcal_mol.to_numpy(float); db=j.db_bde.to_numpy(float); inb=j.in_db_row.to_numpy()
out={}
print("=== restricted to isomeric-key-unique DB rows (db_n==1) ===")
uniq = inb & (j.db_n.to_numpy()==1)
for name,m in [("all matched",inb),("db_n==1 only",uniq),("db_n>1",inb & (j.db_n.to_numpy()>1))]:
    d=db[m]-q[m]; dl=db[m]-l[m]
    r={"n":int(m.sum()),"quantum_max_abs":float(np.abs(d).max()),"quantum_median_abs":float(np.median(np.abs(d))),
       "quantum_frac_le_0.1":float((np.abs(d)<=.1).mean()),"quantum_frac_le_0.5":float((np.abs(d)<=.5).mean()),
       "learned_max_abs":float(np.abs(dl).max()),"learned_frac_le_0.1":float((np.abs(dl)<=.1).mean()),
       "learned_frac_le_0.5":float((np.abs(dl)<=.5).mean()),
       "quantum_median_abs_ratio_to_learned":float(np.median(np.abs(d))/np.median(np.abs(dl)))}
    out[name]=r
    print(f"  {name:14s} n={r['n']:3d} quantum med|d|={r['quantum_median_abs']:.5f} frac<=0.1={r['quantum_frac_le_0.1']:.3f} "
          f"| learned med|d|={np.median(np.abs(dl)):.5f} frac<=0.1={r['learned_frac_le_0.1']:.3f} "
          f"| max {np.abs(d).max():.2f} vs {np.abs(dl).max():.2f}")
print(f"\nratio of median|diff| quantum:learned = {np.median(np.abs(db[inb]-q[inb]))/np.median(np.abs(db[inb]-l[inb])):.4f}")

print("\n=== floor self-consistency (MAE cannot go below the floor) ===")
floors={"co_tight":1.538,"tight_quantum_0.5":1.914,"decomposition_gaussian":2.260}
mae_q=float(np.abs(exp-q).mean()); mae_l=float(np.abs(exp-l).mean())
for k,v in floors.items():
    print(f"  floor {k:24s}={v:.3f}   MAE_quantum={mae_q:.3f} -> {'FEASIBLE' if mae_q>=v else 'INFEASIBLE (floor too high)'}")
    out[f"feasible_{k}"]=bool(mae_q>=v)
print(f"  => the true floor must be <= MAE_quantum = {mae_q:.3f}; empirical tight-match floors (1.54-1.91) satisfy this")
out["floor_upper_bound_from_mae"]=mae_q
out["mae_quantum"]=mae_q; out["mae_learned"]=mae_l
out["mae_gap"]=mae_l-mae_q
print(f"\n  MAE gap learned-quantum = {mae_l-mae_q:.4f}; as a multiple of the tightest floor = {(mae_l-mae_q)/1.538:.3f}")
out["mae_gap_over_tightest_floor"]=float((mae_l-mae_q)/1.538)
out["mae_gap_over_midfloor"]=float((mae_l-mae_q)/2.0)
json.dump(out, open('/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/results/ws5_consistency.json','w'), indent=2, default=float)
print("DONE")
