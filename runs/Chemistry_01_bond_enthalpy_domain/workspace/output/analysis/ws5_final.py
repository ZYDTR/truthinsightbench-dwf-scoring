import sys; sys.path.insert(0,'/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis')
import numpy as np, pandas as pd, common, json, time
from scipy import stats
t0=time.time(); out={}
j = pd.read_pickle('/tmp/bench_joined.pkl')
j['bond_class']=[common.bond_class(a,b) for a,b in zip(j.fragment_a_smiles,j.fragment_b_smiles)]
j['fpair']=j.fragment_a_smiles+'||'+j.fragment_b_smiles
exp=j.experimental_bde_kcal_mol.to_numpy(float); q=j.quantum_estimate_bde_kcal_mol.to_numpy(float)
l=j.learned_estimate_bde_kcal_mol.to_numpy(float); r=j.reference_estimate_bde_kcal_mol.to_numpy(float)
inb=j.in_db_row.to_numpy()

def decomp(mask, lab):
    e,q_,l_=exp[mask],q[mask],l[mask]
    Vf=(np.var(e-q_,ddof=1)+np.var(e-l_,ddof=1)-np.var(q_-l_,ddof=1))/2
    sf=float(np.sqrt(max(Vf,0))); sa=float(np.sqrt(max(np.var(e-q_,ddof=1)-Vf,0))); sb=float(np.sqrt(max(np.var(e-l_,ddof=1)-Vf,0)))
    d={"label":lab,"n":int(mask.sum()),"sigma_ref":sf,"sigma_quantum_vs_truth":sa,"sigma_learned_vs_truth":sb,
       "oracle_floor_normality":float(sf*np.sqrt(2/np.pi)),"oracle_floor_upper_bound":sf,
       "mae_quantum":float(np.abs(e-q_).mean()),"mae_learned":float(np.abs(e-l_).mean()),
       "mae_reference":float(np.abs(e-r[mask]).mean())}
    print(f"  {lab:24s} n={d['n']:3d} sigma_ref={sf:.3f} (q vs truth {sa:.3f}, l vs truth {sb:.3f}) "
          f"floor={d['oracle_floor_normality']:.3f}  MAE q={d['mae_quantum']:.3f} l={d['mae_learned']:.3f}")
    return d
print("=== variance decomposition by subset ===")
out["decomposition_by_subset"]={
 "all_543":decomp(np.ones(len(j),bool),"all 543"),
 "db_matched_331":decomp(inb,"exact DB row (331)"),
 "db_unmatched_212":decomp(~inb,"no exact DB row (212)"),
 "C-H_302":decomp((j.bond_class=='C-H').to_numpy(),"C-H (302)"),
 "C-C_240":decomp((j.bond_class=='C-C').to_numpy(),"C-C (240)"),
}
print("\n=== normality of tight-quantum matched-pair exp differences ===")
o=np.argsort(q); qs=q[o]; ii=[];jj=[]
for a in range(len(qs)):
    lo=np.searchsorted(qs,qs[a]-0.5,'left'); hi=np.searchsorted(qs,qs[a]+0.5,'right')
    for bp in range(lo,hi):
        if bp>a: ii.append(o[a]); jj.append(o[bp])
d=exp[ii]-exp[jj]
out["matched_pair_normality"]={"n_pairs":int(len(ii)),"sd":float(d.std(ddof=1)),"skew":float(stats.skew(d)),
  "excess_kurtosis":float(stats.kurtosis(d)),"shapiro_W":float(stats.shapiro(d[:5000] if len(d)>5000 else d)[0]),
  "shapiro_p":float(stats.shapiro(d[:5000] if len(d)>5000 else d)[1])}
print(f"  n={len(d)} sd={d.std(ddof=1):.3f} skew={stats.skew(d):.3f} exkurt={stats.kurtosis(d):.3f} "
      f"Shapiro W={out['matched_pair_normality']['shapiro_W']:.4f} p={out['matched_pair_normality']['shapiro_p']:.3g}")

# co-tight in both quantum and learned
def pset(a,tol,o):
    o2=np.argsort(a); as_=a[o2]; I=[];J=[]
    for x in range(len(as_)):
        lo=np.searchsorted(as_,as_[x]-tol,'left'); hi=np.searchsorted(as_,as_[x]+tol,'right')
        for y in range(lo,hi):
            if y>x: I.append(o2[x]); J.append(o2[y])
    return I,J
I1,J1=pset(q,0.5,np.arange(len(q))); I2,J2=pset(l,0.5,np.arange(len(l)))
s1=set(zip(np.minimum(I1,J1),np.maximum(I1,J1))); s2=set(zip(np.minimum(I2,J2),np.maximum(I2,J2)))
both=sorted(s1&s2)
print(f"  co-tight pairs (|dq|<=0.5 AND |dl|<=0.5): {len(both)}")
if len(both)>10:
    d2=np.array([exp[a]-exp[b] for a,b in both])
    out["co_tight_pair_estimator"]={"n_pairs":len(both),"sd_exp_diff":float(d2.std(ddof=1)),
      "sigma_ref":float(d2.std(ddof=1)/np.sqrt(2)),"median_abs":float(np.median(np.abs(d2))),
      "quantum_sd_of_diff":float(np.array([q[a]-q[b] for a,b in both]).std(ddof=1)),
      "learned_sd_of_diff":float(np.array([l[a]-l[b] for a,b in both]).std(ddof=1))}
    e=out["co_tight_pair_estimator"]
    print(f"  -> sd(exp_i-exp_j)={e['sd_exp_diff']:.3f} => sigma_ref<={e['sigma_ref']:.3f}  "
          f"(q diff sd {e['quantum_sd_of_diff']:.3f}, l diff sd {e['learned_sd_of_diff']:.3f})")

# reference channel: is it a fragment-pair lookup? final numbers
g=j.groupby('fpair').reference_estimate_bde_kcal_mol.agg(['size','nunique'])
g.columns=['size','nuniq']
out["reference_lookup"]={"n_rows":543,"n_distinct_fragment_pairs":int(j.fpair.nunique()),
  "n_distinct_reference_values":int(j.reference_estimate_bde_kcal_mol.nunique()),
  "n_pairs_with_more_than_one_distinct_ref":int((g['nuniq']>1).sum()),
  "n_pairs_with_more_than_one_row":int((g['size']>1).sum()),
  "one_value_per_fragment_pair":bool((g['nuniq']<=1).all())}
print(f"\nreference lookup: pairs={out['reference_lookup']['n_distinct_fragment_pairs']} "
      f"distinct_ref={out['reference_lookup']['n_distinct_reference_values']} "
      f"pairs_multi_ref={out['reference_lookup']['n_pairs_with_more_than_one_distinct_ref']}")

# reference error matched vs unmatched
for lab,m in [("matched331",inb),("unmatched212",~inb)]:
    e=r[m]-exp[m]
    out[f"reference_error_{lab}"]={"n":int(m.sum()),"mae":float(np.abs(e).mean()),"rmse":float(np.sqrt((e**2).mean())),
      "medae":float(np.median(np.abs(e))),"mean_signed":float(e.mean()),"max_abs":float(np.abs(e).max())}
    print(f"  reference {lab}: n={m.sum()} MAE={np.abs(e).mean():.4f} RMSE={np.sqrt((e**2).mean()):.4f} medianAE={np.median(np.abs(e)):.4f}")
json.dump(out, open('/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/results/ws5_final_raw.json','w'), indent=2, default=float)
print("DONE", round(time.time()-t0,1))
