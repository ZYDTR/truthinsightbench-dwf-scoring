import sys; sys.path.insert(0,'/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis')
import numpy as np, pandas as pd, common, json, time
t0=time.time(); out={}
j = pd.read_pickle('/tmp/bench_joined.pkl')
exp=j.experimental_bde_kcal_mol.to_numpy(float); q=j.quantum_estimate_bde_kcal_mol.to_numpy(float)
l=j.learned_estimate_bde_kcal_mol.to_numpy(float); r=j.reference_estimate_bde_kcal_mol.to_numpy(float)
rng=np.random.default_rng(20260919)

def pairs(a,tol):
    o=np.argsort(a); s=a[o]; I=[];J=[]
    for x in range(len(s)):
        lo=np.searchsorted(s,s[x]-tol,'left'); hi=np.searchsorted(s,s[x]+tol,'right')
        for y in range(lo,hi):
            if y>x: I.append(o[x]); J.append(o[y])
    return np.array(I),np.array(J)

def empirical_floor(I,J,label):
    if len(I)<20: return None
    ad=np.abs(exp[I]-exp[J])
    # E|X_i - X_j| = sqrt(2)*E|X| for iid; => floor = mean|diff|/sqrt(2)
    fl=float(ad.mean()/np.sqrt(2))
    bs=np.array([np.mean(rng.choice(ad,size=len(ad)))/np.sqrt(2) for _ in range(2000)])
    r={"label":label,"n_pairs":int(len(I)),"mean_abs_exp_diff":float(ad.mean()),
       "median_abs_exp_diff":float(np.median(ad)),"floor_empirical_mae":fl,
       "ci_lo":float(np.quantile(bs,.025)),"ci_hi":float(np.quantile(bs,.975)),
       "sigma_equiv":float(ad.std(ddof=1)/np.sqrt(2)),
       "quantum_diff_sd":float((q[I]-q[J]).std(ddof=1)),"learned_diff_sd":float((l[I]-l[J]).std(ddof=1))}
    print(f"  {label:28s} pairs={r['n_pairs']:5d} mean|dexp|={ad.mean():.3f} -> floor MAE={fl:.3f} "
          f"CI=[{r['ci_lo']:.3f},{r['ci_hi']:.3f}] (sigma_eq {r['sigma_equiv']:.3f})")
    return r
print("=== empirical MAE floor preserving the observed tail shape ===")
out["empirical_floor"]={}
out["empirical_floor"]["tight_quantum_0.1"]=empirical_floor(*pairs(q,0.1),"|dq|<=0.1")
out["empirical_floor"]["tight_quantum_0.25"]=empirical_floor(*pairs(q,0.25),"|dq|<=0.25")
out["empirical_floor"]["tight_quantum_0.5"]=empirical_floor(*pairs(q,0.5),"|dq|<=0.5")
out["empirical_floor"]["tight_quantum_1.0"]=empirical_floor(*pairs(q,1.0),"|dq|<=1.0")
I1,J1=pairs(q,0.5); I2,J2=pairs(l,0.5)
s1=set(zip(np.minimum(I1,J1),np.maximum(I1,J1))); s2=set(zip(np.minimum(I2,J2),np.maximum(I2,J2)))
both=np.array(sorted(s1&s2))
out["empirical_floor"]["co_tight_0.5"]=empirical_floor(both[:,0],both[:,1],"co-tight |dq|,|dl|<=0.5")

# NOW: the strongest bound - channels' disagreement vs these floors
print("\n=== channel disagreement relative to the floor ===")
def mae(x): return float(np.abs(x).mean())
out["headline"]={}
fl = out["empirical_floor"]["co_tight_0.5"]["floor_empirical_mae"]
fl_tq = out["empirical_floor"]["tight_quantum_0.5"]["floor_empirical_mae"]
fl_dec = 2.8330*np.sqrt(2/np.pi)
out["headline"]["floor_cotight"]=fl; out["headline"]["floor_tight_quantum"]=fl_tq
out["headline"]["floor_decomposition_gaussian"]=float(fl_dec)
out["headline"]["floor_bootstrap_ci"]=[out["empirical_floor"]["co_tight_0.5"]["ci_lo"],out["empirical_floor"]["co_tight_0.5"]["ci_hi"]]
print(f"  floors: co-tight={fl:.3f}  tight-quantum={fl_tq:.3f}  decomposition(gauss)={fl_dec:.3f}")
out["headline"]["mae_quantum"]=mae(exp-q); out["headline"]["mae_learned"]=mae(exp-l); out["headline"]["mae_reference"]=mae(exp-r)
for k in ['mae_quantum','mae_learned','mae_reference']:
    print(f"  {k}={out['headline'][k]:.4f}  ratio to co-tight floor = {out['headline'][k]/fl:.4f}")
# the quantum-learned gap
out["headline"]["mae_gap_quantum_learned"]=float(mae(exp-q)-mae(exp-l))
out["headline"]["mae_gap_over_floor"]=float((mae(exp-l)-mae(exp-q))/fl)
out["headline"]["mae_gap_over_tight_floor"]=float((mae(exp-l)-mae(exp-q))/fl_tq)
print(f"  MAE(learned)-MAE(quantum) = {out['headline']['mae_gap_quantum_learned']:.4f} "
      f"= {out['headline']['mae_gap_over_floor']:.3f} x co-tight floor, {out['headline']['mae_gap_over_tight_floor']:.3f} x tight-quantum floor")
# reference MAE relative to floor
out["headline"]["reference_mae_over_floor"]=float(out['headline']['mae_reference']/fl)
print(f"  reference MAE = {out['headline']['reference_mae_over_floor']:.3f} x co-tight floor")
json.dump(out, open('/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/results/ws5_floor_empirical.json','w'), indent=2, default=float)
print("DONE", round(time.time()-t0,1))
