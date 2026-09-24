import sys; sys.path.insert(0,'/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis')
import numpy as np, pandas as pd, common, json, time
t0=time.time(); out={}
b = common.load_benchmark()
exp=b.experimental_bde_kcal_mol.to_numpy(float); q=b.quantum_estimate_bde_kcal_mol.to_numpy(float)
l=b.learned_estimate_bde_kcal_mol.to_numpy(float); r=b.reference_estimate_bde_kcal_mol.to_numpy(float)

def V(x): return float(np.var(x, ddof=1))
Veq, Vel, Vql, Vqr, Vlr = V(exp-q), V(exp-l), V(q-l), V(q-r), V(l-r)
print("=== variance decomposition (543 bond observations) ===")
print(f"V(exp-q)={Veq:.4f}  V(exp-l)={Vel:.4f}  V(q-l)={Vql:.4f}  V(q-r)={Vqr:.4f}  V(l-r)={Vlr:.4f}")
# if q_err and l_err are independent of each other and of ref noise:
# V(exp-q)+V(exp-l)-V(q-l) = 2*sigma_ref^2
sr2 = (Veq+Vel-Vql)/2.0
print(f"\n=> 2*sigma_ref^2 = {Veq+Vel-Vql:.4f}  => sigma_ref = {np.sqrt(sr2):.4f}")
sref = float(np.sqrt(sr2))
sig_q = float(np.sqrt(max(Veq-sr2,0))); sig_l = float(np.sqrt(max(Vel-sr2,0)))
print(f"   implied quantum-vs-truth sd = {sig_q:.4f}   learned-vs-truth sd = {sig_l:.4f}")
print(f"   predicted sd(q-l) = {np.sqrt(sig_q**2+sig_l**2):.4f}  (observed {np.sqrt(Vql):.4f}, ratio {np.sqrt(Vql)/np.sqrt(sig_q**2+sig_l**2):.4f})")
out["variance_decomposition"]={"V_exp_minus_quantum":Veq,"V_exp_minus_learned":Vel,"V_quantum_minus_learned":Vql,
   "sigma_ref":sref,"sigma_ref_sq":sr2,"implied_quantum_vs_truth_sd":sig_q,"implied_learned_vs_truth_sd":sig_l,
   "predicted_sd_quantum_minus_learned":float(np.sqrt(sig_q**2+sig_l**2)),"observed_sd_quantum_minus_learned":float(np.sqrt(Vql)),
   "self_consistency_ratio":float(np.sqrt(Vql)/np.sqrt(sig_q**2+sig_l**2))}

# the same decomposition using reference as one of the "channels"? no - reference IS exp-derived.
# bootstrap CI on sigma_ref
rng=np.random.default_rng(20260919); n=len(exp); bs=[]
for _ in range(4000):
    i=rng.integers(0,n,n)
    v=(np.var((exp-q)[i],ddof=1)+np.var((exp-l)[i],ddof=1)-np.var((q-l)[i],ddof=1))/2
    bs.append(np.sqrt(max(v,0)))
bs=np.array(bs)
out["variance_decomposition"]["sigma_ref_ci95"]=[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))]
out["variance_decomposition"]["sigma_ref_boot_sd"]=float(bs.std(ddof=1))
print(f"   bootstrap 95% CI on sigma_ref: [{np.quantile(bs,.025):.4f}, {np.quantile(bs,.975):.4f}] (sd {bs.std(ddof=1):.4f})")

# ---- sensitivity: add the reference channel as a 3rd estimator -------------
# reference is a literature lookup, so its "error vs truth" is sigma_ref itself; skip.
# ---- floor ---------------------------------------------------------------
out["mae_floor"]={}
print("\n=== MAE floor (oracle-channel MAE against a reference with noise sigma_ref) ===")
for name,s in [("decomposition",sref),("matched_pair_tol0.1",2.851),("matched_pair_tol0.5",2.737),
               ("bin_quantum_w1.0",2.745),("bin_learned_w1.0",3.274)]:
    floor = s*np.sqrt(2/np.pi)
    out["mae_floor"][name]={"sigma_ref":float(s),"oracle_mae":float(floor)}
    print(f"  sigma_ref={s:.3f} ({name:20s}) -> oracle MAE = sigma*sqrt(2/pi) = {floor:.4f}")
floor_lo = float(np.quantile(bs,.975)*np.sqrt(2/np.pi)); floor_hi=float(np.quantile(bs,.025)*np.sqrt(2/np.pi))
print(f"  bootstrap range on the oracle MAE: [{floor_hi:.3f}, {floor_lo:.3f}]")
out["mae_floor_range_from_bootstrap"]={"lo":floor_hi,"hi":floor_lo}
# conservative headline: use the tightest (smallest) defensible sigma_ref
floor_cons = float(min(2.737, float(np.quantile(bs,.025)))*np.sqrt(2/np.pi))
out["mae_floor_conservative"]=floor_cons
print(f"  conservative headline floor (smallest defensible sigma_ref) = {floor_cons:.3f}")

# ---- observed MAEs for comparison -----------------------------------------
out["observed_mae"]={c:float(np.abs(v-exp).mean()) for c,v in [('quantum',q),('learned',l),('reference',r)]}
out["observed_mae"]["quantum_minus_learned"]=float(np.abs(q-exp).mean()-np.abs(l-exp).mean())
print(f"\nobserved MAE: quantum={out['observed_mae']['quantum']:.4f} learned={out['observed_mae']['learned']:.4f} "
      f"reference={out['observed_mae']['reference']:.4f}")
print(f"quantum MAE / floor = {out['observed_mae']['quantum']/out['mae_floor']['decomposition']['oracle_mae']:.4f}")
out["ratio_quantum_mae_to_floor"]=float(out['observed_mae']['quantum']/out['mae_floor']['decomposition']['oracle_mae'])
out["ratio_learned_mae_to_floor"]=float(out['observed_mae']['learned']/out['mae_floor']['decomposition']['oracle_mae'])
out["gap_quantum_learned_vs_floor"]=float(abs(out['observed_mae']['quantum_minus_learned'])/out['mae_floor']['decomposition']['oracle_mae'])
print(f"|MAE(q)-MAE(l)| / floor = {out['gap_quantum_learned_vs_floor']:.4f}")
json.dump(out, open('/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/results/ws5_decomposition.json','w'), indent=2, default=float)
print("\nDONE", round(time.time()-t0,1))
