import sys; sys.path.insert(0,'/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis')
import numpy as np, pandas as pd, common, json, time
from scipy import stats
t0=time.time(); out={}
b = common.load_benchmark()
exp=b.experimental_bde_kcal_mol.to_numpy(float); q=b.quantum_estimate_bde_kcal_mol.to_numpy(float)
l=b.learned_estimate_bde_kcal_mol.to_numpy(float); r=b.reference_estimate_bde_kcal_mol.to_numpy(float)
n=len(exp)

# ---- correlation structure of the error signals ----------------------------
er = exp - r; eq = exp - q; el = exp - l
print(f"sd(exp-ref)={er.std(ddof=1):.4f}  sd(exp-quantum)={eq.std(ddof=1):.4f}  sd(exp-learned)={el.std(ddof=1):.4f}")
C = np.corrcoef(np.vstack([er,eq,el]))
out["error_correlations"]={"sd_exp_minus_reference":float(er.std(ddof=1)),"sd_exp_minus_quantum":float(eq.std(ddof=1)),
  "sd_exp_minus_learned":float(el.std(ddof=1)),
  "corr_expminusref_expminusquantum":float(C[0,1]),"corr_expminusref_expminuslearned":float(C[0,2]),
  "corr_expminusquantum_expminuslearned":float(C[1,2])}
print(f"corr(exp-ref, exp-quantum) = {C[0,1]:.4f}")
print(f"corr(exp-ref, exp-learned) = {C[0,2]:.4f}")
print(f"corr(exp-quantum, exp-learned) = {C[1,2]:.4f}")

# if a channel C had error = -(exp-ref) exactly, MAE would be 0. 
# regression slope of (exp-q) on (exp-r):
sl, ic, rv, pv, se = stats.linregress(er, eq)
out["regression_expminusq_on_expminusr"]={"slope":float(sl),"intercept":float(ic),"r2":float(rv**2),"p":float(pv),"se":float(se)}
print(f"regression (exp-q) ~ (exp-r): slope={sl:.4f} r2={rv**2:.4f} p={pv:.3g}")

# ---- distinguishability: minimum detectable MAE difference -----------------
# paired |err_q| - |err_l|
d = np.abs(eq)-np.abs(el)
sd_d = float(d.std(ddof=1)); se_d = sd_d/np.sqrt(n)
# two-sided alpha=0.05, power 0.80
mde = (1.959964+0.8416212)*se_d
print(f"\npaired |err_q|-|err_l|: mean={d.mean():.4f} sd={sd_d:.4f} se={se_d:.4f}")
print(f"  minimum detectable difference (alpha=.05, power=.80, n=543) = {mde:.4f} kcal/mol")
out["minimum_detectable_mae_difference"]={"n":int(n),"sd_paired_diff":sd_d,"se":se_d,"mde_80pct_power":float(mde),
   "observed_mae_difference":float(d.mean()),"observed_p":float(common.paired_t_test(d)[1])}

# ---- how much of the channel MAE is reference noise? ----------------------
sig_ref = 2.8330
floor = sig_ref*np.sqrt(2/np.pi)
print(f"\nreference-noise floor (sigma_ref={sig_ref:.3f}): {floor:.4f}")
print(f"quantum MAE={np.abs(eq).mean():.4f} -> ratio to floor {np.abs(eq).mean()/floor:.4f}")
print(f"learned MAE={np.abs(el).mean():.4f} -> ratio to floor {np.abs(el).mean()/floor:.4f}")
out["mae_vs_floor"]={"sigma_ref":sig_ref,"oracle_floor_mae":float(floor),
  "quantum_mae":float(np.abs(eq).mean()),"learned_mae":float(np.abs(el).mean()),"reference_mae":float(np.abs(er).mean()),
  "quantum_ratio_to_floor":float(np.abs(eq).mean()/floor),"learned_ratio_to_floor":float(np.abs(el).mean()/floor),
  "reference_ratio_to_floor":float(np.abs(er).mean()/floor)}

# ---- leave-one-molecule-out robustness on the floor -----------------------
mols = b.parent_structure_smiles.to_numpy()
uq = pd.unique(mols)
rng=np.random.default_rng(20260919)
floors=[]
for _ in range(300):
    keep = rng.random(len(uq))>0.2
    idx = np.isin(mols, uq[keep])
    if idx.sum()<100: continue
    v=(np.var((exp-q)[idx],ddof=1)+np.var((exp-l)[idx],ddof=1)-np.var((q-l)[idx],ddof=1))/2
    floors.append(np.sqrt(max(v,0)))
floors=np.array(floors)
print(f"\nmolecule-subset (80%) sigma_ref: mean={floors.mean():.3f} sd={floors.std(ddof=1):.3f} "
      f"CI=[{np.quantile(floors,.025):.3f},{np.quantile(floors,.975):.3f}]")
out["molecule_subsample_robustness"]={"n_draws":int(len(floors)),"mean_sigma_ref":float(floors.mean()),
   "sd":float(floors.std(ddof=1)),"ci_lo":float(np.quantile(floors,.025)),"ci_hi":float(np.quantile(floors,.975)),
   "mean_floor":float(floors.mean()*np.sqrt(2/np.pi)),"floor_ci_lo":float(np.quantile(floors,.025)*np.sqrt(2/np.pi)),
   "floor_ci_hi":float(np.quantile(floors,.975)*np.sqrt(2/np.pi))}
# bond-class stratified
print("\nper bond-class MAE:")
out["per_bondclass"]={}
for bc in ['C-H','C-C']:
    m=(b.bond_class if 'bond_class' in b else pd.Series([common.bond_class(x,y) for x,y in zip(b.fragment_a_smiles,b.fragment_b_smiles)])).to_numpy()==bc
    e={"n":int(m.sum()),"mae_quantum":float(np.abs(eq[m]).mean()),"mae_learned":float(np.abs(el[m]).mean()),
       "mae_reference":float(np.abs(er[m]).mean())}
    # per-class sigma_ref
    v=(np.var((exp-q)[m],ddof=1)+np.var((exp-l)[m],ddof=1)-np.var((q-l)[m],ddof=1))/2
    e["sigma_ref"]=float(np.sqrt(max(v,0))); e["floor"]=float(np.sqrt(max(v,0))*np.sqrt(2/np.pi))
    out["per_bondclass"][bc]=e
    print(f"  {bc}: n={e['n']} MAE q={e['mae_quantum']:.3f} l={e['mae_learned']:.3f} r={e['mae_reference']:.3f} "
          f"sigma_ref={e['sigma_ref']:.3f} floor={e['floor']:.3f}")
json.dump(out, open('/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/results/ws5_distinguish.json','w'), indent=2, default=float)
print("\nDONE", round(time.time()-t0,1))
