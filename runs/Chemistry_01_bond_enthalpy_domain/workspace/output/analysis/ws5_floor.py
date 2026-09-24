import sys; sys.path.insert(0,'/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis')
import numpy as np, pandas as pd, common, json, time
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
t0=time.time(); out={}
b = common.load_benchmark()
exp = b.experimental_bde_kcal_mol.to_numpy(float)
q   = b.quantum_estimate_bde_kcal_mol.to_numpy(float)
l   = b.learned_estimate_bde_kcal_mol.to_numpy(float)
ref = b.reference_estimate_bde_kcal_mol.to_numpy(float)

# ---- 1. are there ANY literal replicates in the benchmark? -----------------
def can(s):
    m=Chem.MolFromSmiles(s); return Chem.MolToSmiles(m) if m is not None else s
cm = [can(x) for x in b.parent_structure_smiles]
ca = [can(x) for x in b.fragment_a_smiles]; cb = [can(x) for x in b.fragment_b_smiles]
pair = ['||'.join(sorted([x,y])) for x,y in zip(ca,cb)]
kk = pd.Series([f"{m}@@{p}" for m,p in zip(cm,pair)])
print("unique canonical (molecule, bond) keys:", kk.nunique(), "of", len(kk))
out["canonical_mol_bond_keys"]={"n_rows":int(len(kk)),"n_unique":int(kk.nunique()),
   "n_exact_replicates":int(len(kk)-kk.nunique())}
# also molecule-level repeats
out["canonical_molecule_repeats"]={"n_unique_molecules":int(pd.Series(cm).nunique()),
   "n_rows_on_repeated_molecules":int(pd.Series(cm).map(pd.Series(cm).value_counts()).gt(1).sum())}
print("exact canonical replicates:", out["canonical_mol_bond_keys"]["n_exact_replicates"])

# ---- 2. tight-quantum-bin experimental scatter -----------------------------
# A bin narrow in quantum estimate groups chemically near-identical bonds.
# within-bin scatter of experiment then estimates reference heterogeneity
# (+ the true BDE variation inside the bin, so it is an UPPER bound on sigma_ref... 
#  but quantum also has error, so it is a MIXED bound; report both directions).
def bin_scatter(center, width, values):
    o = np.argsort(center); c=center[o]; v=values[o]
    res=[]
    half=width/2.0
    for i in range(len(c)):
        lo=np.searchsorted(c, c[i]-half, 'left'); hi=np.searchsorted(c, c[i]+half, 'right')
        if hi-lo>=3: res.append(v[lo:hi])
    if not res: return None
    sds=[np.std(x,ddof=1) for x in res]; ns=[len(x) for x in res]
    return {"n_groups":len(res),"n_obs":int(sum(ns)),"mean_n":float(np.mean(ns)),
            "mean_within_sd":float(np.mean(sds)),"pooled_within_sd":float(np.sqrt(np.average(np.array(sds)**2,weights=np.array(ns)-1))),
            "median_within_sd":float(np.median(sds))}
out["bin_scatter_vs_quantum"]={}
print("\n--- within-bin scatter, binning on the QUANTUM estimate ---")
for w in [0.5,1.0,2.0,3.0,5.0]:
    r = bin_scatter(q,w,exp)
    out["bin_scatter_vs_quantum"][f"width_{w}"]=r
    print(f"  width={w}: groups={r['n_groups']} obs={r['n_obs']} mean_n={r['mean_n']:.1f} "
          f"pooled_sd(exp within bin)={r['pooled_within_sd']:.3f} mean_sd={r['mean_within_sd']:.3f}")
    rq = bin_scatter(q,w,q); 
    print(f"            same bins, quantum's own within-bin sd = {rq['pooled_within_sd']:.4f} (calibration check)")
    out["bin_scatter_vs_quantum"][f"width_{w}_quantum_selfcheck"]=rq
print("\n--- same, binning on the LEARNED estimate ---")
out["bin_scatter_vs_learned"]={}
for w in [0.5,1.0,2.0,3.0,5.0]:
    r = bin_scatter(l,w,exp); out["bin_scatter_vs_learned"][f"width_{w}"]=r
    print(f"  width={w}: groups={r['n_groups']} obs={r['n_obs']} pooled_sd(exp)={r['pooled_within_sd']:.3f} mean_sd={r['mean_within_sd']:.3f}")
print("\n--- same, binning on the experimental value itself (reference check) ---")
out["bin_scatter_vs_experimental"]={}
for w in [1.0,2.0,3.0]:
    r=bin_scatter(exp,w,ref); out["bin_scatter_vs_experimental"][f"width_{w}"]=r
    print(f"  width={w}: groups={r['n_groups']} pooled_sd(ref within exp-bin)={r['pooled_within_sd']:.3f}")

# ---- 3. matched-pair estimator: pairs of rows with |dq| tiny --------------
# For each pair (i,j) with |q_i-q_j| <= tol, the true BDEs differ by <=tol+2*q_err.
# The SD of (exp_i-exp_j) over such pairs overestimates sqrt(2)*sigma_ref.
print("\n--- matched-pair (tight quantum difference) estimator ---")
out["matched_pair_estimator"]={}
rng=np.random.default_rng(20260919)
for tol in [0.1,0.25,0.5,1.0,2.0]:
    ii=[];jj=[]
    o=np.argsort(q)
    qs=q[o]
    for a in range(len(qs)):
        lo=np.searchsorted(qs,qs[a]-tol,'left'); hi=np.searchsorted(qs,qs[a]+tol,'right')
        for bpos in range(lo,hi):
            if bpos>a: ii.append(o[a]); jj.append(o[bpos])
    if len(ii)<5: 
        out["matched_pair_estimator"][f"tol_{tol}"]={"n_pairs":len(ii)}; print(f"  tol={tol}: n_pairs={len(ii)} (too few)"); continue
    d = exp[ii]-exp[jj]
    # bootstrap CI
    bs = np.array([np.std(rng.choice(d,size=len(d)),ddof=1)/np.sqrt(2) for _ in range(400)])
    est=float(np.std(d,ddof=1)/np.sqrt(2))
    out["matched_pair_estimator"][f"tol_{tol}"]={"n_pairs":int(len(ii)),"sd_of_exp_diff":float(np.std(d,ddof=1)),
        "sigma_ref_estimate":est,"boot_ci_lo":float(np.quantile(bs,.025)),"boot_ci_hi":float(np.quantile(bs,.975)),
        "median_abs_exp_diff":float(np.median(np.abs(d))),
        "quantum_self_sd":float(np.std(q[ii]-q[jj],ddof=1)/np.sqrt(2))}
    e=out["matched_pair_estimator"][f"tol_{tol}"]
    print(f"  tol={tol}: pairs={e['n_pairs']} sd(exp_i-exp_j)={e['sd_of_exp_diff']:.3f} -> sigma_ref~{est:.3f} "
          f"CI=[{e['boot_ci_lo']:.3f},{e['boot_ci_hi']:.3f}]  (quantum self-sd {e['quantum_self_sd']:.3f})")
json.dump(out, open('/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/results/ws5_floor_raw.json','w'), indent=2, default=float)
print("\nDONE", round(time.time()-t0,1))
