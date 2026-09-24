import sys; sys.path.insert(0,'/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis')
import numpy as np, pandas as pd, common, json, time
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
t0=time.time(); out={}
b = common.load_benchmark()
exp = b.experimental_bde_kcal_mol.to_numpy(float)
ref = b.reference_estimate_bde_kcal_mol.to_numpy(float)
expset = set(np.round(exp,6).tolist())

# ---- coincidence: ref value equals SOME experimental value in the file -----
hit = sum(1 for x in np.round(ref,6) if x in expset)
lo, hi = float(exp.min()), float(exp.max())
grid = np.round(np.arange(lo, hi+0.05, 0.1), 6)   # 0.1 grid, the ref quantization
rng = np.random.default_rng(20260919)
null=[]
for _ in range(2000):
    draw = rng.choice(grid, size=len(ref))
    null.append(sum(1 for x in draw if x in expset))
print(f"ref value in experimental-value set: {hit}/543")
print(f"null (uniform on 0.1 grid over exp range): mean={np.mean(null):.1f} sd={np.std(null):.1f} "
      f"=> z={(hit-np.mean(null))/np.std(null):.2f}")
out["ref_equals_some_experimental_value"]={"observed":int(hit),"n":len(ref),
   "null_mean":float(np.mean(null)),"null_sd":float(np.std(null)),
   "z":float((hit-np.mean(null))/np.std(null)),"n_perm":2000}
# same but restricted to exact same fragment pair
fx = b.fragment_a_smiles+'||'+b.fragment_b_smiles
same=0
for i,(f_,rv) in enumerate(zip(fx, np.round(ref,6))):
    m = (fx==f_).to_numpy() & (np.arange(len(b))!=i)
    if rv in set(np.round(exp[m],6).tolist()): same+=1
print(f"ref equals the experimental value of the SAME fragment pair in a DIFFERENT molecule: {same}/543")
out["ref_equals_exp_same_fpair_other_molecule"]=int(same)

# ---- canonicalised fragment pairs: does any pair collapse further? ---------
def can(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m is not None else s
ca, cb = [can(x) for x in b.fragment_a_smiles], [can(x) for x in b.fragment_b_smiles]
b['can_pair'] = [ '||'.join(sorted([x,y])) for x,y in zip(ca,cb)]
g = b.groupby('can_pair').experimental_bde_kcal_mol.agg(['size','min','max','nunique'])
g = g[g['size']>1]
sp=(g['max']-g['min']).to_numpy(float)
print(f"\ncanonicalised unordered fragment pairs with >1 row: {len(g)} (raw was 4), rows={int(g['size'].sum())}")
print(f"  mean spread={sp.mean():.3f} median={np.median(sp):.3f} rms={np.sqrt(np.mean(sp**2)):.3f} max={sp.max():.3f}")
out["canonical_pair_groups"]={"n_groups":int(len(g)),"n_rows":int(g['size'].sum()),
  "mean_spread":float(sp.mean()),"median_spread":float(np.median(sp)),"max_spread":float(sp.max()),
  "rms_spread":float(np.sqrt(np.mean(sp**2))),"implied_sigma":float(np.sqrt(np.mean(sp**2)/2)),
  "n_rows_same_molecule":int((b.groupby(['can_pair','parent_structure_smiles']).size()>1).sum())}
# same-molecule duplicates under canonicalisation
gm = b.groupby(['can_pair','parent_structure_smiles']).experimental_bde_kcal_mol.agg(['size','min','max'])
gm = gm[gm['size']>1]
spm=(gm['max']-gm['min']).to_numpy(float)
print(f"  of which same parent molecule: {len(gm)} groups, mean spread={spm.mean() if len(spm) else float('nan'):.3f}")
out["canonical_pair_same_molecule_groups"]={"n_groups":int(len(gm)),"n_rows":int(gm['size'].sum()),
  "mean_spread":float(spm.mean()) if len(spm) else None,"max_spread":float(spm.max()) if len(spm) else None}

# ---- is ref constant within canonical pair? -------------------------------
g2 = b.groupby('can_pair').reference_estimate_bde_kcal_mol.nunique()
print(f"\ncanonical pairs with >1 distinct reference value: {int((g2>1).sum())} of {len(g2)}")
out["canonical_pair_multi_ref"]={"n_pairs":int(len(g2)),"n_pairs_multi_ref":int((g2>1).sum())}
json.dump(out, open('/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/results/ws5_ref_origin.json','w'), indent=2, default=float)
print("DONE", round(time.time()-t0,1))
