import sys; sys.path.insert(0,'/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis')
import numpy as np, pandas as pd, common, json, time
t0=time.time(); out={}
j = pd.read_pickle('/tmp/bench_joined.pkl')
j['bond_class'] = [common.bond_class(a,b) for a,b in zip(j.fragment_a_smiles, j.fragment_b_smiles)]
j['fpair'] = j.fragment_a_smiles+'||'+j.fragment_b_smiles
j['fpair_s'] = ['||'.join(sorted(x)) for x in zip(j.fragment_a_smiles,j.fragment_b_smiles)]
print("prepared", round(time.time()-t0,1), flush=True)

def grp_stats(df, keys, val):
    g = df.groupby(keys)[val].agg(['size','min','max','nunique','mean'])
    g = g[g['size']>1]
    if len(g)==0: return None
    sp = (g['max']-g['min']).to_numpy(float)
    # within-group SD (unbiased), pooled
    var = df.groupby(keys)[val].var().dropna()
    # Pooled within-group SD: sqrt( sum_g (n_g-1) * s_g^2 / sum_g (n_g-1) ).
    # NOTE: an earlier version of this line squared the per-group variances
    # inside the weighted average, which inflated the result (165.3 instead of
    # 9.97 kcal/mol).  The verifier for this claim caught it; corrected here.
    dfree = g['size'].to_numpy(float) - 1.0
    pooled_within_sd = float(np.sqrt(np.average(var.to_numpy(float), weights=dfree)))
    return {"n_groups":int(len(g)),"n_rows":int(g['size'].sum()),
      "mean_group_size":float(g['size'].mean()),
      "mean_spread":float(sp.mean()),"median_spread":float(np.median(sp)),
      "q75_spread":float(np.quantile(sp,.75)),"q90_spread":float(np.quantile(sp,.90)),
      "q99_spread":float(np.quantile(sp,.99)),"max_spread":float(sp.max()),
      "rms_spread":float(np.sqrt(np.mean(sp**2))),
      "implied_sigma_from_range":float(np.sqrt(np.mean(sp**2))/np.sqrt(2)),
      "pooled_within_sd":pooled_within_sd,
      "mean_within_sd":float(var.mean()),
      "median_within_sd":float(var.median()),
      "n_groups_with_1_distinct":int((g['nunique']==1).sum()),
      "frac_groups_with_1_distinct":float((g['nunique']==1).mean())}

out['spread_experimental'] = {
 "by_parent_molecule":grp_stats(j,'parent_structure_smiles','experimental_bde_kcal_mol'),
 "by_parent_molecule_and_bondclass":grp_stats(j,['parent_structure_smiles','bond_class'],'experimental_bde_kcal_mol'),
 "by_fragment_pair":grp_stats(j,'fpair','experimental_bde_kcal_mol'),
 "by_fragment_pair_sorted":grp_stats(j,'fpair_s','experimental_bde_kcal_mol'),
}
out['spread_by_channel_same_grouping'] = {
 chan:grp_stats(j,'parent_structure_smiles',f"{chan}_estimate_bde_kcal_mol") for chan in ['quantum','learned','reference']}

for k,v in out['spread_experimental'].items():
    if v: print(f"\n[exp] {k}: groups={v['n_groups']} rows={v['n_rows']} mean_spread={v['mean_spread']:.3f} "
                f"median={v['median_spread']:.3f} q90={v['q90_spread']:.3f} max={v['max_spread']:.3f} "
                f"rms={v['rms_spread']:.3f} implied_sigma={v['implied_sigma_from_range']:.3f} "
                f"mean_within_sd={v['mean_within_sd']:.3f} frac_single={v['frac_groups_with_1_distinct']:.3f}")
for k,v in out['spread_by_channel_same_grouping'].items():
    print(f"[{k}] by_molecule: mean_spread={v['mean_spread']:.3f} rms={v['rms_spread']:.3f} implied_sigma={v['implied_sigma_from_range']:.3f} mean_within_sd={v['mean_within_sd']:.3f}")

# ---- null model: within-molecule spread expected if exp were exact ---------
# permutation: shuffle experimental values within molecule groups -> spread distribution
rng = np.random.default_rng(20260919)
sizes = j.groupby('parent_structure_smiles').size()
msk = j.parent_structure_smiles.isin(sizes[sizes>1].index)
sub = j[msk].copy()
base = sub.groupby('parent_structure_smiles').experimental_bde_kcal_mol.agg(['min','max'])
obs = float(np.mean((base['max']-base['min']).to_numpy()))
null_sp=[]
vals = sub.experimental_bde_kcal_mol.to_numpy(float)
for _ in range(500):
    perm = rng.permutation(vals)
    s2 = sub.copy(); s2['_p']=perm
    g2 = s2.groupby('parent_structure_smiles')._p.agg(['min','max'])
    null_sp.append(float(np.mean((g2['max']-g2['min']).to_numpy())))
print(f"\nnull (permuted within-molecule) mean spread = {np.mean(null_sp):.3f} (sd {np.std(null_sp):.3f}); observed = {obs:.3f}")
out['permutation_null_within_molecule']={"observed_mean_spread":obs,"null_mean":float(np.mean(null_sp)),
   "null_sd":float(np.std(null_sp)),"n_perm":500}
print("DONE", round(time.time()-t0,1))
json.dump(out, open('/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/results/ws5_heterogeneity.json','w'), indent=2, default=float)
