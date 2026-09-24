import sys; sys.path.insert(0,'/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis')
import json, os
import numpy as np, pandas as pd, common
out = {}
bench = common.load_benchmark()
db = pd.read_pickle('/tmp/db.pkl')
dbk = db[['molecule','fragment1','fragment2','bde']].drop_duplicates(subset=['molecule','fragment1','fragment2'])
j = bench.merge(dbk, left_on=['parent_structure_smiles','fragment_a_smiles','fragment_b_smiles'],
                right_on=['molecule','fragment1','fragment2'], how='left', indicator=True)
j['in_db_row'] = (j._merge=='both').to_numpy()
j.to_pickle('/tmp/bench_joined.pkl')
print("in_db_row:", int(j.in_db_row.sum()), "/", len(j))

# ---- 3a: is reference a per-(fragment_a, fragment_b) lookup? ---------------
j['fpair'] = j.fragment_a_smiles + '||' + j.fragment_b_smiles
j['fpair_sorted'] = [ '||'.join(sorted(x)) for x in zip(j.fragment_a_smiles, j.fragment_b_smiles)]
ref = j.reference_estimate_bde_kcal_mol
for keyname in ['fpair','fpair_sorted']:
    g = j.groupby(keyname).reference_estimate_bde_kcal_mol.agg(['size','nunique'])
    out[f"ref_lookup_{keyname}"] = {
        "n_rows": int(len(j)), "n_distinct_fragment_pairs": int(j[keyname].nunique()),
        "n_distinct_reference_values": int(ref.nunique()),
        "n_pairs_with_more_than_one_row": int((g['size']>1).sum()),
        "n_pairs_with_more_than_one_distinct_ref": int((g['nunique']>1).sum()),
        "n_pairs_with_one_row": int((g['size']==1).sum()),
        "max_distinct_ref_per_pair": int(g['nunique'].max()),
        "rows_in_multi_row_pairs": int(g.loc[g['size']>1,'size'].sum()),
        "one_value_per_pair": bool((g['nunique']<=1).all()),
    }
    print(f"\n{keyname}: pairs={j[keyname].nunique()} distinct_ref={ref.nunique()} "
          f"pairs_multi_row={(g['size']>1).sum()} pairs_multi_ref={(g['nunique']>1).sum()} "
          f"one_value_per_pair={(g['nunique']<=1).all()}")
    print("  pair-size distribution:", {str(k):int(v) for k,v in g['size'].value_counts().sort_index().items()})

# also: how many distinct MOLECULES per fragment pair, and how many distinct EXPERIMENTAL per pair
g2 = j.groupby('fpair').agg(n_mol=('parent_structure_smiles','nunique'),
                            n_exp=('experimental_bde_kcal_mol','nunique'),
                            n_ref=('reference_estimate_bde_kcal_mol','nunique'),
                            n_quant=('quantum_estimate_bde_kcal_mol','nunique'),
                            n_learn=('learned_estimate_bde_kcal_mol','nunique'),
                            nrows=('observation_id','size'))
out["per_fragment_pair_uniqueness"] = {
   "pairs_with_multiple_molecules": int((g2.n_mol>1).sum()),
   "pairs_with_multiple_experimental_values": int((g2.n_exp>1).sum()),
   "pairs_with_multiple_reference_values": int((g2.n_ref>1).sum()),
   "pairs_with_multiple_quantum_values": int((g2.n_quant>1).sum()),
   "pairs_with_multiple_learned_values": int((g2.n_learn>1).sum()),
   "n_pairs": int(len(g2)), "n_pairs_multi_row": int((g2.nrows>1).sum()),
}
print("\nper-pair uniqueness:", out["per_fragment_pair_uniqueness"])

print("\n--- the 4 multi-row fragment pairs ---")
mr = g2[g2.nrows>1]
print(j[j.fpair.isin(mr.index)][['observation_id','parent_structure_smiles','fpair','experimental_bde_kcal_mol','reference_estimate_bde_kcal_mol','quantum_estimate_bde_kcal_mol','learned_estimate_bde_kcal_mol']].to_string())
out["multi_row_fragment_pair_examples"] = common.dataframe_records(
    j[j.fpair.isin(mr.index)][['observation_id','parent_structure_smiles','fpair','experimental_bde_kcal_mol','reference_estimate_bde_kcal_mol']])

# ---- 3b: reference channel error, matched (331) vs unmatched (212) ---------
def errstats(mask, chan):
    e = j.loc[mask, f"{chan}_estimate_bde_kcal_mol"].to_numpy(float) - j.loc[mask,'experimental_bde_kcal_mol'].to_numpy(float)
    return {"n": int(mask.sum()), "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e**2).mean())),
            "medae": float(np.median(np.abs(e))), "mean_signed": float(e.mean()), "sd_signed": float(e.std(ddof=1)),
            "max_abs": float(np.abs(e).max())}
out["reference_error_by_db_match"] = {
   "db_matched_331": errstats(j.in_db_row.to_numpy(), 'reference'),
   "db_unmatched_212": errstats(~j.in_db_row.to_numpy(), 'reference'),
   "all_543": errstats(np.ones(len(j),bool), 'reference')}
for k,v in out["reference_error_by_db_match"].items():
    print(f"reference {k:18s} n={v['n']:3d} MAE={v['mae']:8.4f} RMSE={v['rmse']:8.4f} MedAE={v['medae']:8.4f} mean={v['mean_signed']:7.4f} max|e|={v['max_abs']:.4f}")

# full channel MAE table for context
out["channel_error_by_db_match"] = {}
for chan in ['quantum','learned','reference']:
    out["channel_error_by_db_match"][chan] = {
       "db_matched_331": errstats(j.in_db_row.to_numpy(), chan),
       "db_unmatched_212": errstats(~j.in_db_row.to_numpy(), chan),
       "all_543": errstats(np.ones(len(j),bool), chan)}
print("\n--- channel MAE, matched vs unmatched ---")
for chan, d in out["channel_error_by_db_match"].items():
    print(f"  {chan:10s} matched MAE={d['db_matched_331']['mae']:.4f}  unmatched MAE={d['db_unmatched_212']['mae']:.4f}  all={d['all_543']['mae']:.4f}")

# is db_bde only available for matched -> quantum available for all 543?
out["channel_null_counts"] = {c: int(j[f"{c}_estimate_bde_kcal_mol"].isna().sum()) for c in ['quantum','learned','reference']}
print("\nnull counts per channel:", out["channel_null_counts"])

common.save_json(out, "ws5_part3_raw.json")
print("\nDONE part3")
