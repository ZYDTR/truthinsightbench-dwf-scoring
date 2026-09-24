import sys; sys.path.insert(0,'/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis')
import json, os
import numpy as np, pandas as pd, common
out = {}

bench = common.load_benchmark()
if os.path.exists('/tmp/db.pkl'):
    db = pd.read_pickle('/tmp/db.pkl')
else:
    db = common.load_db(); db.to_pickle('/tmp/db.pkl')
print("db", db.shape, "bench", bench.shape)

# ---- join via merge (vectorised) -------------------------------------------
dbk = db[['molecule','fragment1','fragment2','bde','bond_type']].copy()
dbk['_dup'] = dbk.duplicated(subset=['molecule','fragment1','fragment2'], keep=False)
print("db rows on duplicated (mol,f1,f2) keys:", int(dbk._dup.sum()))
dbk = dbk.drop_duplicates(subset=['molecule','fragment1','fragment2'])
print("db unique keys:", len(dbk))
j = bench.merge(dbk, left_on=['parent_structure_smiles','fragment_a_smiles','fragment_b_smiles'],
                right_on=['molecule','fragment1','fragment2'], how='left', indicator=True)
j['in_db_row'] = j._merge == 'both'
out["n_in_db_row"] = int(j.in_db_row.sum()); out["n_not_in_db_row"] = int((~j.in_db_row).sum())
print("in_db_row:", out["n_in_db_row"], "/", len(j))

sub = j[j.in_db_row].copy()
d = sub.bde.to_numpy(float)

for chan, col in [("quantum","quantum_estimate_bde_kcal_mol"), ("learned","learned_estimate_bde_kcal_mol")]:
    p = sub[col].to_numpy(float)
    diff = d - p; ad = np.abs(diff)
    qlev = [0,.01,.05,.25,.5,.75,.95,.99,1.0]
    qs = np.quantile(diff, qlev)
    res = {"n": int(len(diff)),
        "mean_diff": float(diff.mean()), "sd_diff": float(diff.std(ddof=1)),
        "min_diff": float(diff.min()), "max_diff": float(diff.max()),
        "max_abs_diff": float(ad.max()), "mean_abs_diff": float(ad.mean()),
        "quantiles_diff": {f"q{int(round(p_*100)):02d}": float(q) for p_, q in zip(qlev, qs)},
        "quantiles_absdiff": {f"q{int(round(p_*100)):02d}": float(q) for p_, q in zip(qlev, np.quantile(ad, qlev))},
        "n_abs_le_0.01": int((ad <= 0.01).sum()), "frac_abs_le_0.01": float((ad <= 0.01).mean()),
        "n_abs_le_0.1":  int((ad <= 0.1).sum()),  "frac_abs_le_0.1":  float((ad <= 0.1).mean()),
        "n_abs_le_0.5":  int((ad <= 0.5).sum()),  "frac_abs_le_0.5":  float((ad <= 0.5).mean()),
        "n_abs_le_1.0":  int((ad <= 1.0).sum()),  "frac_abs_le_1.0":  float((ad <= 1.0).mean()),
        "n_exact_equality": int((d == p).sum()),
        "n_equal_after_round2_of_db": int((np.round(d,2) == p).sum()),
        "n_equal_after_round4_of_db": int((np.round(d,4) == p).sum()),
    }
    out[f"db_vs_{chan}"] = res
    print(f"\n=== db.bde vs {chan} (n={res['n']}) ===")
    print(f"  max|d|={res['max_abs_diff']:.10g}  mean d={res['mean_diff']:.10g}  sd={res['sd_diff']:.10g}  mean|d|={res['mean_abs_diff']:.10g}")
    print(f"  |d|<=0.01: {res['n_abs_le_0.01']}/{res['n']} ({res['frac_abs_le_0.01']:.6f})")
    print(f"  |d|<=0.1 : {res['n_abs_le_0.1']}/{res['n']} ({res['frac_abs_le_0.1']:.6f})")
    print(f"  |d|<=0.5 : {res['n_abs_le_0.5']}/{res['n']} ({res['frac_abs_le_0.5']:.6f})")
    print(f"  exact equality: {res['n_exact_equality']}/{res['n']}")
    print(f"  quantiles(diff): " + " ".join(f"{k}={v:.6g}" for k,v in res['quantiles_diff'].items()))

# ---- how many rows have an exact DB row but the DB has >1 row per key? -----
dupkeys = dbk[dbk.duplicated(subset=['molecule','fragment1','fragment2'], keep=False)]
out["db_dup_key_rows"] = int(len(dupkeys))
print("\ndb duplicated-key rows:", len(dupkeys))

# ---- does the quantum channel differ from db on ANY row where db exists? ----
qs = out["db_vs_quantum"]; ls = out["db_vs_learned"]
out["quantum_is_db_value_reproduced"] = bool(qs["frac_abs_le_0.01"] == 1.0)
out["learned_is_db_value_reproduced"] = bool(ls["frac_abs_le_0.01"] == 1.0)

# scale reference: what is the precision of db bde values?
out["db_bde_decimal_places_max"] = int(max(len(str(x).split('.')[-1]) for x in sub.bde.head(200)))
print("max decimal places in db bde (first 200):", out["db_bde_decimal_places_max"])

common.save_json(out, "ws5_part2_raw.json")
print("\nDONE part2")
