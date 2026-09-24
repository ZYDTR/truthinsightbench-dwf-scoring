"""WS1: fold the exhaustiveness supplement into the main artifact + inflation drivers."""
import sys, json
sys.path.insert(0, '/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis')
import common, pandas as pd, numpy as np

PROJ = '/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project'
art = json.load(open(f'{PROJ}/output/results/ws1_db_coverage.json'))
supp = json.load(open(f'{PROJ}/output/results/ws1_exhaustiveness_supplement.json'))
art["5_exhaustiveness_supplement"] = supp

# ---- who drives the row inflation ---------------------------------------
db = pd.read_pickle(f'{PROJ}/output/cache/db.pkl')
prep = pd.read_pickle(f'{PROJ}/output/cache/ws1_prep.pkl')
db_pairs = prep['db_pairs']
ps = prep['pair_size'].rename('rows_per_pair')
db_pairs = db_pairs.assign(rows_per_pair=db_pairs.set_index(['molecule','fragment1','fragment2']).index.map(ps))
infl = db_pairs.assign(extra=db_pairs.rows_per_pair - 1).groupby('bond_type').extra.sum()
extra_total = int(infl.sum())
art["2_inflation_drivers_by_class"] = [
    {"bond_class": k, "extra_rows": int(v), "share_pct": round(100*float(v)/extra_total, 3)}
    for k, v in infl.sort_values(ascending=False).items()]
# top molecules by extra rows
top = db_pairs.assign(extra=db_pairs.rows_per_pair-1).groupby('molecule').extra.sum().sort_values(ascending=False)
art["2_top_molecules_by_extra_rows"] = [
    {"molecule": str(m), "extra_rows": int(v), "db_rows": int(prep['pair_size'].loc[m].sum())}
    for m, v in top.head(15).items()]
# within a molecule: molecules whose C-H pairs multiply most
art["2_inflation_note"] = (
    "A (molecule,fragment1,fragment2) unit is stored once per equivalent bond instance: "
    "symmetry-equivalent atoms (e.g. the four H of methane) yield several rows with identical "
    "fragments and identical bde. C-H has 2.145 rows per pair because methyl/methylene H atoms "
    "are numerous and symmetry-equivalent; O-O/H-O/N-N have essentially one row per pair."
)
# HAC support overlap
hb = art["4_hac_db_vs_benchmark"]
art["4_hac_support_gap"] = {
    "db_hac_values": [r["hac"] for r in hb if r["db_unique_molecules"] > 0],
    "benchmark_hac_values": [r["hac"] for r in hb if r["benchmark_observations"] > 0],
    "hac_values_only_in_benchmark": [r["hac"] for r in hb
                                     if r["db_unique_molecules"] == 0 and r["benchmark_observations"] > 0],
    "benchmark_obs_above_db_max": int(sum(r["benchmark_observations"] for r in hb if r["benchmark_hac_gt_db_max"] > 0)),
    "db_molecules_above_benchmark_max": 0,
}
common.save_json(art, 'ws1_db_coverage.json')
print(json.dumps(art["2_inflation_drivers_by_class"], indent=2))
print("top molecules by extra rows:")
print(pd.DataFrame(art["2_top_molecules_by_extra_rows"]).to_string(index=False))
print("hac gap:", json.dumps(art["4_hac_support_gap"], indent=2))
print("artifact keys:", list(art.keys()))
