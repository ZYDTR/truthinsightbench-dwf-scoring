"""WS1 prep: expensive database group-bys, cached to output/cache/ws1_prep.pkl."""
import sys, json, time
sys.path.insert(0, '/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis')
import common, pandas as pd, numpy as np

PROJ = '/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project'
T0 = time.time()
def stamp(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)

db = pd.read_pickle(f'{PROJ}/output/cache/db.pkl')
desc = pd.read_pickle(f'{PROJ}/output/cache/db_mol_desc.pkl')
pair_cols = ['molecule', 'fragment1', 'fragment2']
db_pairs = db.drop_duplicates(subset=pair_cols).copy()
stamp("drop_duplicates pairs")

bt_nuniq = db.groupby(pair_cols, sort=False).bond_type.nunique()
bde_nuniq = db.groupby(pair_cols, sort=False).bde.nunique()
stamp("groupby nunique")

hac_of = dict(zip(desc.molecule, desc.hac))
el_of = dict(zip(desc.molecule, desc.elements))
db_pairs['hac'] = db_pairs.molecule.map(hac_of)
db_pairs['parent_elements'] = db_pairs.molecule.map(el_of)

inventory = {
    "raw_rows": int(len(db)),
    "unique_molecules": int(db.molecule.nunique()),
    "unique_fragment_pairs_mol_f1_f2": int(len(db_pairs)),
    "unique_global_fragment_pairs_f1_f2": int(db.drop_duplicates(subset=['fragment1','fragment2']).shape[0]),
    "unique_bde_values": int(db.bde.nunique()),
    "unique_bond_types": int(db.bond_type.nunique()),
    "unique_bond_index_values": int(db.bond_index.nunique()),
    "row_vs_pair_redundancy_factor": round(len(db)/len(db_pairs), 4),
    "rows_per_molecule": round(len(db)/db.molecule.nunique(), 4),
    "pair_units_with_bond_type_varying": int((bt_nuniq > 1).sum()),
    "pair_units_with_bde_varying": int((bde_nuniq > 1).sum()),
    "extra_rows_beyond_one_per_pair": int(len(db)-len(db_pairs)),
    "distinct_fragment_strings": int(len(pd.unique(pd.concat([db.fragment1, db.fragment2], ignore_index=True)))),
    "duplicate_tuple_group_size_histogram": {int(k): int(v) for k, v in
        db.groupby(pair_cols, sort=False).size().value_counts().sort_index().items()},
}
stamp("inventory")
pair_size = db.groupby(pair_cols, sort=False).size()
pd.to_pickle({'inventory': inventory, 'db_pairs': db_pairs, 'pair_size': pair_size,
              'db_hac_max': int(desc.hac.max()), 'desc': desc},
             f'{PROJ}/output/cache/ws1_prep.pkl')
print(json.dumps(inventory, indent=2))
stamp("saved")
