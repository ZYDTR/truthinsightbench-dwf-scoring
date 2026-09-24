"""WS1 supplement: is the database exhaustive within its own domain?

Compares, per parent molecule, the number of unique fragment pairs stored in the
database against the number of symmetry-unique bonds of the molecule (bond
orbits under canonical atom ranking on the H-added graph).  The database stores
one row per unique fragment pair, so an exhaustive enumeration would give
db_pairs == orbit count.
"""
import sys, json
sys.path.insert(0, '/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis')
import common, pandas as pd, numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

PROJ = '/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project'
db_pairs = pd.read_pickle(f'{PROJ}/output/cache/ws1_prep.pkl')['db_pairs']
fb = pd.read_pickle(f'{PROJ}/output/cache/ws1_fig.pkl')
b = fb['b']
nbp = db_pairs.groupby('molecule', sort=False).size()

def bond_orbits(smi):
    """Symmetry-unique bond count (heavy-H bonds included, H added explicitly)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    keys = set()
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        keys.add(frozenset((ranks[i], ranks[j])))
    return len(keys)

def summarize(mols, label):
    rec = []
    for m in mols:
        o = bond_orbits(m)
        if o is None or o == 0:
            continue
        rec.append((m, int(nbp.get(m, 0)), o, min(1.0, nbp.get(m, 0)/o)))
    r = pd.DataFrame(rec, columns=['molecule','db_pairs','unique_bonds','completeness'])
    s = {
        "subset": label, "n_molecules": int(len(r)),
        "mean_db_pairs": round(float(r.db_pairs.mean()), 3),
        "mean_unique_bonds": round(float(r.unique_bonds.mean()), 3),
        "mean_completeness": round(float(r.completeness.mean()), 4),
        "median_completeness": round(float(r.completeness.median()), 4),
        "pct_molecules_fully_enumerated": round(100*float((r.db_pairs >= r.unique_bonds).mean()), 4),
        "pct_molecules_under_half": round(100*float((r.completeness < 0.5).mean()), 4),
        "total_db_pairs": int(r.db_pairs.sum()), "total_unique_bonds": int(r.unique_bonds.sum()),
        "pooled_completeness": round(float(r.db_pairs.sum()/r.unique_bonds.sum()), 4),
        "completeness_histogram": {str(k): int(v) for k, v in
            pd.cut(r.completeness, [0,0.25,0.5,0.75,0.999,1.0]).value_counts().sort_index().items()},
    }
    return s, r

rng = np.random.default_rng(11)
all_mols = db_pairs.molecule.unique()
samp = list(rng.choice(all_mols, size=2000, replace=False))
s_db, r_db = summarize(samp, "random_2000_db_molecules")
print(json.dumps(s_db, indent=2))

ins = b[b.mol_in_db]
s_ins, r_ins = summarize(list(ins.parent_structure_smiles.unique()), "benchmark_parents_in_db")
print(json.dumps(s_ins, indent=2))

out = {"exhaustiveness_random_sample": s_db, "exhaustiveness_benchmark_parents_in_db": s_ins,
       "note": "completeness = db unique fragment pairs / symmetry-unique bonds (capped at 1.0)"}
# also record per-molecule detail for the benchmark parents that miss an exact pair
miss = fb['miss']
det = r_ins[r_ins.molecule.isin(miss.parent_structure_smiles)]
det = det.assign(molecule=det.molecule.astype(str))
out["benchmark_missing_pair_parent_completeness"] = common.dataframe_records(
    det[['molecule','db_pairs','unique_bonds','completeness']].head(60))
common.save_json(out, 'ws1_exhaustiveness_supplement.json')
print("\nbenchmark parents that miss an exact pair:")
print(det[['molecule','db_pairs','unique_bonds','completeness']].head(30).to_string(index=False))
pd.to_pickle({'exh_random': r_db, 'exh_ins': r_ins}, f'{PROJ}/output/cache/ws1_exh.pkl')
