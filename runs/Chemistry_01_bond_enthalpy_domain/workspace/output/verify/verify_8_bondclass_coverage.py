"""Independent adversarial recomputation of claim 8: bond-class coverage.

Own implementation - does NOT import output/analysis/common.py.
Reads only the raw evidence files.
"""
import gzip, json, random
from collections import Counter

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

PROJ = "/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project"
TSV = f"{PROJ}/data/normalized_views/experimental_benchmark.tsv"
DB = f"{PROJ}/data/source_data/bde_db_full.csv.gz"

ELEM_ORDER = ["C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"]

def radical_element(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    rad = [a for a in m.GetAtoms() if a.GetNumRadicalElectrons() > 0]
    if len(rad) == 1:
        return rad[0].GetSymbol()
    if len(rad) > 1:
        return "+".join(sorted(a.GetSymbol() for a in rad))
    return "NORAD:" + m.GetAtomWithIdx(0).GetSymbol()

def bclass(a, b):
    ea, eb = radical_element(a), radical_element(b)
    if ea is None or eb is None:
        return None
    def key(e):
        e0 = e.split("+")[0]
        return ELEM_ORDER.index(e0) if e0 in ELEM_ORDER else 99
    return "-".join(sorted([ea, eb], key=key))

# ------------------------------------------------------------------ benchmark
bench = pd.read_csv(TSV, sep="\t")
print("benchmark rows:", len(bench))
print("benchmark cols:", list(bench.columns))
print("duplicate observation_id:", int(bench.observation_id.duplicated().sum()))
print("unique molecules:", bench.parent_structure_smiles.nunique())

bench["my_bond_class"] = [bclass(a, b) for a, b in zip(bench.fragment_a_smiles, bench.fragment_b_smiles)]
bc = Counter(bench["my_bond_class"])
print("\nBENCHMARK bond classes (543 raw observation rows, my labeler):")
for k, v in bc.most_common():
    print(f"  {k}: {v}")
print("sum:", sum(bc.values()), "| n distinct:", len(bc), "| nulls:", int(bench.my_bond_class.isna().sum()))

# alternative analysis units
bench["pair_sorted"] = [tuple(sorted(p)) for p in zip(bench.fragment_a_smiles, bench.fragment_b_smiles)]
print("\nunit checks:")
print("  unique (mol,fragA,fragB) rows:", len(bench[["parent_structure_smiles","fragment_a_smiles","fragment_b_smiles"]].drop_duplicates()))
print("  unique unordered fragment pairs:", bench.pair_sorted.nunique())
print("  unique (mol, unordered pair):", bench.assign(k=list(zip(bench.parent_structure_smiles, bench.pair_sorted))).k.nunique())
print("  per-unit class counts (unique mol,fragA,fragB):")
for k, v in Counter(bench.drop_duplicates(["parent_structure_smiles","fragment_a_smiles","fragment_b_smiles"]).my_bond_class).most_common():
    print(f"    {k}: {v}")
print("  per-unit class counts (unique unordered fragment pair):")
for k, v in Counter(bench.drop_duplicates("pair_sorted").my_bond_class).most_common():
    print(f"    {k}: {v}")

# ------------------------------------------------------------------ database
db = pd.read_csv(DB)
print("\nDB rows:", len(db), "| cols:", list(db.columns))
print("DB duplicate rid:", int(db.rid.duplicated().sum()))
db_counts = db["bond_type"].value_counts()
print("\nDB bond_type value counts (raw rows):")
for k, v in db_counts.items():
    print(f"  {k}: {v}")
print("sum:", int(db_counts.sum()), "| n distinct bond_type:", int(db.bond_type.nunique()),
      "| nulls:", int(db.bond_type.isna().sum()))

# element-set sanity of the taxonomy strings
print("\nDB bond_type element-set distribution:", Counter(
    tuple(sorted(t.split('-'))) for t in db.bond_type.unique()))

# labeler validation on a random sample of DB rows (own code vs bond_type column)
rng = random.Random(20260919)
idx = rng.sample(range(len(db)), 20000)
mismatch = 0
mism = Counter()
cache = {}
for i in idx:
    f1 = db.fragment1.iat[i]; f2 = db.fragment2.iat[i]; bt = db.bond_type.iat[i]
    c = bclass(f1, f2)
    if c != bt:
        mismatch += 1
        mism[(c, bt)] += 1
print(f"\nDB labeler validation on random 20000 rows: mismatch={mismatch}")
print("  mismatch pairs (mine, db):", mism.most_common(10))

# DB classes restricted to rows that are actual benchmark rows
keys = list(zip(bench.parent_structure_smiles, bench.fragment_a_smiles, bench.fragment_b_smiles))
db_lookup = dict(zip(zip(db.molecule, db.fragment1, db.fragment2), db.bond_type))
matched_keys = [k for k in keys if k in db_lookup]
print("\nbenchmark rows that are exact DB (molecule,frag1,frag2) rows:", len(matched_keys), "/", len(keys))
print("DB bond_type of matched benchmark rows:", Counter(db_lookup[k] for k in matched_keys))
print("DB taxonomy classes absent from benchmark:", sorted(set(db.bond_type.unique()) - set(db_lookup[k] for k in matched_keys)))

# molecule-level DB presence
db_mols = set(db.molecule)
print("benchmark rows whose parent molecule appears in DB:", int(bench.parent_structure_smiles.isin(db_mols).sum()))

summary = {
    "benchmark_rows": int(len(bench)),
    "benchmark_distinct_classes": len(bc),
    "benchmark_class_counts": {str(k): int(v) for k, v in bc.items()},
    "benchmark_null_classes": int(bench.my_bond_class.isna().sum()),
    "benchmark_unique_molecules": int(bench.parent_structure_smiles.nunique()),
    "benchmark_unique_molffragfrag": int(len(bench[["parent_structure_smiles","fragment_a_smiles","fragment_b_smiles"]].drop_duplicates())),
    "benchmark_unique_unordered_pairs": int(bench.pair_sorted.nunique()),
    "db_rows": int(len(db)),
    "db_distinct_classes": int(db.bond_type.nunique()),
    "db_class_counts": {str(k): int(v) for k, v in db_counts.items()},
    "db_labeler_mismatch_20k_sample": int(mismatch),
    "benchmark_rows_in_db_exact": int(len(matched_keys)),
    "db_classes_absent_from_benchmark": sorted(set(db.bond_type.unique()) - set(db_lookup[k] for k in matched_keys)),
}
with open(f"{PROJ}/output/_verify8_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)
print("\nSUMMARY:", json.dumps(summary, indent=2))
