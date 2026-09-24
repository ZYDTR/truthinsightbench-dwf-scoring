import sys; sys.path.insert(0, '/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis')
import json
import numpy as np
import pandas as pd
import common

out = {}

# ---- hashes ----------------------------------------------------------------
hashes = {}
manifest = json.load(open('/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/data_manifest.json'))
mf = {f["path"]: f for f in manifest["files"]}
for name, p, mkey in [("experimental_benchmark.tsv", common.TSV, "normalized_views/experimental_benchmark.tsv"),
                ("bde_db_full.csv.gz", common.DB_GZ, "source_data/bde_db_full.csv.gz"),
                ("experimental_source_tables.xlsx", common.XLSX, "source_data/experimental_source_tables.xlsx")]:
    h = common.sha256(p)
    hashes[name] = {"sha256": h, "bytes": p.stat().st_size, "path": str(p),
                    "manifest_sha256": mf[mkey]["sha256"],
                    "sha256_matches_manifest": h == mf[mkey]["sha256"],
                    "manifest_bytes": mf[mkey]["bytes"],
                    "bytes_match_manifest": p.stat().st_size == mf[mkey]["bytes"]}
out["file_hashes"] = hashes
for k, v in hashes.items():
    print(f"{k}\n  sha256 {v['sha256']}\n  matches manifest: {v['sha256_matches_manifest']} bytes_ok={v['bytes_match_manifest']}")

bench = common.load_benchmark(); xl = common.load_xlsx(); db = common.load_db()
print("\nshapes: bench", bench.shape, "xlsx", xl.shape, "db", db.shape)
out["shapes"] = {"benchmark_tsv": list(bench.shape), "xlsx_sheet001": list(xl.shape), "db": list(db.shape)}
out["tsv_columns"] = list(bench.columns)
out["xlsx_columns"] = [str(c) for c in xl.columns]
out["db_columns"] = list(db.columns)

# ---- row-for-row comparison of the 7 shared columns ------------------------
colmap = {
    "parent_structure_smiles": "Molecule",
    "fragment_a_smiles": "Radical 1",
    "fragment_b_smiles": "Radical 2",
    "experimental_bde_kcal_mol": "observed_value",
    "quantum_estimate_bde_kcal_mol": "estimate_quantum",
    "learned_estimate_bde_kcal_mol": "estimate_learned",
    "reference_estimate_bde_kcal_mol": "estimate_reference",
}
match_report = {}
for tcol, xcol in colmap.items():
    a = bench[tcol].to_numpy(); b = xl[xcol].to_numpy()
    n = min(len(a), len(b))
    if tcol.endswith("kcal_mol"):
        av = a[:n].astype(float); bv = b[:n].astype(float)
        eq = (av == bv)
        num = {"max_abs_diff": float(np.max(np.abs(av - bv))),
               "n_within_1e-9": int((np.abs(av - bv) <= 1e-9).sum()),
               "n_within_1e-6": int((np.abs(av - bv) <= 1e-6).sum())}
    else:
        eq = np.array([str(x) == str(y) for x, y in zip(a[:n], b[:n])]); num = {}
    exact = int(eq.sum())
    match_report[tcol] = {"xlsx_col": xcol, "n_compared": int(n), "n_exact_match": exact,
                          "frac_exact_match": exact / n if n else None,
                          "first_mismatch_row_index": (int(np.argmin(eq)) if exact < n else None), **num}
    print(f"{tcol:38s} <- {xcol:22s} exact {exact}/{n} maxdiff={num.get('max_abs_diff')}")
out["cross_file_column_match"] = match_report
out["n_shared_columns_tsv_vs_xlsx"] = len(colmap)
out["cross_file_all_shared_columns_identical"] = all(v["n_exact_match"] == v["n_compared"] for v in match_report.values())
out["xlsx_extra_column"] = [c for c in xl.columns if c not in colmap.values()]

# ---- row ORDER identical (keyed on molecule+fragments) ---------------------
key_tsv = list(zip(bench.parent_structure_smiles, bench.fragment_a_smiles, bench.fragment_b_smiles))
key_xl = list(zip(xl["Molecule"], xl["Radical 1"], xl["Radical 2"]))
order_same = int(sum(1 for a, b in zip(key_tsv, key_xl) if a == b))
out["row_order_identical_count"] = order_same
out["row_order_identical_all"] = order_same == len(bench) == len(xl)
# as unordered multiset
out["row_multiset_identical"] = sorted(key_tsv) == sorted(key_xl)
print("\nrow ORDER identical:", order_same, "/", len(bench), " multiset equal:", out["row_multiset_identical"])

# ---- train_flag -------------------------------------------------------------
tf = xl["in_reference_training_set"]
print("\nin_reference_training_set dtype:", tf.dtype, "unique:", sorted(pd.unique(tf.astype(str))))
out["train_flag"] = {"dtype": str(tf.dtype), "n_true": int(tf.astype(bool).sum()),
                     "n_false": int((~tf.astype(bool)).sum()),
                     "value_counts": {str(k): int(v) for k, v in tf.astype(str).value_counts().items()}}

# ---- exact database-row correspondence --------------------------------------
db_keys = set(zip(db.molecule, db.fragment1, db.fragment2))
in_db = np.array([k in db_keys for k in key_tsv])
out["n_in_db_exact_row"] = int(in_db.sum()); out["n_not_in_db_exact_row"] = int((~in_db).sum())
print("\nbenchmark rows with exact DB row:", int(in_db.sum()), "/", len(bench))

tfbool = tf.to_numpy().astype(bool)
ct = pd.crosstab(pd.Series(in_db, name="in_db_exact_row"), pd.Series(tfbool, name="in_reference_training_set"))
print("\ncrosstab in_db_exact_row x train_flag:\n", ct.to_string())
out["train_flag_vs_in_db_crosstab"] = json.loads(ct.to_json())
tp = int((in_db & tfbool).sum()); fp = int((~in_db & tfbool).sum())
fn = int((in_db & ~tfbool).sum()); tn = int((~in_db & ~tfbool).sum())
out["train_flag_vs_in_db"] = {"both_true": tp, "train_only_not_db": fp, "db_only_not_train": fn, "neither": tn,
    "n_agreement": tp + tn, "agreement_fraction": (tp + tn) / len(in_db),
    "contradictions": fp + fn,
    "n_true_total": int(tfbool.sum()), "n_in_db_total": int(in_db.sum())}
print("\ncorrespondence:", out["train_flag_vs_in_db"])

# also: parent molecule present anywhere in DB
_dbset = set(db.molecule)
mol_in_db = np.array([m in _dbset for m in bench.parent_structure_smiles])
out["n_parent_molecule_in_db"] = int(mol_in_db.sum())
print("parent molecules present anywhere in DB:", int(mol_in_db.sum()), "/", len(bench))

common.save_json(out, "ws5_part1_raw.json")
print("\nDONE part1")
