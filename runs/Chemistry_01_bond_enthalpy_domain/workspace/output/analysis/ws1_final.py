"""WS1 final: census, structural domain, coverage boundary, exhaustiveness.

Reads the cached expensive intermediates from ws1_prep.py.  Every statistic is
reported on an explicitly named analysis unit.
"""
import sys, json, time
sys.path.insert(0, '/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis')
import common, pandas as pd, numpy as np

PROJ = '/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project'
T0 = time.time()
def stamp(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)

prep = pd.read_pickle(f'{PROJ}/output/cache/ws1_prep.pkl')
db = pd.read_pickle(f'{PROJ}/output/cache/db.pkl')
desc = prep['desc']
db_pairs = prep['db_pairs']
inventory = prep['inventory']
db_hac_max = prep['db_hac_max']
bench = common.build_analysis_table(verbose=False)
stamp("loaded")

out = {}

# ====================================================== 2. BOND-CLASS CENSUS ==
rows_by_class = db.bond_type.value_counts()
pairs_by_class = db_pairs.bond_type.value_counts()
bde_pairs = db_pairs.bde.to_numpy(float); cls_pairs = db_pairs.bond_type.to_numpy()
bde_rows = db.bde.to_numpy(float); cls_rows = db.bond_type.to_numpy()
mol_pairs = db_pairs.molecule.to_numpy(); mol_rows = db.molecule.to_numpy()

census = []
for cls in rows_by_class.index:
    r = int(rows_by_class[cls]); p = int(pairs_by_class.get(cls, 0))
    pm = cls_pairs == cls; rm = cls_rows == cls; pv = bde_pairs[pm]
    census.append({
        "bond_class": cls, "raw_rows": r, "unique_fragment_pairs": p,
        "n_unique_molecules_with_class": int(pd.unique(mol_pairs[pm]).size),
        "raw_rows_pct": round(100*r/len(db), 4),
        "unique_pairs_pct": round(100*p/len(db_pairs), 4),
        "redundancy_rows_per_pair": round(r/p, 4) if p else None,
        "mean_rows_per_molecule": round(r/int(pd.unique(mol_rows[rm]).size), 3),
        "mean_bde_unique_pairs": round(float(pv.mean()), 4),
        "sd_bde_unique_pairs": round(float(pv.std(ddof=1)), 4),
        "min_bde_unique_pairs": round(float(pv.min()), 4),
        "median_bde_unique_pairs": round(float(np.median(pv)), 4),
        "max_bde_unique_pairs": round(float(pv.max()), 4),
        "mean_bde_row_weighted": round(float(bde_rows[rm].mean()), 4),
    })
census = sorted(census, key=lambda d: -d["raw_rows"])
tot_extra = sum(c["raw_rows"]-c["unique_fragment_pairs"] for c in census)
for c in census:
    c["share_of_row_inflation_pct"] = round(100*(c["raw_rows"]-c["unique_fragment_pairs"])/tot_extra, 3)
    c["mean_bde_row_minus_pair"] = round(c["mean_bde_row_weighted"]-c["mean_bde_unique_pairs"], 4)
print("== 2. BOND-CLASS CENSUS ==")
print(pd.DataFrame(census)[['bond_class','raw_rows','unique_fragment_pairs','redundancy_rows_per_pair',
      'share_of_row_inflation_pct','n_unique_molecules_with_class','mean_bde_unique_pairs',
      'sd_bde_unique_pairs','min_bde_unique_pairs','median_bde_unique_pairs','max_bde_unique_pairs']].to_string(index=False))
print("total rows beyond one-per-pair:", tot_extra, "== raw-pairs:", len(db)-len(db_pairs))
stamp("census")

# labeler validation on a deterministic random sample of unique pairs
rng = np.random.default_rng(20260919)
sidx = rng.choice(len(db_pairs), size=4000, replace=False)
samp = db_pairs.iloc[sidx]
lab = [common.bond_class(a, b) for a, b in zip(samp.fragment1, samp.fragment2)]
samp = samp.assign(labeler_class=lab, labeler_ok=[l == t for l, t in zip(lab, samp.bond_type)])
labeler = {
    "n_sampled_unique_pairs": int(len(samp)),
    "n_labeler_unparsable": int(samp.labeler_class.isna().sum()),
    "labeler_agreement_with_db_bond_type": int(samp.labeler_ok.sum()),
    "labeler_agreement_pct": round(100*float(samp.labeler_ok.mean()), 4),
    "benchmark_labeler_agreement": int(bench.labeler_ok.sum()),
    "benchmark_labeler_compared": int(bench.labeler_ok.notna().sum()),
    "disagreement_examples": common.dataframe_records(
        samp.loc[~samp.labeler_ok, ['molecule','fragment1','fragment2','bond_type','labeler_class']].head(20)),
}
print("\nlabeler:", {k:v for k,v in labeler.items() if k!='disagreement_examples'})
stamp("labeler")

# ================================================== 3. STRUCTURAL DOMAIN ======
db_elements = sorted(set('|'.join(desc.elements.dropna()).split('|')))
hac_counts = desc.hac.value_counts().sort_index()
hac_pairs_counts = db_pairs.hac.value_counts().sort_index()
domain = {
    "database_element_set": db_elements, "database_n_elements": len(db_elements),
    "database_hac_min": int(desc.hac.min()), "database_hac_max": int(desc.hac.max()),
    "database_hac_mean_unique_molecules": round(float(desc.hac.mean()), 4),
    "database_hac_median_unique_molecules": float(desc.hac.median()),
    "database_hac_sd_unique_molecules": round(float(desc.hac.std(ddof=1)), 4),
    "database_hac_mean_unique_pairs": round(float(db_pairs.hac.mean()), 4),
    "database_hac_median_unique_pairs": float(db_pairs.hac.median()),
    "database_hac_mean_weighted_by_raw_rows": round(
        float(db.molecule.map(dict(zip(desc.molecule, desc.hac))).mean()), 4),
    "unparsable_db_molecules": int(desc.hac.isna().sum()),
    "hac_definition": "heavy-atom count of parent SMILES, RDKit GetNumHeavyAtoms",
    "pct_molecules_hac_ge_8": round(100*float((desc.hac >= 8).mean()), 4),
    "pct_molecules_hac_le_6": round(100*float((desc.hac <= 6).mean()), 4),
}
print("\n== 3. STRUCTURAL DOMAIN =="); print(json.dumps(domain, indent=2))

hac_table = pd.DataFrame({
    "hac": hac_counts.index.astype(int),
    "n_unique_molecules": hac_counts.values.astype(int),
    "n_unique_fragment_pairs": hac_pairs_counts.reindex(hac_counts.index).fillna(0).astype(int).values,
})
hac_table["pct_unique_molecules"] = (100*hac_table.n_unique_molecules/len(desc)).round(4)
hac_table["cum_pct_unique_molecules"] = hac_table.pct_unique_molecules.cumsum().round(4)
print(hac_table.to_string(index=False))
molecules_per_element_set = desc.elements.value_counts().to_dict()
print("molecules per element set:", molecules_per_element_set)

ols = [{"bond_class": c["bond_class"], "unit": "unique_fragment_pair", "n": c["unique_fragment_pairs"],
        "mean": c["mean_bde_unique_pairs"], "sd": c["sd_bde_unique_pairs"], "min": c["min_bde_unique_pairs"],
        "median": c["median_bde_unique_pairs"], "max": c["max_bde_unique_pairs"],
        "mean_row_weighted": c["mean_bde_row_weighted"],
        "mean_minus_row_weighted": round(c["mean_bde_unique_pairs"]-c["mean_bde_row_weighted"], 4)} for c in census]
print("\n== 3b. OLS (ordinary, unweighted) BDE summary per class on unique-pair units ==")
print(pd.DataFrame(ols).to_string(index=False))
stamp("domain")

# ================================================== 4. COVERAGE BOUNDARY ======
db_pair_set = set(zip(db.molecule, db.fragment1, db.fragment2))
db_pair_set_rev = set(zip(db.molecule, db.fragment2, db.fragment1))
db_mols = set(db.molecule)
db_el_set = set(db_elements)

b = bench.copy()
# HAC + element set computed for EVERY benchmark parent (not only DB residents)
b_hac = {m: common.heavy_atom_count(m) for m in pd.unique(b.parent_structure_smiles)}
b_el = {m: frozenset(common.element_set(m)) for m in pd.unique(b.parent_structure_smiles)}
b['hac_parent'] = b.parent_structure_smiles.map(b_hac)
b['parent_elements'] = b.parent_structure_smiles.map(lambda m: "|".join(sorted(b_el[m])))
b['n_foreign_elements'] = b.parent_structure_smiles.map(lambda m: len(b_el[m]-db_el_set))
b['foreign_elements'] = b.parent_structure_smiles.map(lambda m: sorted(b_el[m]-db_el_set))
b_keys = list(zip(b.parent_structure_smiles, b.fragment_a_smiles, b.fragment_b_smiles))
b['in_db_exact'] = [k in db_pair_set for k in b_keys]
b['in_db_exact_reverse'] = [k in db_pair_set_rev for k in b_keys]
b['mol_in_db'] = b.parent_structure_smiles.isin(db_mols)
b['hac_gt_db_max'] = b.hac_parent > db_hac_max
n_b = len(b)
stamp("benchmark descriptors")

cov = {
    "benchmark_observations": int(n_b),
    "benchmark_unique_molecules": int(b.parent_structure_smiles.nunique()),
    "benchmark_unique_fragment_pairs": int(b.drop_duplicates(subset=['parent_structure_smiles','fragment_a_smiles','fragment_b_smiles']).shape[0]),
    "a_exact_fragment_pair_in_db": int(b.in_db_exact.sum()),
    "a_exact_fragment_pair_in_db_pct": round(100*float(b.in_db_exact.mean()), 4),
    "a_exact_pair_any_fragment_order": int((b.in_db_exact | b.in_db_exact_reverse).sum()),
    "b_parent_molecule_in_db": int(b.mol_in_db.sum()),
    "b_parent_molecule_in_db_pct": round(100*float(b.mol_in_db.mean()), 4),
    "b_parent_molecule_absent": int((~b.mol_in_db).sum()),
    "c_parent_hac_beyond_db_max": int(b.hac_gt_db_max.sum()),
    "c_parent_hac_beyond_db_max_pct": round(100*float(b.hac_gt_db_max.mean()), 4),
    "c_parent_hac_at_or_below_db_max": int((~b.hac_gt_db_max).sum()),
    "parent_hac_missing_unparsable": int(b.hac_parent.isna().sum()),
    "d_parent_with_elements_never_in_db": int((b.n_foreign_elements > 0).sum()),
    "d_parent_with_elements_never_in_db_pct": round(100*float((b.n_foreign_elements > 0).mean()), 4),
    "d_foreign_element_counts": {e: int(b.parent_structure_smiles.map(lambda m: e in b_el[m]).sum())
                                 for e in sorted(set(e for m in b_el for e in b_el[m]-db_el_set))},
    "db_hac_max": int(db_hac_max),
    "hac_cutoff_defining_boundary": int(db_hac_max),
    "benchmark_hac_min": int(b.hac_parent.min()), "benchmark_hac_max": int(b.hac_parent.max()),
    "benchmark_hac_mean": round(float(b.hac_parent.mean()), 4),
    "benchmark_hac_median": float(b.hac_parent.median()),
    "benchmark_hac_sd": round(float(b.hac_parent.std(ddof=1)), 4),
    "inside_db_by_parent": int(b.mol_in_db.sum()),
    "inside_db_by_exact_pair": int(b.in_db_exact.sum()),
    "outside_db_by_any_criterion": int((~b.mol_in_db).sum()),
    "outside_by_hac_only": int(((~b.mol_in_db) & (b.n_foreign_elements == 0)).sum()),
}
print("\n== 4. COVERAGE BOUNDARY =="); print(json.dumps(cov, indent=2))

b_hac_counts = b.hac_parent.value_counts().sort_index()
inside = b[b.mol_in_db]; outside = b[~b.mol_in_db]
all_hac = sorted(set(list(hac_counts.index.astype(int)) + list(b_hac_counts.index.astype(int))))
hac_cmp = pd.DataFrame({
    "hac": all_hac,
    "db_unique_molecules": hac_counts.reindex(all_hac).fillna(0).astype(int).values,
    "benchmark_observations": b_hac_counts.reindex(all_hac).fillna(0).astype(int).values,
})
hac_cmp["benchmark_inside_db"] = inside.hac_parent.value_counts().reindex(all_hac).fillna(0).astype(int).values
hac_cmp["benchmark_outside_db"] = outside.hac_parent.value_counts().reindex(all_hac).fillna(0).astype(int).values
hac_cmp["benchmark_hac_gt_db_max"] = [
    int((b.hac_parent == h).sum()) if h > db_hac_max else 0 for h in all_hac]
print("\nHAC: DB molecules vs benchmark observations")
print(hac_cmp.to_string(index=False))
print("\nmean HAC: DB molecules=%.3f  benchmark obs=%.3f  benchmark parents in DB=%.3f  outside=%.3f" % (
    desc.hac.mean(), b.hac_parent.mean(), inside.hac_parent.mean(), outside.hac_parent.mean()))
print("\nbenchmark parents absent from DB:")
print(outside[['observation_id','parent_structure_smiles','bond_class','hac_parent',
               'experimental_bde_kcal_mol']].head(20).to_string(index=False))
print("\nbenchmark parents with foreign elements:")
fo = b.loc[b.n_foreign_elements > 0, ['observation_id','parent_structure_smiles','foreign_elements','hac_parent','bond_class']]
print(fo.head(20).to_string(index=False) if len(fo) else "  (none)")
stamp("coverage")

# ============================================ 5. EXHAUSTIVENESS WITHIN DOMAIN ==
ex = {
    "benchmark_parent_in_db": int(len(inside)),
    "of_those_exact_fragment_pair_present": int(inside.in_db_exact.sum()),
    "of_those_exact_fragment_pair_absent": int((~inside.in_db_exact).sum()),
    "pct_exact_when_parent_present": round(100*float(inside.in_db_exact.mean()), 4),
    "pct_pair_absent_when_parent_present": round(100*float((~inside.in_db_exact).mean()), 4),
    "parent_present_pair_absent_pct_of_all_benchmark": round(100*float((b.mol_in_db & ~b.in_db_exact).sum())/n_b, 4),
    "combined_exact_coverage_of_benchmark_pct": round(100*float(b.in_db_exact.sum())/n_b, 4),
}
print("\n== 5. EXHAUSTIVENESS WITHIN DOMAIN =="); print(json.dumps(ex, indent=2))
miss = inside[~inside.in_db_exact]
npairs_of_mol = db_pairs.groupby('molecule', sort=False).size()
miss = miss.assign(db_pairs_for_parent=miss.parent_structure_smiles.map(npairs_of_mol))
print("\nmissing-pair examples:")
print(miss[['observation_id','parent_structure_smiles','fragment_a_smiles','fragment_b_smiles','bond_class',
            'hac_parent','db_pairs_for_parent','experimental_bde_kcal_mol']].head(20).to_string(index=False))
print("\ndb fragment-pair count of those parents:"); print(miss.db_pairs_for_parent.value_counts().sort_index().to_string())
print("\nHAC of missing-pair rows:"); print(miss.hac_parent.value_counts().sort_index().to_string())
print("\nbond class of missing-pair rows:"); print(miss.bond_class.value_counts().to_string())
stamp("exhaustiveness")

# how many benchmark rows have HAC outside the DB HAC support (11+)?
hac_out = b.loc[b.hac_gt_db_max]
print("\nbenchmark rows with parent HAC > db max:", len(hac_out))
if len(hac_out): print(hac_out[['observation_id','parent_structure_smiles','hac_parent']].to_string(index=False))
stamp("hac-out")

# ================================================== write artifact ===========
artifact = {
    "workstream": "WS1",
    "title": "Database inventory, structural domain and coverage boundary",
    "generated_by": "output/analysis/ws1_prep.py + output/analysis/ws1_final.py",
    "units": {
        "raw_row": "one row of bde_db_full.csv.gz (one bond instance in one molecule)",
        "unique_molecule": "distinct parent molecule SMILES",
        "unique_fragment_pair": "distinct (molecule, fragment1, fragment2); bond_type and bde are constant inside a unit",
        "benchmark_observation": "one of the 543 experimental rows (error-analysis unit)",
    },
    "1_inventory": inventory,
    "1_inventory_table": [
        {"unit": "raw rows", "value": int(len(db))},
        {"unit": "unique molecules", "value": int(db.molecule.nunique())},
        {"unit": "unique (molecule,fragment1,fragment2) fragment pairs", "value": int(len(db_pairs))},
        {"unit": "unique (fragment1,fragment2) pairs, molecule-agnostic", "value": inventory["unique_global_fragment_pairs_f1_f2"]},
        {"unit": "unique bde values", "value": int(db.bde.nunique())},
    ],
    "1_duplicate_tuple_group_size_histogram": inventory["duplicate_tuple_group_size_histogram"],
    "2_bond_class_census": census,
    "2_row_inflation_total": int(tot_extra),
    "2_census_csv": "\n".join(pd.DataFrame(census).to_csv(index=False).splitlines()),
    "2_labeler_validation": labeler,
    "3_structural_domain": domain,
    "3_hac_distribution": common.dataframe_records(hac_table),
    "3_molecules_per_element_set": [{"element_set": k, "n_unique_molecules": int(v)}
                                    for k, v in molecules_per_element_set.items()],
    "3_ols_bde_summary_unique_pairs": ols,
    "4_coverage_boundary": cov,
    "4_hac_db_vs_benchmark": common.dataframe_records(hac_cmp),
    "4_outside_examples": common.dataframe_records(
        outside[['observation_id','parent_structure_smiles','bond_class','hac_parent','experimental_bde_kcal_mol']].head(60)),
    "4_foreign_element_examples": common.dataframe_records(fo.head(60)),
    "5_within_domain_exhaustiveness": ex,
    "5_missing_pair_examples": common.dataframe_records(
        miss[['observation_id','parent_structure_smiles','fragment_a_smiles','fragment_b_smiles','bond_class',
              'hac_parent','db_pairs_for_parent','experimental_bde_kcal_mol']].head(60)),
    "5_missing_pair_db_pairs_per_parent": [
        {"db_fragment_pairs_for_parent": int(k), "n_benchmark_observations": int(v)}
        for k, v in miss.db_pairs_for_parent.value_counts().sort_index().items()],
    "5_missing_pair_hac_counts": [{"hac": int(k), "n": int(v)} for k, v in miss.hac_parent.value_counts().sort_index().items()],
    "5_missing_pair_bond_class_counts": [{"bond_class": k, "n": int(v)} for k, v in miss.bond_class.value_counts().items()],
    "benchmark_bond_class_counts": [{"bond_class": k, "n_observations": int(v)}
                                    for k, v in b.bond_class.value_counts().items()],
    "benchmark_parent_elements": sorted(set(e for m in b_el for e in b_el[m])),
    "db_bond_class_row_counts": [{"bond_class": k, "raw_rows": int(v)} for k, v in rows_by_class.items()],
}
common.save_json(artifact, 'ws1_db_coverage.json')
pd.to_pickle({'hac_table': hac_table, 'census': census, 'b': b, 'hac_cmp': hac_cmp, 'miss': miss},
             f'{PROJ}/output/cache/ws1_fig.pkl')
print("\nDONE"); stamp("saved")
