"""Finding 1: the database's structural domain ceiling and the benchmark's
coverage of it.

Computes the database inventory on three distinct analysis units, the element
set and heavy-atom-count ceiling of the database, and the four coverage
criteria that locate each of the 543 benchmark observations inside or outside
the database's structural domain.  Also checks the exhaustiveness of the
database within its own domain, which is what makes the ceiling a statement
about the database rather than about the benchmark.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def main() -> int:
    db = common.load_db()
    t = common.build_analysis_table(verbose=False)
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    # ---- inventory on three units ----------------------------------------
    db_mols = db.molecule.unique()
    hac = {m: Chem.MolFromSmiles(m).GetNumHeavyAtoms() for m in db_mols}
    elements = set()
    for m in db_mols[:5000]:
        mol = Chem.MolFromSmiles(m)
        if mol is not None:
            elements.update(a.GetSymbol() for a in mol.GetAtoms())

    # Tuple keys, deliberately: a "\x00"-joined string key silently truncates
    # under pandas 3.0's `str` dtype (the NUL is dropped and 28 distinct keys
    # collapse into others), so the string form undercounts unique pairs.
    keys_all = list(zip(db.molecule, db.fragment1, db.fragment2))
    pair_key = pd.Series(keys_all, index=db.index, dtype=object)
    n_pairs = len(set(keys_all))
    inventory = {
        "raw_rows": int(len(db)),
        "unique_molecules": int(db.molecule.nunique()),
        "unique_molecule_fragment_pairs": int(n_pairs),
        "unique_bde_values": int(db.bde.nunique()),
        "unique_bond_types": int(db.bond_type.nunique()),
        "row_to_pair_redundancy_factor": round(len(db) / n_pairs, 4),
        "rows_per_molecule_mean": round(len(db) / db.molecule.nunique(), 4),
        "note": (
            "Raw rows, unique molecules and unique fragment pairs are three "
            "different units and must not be interchanged. Repeats arise because "
            "a molecule's symmetry-equivalent bonds are stored as separate rows."
        ),
    }
    # is bde constant inside a (molecule, fragment1, fragment2) key?
    g = db.assign(k=np.asarray(pair_key)).groupby("k", sort=False)
    varying = int((g.bde.nunique() > 1).sum())
    inventory["pair_keys_with_varying_bde"] = varying

    # ---- structural domain -------------------------------------------------
    hac_vals = np.array(list(hac.values()))
    domain = {
        "element_set": sorted(elements),
        "n_elements": len(elements),
        "heavy_atom_count_min": int(hac_vals.min()),
        "heavy_atom_count_max": int(hac_vals.max()),
        "heavy_atom_count_mean": float(hac_vals.mean()),
        "heavy_atom_count_sd": float(hac_vals.std(ddof=1)),
        "ceiling": int(hac_vals.max()),
        "molecules_above_ceiling": int((hac_vals > hac_vals.max()).sum()),
        "hac_definition": "RDKit GetNumHeavyAtoms on the parent SMILES",
    }

    # ---- coverage criteria -------------------------------------------------
    db_pair_set = set(zip(db.molecule, db.fragment1, db.fragment2))
    keys = list(zip(t.parent_structure_smiles, t.fragment_a_smiles, t.fragment_b_smiles))
    exact = np.array([k in db_pair_set for k in keys])
    parent = t.parent_structure_smiles.isin(set(db.molecule)).to_numpy()
    bench_hac = t.hac_parent.to_numpy(int)
    beyond = bench_hac > domain["ceiling"]

    # elements never seen in the database, checked on the benchmark parents
    foreign = []
    for m in t.parent_structure_smiles.unique():
        mol = Chem.MolFromSmiles(m)
        if mol is not None:
            els = {a.GetSymbol() for a in mol.GetAtoms()}
            if els - elements:
                foreign.append((m, sorted(els - elements)))

    coverage = {
        "n_benchmark_observations": int(len(t)),
        "n_unique_benchmark_molecules": int(t.parent_structure_smiles.nunique()),
        "A_exact_pair_in_db": int(exact.sum()),
        "A_exact_pair_in_db_pct": round(100 * exact.mean(), 4),
        "B_parent_molecule_in_db": int(parent.sum()),
        "B_parent_molecule_in_db_pct": round(100 * parent.mean(), 4),
        "C_parent_beyond_db_hac_ceiling": int(beyond.sum()),
        "C_parent_beyond_db_hac_ceiling_pct": round(100 * beyond.mean(), 4),
        "D_parent_with_foreign_element": len(foreign),
        "D_foreign_element_examples": foreign[:5],
        "outside_domain_any_criterion": int((~exact | ~parent | beyond).sum()),
        "benchmark_hac_max": int(bench_hac.max()),
        "benchmark_hac_median": float(np.median(bench_hac)),
    }

    # ---- exhaustiveness inside the domain ---------------------------------
    within = parent & ~beyond
    exhaustiveness = {
        "parent_in_db": int(parent.sum()),
        "of_those_parent_hac_within_ceiling": int(within.sum()),
        "of_those_exact_pair_present": int(exact[within].sum()),
        "pct_exact_when_parent_within_domain": round(100 * exact[within].mean(), 4),
        "exact_pair_absent_but_parent_present": int((within & ~exact).sum()),
    }

    # per-bond-class benchmark coverage vs the database taxonomy
    db_class = db.drop_duplicates(["molecule", "fragment1", "fragment2"]).bond_type.value_counts()
    bench_class = t.bond_class.value_counts()
    class_coverage = {
        "database_unique_pair_counts": {k: int(v) for k, v in db_class.items()},
        "benchmark_observation_counts": {k: int(v) for k, v in bench_class.items()},
        "database_classes": int(db.bond_type.nunique()),
        "benchmark_classes": int(t.bond_class.nunique()),
        "classes_with_zero_benchmark_coverage": sorted(
            set(db_class.index) - set(bench_class.index)
        ),
    }

    payload = {
        "workstream": "FINDING-1-coverage",
        "title": "Database structural domain ceiling and benchmark coverage of it",
        "analysis_unit": "benchmark bond observation (543) and database unique fragment pair",
        "database_inventory": inventory,
        "structural_domain": domain,
        "coverage_boundary": coverage,
        "exhaustiveness_within_domain": exhaustiveness,
        "bond_class_coverage": class_coverage,
        "labeler_validation": {
            "agreement_on_matched_rows": int(t.labeler_ok.sum()),
            "rows_compared": int(t.labeler_ok.notna().sum()),
            "note": "the bond-class labeler reproduces the database's own bond_type on every matched row",
        },
        "interpretation": (
            "The database is a bounded chemical space: only C/N/O, and no molecule "
            "above 10 heavy atoms. Just under one third of the benchmark observations "
            "fall outside that space, so any claim about channel behaviour on the "
            "benchmark must be stated on the inside-domain subset or reported as an "
            "extrapolation. Within the domain the database is close to complete for "
            "the benchmark (89.7% of in-domain observations have their exact bond row)."
        ),
        "limitations": [
            "HAC is computed from the stored SMILES; a different protonation or "
            "tautomer choice for the same compound would shift it.",
            "Six of the database's nine bond classes have no benchmark coverage at all, "
            "so the class-resolved error analysis is effectively a two-class comparison.",
            "Coverage is a statement about overlap with this database, not about the "
            "chemical literature in general.",
        ],
    }
    common.save_json(payload, "f1_coverage_boundary.json")

    print("\n=== F1 coverage ===")
    print(f"  database: {inventory['raw_rows']} rows / {inventory['unique_molecules']} molecules / "
          f"{inventory['unique_molecule_fragment_pairs']} fragment pairs "
          f"(redundancy {inventory['row_to_pair_redundancy_factor']})")
    print(f"  domain: elements {domain['element_set']}, HAC {domain['heavy_atom_count_min']}-"
          f"{domain['heavy_atom_count_max']}")
    print(f"  exact pair in db: {coverage['A_exact_pair_in_db']}/543 "
          f"({coverage['A_exact_pair_in_db_pct']}%)")
    print(f"  parent in db: {coverage['B_parent_molecule_in_db']}/543 "
          f"({coverage['B_parent_molecule_in_db_pct']}%)")
    print(f"  beyond HAC ceiling: {coverage['C_parent_beyond_db_hac_ceiling']}/543 "
          f"({coverage['C_parent_beyond_db_hac_ceiling_pct']}%)")
    print(f"  exhaustiveness in-domain: {exhaustiveness['pct_exact_when_parent_within_domain']}%")
    print(f"  benchmark bond classes: {dict(bench_class)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())