"""WS3: build and cache the enriched analysis table plus environment descriptors.

Run once; writes output/analysis/ws3_table.pkl so the analysis scripts do not
re-pay the RDKit / database-join cost on every iteration.
"""
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, "/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis")

import numpy as np
import pandas as pd

import common

OUT_PKL = Path("/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis/ws3_table.pkl")


def benzene_ring_flag(smiles: str) -> bool:
    """True if the parent contains at least one all-aromatic 6-membered ring."""
    Chem = common._rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if len(ring) == 6 and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            return True
    return False


def main() -> None:
    t0 = time.time()
    df = common.build_analysis_table(verbose=True)
    print(f"[ws3] build_analysis_table {time.time()-t0:.1f}s")

    # ---- radical-centre environment descriptors -------------------------
    ra = [common.radical_center(s) for s in df.fragment_a_smiles]
    rb = [common.radical_center(s) for s in df.fragment_b_smiles]
    df["rad_a_element"] = [x["element"] if x else None for x in ra]
    df["rad_b_element"] = [x["element"] if x else None for x in rb]
    df["rad_a_aromatic"] = [x["aromatic"] if x else None for x in ra]
    df["rad_b_aromatic"] = [x["aromatic"] if x else None for x in rb]
    df["rad_a_in_ring"] = [x["in_ring"] if x else None for x in ra]
    df["rad_b_in_ring"] = [x["in_ring"] if x else None for x in rb]
    df["rad_any_aromatic"] = df.rad_a_aromatic | df.rad_b_aromatic
    df["rad_any_in_ring"] = df.rad_a_in_ring | df.rad_b_in_ring
    # number of radical centres in each fragment (1 everywhere -> sanity check)
    df["rad_a_n"] = [x["n_radical_atoms"] if x else None for x in ra]

    # the carbon radical centre specifically (the heavy radical for C-H bonds)
    def carbon_rad(ela, ela_ar, ela_ring, elb, elb_ar, elb_ring):
        for el, ar, ring in ((ela, ela_ar, ela_ring), (elb, elb_ar, elb_ring)):
            if el == "C":
                return ar, ring
        return (None, None)

    cflags = [
        carbon_rad(a, aa, ar_, b, ba, br)
        for a, aa, ar_, b, ba, br in zip(
            df.rad_a_element, df.rad_a_aromatic, df.rad_a_in_ring,
            df.rad_b_element, df.rad_b_aromatic, df.rad_b_in_ring,
        )
    ]
    df["c_rad_aromatic"] = [c[0] for c in cflags]
    df["c_rad_in_ring"] = [c[1] for c in cflags]

    # ---- parent composition ---------------------------------------------
    cache_benz = {}
    cache_el = {}
    for m in pd.unique(df.parent_structure_smiles):
        cache_benz[m] = benzene_ring_flag(m)
        cache_el[m] = sorted(common.element_set(m))
    df["parent_benzene"] = df.parent_structure_smiles.map(cache_benz)
    df["parent_elements"] = df.parent_structure_smiles.map(lambda m: ",".join(cache_el[m]))
    df["parent_has_N"] = df.parent_structure_smiles.map(lambda m: "N" in cache_el[m])
    df["parent_has_O"] = df.parent_structure_smiles.map(lambda m: "O" in cache_el[m])
    df["parent_has_S"] = df.parent_structure_smiles.map(lambda m: "S" in cache_el[m])
    df["parent_n_elements"] = df.parent_elements.map(lambda s: len(s.split(",")) if s else 0)

    # ---- size bin (parent heavy-atom count) ------------------------------
    bins = [-1, 5, 9, 10 ** 6]
    labels = ["<=5", "6-9", ">=10"]
    df["hac_bin"] = pd.cut(df.hac_parent, bins=bins, labels=labels)

    # ---- experimental magnitude bin (quartiles + explicit weak tail) -----
    q = df.experimental_bde_kcal_mol.quantile([0.25, 0.5, 0.75]).to_dict()
    df["exp_mag_quartile"] = pd.qcut(
        df.experimental_bde_kcal_mol, 4, labels=["Q1 weakest", "Q2", "Q3", "Q4 strongest"]
    )
    edges = [0, 70, 85, 100, 10 ** 6]
    elabels = ["<70", "70-85", "85-100", ">=100"]
    df["exp_mag_bin"] = pd.cut(df.experimental_bde_kcal_mol, bins=edges, labels=elabels)

    # ---- molecule-size flags --------------------------------------------
    df["hac"] = df.hac_parent

    OUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PKL.open("wb") as fh:
        pickle.dump(df, fh)
    print(f"[ws3] wrote {OUT_PKL} shape={df.shape} total {time.time()-t0:.1f}s")
    print("[ws3] quartile thresholds:", q)
    print(df[["bond_class", "hac_bin", "exp_mag_bin"]].value_counts().head(20))


if __name__ == "__main__":
    main()