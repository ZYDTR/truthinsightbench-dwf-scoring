"""Independent recomputation of claim 9 (size interaction / cross-over).

Reads ONLY:
  data/normalized_views/experimental_benchmark.tsv
  data/source_data/bde_db_full.csv.gz
  data/source_data/experimental_source_tables.xlsx
Does NOT import common.py or the workstream scripts.
"""
import json
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

PROJECT = "/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project"
TSV = PROJECT + "/data/normalized_views/experimental_benchmark.tsv"
DB_GZ = PROJECT + "/data/source_data/bde_db_full.csv.gz"

bench = pd.read_csv(TSV, sep="\t")
db = pd.read_csv(DB_GZ)
print("bench", bench.shape, "db", db.shape)

# ---------------- chemistry (own implementation) ----------------
def heavy(smiles):
    m = Chem.MolFromSmiles(smiles)
    return None if m is None else m.GetNumHeavyAtoms()

def heavy_with_h(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return Chem.AddHs(m).GetNumAtoms()

def rad_center(smiles):
    """Element of the atom bearing the radical electron (0 radicals -> first atom)."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    rads = [a for a in m.GetAtoms() if a.GetNumRadicalElectrons() > 0]
    if len(rads) == 1:
        return rads[0].GetSymbol()
    # fall back: atom with unfilled valence / first atom
    return m.GetAtomWithIdx(0).GetSymbol() if not rads else rads[0].GetSymbol()

# sanity: do fragments parse, do radicals appear?
bad_parent = [s for s in bench.parent_structure_smiles.unique() if Chem.MolFromSmiles(s) is None]
print("unparseable parents:", len(bad_parent), bad_parent[:5])
frags = pd.unique(pd.concat([bench.fragment_a_smiles, bench.fragment_b_smiles]))
bad_frag = [s for s in frags if Chem.MolFromSmiles(s) is None]
print("unparseable frags:", len(bad_frag), bad_frag[:5])
nrad0 = sum(1 for s in frags if Chem.MolFromSmiles(s) is not None and
            sum(a.GetNumRadicalElectrons() for a in Chem.MolFromSmiles(s).GetAtoms()) == 0)
print("fragments with 0 radical electrons:", nrad0, "of", len(frags))

def bond_class(a, b):
    ea, eb = rad_center(a), rad_center(b)
    if ea is None or eb is None:
        return None
    order = ["C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
    els = sorted([ea, eb], key=lambda e: order.index(e) if e in order else 99)
    return "-".join(els)

hac_map = {s: heavy(s) for s in bench.parent_structure_smiles.unique()}
hacH_map = {s: heavy_with_h(s) for s in bench.parent_structure_smiles.unique()}
bench["hac"] = bench.parent_structure_smiles.map(hac_map)
bench["hac_H"] = bench.parent_structure_smiles.map(hacH_map)
bench["bond_class"] = [bond_class(a, b) for a, b in
                       zip(bench.fragment_a_smiles, bench.fragment_b_smiles)]

bench["quantum_ae"] = (bench.quantum_estimate_bde_kcal_mol - bench.experimental_bde_kcal_mol).abs()
bench["learned_ae"] = (bench.learned_estimate_bde_kcal_mol - bench.experimental_bde_kcal_mol).abs()
bench["reference_ae"] = (bench.reference_estimate_bde_kcal_mol - bench.experimental_bde_kcal_mol).abs()

print("bond class counts (top):")
print(bench.bond_class.value_counts().head(12))
print("hac distribution:", bench.hac.min(), bench.hac.median(), bench.hac.max())

# ---------------- claimed slices ----------------
def slice_stats(df, label):
    return {
        "label": label, "n": int(len(df)),
        "learned_mae": float(df.learned_ae.mean()) if len(df) else None,
        "quantum_mae": float(df.quantum_ae.mean()) if len(df) else None,
        "reference_mae": float(df.reference_ae.mean()) if len(df) else None,
        "learned_rmse": float(np.sqrt((df.learned_ae ** 2).mean())) if len(df) else None,
        "quantum_rmse": float(np.sqrt((df.quantum_ae ** 2).mean())) if len(df) else None,
    }

results = {}
ch = bench[bench.bond_class == "C-H"]
results["C-H"] = {
    "hac_ge_10": slice_stats(ch[ch.hac >= 10], "C-H HAC>=10"),
    "hac_le_5": slice_stats(ch[ch.hac <= 5], "C-H HAC<=5"),
    "all": slice_stats(ch, "C-H all"),
}
allm = bench
results["all_classes"] = {
    "hac_ge_10": slice_stats(allm[allm.hac >= 10], "all HAC>=10"),
    "hac_le_5": slice_stats(allm[allm.hac <= 5], "all HAC<=5"),
    "all": slice_stats(allm, "all"),
}
# HAC including hydrogens variant
chH = bench[bench.bond_class == "C-H"]
results["C-H_H_included"] = {
    "hac_ge_10": slice_stats(chH[chH.hac_H >= 10], "C-H HAC_H>=10"),
    "hac_le_5": slice_stats(chH[chH.hac_H <= 5], "C-H HAC_H<=5"),
    "all": slice_stats(chH, "C-H all"),
}
# fragmentation-size alternative: heavy atoms of parent fragments combined
bench["hac_fragsum"] = [ (heavy(a) or 0) + (heavy(b) or 0) for a, b in
                         zip(bench.fragment_a_smiles, bench.fragment_b_smiles)]
chF = bench[bench.bond_class == "C-H"]
results["C-H_fragsum"] = {
    "hac_ge_10": slice_stats(chF[chF.hac_fragsum >= 10], "C-H fragsum>=10"),
    "hac_le_5": slice_stats(chF[chF.hac_fragsum <= 5], "C-H fragsum<=5"),
}

# overall edge
results["overall_edge"] = {
    "learned_mae_all": float(bench.learned_ae.mean()),
    "quantum_mae_all": float(bench.quantum_ae.mean()),
    "edge_learned_minus_quantum": float(bench.learned_ae.mean() - bench.quantum_ae.mean()),
}
# C-H edge overall
results["overall_edge_C_H"] = {
    "learned_mae": float(ch.learned_ae.mean()),
    "quantum_mae": float(ch.quantum_ae.mean()),
    "edge": float(ch.learned_ae.mean() - ch.quantum_ae.mean()),
}

# cross-over check: finer bins
bins = []
for lo, hi in [(0, 3), (4, 5), (6, 7), (8, 9), (10, 12), (13, 20), (21, 100)]:
    d = ch[(ch.hac >= lo) & (ch.hac <= hi)]
    bins.append({"bin": f"{lo}-{hi}", "n": int(len(d)),
                 "learned": float(d.learned_ae.mean()) if len(d) else None,
                 "quantum": float(d.quantum_ae.mean()) if len(d) else None})
results["C-H_bins"] = bins

# paired test on HAC>=10 C-H
d10 = ch[ch.hac >= 10]
from scipy import stats
if len(d10) > 1:
    t, p = stats.ttest_rel(d10.learned_ae, d10.quantum_ae)
    results["C-H_hac_ge_10_paired_t"] = {"t": float(t), "p": float(p), "n": int(len(d10))}
d5 = ch[ch.hac <= 5]
if len(d5) > 1:
    t, p = stats.ttest_rel(d5.learned_ae, d5.quantum_ae)
    results["C-H_hac_le_5_paired_t"] = {"t": float(t), "p": float(p), "n": int(len(d5))}

print(json.dumps(results, indent=2))

# also per-unique-molecule and per-unique-fragment-pair dedup sanity
uniq_mol = bench.drop_duplicates(subset=["parent_structure_smiles"])
chm = uniq_mol[uniq_mol.bond_class == "C-H"]
results["dedup_one_row_per_molecule_C-H"] = {
    "hac_ge_10": slice_stats(chm[chm.hac >= 10], "C-H HAC>=10 dedup mol"),
    "hac_le_5": slice_stats(chm[chm.hac <= 5], "C-H HAC<=5 dedup mol"),
}

with open(PROJECT + "/output/results/_verify9_raw.json", "w") as f:
    json.dump(results, f, indent=2)
bench.to_csv(PROJECT + "/output/results/_verify9_table.csv", index=False)
print("wrote raw json")