"""WS1 step 1: per-molecule structural descriptors for every database parent molecule.

Parses each unique parent SMILES once with RDKit and records HAC + element set.
Cross-validated against common.heavy_atom_count / common.element_set below.
Results cached to output/cache/db_mol_desc.pkl.
"""
import sys, time, json
sys.path.insert(0, '/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis')
import common, pandas as pd, numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

PROJ = '/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project'
db = pd.read_pickle(f'{PROJ}/output/cache/db.pkl')
mols = pd.unique(db.molecule.values)
t0 = time.time()
rows = []
n_bad = 0
for i, m in enumerate(mols):
    mol = Chem.MolFromSmiles(m)
    if mol is None:
        n_bad += 1
        rows.append((m, None, None))
        continue
    rows.append((m, mol.GetNumHeavyAtoms(),
                 "|".join(sorted(a.GetSymbol() for a in mol.GetAtoms()))))
desc = pd.DataFrame(rows, columns=['molecule', 'hac', 'elements'])
desc.to_pickle(f'{PROJ}/output/cache/db_mol_desc.pkl')
print(f"parsed {len(mols)} unique parent molecules in {time.time()-t0:.1f}s; unparsable={n_bad}")

# ---- cross-validate the fast path against the shared helper ---------------
sample = list(np.random.default_rng(7).choice(len(mols), size=200, replace=False))
bad = 0
for i in sample:
    m = mols[i]
    if common.heavy_atom_count(m) != desc.hac.iloc[i]:
        bad += 1
    if "|".join(sorted(common.element_set(m))) != desc.elements.iloc[i]:
        bad += 1
print(f"cross-validation on 200 random molecules: mismatches={bad}")
print("HAC range:", desc.hac.min(), "-", desc.hac.max())
print("unparsable molecules:", n_bad)
