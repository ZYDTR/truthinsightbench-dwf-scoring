"""Sensitivity of claim 9 to cut point / domain confound, and method-A variants for claim 5."""
import json
import numpy as np
import pandas as pd

P = "/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project"
b = pd.read_csv(P + "/output/results/_verify9_table.csv")
ch = b[b.bond_class == "C-H"].copy()
out = {}

# ---- 1. cut-point sensitivity for the cross-over ----
tab = {}
for cut in [8, 9, 10, 11, 12]:
    hi = ch[ch.hac >= cut]
    lo = ch[ch.hac <= 5]
    tab[f">= {cut}"] = {"n": int(len(hi)), "learned": float(hi.learned_ae.mean()),
                        "quantum": float(hi.quantum_ae.mean()),
                        "edge": float(hi.learned_ae.mean() - hi.quantum_ae.mean())}
for cut in [3, 4, 5, 6, 7]:
    lo = ch[ch.hac <= cut]
    tab[f"<= {cut}"] = {"n": int(len(lo)), "learned": float(lo.learned_ae.mean()),
                        "quantum": float(lo.quantum_ae.mean()),
                        "edge": float(lo.learned_ae.mean() - lo.quantum_ae.mean())}
out["cut_sensitivity_CH"] = tab

# sign of edge per single HAC value
perhac = {}
for h in sorted(ch.hac.unique()):
    d = ch[ch.hac == h]
    perhac[int(h)] = {"n": int(len(d)), "learned": float(d.learned_ae.mean()),
                      "quantum": float(d.quantum_ae.mean()),
                      "edge": float(d.learned_ae.mean() - d.quantum_ae.mean())}
out["per_hac_CH"] = perhac

# point-biserial / rank correlation of edge with HAC
from scipy import stats
d_ae = (ch.learned_ae - ch.quantum_ae).to_numpy(float)
rho, p = stats.spearmanr(ch.hac.to_numpy(float), d_ae)
out["spearman_hac_vs_edge"] = {"rho": float(rho), "p": float(p), "n": int(len(ch))}
pw, pp = stats.pearsonr(ch.hac.to_numpy(float), d_ae)
out["pearson_hac_vs_edge"] = {"r": float(pw), "p": float(pp)}

# ---- 2. domain confound: DB max HAC and membership per HAC ----
db = pd.read_csv(P + "/data/source_data/bde_db_full.csv.gz")
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
dbm = db.molecule.drop_duplicates()
hac_db = np.array([Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in dbm if Chem.MolFromSmiles(s)])
out["db_domain"] = {"n_unique_molecules": int(dbm.nunique()), "hac_max": int(hac_db.max()),
                    "n_above_10": int((hac_db > 10).sum())}
# all rows (all bond classes) split by domain membership
b["parent_in_db"] = b.parent_structure_smiles.isin(set(db.molecule))
ch["parent_in_db"] = ch.parent_structure_smiles.isin(set(db.molecule))
out["benchmark_domain_by_hac"] = {}
for lo, hi in [(0, 5), (6, 9), (10, 10), (11, 100)]:
    d = b[(b.hac >= lo) & (b.hac <= hi)]
    out["benchmark_domain_by_hac"][f"{lo}-{hi}"] = {
        "n": int(len(d)), "frac_parent_in_db": float(d.parent_in_db.mean()) if len(d) else None}

# domain-stratified cross-over (all classes, and C-H)
def cell(df, label, mask, sub):
    d = df[mask & sub]
    if len(d) == 0:
        return {"n": 0}
    return {"n": int(len(d)), "learned": float(d.learned_ae.mean()),
            "quantum": float(d.quantum_ae.mean()),
            "edge": float(d.learned_ae.mean() - d.quantum_ae.mean())}
strat = {}
for cls_name, df in [("all", b), ("C-H", ch)]:
    strat[cls_name] = {
        "in_db|hac<=5": cell(df, "x", df.parent_in_db, df.hac <= 5),
        "in_db|hac6-9": cell(df, "x", df.parent_in_db, (df.hac >= 6) & (df.hac <= 9)),
        "in_db|hac>=10": cell(df, "x", df.parent_in_db, df.hac >= 10),
        "out_db|hac<=5": cell(df, "x", ~df.parent_in_db, df.hac <= 5),
        "out_db|hac6-9": cell(df, "x", ~df.parent_in_db, (df.hac >= 6) & (df.hac <= 9)),
        "out_db|hac>=10": cell(df, "x", ~df.parent_in_db, df.hac >= 10),
        "in_db|hac==10": cell(df, "x", df.parent_in_db, df.hac == 10),
        "out_db|hac>10": cell(df, "x", ~df.parent_in_db, df.hac > 10),
    }
out["domain_stratified"] = strat

print(json.dumps(out, indent=2, default=str))
json.dump(out, open(P + "/output/results/_verify9_sens.json", "w"), indent=2, default=str)