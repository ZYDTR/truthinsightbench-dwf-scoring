"""Adversarial probes for claim 9: unit sensitivity, domain-boundary confound,
and decomposition of the 'fragile overall edge' interpretation."""
import json
import numpy as np
import pandas as pd
from scipy import stats
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
PROJECT = "/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project"
b = pd.read_csv(PROJECT + "/output/results/_verify9_table.csv")

out = {}
ch = b[b.bond_class == "C-H"].copy()
cc = b[b.bond_class == "C-C"].copy()

# ---- 1. exact claimed slices, parent-HAC unit (bond observation) ----
def mae_ae(d):
    return (float(d.learned_ae.mean()), float(d.quantum_ae.mean()), int(len(d)))

g = ch[ch.hac >= 10]; s = ch[ch.hac <= 5]
out["claimed_slices_recomputed"] = {
    "CH_hac_ge10": mae_ae(g), "CH_hac_le5": mae_ae(s),
}
out["counts"] = {
    "CH_total": int(len(ch)), "CH_hac_ge10": int(len(g)), "CH_hac_le5": int(len(s)),
    "CH_hac_ge10_unique_mols": int(g.parent_structure_smiles.nunique()),
    "CH_hac_le5_unique_mols": int(s.parent_structure_smiles.nunique()),
    "CH_hac_ge10_hac_eq_10": int((g.hac == 10).sum()),
    "CH_hac_ge10_hac_gt_10": int((g.hac > 10).sum()),
}

# ---- 2. domain-boundary split: HAC==10 (in DB domain) vs HAC>10 (outside) ----
d10 = ch[ch.hac == 10]; dgt = ch[ch.hac > 10]
out["domain_split"] = {
    "hac_eq_10": {"n": int(len(d10)), "learned": float(d10.learned_ae.mean()),
                  "quantum": float(d10.quantum_ae.mean())},
    "hac_gt_10": {"n": int(len(dgt)), "learned": float(dgt.learned_ae.mean()),
                  "quantum": float(dgt.quantum_ae.mean())},
}
# DB membership of parents in each slice
db = pd.read_csv(PROJECT + "/data/source_data/bde_db_full.csv.gz")
db_mols = set(db.molecule)
ch["mol_in_db"] = ch.parent_structure_smiles.isin(db_mols)
for name, mask in [("hac>=10", ch.hac >= 10), ("hac<=5", ch.hac <= 5),
                   ("hac==10", ch.hac == 10), ("hac>10", ch.hac > 10)]:
    dd = ch[mask]
    out.setdefault("parent_in_db", {})[name] = {
        "n": int(len(dd)), "frac_parent_in_db": float(dd.mol_in_db.mean()),
        "learned_in_db": float(dd[dd.mol_in_db].learned_ae.mean()) if dd.mol_in_db.any() else None,
        "quantum_in_db": float(dd[dd.mol_in_db].quantum_ae.mean()) if dd.mol_in_db.any() else None,
        "learned_out_db": float(dd[~dd.mol_in_db].learned_ae.mean()) if (~dd.mol_in_db).any() else None,
        "quantum_out_db": float(dd[~dd.mol_in_db].quantum_ae.mean()) if (~dd.mol_in_db).any() else None,
    }

# ---- 3. C-C bond class: does cross-over replicate? ----
out["C-C"] = {
    "hac_ge10": {"n": int(len(cc[cc.hac >= 10])), "learned": float(cc[cc.hac >= 10].learned_ae.mean()),
                 "quantum": float(cc[cc.hac >= 10].quantum_ae.mean())},
    "hac_le5": {"n": int(len(cc[cc.hac <= 5])), "learned": float(cc[cc.hac <= 5].learned_ae.mean()),
                "quantum": float(cc[cc.hac <= 5].quantum_ae.mean())},
}

# ---- 4. overall-edge decomposition by size slice (all classes, bond observation) ----
N = len(b)
def edge(d):
    return float(d.learned_ae.mean() - d.quantum_ae.mean())
out["overall"] = {"n": N, "learned": float(b.learned_ae.mean()),
                  "quantum": float(b.quantum_ae.mean()), "edge": edge(b)}
contrib = {}
for label, mask in [("hac<=5", b.hac <= 5), ("hac 6-9", (b.hac >= 6) & (b.hac <= 9)),
                    ("hac>=10", b.hac >= 10)]:
    d = b[mask]
    e = edge(d)
    contrib[label] = {"n": int(len(d)), "weight": len(d) / N, "slice_edge": e,
                      "absolute_contribution_to_overall_edge": (len(d) / N) * e}
out["overall_edge_decomposition"] = contrib
out["overall_edge_leaving_out_hac_ge10"] = edge(b[b.hac < 10])
out["overall_edge_only_hac_le5"] = edge(b[b.hac <= 5])
out["overall_edge_only_hac_ge10"] = edge(b[b.hac >= 10])

# C-H-only decomposition
contrib_ch = {}
for label, mask in [("hac<=5", ch.hac <= 5), ("hac 6-9", (ch.hac >= 6) & (ch.hac <= 9)),
                    ("hac>=10", ch.hac >= 10)]:
    d = ch[mask]
    contrib_ch[label] = {"n": int(len(d)), "slice_edge": edge(d),
                         "abs_contrib_to_CH_edge": (len(d) / len(ch)) * edge(d)}
out["CH_edge_decomposition"] = {"overall_CH_edge": edge(ch), "slices": contrib_ch}
out["CH_edge_excluding_hac_ge10"] = edge(ch[ch.hac < 10])

# ---- 5. bootstrap CI of the slice edges and of the overall edge ----
rng = np.random.default_rng(12345)
def boot_edge(d, nb=10000):
    dl = d.learned_ae.to_numpy(); dq = d.quantum_ae.to_numpy(); n = len(d)
    idx = rng.integers(0, n, size=(nb, n))
    diffs = dl[idx].mean(axis=1) - dq[idx].mean(axis=1)
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))
out["bootstrap95"] = {
    "overall_edge": boot_edge(b), "CH_hac_ge10_edge": boot_edge(g),
    "CH_hac_le5_edge": boot_edge(s), "overall_edge_without_hac_ge10": boot_edge(b[b.hac < 10]),
}

# ---- 6. paired significance on slices ----
for lbl, d in [("CH_hac_ge10", g), ("CH_hac_le5", s), ("overall", b)]:
    t, p = stats.ttest_rel(d.learned_ae, d.quantum_ae)
    out.setdefault("paired_t", {})[lbl] = {"t": float(t), "p": float(p), "n": int(len(d))}
# interaction test: (learned-quantum) diff-of-diffs between size groups on C-H
diff_g = (g.learned_ae - g.quantum_ae).to_numpy()
diff_s = (s.learned_ae - s.quantum_ae).to_numpy()
tt, pp = stats.ttest_ind(diff_g, diff_s, equal_var=False)
out["interaction_dod"] = {"mean_diff_ge10": float(diff_g.mean()), "mean_diff_le5": float(diff_s.mean()),
                          "difference": float(diff_g.mean() - diff_s.mean()), "welch_t": float(tt),
                          "p": float(pp)}
# permutation test
obs = diff_g.mean() - diff_s.mean()
pool = np.concatenate([diff_g, diff_s])
ng = len(diff_g)
cnt = 0
for _ in range(20000):
    perm = rng.permutation(pool)
    if abs(perm[:ng].mean() - perm[ng:].mean()) >= abs(obs):
        cnt += 1
out["interaction_perm_p"] = (cnt + 1) / 20001

# ---- 7. unit sensitivity: unique molecules (one row per molecule) ----
def dedup_slices(d):
    u = d.drop_duplicates(subset=["parent_structure_smiles"])
    return u
uu = dedup_slices(ch)
out["unit_unique_molecule_CH"] = {
    "hac_ge10": {"n": int(len(uu[uu.hac >= 10])), "learned": float(uu[uu.hac >= 10].learned_ae.mean()),
                 "quantum": float(uu[uu.hac >= 10].quantum_ae.mean())},
    "hac_le5": {"n": int(len(uu[uu.hac <= 5])), "learned": float(uu[uu.hac <= 5].learned_ae.mean()),
                "quantum": float(uu[uu.hac <= 5].quantum_ae.mean())},
}
# molecule-mean aggregation (all rows of a molecule averaged)
agg = ch.groupby("parent_structure_smiles").agg(
    learned_ae=("learned_ae", "mean"), quantum_ae=("quantum_ae", "mean"), hac=("hac", "first")).reset_index()
out["unit_molecule_mean_CH"] = {
    "hac_ge10": {"n": int(len(agg[agg.hac >= 10])), "learned": float(agg[agg.hac >= 10].learned_ae.mean()),
                 "quantum": float(agg[agg.hac >= 10].quantum_ae.mean())},
    "hac_le5": {"n": int(len(agg[agg.hac <= 5])), "learned": float(agg[agg.hac <= 5].learned_ae.mean()),
                "quantum": float(agg[agg.hac <= 5].quantum_ae.mean())},
}

# within-molecule duplication check
out["duplication"] = {
    "CH_hac_le5_rows_vs_mols": [int(len(s)), int(s.parent_structure_smiles.nunique())],
    "CH_hac_ge10_rows_vs_mols": [int(len(g)), int(g.parent_structure_smiles.nunique())],
    "top_dup_mols_CH_hac_le5": s.parent_structure_smiles.value_counts().head(5).to_dict(),
}

print(json.dumps(out, indent=2, default=str))
json.dump(out, open(PROJECT + "/output/results/_verify9_probe.json", "w"), indent=2, default=str)