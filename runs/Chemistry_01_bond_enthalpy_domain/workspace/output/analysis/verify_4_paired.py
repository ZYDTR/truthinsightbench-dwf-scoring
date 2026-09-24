"""Independent adversarial verification of claim 4 (paired channel comparisons).

Recomputes, from the raw TSV only, with an independent implementation:
  * paired t, Wilcoxon signed-rank (two independent implementations), sign test
  * robustness: tie handling, outlier drop, molecule clustering, cluster sign-flip
  * the ~165 pooled-within-sd anomaly seen in the workstream's JSON
"""
import numpy as np
import pandas as pd
from scipy import stats

TSV = "/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/data/normalized_views/experimental_benchmark.tsv"
df = pd.read_csv(TSV, sep="\t")
assert len(df) == 543 and df.notna().all().all()

exp = df.experimental_bde_kcal_mol.to_numpy(float)
ae = {c: np.abs(df[f"{c}_estimate_bde_kcal_mol"].to_numpy(float) - exp)
      for c in ("quantum", "learned", "reference")}
mols = df.parent_structure_smiles.to_numpy()

print("unit check: rows=%d unique obs=%d unique (mol,f1,f2)=%d unique molecules=%d"
      % (len(df), df.observation_id.nunique(),
         df[["parent_structure_smiles", "fragment_a_smiles", "fragment_b_smiles"]]
         .drop_duplicates().shape[0], df.parent_structure_smiles.nunique()))
print("MAE:", {c: round(ae[c].mean(), 5) for c in ae})


def wilcoxon_manual(d):
    """Independent tie-corrected normal-approximation Wilcoxon signed-rank."""
    d = d[d != 0]
    nz = len(d)
    r = stats.rankdata(np.abs(d), method="average")
    W = min(r[d > 0].sum(), r[d < 0].sum())
    mu = nz * (nz + 1) / 4.0
    _, cnt = np.unique(np.abs(d), return_counts=True)
    sd = np.sqrt(nz * (nz + 1) * (2 * nz + 1) / 24.0 - ((cnt ** 3 - cnt).sum()) / 48.0)
    z = (W - mu) / sd
    return W, 2 * stats.norm.cdf(z)


def cluster_signflip(d, groups, reps=100000, seed=20260919):
    """Vectorised sign-flip permutation at the molecule level (paired dispersion)."""
    rng = np.random.default_rng(seed)
    codes, _ = pd.factorize(pd.Series(groups))
    ncl = codes.max() + 1
    sums = np.bincount(codes, weights=d, minlength=ncl)
    obs = abs(d.mean())
    signs = rng.choice(np.array([-1.0, 1.0]), size=(reps, ncl))
    tot = np.abs(signs @ sums) / len(d)
    return float((tot >= obs - 1e-12).sum() + 1) / (reps + 1)


results = {}
for a, b in (("quantum", "learned"), ("quantum", "reference")):
    d = ae[a] - ae[b]
    t, pt = stats.ttest_rel(ae[a], ae[b])
    W_s, p_s = stats.wilcoxon(d, zero_method="wilcox")
    W_m, p_m = wilcoxon_manual(d)
    nb, nw, ties = int((d < 0).sum()), int((d > 0).sum()), int((d == 0).sum())
    p_sign = stats.binomtest(nb, nb + nw, 0.5).pvalue
    results[f"{a}_vs_{b}"] = dict(
        mean_ae_diff=float(d.mean()), mae_a=float(ae[a].mean()), mae_b=float(ae[b].mean()),
        median_diff=float(np.median(d)),
        t=float(t), t_p=float(pt), wilcoxon_W=float(W_s), wilcoxon_p_scipy=float(p_s),
        wilcoxon_p_manual_normal=float(p_m),
        sign_a_better=nb, sign_b_better=nw, ties=ties, sign_p=float(p_sign),
    )
    print(f"\n{a} vs {b}: mean(ae_{a}-ae_{b})={d.mean():.5f}  median={np.median(d):.4f}")
    print(f"  t={t:.5f} p={pt:.6g} | wilcoxon scipy p={p_s:.6g} | manual p={p_m:.6g} agree={abs(p_s-p_m)<1e-9}")
    print(f"  sign {a}-better={nb} {b}-better={nw} ties={ties} p={p_sign:.6g}")
    for meth in ("wilcox", "pratt", "zsplit", "approx"):
        print(f"    zero_method={meth}: p={stats.wilcoxon(d, zero_method=meth if meth!='approx' else 'wilcox', method='approx').pvalue:.6g}")
    # cluster sign-flip permutation on molecule clusters (mean statistic)
    p_cl = cluster_signflip(d, mols)
    print(f"  molecule-cluster sign-flip permutation p (100k) = {p_cl:.6g}")

print("\n--- pooled within-molecule SD forensics (the ~165 anomaly) ---")
g = df.groupby("parent_structure_smiles")["experimental_bde_kcal_mol"]
multi = df[df.parent_structure_smiles.isin(g.size()[g.size() > 1].index)]
gg = multi.groupby("parent_structure_smiles")["experimental_bde_kcal_mol"]
var = gg.var()                      # unbiased per-group variance
w = gg.size() - 1
ss = gg.apply(lambda s: ((s - s.mean()) ** 2).sum()).sum()
correct_df_weighted = float(np.sqrt(ss / w.sum()))
variance_pooled = float(np.sqrt((w * var).sum() / w.sum()))
print(f"correct df-weighted pooled SD  = {correct_df_weighted:.4f} kcal/mol")
print(f"naive variance-pooled SD       = {variance_pooled:.4f} kcal/mol")
print(f"reproducing original artefact  = sqrt(sum(w*var^2)/sum(w)) = "
      f"{float(np.sqrt((w * var ** 2).sum() / w.sum())):.4f} (variances squared again)")
print(f"units of artefact numerator    = kcal^4 -> result is sqrt(E[var^2]) not an SD")

results["pooled_within_sd_by_molecule"] = {
    "correct_df_weighted_pooled_sd": correct_df_weighted,
    "naive_variance_pooled_sd": variance_pooled,
    "artefact_variances_squared": float(np.sqrt((w * var ** 2).sum() / w.sum())),
}
np.save("/tmp/verify4_results.npy", results, allow_pickle=True)
print("\nDONE")