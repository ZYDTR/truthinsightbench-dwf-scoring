import os
import pandas as pd
from scipy import stats

DATA = "/data/user/conversations/conv-7eebfb61-13ed-4509-8100-c75a3d34d930/project/data/tracked_unit_spatial_stability.tsv"
OUT = "/data/user/conversations/conv-7eebfb61-13ed-4509-8100-c75a3d34d930/project/output/results"
FIG = "/data/user/conversations/conv-7eebfb61-13ed-4509-8100-c75a3d34d930/project/output/figures"
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

df = pd.read_csv(DATA, sep="\t")
df["combined_metric"] = df[
    ["new_environment_stability_1", "new_environment_stability_2", "new_environment_stability_3"]
].mean(axis=1)

METRICS = [
    "repeated_environment_stability_1",
    "repeated_environment_stability_2",
    "new_environment_stability_1",
    "new_environment_stability_2",
    "new_environment_stability_3",
    "between_repeated_environment_stability",
    "repeated_environment_change",
    "novel_vs_repeated_change",
]

def welch_stats(a, b):
    """Welch t-test; returns t, p, diff (b-a), CI95 for diff (Satterthwaite)."""
    a, b = a, b
    t, p = stats.ttest_ind(a, b, equal_var=False)
    va, vb, na, nb = a.var(ddof=1), b.var(ddof=1), len(a), len(b)
    se = (va / na + vb / nb) ** 0.5
    ss_df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    diff = b.mean() - a.mean()
    margin = stats.t.ppf(0.975, df=ss_df) * se
    return t, p, diff, diff - margin, diff + margin

rows = []
for m in METRICS:
    r = df.loc[df.condition == "reference", m]
    p = df.loc[df.condition == "matrix_perturbed", m]
    t, pw, diff, _, _ = welch_stats(r, p)
    rows.append({
        "metric": m,
        "ref_n_units": int(len(r)),
        "pert_n_units": int(len(p)),
        "ref_unit_mean": r.mean(),
        "pert_unit_mean": p.mean(),
        "diff_pert_minus_ref": diff,
        "t": t,
        "p_welch": pw,
        "cohens_d": diff / ((r.var(ddof=1) + p.var(ddof=1)) / 2) ** 0.5,
    })
unit_df = pd.DataFrame(rows)

arows = []
df_a = df.groupby("animal_id").agg(
    condition=("condition", lambda s: s.iloc[0]),
    **{m: (m, "mean") for m in METRICS + ["combined_metric"]},
)
for m in METRICS:
    ra = df_a.loc[df_a.condition == "reference", m]
    pa = df_a.loc[df_a.condition == "matrix_perturbed", m]
    t, pw, diff, lo, hi = welch_stats(ra, pa)
    arows.append({
        "metric": m,
        "ref_n_animals": int(len(ra)),
        "pert_n_animals": int(len(pa)),
        "ref_animal_mean": ra.mean(),
        "pert_animal_mean": pa.mean(),
        "diff": diff,
        "p_welch": pw,
        "ci_lo": lo,
        "ci_hi": hi,
    })
animal_df = pd.DataFrame(arows)

alt = []
for lvl, idx in [("unit", "tracked_unit_id"), ("animal", "animal_id")]:
    if lvl == "unit":
        r = df.loc[df.condition == "reference", "combined_metric"]
        p = df.loc[df.condition == "matrix_perturbed", "combined_metric"]
    else:
        cm = df.groupby("animal_id").combined_metric.mean()
        ref_ids = set(df.loc[df.condition == "reference", "animal_id"])
        pert_ids = set(df.loc[df.condition == "matrix_perturbed", "animal_id"])
        r, p = cm[cm.index.isin(ref_ids)], cm[cm.index.isin(pert_ids)]
    t, pw, diff, _, _ = welch_stats(r, p)
    alt.append({"combined_metric": "mean_of_3_new_env", "level": lvl,
                "ref_mean": r.mean(), "pert_mean": p.mean(), "diff": diff, "p_welch": pw})
alt_df = pd.DataFrame(alt)

unit_df.to_csv(OUT + "/stability_unit_level_effect.csv", index=False)
animal_df.to_csv(OUT + "/stability_animal_level_effect.csv", index=False)
alt_df.to_csv(OUT + "/stability_alt_definition.csv", index=False)

print("=== DESIGN VERIFICATION ===")
print("distinct animals:", int(df.animal_id.nunique()))
print("ref animals:", sorted(df.loc[df.condition == "reference", "animal_id"].unique()))
print("pert animals:", sorted(df.loc[df.condition == "matrix_perturbed", "animal_id"].unique()))
print("overlap:", int(len(set(df.loc[df.condition == "reference", "animal_id"]) &
                           set(df.loc[df.condition == "matrix_perturbed", "animal_id"]))))

for f in ["stability_unit_level_effect.csv", "stability_animal_level_effect.csv", "stability_alt_definition.csv"]:
    print("----", OUT + "/" + f)
    print(pd.read_csv(OUT + "/" + f).to_string(index=False))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, m in zip(axes, ["between_repeated_environment_stability", "repeated_environment_change", "novel_vs_repeated_change"]):
        for cond, col in [("reference", "tab:blue"), ("matrix_perturbed", "tab:orange")]:
            ax.boxplot(df.loc[df.condition == cond, m], label=cond)
        ax.set_title(m)
    fig.tight_layout()
    fig.savefig(FIG + "/stability_by_condition.png", dpi=150)
    print("boxplot saved")
except Exception as e:
    print("boxplot failed:", e)
