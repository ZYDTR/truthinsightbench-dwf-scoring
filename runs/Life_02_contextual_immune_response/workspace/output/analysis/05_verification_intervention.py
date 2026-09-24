import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata, spearmanr
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

ROOT = "/data/user/conversations/conv-beae8ffb-0726-4214-9c89-748e1330ccf7/project"
checks, errors = [], []


def add(claim, recomputed, agrees, expression):
    checks.append({"claim": claim, "recomputed": recomputed,
                   "agrees": bool(agrees), "expression": expression})


# ---------------- (A) mechanistic_intervention ----------------
A = pd.read_csv(f"{ROOT}/data/mechanistic_intervention_replicates.tsv", sep="\t")
meanA = A.groupby(["assay_id", "intervention_setting_id"])["response_value"].mean().unstack()
settings = sorted(meanA.columns)
assays = sorted(meanA.index)
# rank each assay's settings 1 = lowest mean
rank_mat = meanA.rank(axis=1, method="average")[settings]
print("Mean matrix (rows assays, cols settings):")
print(meanA.round(5))
print("Rank matrix:")
print(rank_mat)

R = rank_mat.values  # rows = assays (blocks), cols = settings (treatments)
n, k = R.shape
col_sums = R.sum(axis=0)
S = ((col_sums - col_sums.mean()) ** 2).sum()
# tie correction per block: sum(t^3 - t)
T = 0.0
for row in R:
    _, cnt = np.unique(row, return_counts=True)
    T += (cnt ** 3 - cnt).sum()
W = 12.0 * S / (n ** 2 * (k ** 3 - k) - n * T)
add("V3a Kendall's W over 3x5 rank matrix", round(W, 6), abs(W - 0.955556) < 5e-6,
    "W = 12*S/(n^2*(k^3-k)-n*T) with S=sum(col_rank_sums-mean)^2, n=3,k=5")

chi2 = n * (k - 1) * W
# also tie-corrected Friedman chi2 from scipy for comparison
fr = stats.friedmanchisquare(*[R[i, :] for i in range(n)])
p_chi = stats.chi2.sf(chi2, k - 1)
print(f"chi2={chi2:.6f} df={k-1} p={p_chi:.6f} | scipy friedman chi2={fr.statistic:.6f} p={fr.pvalue:.6f}")
add("V3b chi2 = n_assays*(n_settings-1)*W, df=4, p", {"chi2": round(chi2, 4), "df": k - 1,
    "p": round(p_chi, 6)}, abs(chi2 - 11.4667) < 5e-4 and abs(p_chi - 0.0218) < 5e-4,
    "chi2 = 3*(5-1)*W; p = scipy.stats.chi2.sf(chi2, 4)")

lowest = meanA[settings].idxmin(axis=1)
print("Lowest-mean setting per assay:", dict(lowest))
add("V3c lowest-ranked setting per assay", dict(lowest),
    all(v == "intervention_setting_1" for v in lowest),
    "meanA.idxmin(axis=1) over setting columns")

rho13 = spearmanr(rank_mat.loc["assay_1"].values, rank_mat.loc["assay_3"].values).statistic
rho23 = spearmanr(rank_mat.loc["assay_2"].values, rank_mat.loc["assay_3"].values).statistic
print(f"spearman a1-a3={rho13} a2-a3={rho23}")
add("V3d Spearman assay_1 vs assay_3 = 1.000", round(rho13, 6), abs(rho13 - 1.0) < 1e-9,
    "scipy.stats.spearmanr(rank_assay1, rank_assay3)")
add("V3d Spearman assay_2 vs assay_3 = 0.900", round(rho23, 6), abs(rho23 - 0.9) < 1e-6,
    "scipy.stats.spearmanr(rank_assay2, rank_assay3)")

# ---------------- (B) multi_condition_response ----------------
B = pd.read_csv(f"{ROOT}/data/multi_condition_response_replicates.tsv", sep="\t")
B["setting_index"] = B["compound_setting_id"].str.extract(r"(\d+)").astype(int)


def fit_class(df, name):
    m = smf.ols("response_value ~ setting_index", data=df).fit()
    return m, m.params["setting_index"], m.rsquared, m.pvalues["setting_index"], name


mA, slA, r2A, pA, _ = fit_class(B[B.response_class_id == "response_class_A"], "A")
mB, slB, r2B, pB, _ = fit_class(B[B.response_class_id == "response_class_B"], "B")
print(f"A: slope={slA:.6f} R2={r2A:.6f} p={pA:.6f} | B: slope={slB:.6f} R2={r2B:.6f} p={pB:.6f}")
add("V4a class A slope 8.593, R2 0.9587", {"slope": round(slA, 4), "R2": round(r2A, 4)},
    abs(slA - 8.593) < 5e-3 and abs(r2A - 0.9587) < 5e-4,
    "smf.ols('response_value ~ setting_index') on class A")
add("V4b class B slope 0.191, p 0.5837", {"slope": round(slB, 4), "p": round(pB, 4)},
    abs(slB - 0.191) < 5e-3 and abs(pB - 0.5837) < 5e-4,
    "smf.ols('response_value ~ setting_index') on class B")

full = smf.ols("response_value ~ C(response_class_id) * C(compound_setting_id)", data=B).fit()
tab = anova_lm(full, typ=2)
inter = tab.loc["C(response_class_id):C(compound_setting_id)"]
print(tab)
add("V4c class x setting interaction F 45.415, p 4.73e-08",
    {"F": round(inter["F"], 4), "p": inter["PR(>F)"]},
    abs(inter["F"] - 45.415) < 5e-3 and abs(inter["PR(>F)"] - 4.73e-08) < 1e-9,
    "anova_lm(ols('response_value ~ C(response_class_id)*C(compound_setting_id)'), typ=2)")

rep_slopes = {}
for (cls, rep), g in B.groupby(["response_class_id", "replicate_id"]):
    rep_slopes.setdefault(cls, {})[rep] = float(smf.ols("response_value ~ setting_index", data=g).fit().params["setting_index"])
a_vals = sorted(rep_slopes["response_class_A"].values())
b_vals = sorted(rep_slopes["response_class_B"].values())
sep = min(a_vals) > max(b_vals)
print("A replicate slopes:", rep_slopes["response_class_A"])
print("B replicate slopes:", rep_slopes["response_class_B"], "separation:", sep)
add("V4d class A slopes [8.782, 9.291, 7.706], class B [0.403, 0.825, -0.654], complete separation",
    {"A": {k2: round(v, 3) for k2, v in rep_slopes["response_class_A"].items()},
     "B": {k2: round(v, 3) for k2, v in rep_slopes["response_class_B"].items()},
     "min_A": round(min(a_vals), 3), "max_B": round(max(b_vals), 3), "complete_separation": bool(sep)},
    sep and sorted(round(v, 3) for v in a_vals) == [7.706, 8.782, 9.291]
    and sorted(round(v, 3) for v in b_vals) == [-0.654, 0.403, 0.825],
    "per (class,replicate) smf.ols('response_value ~ setting_index') slopes; min(A) > max(B)")

conclusion = ("All checks verified: " + str(sum(c["agrees"] for c in checks)) + "/" + str(len(checks)) +
              " claims agree with recomputation from raw TSVs.") if all(c["agrees"] for c in checks) else \
             ("Mismatches: " + "; ".join(c["claim"] for c in checks if not c["agrees"]))

out = {"unit": "intervention", "checks": checks, "conclusion": conclusion, "errors": errors}
with open(f"{ROOT}/output/results/verification_intervention.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))