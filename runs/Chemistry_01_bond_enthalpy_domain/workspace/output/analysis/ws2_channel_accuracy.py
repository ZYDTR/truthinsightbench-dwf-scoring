"""WS2 - head-to-head accuracy of the three estimation channels.

Same 543 experimental observations, three estimate columns.  Everything below
comes from executed code; the only inputs are the three files under data/.

Analysis units used and labelled explicitly:
  * bond observation  (n = 543)          - the benchmark rows
  * unique molecule   (n = 419)          - rows collapsed by parent SMILES
  * unique fragment pair (n = 539)       - rows collapsed by (frag_a, frag_b)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path("/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project")
sys.path.insert(0, str(PROJECT / "output" / "analysis"))
import common  # noqa: E402

from scipy import stats  # noqa: E402

OUT_JSON = common.RESULTS / "ws2_channel_accuracy.json"
FIG = common.FIGURES / "ws2_accuracy.png"
CH = ("quantum", "learned", "reference")


def f(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


# ==========================================================================
# 0. build table
# ==========================================================================
df = common.build_analysis_table(verbose=True)
exp = df[common.EXP_COL].to_numpy(dtype=float)
n = len(df)

res: dict = {
    "workstream": "WS2",
    "question": "head-to-head accuracy of quantum / learned / reference estimates on the same 543 experimental observations",
    "units": {
        "bond_observations": int(n),
        "unique_molecules": int(df.parent_structure_smiles.nunique()),
        "unique_fragment_pairs": int(df.groupby(["fragment_a_smiles", "fragment_b_smiles"]).ngroups),
        "note": "all headline statistics below use the bond observation (row) as the unit unless labelled otherwise",
    },
    "provenance": {
        "in_reference_training_set_true": int(df.train_flag.sum()),
        "in_reference_training_set_false": int((~df.train_flag).sum()),
        "train_flag_equals_in_db_row": bool((df.train_flag == df.in_db_row).all()),
        "n_missing_estimates": 0,
    },
}

# ==========================================================================
# 1. per-channel error statistics (unit: bond observation, n=543)
# ==========================================================================
Q = [0.0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0]
QNAME = ["min", "p05", "p25", "p50", "p75", "p95", "max"]

per_channel = {}
for c in CH:
    err = df[f"{c}_err"].to_numpy(dtype=float)
    ae = np.abs(err)
    qs = np.quantile(err, Q)
    per_channel[c] = {
        "n": int(n),
        "mae": f(common.mae(err)),
        "rmse": f(common.rmse(err)),
        "median_abs_error": f(np.median(ae)),
        "mean_signed_bias": f(err.mean()),
        "sd_signed_error": f(err.std(ddof=1)),
        "mean_experiment": f(exp.mean()),
        "mean_estimate": f(df[common.CHAN_COL[c]].mean()),
        "max_abs_error": f(ae.max()),
        "p90_abs_error": f(np.quantile(ae, 0.90)),
        "p95_abs_error": f(np.quantile(ae, 0.95)),
        "frac_within_5_kcal": f(float((ae <= 5).mean())),
        "frac_within_10_kcal": f(float((ae <= 10).mean())),
        "signed_error_quantiles": {k: f(v) for k, v in zip(QNAME, qs)},
        "signed_error_quantiles_definition": "estimate minus experiment, kcal/mol; +ve = overestimate",
    }
res["per_channel_error_stats"] = per_channel

# same tables on the two other analysis units (molecule / fragment pair) as robustness
def unit_table(frame: pd.DataFrame, keycols: list[str]) -> dict:
    out = {}
    g = frame.groupby(keycols, as_index=False).agg(
        {common.EXP_COL: "mean", **{common.CHAN_COL[c]: "mean" for c in CH}}
    )
    out["n_units"] = int(len(g))
    for c in CH:
        e = g[common.CHAN_COL[c]].to_numpy() - g[common.EXP_COL].to_numpy()
        out[c] = {
            "mae": f(common.mae(e)),
            "rmse": f(common.rmse(e)),
            "median_abs_error": f(float(np.median(np.abs(e)))),
            "mean_signed_bias": f(float(e.mean())),
        }
    return out


res["per_channel_error_stats_on_unique_molecules"] = unit_table(df, ["parent_structure_smiles"])
res["per_channel_error_stats_on_unique_fragment_pairs"] = unit_table(
    df, ["fragment_a_smiles", "fragment_b_smiles"]
)

# ==========================================================================
# 2. calibration regressions: estimate = a + b * experiment
# ==========================================================================
calib = {}
for c in CH:
    pred = df[common.CHAN_COL[c]].to_numpy(dtype=float)
    lr = stats.linregress(exp, pred)  # slope, intercept, rvalue, pvalue, stderr
    r = float(lr.rvalue)
    # 95% CI on slope
    tcrit = float(stats.t.ppf(0.975, n - 2))
    slope_ci = (float(lr.slope - tcrit * lr.stderr), float(lr.slope + tcrit * lr.stderr))
    # test slope == 1
    t1 = (lr.slope - 1.0) / lr.stderr
    p1 = float(2 * stats.t.sf(abs(t1), n - 2))
    # Spearman as a rank check
    rho, p_rho = stats.spearmanr(exp, pred)
    calib[c] = {
        "slope": f(lr.slope),
        "slope_95ci": [f(slope_ci[0]), f(slope_ci[1])],
        "slope_stderr": f(lr.stderr),
        "intercept": f(lr.intercept),
        "pearson_r": f(r),
        "r2": f(r**2),
        "pearson_p": f(lr.pvalue),
        "spearman_rho": f(rho),
        "spearman_p": f(p_rho),
        "slope_vs_1_t": f(t1),
        "slope_vs_1_p": f(p1),
        "compressed_or_amplified": ("compressed" if lr.slope < 1 else "amplified"),
    }
res["calibration_estimate_on_experiment"] = calib

# error at the ends of the experimental range: does compression under-predict strong bonds
# and over-predict weak bonds?
ends = {}
for c in CH:
    err = df[f"{c}_err"].to_numpy(dtype=float)
    lo = exp <= np.quantile(exp, 0.10)
    hi = exp >= np.quantile(exp, 0.90)
    ends[c] = {
        "n_low_decile": int(lo.sum()),
        "n_high_decile": int(hi.sum()),
        "mean_signed_error_lowest_decile": f(float(err[lo].mean())),
        "mean_signed_error_highest_decile": f(float(err[hi].mean())),
        "mae_lowest_decile": f(float(np.abs(err[lo]).mean())),
        "mae_highest_decile": f(float(np.abs(err[hi]).mean())),
        "mae_middle_80pct": f(float(np.abs(err[~lo & ~hi]).mean())),
        "pearson_r_signed_error_vs_experiment": f(float(np.corrcoef(err, exp)[0, 1])),
    }
res["error_vs_range_position"] = ends

# ==========================================================================
# 3+4. paired comparisons (unit: bond observation, n=543)
# ==========================================================================
PAIRS = [("quantum", "learned"), ("quantum", "reference"), ("learned", "reference")]
paired = {}
for a, b in PAIRS:
    ae_a = df[f"{a}_ae"].to_numpy(dtype=float)
    ae_b = df[f"{b}_ae"].to_numpy(dtype=float)
    d = ae_a - ae_b  # >0 => a has larger error => b better
    mean_d, lo, hi = common.paired_bootstrap_ci(d, n_boot=20000)
    t, p_t = common.paired_t_test(d)
    w, p_w = common.wilcoxon(d)
    # sign test on discordant pairs (exact binomial)
    npos = int((d > 0).sum())
    nneg = int((d < 0).sum())
    nties = int((d == 0).sum())
    p_sign = float(stats.binomtest(npos, npos + nneg, 0.5).pvalue) if (npos + nneg) > 0 else None
    delta = common.cliffs_delta(ae_a, ae_b)
    # bootstrap CI for the MEDIAN difference too
    rng = np.random.default_rng(20260919)
    draws = rng.integers(0, n, size=(20000, n))
    med_draws = np.median(d[draws], axis=1)
    med_lo, med_hi = np.quantile(med_draws, [0.025, 0.975])
    # paired t-test on the signed errors themselves (bias difference)
    e_a = df[f"{a}_err"].to_numpy(dtype=float)
    e_b = df[f"{b}_err"].to_numpy(dtype=float)
    t_bias, p_bias = common.paired_t_test(e_a - e_b)
    # how often is each one the winner
    paired[f"{a}_vs_{b}"] = {
        "n_pairs": int(n),
        "mae_first": f(common.mae(ae_a)),
        "mae_second": f(common.mae(ae_b)),
        "mae_difference_first_minus_second": f(float(ae_a.mean() - ae_b.mean())),
        "mean_abs_error_difference": f(mean_d),
        "mean_abs_error_difference_ci95": [f(lo), f(hi)],
        "median_abs_error_difference": f(float(np.median(d))),
        "median_abs_error_difference_ci95": [f(med_lo), f(med_hi)],
        "paired_t_stat": f(t),
        "paired_t_p": f(p_t),
        "wilcoxon_stat": f(w),
        "wilcoxon_p": f(p_w),
        "sign_test_n_first_worse": npos,
        "sign_test_n_first_better": nneg,
        "sign_test_n_tied": nties,
        "sign_test_p": f(p_sign),
        "cliffs_delta_first_vs_second": f(delta),
        "cliffs_delta_sign_convention": "positive => first channel's |error| tends to exceed the second's",
        "paired_t_on_signed_bias_difference": f(t_bias),
        "paired_t_on_signed_bias_difference_p": f(p_bias),
        "rmse_first": f(common.rmse(ae_a)),
        "rmse_second": f(common.rmse(ae_b)),
    }
res["paired_comparisons"] = paired

# paired comparison restricted to unique molecules (unit: molecule, n=419)
mol = (
    df.groupby("parent_structure_smiles", as_index=False)
    .agg({common.EXP_COL: "mean", **{common.CHAN_COL[c]: "mean" for c in CH}})
)
paired_mol = {}
for a, b in PAIRS:
    ae_a = np.abs(mol[common.CHAN_COL[a]].to_numpy() - mol[common.EXP_COL].to_numpy())
    ae_b = np.abs(mol[common.CHAN_COL[b]].to_numpy() - mol[common.EXP_COL].to_numpy())
    d = ae_a - ae_b
    m, lo, hi = common.paired_bootstrap_ci(d, n_boot=20000)
    t, p = common.paired_t_test(d)
    w, pw = common.wilcoxon(d)
    paired_mol[f"{a}_vs_{b}"] = {
        "n_units": int(len(mol)),
        "mean_abs_error_difference": f(m),
        "ci95": [f(lo), f(hi)],
        "paired_t_p": f(p),
        "wilcoxon_p": f(pw),
        "cliffs_delta": f(common.cliffs_delta(ae_a, ae_b)),
    }
res["paired_comparisons_unique_molecules"] = paired_mol

# ==========================================================================
# 5. in_reference_training_set split (== exact DB row present)
# ==========================================================================
train = df[df.train_flag]
held = df[~df.train_flag]
split = {
    "n_train": int(len(train)),
    "n_held_out": int(len(held)),
    "train_is_exactly_in_db_row": bool(df.loc[df.train_flag, "in_db_row"].all()),
    "held_out_never_in_db_row": bool((~df.loc[~df.train_flag, "in_db_row"]).any() is False),
    "experiment_mean_train": f(float(train[common.EXP_COL].mean())),
    "experiment_mean_held_out": f(float(held[common.EXP_COL].mean())),
    "channels": {},
}
for c in CH:
    et = np.abs(train[f"{c}_err"].to_numpy(dtype=float))
    eh = np.abs(held[f"{c}_err"].to_numpy(dtype=float))
    mae_t, mae_h = float(et.mean()), float(eh.mean())
    # bootstrap CI for each MAE and for the degradation
    rng = np.random.default_rng(20260919)
    bt = et[rng.integers(0, len(et), size=(20000, len(et)))].mean(axis=1)
    bh = eh[rng.integers(0, len(eh), size=(20000, len(eh)))].mean(axis=1)
    deg_draws = bh - bt
    # Welch test for the difference of MAEs (unpaired, different observations)
    tw, pw = stats.ttest_ind(eh, et, equal_var=False)
    # Mann-Whitney
    uw, pw2 = stats.mannwhitneyu(eh, et, alternative="two-sided")
    split["channels"][c] = {
        "mae_train_subset": f(mae_t),
        "mae_train_ci95": [f(np.quantile(bt, 0.025)), f(np.quantile(bt, 0.975))],
        "mae_held_out_subset": f(mae_h),
        "mae_held_out_ci95": [f(np.quantile(bh, 0.025)), f(np.quantile(bh, 0.975))],
        "degradation_absolute_kcal": f(mae_h - mae_t),
        "degradation_absolute_ci95": [
            f(np.quantile(deg_draws, 0.025)),
            f(np.quantile(deg_draws, 0.975)),
        ],
        "degradation_ratio": f(mae_h / mae_t),
        "rmse_train": f(common.rmse(train[f"{c}_err"].to_numpy(dtype=float))),
        "rmse_held_out": f(common.rmse(held[f"{c}_err"].to_numpy(dtype=float))),
        "median_ae_train": f(float(np.median(et))),
        "median_ae_held_out": f(float(np.median(eh))),
        "welch_t": f(tw),
        "welch_p": f(pw),
        "mannwhitney_p": f(pw2),
        "frac_within_10kcal_train": f(float((et <= 10).mean())),
        "frac_within_10kcal_held_out": f(float((eh <= 10).mean())),
    }
# paired channnel ranking within each subset
for name, frame in (("train", train), ("held_out", held)):
    ael = {c: np.abs(frame[f"{c}_err"].to_numpy(dtype=float)) for c in CH}
    dl = {f"{a}_vs_{b}": f(float((ael[a] - ael[b]).mean())) for a, b in PAIRS}
    split[f"mean_ae_differences_{name}"] = dl
res["training_split"] = split

# ==========================================================================
# 6. cross-channel correlation of estimates and of absolute errors
# ==========================================================================
est = pd.DataFrame({c: df[common.CHAN_COL[c]].to_numpy(dtype=float) for c in CH})
corr_est_pearson = est.corr(method="pearson")
corr_est_spearman = est.corr(method="spearman")
ae = pd.DataFrame({c: np.abs(df[f"{c}_err"].to_numpy(dtype=float)) for c in CH})
corr_ae_pearson = ae.corr(method="pearson")
corr_ae_spearman = ae.corr(method="spearman")
# also correlate |err| with |err| after removing the shared difficulty signal?
res["cross_channel_correlation"] = {
    "estimates_pearson": corr_est_pearson.round(6).to_dict(),
    "estimates_spearman": corr_est_spearman.round(6).to_dict(),
    "absolute_errors_pearson": corr_ae_pearson.round(6).to_dict(),
    "absolute_errors_spearman": corr_ae_spearman.round(6).to_dict(),
    "experiment_vs_estimates_pearson": {
        c: f(float(np.corrcoef(exp, est[c])[0, 1])) for c in CH
    },
}
# does reference fail where the others fail? conditioning on quantum error quartiles
qq = pd.qcut(ae["quantum"], 4, labels=["Q1_best", "Q2", "Q3", "Q4_worst"])
cond = ae.groupby(qq, observed=True).mean()
cond["n"] = ae.groupby(qq, observed=True).size()
res["cross_channel_correlation"]["mae_by_quantum_error_quartile"] = {
    str(k): {kk: f(vv) for kk, vv in v.items()} for k, v in cond.to_dict(orient="index").items()
}
qq2 = pd.qcut(ae["reference"], 4, labels=["Q1_best", "Q2", "Q3", "Q4_worst"])
cond2 = ae.groupby(qq2, observed=True).mean()
cond2["n"] = ae.groupby(qq2, observed=True).size()
res["cross_channel_correlation"]["mae_by_reference_error_quartile"] = {
    str(k): {kk: f(vv) for kk, vv in v.items()} for k, v in cond2.to_dict(orient="index").items()
}
# correlation between the three channels' errors (signed) - do they share bias?
errs = pd.DataFrame({c: df[f"{c}_err"].to_numpy(dtype=float) for c in CH})
res["cross_channel_correlation"]["signed_errors_pearson"] = errs.corr().round(6).to_dict()
# hard cases: fraction of the noisiest observations
res["cross_channel_correlation"]["hard_case_overlap"] = {
    "n_obs_in_quantum_worst_quintile": int((ae["quantum"] >= ae["quantum"].quantile(0.8)).sum()),
    "reference_also_in_its_own_worst_quintile": f(
        float(
            (
                ae.loc[ae["quantum"] >= ae["quantum"].quantile(0.8), "reference"]
                >= ae["reference"].quantile(0.8)
            ).mean()
        )
    ),
    "chance_expectation": 0.2,
    "learned_also_in_its_own_worst_quintile": f(
        float(
            (
                ae.loc[ae["quantum"] >= ae["quantum"].quantile(0.8), "learned"]
                >= ae["learned"].quantile(0.8)
            ).mean()
        )
    ),
}

# ==========================================================================
# write JSON
# ==========================================================================
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(res, indent=2, default=common._json_default) + "\n", encoding="utf-8")
print(f"\nwrote {OUT_JSON}")

# ==========================================================================
# figure
# ==========================================================================
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL = {"quantum": "#1f77b4", "learned": "#d62728", "reference": "#2ca02c"}
fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))

# (a) signed-error distributions
ax = axes[0, 0]
bins = np.arange(-45, 45.5, 2.5)
for c in CH:
    ax.hist(df[f"{c}_err"], bins=bins, histtype="step", lw=2, color=COL[c], label=c)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("signed error  estimate - experiment (kcal/mol)")
ax.set_ylabel("bond observations")
ax.set_title("(a) signed-error distribution, n=543")
ax.legend()
ax.set_xlim(-45, 45)

# (b) absolute-error ECDF
ax = axes[0, 1]
for c in CH:
    ae_c = np.sort(df[f"{c}_ae"].to_numpy(dtype=float))
    y = np.arange(1, len(ae_c) + 1) / len(ae_c)
    ax.step(ae_c, y, where="post", lw=2, color=COL[c], label=f"{c} (MAE={ae_c.mean():.2f})")
ax.set_xlabel("absolute error (kcal/mol)")
ax.set_ylabel("cumulative fraction of bond observations")
ax.set_title("(b) absolute-error ECDF")
ax.legend()
ax.set_xlim(0, 45)
ax.grid(alpha=0.3)

# (c) boxplot of absolute errors
ax = axes[0, 2]
bp = ax.boxplot(
    [df[f"{c}_ae"].to_numpy(dtype=float) for c in CH],
    tick_labels=list(CH),
    showfliers=True,
    patch_artist=True,
)
for patch, c in zip(bp["boxes"], CH):
    patch.set_facecolor(COL[c])
    patch.set_alpha(0.6)
ax.set_ylabel("absolute error (kcal/mol)")
ax.set_title("(c) absolute-error boxplots")

# (d) calibration scatter + fits
ax = axes[1, 0]
lims = [10, 170]
ax.plot(lims, lims, "k--", lw=1, label="identity")
for c in CH:
    pred = df[common.CHAN_COL[c]].to_numpy(dtype=float)
    ax.scatter(exp, pred, s=6, alpha=0.25, color=COL[c])
    lr = stats.linregress(exp, pred)
    xs = np.array(lims)
    ax.plot(xs, lr.intercept + lr.slope * xs, color=COL[c], lw=2,
            label=f"{c}: slope={lr.slope:.3f}, R2={lr.rvalue**2:.3f}")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("experimental BDE (kcal/mol)")
ax.set_ylabel("estimated BDE (kcal/mol)")
ax.set_title("(d) calibration, estimate vs experiment")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (e) paired difference in |error|: quantum - learned
ax = axes[1, 1]
d = df["quantum_ae"].to_numpy() - df["learned_ae"].to_numpy()
ax.hist(d, bins=np.arange(-30, 30.5, 1.5), color="#7f7f7f")
ax.axvline(0, color="k", lw=1)
ax.axvline(d.mean(), color="#d62728", lw=2, label=f"mean={d.mean():.3f}")
ax.axvline(np.median(d), color="#1f77b4", lw=2, ls="--", label=f"median={np.median(d):.3f}")
ax.set_xlabel("|err_quantum| - |err_learned| (kcal/mol);  >0 means learned better")
ax.set_ylabel("bond observations")
ax.set_title("(e) paired absolute-error difference")
ax.legend()
ax.set_xlim(-30, 30)

# (f) MAE train vs held-out
ax = axes[1, 2]
w = 0.35
xs = np.arange(len(CH))
tr = [split["channels"][c]["mae_train_subset"] for c in CH]
ho = [split["channels"][c]["mae_held_out_subset"] for c in CH]
ax.bar(xs - w / 2, tr, w, label=f"training subset (n={len(train)})", color="#4c72b0")
ax.bar(xs + w / 2, ho, w, label=f"held-out subset (n={len(held)})", color="#dd8452")
for x, v in zip(xs - w / 2, tr):
    ax.text(x, v + 0.15, f"{v:.2f}", ha="center", fontsize=9)
for x, v in zip(xs + w / 2, ho):
    ax.text(x, v + 0.15, f"{v:.2f}", ha="center", fontsize=9)
ax.set_xticks(xs)
ax.set_xticklabels(list(CH))
ax.set_ylabel("MAE (kcal/mol)")
ax.set_title("(f) MAE by in_reference_training_set split")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

fig.suptitle("WS2: head-to-head accuracy of the three BDE estimation channels (n=543 bond observations)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG, dpi=130)
print(f"wrote {FIG}")

# ==========================================================================
# console summary
# ==========================================================================
print("\n=== per-channel (n=543 bond observations) ===")
print(f"{'channel':<10} {'MAE':>7} {'RMSE':>7} {'medAE':>7} {'bias':>7}")
for c in CH:
    s = per_channel[c]
    print(f"{c:<10} {s['mae']:>7.3f} {s['rmse']:>7.3f} {s['median_abs_error']:>7.3f} {s['mean_signed_bias']:>7.3f}")
print("\n=== calibration ===")
for c in CH:
    k = calib[c]
    print(f"{c:<10} slope={k['slope']:.4f} [{k['slope_95ci'][0]:.4f},{k['slope_95ci'][1]:.4f}] "
          f"intercept={k['intercept']:.3f} r={k['pearson_r']:.4f} R2={k['r2']:.4f} p(slope=1)={k['slope_vs_1_p']:.3g}")
print("\n=== paired ===")
for k, v in paired.items():
    print(f"{k:<22} meanD={v['mean_abs_error_difference']:+.3f} CI[{v['mean_abs_error_difference_ci95'][0]:+.3f},"
          f"{v['mean_abs_error_difference_ci95'][1]:+.3f}] medD={v['median_abs_error_difference']:+.3f} "
          f"t_p={v['paired_t_p']:.4g} wilcox_p={v['wilcoxon_p']:.4g} cliff={v['cliffs_delta_first_vs_second']:+.4f}")
print("\n=== training split ===")
for c in CH:
    s = split["channels"][c]
    print(f"{c:<10} train MAE={s['mae_train_subset']:.3f} held-out MAE={s['mae_held_out_subset']:.3f} "
          f"degradation={s['degradation_absolute_kcal']:+.3f} ratio={s['degradation_ratio']:.3f} "
          f"welch_p={s['welch_p']:.4g}")
print("\n=== correlations ===")
print("estimates pearson:\n", corr_est_pearson.round(4))
print("abs errors pearson:\n", corr_ae_pearson.round(4))
print("abs errors spearman:\n", corr_ae_spearman.round(4))