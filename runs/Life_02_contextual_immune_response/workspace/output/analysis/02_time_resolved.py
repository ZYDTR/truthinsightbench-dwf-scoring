#!/usr/bin/env python3
"""Time-resolved response: when, if at all, do the two condition arms separate?

The file carries normalized_response_percent for two condition roles measured at
seven time points (3..21) with four biological replicates each.  Replicates are
paired across roles by replicate_id, so every time point yields four paired
differences; the assay-level question is whether the role contrast is present at
all times or only after a kinetic boundary.

Covers
  T1  per-time-point role contrast (paired, n=4 pairs) with equivalence testing
  T2  time x role interaction (mixed model with a replicate random intercept)
  T3  segmented (early vs late) slope comparison and the crossover time
  T4  robustness: unpaired test, per-replicate slopes, leave-one-replicate-out

Run:  python3 output/analysis/02_time_resolved.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import mixedlm

from common import (
    LOGS,
    ensure_dirs,
    load,
    mean_ci,
    pformat,
    write_json,
    write_table,
)

ensure_dirs()
SECTION_LOG: list[str] = []


def say(message: str = "") -> None:
    print(message)
    SECTION_LOG.append(message)


time_df = load("time_resolved")
assert time_df["normalized_response_percent"].notna().all()
times = sorted(time_df["time_units"].unique())
roles = sorted(time_df["condition_role"].unique())

say("=" * 78)
say("T1  Time-resolved response: design")
say("=" * 78)
say(f"rows={len(time_df)}  times={times}  roles={roles}")
say(f"replicates per role x time: "
    f"{sorted(time_df.groupby(['condition_role', 'time_units']).size().unique())} "
    f"(balanced = {time_df.groupby(['condition_role', 'time_units']).size().nunique() == 1})")

# wide table: one row per time x replicate, baseline and perturbation side by side
wide = time_df.pivot_table(
    index=["time_units", "replicate_id"],
    columns="condition_role",
    values="normalized_response_percent",
).reset_index()
wide.columns.name = None
wide["difference"] = wide["perturbation"] - wide["baseline"]
say(f"paired rows available at every time point: "
    f"{bool(wide['difference'].notna().all())} ({len(wide)} pairs)")

say("\n   per-time-point means (percent of the common normalization reference):")
say(f"   {'time':>5} {'baseline':>10} {'perturbation':>13} {'difference':>11}")
for t in times:
    block = wide[wide["time_units"] == t]
    say(f"   {t:5.0f} {block['baseline'].mean():10.3f} "
        f"{block['perturbation'].mean():13.3f} {block['difference'].mean():+11.3f}")

# --------------------------------------------------------------------------
# T1  per-time-point paired contrast + equivalence testing
# --------------------------------------------------------------------------
say("\n" + "=" * 78)
say("T2  Per-time-point role contrast (paired by replicate_id, n = 4 pairs)")
say("=" * 78)

time_rows = []
for t in times:
    block = wide[wide["time_units"] == t]
    diff = block["difference"].to_numpy()
    test = stats.ttest_rel(block["perturbation"], block["baseline"])
    ci = mean_ci(diff)
    # equivalence (TOST) against a +/-3 percentage-point margin, a margin set to
    # the size of the late-time divergence so it is a claim about the scale that
    # matters for this assay rather than an arbitrary constant
    margin = 3.0
    low, high = -margin, margin
    se = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
    dfree = len(diff) - 1
    t_low = (float(np.mean(diff)) - low) / se
    t_high = (float(np.mean(diff)) - high) / se
    p_lower = float(stats.t.sf(t_low, dfree))       # H0: mean <= -3
    p_upper = float(stats.t.cdf(t_high, dfree))     # H0: mean >= +3
    tost_p = max(p_lower, p_upper)
    time_rows.append(
        {
            "time_units": float(t),
            "n_pairs": int(len(diff)),
            "baseline_mean": float(block["baseline"].mean()),
            "perturbation_mean": float(block["perturbation"].mean()),
            "difference_mean": ci["mean"],
            "difference_sd": ci["sd"],
            "difference_ci_low": ci["ci_low"],
            "difference_ci_high": ci["ci_high"],
            "paired_t": float(test.statistic),
            "paired_p": float(test.pvalue),
            "tost_margin": margin,
            "tost_p": tost_p,
            "equivalent_within_margin": bool(tost_p < 0.05),
        }
    )
time_frame = pd.DataFrame(time_rows)

for _, row in time_frame.iterrows():
    say(f"   t={row['time_units']:5.0f}  diff = {row['difference_mean']:+.3f} "
        f"[{row['difference_ci_low']:+.3f}, {row['difference_ci_high']:+.3f}]  "
        f"paired t = {row['paired_t']:+.3f} (p = {pformat(row['paired_p'])})  "
        f"TOST vs +/-3pp: p = {pformat(row['tost_p'])} "
        f"equivalent={row['equivalent_within_margin']}")

n_sig = int((time_frame["paired_p"] < 0.05).sum())
sig_times = time_frame.loc[time_frame["paired_p"] < 0.05, "time_units"].tolist()
say(f"   time points with p < 0.05: {n_sig}/7 -> {sig_times}")
say(f"   early window (t <= 15) all non-significant: "
    f"{bool((time_frame.loc[time_frame.time_units <= 15, 'paired_p'] >= 0.05).all())}")
early_eq = time_frame[time_frame["time_units"] <= 15]["equivalent_within_margin"].tolist()
say(f"   early window equivalent within +/-3pp: {early_eq}")

# --------------------------------------------------------------------------
# T2  time x role interaction
# --------------------------------------------------------------------------
say("\n" + "=" * 78)
say("T3  Time x role interaction")
say("=" * 78)

long = time_df.rename(columns={"normalized_response_percent": "response"})
long["role"] = pd.Categorical(long["condition_role"], categories=["baseline", "perturbation"])
interaction_fit = None
try:
    model = mixedlm(
        "response ~ time_units * C(condition_role)",
        long,
        groups=long["replicate_id"],
    ).fit(reml=True)
    interaction_fit = {
        "formula": "response ~ time_units * C(condition_role); random intercept for replicate_id",
        "params": {str(k): float(v) for k, v in model.params.items()},
        "bse": {str(k): float(v) for k, v in model.bse.items()},
        "pvalues": {str(k): float(v) for k, v in model.pvalues.items()},
        "interaction_coef": float(
            model.params.get("time_units:C(condition_role)[T.perturbation]", np.nan)
        ),
        "interaction_p": float(
            model.pvalues.get("time_units:C(condition_role)[T.perturbation]", np.nan)
        ),
        "group_variance": float(model.cov_re.iloc[0, 0]),
        "residual_variance": float(model.scale),
    }
    say(f"   mixed model {interaction_fit['formula']}")
    say(f"   time x role interaction coef = {interaction_fit['interaction_coef']:+.4f} "
        f"percent per time unit (p = {pformat(interaction_fit['interaction_p'])})")
    say(f"   baseline time slope = "
        f"{interaction_fit['params']['time_units']:+.4f}; "
        f"perturbation slope = "
        f"{interaction_fit['params']['time_units'] + interaction_fit['interaction_coef']:+.4f}")
    say(f"   variance: replicate = {interaction_fit['group_variance']:.4f}, "
        f"residual = {interaction_fit['residual_variance']:.4f}")
except Exception as error:  # reported, never hidden
    say(f"   [mixed model FAILED] {type(error).__name__}: {error}")
    interaction_fit = {"error": f"{type(error).__name__}: {error}"}

# difference-series regression: slope of (perturbation - baseline) over time
slope_fit = stats.linregress(wide["time_units"], wide["difference"])
say(f"   difference-series OLS: slope = {slope_fit.slope:+.4f} percent per time unit "
    f"(SE {slope_fit.stderr:.4f}, p = {pformat(float(slope_fit.pvalue))}, "
    f"R2 = {float(slope_fit.rvalue) ** 2:.3f})")

# --------------------------------------------------------------------------
# T3  segmented slopes: early vs late
# --------------------------------------------------------------------------
say("\n" + "=" * 78)
say("T4  Kinetic boundary: segmented slopes and crossover")
say("=" * 78)

segments = {"early_3_to_12": (3.0, 12.0), "late_15_to_21": (15.0, 21.0)}
segment_rows = []
for label, (lo, hi) in segments.items():
    block = long[(long["time_units"] >= lo) & (long["time_units"] <= hi)]
    slopes = {}
    for role, sub in block.groupby("condition_role"):
        fit = stats.linregress(sub["time_units"], sub["response"])
        slopes[role] = fit
        say(f"   {label:14s} {role:13s} slope = {fit.slope:+.4f} percent/unit "
            f"(SE {fit.stderr:.4f}, p = {pformat(float(fit.pvalue))}, R2 = {float(fit.rvalue) ** 2:.3f})")
    difference = slopes["perturbation"].slope - slopes["baseline"].slope
    say(f"   {label:14s} slope difference (perturbation - baseline) = {difference:+.4f}")
    segment_rows.append(
        {
            "segment": label,
            "time_low": lo,
            "time_high": hi,
            "baseline_slope": float(slopes["baseline"].slope),
            "baseline_slope_se": float(slopes["baseline"].stderr),
            "baseline_p": float(slopes["baseline"].pvalue),
            "perturbation_slope": float(slopes["perturbation"].slope),
            "perturbation_slope_se": float(slopes["perturbation"].stderr),
            "perturbation_p": float(slopes["perturbation"].pvalue),
            "slope_difference": float(difference),
        }
    )
segment_frame = pd.DataFrame(segment_rows)

# per-replicate slopes of the difference series -> tests whether the drift is
# consistent across biological replicates
rep_slope_rows = []
for replicate, block in wide.groupby("replicate_id", sort=True):
    fit = stats.linregress(block["time_units"], block["difference"])
    rep_slope_rows.append(
        {
            "replicate_id": replicate,
            "difference_slope": float(fit.slope),
            "difference_slope_se": float(fit.stderr),
            "difference_slope_p": float(fit.pvalue),
        }
    )
rep_slopes = pd.DataFrame(rep_slope_rows)
rep_slope_test = stats.ttest_1samp(rep_slopes["difference_slope"], 0.0)
say("\n   per-replicate slope of (perturbation - baseline) over time:")
for _, row in rep_slopes.iterrows():
    say(f"      {row['replicate_id']}: slope = {row['difference_slope']:+.4f} "
        f"(p = {pformat(row['difference_slope_p'])})")
say(f"   one-sample t on the 4 replicate slopes vs 0: t = "
    f"{float(rep_slope_test.statistic):+.3f}, p = {pformat(float(rep_slope_test.pvalue))}")

# crossover: first time at which the paired contrast is significant
first_sig = (
    float(time_frame.loc[time_frame["paired_p"] < 0.05, "time_units"].min())
    if n_sig
    else None
)
say(f"   first time point with a significant paired contrast: {first_sig}")

# magnitude of the late divergence relative to its own baseline arm
late_block = time_frame[time_frame["time_units"] >= 18]
say(f"   late window mean divergence: "
    f"{late_block['difference_mean'].mean():+.3f} percentage points "
    f"(baseline arm mean {late_block['baseline_mean'].mean():.3f})")

# --------------------------------------------------------------------------
# T4  robustness
# --------------------------------------------------------------------------
say("\n" + "=" * 78)
say("T5  Robustness of the kinetic boundary")
say("=" * 78)

# (a) unpaired test ignoring replicate pairing
unpaired_rows = []
for t in times:
    block = time_df[time_df["time_units"] == t]
    base = block.loc[block["condition_role"] == "baseline", "normalized_response_percent"]
    pert = block.loc[block["condition_role"] == "perturbation", "normalized_response_percent"]
    test = stats.ttest_ind(pert, base, equal_var=False)
    unpaired_rows.append(
        {
            "time_units": float(t),
            "welch_t": float(test.statistic),
            "welch_p": float(test.pvalue),
        }
    )
unpaired = pd.DataFrame(unpaired_rows).merge(
    time_frame[["time_units", "paired_p"]], on="time_units"
)
say("   (a) Welch unpaired tests vs the paired tests:")
for _, row in unpaired.iterrows():
    say(f"      t={row['time_units']:5.0f}  Welch p = {pformat(row['welch_p'])}  "
        f"paired p = {pformat(row['paired_p'])}")
say(f"      same significance calls: "
    f"{bool(((unpaired['welch_p'] < 0.05) == (unpaired['paired_p'] < 0.05)).all())}")

# (b) leave-one-replicate-out on the difference-series slope
loo_rows = []
for replicate in sorted(wide["replicate_id"].unique()):
    block = wide[wide["replicate_id"] != replicate]
    fit = stats.linregress(block["time_units"], block["difference"])
    loo_rows.append(
        {
            "left_out_replicate": replicate,
            "difference_slope": float(fit.slope),
            "p_value": float(fit.pvalue),
        }
    )
loo_frame = pd.DataFrame(loo_rows)
say("   (b) leave-one-replicate-out difference-series slope:")
for _, row in loo_frame.iterrows():
    say(f"      omit {row['left_out_replicate']}: slope = {row['difference_slope']:+.4f} "
        f"(p = {pformat(row['p_value'])})")
say(f"      slope range across folds: "
    f"{loo_frame['difference_slope'].min():+.4f} to {loo_frame['difference_slope'].max():+.4f}")

# (c) alternative endpoint: ratio instead of difference
wide_ratio = wide.assign(ratio=wide["perturbation"] / wide["baseline"])
ratio_fit = stats.linregress(wide_ratio["time_units"], wide_ratio["ratio"])
say(f"   (c) alternative contrast (ratio perturbation/baseline): slope = "
    f"{ratio_fit.slope:+.5f} per time unit, p = {pformat(float(ratio_fit.pvalue))}")
ratio_by_time = wide_ratio.groupby("time_units")["ratio"].mean()
say(f"      ratio by time: "
    f"{ {int(k): round(v, 4) for k, v in ratio_by_time.items()} }")

# (d) trend test that does not assume linearity: Spearman of difference vs time
rho_diff = stats.spearmanr(wide["time_units"], wide["difference"])
say(f"   (d) Spearman(difference, time) = {float(rho_diff.statistic):+.3f} "
    f"(p = {pformat(float(rho_diff.pvalue))})")

write_table(time_frame, "time_resolved_contrasts.csv")
write_table(segment_frame, "time_resolved_segments.csv")
write_table(rep_slopes, "time_resolved_replicate_slopes.csv")
write_table(unpaired, "time_resolved_unpaired_robustness.csv")
write_table(loo_frame, "time_resolved_leave_one_replicate_out.csv")
write_table(wide, "time_resolved_paired_wide.csv")

summary = {
    "design": {
        "n_rows": int(len(time_df)),
        "times": [float(t) for t in times],
        "roles": roles,
        "replicates_per_cell": int(time_df.groupby(["condition_role", "time_units"]).size().iloc[0]),
        "balanced": bool(time_df.groupby(["condition_role", "time_units"]).size().nunique() == 1),
    },
    "per_time_point": time_frame.to_dict(orient="records"),
    "significant_time_points": sig_times,
    "first_significant_time": first_sig,
    "early_window_all_nonsignificant": bool(
        (time_frame.loc[time_frame.time_units <= 15, "paired_p"] >= 0.05).all()
    ),
    "mixed_model": interaction_fit,
    "difference_series": {
        "slope": float(slope_fit.slope),
        "stderr": float(slope_fit.stderr),
        "p": float(slope_fit.pvalue),
        "r_squared": float(slope_fit.rvalue) ** 2,
    },
    "segments": segment_frame.to_dict(orient="records"),
    "replicate_slopes": rep_slopes.to_dict(orient="records"),
    "replicate_slope_ttest_p": float(rep_slope_test.pvalue),
    "robustness": {
        "unpaired_matches_paired": bool(
            ((unpaired["welch_p"] < 0.05) == (unpaired["paired_p"] < 0.05)).all()
        ),
        "leave_one_replicate_slope_min": float(loo_frame["difference_slope"].min()),
        "leave_one_replicate_slope_max": float(loo_frame["difference_slope"].max()),
        "ratio_contrast_slope": float(ratio_fit.slope),
        "ratio_contrast_p": float(ratio_fit.pvalue),
        "spearman_difference_time": float(rho_diff.statistic),
        "spearman_p": float(rho_diff.pvalue),
    },
}
write_json("time_resolved_summary.json", summary)

(LOGS / "02_time_resolved.log").write_text("\n".join(SECTION_LOG) + "\n", encoding="utf-8")
print("\n[done] 02_time_resolved.py")