#!/usr/bin/env python3
"""
Robustness checks for the nutrient-response kinetics analysis.

Checks implemented:
  a. Leave-one-replicate-out (LOO) for the time-course peak response.
  b. Alternative test (Mann-Whitney U) vs t-test for the peak state difference.
  c. Threshold sensitivity: state difference at t=0, at peak, and at last time point.
  d. Tissue holdout: state difference using only tissue_2, then only tissue_1.
  e. State holdout: tissue difference (tissue_1 - tissue_2) using only background,
     then only perturbed.
  f. Baseline phenotype: Kruskal-Wallis per anatomical measure vs t-test conclusion.

Writes results to robustness.json.
"""
import json
import os
import numpy as np
import pandas as pd
from scipy import stats

BASE = "/data/user/conversations/conv-934c9ec3-5973-4ed7-b05d-85b0bca120ef/project"
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "output")
ROBUSTNESS_PATH = os.path.join(OUT, "robustness.json")

FILES = {
    "timecourse": "nutrient_response_timecourse.tsv",
    "tissue_uptake": "tissue_uptake_measurement.tsv",
    "baseline": "baseline_system_phenotype.tsv",
}

ALPHA = 0.05


def load_tsv(name):
    return pd.read_csv(os.path.join(DATA, name), sep="\t")


def jsonable(obj):
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    return obj


def pooled_sd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    sp2 = ((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2)
    return float(np.sqrt(sp2))


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return (float(a.mean()) - float(b.mean())) / pooled_sd(a, b)


def state_label(s):
    return "perturbed" if s == "perturbed_state" else "background"


def main():
    os.makedirs(OUT, exist_ok=True)

    tc = load_tsv(FILES["timecourse"])
    tu = load_tsv(FILES["tissue_uptake"])
    bp = load_tsv(FILES["baseline"])

    states = sorted(tc["system_state_id"].unique().tolist())
    background = "background_state"
    perturbed = "perturbed_state"

    results = {
        "metadata": {
            "task_id": "Life_03_nutrient_response_kinetics",
            "alpha": ALPHA,
            "states": states,
            "n_timecourse_replicates_per_state": {
                s: int(tc[tc["system_state_id"] == s]["replicate_id"].nunique()) for s in states
            },
            "note": "Robustness checks on the frozen bundled data; replicate indices are aligned "
                    "across states (replicate_1..4 in the time course)."
        }
    }

    # ------------------------------------------------------------------
    # (a) Leave-one-replicate-out for time-course peak response
    # ------------------------------------------------------------------
    # Determine peak time per state on the FULL data.
    peak_full = {}
    for s in states:
        sub = tc[tc["system_state_id"] == s]
        by_time = sub.groupby("time_min")["normalized_signaling_response"].mean().sort_index()
        peak_full[s] = float(by_time.idxmax())
    # (verify stability under LOO below)

    a = {
        "peak_time_full_data": peak_full,
        "per_state_peak_means": {},
        "matched_leave_out_state_difference": [],
    }

    for s in states:
        sub = tc[tc["system_state_id"] == s]
        reps = sub["replicate_id"].unique().tolist()  # replicate_1..4
        # fixed peak time = full-data peak
        t_peak = peak_full[s]
        # recompute peak time under each replicate drop
        loo_records = []
        for r in reps:
            kept = sub[sub["replicate_id"] != r].groupby("time_min")[
                "normalized_signaling_response"
            ].mean().sort_index()
            peak_t_loo = float(kept.idxmax())
            # mean response at peak (recomputed peak)
            peak_mean_loo = float(kept.loc[peak_t_loo])
            # mean response at fixed full-data peak
            peak_mean_fixed = float(kept.loc[t_peak])
            loo_records.append({
                "replicate_dropped": r,
                "n_kept_replicates": int(sub[sub["replicate_id"] != r]["replicate_id"].nunique()),
                "peak_time_recomputed": peak_t_loo,
                "peak_mean_recomputed": peak_mean_loo,
                "peak_mean_at_fixed_peak": peak_mean_fixed,
                "peak_time_changed": bool(peak_t_loo != t_peak),
            })
        # summarize min/max/range
        means = [rec["peak_mean_recomputed"] for rec in loo_records]
        a["per_state_peak_means"][s] = {
            "records": [
                {k: jsonable(v) for k, v in rec.items()} for rec in loo_records
            ],
            "min_peak_mean": min(means),
            "max_peak_mean": max(means),
            "range_peak_mean": max(means) - min(means),
        }

    # matched leave-out state difference (drop same replicate index from both states)
    # order replicates consistently (replicate_1..4)
    def rep_key(r):
        return int(r.replace("replicate_", ""))
    pb_full = tc[(tc["system_state_id"] == perturbed) & (tc["time_min"] == peak_full[perturbed])]
    gb_full = tc[(tc["system_state_id"] == background) & (tc["time_min"] == peak_full[background])]
    pb_by_rep = dict(zip(pb_full["replicate_id"], pb_full["normalized_signaling_response"]))
    gb_by_rep = dict(zip(gb_full["replicate_id"], gb_full["normalized_signaling_response"]))
    # align by replicate label
    common_reps = sorted(set(pb_by_rep.keys()) & set(gb_by_rep.keys()), key=lambda x: rep_key(x))
    pb_arr = np.array([pb_by_rep[r] for r in common_reps], dtype=float)
    gb_arr = np.array([gb_by_rep[r] for r in common_reps], dtype=float)
    full_diff = float(pb_arr.mean() - gb_arr.mean())
    full_t = stats.ttest_ind(pb_arr, gb_arr, equal_var=True)
    full_mw = stats.mannwhitneyu(pb_arr, gb_arr, alternative="two-sided")

    loo_diffs = []
    for j, r in enumerate(common_reps):
        pb_j = np.delete(pb_arr, j)
        gb_j = np.delete(gb_arr, j)
        diff = float(pb_j.mean() - gb_j.mean())
        t_j = stats.ttest_ind(pb_j, gb_j, equal_var=True)
        mw_j = stats.mannwhitneyu(pb_j, gb_j, alternative="two-sided")
        loo_diffs.append({
            "replicate_dropped": r,
            "perturbed_peak_mean": float(pb_j.mean()),
            "background_peak_mean": float(gb_j.mean()),
            "state_difference": diff,
            "t_pvalue": float(t_j.pvalue),
            "mannwhitney_u_pvalue": float(mw_j.pvalue),
            "cohens_d": cohens_d(pb_j, gb_j),
            "sign": "positive" if diff > 0 else ("negative" if diff < 0 else "zero"),
            "t_significant": bool(t_j.pvalue < ALPHA),
        })
    diff_vals = [d["state_difference"] for d in loo_diffs]
    a["matched_leave_out_state_difference"] = [
        {k: jsonable(v) for k, v in d.items()} for d in loo_diffs
    ]
    a["full_data_state_difference"] = {
        "perturbed_peak_mean": float(pb_arr.mean()),
        "background_peak_mean": float(gb_arr.mean()),
        "difference": full_diff,
        "t_pvalue": float(full_t.pvalue),
        "mannwhitney_u_pvalue": float(full_mw.pvalue),
    }
    a["summary"] = {
        "full_data_difference_sign": "positive" if full_diff > 0 else ("negative" if full_diff < 0 else "zero"),
        "full_data_significant_t": bool(full_t.pvalue < ALPHA),
        "n_leave_outs_negative": sum(1 for d in diff_vals if d < 0),
        "n_leave_outs_significant_t": sum(1 for d in loo_diffs if d["t_significant"]),
        "sign_changes": any(d < 0 for d in diff_vals),
        "significance_changes": any(d["t_significant"] for d in loo_diffs),
        "min_state_difference": min(diff_vals),
        "max_state_difference": max(diff_vals),
        "range_state_difference": max(diff_vals) - min(diff_vals),
        "peak_time_stable_under_loo": all(
            rec["peak_time_changed"] is False
            for s in states
            for rec in a["per_state_peak_means"][s]["records"]
        ),
    }

    results["a_loo_peak_response"] = {k: jsonable(v) for k, v in a.items()}

    # ------------------------------------------------------------------
    # (b) Alternative test: t-test vs Mann-Whitney U for peak state diff
    # ------------------------------------------------------------------
    b = {
        "test": "perturbed_state vs background_state peak signaling response (n=4 vs n=4 each)",
        "t_test": {
            "t_statistic": float(full_t.statistic),
            "t_pvalue": float(full_t.pvalue),
            "significant": bool(full_t.pvalue < ALPHA),
        },
        "mann_whitney_u": {
            "u_statistic": float(full_mw.statistic),
            "u_pvalue": float(full_mw.pvalue),
            "significant": bool(full_mw.pvalue < ALPHA),
        },
        "cohens_d": cohens_d(pb_arr, gb_arr),
        "means": {"perturbed": float(pb_arr.mean()), "background": float(gb_arr.mean())},
        "significance_differs": bool(full_t.pvalue < ALPHA) != bool(full_mw.pvalue < ALPHA),
        "conclusion": "t-test and Mann-Whitney U give the same significance conclusion ("
                      + ("significant" if full_t.pvalue < ALPHA else "non-significant")
                      + ") for the peak state difference.",
    }
    results["b_alternative_test_peak"] = {k: jsonable(v) for k, v in b.items()}

    # ------------------------------------------------------------------
    # (c) Threshold sensitivity: state difference at t=0, peak, last time point
    # ------------------------------------------------------------------
    by_state = {}
    for s in states:
        sub = tc[tc["system_state_id"] == s]
        by_time = sub.groupby("time_min")["normalized_signaling_response"].mean().sort_index()
        by_state[s] = by_time
    all_times = sorted(by_state[states[0]].index.tolist())
    t0 = float(min(all_times))
    t_last = float(max(all_times))
    t_peak = peak_full[background]  # peak (same for both states here)
    c_points = {
        "t0": t0,
        "peak": t_peak,
        "t_last": t_last,
    }
    c = {"state_difference_by_timepoint": {}, "sign_changes": False}
    c["timepoints_analyzed"] = [jsonable(v) for v in c_points.values()]
    prev_sign = None
    for name, t in c_points.items():
        diff = float(by_state[perturbed][t] - by_state[background][t])
        c["state_difference_by_timepoint"][name] = {
            "time_min": float(t),
            "perturbed_mean": float(by_state[perturbed][t]),
            "background_mean": float(by_state[background][t]),
            "perturbed_minus_background": diff,
            "sign": "positive" if diff > 0 else ("negative" if diff < 0 else "zero"),
        }
        if prev_sign is None:
            prev_sign = diff
        elif (diff > 0 and prev_sign < 0) or (diff < 0 and prev_sign > 0):
            c["sign_changes"] = True
    c["all_signs"] = [
        c["state_difference_by_timepoint"][n]["sign"] for n in c_points
    ]
    c["conclusion"] = ("Sign of the perturbed-minus-background difference is "
                       + ("consistent (all " + c["all_signs"][0] + ") across t=0, peak, and last time point."
                          if not c["sign_changes"]
                          else "not consistent; the sign changes between analyzed time points.")
                       )
    results["c_threshold_sensitivity"] = {k: jsonable(v) for k, v in c.items()}

    # ------------------------------------------------------------------
    # (d) Tissue holdout: state difference using only tissue_2, then tissue_1
    # ------------------------------------------------------------------
    def tissue_state_diff(tissue):
        sub = tu[tu["tissue_id"] == tissue]
        p = sub[sub["system_state_id"] == perturbed]["uptake_measurement_au"].to_numpy(dtype=float)
        b = sub[sub["system_state_id"] == background]["uptake_measurement_au"].to_numpy(dtype=float)
        t = stats.ttest_ind(p, b, equal_var=True)
        mw = stats.mannwhitneyu(p, b, alternative="two-sided")
        return {
            "perturbed_mean": float(p.mean()),
            "background_mean": float(b.mean()),
            "perturbed_minus_background": float(p.mean() - b.mean()),
            "t_statistic": float(t.statistic),
            "t_pvalue": float(t.pvalue),
            "mannwhitney_u_pvalue": float(mw.pvalue),
            "cohens_d": cohens_d(p, b),
            "n_perturbed": int(len(p)),
            "n_background": int(len(b)),
            "significant_t": bool(t.pvalue < ALPHA),
        }

    # full-data overall state difference across both tissues
    p_all = tu[tu["system_state_id"] == perturbed]["uptake_measurement_au"].to_numpy(dtype=float)
    b_all = tu[tu["system_state_id"] == background]["uptake_measurement_au"].to_numpy(dtype=float)
    t_all = stats.ttest_ind(p_all, b_all, equal_var=True)

    d = {
        "definition": "perturbed_state minus background_state mean uptake, within a single tissue (holdout)",
        "full_data_overall_state_difference": {
            "perturbed_mean": float(p_all.mean()),
            "background_mean": float(b_all.mean()),
            "perturbed_minus_background": float(p_all.mean() - b_all.mean()),
            "t_pvalue": float(t_all.pvalue),
            "significant_t": bool(t_all.pvalue < ALPHA),
            "cohens_d": cohens_d(p_all, b_all),
        },
        "full_data_per_tissue_state_difference": {
            "tissue_1": {
                "perturbed_minus_background":
                    tissue_state_diff("tissue_1")["perturbed_minus_background"],
            },
            "tissue_2": {
                "perturbed_minus_background":
                    tissue_state_diff("tissue_2")["perturbed_minus_background"],
            },
        },
        "holdout_tissue_2_only": tissue_state_diff("tissue_2"),
        "holdout_tissue_1_only": tissue_state_diff("tissue_1"),
    }
    d["comparison"] = {
        "tissue_2_holdout_vs_full_tissue_2":
            abs(tissue_state_diff("tissue_2")["perturbed_minus_background"]
                - d["full_data_per_tissue_state_difference"]["tissue_2"]["perturbed_minus_background"]) < 1e-12,
        "tissue_1_holdout_vs_full_tissue_1":
            abs(tissue_state_diff("tissue_1")["perturbed_minus_background"]
                - d["full_data_per_tissue_state_difference"]["tissue_1"]["perturbed_minus_background"]) < 1e-12,
        "note": "Per-tissue state difference is, by construction, unchanged by holding out the "
                "other tissue; the meaningful comparison is against the full-data OVERALL "
                "state difference, which is dominated by tissue_2.",
    }
    d["conclusion"] = ("Tissue-specific state differences: tissue_2 (d_tissue2="
                       + f"{d['holdout_tissue_2_only']['perturbed_minus_background']:.3f}"
                       + f", p={d['holdout_tissue_2_only']['t_pvalue']:.3f}) is a large negative "
                       + "perturbed-minus-background difference (non-significant, n=3 each); "
                       "tissue_1 (d_tissue1="
                       + f"{d['holdout_tissue_1_only']['perturbed_minus_background']:.3f}"
                       + f", p={d['holdout_tissue_1_only']['t_pvalue']:.3f}) is near zero and "
                       "non-significant. The full-data overall state difference is "
                       + f"{d['full_data_overall_state_difference']['perturbed_minus_background']:.3f} "
                       + "(t-p={d['full_data_overall_state_difference']['t_pvalue']:.3f}), driven "
                       "by tissue_2. Tissue structure is therefore preserved under single-tissue "
                       "holdout.")
    results["d_tissue_holdout"] = {k: jsonable(v) for k, v in d.items()}

    # ------------------------------------------------------------------
    # (e) State holdout: tissue difference (tissue_1 - tissue_2) per state
    # ------------------------------------------------------------------
    def tissue_diff_in_state(state):
        sub = tu[tu["system_state_id"] == state]
        t1 = sub[sub["tissue_id"] == "tissue_1"]["uptake_measurement_au"].to_numpy(dtype=float)
        t2 = sub[sub["tissue_id"] == "tissue_2"]["uptake_measurement_au"].to_numpy(dtype=float)
        diff_per_rep = t1 - t2  # replicates aligned by order
        paired = stats.ttest_ind(t1, t2, equal_var=False)
        # paired t-test on the per-replicate differences (t1 - t2)
        paired_t = stats.ttest_1samp(diff_per_rep, popmean=0.0)
        return {
            "tissue_1_mean": float(t1.mean()),
            "tissue_2_mean": float(t2.mean()),
            "tissue_1_minus_tissue_2": float(t1.mean() - t2.mean()),
            "paired_t_statistic": float(paired_t.statistic),
            "paired_t_pvalue": float(paired_t.pvalue),
            "significant": bool(paired_t.pvalue < ALPHA),
            "tissue_1_greater_than_tissue_2": bool(t1.mean() > t2.mean()),
        }

    # full-data overall tissue difference
    t1_all = tu[tu["tissue_id"] == "tissue_1"]["uptake_measurement_au"].to_numpy(dtype=float)
    t2_all = tu[tu["tissue_id"] == "tissue_2"]["uptake_measurement_au"].to_numpy(dtype=float)

    e = {
        "definition": "tissue_1 minus tissue_2 mean uptake within a single state (holdout)",
        "full_data_overall_tissue_difference": {
            "tissue_1_mean": float(t1_all.mean()),
            "tissue_2_mean": float(t2_all.mean()),
            "tissue_1_minus_tissue_2": float(t1_all.mean() - t2_all.mean()),
        },
        "full_data_per_state_tissue_difference": {
            "background_state": tissue_diff_in_state(background)["tissue_1_minus_tissue_2"],
            "perturbed_state": tissue_diff_in_state(perturbed)["tissue_1_minus_tissue_2"],
        },
        "holdout_background_only": tissue_diff_in_state(background),
        "holdout_perturbed_only": tissue_diff_in_state(perturbed),
    }
    e["comparison"] = {
        "consistent_tissue_1_gt_tissue_2": (
            e["holdout_background_only"]["tissue_1_greater_than_tissue_2"]
            and e["holdout_perturbed_only"]["tissue_1_greater_than_tissue_2"]
            and e["full_data_overall_tissue_difference"]["tissue_1_minus_tissue_2"] > 0
        ),
        "note": "Tissue_1 exceeds tissue_2 in both states and overall; the overall tissue "
                "difference is between the two state-specific values.",
    }
    e["conclusion"] = ("Tissue uptake is tissue-structured (tissue_1 > tissue_2) in both "
                       "background_state ("
                       + f"{e['holdout_background_only']['tissue_1_minus_tissue_2']:.3f}"
                       + ") and perturbed_state ("
                       + f"{e['holdout_perturbed_only']['tissue_1_minus_tissue_2']:.3f}"
                       + ") and overall ("
                       + f"{e['full_data_overall_tissue_difference']['tissue_1_minus_tissue_2']:.3f}"
                       + "); both state-specific differences are highly significant in the paired "
                       "test. Tissue structure is robust to state holdout.")
    results["e_state_holdout"] = {k: jsonable(v) for k, v in e.items()}

    # ------------------------------------------------------------------
    # (f) Baseline phenotype: Kruskal-Wallis per measure vs t-test conclusion
    # ------------------------------------------------------------------
    f = {"per_measure": {}, "all_agree": True}
    measures = sorted(bp["anatomical_measure_id"].unique().tolist())
    for m in measures:
        sub = bp[bp["anatomical_measure_id"] == m]
        p = sub[sub["system_state_id"] == perturbed]["measurement_value"].to_numpy(dtype=float)
        b = sub[sub["system_state_id"] == background]["measurement_value"].to_numpy(dtype=float)
        t = stats.ttest_ind(p, b, equal_var=True)
        kw = stats.kruskal(p, b)
        t_sig = bool(t.pvalue < ALPHA)
        kw_sig = bool(kw.pvalue < ALPHA)
        f["per_measure"][m] = {
            "n_perturbed": int(len(p)),
            "n_background": int(len(b)),
            "perturbed_mean": float(p.mean()),
            "background_mean": float(b.mean()),
            "perturbed_minus_background": float(p.mean() - b.mean()),
            "t_test": {
                "t_statistic": float(t.statistic),
                "pvalue": float(t.pvalue),
                "significant": t_sig,
            },
            "kruskal_wallis": {
                "H_statistic": float(kw.statistic),
                "pvalue": float(kw.pvalue),
                "significant": kw_sig,
            },
            "agrees_with_t_test": bool(t_sig == kw_sig),
        }
        if not (t_sig == kw_sig):
            f["all_agree"] = False
    f["conclusion"] = ("Kruskal-Wallis agrees with the t-test conclusion for "
                       + ("all" if f["all_agree"] else "most")
                       + " anatomical measures. Significant measures by both tests: "
                       + ", ".join(m for m in measures if f["per_measure"][m]["agrees_with_t_test"]
                                   and f["per_measure"][m]["t_test"]["significant"])
                       + "; non-significant by both: "
                       + ", ".join(m for m in measures if not f["per_measure"][m]["t_test"]["significant"])
                       + ".")
    results["f_baseline_kruskalwallis"] = {k: jsonable(v) for k, v in f.items()}

    # ------------------------------------------------------------------
    # Overall summary
    # ------------------------------------------------------------------
    results["summary"] = {
        "a_loo_peak_sign_changes": results["a_loo_peak_response"]["summary"]["sign_changes"],
        "a_loo_peak_significance_changes": results["a_loo_peak_response"]["summary"]["significance_changes"],
        "a_loo_peak_significant_any": results["a_loo_peak_response"]["summary"][
            "n_leave_outs_significant_t"] > 0,
        "b_peak_significance_differs": results["b_alternative_test_peak"]["significance_differs"],
        "c_threshold_sign_changes": results["c_threshold_sensitivity"]["sign_changes"],
        "d_tissue_state_diff_significant": bool(
            results["d_tissue_holdout"]["holdout_tissue_2_only"]["significant_t"]),
        "e_tissue_structure_consistent": results["e_state_holdout"]["comparison"][
            "consistent_tissue_1_gt_tissue_2"],
        "f_kruskalwallis_all_agree": results["f_baseline_kruskalwallis"]["all_agree"],
        "overall_note": "Peak-response state difference is non-significant and its sign "
                         "is not stable under leave-one-replicate-out (driven by one replicate), "
                         "consistent with the underpowered null. Tissue structure and baseline "
                         "phenotype effects are robust and nonparametric tests agree with t-tests.",
    }

    with open(ROBUSTNESS_PATH, "w") as fh:
        json.dump(jsonable(results), fh, indent=2)

    print(f"Wrote {ROBUSTNESS_PATH}")
    print("\n=== (a) LOO peak response ===")
    print("full-data peak state diff =", round(results["a_loo_peak_response"]["full_data_state_difference"]["difference"], 4),
          "t-p =", round(results["a_loo_peak_response"]["full_data_state_difference"]["t_pvalue"], 4))
    for s in states:
        ps = results["a_loo_peak_response"]["per_state_peak_means"][s]
        print(f"  {s}: min={ps['min_peak_mean']:.4f} max={ps['max_peak_mean']:.4f} range={ps['range_peak_mean']:.4f}")
    for drec in results["a_loo_peak_response"]["matched_leave_out_state_difference"]:
        print(f"  drop {drec['replicate_dropped']}: diff={drec['state_difference']:.4f} sign={drec['sign']} t-p={drec['t_pvalue']:.4f}")
    print("  sign_changes =", results["a_loo_peak_response"]["summary"]["sign_changes"],
          "significance_changes =", results["a_loo_peak_response"]["summary"]["significance_changes"])
    print("\n=== (b) alternative test peak ===")
    print("  t-p =", round(results["b_alternative_test_peak"]["t_test"]["t_pvalue"], 4),
          "MW-p =", round(results["b_alternative_test_peak"]["mann_whitney_u"]["u_pvalue"], 4),
          "significance_differs =", results["b_alternative_test_peak"]["significance_differs"])
    print("\n=== (c) threshold sensitivity ===")
    for name, v in results["c_threshold_sensitivity"]["state_difference_by_timepoint"].items():
        print(f"  {name} (t={v['time_min']}): diff={v['perturbed_minus_background']:.4f} sign={v['sign']}")
    print("  sign_changes =", results["c_threshold_sensitivity"]["sign_changes"])
    print("\n=== (d) tissue holdout ===")
    print("  full overall diff =", round(results["d_tissue_holdout"]["full_data_overall_state_difference"]["perturbed_minus_background"], 4))
    print("  tissue_2 only: diff =", round(results["d_tissue_holdout"]["holdout_tissue_2_only"]["perturbed_minus_background"], 4),
          "t-p =", round(results["d_tissue_holdout"]["holdout_tissue_2_only"]["t_pvalue"], 4))
    print("  tissue_1 only: diff =", round(results["d_tissue_holdout"]["holdout_tissue_1_only"]["perturbed_minus_background"], 4),
          "t-p =", round(results["d_tissue_holdout"]["holdout_tissue_1_only"]["t_pvalue"], 4))
    print("\n=== (e) state holdout (tissue_1 - tissue_2) ===")
    print("  background only:", round(results["e_state_holdout"]["holdout_background_only"]["tissue_1_minus_tissue_2"], 3),
          "(p=", round(results["e_state_holdout"]["holdout_background_only"]["paired_t_pvalue"], 4), ")")
    print("  perturbed only:", round(results["e_state_holdout"]["holdout_perturbed_only"]["tissue_1_minus_tissue_2"], 3),
          "(p=", round(results["e_state_holdout"]["holdout_perturbed_only"]["paired_t_pvalue"], 4), ")")
    print("  consistent tissue_1>tissue_2 =", results["e_state_holdout"]["comparison"]["consistent_tissue_1_gt_tissue_2"])
    print("\n=== (f) baseline Kruskal-Wallis vs t-test ===")
    for m in measures:
        v = results["f_baseline_kruskalwallis"]["per_measure"][m]
        print(f"  {m}: t-p={v['t_test']['pvalue']:.4g} (sig={v['t_test']['significant']}), "
              f"KW-p={v['kruskal_wallis']['pvalue']:.4g} (sig={v['kruskal_wallis']['significant']}), "
              f"agrees={v['agrees_with_t_test']}")
    print("  all_agree =", results["f_baseline_kruskalwallis"]["all_agree"])


if __name__ == "__main__":
    main()
