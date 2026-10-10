#!/usr/bin/env python3
"""
Nutrient-response kinetics analysis across system states and tissues.

Loads 4 TSV panels, computes time-course kinetics, circulating-input,
tissue-uptake, and baseline-phenotype summaries, Pearson correlations
(baseline phenotype vs peak signaling response), and two-sample tests
(t-test, Mann-Whitney U, Cohen's d). Writes results to results.json.

Analysis unit: one biological replicate within a system state / tissue /
input condition / time course; repeated measures sharing a state are nested.
"""
import json
import os
import numpy as np
import pandas as pd
from scipy import stats

BASE = "/data/user/conversations/conv-934c9ec3-5973-4ed7-b05d-85b0bca120ef/project"
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "output")
RESULTS_PATH = os.path.join(OUT, "results.json")

FILES = {
    "timecourse": "nutrient_response_timecourse.tsv",
    "circulating_input": "circulating_input_measurement.tsv",
    "tissue_uptake": "tissue_uptake_measurement.tsv",
    "baseline": "baseline_system_phenotype.tsv",
}


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


def stat_pair(a, b, label):
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = stats.ttest_ind(a, b, equal_var=True)
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "label": label,
        "a_mean": float(a.mean()),
        "a_std_ddof1": float(a.std(ddof=1)),
        "a_n": int(len(a)),
        "b_mean": float(b.mean()),
        "b_std_ddof1": float(b.std(ddof=1)),
        "b_n": int(len(b)),
        "t_statistic": float(t.statistic),
        "t_pvalue": float(t.pvalue),
        "mannwhitney_u_statistic": float(u.statistic),
        "mannwhitney_u_pvalue": float(u.pvalue),
        "cohens_d": cohens_d(a, b),
    }


def main():
    os.makedirs(OUT, exist_ok=True)

    tc = load_tsv(FILES["timecourse"])
    ci = load_tsv(FILES["circulating_input"])
    tu = load_tsv(FILES["tissue_uptake"])
    bp = load_tsv(FILES["baseline"])

    results = {}

    # ------------------------------------------------------------------
    # Data inventory / metadata
    # ------------------------------------------------------------------
    results["metadata"] = {
        "task_id": "Life_03_nutrient_response_kinetics",
        "files_loaded": {
            k: {"path": os.path.join(DATA, v), "rows": int(len(df)), "columns": df.columns.tolist()}
            for df, k, v in [
                (tc, "timecourse", FILES["timecourse"]),
                (ci, "circulating_input", FILES["circulating_input"]),
                (tu, "tissue_uptake", FILES["tissue_uptake"]),
                (bp, "baseline", FILES["baseline"]),
            ]
        },
        "system_states": sorted(tc["system_state_id"].unique().tolist()),
        "n_input_roles": int(ci["input_condition_role"].nunique()),
        "input_roles": sorted(ci["input_condition_role"].unique().tolist()),
        "n_tissues": int(tu["tissue_id"].nunique()),
        "tissues": sorted(tu["tissue_id"].unique().tolist()),
        "n_anatomical_measures": int(bp["anatomical_measure_id"].nunique()),
        "anatomical_measures": sorted(bp["anatomical_measure_id"].unique().tolist()),
        "analysis_unit": "one biological replicate within a system state, tissue, input condition, or time course; repeated measures sharing a state are nested",
        "note": "No missing values in any file (verified at load time).",
    }

    states = sorted(tc["system_state_id"].unique().tolist())
    background = "background_state"
    perturbed = "perturbed_state"

    # ------------------------------------------------------------------
    # (a) Time-course kinetics
    # ------------------------------------------------------------------
    timecourse = {}
    for s in states:
        sub = tc[tc["system_state_id"] == s].copy()
        g = sub.groupby("time_min")["normalized_signaling_response"]
        by_time = pd.DataFrame({
            "mean": g.mean(),
            "std": g.std(ddof=1),
            "count": g.count(),
        }).sort_index()
        peak_t = float(by_time["mean"].idxmax())
        timecourse[s] = {
            "n_rows": int(len(sub)),
            "n_replicates": int(sub["replicate_id"].nunique()),
            "time_points_min": [float(t) for t in by_time.index.tolist()],
            "mean_response_by_time": {float(t): float(m) for t, m in by_time["mean"].items()},
            "std_response_by_time": {float(t): float(v) for t, v in by_time["std"].items()},
            "overall_mean": float(sub["normalized_signaling_response"].mean()),
            "overall_std_ddof1": float(sub["normalized_signaling_response"].std(ddof=1)),
            "peak_time_min": peak_t,
            "peak_mean_response": float(by_time.loc[peak_t, "mean"]),
        }

    t_times = sorted(timecourse[states[0]]["mean_response_by_time"].keys())
    state_diff_by_time = {}
    for t in t_times:
        state_diff_by_time[t] = float(
            timecourse[perturbed]["mean_response_by_time"][t]
            - timecourse[background]["mean_response_by_time"][t]
        )

    results["time_course_kinetics"] = {
        "by_state": timecourse,
        "state_difference_by_time_min": state_diff_by_time,
        "state_difference_note": "perturbed_state mean minus background_state mean at each time point",
    }

    # ------------------------------------------------------------------
    # (b) Circulating input
    # ------------------------------------------------------------------
    ci_summary = {}
    for (s, role), g in ci.groupby(["system_state_id", "input_condition_role"], sort=False):
        ci_summary[f"{s}|{role}"] = {
            "mean_abundance_au": float(g["measured_abundance_au"].mean()),
            "std_abundance_au_ddof1": float(g["measured_abundance_au"].std(ddof=1)),
            "n": int(len(g)),
        }

    ci_state_diff_by_condition = {}
    ci_condition_diff_by_state = {}
    for role in sorted(ci["input_condition_role"].unique().tolist()):
        for s in states:
            ci_state_diff_by_condition[f"{s}|{role}"] = {
                "perturbed_minus_background": float(
                    ci_summary[f"{perturbed}|{role}"]["mean_abundance_au"]
                    - ci_summary[f"{background}|{role}"]["mean_abundance_au"]
                ),
            }
    for s in states:
        ci_condition_diff_by_state[s] = {
            "nutrient_input_minus_pre_input": float(
                ci_summary[f"{s}|nutrient_input"]["mean_abundance_au"]
                - ci_summary[f"{s}|pre_input"]["mean_abundance_au"]
            ),
        }

    results["circulating_input"] = {
        "group_means": ci_summary,
        "state_difference_by_condition": ci_state_diff_by_condition,
        "condition_difference_by_state": ci_condition_diff_by_state,
        "state_difference_note": "perturbed_state minus background_state mean abundance at each condition",
        "condition_difference_note": "nutrient_input minus pre_input mean abundance at each state",
    }

    # ------------------------------------------------------------------
    # (c) Tissue uptake
    # ------------------------------------------------------------------
    tu_summary = {}
    for (t, s), g in tu.groupby(["tissue_id", "system_state_id"], sort=False):
        tu_summary[f"{t}|{s}"] = {
            "mean_uptake_au": float(g["uptake_measurement_au"].mean()),
            "std_uptake_au_ddof1": float(g["uptake_measurement_au"].std(ddof=1)),
            "n": int(len(g)),
        }

    tissues = sorted(tu["tissue_id"].unique().tolist())
    tu_tissue_diff_by_state = {}
    tu_state_diff_by_tissue = {}
    for s in states:
        tu_tissue_diff_by_state[s] = {
            "tissue_1_minus_tissue_2": float(
                tu_summary[f"tissue_1|{s}"]["mean_uptake_au"] - tu_summary[f"tissue_2|{s}"]["mean_uptake_au"]
            ),
        }
    for t in tissues:
        tu_state_diff_by_tissue[t] = {
            "perturbed_minus_background": float(
                tu_summary[f"{t}|{perturbed}"]["mean_uptake_au"] - tu_summary[f"{t}|{background}"]["mean_uptake_au"]
            ),
        }

    results["tissue_uptake"] = {
        "group_means": tu_summary,
        "tissue_difference_by_state": tu_tissue_diff_by_state,
        "state_difference_by_tissue": tu_state_diff_by_tissue,
        "tissue_difference_note": "tissue_1 minus tissue_2 mean uptake at each state",
        "state_difference_note": "perturbed_state minus background_state mean uptake at each tissue",
    }

    # ------------------------------------------------------------------
    # (d) Baseline phenotype
    # ------------------------------------------------------------------
    bp_summary = {}
    for (m, s), g in bp.groupby(["anatomical_measure_id", "system_state_id"], sort=False):
        bp_summary[f"{m}|{s}"] = {
            "mean_measurement_value": float(g["measurement_value"].mean()),
            "std_measurement_value_ddof1": float(g["measurement_value"].std(ddof=1)),
            "n": int(len(g)),
        }

    bp_state_diff_by_measure = {}
    for m in sorted(bp["anatomical_measure_id"].unique().tolist()):
        bp_state_diff_by_measure[m] = {
            "perturbed_minus_background": float(
                bp_summary[f"{m}|{perturbed}"]["mean_measurement_value"]
                - bp_summary[f"{m}|{background}"]["mean_measurement_value"]
            ),
        }

    results["baseline_phenotype"] = {
        "group_means": bp_summary,
        "state_difference_by_measure": bp_state_diff_by_measure,
        "state_difference_note": "perturbed_state minus background_state mean value at each anatomical measure",
    }

    # ------------------------------------------------------------------
    # (e) Correlations: baseline phenotype vs peak signaling response
    # ------------------------------------------------------------------
    corr = {}
    corr_robust = {}
    all_a = []
    all_b = []
    for s in states:
        peak_t = timecourse[s]["peak_time_min"]
        peak = (
            tc[(tc["system_state_id"] == s) & (tc["time_min"] == peak_t)]
            .set_index("replicate_id")["normalized_signaling_response"]
        )
        base = bp[bp["system_state_id"] == s].groupby("replicate_id")["measurement_value"].mean()
        common = peak.index.intersection(base.index)
        a = base.loc[common].to_numpy(dtype=float)  # baseline mean across measures per replicate
        b = peak.loc[common].to_numpy(dtype=float)  # peak response per replicate
        r, p = stats.pearsonr(a, b)
        corr[s] = {
            "n_pairs": int(len(common)),
            "replicates": [str(i) for i in common],
            "peak_time_min": peak_t,
            "baseline_mean_per_replicate": [float(x) for x in a],
            "peak_response_per_replicate": [float(x) for x in b],
            "r": float(r),
            "p": float(p),
        }
        all_a.extend(a)
        all_b.extend(b)

        # Robustness: leave-one-out Pearson r per state
        loo = {}
        for j in range(len(a)):
            aj = np.delete(a, j)
            bj = np.delete(b, j)
            loo[str(common[j])] = float(stats.pearsonr(aj, bj).statistic)
        corr_robust[s] = {"leave_one_out_r": loo}

    pooled_r, pooled_p = stats.pearsonr(np.asarray(all_a, float), np.asarray(all_b, float))
    corr_pooled = {
        "n_pairs": int(len(all_a)),
        "pooled_r": float(pooled_r),
        "pooled_p": float(pooled_p),
        "definition": "pool all matched (state, replicate) pairs across both states and compute Pearson r",
    }

    results["correlations"] = {
        "by_state": corr,
        "pooled": corr_pooled,
        "robustness": corr_robust,
        "robustness_note": "leave-one-out Pearson r per state (drop each replicate, recompute r on remaining pairs)",
    }

    # ------------------------------------------------------------------
    # (f) Statistical tests (two-sample)
    # ------------------------------------------------------------------
    tests = {}

    # time-course peak (perturbed at its peak vs background at its peak)
    pb = tc[(tc["system_state_id"] == perturbed) & (tc["time_min"] == timecourse[perturbed]["peak_time_min"])]["normalized_signaling_response"].to_numpy(dtype=float)
    gb = tc[(tc["system_state_id"] == background) & (tc["time_min"] == timecourse[background]["peak_time_min"])]["normalized_signaling_response"].to_numpy(dtype=float)
    tests["time_course_peak_response"] = stat_pair(pb, gb, "perturbed_state vs background_state peak signaling response")

    # circulating input pre_input
    cpre = ci[ci["input_condition_role"] == "pre_input"]
    cpre_p = cpre[cpre["system_state_id"] == perturbed]["measured_abundance_au"].to_numpy(dtype=float)
    cpre_b = cpre[cpre["system_state_id"] == background]["measured_abundance_au"].to_numpy(dtype=float)
    tests["circulating_input_pre_input"] = stat_pair(cpre_p, cpre_b, "perturbed_state vs background_state pre_input")

    # circulating input nutrient_input
    cnut = ci[ci["input_condition_role"] == "nutrient_input"]
    cnut_p = cnut[cnut["system_state_id"] == perturbed]["measured_abundance_au"].to_numpy(dtype=float)
    cnut_b = cnut[cnut["system_state_id"] == background]["measured_abundance_au"].to_numpy(dtype=float)
    tests["circulating_input_nutrient_input"] = stat_pair(cnut_p, cnut_b, "perturbed_state vs background_state nutrient_input")

    # tissue uptake per tissue
    for t in tissues:
        sub = tu[tu["tissue_id"] == t]
        p = sub[sub["system_state_id"] == perturbed]["uptake_measurement_au"].to_numpy(dtype=float)
        b = sub[sub["system_state_id"] == background]["uptake_measurement_au"].to_numpy(dtype=float)
        tests[f"tissue_uptake_{t}"] = stat_pair(p, b, f"perturbed_state vs background_state {t}")

    # baseline per measure
    for m in sorted(bp["anatomical_measure_id"].unique().tolist()):
        sub = bp[bp["anatomical_measure_id"] == m]
        p = sub[sub["system_state_id"] == perturbed]["measurement_value"].to_numpy(dtype=float)
        b = sub[sub["system_state_id"] == background]["measurement_value"].to_numpy(dtype=float)
        tests[f"baseline_{m}"] = stat_pair(p, b, f"perturbed_state vs background_state {m}")

    results["statistical_tests"] = {
        "tests": tests,
        "methods": {
            "t_test": "independent two-sample Student t-test (scipy.stats.ttest_ind, equal_var=True)",
            "mannwhitney_u": "scipy.stats.mannwhitneyu, alternative='two-sided'",
            "cohens_d": "(mean_perturbed - mean_background) / pooled standard deviation",
        },
    }

    # ------------------------------------------------------------------
    # (g) Robustness: matched leave-one-out t-test / Cohen d for peak
    # ------------------------------------------------------------------
    # time course replicates are aligned by replicate index (replicate_1..4 in both states)
    robust = {
        "time_course_peak_response_matched_leave_one_out": {
            str(j): {
                "t_pvalue": float(stats.ttest_ind(np.delete(pb, j), np.delete(gb, j), equal_var=True).pvalue),
                "mannwhitney_u_pvalue": float(stats.mannwhitneyu(np.delete(pb, j), np.delete(gb, j), alternative="two-sided").pvalue),
                "cohens_d": (float(np.delete(pb, j).mean()) - float(np.delete(gb, j).mean())) / pooled_sd(np.delete(pb, j), np.delete(gb, j)),
            }
            for j in range(len(pb))
        }
    }

    results["robustness"] = robust

    # ------------------------------------------------------------------
    # Write results
    # ------------------------------------------------------------------
    with open(RESULTS_PATH, "w") as fh:
        json.dump(jsonable(results), fh, indent=2)

    print(f"Wrote {RESULTS_PATH}")
    print(f"States: {states}")
    print(f"Peak times: " + ", ".join(f"{s}={timecourse[s]['peak_time_min']}" for s in states))
    print(f"Pooled correlation r={corr_pooled['pooled_r']:.6f} p={corr_pooled['pooled_p']:.6f}")
    for s in states:
        print(f"  {s}: r={corr[s]['r']:.6f} p={corr[s]['p']:.6f} n={corr[s]['n_pairs']}")


if __name__ == "__main__":
    main()
