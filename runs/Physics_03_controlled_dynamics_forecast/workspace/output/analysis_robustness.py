import json
import numpy as np
import pandas as pd

OUT = "/data/user/conversations/conv-ef9804c2-78a8-407c-8722-19f66e9fbe2d/project/output"
DATA = "/data/user/conversations/conv-ef9804c2-78a8-407c-8722-19f66e9fbe2d/project/data"
NV = DATA + "/normalized_views"

with open(OUT + "/frequency_results.json") as f:
    freq_res = json.load(f)
with open(OUT + "/trajectory_results.json") as f:
    traj_res = json.load(f)

# ------------------------------------------------------------------
# 1) cross_channel
# ------------------------------------------------------------------
freq_conds = ["control_a", "control_b", "signed_control"]
freq_mae = {c: float(freq_res[c]["mean_abs_error"]) for c in freq_conds}
freq_exact = {c: int(freq_res[c]["exact_match_count"]) for c in freq_conds}
freq_n = {c: int(freq_res[c]["n_pairs"]) for c in freq_conds}

freq_over_mean = float(np.mean(list(freq_mae.values())))
freq_over_max = float(np.max(list(freq_mae.values())))
freq_best = min(freq_mae, key=freq_mae.get)
freq_worst = max(freq_mae, key=freq_mae.get)

traj_conds = ["condition_01", "condition_02", "condition_03"]
traj_rmse = {c: float(traj_res[c]["rmse"]) for c in traj_conds}
traj_mae = {c: float(traj_res[c]["max_abs_error"]) for c in traj_conds}

rmse_over_mean = float(np.mean(list(traj_rmse.values())))
rmse_over_max = float(np.max(list(traj_rmse.values())))
rmse_best = min(traj_rmse, key=traj_rmse.get)
rmse_worst = max(traj_rmse, key=traj_rmse.get)

mae_over_mean = float(np.mean(list(traj_mae.values())))
mae_over_max = float(np.max(list(traj_mae.values())))
mae_best = min(traj_mae, key=traj_mae.get)
mae_worst = max(traj_mae, key=traj_mae.get)

# magnitude-scale comparison
ratio_rmse_mean = rmse_over_mean / freq_over_mean
ratio_mae_max = mae_over_max / freq_over_max

if ratio_rmse_mean > 3.0 or ratio_mae_max > 3.0:
    verdict = "CONFLICT"
    note = ("Time-channel errors are ~%.1fx larger (rmse mean %.4f vs freq mean_abs_error %.4f) "
            "and max_abs_error reaches %.4f vs freq max %.4f. Frequency channel shows near-exact "
            "matches (control_a %d/%d, control_b %d/%d, signed_control %d/%d), while time channel "
            "shows substantial divergence. The two channels do NOT agree on overall quality."
            % (ratio_rmse_mean, rmse_over_mean, freq_over_mean, mae_over_max, freq_over_max,
               freq_exact["control_a"], freq_n["control_a"],
               freq_exact["control_b"], freq_n["control_b"],
               freq_exact["signed_control"], freq_n["signed_control"]))
else:
    verdict = "AGREE"
    note = ("Magnitude scales are comparable (rmse mean %.4f vs freq mean_abs_error %.4f)."
            % (rmse_over_mean, freq_over_mean))

cross_channel = {
    "frequency": {
        "mean_abs_error_by_condition": freq_mae,
        "overall_mean": freq_over_mean,
        "overall_max": freq_over_max,
        "best_condition": freq_best,
        "worst_condition": freq_worst,
    },
    "time": {
        "rmse_by_condition": traj_rmse,
        "max_abs_error_by_condition": traj_mae,
        "rmse_overall_mean": rmse_over_mean,
        "rmse_overall_max": rmse_over_max,
        "rmse_best_condition": rmse_best,
        "rmse_worst_condition": rmse_worst,
        "max_abs_error_overall_mean": mae_over_mean,
        "max_abs_error_overall_max": mae_over_max,
        "max_abs_error_best_condition": mae_best,
        "max_abs_error_worst_condition": mae_worst,
    },
    "magnitude_comparison": {
        "ratio_time_rmse_mean_to_freq_mae": ratio_rmse_mean,
        "ratio_time_max_abs_max_to_freq_mae_max": ratio_mae_max,
    },
    "verdict": verdict,
    "explanation": note,
}

# ------------------------------------------------------------------
# 2) leave_one_condition_out (TIME domain) - re-read raw TSVs
# ------------------------------------------------------------------
traj_files = {
    "condition_01": NV + "/trajectory_condition_01.tsv",
    "condition_02": NV + "/trajectory_condition_02.tsv",
    "condition_03": NV + "/trajectory_condition_03.tsv",
}
raw = {}
for c, path in traj_files.items():
    df = pd.read_csv(path, sep="\t")
    err = (df["reference_response"].to_numpy() - df["forecast_response"].to_numpy())
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.max(np.abs(err)))
    raw[c] = {"rmse": rmse, "max_abs_error": mae, "n_points": int(len(df))}

global_rank_rmse = [c for c, _ in sorted(raw.items(), key=lambda kv: kv[1]["rmse"])]
global_rank_mae = [c for c, _ in sorted(raw.items(), key=lambda kv: kv[1]["max_abs_error"])]

leave_out = {}
for c_out in traj_conds:
    remaining = [c for c in traj_conds if c != c_out]
    rem_rmse = float(np.mean([raw[c]["rmse"] for c in remaining]))
    rem_mae = float(np.max([raw[c]["max_abs_error"] for c in remaining]))
    rem_rank_rmse = [c for c in sorted(remaining, key=lambda c2: raw[c2]["rmse"])]
    rem_rank_mae = [c for c in sorted(remaining, key=lambda c2: raw[c2]["max_abs_error"])]
    global_restricted_rmse = [c for c in global_rank_rmse if c in remaining]
    global_restricted_mae = [c for c in global_rank_mae if c in remaining]
    leave_out[c_out] = {
        "remaining_conditions": remaining,
        "mean_rmse": rem_rmse,
        "max_abs_error": rem_mae,
        "remaining_ranking_by_rmse": rem_rank_rmse,
        "global_restricted_ranking_by_rmse": global_restricted_rmse,
        "rmse_ranking_stable": rem_rank_rmse == global_restricted_rmse,
        "remaining_ranking_by_max_abs_error": rem_rank_mae,
        "global_restricted_ranking_by_max_abs_error": global_restricted_mae,
        "max_abs_error_ranking_stable": rem_rank_mae == global_restricted_mae,
    }

all_rmse_stable = all(leave_out[c]["rmse_ranking_stable"] for c in traj_conds)
all_mae_stable = all(leave_out[c]["max_abs_error_ranking_stable"] for c in traj_conds)
loco = {
    "per_condition_from_raw_tsv": raw,
    "global_ranking_by_rmse": global_rank_rmse,
    "global_ranking_by_max_abs_error": global_rank_mae,
    "leave_one_out": leave_out,
    "per_condition_ranking_changes": (not all_rmse_stable),
    "per_condition_ranking_changes_by_max_abs_error": (not all_mae_stable),
}

# ------------------------------------------------------------------
# 3) threshold_sensitivity (FREQ) - re-read raw TSVs
# ------------------------------------------------------------------
freq_files = {
    "control_a": NV + "/control_a_frequency_response.tsv",
    "control_b": NV + "/control_b_frequency_response.tsv",
    "signed_control": NV + "/signed_control_frequency_response.tsv",
}
thresholds = [1e-6, 1e-3, 1e-1]
thr_keys = ["1e-6", "1e-3", "1e-1"]
threshold_sens = {}
for c, path in freq_files.items():
    df = pd.read_csv(path, sep="\t")
    diff = np.abs(df["reference_frequency"].to_numpy() - df["forecast_frequency"].to_numpy())
    n = int(len(df))
    row = {"n_pairs": n}
    for t, k in zip(thresholds, thr_keys):
        row[k] = int(np.sum(diff <= t))
    threshold_sens[c] = row

# ------------------------------------------------------------------
# 4) signed_control comparison
# ------------------------------------------------------------------
sc_mre = float(freq_res["signed_control"]["mean_relative_error"])
ca_mre = float(freq_res["control_a"]["mean_relative_error"])
cb_mre = float(freq_res["control_b"]["mean_relative_error"])
avg_ca_b = (ca_mre + cb_mre) / 2.0
diff_sc = sc_mre - avg_ca_b

# recompute from raw TSVs to verify (mean |fc-ref|/ref over all pairs)
def recompute_mre(path):
    df = pd.read_csv(path, sep="\t")
    ref = df["reference_frequency"].to_numpy()
    fc = df["forecast_frequency"].to_numpy()
    return float(np.mean(np.abs(fc - ref) / ref))

sc_mre_raw = recompute_mre(freq_files["signed_control"])
ca_mre_raw = recompute_mre(freq_files["control_a"])
cb_mre_raw = recompute_mre(freq_files["control_b"])

signed_ctrl = {
    "signed_control_mean_relative_error": sc_mre,
    "control_a_mean_relative_error": ca_mre,
    "control_b_mean_relative_error": cb_mre,
    "avg_control_a_b": avg_ca_b,
    "signed_control_minus_avg": diff_sc,
    "signed_control_is_better_or_worse": ("below average (closer to 0)" if diff_sc < 0 else "above average (larger)"),
    "recomputed_from_raw_tsv": {
        "signed_control": sc_mre_raw,
        "control_a": ca_mre_raw,
        "control_b": cb_mre_raw,
        "avg_control_a_b": (ca_mre_raw + cb_mre_raw) / 2.0,
    },
}

# ------------------------------------------------------------------
# assemble + write
# ------------------------------------------------------------------
results = {
    "cross_channel": cross_channel,
    "leave_one_condition_out": loco,
    "threshold_sensitivity": threshold_sens,
    "signed_control_comparison": signed_ctrl,
}

with open(OUT + "/robustness_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Wrote robustness_results.json")
print("cross_channel verdict:", verdict)
print("leave_one_condition_out ranking changes (rmse):", not all_rmse_stable,
      "(max_abs):", not all_mae_stable)
print("signed_control mre:", sc_mre, "avg(ca,cb):", avg_ca_b, "diff:", diff_sc)
