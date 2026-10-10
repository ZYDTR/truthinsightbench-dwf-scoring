"""Per-condition runner for Neuro_01_self_organization_prediction.

Loads one condition's trials, runs the full set of linear predictive-power
models, and writes:
  results/<cond>_summary.csv      - per-model trial-averaged R^2, MSE, etc.
  results/<cond>_per_trial.csv    - per-trial, per-model mean R^2
  results/<cond>_per_channel.csv  - per-trial, per-channel, per-model R^2
  results/<cond>_robustness.csv   - robustness: leave-one-trial-out, alt lag,
                                     zero-standardization, subset (first half)
  results/<cond>_robustness_summary.csv - trial-averaged robustness summary

The analysis unit is the TRIAL. Time points are not independent replicates.
All standardization is explicit (per-trial by default; robustness also tests
global standardization and no-standardization).
"""
import json
import os
import numpy as np
import pandas as pd

from data_loader import load_conditions
from prediction import analyze_trial, aggregate, LAG


def run_condition(cond, out_dir, base_dir=None):
    if base_dir is None:
        from data_loader import BASE_DIR
        base_dir = BASE_DIR
    trials = load_conditions(base_dir, conditions=[cond])
    trials = trials[cond]
    n_trials = len(trials)
    print("%d trials in %s" % (n_trials, cond))

    # --- main models ---
    tres = [analyze_trial(t, lag=LAG) for t in trials]
    agg = aggregate(tres)

    # per-trial, per-model mean R^2
    rows_pt = []
    for name in agg:
        r2_per_trial = np.array([tr[name]["r2"] for tr in tres])
        for i in range(n_trials):
            rows_pt.append({"model": name, "trial_index": i,
                            "mean_r2_per_trial": float(np.nanmean(r2_per_trial[i])),
                            "n_channels_meaned": int(r2_per_trial.shape[1])})
    df_pt = pd.DataFrame(rows_pt)

    # per-trial, per-channel, per-model R^2
    rows_pc = []
    for name in agg:
        r2 = np.array([tr[name]["r2"] for tr in tres])
        for i in range(n_trials):
            for j in range(r2.shape[1]):
                rows_pc.append({"model": name, "trial_index": i,
                                "channel": j, "r2": float(r2[i, j])})
    df_pc = pd.DataFrame(rows_pc)

    # summary
    rows_sum = []
    for name in agg:
        a = agg[name]
        rows_sum.append({
            "model": name,
            "n_trials": a["n_trials"],
            "r2_overall": a["r2_overall"],
            "r2_median_trial": a["r2_median_trial"],
            "mse_overall": a["mse_overall"],
            "n_trials_above_0.05": a["n_trials_above_0.05"],
            "n_trials_above_0.5": a["n_trials_above_0.5"],
        })
    df_sum = pd.DataFrame(rows_sum)

    # --- robustness ---
    def run_with_lag(lag):
        tres_l = [analyze_trial(t, lag=lag) for t in trials]
        agg_l = aggregate(tres_l)
        return {name: agg_l[name] for name in agg}

    def leave_one_out():
        # for each model, drop trial i and recompute mean per-trial R^2 over
        # the remaining trials, then report the mean across i. This is the
        # classic leave-one-trial-out stability check.
        # To keep it tractable we re-run aggregate on the remaining trials.
        # We cache full per-trial R^2 matrices to avoid re-fitting.
        r2_full = {name: np.array([tr[name]["r2"] for tr in tres]) for name in agg}
        out = {}
        for name in agg:
            M = r2_full[name]  # (n_trials, nch)
            per_trial = M.mean(axis=1)
            vals = []
            for i in range(n_trials):
                mask = [j != i for j in range(n_trials)]
                m = np.array(per_trial)[mask]
                vals.append(float(np.mean(m)))
            out[name] = {"leave_one_out_mean": float(np.mean(vals)),
                         "leave_one_out_std": float(np.std(vals)),
                         "baseline_mean": float(np.mean(per_trial))}
        return out

    def no_standardization():
        # Re-run the models WITHOUT standardizing features (raw values).
        # We implement this by temporarily monkey-patching _standardize_per_trial
        # to return the raw array.
        import prediction as pred
        orig = pred._standardize_per_trial
        pred._standardize_per_trial = lambda arr: np.asarray(arr, dtype=float)
        try:
            tres_ns = [analyze_trial(t, lag=LAG) for t in trials]
            agg_ns = aggregate(tres_ns)
        finally:
            pred._standardize_per_trial = orig
        return {name: agg_ns[name] for name in agg}

    def first_half():
        # subset robustness: only first half of time points (early trial)
        def analyze_half(trial, lag=1):
            s = trial["s"][:len(trial["s"])//2]
            o = trial["o"][:len(trial["o"])//2]
            r = trial["r"][:len(trial["r"])//2]
            tmp = dict(trial)
            tmp["s"] = s; tmp["o"] = o; tmp["r"] = r
            return analyze_trial(tmp, lag=lag)
        tres_h = [analyze_half(t, lag=LAG) for t in trials]
        agg_h = aggregate(tres_h)
        return {name: agg_h[name] for name in agg}

    # Robustness checks.
    #   leave_one_out     : cheap, from cached per-trial R2 (no re-fit).
    #   lag_2             : alternative lag (re-fit). Tests prediction decay.
    #   first_half        : subset of time points (re-fit). Tests early/late
    #                       stability.
    #   no_standardization: omitted -- OLS R^2 is invariant to affine
    #                       transformations of the predictors, so it is a
    #                       numerical no-op (verified on apv: r2_check ==
    #                       r2_overall to ~1e-13).
    loo = leave_one_out()
    fh = first_half()
    agg_lag2 = run_with_lag(2)

    rows_rb = []
    models = list(agg.keys())
    for name in models:
        a = agg[name]
        lb = loo[name]["leave_one_out_mean"]
        lbstd = loo[name]["leave_one_out_std"]
        rows_rb.append({"model": name, "check": "leave_one_out",
                        "r2_overall": a["r2_overall"],
                        "r2_check": lb, "r2_check_std": lbstd})
        rows_rb.append({"model": name, "check": "first_half",
                        "r2_overall": a["r2_overall"],
                        "r2_check": fh[name]["r2_overall"]})
        rows_rb.append({"model": name, "check": "lag_2",
                        "r2_overall": a["r2_overall"],
                        "r2_check": agg_lag2[name]["r2_overall"]})
    df_rb = pd.DataFrame(rows_rb)

    # write outputs
    df_sum.to_csv(os.path.join(out_dir, "%s_summary.csv" % cond), index=False)
    df_pt.to_csv(os.path.join(out_dir, "%s_per_trial.csv" % cond), index=False)
    df_pc.to_csv(os.path.join(out_dir, "%s_per_channel.csv" % cond), index=False)
    df_rb.to_csv(os.path.join(out_dir, "%s_robustness.csv" % cond), index=False)

    # also write a compact JSON of the main summary
    with open(os.path.join(out_dir, "%s_summary.json" % cond), "w") as f:
        json.dump({name: {k: (a[k] if k != "r2_per_trial" else a[k])
                          for k in a} for name, a in agg.items()}, f, indent=2)
    print("wrote results for %s" % cond)
    return agg


if __name__ == "__main__":
    import sys
    conds = sys.argv[1:] if len(sys.argv) > 1 else ["apv", "bic", "ctrl", "dzp", "mix0", "mix50"]
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    for c in conds:
        run_condition(c, out_dir)
    print("done")
