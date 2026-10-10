"""Cross-condition aggregation for Neuro_01_self_organization_prediction.

Reads all per-condition *summary.csv and *robustness.csv files in the results
directory and produces cross-condition pivots.

The independent analysis unit is the TRIAL. Within a condition we average
over the independent trials; across conditions we report the per-condition
r2_overall and its spread. This is a descriptive summary, not a pooled
independent-sample inference (conditions are not independent replicates of a
single population).
"""
import glob
import os
import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def _cond_from_name(fname):
    return os.path.basename(fname).replace("_summary.csv", "").replace("_robustness.csv", "")


def load_summary():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_summary.csv")))
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["condition"] = _cond_from_name(f)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_robust():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_robustness.csv")))
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["condition"] = _cond_from_name(f)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def make_pivot(df, value_col, index_col="model", col_col="condition"):
    return df.pivot(index=index_col, columns=col_col, values=value_col)


def main():
    summary = load_summary()
    robust = load_robust()
    if summary.empty:
        raise SystemExit("no summary CSVs found; run per-condition analysis first")

    order = ["zero", "mean", "persist_r", "s", "o", "s_o", "r_s_o", "s_self", "s_o_s"]
    order = [m for m in order if m in summary["model"].unique()] + \
            [m for m in summary["model"].unique() if m not in order]

    # pivots
    p_r2 = make_pivot(summary, "r2_overall")
    p_r2 = p_r2.reindex(index=order)
    p_r2.to_csv(os.path.join(RESULTS_DIR, "cross_condition_r2.csv"))

    p_med = make_pivot(summary, "r2_median_trial")
    p_med = p_med.reindex(index=order)
    p_med.to_csv(os.path.join(RESULTS_DIR, "cross_condition_median.csv"))

    p_na = make_pivot(summary, "n_trials_above_0.05")
    p_na = p_na.reindex(index=order)
    p_na.to_csv(os.path.join(RESULTS_DIR, "cross_condition_nabove.csv"))

    # per-model across-condition summary
    rows = []
    for m in order:
        g = summary[summary["model"] == m]
        vmax = g["r2_overall"].max()
        vmin = g["r2_overall"].min()
        rows.append({
            "model": m,
            "n_conditions": int(len(g)),
            "mean": float(g["r2_overall"].mean()),
            "std": float(g["r2_overall"].std()),
            "min": float(vmin),
            "max": float(vmax),
            "max_condition": str(g.loc[g["r2_overall"].idxmax(), "condition"]),
            "min_condition": str(g.loc[g["r2_overall"].idxmin(), "condition"]),
        })
    df_across = pd.DataFrame(rows)
    df_across.to_csv(os.path.join(RESULTS_DIR, "cross_condition_across.csv"), index=False)

    # robustness: per model x check, mean r2_check across conditions
    rows_rb = []
    for m in order:
        g = robust[robust["model"] == m]
        for check in sorted(g["check"].unique()):
            sub = g[g["check"] == check]["r2_check"].dropna()
            rows_rb.append({
                "model": m,
                "check": check,
                "n_conditions": int(sub.count()),
                "mean": float(sub.mean()),
                "std": float(sub.std()) if sub.count() > 1 else np.nan,
                "min": float(sub.min()),
                "max": float(sub.max()),
            })
    df_rb = pd.DataFrame(rows_rb)
    df_rb.to_csv(os.path.join(RESULTS_DIR, "cross_condition_robust.csv"), index=False)

    print("wrote cross-condition artifacts to", RESULTS_DIR)
    print("\n=== r2_overall by model and condition ===")
    print(p_r2.to_string())
    print("\n=== across-condition summary ===")
    print(df_across.to_string())
    print("\n=== robustness (mean r2_check across conditions) ===")
    print(df_rb.to_string())


if __name__ == "__main__":
    main()
