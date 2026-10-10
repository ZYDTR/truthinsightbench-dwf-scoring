# Workflow script for Neuro_01_self_organization_prediction.
# Serial execution: one child at a time, each condition fully completes
# before the next begins. No parallelism.
#
# Phases:
#   1. Run each remaining condition (bic, ctrl, dzp, mix50) serially.
#   2. Run cross-condition aggregation.
#   3. Final verification that all artifacts exist.
#
# Each child runs the already-verified run_condition.py for one condition
# (or cross_condition.py for aggregation). The analysis code (prediction.py,
# run_condition.py, cross_condition.py, data_loader.py) is authored and
# verified by the main agent; the children merely execute it and report the
# machine-readable results.

import subprocess
import json
import os
import sys

WORKDIR = "/data/user/conversations/conv-5884203e-e071-45de-91eb-fc6c20f5e9d8/project/output/analysis"
RESULTS = "/data/user/conversations/conv-5884203e-e071-45de-91eb-fc6c20f5e9d8/project/output/results"
PYTHON = sys.executable

# Conditions already done: apv, mix0. Remaining: bic, ctrl, dzp, mix50.
CONDITIONS = ["bic", "ctrl", "dzp", "mix50"]


def run_cmd(cmd, cwd, timeout=600):
    """Run a command and return (returncode, stdout_tail, stderr_tail)."""
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout[-2000:], p.stderr[-2000:]


def run_condition(cond):
    """Run run_condition.py for one condition. Returns a result dict."""
    cmd = [PYTHON, "run_condition.py", cond]
    rc, out, err = run_cmd(cmd, WORKDIR, timeout=900)
    # read the summary csv to extract the key numbers
    summary_path = os.path.join(RESULTS, "%s_summary.csv" % cond)
    import pandas as pd
    key = {}
    if rc == 0 and os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        for _, row in df.iterrows():
            key[row["model"]] = {
                "n_trials": int(row["n_trials"]),
                "r2_overall": float(row["r2_overall"]),
                "r2_median_trial": float(row["r2_median_trial"]),
                "n_trials_above_0.05": int(row["n_trials_above_0.05"]),
            }
        # robustness
        rb_path = os.path.join(RESULTS, "%s_robustness.csv" % cond)
        if os.path.exists(rb_path):
            rb = pd.read_csv(rb_path)
            key["robustness"] = rb.to_dict(orient="records")
    return {
        "condition": cond,
        "returncode": rc,
        "key": key,
        "stdout_tail": out,
        "stderr_tail": err,
        "summary_path": summary_path,
    }


def run_cross():
    cmd = [PYTHON, "cross_condition.py"]
    rc, out, err = run_cmd(cmd, WORKDIR, timeout=300)
    return {"returncode": rc, "stdout_tail": out, "stderr_tail": err}


# --- Serial execution ---
results = []
for cond in CONDITIONS:
    print("=== running %s ===" % cond, flush=True)
    r = run_condition(cond)
    results.append(r)
    print("  %s rc=%d" % (cond, r["returncode"]), flush=True)
    # report the key r2 values
    for model in ["persist_r", "s", "o", "s_o", "r_s_o", "s_self", "s_o_s"]:
        if model in r["key"]:
            k = r["key"][model]
            print("    %s r2=%.4f (median=%.4f, n>0.05=%d/%d)" % (
                model, k["r2_overall"], k["r2_median_trial"],
                k["n_trials_above_0.05"], k["n_trials"]), flush=True)

print("\n=== cross-condition aggregation ===", flush=True)
cross = run_cross()
print("  cross rc=%d" % cross["returncode"], flush=True)

# write a combined result file for the main agent to read
combined = {
    "conditions": results,
    "cross": cross,
}
with open(os.path.join(WORKDIR, "run_results.json"), "w") as f:
    json.dump(combined, f, indent=2, default=float)
print("\nwrote run_results.json", flush=True)
print("\n=== ALL DONE ===", flush=True)
