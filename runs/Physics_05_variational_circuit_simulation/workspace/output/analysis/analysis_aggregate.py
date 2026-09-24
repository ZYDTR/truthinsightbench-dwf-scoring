#!/usr/bin/env python3
"""Aggregation / robustness branch.

Cross-checks the three branch artifacts:
  - action_a_cost_layer_rows.json   (rbm_exact_cost_layer: 12 rows)
  - action_b_pipeline_rows.json     (rbm_p2_multigraph_pipeline: 3 rows)
  - integrity_receipt.json

Declared minima are read from the workspace source data file
(data/source_data/computational_observations.json); recomputed minima are
recomputed from the branch row artifacts. Integrity is taken from the
receipt AND independently re-verified (sha256 + byte size of the source
file).

Output: /data/user/conversations/conv-8ab5fa73-141f-470a-b3e5-45582443e897/project/output/results/robustness.json
"""

import hashlib
import json
import os

BASE = "/data/user/conversations/conv-8ab5fa73-141f-470a-b3e5-45582443e897/project"
DATA_PATH = os.path.join(BASE, "data", "source_data", "computational_observations.json")
RECEIPT_PATH = os.path.join(BASE, "output", "results", "integrity_receipt.json")
COST_ROWS_PATH = os.path.join(BASE, "output", "results", "action_a_cost_layer_rows.json")
PIPE_ROWS_PATH = os.path.join(BASE, "output", "results", "action_b_pipeline_rows.json")
OUT_PATH = os.path.join(BASE, "output", "results", "robustness.json")

errors = []


def load_json(path, label):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        errors.append("failed to load %s: %s" % (label, e))
        return None


data = load_json(DATA_PATH, "source data")
receipt = load_json(RECEIPT_PATH, "integrity receipt")
cost_rows = load_json(COST_ROWS_PATH, "cost layer rows")
pipe_rows = load_json(PIPE_ROWS_PATH, "pipeline rows")

if data is None:
    raise SystemExit("aborted: cannot continue without source data; errors: " + str(errors))
if cost_rows is None or pipe_rows is None:
    raise SystemExit("aborted: cannot continue without branch artifacts; errors: " + str(errors))

actions = {a["action_id"]: a for a in data["actions"]}
cost_action = actions.get("rbm_exact_cost_layer", {})
pipe_action = actions.get("rbm_p2_multigraph_pipeline", {})
if not cost_action or not pipe_action:
    raise SystemExit("aborted: expected actions not found in source data")

# ---------------------- cost layer ----------------------
declared_min_cl = cost_action.get("observed_minimum_exact_state_fidelity")
recomputed_min_cl = min(r["exact_state_fidelity"] for r in cost_rows)
max_defect_cl = max(r["defect"] for r in cost_rows)

def leave_one_out(rows, key, numeric=False):
    out = {}
    keys = sorted({r[key] for r in rows}, key=lambda k: (k < 0, k) if numeric else k)
    for k in keys:
        rest = [r for r in rows if r[key] != k]
        out[str(k)] = {
            "remaining_n": len(rest),
            "min_fidelity": min(r["exact_state_fidelity"] for r in rest),
            "max_defect": max(r["defect"] for r in rest),
        }
    return out

leave_one_graph_out = leave_one_out(cost_rows, "graph")
leave_one_gamma_out = leave_one_out(cost_rows, "gamma", numeric=True)

cost_layer = {
    "declared_min": declared_min_cl,
    "recomputed_min": recomputed_min_cl,
    "declared_matches_recomputed": declared_min_cl == recomputed_min_cl,
    "max_defect": max_defect_cl,
    "leave_one_graph_out": leave_one_graph_out,
    "leave_one_gamma_out": leave_one_gamma_out,
}

# ---------------------- pipeline ----------------------
declared_mins = {
    "compression_exact_fidelity": pipe_action.get("minimum_compression_exact_fidelity"),
    "gate_terminal_fidelity": pipe_action.get("minimum_gate_terminal_fidelity"),
    "p1_final_fidelity": pipe_action.get("minimum_p1_final_fidelity"),
    "p2_final_fidelity": pipe_action.get("minimum_p2_final_fidelity"),
}
recomputed_mins = {
    "compression_exact_fidelity": min(r["compression_exact_fidelity"] for r in pipe_rows),
    "gate_terminal_fidelity": min(r["min_gate_terminal_fidelity"] for r in pipe_rows),
    "p1_final_fidelity": min(r["p1_final_fidelity"] for r in pipe_rows),
    "p2_final_fidelity": min(r["p2_final_fidelity"] for r in pipe_rows),
}
declared_matches_recomputed = {
    k: (declared_mins[k] == recomputed_mins[k]) for k in declared_mins
}

threshold_map = {
    "compression_fidelity": "compression_exact_fidelity",
    "p1_final_fidelity": "p1_final_fidelity",
    "p2_final_fidelity": "p2_final_fidelity",
}
frozen = pipe_action.get("frozen_thresholds_from_prior_single_graph_execution", {})

per_case_p2_minus_p1 = {}
thresholds = {}
for r in pipe_rows:
    case_key = "case_%d" % r["case_index"]
    per_case_p2_minus_p1[case_key] = r["p2_minus_p1"]
    per_case = {}
    for tkey, field in threshold_map.items():
        value = r[field]
        thr = frozen[tkey]
        per_case[tkey] = {
            "value": value,
            "threshold": thr,
            "margin": value - thr,
            "passed": value >= thr,
        }
    thresholds[case_key] = per_case

pipeline = {
    "declared_mins": declared_mins,
    "recomputed_mins": recomputed_mins,
    "declared_matches_recomputed": declared_matches_recomputed,
    "min_p1": recomputed_mins["p1_final_fidelity"],
    "min_p2": recomputed_mins["p2_final_fidelity"],
    "per_case_p2_minus_p1": per_case_p2_minus_p1,
    "thresholds": thresholds,
}

# ---------------------- integrity ----------------------
hash_match = bool(receipt.get("hash_match", False))
bytes_match = bool(receipt.get("bytes") == receipt.get("manifest_bytes"))

try:
    raw = open(DATA_PATH, "rb").read()
    recomputed_sha = hashlib.sha256(raw).hexdigest()
    hash_match = hash_match and (recomputed_sha == receipt.get("sha256")) \
        and (recomputed_sha == receipt.get("manifest_sha256"))
    bytes_match = bytes_match and (len(raw) == receipt.get("bytes")) \
        and (len(raw) == receipt.get("manifest_bytes"))
except Exception as e:
    errors.append("failed to independently verify integrity: %s" % e)

integrity = {
    "hash_match": hash_match,
    "bytes_match": bytes_match,
}

result = {
    "cost_layer": cost_layer,
    "pipeline": pipeline,
    "integrity": integrity,
}

with open(OUT_PATH, "w") as f:
    json.dump(result, f, indent=2)

# ---------------------- summary ----------------------
print("Wrote", OUT_PATH)
print()
print("=== cost_layer ===")
print("declared_min       :", repr(declared_min_cl))
print("recomputed_min     :", repr(recomputed_min_cl))
print("declared==recomputed:", cost_layer["declared_matches_recomputed"])
print("max_defect         :", repr(max_defect_cl))
print("leave_one_graph_out (n remaining 9):")
for g in sorted(cost_layer["leave_one_graph_out"]):
    d = cost_layer["leave_one_graph_out"][g]
    print("  %-28s n=%d min_fidelity=%r max_defect=%r"
          % (g, d["remaining_n"], d["min_fidelity"], d["max_defect"]))
print("leave_one_gamma_out (n remaining 8):")
for g in sorted(cost_layer["leave_one_gamma_out"], key=float):
    d = cost_layer["leave_one_gamma_out"][g]
    print("  gamma=%8s n=%d min_fidelity=%r max_defect=%r"
          % (g, d["remaining_n"], d["min_fidelity"], d["max_defect"]))
print()
print("=== pipeline ===")
for k in declared_mins:
    print("  %s: declared=%r recomputed=%r match=%r"
          % (k, declared_mins[k], recomputed_mins[k], declared_matches_recomputed[k]))
print("  min_p1 =", repr(pipeline["min_p1"]))
print("  min_p2 =", repr(pipeline["min_p2"]))
print("  per_case_p2_minus_p1:")
for ck in per_case_p2_minus_p1:
    print("    %s = %r" % (ck, per_case_p2_minus_p1[ck]))
total = sum(1 for c in thresholds.values() for t in c.values() if t["passed"])
total_checks = sum(len(c) for c in thresholds.values())
print("  thresholds: %d/%d passed" % (total, total_checks))
print()
print("=== integrity ===")
print("hash_match:", integrity["hash_match"])
print("bytes_match:", integrity["bytes_match"])
print()
if errors:
    print("ERRORS:")
    for e in errors:
        print("  -", e)
else:
    print("no errors")
