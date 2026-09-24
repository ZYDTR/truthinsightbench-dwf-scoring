#!/usr/bin/env python3
"""Action A (rbm_exact_cost_layer) analysis: defect, summaries, leave-one-out, min-fidelity check."""

import csv
import json

DATA_PATH = "/data/user/conversations/conv-8ab5fa73-141f-470a-b3e5-45582443e897/project/data/source_data/computational_observations.json"
CSV_PATH = "/data/user/conversations/conv-8ab5fa73-141f-470a-b3e5-45582443e897/project/output/results/action_a_cost_layer.csv"
JSON_PATH = "/data/user/conversations/conv-8ab5fa73-141f-470a-b3e5-45582443e897/project/output/results/action_a_cost_layer_rows.json"
ACTION_ID = "rbm_exact_cost_layer"

with open(DATA_PATH) as f:
    data = json.load(f)

action = next(
    (a for a in data["actions"] if a.get("action_id") == ACTION_ID),
    None,
)
if action is None:
    raise SystemExit("action %s not found" % ACTION_ID)

cases = action["cases"]
assert len(cases) == action.get("case_count", len(cases)), "case count mismatch"

# Attach defect to each case (exact float arithmetic, no rounding)
rows = []
for c in cases:
    fidelity = c["exact_state_fidelity"]
    defect = abs(fidelity - 1.0)
    rows.append(
        {
            "graph": c["graph"],
            "gamma": c["gamma"],
            "nodes": c["nodes"],
            "edges": c["edges"],
            "rbm_hidden_units_after_cost_layer": c["rbm_hidden_units_after_cost_layer"],
            "exact_state_fidelity": fidelity,
            "defect": defect,
        }
    )

# ---------- Per-case table ----------
print("=== PER CASE (defect = abs(fidelity - 1.0)) ===")
for r in rows:
    print(
        "%-28s gamma=%s  fidelity=%r  defect=%r"
        % (r["graph"], r["gamma"], r["exact_state_fidelity"], r["defect"])
    )

# ---------- Per-graph summaries ----------
print("\n=== PER-GRAPH (min, max, mean defect, n) ===")
graphs = {}
for r in rows:
    graphs.setdefault(r["graph"], []).append(r)
for g in sorted(graphs):
    ds = [r["defect"] for r in graphs[g]]
    mean_d = sum(ds) / len(ds)
    print(
        "%-28s n=%d  min_defect=%r  max_defect=%r  mean_defect=%r"
        % (g, len(ds), min(ds), max(ds), mean_d)
    )

# ---------- Per-gamma summaries ----------
print("\n=== PER-GAMMA (min, max, mean defect, n) ===")
gammas = {}
for r in rows:
    gammas.setdefault(r["gamma"], []).append(r)
for gamma in sorted(gammas, key=lambda x: (x < 0, x)):
    ds = [r["defect"] for r in gammas[gamma]]
    mean_d = sum(ds) / len(ds)
    print(
        "gamma=%8.2f n=%d  min_defect=%r  max_defect=%r  mean_defect=%r"
        % (gamma, len(ds), min(ds), max(ds), mean_d)
    )

# ---------- Leave-one-graph-out ----------
print("\n=== LEAVE-ONE-GRAPH-OUT (min fidelity, max defect of remaining 9) ===")
for g in sorted(graphs):
    rest = [r for r in rows if r["graph"] != g]
    print(
        "drop %-28s  n=%d  min_fidelity=%r  max_defect=%r"
        % (g, len(rest), min(r["exact_state_fidelity"] for r in rest), max(r["defect"] for r in rest))
    )

# ---------- Leave-one-gamma-out ----------
print("\n=== LEAVE-ONE-GAMMA-OUT (min fidelity, max defect of remaining 8) ===")
for gamma in sorted(gammas, key=lambda x: (x < 0, x)):
    rest = [r for r in rows if r["gamma"] != gamma]
    print(
        "drop gamma=%8.2f  n=%d  min_fidelity=%r  max_defect=%r"
        % (gamma, len(rest), min(r["exact_state_fidelity"] for r in rest), max(r["defect"] for r in rest))
    )

# ---------- Recomputed global minimum vs declared ----------
recomputed_min = min(r["exact_state_fidelity"] for r in rows)
declared_min = action["observed_minimum_exact_state_fidelity"]
match = recomputed_min == declared_min
print("\n=== MIN FIDELITY CHECK ===")
print("declared       observed_minimum_exact_state_fidelity = %r" % declared_min)
print("recomputed     min(exact_state_fidelity) over 12      = %r" % recomputed_min)
print("match (exact float equality): %r" % match)

# ---------- Write artifacts ----------
with open(CSV_PATH, "w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "graph",
            "gamma",
            "nodes",
            "edges",
            "rbm_hidden_units_after_cost_layer",
            "exact_state_fidelity",
            "defect",
        ],
    )
    w.writeheader()
    w.writerows(rows)

with open(JSON_PATH, "w") as f:
    json.dump(rows, f, indent=2)

print("\nWrote %s and %s" % (CSV_PATH, JSON_PATH))
print("Global max defect = %r" % max(r["defect"] for r in rows))
print("Global min fidelity = %r" % recomputed_min)
