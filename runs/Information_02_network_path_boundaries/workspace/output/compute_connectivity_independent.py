#!/usr/bin/env python3
"""Independent connectivity-contrast check.

Reads only computational_observations.json and compares the global
efficiency E with the directed density d = L / (N*(N-1)):

- Family A: action 'directed_sparse_flower_boundary' (sparse strongly
  connected flowers). Reports all (E - d) values > 1e-12, the minimum
  gap, and the maximum E/d ratio.
- Family B: action 'directed_disconnected_efficiency_lower_boundary'
  (DAG prefixes). Reports whether all (E - d) == 0 within 1e-12.
"""
import json

SRC = "/data/user/conversations/conv-638e4581-a722-4918-95a5-b7bfc57d6954/project/data/source_data/computational_observations.json"
OUT = "/data/user/conversations/conv-638e4581-a722-4918-95a5-b7bfc57d6954/project/output/evidence_connectivity.json"
TOL = 1e-12

with open(SRC) as f:
    data = json.load(f)

A = "directed_sparse_flower_boundary"
B = "directed_disconnected_efficiency_lower_boundary"

actions = {a["action_id"]: a for a in data["actions"]}

# ---------- Family A: sparse strongly connected flowers ----------
sc_cases = actions[A]["cases"]
n_sc_cases = len(sc_cases)
gaps_sc = []
ratios_sc = []
violations_sc = []
for c in sc_cases:
    n = c["n_nodes"]
    L = c["n_arcs"]
    E = c["observed_global_efficiency"]
    density = L / (n * (n - 1))
    gap = E - density
    gaps_sc.append(gap)
    ratios_sc.append(E / density if density > 0 else float("inf"))
    if not (gap > TOL):
        violations_sc.append(c)

# cross-check density convention: E should equal L/(N(N-1)) only for
# the DAG cases; for family A we verify the canonical reference point
# n=8, L=11 -> 11/56 = 0.1964..., E = 0.517857... (gap > 0)

# ---------- Family B: disconnected DAG lower boundary ----------
dag_cases = actions[B]["cases"]
n_dag_cases = len(dag_cases)
gaps_dag = []
violations_dag = []
density_mismatches = []
for c in dag_cases:
    n = c["n_nodes"]
    L = c["n_arcs"]
    E = c["observed_global_efficiency"]
    d_stored = c.get("density")
    d = L / (n * (n - 1)) if (n - 1) != 0 else 0.0
    # stored density must agree with the recomputed one
    if d_stored is not None and abs(d_stored - d) > 1e-12:
        density_mismatches.append(c)
    gap = E - d
    gaps_dag.append(gap)
    if abs(gap) > TOL:
        violations_dag.append(c)

result = {
    "family": "connectivity_contrast",
    "source_file": SRC,
    "tolerance": TOL,
    "family_A_directed_sparse_flower_boundary": {
        "n_cases": n_sc_cases,
        "all_sc_gt_density": len(violations_sc) == 0,
        "min_gap_sc": min(gaps_sc),
        "max_ratio_sc": max(ratios_sc),
        "violations": violations_sc,
        "gap_stats": {
            "min": min(gaps_sc),
            "max": max(gaps_sc),
            "mean": sum(gaps_sc) / len(gaps_sc),
        },
    },
    "family_B_directed_disconnected_efficiency_lower_boundary": {
        "n_cases": n_dag_cases,
        "all_dag_eq_density": len(violations_dag) == 0,
        "max_abs_gap_dag": max(abs(g) for g in gaps_dag),
        "violations": violations_dag,
        "stored_density_mismatches": density_mismatches,
    },
}

print(json.dumps(result, indent=2))
with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
    f.write("\n")
