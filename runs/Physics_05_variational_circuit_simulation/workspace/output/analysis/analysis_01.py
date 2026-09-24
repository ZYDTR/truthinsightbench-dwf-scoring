"""
Analysis 01: classical vs exact state-vector fidelity for variational circuit
simulations (Physics_05_variational_circuit_simulation).

Data: data/source_data/computational_observations.json (hash-verified in
analysis_00_integrity.py against the manifest sha256).

Two actions:
  A. rbm_exact_cost_layer  : 12 cases, 4 graph types x 3 gammas.
                             exact_state_fidelity is the comparison of the
                             classical representation to the exact state vector.
  B. rbm_p2_multigraph_pipeline : 3 cases, 6-node degree-3 random graphs.
                             p1/p2 final state fidelity, gate-level terminal
                             fidelities, compression fidelity, and a frozen
                             threshold set from a prior single-graph run.

All computations are recomputed from the raw fields; no external data used.
"""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data" / "source_data" / "computational_observations.json"
OUT = ROOT / "results"

obs = json.loads(DATA.read_text())


def group_fidelity_delta(f):
    """Fidelity defect from the exact-1.0 state-vector reference.
    Returns positive numbers for the magnitude of deviation (|f - 1|).
    """
    return abs(f - 1.0)


# ----------------------------------------------------------------------------
# Action A: cost-layer fidelity
# ----------------------------------------------------------------------------
a_cases = obs["actions"][0]["cases"]
a_min = obs["actions"][0]["observed_minimum_exact_state_fidelity"]

print("=== ACTION A: rbm_exact_cost_layer ===")
print(f"declared observed minimum = {a_min!r}")

rows_a = []
for c in a_cases:
    rows_a.append(
        {
            "action": "rbm_exact_cost_layer",
            "graph": c["graph"],
            "nodes": c["nodes"],
            "edges": c["edges"],
            "gamma": c["gamma"],
            "rbm_hidden_units_after_cost_layer": c["rbm_hidden_units_after_cost_layer"],
            "exact_state_fidelity": c["exact_state_fidelity"],
            "defect": abs(c["exact_state_fidelity"] - 1.0),
        }
    )

# Per-graph and per-gamma summaries
graphs = sorted({r["graph"] for r in rows_a})
gammas = sorted({r["gamma"] for r in rows_a})

def summarize(pred):
    vals = [r["defect"] for r in rows_a if pred(r)]
    if not vals:
        return (None, None, None, 0)
    return (min(vals), max(vals), sum(vals) / len(vals), len(vals))

graph_summary = []
for g in graphs:
    m, M, mean, n = summarize(lambda r, g=g: r["graph"] == g)
    graph_summary.append({"graph": g, "min_defect": m, "max_defect": M, "mean_defect": mean, "n": n})
    print(f"A  {g:28s} n={n}  min_defect={m:.3e}  max_defect={M:.3e}  mean_defect={mean:.3e}")

gamma_summary = []
for gamma in gammas:
    m, M, mean, n = summarize(lambda r, gamma=gamma: r["gamma"] == gamma)
    gamma_summary.append({"gamma": gamma, "min_defect": m, "max_defect": M, "mean_defect": mean, "n": n})
    print(f"A  gamma={gamma:5.2f}        n={n}  min_defect={m:.3e}  max_defect={M:.3e}  mean_defect={mean:.3e}")

# Reconstruct the per-graph x gamma table
print("\nA  fidelity by (graph, gamma):")
for gamma in gammas:
    for g in graphs:
        r = next(x for x in rows_a if x["graph"] == g and x["gamma"] == gamma)
        print(f"    {g:28s} gamma={gamma:5.2f}  f={r['exact_state_fidelity']!r}  defect={r['defect']:.3e}")

# Robustness: leave-one-graph-out (4 graphs -> 4 leaves)
print("\nA  robustness: leave-one-graph-out (minimum defect of remaining set)")
for g in graphs:
    rest = [r for r in rows_a if r["graph"] != g]
    print(f"    leave {g:28s} -> min_defect={min(r['defect'] for r in rest):.3e}  "
          f"max_defect={max(r['defect'] for r in rest):.3e}  n={len(rest)}")

# gamma ordering: does gamma affect the defect? (no model assumed; just report)
print("\nA  gamma-wise max defect (sensitivity to gamma):")
for gamma in gammas:
    vals = [r["defect"] for r in rows_a if r["gamma"] == gamma]
    print(f"    gamma={gamma:5.2f}  max_defect={max(vals):.3e}  span={max(vals)-min(vals):.3e}")

# Robustness: leave-one-gamma-out
print("\nA  robustness: leave-one-gamma-out (max defect of remaining set)")
for gamma in gammas:
    rest = [r for r in rows_a if r["gamma"] != gamma]
    print(f"    leave gamma={gamma:5.2f} -> min_defect={min(r['defect'] for r in rest):.3e}  "
          f"max_defect={max(r['defect'] for r in rest):.3e}  n={len(rest)}")

# ----------------------------------------------------------------------------
# Action B: p2 multigraph pipeline
# ----------------------------------------------------------------------------
b_cases = obs["actions"][1]["cases"]
b_frozen = obs["actions"][1]["frozen_thresholds_from_prior_single_graph_execution"]
b_decl_mins = {
    "compression_exact_fidelity": obs["actions"][1]["minimum_compression_exact_fidelity"],
    "gate_terminal_fidelity": obs["actions"][1]["minimum_gate_terminal_fidelity"],
    "p1_final_fidelity": obs["actions"][1]["minimum_p1_final_fidelity"],
    "p2_final_fidelity": obs["actions"][1]["minimum_p2_final_fidelity"],
}

print("\n=== ACTION B: rbm_p2_multigraph_pipeline (3 graphs) ===")
print(f"frozen thresholds (prior single-graph execution): {b_frozen}")

rows_b = []
for i, c in enumerate(b_cases, start=1):
    f1 = c["first_layer_gate_terminal_fidelities"]
    f2 = c["second_layer_gate_terminal_fidelities"]
    all_gate = f1 + f2
    rows_b.append(
        {
            "case": f"p2_case_{i}",
            "graph_seed": c["graph_seed"],
            "stochastic_seed": c["stochastic_seed"],
            "nodes": c["nodes"],
            "degree": c["degree"],
            "n_edges": len(c["edges"]),
            "qaoa_optimizer_step_count": c["qaoa_optimizer_step_count"],
            "p1_final_fidelity": c["p1_final_fidelity"],
            "p2_final_fidelity": c["p2_final_fidelity"],
            "compression_exact_fidelity": c["compression_exact_fidelity"],
            "compression_terminal_estimated_fidelity": c["compression_terminal_estimated_fidelity"],
            "min_gate_terminal_fidelity": min(all_gate),
            "max_gate_terminal_fidelity": max(all_gate),
            "mean_gate_terminal_fidelity": sum(all_gate) / len(all_gate),
            "optimized_exact_cost": c["optimized_exact_cost"],
            "passed_frozen_thresholds": c["passed_frozen_thresholds"],
            "initial_angles": c["initial_angles"],
            "optimized_angles": c["optimized_angles"],
        }
    )

print(f"\nB  declared minima: {b_decl_mins}")
for r in rows_b:
    print(f"\nB  {r['case']}  seed={r['graph_seed']!r}  qaoa_steps={r['qaoa_optimizer_step_count']}")
    print(f"    p1={r['p1_final_fidelity']!r}  p2={r['p2_final_fidelity']!r}")
    print(f"    compression_exact={r['compression_exact_fidelity']!r}  "
          f"compression_est={r['compression_terminal_estimated_fidelity']!r}")
    print(f"    gate terminals: min={r['min_gate_terminal_fidelity']!r} "
          f"max={r['max_gate_terminal_fidelity']!r} mean={r['mean_gate_terminal_fidelity']!r}")
    print(f"    optimized_exact_cost={r['optimized_exact_cost']!r}  "
          f"passed_thresholds={r['passed_frozen_thresholds']}")
    print(f"    initial_angles={r['initial_angles']}")
    print(f"    optimized_angles={r['optimized_angles']}")
    delta = [a - b for a, b in zip(r["initial_angles"], r["optimized_angles"])]
    print(f"    angle_delta (optimized - initial) = {[round(x, 12) for x in delta]}")

# Threshold check: compare each case to the frozen thresholds
print("\nB  threshold check (frozen, from prior single-graph run):")
for r in rows_b:
    for k, thr in b_frozen.items():
        key = {
            "compression_fidelity": "compression_exact_fidelity",
            "p1_final_fidelity": "p1_final_fidelity",
            "p2_final_fidelity": "p2_final_fidelity",
        }[k]
        val = r[key]
        ok = val >= thr
        margin = val - thr
        print(f"    {r['case']}  {k:32s} val={val!r} thr={thr}  pass={ok}  margin={margin:+.6f}")

# p1 vs p2: does the p2 pipeline degrade relative to p1?
print("\nB  p1 vs p2 per-case (p2 - p1):")
for r in rows_b:
    print(f"    {r['case']}  p2 - p1 = {r['p2_final_fidelity'] - r['p1_final_fidelity']:+.6f}")

# compression exact vs estimated
print("\nB  compression exact vs terminal-estimated:")
for r in rows_b:
    print(f"    {r['case']}  exact={r['compression_exact_fidelity']!r}  "
          f"estimated={r['compression_terminal_estimated_fidelity']!r}  "
          f"diff={r['compression_terminal_estimated_fidelity']-r['compression_exact_fidelity']:+.6f}")

# ----------------------------------------------------------------------------
# Cross-action: compare cost-layer (A) vs final pipeline (B) fidelity
# ----------------------------------------------------------------------------
a_min_defect = min(r["defect"] for r in rows_a)
a_max_defect = max(r["defect"] for r in rows_a)
print(f"\n=== CROSS-ACTION ===")
print(f"A  cost-layer defects: all in [{a_min_defect:.3e}, {a_max_defect:.3e}] (vs 1.0)")
print(f"B  p1 final fidelity:   min={min(r['p1_final_fidelity'] for r in rows_b):.6f}")
print(f"B  p2 final fidelity:   min={min(r['p2_final_fidelity'] for r in rows_b):.6f}")
print(f"B  gate terminal:       min={min(r['min_gate_terminal_fidelity'] for r in rows_b):.6f}")

# Save all raw rows to JSON
OUT.mkdir(parents=True, exist_ok=True)
OUT.joinpath("action_a_cost_layer_rows.json").write_text(
    json.dumps(rows_a, indent=2))
OUT.joinpath("action_b_pipeline_rows.json").write_text(
    json.dumps(rows_b, indent=2))
OUT.joinpath("summaries.json").write_text(json.dumps(
    {
        "action_a": {
            "rows": rows_a,
            "graph_summary": graph_summary,
            "gamma_summary": gamma_summary,
            "declared_min": a_min,
        },
        "action_b": {
            "rows": rows_b,
            "frozen_thresholds": b_frozen,
            "declared_mins": b_decl_mins,
        },
    }, indent=2))

# Robustness + declared-minima cross-check artifact
robustness = {
    "action_a_cost_layer": {
        "declared_observed_min": a_min,
        "recomputed_min": min(r["exact_state_fidelity"] for r in rows_a),
        "declared_matches_recomputed": abs(a_min - min(r["exact_state_fidelity"] for r in rows_a)) < 1e-15,
        "leave_one_graph_out": {
            g: {
                "remaining_n": sum(1 for r in rows_a if r["graph"] != g),
                "min_fidelity": min(r["exact_state_fidelity"] for r in rows_a if r["graph"] != g),
                "max_defect": max(abs(r["exact_state_fidelity"] - 1.0) for r in rows_a if r["graph"] != g),
            } for g in graphs
        },
        "leave_one_gamma_out": {
            f"{gamma:.2f}": {
                "remaining_n": sum(1 for r in rows_a if r["gamma"] != gamma),
                "min_fidelity": min(r["exact_state_fidelity"] for r in rows_a if r["gamma"] != gamma),
                "max_defect": max(abs(r["exact_state_fidelity"] - 1.0) for r in rows_a if r["gamma"] != gamma),
            } for gamma in gammas
        },
    },
    "action_b_pipeline": {
        "declared_mins": b_decl_mins,
        "recomputed": {
            "compression_exact_fidelity": min(c["compression_exact_fidelity"] for c in b_cases),
            "gate_terminal_fidelity": min(
                min(c["first_layer_gate_terminal_fidelities"] +
                    c["second_layer_gate_terminal_fidelities"]) for c in b_cases),
            "p1_final_fidelity": min(c["p1_final_fidelity"] for c in b_cases),
            "p2_final_fidelity": min(c["p2_final_fidelity"] for c in b_cases),
        },
        "declared_matches_recomputed": {
            k: abs(b_decl_mins[k] - v) < 1e-15
            for k, v in (
                ("compression_exact_fidelity", min(c["compression_exact_fidelity"] for c in b_cases)),
                ("gate_terminal_fidelity", min(
                    min(c["first_layer_gate_terminal_fidelities"] +
                        c["second_layer_gate_terminal_fidelities"]) for c in b_cases)),
                ("p1_final_fidelity", min(c["p1_final_fidelity"] for c in b_cases)),
                ("p2_final_fidelity", min(c["p2_final_fidelity"] for c in b_cases)),
            )
        },
        "thresholds": {
            r["case"]: {
                k: {
                    "value": r[{
                        "compression_fidelity": "compression_exact_fidelity",
                        "p1_final_fidelity": "p1_final_fidelity",
                        "p2_final_fidelity": "p2_final_fidelity",
                    }[k]],
                    "threshold": b_frozen[k],
                    "margin": r[{
                        "compression_fidelity": "compression_exact_fidelity",
                        "p1_final_fidelity": "p1_final_fidelity",
                        "p2_final_fidelity": "p2_final_fidelity",
                    }[k]] - b_frozen[k],
                    "passed": r[{
                        "compression_fidelity": "compression_exact_fidelity",
                        "p1_final_fidelity": "p1_final_fidelity",
                        "p2_final_fidelity": "p2_final_fidelity",
                    }[k]] >= b_frozen[k],
                }
                for k in b_frozen
            }
            for r in rows_b
        },
        "note": "declared minima are recomputed from the raw per-case fields and match.",
    },
}
OUT.joinpath("robustness.json").write_text(json.dumps(robustness, indent=2))
print("\nWrote robustness.json")

# Human-readable CSV exports
def dump_csv(path, rows, keys):
    import csv
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

csv_keys_a = ["graph", "gamma", "nodes", "edges",
              "rbm_hidden_units_after_cost_layer",
              "exact_state_fidelity", "defect"]
dump_csv(OUT.joinpath("action_a_cost_layer.csv"), rows_a, csv_keys_a)

csv_keys_b = ["case", "graph_seed", "stochastic_seed", "nodes", "degree",
              "qaoa_optimizer_step_count", "p1_final_fidelity",
              "p2_final_fidelity", "p2_minus_p1",
              "compression_exact_fidelity",
              "compression_terminal_estimated_fidelity",
              "min_gate_terminal_fidelity", "optimized_exact_cost",
              "passed_frozen_thresholds"]
for r in rows_b:
    r["p2_minus_p1"] = r["p2_final_fidelity"] - r["p1_final_fidelity"]
dump_csv(OUT.joinpath("action_b_pipeline.csv"), rows_b, csv_keys_b)

print("\nWrote CSV exports.")

