#!/usr/bin/env python3
"""Action B (rbm_p2_multigraph_pipeline) verification pipeline.

Loads the single source JSON, extracts the action, extracts per-case fields,
computes p2_minus_p1 and angle deltas, checks frozen thresholds, and
recomputes the 4 declared minima over the 3 cases for comparison.
"""
import json
import csv
import os

DATA_PATH = "/data/user/conversations/conv-8ab5fa73-141f-470a-b3e5-45582443e897/project/data/source_data/computational_observations.json"
OUT_DIR = "/data/user/conversations/conv-8ab5fa73-141f-470a-b3e5-45582443e897/project/output/results"
CSV_PATH = os.path.join(OUT_DIR, "action_b_pipeline.csv")
JSON_PATH = os.path.join(OUT_DIR, "action_b_pipeline_rows.json")

os.makedirs(OUT_DIR, exist_ok=True)


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    action = next(
        a for a in data["actions"] if a["action_id"] == "rbm_p2_multigraph_pipeline"
    )
    cases = action["cases"]
    n = len(cases)
    assert n == 3, f"expected 3 cases, got {n}"

    thresholds = action["frozen_thresholds_from_prior_single_graph_execution"]
    # threshold key -> field name to check
    threshold_map = {
        "compression_fidelity": "compression_exact_fidelity",
        "p1_final_fidelity": "p1_final_fidelity",
        "p2_final_fidelity": "p2_final_fidelity",
    }

    rows = []
    for i, c in enumerate(cases):
        gate_fids = (
            c["first_layer_gate_terminal_fidelities"]
            + c["second_layer_gate_terminal_fidelities"]
        )
        min_gate = min(gate_fids)
        p2_minus_p1 = c["p2_final_fidelity"] - c["p1_final_fidelity"]
        angle_delta = [
            o - in_ for o, in_ in zip(c["optimized_angles"], c["initial_angles"])
        ]
        threshold_checks = []
        for tkey, field in threshold_map.items():
            value = c[field]
            thr = thresholds[tkey]
            margin = value - thr
            threshold_checks.append(
                {
                    "threshold": tkey,
                    "value": value,
                    "threshold_value": thr,
                    "margin": margin,
                    "passed": value >= thr,
                }
            )
        rows.append(
            {
                "case_index": i + 1,
                "graph_seed": c["graph_seed"],
                "stochastic_seed": c["stochastic_seed"],
                "qaoa_optimizer_step_count": c["qaoa_optimizer_step_count"],
                "p1_final_fidelity": c["p1_final_fidelity"],
                "p2_final_fidelity": c["p2_final_fidelity"],
                "p2_minus_p1": p2_minus_p1,
                "compression_exact_fidelity": c["compression_exact_fidelity"],
                "compression_terminal_estimated_fidelity": c["compression_terminal_estimated_fidelity"],
                "min_gate_terminal_fidelity": min_gate,
                "optimized_exact_cost": c["optimized_exact_cost"],
                "passed_frozen_thresholds": c["passed_frozen_thresholds"],
                "initial_angles": c["initial_angles"],
                "optimized_angles": c["optimized_angles"],
                "angle_deltas": angle_delta,
                "threshold_checks": threshold_checks,
                "all_thresholds_passed": all(tc["passed"] for tc in threshold_checks),
            }
        )

    # ---- recomputed minima vs declared minima ----
    recomputed = {
        "minimum_compression_exact_fidelity": min(
            c["compression_exact_fidelity"] for c in cases
        ),
        "minimum_gate_terminal_fidelity": min(r["min_gate_terminal_fidelity"] for r in rows),
        "minimum_p1_final_fidelity": min(r["p1_final_fidelity"] for r in rows),
        "minimum_p2_final_fidelity": min(r["p2_final_fidelity"] for r in rows),
    }
    declared = {
        k: action[k]
        for k in action
        if k.startswith("minimum_")
    }
    min_comparisons = []
    for k in sorted(recomputed):
        decl = declared.get(k)
        match = decl is not None and recomputed[k] == decl
        min_comparisons.append(
            {
                "field": k,
                "recomputed": recomputed[k],
                "declared": decl,
                "match": match,
            }
        )

    # ---- summary print ----
    print("== Action B: rbm_p2_multigraph_pipeline ==")
    for r in rows:
        print(f"\nCase {r['case_index']}:")
        print(f"  graph_seed={r['graph_seed']} stochastic_seed={r['stochastic_seed']} qaoa_steps={r['qaoa_optimizer_step_count']}")
        print(f"  p1={r['p1_final_fidelity']} p2={r['p2_final_fidelity']} p2_minus_p1={r['p2_minus_p1']}")
        print(f"  comp_exact={r['compression_exact_fidelity']} comp_term_est={r['compression_terminal_estimated_fidelity']}")
        print(f"  min_gate_terminal_fidelity={r['min_gate_terminal_fidelity']}")
        print(f"  optimized_exact_cost={r['optimized_exact_cost']} passed={r['passed_frozen_thresholds']}")
        print(f"  angle_deltas={r['angle_deltas']}")
        for tc in r["threshold_checks"]:
            print(f"  threshold {tc['threshold']}: value={tc['value']} thr={tc['threshold_value']} margin={tc['margin']} passed={tc['passed']}")

    total_checks = sum(len(r["threshold_checks"]) for r in rows)
    total_passed = sum(1 for r in rows for tc in r["threshold_checks"] if tc["passed"])
    print(f"\n== Thresholds: {total_passed}/{total_checks} checks passed ==")

    print("== Recomputed vs declared minima ==")
    all_match = all(m["match"] for m in min_comparisons)
    for m in min_comparisons:
        print(f"  {m['field']}: recomputed={m['recomputed']} declared={m['declared']} match={m['match']}")
    print(f"  all match: {all_match}")

    # ---- write JSON rows ----
    with open(JSON_PATH, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {JSON_PATH}")

    # ---- write CSV ----
    angle_keys = [f"angle_delta_{j}" for j in range(len(rows[0]["initial_angles"]))]
    csv_fields = [
        "case_index",
        "graph_seed",
        "stochastic_seed",
        "qaoa_optimizer_step_count",
        "p1_final_fidelity",
        "p2_final_fidelity",
        "p2_minus_p1",
        "compression_exact_fidelity",
        "compression_terminal_estimated_fidelity",
        "min_gate_terminal_fidelity",
        "optimized_exact_cost",
        "passed_frozen_thresholds",
        "all_thresholds_passed",
    ] + angle_keys
    for k in ["threshold_compression_pass", "threshold_p1_pass", "threshold_p2_pass"]:
        csv_fields.append(k)
    # map threshold check order
    tmap = {"threshold_compression_pass": 0, "threshold_p1_pass": 1, "threshold_p2_pass": 2}
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for r in rows:
            rowd = {k: r[k] for k in csv_fields if k not in angle_keys and k not in tmap}
            for j, v in enumerate(r["angle_deltas"]):
                rowd[f"angle_delta_{j}"] = v
            for ck, idx in tmap.items():
                rowd[ck] = r["threshold_checks"][idx]["passed"]
            w.writerow(rowd)
    print(f"Wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
