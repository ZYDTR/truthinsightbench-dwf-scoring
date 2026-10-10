#!/usr/bin/env python3
"""
Independent re-derivation of Finding 1: within-domain graph invariance.

This script computes every number ONLY from the raw source data
(data/source_data/computational_observations.json). It does NOT read
output/analysis.py or any output/*.json result file.

Steps:
 (a) Load computational_observations.json.
 (b) From actions[0].panels (6 within-domain panels) compute/derive:
      - per-panel mean/median/p95/max (percent)
      - overall mean spread (max mean - min mean, percentage points)
      - spread over the 5 panels excluding stored_training_pairs
      - max of (p95 - median) across panels
      - per new-graph panel mean difference vs stored_training_pairs (pp)
 (c) Read the same numbers from diagnostic_panels.within_domain and check
      exact mean agreement for 5 mappings.
 (d) Write JSON to output/results/independent_f1.json.
"""

import json
import os

BASE = "/data/user/conversations/conv-07581fcf-2edd-4246-8da8-dc035efd1751/project"
SRC = os.path.join(BASE, "data", "source_data", "computational_observations.json")
OUT_DIR = os.path.join(BASE, "output", "results")
OUT = os.path.join(OUT_DIR, "independent_f1.json")


def main():
    # ---- (a) Load raw data ----
    with open(SRC, "r", encoding="utf-8") as f:
        data = json.load(f)

    action = data["actions"][0]
    assert action["action_id"] == "network_dynamics_ood_generalization_within_domain", action["action_id"]
    assert action["panel_count"] == 6, action["panel_count"]
    panels = action["panels"]
    assert len(panels) == 6, list(panels.keys())

    # ---- (b) Per-panel stats (mean/median/p95/max percent) ----
    panel_stats = {}
    for pname in panels:
        p = panels[pname]
        panel_stats[pname] = {
            "mean_percent": p["mean_percent"],
            "median_percent": p["median_percent"],
            "p95_percent": p["p95_percent"],
            "max_percent": p["max_percent"],
        }

    means = {pname: panel_stats[pname]["mean_percent"] for pname in panel_stats}

    # Overall mean spread (max mean - min mean, in percentage points)
    overall_spread_pp = max(means.values()) - min(means.values())

    # Spread over the 5 panels excluding stored_training_pairs
    new_graph_panels = [pn for pn in panel_stats if pn != "stored_training_pairs"]
    assert len(new_graph_panels) == 5, new_graph_panels
    new_means = [means[pn] for pn in new_graph_panels]
    new_graph_spread_pp = max(new_means) - min(new_means)

    # Max of (p95 - median) across panels
    p95_minus_median = {
        pn: panel_stats[pn]["p95_percent"] - panel_stats[pn]["median_percent"]
        for pn in panel_stats
    }
    max_p95_minus_median_pp = max(p95_minus_median.values())
    max_p95_minus_median_panel = max(p95_minus_median, key=p95_minus_median.get)

    # Per new-graph panel mean difference vs stored_training_pairs baseline (pp)
    baseline_name = "stored_training_pairs"
    baseline_mean = means[baseline_name]
    new_graph_diff_from_baseline_pp = {
        pn: means[pn] - baseline_mean for pn in new_graph_panels
    }

    # ---- (c) Consistency: diagnostic_panels.within_domain vs actions[0].panels ----
    wd = data["diagnostic_panels"]["within_domain"]
    mappings = [
        ("training_graph", "uniform_0_1", "training_graph_uniform"),
        ("equivalent_novel_graph", "uniform_0_1", "equivalent_graph_uniform"),
        ("equivalent_novel_graph", "beta_5_2", "equivalent_graph_beta_5_2"),
        ("higher_degree_graph", "uniform_0_1", "higher_degree_graph_uniform"),
        ("higher_degree_graph", "beta_5_2", "higher_degree_graph_beta_5_2"),
    ]
    consistency_details = []
    consistency_equal = []
    for graph, dist, action_panel in mappings:
        diag_mean = wd[graph]["distributions"][dist]["mean_percent"]
        act_mean = means[action_panel]
        equal = diag_mean == act_mean
        consistency_equal.append(equal)
        consistency_details.append({
            "mapping": f"{graph}.{dist} vs {action_panel}",
            "diagnostic_mean_percent": diag_mean,
            "action_panel_mean_percent": act_mean,
            "exact_mean_equal": equal,
        })

    # ---- (d) Write JSON ----
    os.makedirs(OUT_DIR, exist_ok=True)
    result = {
        "source_file": SRC,
        "derived_from": "data/source_data/computational_observations.json (actions[0].panels + diagnostic_panels.within_domain)",
        "panel_stats": panel_stats,
        "panel_means": means,
        "overall_spread_pp": overall_spread_pp,
        "new_graph_spread_pp": new_graph_spread_pp,
        "max_p95_minus_median_pp": max_p95_minus_median_pp,
        "max_p95_minus_median_panel": max_p95_minus_median_panel,
        "p95_minus_median_by_panel": p95_minus_median,
        "baseline": {"name": baseline_name, "mean_percent": baseline_mean},
        "new_graph_diff_from_baseline_pp": new_graph_diff_from_baseline_pp,
        "consistency_equal": consistency_equal,
        "consistency_details": consistency_details,
        "new_graph_panels": new_graph_panels,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # ---- Print summary to stdout ----
    print("=== Independent re-derivation: Finding 1 (within-domain graph invariance) ===")
    print("Source:", SRC)
    print("\nPer-panel stats (percent):")
    for pn in panel_stats:
        s = panel_stats[pn]
        print(f"  {pn}: mean={s['mean_percent']:.16f} median={s['median_percent']:.16f} "
              f"p95={s['p95_percent']:.16f} max={s['max_percent']:.16f}")
    print(f"\nOverall mean spread (max-min, all 6 panels) = {overall_spread_pp:.16f} pp")
    print(f"New-graph spread (5 panels excl. stored_training_pairs) = {new_graph_spread_pp:.16f} pp")
    print(f"Max (p95 - median) = {max_p95_minus_median_pp:.16f} pp  [{max_p95_minus_median_panel}]")
    print("\nMean diff vs stored_training_pairs baseline (pp):")
    for pn in new_graph_panels:
        print(f"  {pn}: {new_graph_diff_from_baseline_pp[pn]:.16f}")
    print("\nConsistency (diagnostic_panels.within_domain == actions[0].panels):")
    for d in consistency_details:
        print(f"  {d['mapping']}: {d['diagnostic_mean_percent']} == {d['action_panel_mean_percent']} -> {d['exact_mean_equal']}")
    print(f"\nAll 5 mappings exact-match: {all(consistency_equal)}")
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    main()
