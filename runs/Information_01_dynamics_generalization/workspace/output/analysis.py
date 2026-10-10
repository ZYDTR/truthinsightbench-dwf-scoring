#!/usr/bin/env python3
"""
Executable analysis for TruthInsightBench task
Information_01_dynamics_generalization.

Purpose
-------
Quantify how the prediction error of a single persisted network-dynamics model
(the 'MAK' checkpoint) changes with (a) graph type, (b) state distribution
(input distribution), and (c) state range / out-of-domain paired state shift.

Data boundary
-------------
- data/source_data/computational_observations.json : precomputed per-panel
  relative-error summaries (the direct evidentiary numbers).
- data/source_data/er_n_100_p_*.npy               : 0/1 undirected adjacency
  matrices (graph structure, used to verify the graph panel properties).
- The pickled Torch checkpoint / data files are NOT replayed here: no PyTorch
  runtime is installed in this environment. Per DATA_GUIDE.md we therefore rely
  on the portable computational_observations.json + NumPy matrices and state
  explicitly that model replay was not executed.

This script is fully deterministic and writes machine-readable result artifacts
to output/results/.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]          # .../project
DATA = PROJECT / "data" / "source_data"
OUT = PROJECT / "output"
RESULTS = OUT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 0. Load the graph matrices and verify structure + hashes
# ---------------------------------------------------------------------------
def sha256_bytes(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()

def load_graphs():
    graphs = {}
    for f in sorted(DATA.glob("er_n_100_p_*.npy")):
        a = np.load(f)
        graphs[f.name] = {
            "path": str(f),
            "sha256": sha256_bytes(f),
            "dtype": str(a.dtype),
            "shape": list(a.shape),
            "node_count": int(a.shape[0]),
            "symmetric": bool(np.array_equal(a, a.T)),
            "self_loops": int(np.count_nonzero(np.diag(a))),
            "off_diag_edges": int(np.count_nonzero(np.triu(a, 1))),
            "degree": a.sum(axis=1).astype(float),
            "mean_degree_incl_self": float(a.sum() / a.shape[0]),
            "mean_degree_excl_self": float((a - a * np.eye(a.shape[0])).sum() / a.shape[0]),
            "degree_min": float(np.min(a.sum(axis=1))),
            "degree_median": float(np.median(a.sum(axis=1))),
            "degree_max": float(np.max(a.sum(axis=1))),
        }
    return graphs

# ---------------------------------------------------------------------------
# 1. Load the computational observations
# ---------------------------------------------------------------------------
def load_obs():
    with open(DATA / "computational_observations.json", "r", encoding="utf-8") as fh:
        return json.load(fh)

def panel_mean(p):
    return p["mean_percent"]

def get(d, *keys):
    for k in keys:
        d = d[k]
    return d

# ---------------------------------------------------------------------------
# 2. Derived comparisons (the core numerical evidence)
# ---------------------------------------------------------------------------
def analyze(obs, graphs):
    report = {}

    # --- Internal consistency: the two action-block panels must equal the
    #     diagnostic_panels (they are the same stored numbers).
    within_block = obs["actions"][0]
    within = within_block["panels"]
    diag = obs["diagnostic_panels"]["within_domain"]
    ood_block = obs["actions"][1]
    ood_action = ood_block["panels"]
    ood_diag = obs["diagnostic_panels"]["paired_out_of_domain"]

    cons = {"n_action_panels": len(within),
            "n_ood_graphs": len(ood_action)}
    # numeric cross-check: within-domain action panels vs diagnostic_panels.
    # Keys differ by structure, so map explicitly:
    #   action panel  <-> diagnostic (graph, distribution)
    #   within "training_graph_uniform"      <-> diag training_graph.uniform_0_1
    #   within "equivalent_graph_uniform"    <-> diag equivalent_novel_graph.uniform_0_1
    #   within "equivalent_graph_beta_5_2"   <-> diag equivalent_novel_graph.beta_5_2
    #   within "higher_degree_graph_uniform" <-> diag higher_degree_graph.uniform_0_1
    #   within "higher_degree_graph_beta_5_2" <-> diag higher_degree_graph.beta_5_2
    within_map = {
        "training_graph_uniform": ("training_graph", "uniform_0_1"),
        "equivalent_graph_uniform": ("equivalent_novel_graph", "uniform_0_1"),
        "equivalent_graph_beta_5_2": ("equivalent_novel_graph", "beta_5_2"),
        "higher_degree_graph_uniform": ("higher_degree_graph", "uniform_0_1"),
        "higher_degree_graph_beta_5_2": ("higher_degree_graph", "beta_5_2"),
    }
    cons["within_cross_check"] = {}
    for akey, (gk, dk) in within_map.items():
        va = within[akey]
        vb = diag[gk]["distributions"][dk]
        cons["within_cross_check"][akey] = {
            "action_mean": round(va["mean_percent"], 12),
            "diag_mean": round(vb["mean_percent"], 12),
            "equal": bool(np.allclose(va["mean_percent"], vb["mean_percent"])),
        }
    # OOD: action panels keyed by graph, diag by graph too.
    ood_map = {
        "training_graph": "training_graph",
        "equivalent_novel_graph": "equivalent_novel_graph",
        "higher_degree_graph": "higher_degree_graph",
    }
    cons["ood_cross_check"] = {}
    for gk in ood_map:
        va = ood_action[gk]
        vb = ood_diag[gk]
        cons["ood_cross_check"][gk] = {
            "in_domain_mean": [round(va["in_domain_uniform_0_1"]["mean_percent"], 12),
                               round(vb["in_domain_uniform_0_1"]["mean_percent"], 12)],
            "out_domain_mean": [round(va["out_of_domain_uniform_0_2_1_2"]["mean_percent"], 12),
                                round(vb["out_of_domain_uniform_0_2_1_2"]["mean_percent"], 12)],
            "ratio_equal": bool(np.allclose(va["mean_error_ratio"], vb["mean_error_ratio"])),
            "wilcoxon_p_equal": bool(np.allclose(va["paired_wilcoxon_one_sided_p_value"],
                                                 vb["paired_wilcoxon_one_sided_p_value"])),
        }
    report["internal_consistency"] = cons

    # --- Finding 1: within-domain, graph-type invariance (null/unchanged)
    #     Panels: stored_training_pairs, training/equivalent/higher x uniform/beta
    fd1 = {
        "panels": {},
    }
    for name, p in within.items():
        fd1["panels"][name] = {
            "count": p["count"],
            "mean": p["mean_percent"],
            "median": p["median_percent"],
            "p95": p["p95_percent"],
            "max": p["max_percent"],
        }
    # new-graph-only (exclude stored_training_pairs) to test robustness
    new = {k: v for k, v in within.items() if k != "stored_training_pairs"}
    means = [v["mean_percent"] for v in new.values()]
    all_means = [v["mean_percent"] for v in within.values()]
    fd1["robustness"] = {
        "all_panel_mean_spread_pp": round(max(all_means) - min(all_means), 6),
        "new_graph_mean_spread_pp": round(max(means) - min(means), 6),
        "new_graph_mean_range": [round(min(means), 6), round(max(means), 6)],
        "new_graph_means": {k: round(v["mean_percent"], 6) for k, v in new.items()},
        "max_p95_across_panels": round(max(v["p95_percent"] for v in within.values()), 6),
        "within_panel_p95_minus_median_max_pp": round(
            max(v["p95_percent"] - v["median_percent"] for v in within.values()), 6),
        "graph_type_spread_vs_within_panel_spread": round(
            (max(all_means) - min(all_means)) /
            max(v["p95_percent"] - v["median_percent"] for v in within.values()), 4),
    }
    # directional: is any new graph worse (higher mean) than the training baseline?
    base = within["stored_training_pairs"]["mean_percent"]
    fd1["robustness"]["new_graph_mean_higher_than_training"] = [
        (k, round(v["mean_percent"], 6), round(v["mean_percent"] - base, 6))
        for k, v in sorted(new.items())
    ]
    report["finding1_within_domain_graph_invariance"] = fd1

    # --- Finding 2: within-domain, degradation with boundary state distribution
    #     (beta_0.5_0.5) vs uniform / beta_5.2, across the 3 graphs.
    fd2 = {"graphs": {}}
    for gname, g in diag.items():
        un = g["distributions"]["uniform_0_1"]
        b52 = g["distributions"]["beta_5_2"]
        bnd = g["distributions"]["beta_0_5_0_5_boundary"]
        fd2["graphs"][gname] = {
            "source": g["source"]["path"],
            "mean_degree": g["mean_degree"],
            "uniform": {"mean": un["mean_percent"], "median": un["median_percent"], "p95": un["p95_percent"]},
            "beta_5_2": {"mean": b52["mean_percent"], "median": b52["median_percent"], "p95": b52["p95_percent"]},
            "boundary": {"mean": bnd["mean_percent"], "median": bnd["median_percent"], "p95": bnd["p95_percent"]},
            "boundary_over_uniform_ratio": round(bnd["mean_percent"] / un["mean_percent"], 4),
            "boundary_over_beta52_ratio": round(bnd["mean_percent"] / b52["mean_percent"], 4),
            "median_boundary_over_uniform_ratio": round(bnd["median_percent"] / un["median_percent"], 4),
        }
    ratios = [v["boundary_over_uniform_ratio"] for v in fd2["graphs"].values()]
    fd2["summary"] = {
        "boundary_over_uniform_ratio_range": [round(min(ratios), 4), round(max(ratios), 4)],
        "boundary_over_beta52_ratio_range": [
            round(min(v["boundary_over_beta52_ratio"] for v in fd2["graphs"].values()), 4),
            round(max(v["boundary_over_beta52_ratio"] for v in fd2["graphs"].values()), 4),
        ],
        "boundary_mean_range": [round(min(v["boundary"]["mean"] for v in fd2["graphs"].values()), 4),
                                round(max(v["boundary"]["mean"] for v in fd2["graphs"].values()), 4)],
        "uniform_mean_range": [round(min(v["uniform"]["mean"] for v in fd2["graphs"].values()), 4),
                               round(max(v["uniform"]["mean"] for v in fd2["graphs"].values()), 4)],
    }
    report["finding2_boundary_distribution_degradation"] = fd2

    # --- Finding 3: OOD paired state-shift degradation (U(0,1) -> shifted)
    alpha = ood_block["predeclared_significance_alpha"]
    fd3 = {"graphs": {}}
    for gname, g in ood_action.items():
        inp = g["in_domain_uniform_0_1"]
        outp = g["out_of_domain_uniform_0_2_1_2"]
        fd3["graphs"][gname] = {
            "source": g["source"]["path"],
            "node_count": g["node_count"],
            "mean_degree": g["mean_degree"],
            "paired_seed": g["paired_seed"],
            "pair_count": g["pair_count"],
            "in_domain": {"mean": inp["mean_percent"], "median": inp["median_percent"],
                          "p95": inp["p95_percent"], "max": inp["max_percent"]},
            "out_of_domain": {"mean": outp["mean_percent"], "median": outp["median_percent"],
                              "p95": outp["p95_percent"], "max": outp["max_percent"]},
            "mean_error_ratio": g["mean_error_ratio"],
            "mean_error_increase_pp": g["mean_error_increase_percent_points"],
            "median_error_increase_pp": g["median_error_increase_percent_points"],
            "fraction_pairs_shifted_error_greater": g["fraction_pairs_shifted_error_greater"],
            "wilcoxon_stat": g["paired_wilcoxon_statistic"],
            "wilcoxon_p": g["paired_wilcoxon_one_sided_p_value"],
            "predeclared_alpha": alpha,
            "out_domain_median_vs_in_domain_max": round(
                outp["median_percent"] / inp["max_percent"], 3),
        }
    ratios = [v["mean_error_ratio"] for v in fd3["graphs"].values()]
    fd3["summary"] = {
        "mean_error_ratio_range": [round(min(ratios), 4), round(max(ratios), 4)],
        "all_graphs_directionally_worse": ood_block["all_graphs_directionally_worse"],
        "minimum_mean_error_ratio": ood_block["minimum_mean_error_ratio"],
        "maximum_one_sided_p_value": ood_block["maximum_one_sided_p_value"],
        "predeclared_alpha": alpha,
        "all_p_below_alpha": all(
            v["wilcoxon_p"] < alpha
            for v in fd3["graphs"].values()),
        "all_fraction_pairs_greater_1.0": all(
            v["fraction_pairs_shifted_error_greater"] == 1.0 for v in fd3["graphs"].values()),
        "graph_heterogeneity_note": {
            "largest_ratio_graph": max(fd3["graphs"], key=lambda k: fd3["graphs"][k]["mean_error_ratio"]),
            "ratio_by_graph": {k: round(v["mean_error_ratio"], 4)
                               for k, v in sorted(fd3["graphs"].items())},
        },
    }
    report["finding3_ood_paired_state_shift"] = fd3

    return report

# ---------------------------------------------------------------------------
# 3. Run + write artifacts
# ---------------------------------------------------------------------------
def main():
    print("Loading graph matrices ...")
    graphs = load_graphs()
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("degree",)}
                      for k, v in graphs.items()}, indent=2, default=str))

    print("\nLoading computational observations ...")
    obs = load_obs()

    print("\nAnalyzing ...")
    report = analyze(obs, graphs)

    # write machine-readable result artifacts
    (RESULTS / "graph_panels.json").write_text(
        json.dumps(graphs, indent=2, default=str), encoding="utf-8")
    (RESULTS / "finding1_within_domain_graph_invariance.json").write_text(
        json.dumps(report["finding1_within_domain_graph_invariance"], indent=2), encoding="utf-8")
    (RESULTS / "finding2_boundary_distribution_degradation.json").write_text(
        json.dumps(report["finding2_boundary_distribution_degradation"], indent=2), encoding="utf-8")
    (RESULTS / "finding3_ood_paired_state_shift.json").write_text(
        json.dumps(report["finding3_ood_paired_state_shift"], indent=2), encoding="utf-8")
    (RESULTS / "internal_consistency.json").write_text(
        json.dumps(report["internal_consistency"], indent=2), encoding="utf-8")
    (RESULTS / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # also a CSV for the three OOD graphs (machine-readable table)
    import csv
    with (RESULTS / "ood_paired_shift.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["graph", "source", "mean_degree", "in_domain_mean", "out_domain_mean",
                    "mean_error_ratio", "fraction_pairs_greater", "wilcoxon_p"])
        for gname, v in report["finding3_ood_paired_state_shift"]["graphs"].items():
            w.writerow([gname, v["source"], v["mean_degree"],
                        round(v["in_domain"]["mean"], 6), round(v["out_of_domain"]["mean"], 6),
                        round(v["mean_error_ratio"], 6), v["fraction_pairs_shifted_error_greater"],
                        v["wilcoxon_p"]])

    print("\n=== SUMMARY ===")
    print(json.dumps(report["finding3_ood_paired_state_shift"]["summary"], indent=2))
    print("\n=== F1 robustness ===")
    print(json.dumps(report["finding1_within_domain_graph_invariance"]["robustness"], indent=2))
    print("\n=== F2 summary ===")
    print(json.dumps(report["finding2_boundary_distribution_degradation"]["summary"], indent=2))
    print("\nWrote artifacts to", RESULTS)
    return 0

if __name__ == "__main__":
    sys.exit(main())
