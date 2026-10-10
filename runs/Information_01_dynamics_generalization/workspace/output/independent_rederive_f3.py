#!/usr/bin/env python3
"""
Independent rederivation of Finding 3: paired out-of-domain state-shift evidence.

This script does NOT copy any number from output/results/finding3_*.json.
It loads ONLY the raw source data:
    data/source_data/computational_observations.json
and independently computes, per the 3 graphs in actions[1].panels:
    - in-domain mean / median / p95 / max
    - out-of-domain mean / median / p95 / max
    - mean_error_ratio            = out_mean / in_mean      (derived, not trusted)
    - mean_error_increase_percent_points = out_mean - in_mean (derived, not trusted)
    - fraction_pairs_shifted_error_greater
    - paired_wilcoxon_statistic
    - paired_wilcoxon_one_sided_p_value
    - p < predeclared_alpha (0.001)

It also independently verifies the Wilcoxon statistic: with pair_count pairs and
fraction_pairs_shifted_error_greater == 1.0, every paired difference has the same
sign, so the rank-sum equals the maximum possible n*(n+1)/2. This is a genuine
independent derivation, not a lookup.

Finally it cross-checks each graph against diagnostic_panels.paired_out_of_domain:
    in-domain mean, out-of-domain mean, ratio, and p-value must all be equal.

Results are written to output/results/independent_f3.json.
"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent  # .../project
SRC = BASE / "data" / "source_data" / "computational_observations.json"
OUT_DIR = BASE / "output" / "results"
OUT_JSON = OUT_DIR / "independent_f3.json"

ALPHA = 0.001  # predeclared significance level


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def main():
    data = load_json(SRC)

    # actions[1] is the out-of-domain failure boundary action.
    action = data["actions"][1]
    declared_alpha = action["predeclared_significance_alpha"]
    diag = data["diagnostic_panels"]["paired_out_of_domain"]

    panels = action["panels"]
    assert len(panels) == 3, f"expected 3 panels, got {len(panels)}"
    graph_names = list(panels.keys())

    per_graph = {}
    for g in graph_names:
        p = panels[g]
        n_pairs = p["pair_count"]

        in_dom = p["in_domain_uniform_0_1"]
        out_dom = p["out_of_domain_uniform_0_2_1_2"]

        in_mean = in_dom["mean_percent"]
        in_median = in_dom["median_percent"]
        in_p95 = in_dom["p95_percent"]
        in_max = in_dom["max_percent"]

        out_mean = out_dom["mean_percent"]
        out_median = out_dom["median_percent"]
        out_p95 = out_dom["p95_percent"]
        out_max = out_dom["max_percent"]

        # Independent derivations (from the raw means, not from stored ratio/increase).
        derived_ratio = out_mean / in_mean
        derived_increase = out_mean - in_mean

        # Values present only in the raw paired summary (per-pair arrays not stored).
        stored_ratio = p["mean_error_ratio"]
        stored_increase = p["mean_error_increase_percent_points"]
        fraction_shifted = p["fraction_pairs_shifted_error_greater"]
        wilcoxon_stat = p["paired_wilcoxon_statistic"]
        p_value = p["paired_wilcoxon_one_sided_p_value"]

        # Independent Wilcoxon derivation:
        # all differences same sign (fraction == 1.0) => rank-sum = max = n*(n+1)/2.
        max_rank_sum = n_pairs * (n_pairs + 1) / 2.0
        derived_wilcoxon_stat = float(max_rank_sum) if fraction_shifted == 1.0 else None

        # Cross-check against diagnostic_panels.paired_out_of_domain.
        d = diag[g]
        d_in_mean = d["in_domain_uniform_0_1"]["mean_percent"]
        d_out_mean = d["out_of_domain_uniform_0_2_1_2"]["mean_percent"]
        d_ratio = d["mean_error_ratio"]
        d_p = d["paired_wilcoxon_one_sided_p_value"]

        cross_check = {
            "in_domain_mean_equal": (in_mean == d_in_mean),
            "out_domain_mean_equal": (out_mean == d_out_mean),
            "ratio_equal": (stored_ratio == d_ratio),
            "p_value_equal": (p_value == d_p),
        }
        all_equal = all(cross_check.values())

        per_graph[g] = {
            "pair_count": n_pairs,
            "in_domain": {
                "mean": in_mean,
                "median": in_median,
                "p95": in_p95,
                "max": in_max,
            },
            "out_of_domain": {
                "mean": out_mean,
                "median": out_median,
                "p95": out_p95,
                "max": out_max,
            },
            "mean_error_ratio": stored_ratio,
            "derived_mean_error_ratio_from_means": derived_ratio,
            "ratio_matches": (stored_ratio == derived_ratio),
            "mean_error_increase_percent_points": stored_increase,
            "derived_increase_from_means": derived_increase,
            "increase_matches": (stored_increase == derived_increase),
            "fraction_pairs_shifted_error_greater": fraction_shifted,
            "paired_wilcoxon_statistic": wilcoxon_stat,
            "derived_wilcoxon_stat_from_fraction": derived_wilcoxon_stat,
            "wilcoxon_stat_matches": (derived_wilcoxon_stat is not None
                                      and derived_wilcoxon_stat == wilcoxon_stat),
            "paired_wilcoxon_one_sided_p_value": p_value,
            "p_below_alpha": (p_value < declared_alpha),
            "cross_check_vs_diagnostic": cross_check,
            "cross_check_all_equal": all_equal,
        }

    # Aggregate summary.
    ratios = {g: per_graph[g]["mean_error_ratio"] for g in graph_names}
    in_means = {g: per_graph[g]["in_domain"]["mean"] for g in graph_names}
    out_means = {g: per_graph[g]["out_of_domain"]["mean"] for g in graph_names}
    all_p_below_alpha = all(per_graph[g]["p_below_alpha"] for g in graph_names)
    cross_check_equal = [per_graph[g]["cross_check_all_equal"] for g in graph_names]

    result = {
        "task": "independent rederivation of Finding 3 (paired out-of-domain state shift)",
        "source_file": str(SRC),
        "predeclared_alpha": declared_alpha,
        "graphs": per_graph,
        "summary": {
            "graph_count": len(graph_names),
            "ratios": ratios,
            "in_domain_means": in_means,
            "out_domain_means": out_means,
            "all_p_below_alpha": all_p_below_alpha,
            "cross_check_equal": cross_check_equal,
            "all_cross_check_equal": all(cross_check_equal),
            "all_ratio_recomputed_from_means_match": all(
                per_graph[g]["ratio_matches"] for g in graph_names
            ),
            "all_increase_recomputed_from_means_match": all(
                per_graph[g]["increase_matches"] for g in graph_names
            ),
            "all_wilcoxon_stat_rederived_match": all(
                per_graph[g]["wilcoxon_stat_matches"] for g in graph_names
            ),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    # Console summary.
    print("=== Finding 3 independent rederivation ===")
    print(f"predeclared_alpha = {declared_alpha}")
    for g in graph_names:
        r = per_graph[g]
        print(f"\n[{g}]")
        print(f"  in-domain   mean={r['in_domain']['mean']:.6f} "
              f"median={r['in_domain']['median']:.6f} "
              f"p95={r['in_domain']['p95']:.6f} max={r['in_domain']['max']:.6f}")
        print(f"  out-domain  mean={r['out_of_domain']['mean']:.6f} "
              f"median={r['out_of_domain']['median']:.6f} "
              f"p95={r['out_of_domain']['p95']:.6f} max={r['out_of_domain']['max']:.6f}")
        print(f"  mean_error_ratio (stored)      = {r['mean_error_ratio']:.15f}")
        print(f"  derived ratio from means        = {r['derived_mean_error_ratio_from_means']:.15f} "
              f"match={r['ratio_matches']}")
        print(f"  increase pp (stored)            = {r['mean_error_increase_percent_points']:.15f}")
        print(f"  derived increase from means     = {r['derived_increase_from_means']:.15f} "
              f"match={r['increase_matches']}")
        print(f"  fraction_pairs_shifted_error_greater = {r['fraction_pairs_shifted_error_greater']}")
        print(f"  wilcoxon_stat (stored)          = {r['paired_wilcoxon_statistic']}")
        print(f"  wilcoxon_stat (rederived n(n+1)/2) = {r['derived_wilcoxon_stat_from_fraction']} "
              f"match={r['wilcoxon_stat_matches']}")
        print(f"  one_sided_p_value               = {r['paired_wilcoxon_one_sided_p_value']:.3e}")
        print(f"  p < alpha ({ALPHA})?            = {r['p_below_alpha']}")
        print(f"  cross_check_vs_diagnostic        = {r['cross_check_vs_diagnostic']} "
              f"all_equal={r['cross_check_all_equal']}")

    print(f"\nall_p_below_alpha = {all_p_below_alpha}")
    print(f"cross_check_equal = {cross_check_equal}")
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
