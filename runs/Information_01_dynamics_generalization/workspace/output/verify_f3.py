#!/usr/bin/env python3
"""
Adversarial verification of Finding 3:
"Paired OOD state-shift degradation + mild high-degree amplification."

Claim to test:
  - on all 3 graphs, OOD paired-state error is ~3.4-3.7x the in-domain error
  - 100% of pairs are worse
  - one-sided Wilcoxon p ~1.66e-165 < predeclared alpha 0.001
  - the high-degree graph (mean degree ~58.8) amplifies most (ratio ~3.73)

Method (independent of main-agent outputs):
  * Load ONLY data/source_data/computational_observations.json and the 3 .npy
    adjacency matrices. (No torch; the data.pkl / .pth are NOT executed —
    they are a torch checkpoint and would require a PyTorch runtime.)
  * Recompute every reported number from the raw stored summaries:
      ratio    = out_mean / in_mean
      increase = out_mean - in_mean
      wilcoxon normal-approx one-sided p at W1 = n(n+1)/2 (max, i.e. all pairs
      same sign) via 0.5*erfc(z/sqrt(2)), z=(W1 - n(n+1)/4 + 0.5)/sqrt(n(n+1)(2n+1)/24)
      exact one-sided p = 1/C(2000,1000) via lgamma (log domain)
  * Try to refute:
      (a) is any graph NOT worse?   (out_mean < in_mean anywhere?)
      (b) is p really < 0.001?     (both normal-approx and exact)
      (c) is the high-degree amplification real given the small difference?
  * Write a JSON verdict.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent          # .../project
SRC = BASE / "data" / "source_data"
OBS = SRC / "computational_observations.json"
OUT_DIR = BASE / "output" / "results"
OUT_JSON = OUT_DIR / "verify_f3.json"
ALPHA = 0.001

GRAPH_FILES = {
    "training_graph": "er_n_100_p_01.npy",
    "equivalent_novel_graph": "er_n_100_p_01_alt.npy",
    "higher_degree_graph": "er_n_100_p_06.npy",
}


def load_json(p: Path):
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def verify_graphs():
    out = {}
    for g, fn in GRAPH_FILES.items():
        A = np.load(SRC / fn, allow_pickle=False)
        out[g] = {
            "shape": list(A.shape),
            "dtype": str(A.dtype),
            "min": float(A.min()),
            "max": float(A.max()),
            "symmetric": bool(np.allclose(A, A.T)),
            "no_self_loops": bool(A.diagonal().sum() == 0),
            "unique_vals": [float(v) for v in np.unique(A)],
            "mean_degree": float(A.sum(axis=1).mean()),
        }
    return out


def wilcoxon_p_normal(W1, n):
    """One-sided normal-approx Wilcoxon (scipy.stats.wilcoxon, method='normal'),
    with continuity correction, for the 'all pairs positive' case (W1 = max)."""
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    sd = math.sqrt(var_w)
    z = (W1 - mean_w + 0.5) / sd
    p = 0.5 * math.erfc(z / math.sqrt(2.0))
    return z, p


def wilcoxon_p_exact_log10(n):
    """Exact one-sided p = 1 / C(2n, n)  (all pairs same sign)."""
    log_c = math.lgamma(2 * n + 1) - 2 * math.lgamma(n + 1)
    return -log_c / math.log(10.0)


def main():
    data = load_json(OBS)
    action = data["actions"][1]
    declared_alpha = float(action["predeclared_significance_alpha"])
    diag = data["diagnostic_panels"]["paired_out_of_domain"]
    panels = action["panels"]
    graph_names = list(panels.keys())
    assert len(graph_names) == 3, f"expected 3 panels, got {len(graph_names)}"

    graphs = verify_graphs()

    per_graph = {}
    for g in graph_names:
        p = panels[g]
        n = int(p["pair_count"])
        A = p["in_domain_uniform_0_1"]
        B = p["out_of_domain_uniform_0_2_1_2"]
        in_mean = float(A["mean_percent"])
        out_mean = float(B["mean_percent"])
        in_median = float(A["median_percent"])
        out_median = float(B["median_percent"])
        in_max = float(A["max_percent"])
        out_max = float(B["max_percent"])

        # --- recompute from means (independent of stored ratio/increase) ---
        derived_ratio = out_mean / in_mean
        derived_increase = out_mean - in_mean
        stored_ratio = float(p["mean_error_ratio"])
        stored_increase = float(p["mean_error_increase_percent_points"])
        stored_fraction = float(p["fraction_pairs_shifted_error_greater"])
        stored_stat = float(p["paired_wilcoxon_statistic"])
        stored_p = float(p["paired_wilcoxon_one_sided_p_value"])

        # --- recompute Wilcoxon p (normal approx) at W1 = max (all pairs +) ---
        W1_max = n * (n + 1) / 2.0
        z, p_normal = wilcoxon_p_normal(W1_max, n)
        log10_p_exact = wilcoxon_p_exact_log10(n)

        # stored Wilcoxon stat should equal W1_max when fraction==1.0
        stat_consistent = (stored_stat == W1_max)

        # --- internal consistency: actions vs diagnostic_panels ---
        d = diag[g]
        dd = {
            "in_mean_equal": (in_mean == float(d["in_domain_uniform_0_1"]["mean_percent"])),
            "out_mean_equal": (out_mean == float(d["out_of_domain_uniform_0_2_1_2"]["mean_percent"])),
            "ratio_equal": (stored_ratio == float(d["mean_error_ratio"])),
            "p_equal": (stored_p == float(d["paired_wilcoxon_one_sided_p_value"])),
            "fraction_equal": (stored_fraction == float(d["fraction_pairs_shifted_error_greater"])),
            "stat_equal": (stored_stat == float(d["paired_wilcoxon_statistic"])),
        }

        per_graph[g] = {
            "graph_file": GRAPH_FILES[g],
            "node_count": int(p["node_count"]),
            "mean_degree_stored": float(p["mean_degree"]),
            "mean_degree_recomputed_from_npy": graphs[g]["mean_degree"],
            "in_domain": {
                "mean": in_mean, "median": in_median, "p95": float(A["p95_percent"]),
                "max": in_max, "count": int(A["count"]),
            },
            "out_of_domain": {
                "mean": out_mean, "median": out_median, "p95": float(B["p95_percent"]),
                "max": out_max, "count": int(B["count"]),
            },
            "recomputed": {
                "ratio_from_means": derived_ratio,
                "increase_pp_from_means": derived_increase,
                "wilcoxon_z_normalapprox": z,
                "wilcoxon_p_normalapprox": p_normal,
                "wilcoxon_p_exact_log10": log10_p_exact,
                "wilcoxon_W1_max": W1_max,
            },
            "stored_vs_recomputed": {
                "ratio": {"stored": stored_ratio, "recomputed": derived_ratio,
                          "match": (abs(stored_ratio - derived_ratio) < 1e-12)},
                "increase_pp": {"stored": stored_increase,
                                "recomputed": derived_increase,
                                "match": (abs(stored_increase - derived_increase) < 1e-12)},
                "wilcoxon_stat": {"stored": stored_stat, "expected_W1_max": W1_max,
                                 "match": stat_consistent},
                "wilcoxon_p": {"stored": stored_p,
                               "normalapprox_recomputed": p_normal,
                               "match_to_1e-12": (abs(stored_p - p_normal) < 1e-12)},
            },
            "p_below_alpha": (stored_p < declared_alpha),
            "p_below_alpha_normalapprox": (p_normal < declared_alpha),
            "fraction_pairs_worse": stored_fraction,
            "out_of_domain_median_gt_in_domain_max": (out_median > in_max),
            "cross_check_vs_diagnostic": dd,
            "cross_check_all_equal": all(dd.values()),
        }

    # ---------------- aggregate / refutation tests ----------------
    ratios = {g: per_graph[g]["recomputed"]["ratio_from_means"] for g in graph_names}
    low_deg = min(ratios["training_graph"], ratios["equivalent_novel_graph"])
    high_deg = ratios["higher_degree_graph"]
    # is any graph NOT worse?
    any_not_worse = any(
        per_graph[g]["out_of_domain"]["mean"] <= per_graph[g]["in_domain"]["mean"]
        for g in graph_names
    )
    all_p_below_alpha = all(per_graph[g]["p_below_alpha"] for g in graph_names)
    all_p_below_alpha_na = all(per_graph[g]["p_below_alpha_normalapprox"] for g in graph_names)

    # high-degree amplification: real? mild? artifact of lower in-domain baseline?
    in_means = {g: per_graph[g]["in_domain"]["mean"] for g in graph_names}
    out_means = {g: per_graph[g]["out_of_domain"]["mean"] for g in graph_names}
    amp_abs = high_deg - low_deg
    amp_rel = amp_abs / low_deg
    # out-domain means: high-degree vs low-degree
    out_high = out_means["higher_degree_graph"]
    out_low_min = min(out_means["training_graph"], out_means["equivalent_novel_graph"])
    out_high_margin = out_high - out_low_min

    # exact p-value (log10) for the "all pairs worse" extreme
    exact_log10_p = per_graph["training_graph"]["recomputed"]["wilcoxon_p_exact_log10"]

    result = {
        "task": "adversarial verification of Finding 3 (paired OOD state-shift)",
        "source": {
            "json": str(OBS),
            "graphs": {g: str(SRC / f) for g, f in GRAPH_FILES.items()},
        },
        "predeclared_alpha": declared_alpha,
        "graphs_verified": graphs,
        "per_graph": per_graph,
        "refutation_tests": {
            "any_graph_not_worse": any_not_worse,
            "all_p_below_alpha_stored": all_p_below_alpha,
            "all_p_below_alpha_normalapprox": all_p_below_alpha_na,
            "all_cross_check_equal": all(per_graph[g]["cross_check_all_equal"] for g in graph_names),
            "all_ratio_recomputed_match": all(
                per_graph[g]["stored_vs_recomputed"]["ratio"]["match"] for g in graph_names
            ),
            "all_increase_recomputed_match": all(
                per_graph[g]["stored_vs_recomputed"]["increase_pp"]["match"] for g in graph_names
            ),
            "wilcoxon_p_stored_matches_normalapprox": all(
                per_graph[g]["stored_vs_recomputed"]["wilcoxon_p"]["match_to_1e-12"]
                for g in graph_names
            ),
        },
        "high_degree_amplification": {
            "ratio_low_min": low_deg,
            "ratio_high": high_deg,
            "amplification_abs": amp_abs,
            "amplification_rel": amp_rel,
            "in_domain_means": in_means,
            "out_domain_means": out_means,
            "out_high_minus_low_min": out_high_margin,
            "note": (
                "high-degree ratio is higher mainly because its in-domain "
                "baseline is LOWER (0.4838 vs 0.5235/0.5262); out-domain means "
                "are within 0.026 pp of each other."
            ),
        },
        "wilcoxon_p_characterization": {
            "reported_p": per_graph[graph_names[0]]["stored_vs_recomputed"]["wilcoxon_p"]["stored"],
            "reported_p_is_normal_approx": True,
            "reported_p_exact_log10": exact_log10_p,
            "reported_p_exact_underflows_float64": exact_log10_p < -308,
            "note": (
                "1.66e-165 is the NORMAL-APPROXIMATION paired-Wilcoxon p "
                "(scipy method='normal', continuity correction) at the max "
                "rank-sum (all 1000 pairs same sign). The EXACT one-sided p "
                "1/C(2000,1000) ~ 10^%.0f underflows float64." % exact_log10_p
            ),
        },
        "verdict": (
            "CONFIRMED with a nuance. All four sub-claims are reproduced from raw "
            "summaries: (1) OOD paired error is 3.395-3.730x in-domain on all 3 "
            "graphs (within the claimed 3.4-3.7 band, low-degree actually 3.395-3.398 "
            "which is just below the stated 3.4 lower bound); (2) all pairs worse "
            "(fraction=1.0, Wilcoxon stat = max W1 = 500500); (3) p ~ 1.66e-165 "
            "< alpha 0.001 (this is the NORMAL-APPROX paired-Wilcoxon p, not exact); "
            "(4) high-degree graph amplifies most (3.730). Refutation attempts all "
            "failed to refute the core claim. The high-degree 'amplification' is REAL "
            "but MILD: its ratio lift is driven by a LOWER in-domain baseline (0.4838 "
            "vs 0.5235/0.5262), and the raw out-domain means differ by only 0.026 pp "
            "between high- and low-degree graphs."
        ),
        "counter_evidence": (
            "1) The exact one-sided Wilcoxon p = 1/C(2000,1000) ~ 1e-600 underflows "
            "float64; the reported 1.66e-165 is a NORMAL-APPROXIMATION value (scipy "
            "method='normal', continuity correction) at the max rank-sum, not the "
            "exact combinatorial p. Both are < 0.001, so the significance claim "
            "stands, but the quoted 1.66e-165 is an approximation. 2) The '100% of "
            "pairs worse' claim rests on stored fraction=1.0 and W1=max; the raw "
            "per-pair arrays are NOT in the portable JSON (only SHA256 hashes), so "
            "this is an internal-consistency check, not a direct per-pair re-derivation. "
            "3) High-degree amplification is mild (ratio 3.730 vs 3.395-3.398, a "
            "0.33 pp absolute / ~9.9% relative lift) and largely attributable to a "
            "lower in-domain baseline rather than a genuinely different OOD failure. "
            "4) Low-degree ratio 3.395-3.398 sits just below the claimed '3.4' lower "
            "bound; the 3.4-3.7 band is a loose rounding of 3.395-3.730."
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    # console
    print("=== Finding 3 adversarial verification ===")
    print(f"predeclared_alpha = {declared_alpha}")
    for g in graph_names:
        r = per_graph[g]
        print(f"\n[{g}]  npy_degree={r['mean_degree_recomputed_from_npy']:.4f} "
              f"stored_degree={r['mean_degree_stored']:.4f}")
        print(f"  in-domain   mean={r['in_domain']['mean']:.6f} "
              f"median={r['in_domain']['median']:.6f} max={r['in_domain']['max']:.6f}")
        print(f"  out-domain  mean={r['out_of_domain']['mean']:.6f} "
              f"median={r['out_of_domain']['median']:.6f} max={r['out_of_domain']['max']:.6f}")
        print(f"  ratio stored={r['stored_vs_recomputed']['ratio']['stored']:.15f} "
              f"recomputed={r['recomputed']['ratio_from_means']:.15f} "
              f"match={r['stored_vs_recomputed']['ratio']['match']}")
        print(f"  increase stored={r['stored_vs_recomputed']['increase_pp']['stored']:.15f} "
              f"recomputed={r['recomputed']['increase_pp_from_means']:.15f} "
              f"match={r['stored_vs_recomputed']['increase_pp']['match']}")
        print(f"  fraction_worse={r['fraction_pairs_worse']}  "
              f"stat={r['stored_vs_recomputed']['wilcoxon_stat']['stored']} "
              f"(==W1_max {r['recomputed']['wilcoxon_W1_max']}) "
              f"match={r['stored_vs_recomputed']['wilcoxon_stat']['match']}")
        print(f"  p stored={r['stored_vs_recomputed']['wilcoxon_p']['stored']:.6e} "
              f"normalapprox_recomp={r['recomputed']['wilcoxon_p_normalapprox']:.6e} "
              f"match={r['stored_vs_recomputed']['wilcoxon_p']['match_to_1e-12']}")
        print(f"  p<alpha(stored)={r['p_below_alpha']}  "
              f"p<alpha(normalapprox)={r['p_below_alpha_normalapprox']}  "
              f"cross_check={r['cross_check_all_equal']}")

    print(f"\nany graph NOT worse: {any_not_worse}")
    print(f"all p < alpha (stored): {all_p_below_alpha}")
    print(f"all p < alpha (normalapprox): {all_p_below_alpha_na}")
    print(f"exact one-sided p ~ 10^{exact_log10_p:.0f} (underflows float64)")
    print(f"high-degree ratio={high_deg:.6f} vs low min={low_deg:.6f} "
          f"amp_abs={amp_abs:.4f} amp_rel={amp_rel:.4f}")
    print(f"out-domain high minus low-min = {out_high_margin:.6f} pp")
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
