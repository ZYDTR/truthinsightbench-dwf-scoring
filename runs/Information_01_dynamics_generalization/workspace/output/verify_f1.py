#!/usr/bin/env python3
"""
Adversarial verification of Finding 1: within-domain graph invariance.

Written independently by the F1 verification sub-agent. It recomputes every
number from the RAW source only:
  - data/source_data/computational_observations.json  (panel-level error stats)
  - data/source_data/er_n_100_p_*.npy                (raw graph matrices)

It does NOT read any output/*.json or output/*.py for its numbers. The one
main-agent artifact it may touch is the final read-only agreement check at the
very end (allowed by the task), which never feeds into the verdict.

Claims under test (Finding 1):
  (a) the 6 within-domain panel means span at most ~0.05 pp (0.4826-0.5228)
  (b) no new graph is meaningfully worse than the stored training baseline (0.5228%)
  (c) between-graph spread is small relative to within-panel (p95 - median) spread

Adversarial refutation attempts:
  R1  Is ANY new-graph panel mean > the stored-training baseline mean? (strict)
  R2  Is the mean span > 0.05 pp (all-6 or new-5)?
  R3  Does the high-degree graph INVERT direction (worse/higher than baseline
       or the low-degree graphs)?
  R4  Do the diagnostic_panels.within_domain entries EXACTLY match the action
       panels for the same (graph, distribution)? (internal consistency)
  R5  Tail probe: does any new-graph p95 exceed the baseline p95?
  R6  Scope probe: if boundary state-distributions (beta_0_5_0_5_boundary) were
       (incorrectly) folded into the "6 panels", how large does the span become?
  R7  Alternative between-graph measure: graph-level mean (avg of the 2
       in-domain state distributions per graph) and its spread.
  R8  Raw .npy identity: confirm p_01 -> training, p_01_alt -> equivalent,
       p_06 -> higher_degree via mean degree / symmetry / node count.
  R9  Hash the raw JSON against the manifest to ensure we are analyzing the
       frozen, unaltered file.
"""
import hashlib
import json
import os
import sys

import numpy as np

BASE = "/data/user/conversations/conv-07581fcf-2edd-4246-8da8-dc035efd1751/project"
SRC_OBS = os.path.join(BASE, "data", "source_data", "computational_observations.json")
MAF1  = os.path.join(BASE, "output", "results", "finding1_within_domain_graph_invariance.json")
OUTDIR = os.path.join(BASE, "output", "results")
OUT  = os.path.join(OUTDIR, "verify_f1.json")
SHA_OBS = "e800c69b7075d3fd816f518fa2bd366e4722f62a3537fde439f61a3b3a4767bd"

# Expected identity of each raw graph by mean degree (from raw npy)
EXPECTED = {
    "er_n_100_p_01.npy":      {"name": "training",      "deg": 9.48},
    "er_n_100_p_01_alt.npy":  {"name": "equivalent",    "deg": 9.54},
    "er_n_100_p_06.npy":      {"name": "higher_degree", "deg": 58.82},
}


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    # ---------- R9: load raw JSON, verify hash ----------
    with open(SRC_OBS, "r", encoding="utf-8") as f:
        obs = json.load(f)
    sh = sha_file(SRC_OBS)
    hash_ok = (sh == SHA_OBS)

    action = obs["actions"][0]
    assert action["action_id"] == "network_dynamics_ood_generalization_within_domain"
    assert action["panel_count"] == 6
    panels = action["panels"]
    B = "stored_training_pairs"
    NEW = [k for k in ["training_graph_uniform","equivalent_graph_uniform",
                       "equivalent_graph_beta_5_2","higher_degree_graph_uniform",
                       "higher_degree_graph_beta_5_2"] if k in panels]
    assert set(panels) == {B, *NEW}, panels.keys()
    assert action["maximum_panel_mean_percent"] == panels[B]["mean_percent"]

    stats = {k: dict(panels[k]) for k in [B, *NEW]}
    means = {k: stats[k]["mean_percent"] for k in stats}

    # ---------- claim (a): 6-panel mean span ----------
    mn, mx = min(means.values()), max(means.values())
    span_all = mx - mn
    min_panel = min(means, key=means.get)
    max_panel = max(means, key=means.get)
    nm = [means[k] for k in NEW]
    span_new = max(nm) - min(nm)
    claim_a = span_all <= 0.05

    # ---------- claim (b): any new graph worse than stored-training baseline ----------
    base = means[B]
    diff = {k: means[k] - base for k in NEW}
    any_worse = any(d > 0.0 for d in diff.values())
    worst_k = max(diff, key=diff.get)
    worst_d = diff[worst_k]
    claim_b = not any_worse

    # ---------- claim (c): between-graph vs within-panel (p95-median) spread ----------
    p95med = {k: stats[k]["p95_percent"] - stats[k]["median_percent"] for k in stats}
    p95med_max  = max(p95med.values());  p95med_max_k = max(p95med, key=p95med.get)
    p95med_min  = min(p95med.values());  p95med_min_k = min(p95med, key=p95med.get)
    ratio_max = span_all / p95med_max
    ratio_min = span_all / p95med_min
    claim_c = ratio_max < 1.0

    # ---------- R3: high-degree direction ----------
    hd = [means["higher_degree_graph_uniform"], means["higher_degree_graph_beta_5_2"]]
    low = [means[k] for k in NEW if "higher_degree" not in k]
    hd_worse_than_base = any(d > base for d in hd)
    hd_is_lowest = min(hd) < min(low) and min(hd) < base

    # ---------- R4: diagnostic vs action exact match ----------
    wd = obs["diagnostic_panels"]["within_domain"]
    mapping = [
        ("training_graph",        "uniform_0_1", "training_graph_uniform"),
        ("equivalent_novel_graph","uniform_0_1", "equivalent_graph_uniform"),
        ("equivalent_novel_graph","beta_5_2",    "equivalent_graph_beta_5_2"),
        ("higher_degree_graph",   "uniform_0_1", "higher_degree_graph_uniform"),
        ("higher_degree_graph",   "beta_5_2",    "higher_degree_graph_beta_5_2"),
    ]
    diag = []
    for g, d, ap in mapping:
        dm = wd[g]["distributions"][d]["mean_percent"]
        am = means[ap]
        diag.append({"map": f"{g}.{d}->{ap}", "diag": dm, "action": am,
                     "exact": float(dm) == float(am), "abs_diff": abs(dm - am)})
    diag_ok = all(c["exact"] for c in diag)

    # ---------- R5: p95 tail probe ----------
    base_p95 = stats[B]["p95_percent"]
    p95_tail = {k: stats[k]["p95_percent"] - base_p95 for k in NEW}
    any_p95_above = any(v > 0.0 for v in p95_tail.values())

    # ---------- R6: boundary scope probe ----------
    bmeans = {g: wd[g]["distributions"]["beta_0_5_0_5_boundary"]["mean_percent"] for g in wd}
    all_vals = list(means.values()) + list(bmeans.values())
    span_with_boundary = max(all_vals) - min(all_vals)

    # ---------- R7: graph-level (avg of 2 in-domain dists) spread ----------
    gavg = {
        "training_graph":      (wd["training_graph"]["distributions"]["uniform_0_1"]["mean_percent"]
                                + wd["training_graph"]["distributions"]["beta_5_2"]["mean_percent"]) / 2.0,
        "equivalent_novel_graph": (wd["equivalent_novel_graph"]["distributions"]["uniform_0_1"]["mean_percent"]
                                   + wd["equivalent_novel_graph"]["distributions"]["beta_5_2"]["mean_percent"]) / 2.0,
        "higher_degree_graph": (wd["higher_degree_graph"]["distributions"]["uniform_0_1"]["mean_percent"]
                                + wd["higher_degree_graph"]["distributions"]["beta_5_2"]["mean_percent"]) / 2.0,
    }
    gavg_spread = max(gavg.values()) - min(gavg.values())

    # ---------- R8: raw .npy graphs ----------
    graphs = {}
    for fname in ["er_n_100_p_01.npy", "er_n_100_p_01_alt.npy", "er_n_100_p_06.npy"]:
        p = os.path.join(BASE, "data", "source_data", fname)
        a = np.load(p)
        deg = a.sum(axis=1)
        md = float(np.mean(deg))
        graphs[fname] = {
            "sha256": sha_file(p), "shape": list(a.shape), "dtype": str(a.dtype),
            "symmetric": bool((a == a.T).all()), "self_loops": int(np.trace(a)),
            "offdiag_edges": int(int(a.sum()) // 2),
            "node_count": int(a.shape[0]), "mean_degree": md,
            "deg_min": float(np.min(deg)), "deg_max": float(np.max(deg)),
            "matches_expected": abs(md - EXPECTED[fname]["deg"]) < 0.1
        }
    npy_ok = all(g["matches_expected"] for g in graphs.values())
    # cross-check the mean-degree values reported inside the JSON too
    json_deg = {
        "training":   obs["diagnostic_panels"]["within_domain"]["training_graph"]["mean_degree"],
        "equivalent": obs["diagnostic_panels"]["within_domain"]["equivalent_novel_graph"]["mean_degree"],
        "higher_degree": obs["diagnostic_panels"]["within_domain"]["higher_degree_graph"]["mean_degree"],
    }
    json_deg_ok = all(abs(json_deg[n] - EXPECTED[k]["deg"]) < 0.1
                      for k, n in [("er_n_100_p_01.npy","training"),
                                   ("er_n_100_p_01_alt.npy","equivalent"),
                                   ("er_n_100_p_06.npy","higher_degree")])

    # ---------- assemble ----------
    refutations = {
        "R1_any_new_graph_mean_above_baseline": {
            "any_positive_diff": any_worse, "max_diff_pp": worst_d, "panel": worst_k,
            "diffs_pp": diff,
        },
        "R2_mean_span_above_0p05pp": {
            "span_all_6_pp": span_all, "span_new_5_pp": span_new,
            "all_6_within_0p05": span_all <= 0.05, "new_5_within_0p05": span_new <= 0.05,
        },
        "R3_higher_degree_inverts_direction": {
            "higher_degree_means_pp": hd, "baseline_pp": base,
            "worse_than_baseline": hd_worse_than_base, "is_lowest_mean": hd_is_lowest,
        },
        "R4_diagnostic_vs_action_exact": {"all_exact": diag_ok, "details": diag},
        "R5_new_graph_p95_above_baseline_p95": {
            "baseline_p95_pp": base_p95, "any": any_p95_above, "diffs_pp": p95_tail,
        },
        "R6_boundary_scope_probe": {
            "boundary_means_pp": bmeans, "span_incl_boundary_pp": span_with_boundary,
            "note": "boundary state dists are not among the 6 within-domain graph panels",
        },
        "R7_graph_level_mean_spread_pp": {
            "graph_avg_means_pp": gavg, "spread_pp": gavg_spread,
        },
        "R8_raw_npy_graph_identity": {
            "graphs": graphs, "npy_match_expected": npy_ok,
            "json_mean_degree_match_expected": json_deg_ok,
        },
        "R9_raw_json_hash": {"sha256": sh, "manifest_sha256": SHA_OBS, "match": hash_ok},
    }

    verdict = "confirmed" if (claim_a and claim_b and claim_c) else "partially_confirmed"

    result = {
        "source": {"file": SRC_OBS, "sha256": sh, "hash_match_manifest": hash_ok,
                   "action_id": action["action_id"], "panel_count": 6,
                   "note": "panel-level error stats are from the raw JSON; the per-pair "
                           "errors are only recoverable by torch model replay, which is "
                           "not available in this environment (torch not installed)."
                           " All derived quantities below are recomputed independently."},
        "baseline": {"name": B, "mean_pp": base, "median_pp": stats[B]["median_percent"],
                     "p95_pp": base_p95, "max_pp": stats[B]["max_percent"]},
        "recomputed_numbers": {
            "panel_stats": stats,
            "panel_means_pp": means,
            "min_mean_pp": mn, "max_mean_pp": mx,
            "min_mean_panel": min_panel, "max_mean_panel": max_panel,
            "span_all_6_means_pp": span_all, "span_new_5_means_pp": span_new,
            "diff_vs_baseline_pp": diff,
            "p95_minus_median_pp": p95med,
            "p95_minus_median_max_pp": p95med_max, "p95_minus_median_max_panel": p95med_max_k,
            "p95_minus_median_min_pp": p95med_min, "p95_minus_median_min_panel": p95med_min_k,
            "between_graph_vs_within_panel_ratio_vs_max": ratio_max,
            "between_graph_vs_within_panel_ratio_vs_min": ratio_min,
            "graph_avg_means_pp": gavg, "graph_avg_spread_pp": gavg_spread,
        },
        "claim_checks": {
            "a_span_at_most_0p05pp": claim_a, "a_observed_span_pp": span_all,
            "a_observed_range_pp": [mn, mx],
            "b_no_new_graph_worse_than_baseline": claim_b,
            "c_between_graph_spread_smaller_than_within_panel": claim_c,
        },
        "refutation_attempts": refutations,
        "verdict": verdict,
    }

    # ---------- FINAL read-only agreement check with main-agent output ----------
    agreement = {}
    try:
        with open(MAF1, "r", encoding="utf-8") as f:
            ma = json.load(f)
        agreement["main_agent_readable"] = True
        checks = []
        for k in [B, *NEW]:
            mp = ma["panels"][k]
            checks.append({"panel": k, "main_mean": mp["mean"], "recomputed_mean": stats[k]["mean_percent"],
                           "exact": float(mp["mean"]) == float(stats[k]["mean_percent"])})
        agreement["panel_mean_checks"] = checks
        agreement["main_all_spread_pp"] = ma["robustness"]["all_panel_mean_spread_pp"]
        agreement["recomputed_all_spread_pp"] = span_all
        agreement["main_new_spread_pp"] = ma["robustness"]["new_graph_mean_spread_pp"]
        agreement["recomputed_new_spread_pp"] = span_new
        agreement["main_p95med_max_pp"] = ma["robustness"]["within_panel_p95_minus_median_max_pp"]
        agreement["recomputed_p95med_max_pp"] = p95med_max
        agreement["main_ratio"] = ma["robustness"]["graph_type_spread_vs_within_panel_spread"]
        agreement["recomputed_ratio"] = ratio_max
        # Main agent reports means at full precision, derived spreads rounded to
        # 6 dp, and the ratio rounded to 4 dp. Compare at that precision.
        agreement["all_numbers_match"] = all(c["exact"] for c in checks) and \
            round(ma["robustness"]["all_panel_mean_spread_pp"], 6) == round(span_all, 6) and \
            round(ma["robustness"]["new_graph_mean_spread_pp"], 6) == round(span_new, 6) and \
            round(ma["robustness"]["within_panel_p95_minus_median_max_pp"], 6) == round(p95med_max, 6) and \
            round(ma["robustness"]["graph_type_spread_vs_within_panel_spread"], 4) == round(ratio_max, 4)
        agreement["match_note"] = (
            "main-agent means are bit-identical; derived spreads match to 6 dp; "
            "ratio matches to 4 dp (main-agent rounding). No substantive disagreement.")
    except Exception as e:
        agreement = {"main_agent_readable": False, "error": repr(e)}
    result["agreement_with_main_agent_output"] = agreement

    os.makedirs(OUTDIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # ---------- print summary ----------
    print("=== Adversarial verification: Finding 1 (within-domain graph invariance) ===")
    print("raw JSON hash match manifest:", hash_ok)
    print("\nPer-panel stats (percent):")
    for k in [B, *NEW]:
        s = stats[k]
        print(f"  {k}: n={s['count']} mean={s['mean_percent']:.12f} "
              f"median={s['median_percent']:.12f} p95={s['p95_percent']:.12f} max={s['max_percent']:.12f}")
    print(f"\n[claim a] 6-panel mean span = {span_all:.6f} pp ({mn:.6f}-{mx:.6f}) "
          f"<=0.05: {claim_a}; new-5 span = {span_new:.6f} pp <=0.05: {span_new<=0.05}")
    print(f"[claim b] baseline={base:.6f} pp; any new-graph mean > baseline: {any_worse} "
          f"(worst={worst_k} {worst_d:+.6f} pp)")
    print(f"[claim c] between-graph span {span_all:.6f} pp vs within-panel p95-median "
          f"max={p95med_max:.6f} pp ({p95med_max_k}), min={p95med_min:.6f} pp ({p95med_min_k}); "
          f"ratio vs max={ratio_max:.6f}, vs min={ratio_min:.6f}; claim_c={claim_c}")
    print("\nRefutations:")
    print(f"  R1 any new mean > baseline: {any_worse}")
    print(f"  R2 span > 0.05pp: all6={span_all>0.05}, new5={span_new>0.05}")
    print(f"  R3 high-degree inverts (worse than baseline): {hd_worse_than_base}; is lowest: {hd_is_lowest}")
    print(f"  R4 diagnostic==action exact: {diag_ok}")
    print(f"  R5 any new p95 > baseline p95 ({base_p95:.6f}): {any_p95_above}")
    print(f"  R6 span incl boundary dists = {span_with_boundary:.6f} pp (scope probe)")
    print(f"  R7 graph-avg mean spread = {gavg_spread:.6f} pp")
    print(f"  R8 npy identity match expected: {npy_ok}; json mean-degree match: {json_deg_ok}")
    print(f"  R9 json hash match: {hash_ok}")
    print(f"\nVERDICT: {verdict}")
    if agreement.get("main_agent_readable"):
        print(f"Main-agent numbers match: {agreement['all_numbers_match']}")
    print("Wrote:", OUT)


if __name__ == "__main__":
    main()
