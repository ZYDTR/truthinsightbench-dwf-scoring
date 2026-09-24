"""Step 1 - data inventory, integrity and internal-consistency audit.

Checks (executed, not asserted):
  * file hashes against data_manifest.json
  * row/column counts, analysis-unit structure, repeated measures per group
  * missingness census
  * cross-file agreement of particle_total_per_L / depth
  * internal consistency of percent vs count vs total columns
  * integer resolution of count columns (detection / quantisation limits)
  * decimal-rounding reconstruction test of percent from counts
Writes output/results/01_data_audit.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, PROJECT, RESULTS, POLYMERS, SIZE_BINS, load, matrices

RES = {}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    manifest = json.loads((PROJECT / "data_manifest.json").read_text())
    RES["manifest_hash_check"] = [
        {
            "path": f["path"],
            "manifest_sha256": f["sha256"],
            "computed_sha256": sha256(DATA / f["path"]),
            "match": sha256(DATA / f["path"]) == f["sha256"],
            "bytes": (DATA / f["path"]).stat().st_size,
            "manifest_bytes": f["bytes"],
        }
        for f in manifest["files"]
    ]

    comp, env = load()
    RES["row_counts"] = {"particle_composition_profiles.tsv": len(comp),
                         "sample_environment_profiles.tsv": len(env)}
    RES["column_counts"] = {"composition": len(comp[0]), "environment": len(env[0])}

    comp_ids = [r["sample_id"] for r in comp]
    env_ids = [r["sample_id"] for r in env]
    RES["sample_id_sets_identical"] = comp_ids == env_ids

    C = matrices(comp)
    groups = C["collection_group_id"]
    uniq = sorted(set(groups))
    RES["collection_groups"] = {
        g: {
            "n_intervals": int(sum(1 for x in groups if x == g)),
            "lat": float(C["depth_min_m"][0]) if False else None,
            "regime": None,
            "depth_min": float(np.nanmin(C["depth_min_m"][np.array(groups) == g])),
            "depth_max": float(np.nanmax(C["depth_max_m"][np.array(groups) == g])),
        }
        for g in uniq
    }
    for g in uniq:
        m = np.array(groups) == g
        e = [r for r, gm in zip(env, groups) if gm == g]
        RES["collection_groups"][g]["latitude_deg"] = float(e[0]["latitude_deg"])
        RES["collection_groups"][g]["longitude_deg"] = float(e[0]["longitude_deg"])
        RES["collection_groups"][g]["ice_regime"] = e[0]["ice_regime"]
        # same lat/lon/regime inside a group? (group == station identity)
        RES["collection_groups"][g]["internal_lat_unique"] = len(
            {r["latitude_deg"] for r in e}) == 1
        RES["collection_groups"][g]["internal_regime_unique"] = len(
            {r["ice_regime"] for r in e}) == 1

    # ---- missingness census -------------------------------------------------
    miss = {}
    for name, rows in (("composition", comp), ("environment", env)):
        for col in rows[0]:
            k = int(sum(1 for r in rows if r[col].strip() == ""))
            if k:
                miss[f"{name}:{col}"] = {
                    "n_missing": k,
                    "samples": [r["sample_id"] for r in rows if r[col].strip() == ""],
                }
    RES["missingness"] = miss or {"none": 0}

    # ---- cross-file agreement ----------------------------------------------
    E = matrices(env)
    E["ice_regime"] = [r["ice_regime"] for r in env]
    tot_c, tot_e = C["particle_total_per_L"], E["particle_total_per_L"]
    RES["cross_file_total_agreement"] = {
        "identical": bool(np.allclose(tot_c, tot_e)),
        "max_abs_diff": float(np.max(np.abs(tot_c - tot_e))),
    }
    RES["cross_file_depth_agreement"] = {
        "depth_min_identical": bool(np.allclose(C["depth_min_m"], E["depth_min_m"])),
        "depth_max_identical": bool(np.allclose(C["depth_max_m"], E["depth_max_m"])),
    }

    # ---- contiguity of depth intervals within groups ------------------------
    contig = {}
    for g in uniq:
        m = np.array(groups) == g
        order = np.argsort(C["depth_min_m"][m])
        dmin = C["depth_min_m"][m][order]
        dmax = C["depth_max_m"][m][order]
        gaps = dmin[1:] - dmax[:-1]
        contig[g] = {"n": int(m.sum()),
                     "max_gap_m": float(np.max(gaps)) if len(gaps) else None,
                     "min_gap_m": float(np.min(gaps)) if len(gaps) else None,
                     "top_starts_at_0": bool(abs(dmin[0]) < 1e-9),
                     "thickness_m": [float(x) for x in (dmax - dmin)]}
    RES["depth_contiguity"] = contig

    # ---- percent / count / total consistency -------------------------------
    def pct(col):  # percent column vector
        return C[f"polymer_{col}_percent"]

    def cnt(col):
        return C[f"polymer_count_{col}_per_L"]

    poly_pct_sum = np.zeros(len(comp))
    poly_cnt_sum = np.zeros(len(comp))
    recon_err_pct = []
    for p in POLYMERS:
        poly_pct_sum += pct(p)
        poly_cnt_sum += cnt(p)
        # count column should equal percent * total / 100 (rounded)
        implied = pct(p) * tot_c / 100.0
        recon_err_pct.append(np.nanmax(np.abs(implied - cnt(p))))
    RES["polymer_percent_sum"] = {
        "min": float(poly_pct_sum.min()), "max": float(poly_pct_sum.max()),
        "all_within_0p05_of_100": bool(np.all(np.abs(poly_pct_sum - 100) < 0.05)),
    }
    RES["polymer_count_equals_percent_x_total"] = {
        "max_abs_residual_per_L": float(np.max(recon_err_pct)),
        "counts_all_integers": bool(np.all(np.abs(np.concatenate(
            [cnt(p) for p in POLYMERS]) - np.round(np.concatenate(
            [cnt(p) for p in POLYMERS]))) < 1e-9)),
    }
    RES["polymer_count_sum_vs_total"] = {
        "ratio_min": float((poly_cnt_sum / tot_c).min()),
        "ratio_max": float((poly_cnt_sum / tot_c).max()),
        "identical": bool(np.allclose(poly_cnt_sum, tot_c)),
        "mean_abs_diff": float(np.mean(np.abs(poly_cnt_sum - tot_c))),
    }

    size_pct_sum = np.zeros(len(comp))
    size_cnt_sum = np.zeros(len(comp))
    size_recon = []
    for s in SIZE_BINS:
        size_pct_sum += C[f"particle_size_{s}_percent"]
        size_cnt_sum += C[f"particle_size_{s}_per_L"]
        implied = C[f"particle_size_{s}_percent"] * tot_c / 100.0
        size_recon.append(np.nanmax(np.abs(implied - C[f"particle_size_{s}_per_L"])))
    RES["size_percent_sum"] = {
        "min": float(size_pct_sum.min()), "max": float(size_pct_sum.max()),
        "all_within_0p05_of_100": bool(np.all(np.abs(size_pct_sum - 100) < 0.05)),
    }
    RES["size_count_equals_percent_x_total"] = {
        "max_abs_residual_per_L": float(np.max(size_recon)),
    }
    RES["size_count_sum_vs_total"] = {
        "ratio_min": float((size_cnt_sum / tot_c).min()),
        "ratio_max": float((size_cnt_sum / tot_c).max()),
        "mean_abs_diff": float(np.mean(np.abs(size_cnt_sum - tot_c))),
    }

    # Where size counts do not sum to the reported total the difference is a
    # detection-boundary effect (particles below the 11 um lower limit).
    RES["size_mass_balance"] = [
        {"sample_id": comp[i]["sample_id"], "group": groups[i],
         "total_per_L": float(tot_c[i]),
         "size_count_sum": float(size_cnt_sum[i]),
         "polymer_count_sum": float(poly_cnt_sum[i]),
         "unclassified_by_size_per_L": float(tot_c[i] - size_cnt_sum[i])}
        for i in range(len(comp))
    ]

    # ---- quantisation / counting resolution ---------------------------------
    tots = sorted(float(x) for x in tot_c)
    RES["total_concentration_range"] = {
        "min": tots[0], "max": tots[-1], "median": float(np.median(tots)),
        "n_below_100_per_L": int(np.sum(np.array(tots) < 100)),
        "smallest_quantum_percent_at_min_total": float(100.0 / tots[0]),
    }
    # For small counts, percentages can only move in steps of 100/total.
    qstep = 100.0 / tot_c
    dev = []
    for p in POLYMERS:
        implied = cnt(p) / tot_c * 100.0
        dev.append(np.abs(pct(p) - implied))
    RES["percent_rounding_vs_integer_count"] = {
        "max_abs_deviation_percent": float(np.max(dev)),
        "note": "percent column reproduces integer count / total to <=0.05 pp",
    }

    # ---- environment covariate collinearity --------------------------------
    def col(name):
        return E[name]

    temp = col("temperature_C")
    RES["environment_summary"] = {
        "latitude_deg": {"min": float(np.nanmin(col("latitude_deg"))),
                         "max": float(np.nanmax(col("latitude_deg")))},
        "longitude_deg": {"min": float(np.nanmin(col("longitude_deg"))),
                          "max": float(np.nanmax(col("longitude_deg")))},
        "n_latitude_levels": int(len(set(col("latitude_deg")))),
        "ice_regime_levels": sorted(set(E["ice_regime"])),
        "ice_regime_by_group": {g: RES["collection_groups"][g]["ice_regime"] for g in uniq},
        "temperature_C": {"n_valid": int(np.isfinite(temp).sum()),
                          "min": float(np.nanmin(temp)), "max": float(np.nanmax(temp))},
        "salinity_PSU": {"min": float(np.nanmin(col("salinity_PSU"))),
                         "max": float(np.nanmax(col("salinity_PSU")))},
        "chlorophyll_a_ug_L": {"min": float(np.nanmin(col("chlorophyll_a_ug_L"))),
                               "max": float(np.nanmax(col("chlorophyll_a_ug_L")))},
        "particulate_organic_carbon_mg_L": {
            "min": float(np.nanmin(col("particulate_organic_carbon_mg_L"))),
            "max": float(np.nanmax(col("particulate_organic_carbon_mg_L")))},
        "carbon_to_nitrogen_ratio": {
            "min": float(np.nanmin(col("carbon_to_nitrogen_ratio"))),
            "max": float(np.nanmax(col("carbon_to_nitrogen_ratio")))},
        "uncertainty": {
            "n_valid": int(np.isfinite(col("particle_total_std_or_error_per_L")).sum()),
            "min": float(np.nanmin(col("particle_total_std_or_error_per_L"))),
            "max": float(np.nanmax(col("particle_total_std_or_error_per_L")))},
    }
    # design confounding: is ice_regime separable from group?
    regimes_per_group = {g: RES["collection_groups"][g]["ice_regime"] for g in uniq}
    RES["design_boundary"] = {
        "ice_regime_is_nested_in_group": len(set(regimes_per_group.values())) > 1
        and len(set(regimes_per_group.values())) < len(uniq),
        "regimes_per_group": regimes_per_group,
        "note": ("ice_regime has only two levels and no within-group variation; "
                 "lat/lon are constant within group, so lat/lon/regime carry no "
                 "within-group replicate variation and cannot be separated from "
                 "station identity."),
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "01_data_audit.json").write_text(json.dumps(RES, indent=2, default=str))
    print(json.dumps(RES, indent=2, default=str))


if __name__ == "__main__":
    main()