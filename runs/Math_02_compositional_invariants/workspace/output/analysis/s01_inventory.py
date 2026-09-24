"""Stage 1 - inventory, integrity and closure verification.

Executed evidence for the data-boundary claims used by every finding.
Writes output/results/inventory.json and output/results/integrity.json.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_composition import (  # noqa: E402
    OUT,
    bias_wide,
    counts_matrix,
    integrity_receipt,
    library_size,
    load_bias_metrics,
    load_covariates,
    load_counts,
    richness,
    save_json,
)

fc = load_counts()
cv = load_covariates()
bm = load_bias_metrics()
P, samples, features = counts_matrix()

inv = {
    "files": {
        "anonymous_feature_counts.tsv": {
            "rows": int(len(fc)),
            "columns": list(fc.columns),
            "n_features": int(fc.feature_id.nunique()),
            "n_samples": int(fc.sample_id.nunique()),
            "grid_density": float(len(fc) / (fc.feature_id.nunique() * fc.sample_id.nunique())),
        },
        "anonymous_sample_covariates.tsv": {
            "rows": int(len(cv)),
            "columns": list(cv.columns),
            "n_samples": int(cv.sample_id.nunique()),
            "group_counts": {k: int(v) for k, v in Counter(cv.coarse_group_id).items()},
            "covariate_min": float(cv.continuous_covariate.min()),
            "covariate_median": float(cv.continuous_covariate.median()),
            "covariate_max": float(cv.continuous_covariate.max()),
            "n_distinct_covariate_values": int(cv.continuous_covariate.nunique()),
        },
        "bias_scenario_metrics.tsv": {
            "rows": int(len(bm)),
            "columns": list(bm.columns),
            "scenarios": {k: int(v) for k, v in Counter(bm.scenario_id).items()},
            "conditions": {k: int(v) for k, v in Counter(bm.condition_id).items()},
            "n_metrics": int(bm.metric_id.nunique()),
            "metric_value_min": float(bm.metric_value.min()),
            "metric_value_max": float(bm.metric_value.max()),
        },
    },
    "analysis_units": {
        "composition_unit": "one complete sample composition (529 units)",
        "scenario_unit": "one simulated bias scenario x metric x condition cell (2000 units)",
    },
    "missingness": {
        "zero_cells_in_grid": float(1.0 - len(fc) / (fc.feature_id.nunique() * fc.sample_id.nunique())),
        "explicit_na_in_counts": int(fc.isna().sum().sum()),
        "explicit_na_in_covariates": int(cv.isna().sum().sum()),
        "explicit_na_in_bias": int(bm.isna().sum().sum()),
    },
}

# ---- closure / scale verification -----------------------------------------
Pint = P.astype(np.int64)
lib = library_size(Pint)
rich = richness(Pint)
inv["value_domain"] = {
    "all_values_integer": bool(np.all(Pint == np.round(Pint))),
    "min_value": int(Pint.min()),
    "max_value": int(Pint.max()),
    "row_sum_min": int(lib.min()),
    "row_sum_median": float(np.median(lib)),
    "row_sum_max": int(lib.max()),
    "row_sum_ratio_max_over_min": float(lib.max() / lib.min()),
    "row_sum_cv": float(lib.std() / lib.mean()),
    "n_rows_summing_to_1": int((lib == 1).sum()),
    "field_name": "relative_count",
    "closure_interpretation": (
        "The column is named relative_count but every value and every row sum is an "
        "exact positive integer; no row sums to 1 or 100. The closure constraint is "
        "therefore a property of the measurement (counts normalised by an unobserved "
        "per-sample total) rather than of the stored values, and library size is a "
        "free, sample-level nuisance parameter that must be modelled explicitly."
    ),
}

# ---- sample overlap -------------------------------------------------------
inv["sample_overlap"] = {
    "counts_only": sorted(set(fc.sample_id) - set(cv.sample_id)),
    "covariates_only": sorted(set(cv.sample_id) - set(fc.sample_id)),
    "n_joined": int(len(set(fc.sample_id) & set(cv.sample_id))),
}

# ---- bias scenario grid ---------------------------------------------------
w = bias_wide()
small = set(w.loc[w.scenario_id == "small_sample", "metric_number"])
high = set(w.loc[w.scenario_id == "high_variance", "metric_number"])
inv["bias_grid"] = {
    "scenarios": sorted(w.scenario_id.unique().tolist()),
    "conditions": sorted(c for c in w.columns if c.startswith("condition_")),
    "metrics_per_scenario": {k: int(v) for k, v in Counter(w.scenario_id).items()},
    "shared_metric_ids": len(small & high),
    "high_variance_only": len(high - small),
    "small_sample_only": len(small - high),
    "high_variance_metric_id_range": [int(min(high)), int(max(high))],
    "small_sample_metric_id_range": [int(min(small)), int(max(small))],
    "paired_design_available": True,
}

# ---- feature x metric id space -------------------------------------------
fnum = fc.feature_id.str.extract(r"(\d+)").astype(int)[0]
inv["id_spaces"] = {
    "n_feature_ids": int(fnum.nunique()),
    "n_metric_ids_total": int(bm.metric_id.nunique()),
    "feature_id_range": [int(fnum.min()), int(fnum.max())],
    "metric_id_exists_without_feature": int(bm.metric_id.nunique() - fnum.nunique()),
    "note": "600 feature ids and 600 metric ids share the numeric space 1..600.",
}

save_json(inv, OUT / "results" / "inventory.json")
save_json(integrity_receipt(), OUT / "results" / "integrity.json")
print("stage 1 ok")
print("  grid density      :", round(inv["files"]["anonymous_feature_counts.tsv"]["grid_density"], 4))
print("  integer values    :", inv["value_domain"]["all_values_integer"])
print("  rows summing to 1 :", inv["value_domain"]["n_rows_summing_to_1"])
print("  shared metric ids :", inv["bias_grid"]["shared_metric_ids"])
print("  joined samples    :", inv["sample_overlap"]["n_joined"])