#!/usr/bin/env python3
"""Step 01 - Data audit and integrity check for the stage-indexed interaction table.

Task: Life_05_developmental_interaction_structure (T0 = 2022-01-01)

Reads  ../data/stage_indexed_interactions.tsv (hash-verified against
data_manifest.json), clears exact-replicated rows caused by the
(interaction_id, cell_pair_id) row-multiset expansion, and writes the audit
artifact plus the canonical de-duplicated analysis table.

Analysis unit: one (interaction_id, cell_pair_id) combination followed across
ordered stages.  Repeated stages for the same pair are dependent.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "stage_indexed_interactions.tsv"
OUT = ROOT / "output" / "results"
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED_SHA256 = "67ab6ace870c985522f905801f0eed680818a0c73225ca113057ba12fdea992f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    audit: dict[str, object] = {}

    digest = sha256(DATA)
    audit["input_path"] = str(DATA.relative_to(ROOT))
    audit["input_sha256"] = digest
    audit["input_sha256_matches_manifest"] = digest == EXPECTED_SHA256
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"input hash mismatch: {digest} != {EXPECTED_SHA256}")

    raw = pd.read_csv(DATA, sep="\t")

    # ---- schema -----------------------------------------------------------
    audit["raw_rows"] = int(len(raw))
    audit["columns"] = list(raw.columns)
    audit["dtypes"] = {c: str(t) for c, t in raw.dtypes.items()}
    audit["missing_cells_per_column"] = {
        c: int(raw[c].isna().sum()) for c in raw.columns
    }

    # ---- exact replicated rows -------------------------------------------
    # The frozen table stores some (stage, interaction, cell_pair) records as
    # an exact row multiset.  Replicates carry identical p-values AND identical
    # strength values, so they add no information and would inflate
    # record-weighted stage summaries.  Collapse to one record per unit-stage.
    key = ["stage_index", "interaction_id", "cell_pair_id"]
    repl = raw.duplicated(subset=key, keep=False)
    audit["rows_in_replicated_groups"] = int(repl.sum())
    audit["n_replicated_groups"] = int(
        raw.loc[repl, key].drop_duplicates().shape[0]
    )
    grouped = raw.groupby(key, sort=False)
    audit["all_replicates_identical_pvalue"] = bool(
        grouped["association_pvalue"].nunique().eq(1).all()
    )
    audit["all_replicates_identical_strength"] = bool(
        grouped["interaction_strength_like_value"].nunique().eq(1).all()
    )
    audit["max_replicate_multiplicity"] = int(grouped.size().max())

    units = raw.drop_duplicates(subset=key, keep="first").copy()
    units = units.sort_values(key).reset_index(drop=True)
    units["unit_id"] = units["interaction_id"] + "|" + units["cell_pair_id"]

    audit["analysis_rows"] = int(len(units))
    audit["n_stages"] = int(units["stage_index"].nunique())
    audit["stage_index_values"] = sorted(int(v) for v in units["stage_index"].unique())
    audit["n_interaction_families"] = int(units["interaction_id"].nunique())
    audit["n_cell_pairs"] = int(units["cell_pair_id"].nunique())
    audit["n_units"] = int(units["unit_id"].nunique())
    audit["rows_per_stage"] = {
        int(k): int(v) for k, v in units.groupby("stage_index").size().items()
    }

    # ---- value domains ----------------------------------------------------
    audit["association_pvalue_levels"] = sorted(
        float(v) for v in units["association_pvalue"].unique()
    )
    audit["strength_min"] = float(units["interaction_strength_like_value"].min())
    audit["strength_max"] = float(units["interaction_strength_like_value"].max())
    audit["strength_mean"] = float(units["interaction_strength_like_value"].mean())
    audit["strength_sd"] = float(units["interaction_strength_like_value"].std(ddof=1))

    # ---- design balance ---------------------------------------------------
    occ = units.groupby("unit_id").size()
    audit["stages_observed_per_unit"] = {
        "min": int(occ.min()),
        "median": float(occ.median()),
        "max": int(occ.max()),
        "n_units_all_18_stages": int((occ == 18).sum()),
        "n_units_single_stage": int((occ == 1).sum()),
    }
    # units missing any intermediate stage -> unequally spaced panel
    span = units.groupby("unit_id")["stage_index"].agg(["min", "max", "nunique"])
    span["gaps"] = (span["max"] - span["min"] + 1) - span["nunique"]
    audit["n_units_with_internal_gaps"] = int((span["gaps"] > 0).sum())

    fam_share = units.groupby("interaction_id").size() / len(units)
    audit["family_size"] = {
        "n": int(len(fam_share)),
        "max": int(units.groupby("interaction_id").size().max()),
        "median": float(units.groupby("interaction_id").size().median()),
        "top20_share_of_rows": float(fam_share.sort_values(ascending=False).head(20).sum()),
    }
    audit["cell_pair_size"] = {
        str(k): int(v) for k, v in units.groupby("cell_pair_id").size().items()
    }

    # ---- variance decomposition (nestedness) -----------------------------
    y = units["interaction_strength_like_value"]
    fam_mean = units.groupby("interaction_id")["interaction_strength_like_value"].transform("mean")
    unit_mean = units.groupby("unit_id")["interaction_strength_like_value"].transform("mean")
    total_var = float(y.var(ddof=0))
    audit["variance_decomposition"] = {
        "total_var": total_var,
        "family_mean_var": float(fam_mean.var(ddof=0)),
        "unit_mean_var": float(unit_mean.var(ddof=0)),
        "residual_within_unit_var": float((y - unit_mean).var(ddof=0)),
        "icc_family": float(fam_mean.var(ddof=0) / total_var),
        "icc_unit": float(unit_mean.var(ddof=0) / total_var),
    }
    # family vs cell-pair: which grouping absorbs more variance?
    cp_mean = units.groupby("cell_pair_id")["interaction_strength_like_value"].transform("mean")
    audit["variance_decomposition"]["cell_pair_mean_var"] = float(cp_mean.var(ddof=0))
    audit["variance_decomposition"]["icc_cell_pair"] = float(cp_mean.var(ddof=0) / total_var)

    # ---- persist ----------------------------------------------------------
    units.to_csv(OUT / "analysis_table.tsv", sep="\t", index=False)
    audit["analysis_table"] = "output/results/analysis_table.tsv"
    (OUT / "data_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()