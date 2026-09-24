#!/usr/bin/env python3
"""Step 1 - Data inventory, schema, integrity, and missing-data boundary.

Reads the frozen task table and emits a machine-readable profile.  No
scientific claim is made here; this establishes the analysis unit and the
usable denominators for every later step.

Outputs
-------
output/results/01_data_profile.json
output/results/01_depth_series.csv
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "observations.tsv"
OUT = ROOT / "output" / "results"
OUT.mkdir(parents=True, exist_ok=True)

COLUMNS = ["depth_m", "age_ka_bp_1", "age_ka_bp_2", "d18o_pdb"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    raw = pd.read_csv(DATA, sep="\t")
    raw.columns = COLUMNS
    df = raw.copy()

    n = len(df)
    missing = {c: int(df[c].isna().sum()) for c in df.columns}

    # ---- integrity vs the manifest -------------------------------------
    manifest = json.loads((ROOT / "data_manifest.json").read_text())
    declared = manifest["files"][0]
    integrity = {
        "declared_sha256": declared["sha256"],
        "observed_sha256": sha256(DATA),
        "declared_bytes": declared["bytes"],
        "observed_bytes": DATA.stat().st_size,
    }
    integrity["sha256_match"] = integrity["declared_sha256"] == integrity["observed_sha256"]
    integrity["bytes_match"] = integrity["declared_bytes"] == integrity["observed_bytes"]

    # ---- ordering / monotonicity ---------------------------------------
    ordering = {}
    for col in ["depth_m", "age_ka_bp_1", "age_ka_bp_2"]:
        diff = np.diff(df[col].to_numpy())
        ordering[col] = {
            "strictly_increasing": bool(np.all(diff > 0)),
            "n_nonpositive_steps": int((diff <= 0).sum()),
            "min_step": float(diff.min()),
            "max_step": float(diff.max()),
            "median_step": float(np.median(diff)),
        }

    # ---- value resolution ----------------------------------------------
    resolution = {}
    for col in COLUMNS:
        vals = df[col].dropna().to_numpy()
        resolution[col] = {
            "n_unique": int(len(np.unique(vals))),
            "quantised_to_0.01": bool(np.allclose(np.round(vals, 2), vals)),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }

    # ---- depth sampling gaps -------------------------------------------
    ddepth = np.diff(df["depth_m"].to_numpy())
    gap_idx = np.where(ddepth > 0.011)[0]
    gaps = [
        {
            "after_row_index": int(i),
            "depth_from_m": float(df["depth_m"].iloc[i]),
            "depth_to_m": float(df["depth_m"].iloc[i + 1]),
            "gap_m": float(ddepth[i]),
        }
        for i in gap_idx
    ]

    # ---- dual age-coordinate divergence ---------------------------------
    dad = (df["age_ka_bp_1"] - df["age_ka_bp_2"]).to_numpy()
    diverged = np.abs(dad) > 0.005
    depth_span = float(df["depth_m"].max() - df["depth_m"].min())
    age_div = {
        "n_rows_total": n,
        "n_rows_diverged": int(diverged.sum()),
        "frac_rows_diverged": float(diverged.mean()),
        "max_abs_diff_ka": float(np.abs(dad).max()),
        "signed_diff_min_ka": float(dad.min()),
        "signed_diff_max_ka": float(dad.max()),
        "depth_max_ka": float(df["depth_m"].to_numpy()[diverged].max()),
        "depth_min_ka": float(df["depth_m"].to_numpy()[diverged].min()),
        "frac_depth_span_affected": float(
            (df["depth_m"].to_numpy()[diverged].max()
             - df["depth_m"].to_numpy()[diverged].min()) / depth_span
        ),
        "age1_equals_age2_below_depth_m": float(
            df["depth_m"].to_numpy()[np.where(diverged)[0].max() + 1]
        ),
    }

    # ---- missing-data boundary ------------------------------------------
    miss_mask = df["d18o_pdb"].isna().to_numpy()
    idx = np.where(miss_mask)[0]
    d_miss = np.diff(idx)
    missing_block = {
        "n_missing": int(miss_mask.sum()),
        "frac_missing": float(miss_mask.mean()),
        "n_complete": int((~miss_mask).sum()),
        "isolated_missing_rows": int((d_miss > 1).sum() + (idx[0] != 0)),
        "max_run_length": int(
            max((len(g) for g in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)), default=0)
        ),
        "depth_min_m": float(df["depth_m"].to_numpy()[miss_mask].min()),
        "depth_max_m": float(df["depth_m"].to_numpy()[miss_mask].max()),
        "age_min_ka": float(df["age_ka_bp_1"].to_numpy()[miss_mask].min()),
        "age_max_ka": float(df["age_ka_bp_1"].to_numpy()[miss_mask].max()),
        "missingness_concentrated_below_depth_m": float(
            np.median(df["depth_m"].to_numpy()[miss_mask])
        ),
    }

    # ---- analysis unit statement ----------------------------------------
    median_age_step = float(np.median(np.diff(df["age_ka_bp_1"].to_numpy())))
    analysis_unit = {
        "declared_unit": "one contiguous depth or age interval",
        "median_age_step_ka": median_age_step,
        "median_depth_step_m": float(np.median(ddepth)),
        "record_span_ka": float(df["age_ka_bp_1"].max() - df["age_ka_bp_1"].min()),
        "record_span_m": float(df["depth_m"].max() - df["depth_m"].min()),
        "n_rows": n,
        "note": (
            "Rows are serially correlated at the 0.01 m / ~0.45 ka sampling step; "
            "effective independent units are contiguous intervals, not rows."
        ),
    }

    profile = {
        "step": "01_data_profile",
        "source_file": "data/observations.tsv",
        "sha256": integrity["observed_sha256"],
        "integrity": integrity,
        "n_rows": n,
        "n_columns": int(df.shape[1]),
        "missing_counts": missing,
        "ordering": ordering,
        "resolution": resolution,
        "depth_sampling_gaps": gaps,
        "dual_age_coordinate_divergence": age_div,
        "missing_data_boundary": missing_block,
        "analysis_unit": analysis_unit,
        "duplicate_rows": int(df.duplicated().sum()),
    }

    (OUT / "01_data_profile.json").write_text(
        json.dumps(profile, indent=2) + "\n", encoding="utf-8"
    )

    # tidy depth series used by the plotting step
    df.to_csv(OUT / "01_depth_series.csv", index=False)

    print(json.dumps(
        {k: profile[k] for k in
         ["n_rows", "missing_counts", "integrity"]}, indent=2))
    print("age divergence:", json.dumps(age_div, indent=2))
    print("missing block:", json.dumps(missing_block, indent=2))


if __name__ == "__main__":
    main()