#!/usr/bin/env python3
"""Descriptive coverage of profiles b & c plus annotation covariance structure."""
import json

import pandas as pd

DATA = "/data/user/conversations/conv-88510242-47d4-46dc-a7c4-b833ab46f409/project/data"
OUT = "/data/user/conversations/conv-88510242-47d4-46dc-a7c4-b833ab46f409/project/output"


def summarize(coords, annots, n_nodes):
    out = {"n_nodes": int(n_nodes)}
    for name in coords:
        out[f"coord_{name}_min"] = float(coords[name].min())
        out[f"coord_{name}_max"] = float(coords[name].max())
    for name in annots:
        out[f"{name}_min"] = float(annots[name].min())
        out[f"{name}_max"] = float(annots[name].max())
    return out


def main():
    b = pd.read_csv(f"{DATA}/network_profile_b.tsv", sep="\t")
    c = pd.read_csv(f"{DATA}/network_profile_c.tsv", sep="\t")
    a = pd.read_csv(f"{DATA}/network_profile_a.tsv", sep="\t")

    profile_b = summarize(
        b[["coord_x", "coord_y", "coord_z"]],
        b[["annotation_1"]],
        len(b),
    )
    profile_c = summarize(
        c[["coord_x", "coord_y", "coord_z"]],
        c[["annotation_1", "annotation_2", "annotation_3"]],
        len(c),
    )

    annot_cols = [f"annotation_{i}" for i in range(1, 6)]
    cov = a[annot_cols].corr(method="pearson")
    annotation_pairwise = {}
    for i in range(5):
        for j in range(i + 1, 5):
            key = f"{i + 1}_{j + 1}"
            annotation_pairwise[key] = float(cov.loc[annot_cols[i], annot_cols[j]])

    result = {
        "profile_b": profile_b,
        "profile_c": profile_c,
        "annotation_pairwise": annotation_pairwise,
        "n_nodes_a": int(len(a)),
    }

    with open(f"{OUT}/f5_profiles_annotations_result.json", "w") as fh:
        json.dump(result, fh, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
