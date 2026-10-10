#!/usr/bin/env python3
"""Aggregate the 5 per-dimension JSON artifacts into final_results.json."""
import json
import os

OUT_DIR = "/data/user/conversations/conv-88510242-47d4-46dc-a7c4-b833ab46f409/project/output"

DIMENSION_KEY_TO_ARTIFACT = {
    "f1_strength_distance": "f1_strength_distance_result.json",
    "f2_spatial_gradient": "f2_spatial_gradient_result.json",
    "f3_homophily_null": "f3_homophily_null_result.json",
    "f4_trimming": "f4_trimming_result.json",
    "f5_profiles_annotations": "f5_profiles_annotations_result.json",
}

FINDINGS_TO_EVIDENCE = {
    "1": {"artifact": "f1_strength_distance_result.json", "script": "f1_strength_distance.py"},
    "2": {"artifact": "f2_spatial_gradient_result.json", "script": "f2_spatial_gradient.py"},
    "3": {"artifact": "f3_homophily_null_result.json", "script": "f3_homophily_null.py"},
    "4": {"artifact": "f4_trimming_result.json", "script": "f4_trimming.py"},
    "5": {"artifact": "f5_profiles_annotations_result.json", "script": "f5_profiles_annotations.py"},
}

META = {
    "task_id": "Information_04_connectome_assortative_structure",
    "n_nodes": 800,
    "n_systems": 4,
    "nodes_per_system": 200,
}


def main():
    consolidated = {}
    missing = []
    loaded = 0

    for key, artifact in DIMENSION_KEY_TO_ARTIFACT.items():
        path = os.path.join(OUT_DIR, artifact)
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            consolidated[key] = data
            loaded += 1
        except Exception as exc:
            missing.append(artifact)

    consolidated["meta"] = dict(META)
    consolidated["meta"]["n_artifacts_loaded"] = loaded
    consolidated["meta"]["missing"] = missing
    consolidated["meta"]["findings_to_evidence"] = FINDINGS_TO_EVIDENCE

    out_path = os.path.join(OUT_DIR, "final_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2, ensure_ascii=False)

    print("Wrote", out_path)
    print("n_artifacts_loaded =", loaded, "missing =", missing)


if __name__ == "__main__":
    main()
