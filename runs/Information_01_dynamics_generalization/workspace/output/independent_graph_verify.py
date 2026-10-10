#!/usr/bin/env python3
"""Independent verification of the three ER graph matrices (input structure).

Loads the three npy adjacency matrices, checks shape/dtype/symmetry/
diagonal/edge/degree statistics, verifies sha256 against the hashes declared
in data_manifest.json, and writes results to
output/results/independent_graph_verify.json.
"""

import hashlib
import json
import os

import numpy as np

PROJECT = "/data/user/conversations/conv-07581fcf-2edd-4246-8da8-dc035efd1751/project"
DATA_DIR = os.path.join(PROJECT, "data", "source_data")
MANIFEST_PATH = os.path.join(PROJECT, "data_manifest.json")
OUTPUT_DIR = os.path.join(PROJECT, "output", "results")

GRAPH_FILES = [
    "er_n_100_p_01.npy",
    "er_n_100_p_01_alt.npy",
    "er_n_100_p_06.npy",
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_expected_hashes(manifest_path):
    with open(manifest_path) as f:
        manifest = json.load(f)
    expected = {}
    for entry in manifest.get("files", []):
        name = os.path.basename(entry["path"])
        if name in GRAPH_FILES:
            expected[name] = entry["sha256"]
    missing = [n for n in GRAPH_FILES if n not in expected]
    if missing:
        raise ValueError(f"Expected hashes not found in manifest for: {missing}")
    return expected


def verify_graph(path, name, expected_hash):
    A = np.load(path)
    shape = A.shape
    dtype = str(A.dtype)
    symmetric = bool(np.array_equal(A, A.T))
    diagonal = np.diag(A)
    zero_diagonal = bool(np.all(diagonal == 0))
    n_nonzero = int(np.count_nonzero(A))
    n_diag_nonzero = int(np.count_nonzero(diagonal))
    nonzero_off_diagonal = n_nonzero - n_diag_nonzero
    undirected_edges = nonzero_off_diagonal // 2
    degree = A.sum(axis=1)
    mean_degree = float(np.sum(A) / A.shape[0])
    actual_hash = sha256_file(path)
    return {
        "file": name,
        "path": path,
        "shape": [int(s) for s in shape],
        "shape_is_100x100": list(shape) == [100, 100],
        "dtype": dtype,
        "dtype_is_int64": (dtype == "int64"),
        "symmetric": symmetric,
        "zero_diagonal": zero_diagonal,
        "nonzero_off_diagonal_entries": nonzero_off_diagonal,
        "undirected_edges": undirected_edges,
        "mean_degree": mean_degree,
        "degree_min": int(degree.min()),
        "degree_median": float(np.median(degree)),
        "degree_max": int(degree.max()),
        "sha256": actual_hash,
        "expected_sha256": expected_hash,
        "hash_match": actual_hash == expected_hash,
    }


def main():
    expected_hashes = load_expected_hashes(MANIFEST_PATH)
    results = {}
    hash_match = []
    all_passed = True
    for name in GRAPH_FILES:
        path = os.path.join(DATA_DIR, name)
        r = verify_graph(path, name, expected_hashes[name])
        results[name] = r
        hash_match.append(r["hash_match"])
        if not (
            r["shape_is_100x100"]
            and r["dtype_is_int64"]
            and r["symmetric"]
            and r["zero_diagonal"]
            and r["hash_match"]
        ):
            all_passed = False

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "independent_graph_verify.json")
    output = {
        "task": "independent_graph_verify",
        "all_checks_passed": all_passed,
        "hash_match": hash_match,
        "graphs": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(output, indent=2))
    print(f"\nWrote: {out_path}")
    print(f"all_checks_passed = {all_passed}")
    print(f"hash_match = {hash_match}")


if __name__ == "__main__":
    main()
