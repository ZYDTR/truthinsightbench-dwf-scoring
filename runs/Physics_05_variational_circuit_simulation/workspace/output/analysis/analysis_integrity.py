#!/usr/bin/env python3
"""Integrity check for the variational quantum circuit benchmark dataset (branch: integrity).

Loads the source data file and the data manifest, verifies sha256 and byte size,
and enumerates the schema of the data file.
"""

import hashlib
import json
import os

BASE = "/data/user/conversations/conv-8ab5fa73-141f-470a-b3e5-45582443e897/project"
DATA_PATH = os.path.join(BASE, "data", "source_data", "computational_observations.json")
MANIFEST_PATH = os.path.join(BASE, "data_manifest.json")
RECEIPT_PATH = os.path.join(BASE, "output", "results", "integrity_receipt.json")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    # 1. Load manifest and data
    with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    with open(DATA_PATH, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    # 2. Hash and size verification
    file_sha = sha256_of(DATA_PATH)
    file_bytes = os.path.getsize(DATA_PATH)

    mfile = manifest["files"][0]
    manifest_sha = mfile["sha256"]
    manifest_bytes = mfile["bytes"]
    manifest_task_id = manifest["task_id"]
    manifest_t0 = manifest["t0"]

    hash_match = file_sha == manifest_sha
    bytes_match = file_bytes == manifest_bytes

    # 3. Schema enumeration
    top_level_keys = sorted(data.keys())

    print("=" * 72)
    print("INTEGRITY VERIFICATION REPORT (branch: integrity)")
    print("=" * 72)
    print(f"data file:    {DATA_PATH}")
    print(f"manifest:     {MANIFEST_PATH}")
    print("-" * 72)
    print(f"computed sha256 : {file_sha}")
    print(f"manifest sha256 : {manifest_sha}")
    print(f"hash_match      : {hash_match}")
    print(f"computed bytes  : {file_bytes}")
    print(f"manifest bytes  : {manifest_bytes}")
    print(f"bytes_match     : {bytes_match}")
    print(f"manifest task_id: {manifest_task_id}")
    print(f"manifest t0     : {manifest_t0}")
    print("-" * 72)
    print(f"top-level keys: {top_level_keys}")

    actions = data.get("actions", [])
    print(f"n_actions: {len(actions)}")
    for action in actions:
        action_id = action.get("action_id")
        cases = action.get("cases", [])
        case_count = len(cases)
        # union of keys present in each case (list of keys, then sorted)
        key_union = set()
        for case in cases:
            if isinstance(case, dict):
                key_union.update(case.keys())
        print(f"  action_id={action_id!r}, case_count={case_count}, "
              f"sorted union of case keys: {sorted(key_union)}")

    # 4. Write receipt
    receipt = {
        "file": DATA_PATH,
        "sha256": file_sha,
        "manifest_sha256": manifest_sha,
        "hash_match": hash_match,
        "bytes": file_bytes,
        "manifest_bytes": manifest_bytes,
        "manifest_task_id": manifest_task_id,
        "manifest_t0": manifest_t0,
    }
    os.makedirs(os.path.dirname(RECEIPT_PATH), exist_ok=True)
    with open(RECEIPT_PATH, "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=2)
        fh.write("\n")

    print("-" * 72)
    print(f"receipt written to: {RECEIPT_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
