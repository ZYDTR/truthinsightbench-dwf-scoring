"""
Analysis 00: data integrity — verify the bundled observation file against the
manifest hash, and enumerate the complete dataset schema/content.

This is the first-executed step. It is the evidence that the exact manifest-
declared file was read, not a re-keyed or truncated copy.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data" / "source_data" / "computational_observations.json"
MANIFEST = ROOT.parent / "data_manifest.json"

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

obs = json.loads(DATA.read_text())
manifest = json.loads(MANIFEST.read_text())

actual_hash = sha256(DATA)
manifest_hash = manifest["files"][0]["sha256"]
declared_bytes = manifest["files"][0]["bytes"]
actual_bytes = DATA.stat().st_size

print("=== DATA INTEGRITY ===")
print(f"file: {DATA}")
print(f"actual sha256 : {actual_hash}")
print(f"manifest sha256: {manifest_hash}")
print(f"hash match: {actual_hash == manifest_hash}")
print(f"actual bytes : {actual_bytes}")
print(f"manifest bytes: {declared_bytes}")
print(f"byte match: {actual_bytes == declared_bytes}")

# Schema enumeration
print("\n=== SCHEMA ===")
print("top-level keys:", list(obs.keys()))
for action in obs["actions"]:
    print(f"\naction_id: {action['action_id']}  case_count: {action['case_count']}")
    keys = sorted(set(k for c in action["cases"] for k in c))
    print("case keys:", keys)
    print("action-level keys:", sorted(k for k in action if k not in ("action_id", "case_count", "cases")))

# Complete inventory
print("\n=== COMPLETE INVENTORY ===")
for action in obs["actions"]:
    for c in action["cases"]:
        print(f"[{action['action_id']}] {json.dumps(c, sort_keys=True)[:160]}...")

# Write a hash receipt
receipt = {
    "file": str(DATA),
    "sha256": actual_hash,
    "manifest_sha256": manifest_hash,
    "hash_match": actual_hash == manifest_hash,
    "bytes": actual_bytes,
    "manifest_bytes": declared_bytes,
    "manifest_task_id": manifest["task_id"],
    "manifest_t0": manifest["t0"],
}
OUT = ROOT.parent / "output" / "results"
OUT.mkdir(parents=True, exist_ok=True)
OUT.joinpath("integrity_receipt.json").write_text(json.dumps(receipt, indent=2))
print(f"\nintegrity_receipt.json written; hash_match={actual_hash == manifest_hash}")
