# Export design

The benchmark's `evaluation/evaluate.py` requires `run_receipt.json` beside a participant `workspace/`. The DWF harness instead stores inputs in its seeded workspace and final output in an immutable collect snapshot. `scripts/export_completed.py` assembles those two trees only for tasks with `run.json` status `RELEASED`, a release receipt, and passing output validation. It refuses to overwrite any existing export.

The source evaluator stays under `TruthInsightBench/` so its relative lookup of `registry.json`, `agents/workspace_integrity.py`, specification, and frozen task assets remains unchanged. Each new run receipt binds exact input and output trees. `export_evidence.json` separately records hashes of source control-plane receipts and models seen in the frozen raw SDK JSONL; those logs are not uploaded.

The `platform_exit_code=0` field is an evaluator compatibility mapping, not a captured SAP exit code. It is explicitly marked as such. `source_release_json_sha256` and preserved release-check booleans prevent the mapping from hiding contrary source evidence.

This archive is a fixed 27-task snapshot. Future additions should be released as a distinct version or an explicitly versioned extension, never by silently replacing the existing `runs/` tree.
