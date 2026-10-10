# Export design

The benchmark's `evaluation/evaluate.py` requires `run_receipt.json` beside a participant `workspace/`. The DWF harness instead stores inputs in its seeded workspace and final output in an immutable collect snapshot. `scripts/export_completed.py` assembles those two trees only for tasks with `run.json` status `RELEASED`, a release receipt, and passing output validation. It refuses to overwrite any existing export.

The source evaluator stays under `TruthInsightBench/` so its relative lookup of `registry.json`, `agents/workspace_integrity.py`, specification, and frozen task assets remains unchanged. Each new run receipt binds exact input and output trees. `export_evidence.json` separately records hashes of source control-plane receipts and models seen in the frozen raw SDK JSONL; those logs are not uploaded.

The `platform_exit_code=0` field is an evaluator compatibility mapping, not a captured SAP exit code. It is explicitly marked as such. `source_release_json_sha256` and preserved release-check booleans prevent the mapping from hiding contrary source evidence.

The original 27-task snapshot is preserved in `versions/20260924-27/export_manifest.json`. The `20261010-40` extension adds 13 new run directories and updates the top-level manifest to 40 tasks without replacing any original submission or scorer bytes.

`scripts/extend_completed.py` selects exactly one genuinely released canonical or independent-attempt source per missing registered task, requiring matching task/release identity, passing validation, and a complete frozen collect snapshot. Temporary logical-task views let the original exporter handle attempt-directory names without changing that exporter or the private Run. New exports are staged outside the repository before being copied to absent destinations. The extension ledger compares all 2,686 protected files against the actual original Git commit, not only a before/after working-tree scan.

The historical manifest's aggregate scoring-tree checksum disagreed with the tree actually published in that commit. All 588 original scoring files individually matched the commit. The old manifest is preserved verbatim, and the new manifest binds the unchanged published tree's actual aggregate checksum; this is a versioned metadata correction, not a scorer change.

The read-only publication audit scans publishable file bytes for credential patterns, optionally checks exact private credential values in memory, detects serialized SDK/Codex/Workflow rows regardless of text-file extension, and verifies the frozen-base ledger. It does not copy source secrets or source logs. `privacy_scope.json` permits only the owner's documented original journal/cache exceptions while their SHA-256 hashes still match the protected baseline; credential or integrity findings cannot be waived. Scientific verification prose with a `.jsonl.md` filename is distinguished from serialized runtime rows.
