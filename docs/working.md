# Working log

## Changelog

### 2026-09-24

- Established a scoring-only project separate from the live DWF harness, with public-release checks.
- Preserved the V1.0 evaluator, license, registry, and third-party notices without modifying scorer logic.
- Added a frozen, no-overwrite export adapter and evaluator wrappers for the first 27 released runs.
- Exported 27 released run roots with 128 registered findings from complete collect snapshots.
- Unit tests passed (2/2); the original evaluator dry-run prepared 27/27 runs and 3,712 scoring atoms without an external model call.
- Checked 41 locally configured secret values against the entire export; none appeared. No exported file exceeds 45 MB and no symlinks are present.
- Removed generated submission validation receipts because they contained workstation absolute paths and are not scorer inputs; the scorer regenerates them in its work directory.

## Lessons Learned

- The old harness `make_tib_run.py` hardcodes historical DeepSeek/Kimi metadata and deletes a prior export; do not use it for these releases.
- A DWF release receipt is not a captured Agent-run process exit code. Keep the compatibility mapping visible in each exported receipt.
- Some released snapshots contain more than one real assistant model; preserve observed model counts rather than assigning every task to Bonsai.
- Source snapshots can contain an incomplete earlier collect directory; select the latest snapshot that has both `manifest.json` and `output/`.
