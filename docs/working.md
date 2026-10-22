# Working log

## Changelog

### 2026-10-22: Scientific Evidence 4.0.2 context publication

- Added the frozen participant-facing context under `harnesses/scientific-evidence/4.0.2/`: the system prompt, project CLAUDE.md, two context entrypoints, core Skill, and all eight reference modules. All 13 original files (40,244 bytes) match source commit `ef5f1d78ffde83662df42da813a250360a5f8ef1`; per-file hashes are recorded in the publication manifest.
- Added package installation/dependency instructions and links from the repository README and Skill index. Kept runtime/operator infrastructure separate and identified this as the historical V4.0.2 version, without mixing in V5 or claiming reliable scientific acceptance.
- Verified 21 relative links in the package and Skill index. A temporary clean task assembly retained all 11 public Life_03 input files and all 12 project context files byte-for-byte, including the hidden `.claude/` directory. The system prompt remains an explicitly configured host input.
- Publication audit passed: no new privacy issues, 40 exact credential values checked, and all 2,686 protected historical files unchanged. Previously documented legacy exceptions remain unchanged. This release changes no scoring logic or scientific artifacts and starts no model, test conversation, or scoring run.
- The working tree was clean and based on current `origin/main` before this scoped addition. Reverting the publication commit removes the context package and its navigation entries; there is no runtime deployment or database change.

### 2026-10-20: Teach Through Breakdown publication

- Added the teaching Skill, four original reference contracts, Agent metadata, HTML generator, and bundle validator under `skills/teach-through-breakdown/`; added a discoverable Skill index and portable usage instructions.
- Adapted workstation-specific paths and conditional workspace integration; retained teaching contracts and rendering behavior. Added a self-contained evidence/delegation reference and a small wrapper for the original prose lint.
- Bundled `marked` 17.0.5 runtime files and the fixed `grapeot/writing-skill` source at `c68f46620711e10f2c16f1f39bfab5badcbae59f`, with their original MIT licenses. Excluded environment examples, caches, and repository-only Agent instructions.
- Detached-copy verification found an existing CLI entry-point bug: symlinked paths silently skipped execution. The public copy now compares resolved file identities; a regression test covers actual execution and nonzero argument errors through a symlink.
- Generator self-tests, a detached-copy HTML/manifest round trip, stale-HTML rejection, clean/dirty prose-lint cases, the new symlink regression, and all 16 bundled writing-skill tests passed. Runtime checks used only bundled dependencies, with environment overrides cleared and no npm/pip installation.
- Publication audit passed with no new issues and 40 exact credential values checked; all 2,686 protected historical files and the 11 Life_03 workspace hashes remain unchanged. Authored files pass the whitespace check; one upstream test's existing blank line at EOF is preserved to retain exact source bytes.
- This publishes reusable tooling; it does not run a scientific task, grade an answer, or certify a new article's reader/browser acceptance. Historical scoring inputs and the clean Life_03 task remain unchanged.

### 2026-10-20: Life_03 clean participant input

- Added `clean_tasks/Life_03_nutrient_response_kinetics/` as a separate participant-input package with clean-run instructions and a publication hash ledger.
- Used the original V1.0 workspace assembler: preserved all seven task files and three public runner files, then generated the standard workspace receipt. The four data files contain 144 rows and 7,685 bytes.
- Kept historical `runs/`, scoring assets, export manifests, and published scores unchanged. The new workspace contains no answers, previous outputs, evaluator assets, model configuration, or runtime logs.
- Verified all seven original task files against both the frozen source and the existing published run inputs; all three public runner files also match their source bytes. Registry checks, data hashes, row counts, and the empty-output check passed.
- Existing unit tests passed (13/13). The publication audit passed with no new issues, checked 40 exact locally configured credential values, and confirmed all 2,686 protected historical files remain unchanged; the previously documented legacy exceptions remain unchanged.
- Preserved the four original TSV files' CRLF line endings to retain their registered hashes; Git's default whitespace check flags those 148 source lines. Authored packaging files pass the whitespace check.

### 2026-10-10: historical manual review publication

- Added the original 27-task DWF manual provisional review under `reports/manual_review_20260924/`: unchanged Excel workbook, unchanged Chinese review report, sanitized JSON score summary, and source/publication hash receipt.
- Preserved all historical score values and review judgments. Replaced only the JSON summary's private source-path metadata with a repository-relative report filename; original local files remain unchanged.
- Checked all 270 cached summary fields across the workbook's 27 task rows against JSON scores, retained all eight sheets and formulas, and found zero cached error cells. No workbook recalculation or scientific regrading was performed.
- Scanned expanded XLSX entries as well as ordinary publishable files for credential patterns, 52 exact private credential values, workstation paths, and runtime-log rows. Kept the earlier documented legacy exceptions unchanged.
- The historical review covers exactly the original 27 exports, with mean 66.3 and median 68.2; it excludes the later 13 additions and the separate first-round Conversation scores. Additional review files merely named in the old report are not fabricated or claimed as included.

### 2026-10-10

- Extended the archive to all 40 released tasks by adding 13 independent-attempt exports with 49 findings; the full package has 177 findings.
- Preserved all 2,686 originally published submission/scoring files byte-for-byte, including all 588 original scoring-asset files. Kept the original adapter and scoring wrappers unchanged.
- Preserved the original 27-task manifest verbatim and added the explicitly versioned `20261010-40` extension ledger, source receipt commitments, privacy scope, and sanitized verification receipt.
- Resolved a historical aggregate scoring-tree checksum discrepancy by binding the new manifest to the actual unchanged published tree; did not rewrite the historical manifest or scorer files.
- The original evaluator dry-run prepared 40/40 runs and 5,133 scoring atoms without an external model call. Scientific scoring remains opt-in and unperformed.
- Unit tests passed (13/13), including the two original tests; source cross-checks verified all 13 new input/output trees and 52 source receipt hashes.
- The full publication audit checked 52 exact locally configured credential values and common credential patterns; no match was found. New exports contain no raw runtime logs or workstation paths.
- The owner explicitly requested that old tasks be left alone after disclosure of two already-published Energy_03 Workflow journals and an original compiled scoring cache containing a workstation path. Those three files remain unchanged under exact-hash legacy exceptions; the repository is not represented as globally free of historical logs or paths.
- Moved a newly generated validation cache outside the publication tree, retained the original tracked cache, and ignored future compiled caches. Added extension/audit tests and a fresh-clone verification gate before pushing.
- Authored packaging files passed the Git whitespace check. The whole staged archive reports 73,145 trailing-whitespace warnings in 60 frozen scientific artifact files (including original CRLF data); retained their exact source bytes rather than reformatting scientific evidence.

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
