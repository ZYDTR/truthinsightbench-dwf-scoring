# TruthInsightBench DWF scoring export

Public, self-contained scoring package for **all 40 released TruthInsightBench V1.0 DWF tasks**, covering **177 registered findings**. It contains the original evaluator and frozen scoring assets, the registered task inputs, the submitted outputs, and integrity-bound run receipts. `export_manifest.json` lists the exact included tasks and input/output commitments.

All 40 exports are ready for the original Evaluator's formal scoring. A historical **manual provisional review of the original 27** is also preserved below; it is separate from an original Evaluator judge run. The original 27 exports (128 findings) remain byte-identical to commit `ace8ab84c407201a089a45ff7e572092c6f27f87`. Their model provenance is mixed: early runs used DeepSeek, later runs used Bonsai, and a few snapshots contain both. The versioned extension adds the remaining 13 released tasks (49 findings), whose observed assistant model is `bonsai2-27b-pq2`. Each `runs/<task_id>/export_evidence.json` records model counts read from source SDK JSONL; those SDK logs are not copied into the package.

## Layout

- `TruthInsightBench/evaluation/`: original V1.0 scorer, validator, rules, and frozen evaluator task assets.
- `TruthInsightBench/registry.json`, `agents/workspace_integrity.py`, `LICENSE`, `provenance/`: dependencies and attribution retained from the benchmark.
- `runs/<task_id>/workspace/`: registered task data and frozen final `output/` from the DWF collect snapshot.
- `runs/<task_id>/run_receipt.json`: evaluator-compatible integrity receipt.
- `runs/<task_id>/export_evidence.json`: source receipt hashes, selected snapshot, model counts, and release checks.
- `versions/20260924-27/export_manifest.json`: byte-identical copy of the original 27-task manifest.
- `versions/20261010-40/`: extension manifest, complete frozen-base hash ledger, privacy scope, and verification receipt.
- `scripts/export_completed.py`: unchanged original 27-task, no-overwrite adapter.
- `scripts/extend_completed.py`: one-time versioned extension adapter; resolves released independent attempts and refuses existing destinations.
- `scripts/audit_exports.py`: read-only credential, runtime-log, and frozen-base audit.
- `reports/manual_review_20260924/`: historical manual provisional scores for the original 27 DWF exports.
- `clean_tasks/Life_03_nutrient_response_kinetics/`: original participant input and public launch/validation files for a separate clean run; contains no historical answers.

## Clean task input: Life_03

[Nutrient-response kinetics — original task and clean-run instructions](clean_tasks/Life_03_nutrient_response_kinetics/README.md) packages the original seven task files plus the public contract, launch prompt, validator, and workspace receipt. All original input bytes are preserved. Copy only its `workspace/` into a separate participant environment; the rest of this repository contains historical answers and evaluator material. This addition does not change the 40 frozen exports or perform a new scientific run or grading.

## Historical manual scores: original 27 DWF tasks

The review dated **2026-09-24** covers exactly the original 27 DWF exports and 128 registered findings. Its mean is **66.3/100** and median **68.2**. These are historical manual provisional scores, not original Evaluator judge results. The later 13 additions are outside this review, and the earlier Conversation round's scores are not mixed into it.

- [Excel score workbook](reports/manual_review_20260924/truthinsightbench_manual_scores.xlsx)
- [Chinese per-task review report](reports/manual_review_20260924/review_report_zh.md)
- [Machine-readable score summary](reports/manual_review_20260924/review_summary.json)
- [Publication provenance and hash receipt](reports/manual_review_20260924/publication_manifest.json)

The Excel workbook and review report are byte-identical to the local historical archive. The JSON summary preserves all scores and review content; its private `source` path is replaced with a relative report filename. Workbook formulas and cached values were retained, with 270 cached summary fields checked against the JSON. This publication does not recalculate or rejudge the historical review. Novelty item 5.2 was unverified and provisionally assigned zero, as explained in the report. Additional JSON/CSV files mentioned by that report were not found in the supplied local archive and are not represented as included files.

## Validate without a judge call

Requires Python 3 and Node.js 20 or newer. From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/validate_exports.py
```

This invokes the original `evaluate.py --dry-run` for all 40 runs in a temporary work directory. It verifies run receipts, registered task input bytes, output hashes, submission format, and scorer cohort construction. The 40-task dry-run prepared **177 findings and 5,133 scoring atoms**, with `external_model_called=false`. Its `READY`/provisional status is technical readiness, **not** a scientific grade or 40 passing scientific answers.

Run the adapter and audit unit tests with `python3 -B -m unittest discover -s tests -v`. Run the publication audit with `python3 -B scripts/audit_exports.py`; the optional `--source-root /path/to/private-harness` additionally checks exact locally configured credential values in memory, without printing them.

## Score with a judge

Install the evaluator's Python dependencies from `TruthInsightBench/evaluation/evaluator/requirements.txt`, configure an authorized judge endpoint, and run:

```bash
python3 -B scripts/score_exports.py --judge-endpoint "$JUDGE_ENDPOINT" --judge-model "$JUDGE_MODEL" --work-dir scoring-work
```

The scoring work directory must be empty. The final results are written to `scoring-work/final/final_scoring_result.json`. The default judge model in the original evaluator is `GLM-5.1-W4A8`; set the endpoint and model explicitly for reproducibility. Judge credentials belong in the local environment, never in Git.

## Receipt semantics and limits

The original evaluator expects an Agent-run `platform_exit_code=0`. SAP DWF runs instead have release, validation, and frozen snapshot receipts rather than an Agent-run process exit code. The adapter sets the required zero **only as a compatibility normalization** after checking `RELEASED`, passing validation, unchanged registered inputs, and a frozen nonempty output. It is **not** evidence that a platform process emitted exit code zero. Original release flags, including any false or unavailable `jsonl_result_event`, are preserved in `export_evidence.json`. No manual-review or dry-run result is represented as an original Evaluator score.

The evaluator and task assets are copied unchanged from TruthInsightBench V1.0. The export adapter and documentation are new. The original Apache-2.0 license and third-party notices are retained under `TruthInsightBench/`.

The extension binds all **2,686 original submission/scoring files**, including all 588 scoring-asset files, to their published Git bytes. The old manifest's aggregate scoring-asset checksum did not match that published tree; the preserved old manifest keeps its historical value, while the 40-task manifest records the verified unchanged tree's actual checksum. No scorer or asset file was rewritten to resolve this metadata discrepancy.

## Privacy

The **new 13 exports** contain no raw SDK/SSE/Workflow logs, runtime credentials, tokens, or local workstation paths. Only registered inputs, frozen final scientific outputs, and sanitized hash-bound receipts were added. The publication audit checked the full tracked/publishable tree against common credential patterns and 52 exact locally configured credential values; no credential match was found. Scientific outputs remain byte-identical to their released snapshots, including generic sandbox paths embedded by the agents.

**Historical exceptions are preserved, not newly introduced.** At the owner's explicit request to leave old tasks and original scoring assets unchanged, the original Energy_03 export retains `workspace/output/native/provenance/verifier_verdicts/round1_journal.jsonl` and `round2_journal.jsonl`, and the original scoring tree retains `TruthInsightBench/agents/__pycache__/workspace_integrity.cpython-314.pyc` with an embedded workstation path. Thus this repository is **not described as entirely free of historical runtime logs or workstation paths**. These three files are individually hash-bound to the original commit in the extension ledger; the privacy policy permits only these exact, unchanged legacy exceptions and never waives a credential finding. Locally generated publication-validation caches are ignored; frozen scientific output files remain preserved in full.
