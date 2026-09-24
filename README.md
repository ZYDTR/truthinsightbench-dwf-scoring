# TruthInsightBench DWF scoring export

Public, self-contained scoring package for the first **27 released TruthInsightBench V1.0 DWF tasks**. It contains the original evaluator and frozen scoring assets, the registered task inputs, the submitted outputs, and integrity-bound run receipts. `export_manifest.json` lists the exact included tasks and input/output commitments.

The 27 tasks are **ready to score, not already scored**. Their model provenance is mixed: early runs used DeepSeek, later runs used Bonsai, and a few snapshots contain both. Each `runs/<task_id>/export_evidence.json` records models observed in the raw SDK JSONL; the large JSONL files themselves are intentionally excluded because they are not scorer inputs and can contain private runtime context.

## Layout

- `TruthInsightBench/evaluation/`: original V1.0 scorer, validator, rules, and frozen evaluator task assets.
- `TruthInsightBench/registry.json`, `agents/workspace_integrity.py`, `LICENSE`, `provenance/`: dependencies and attribution retained from the benchmark.
- `runs/<task_id>/workspace/`: registered task data and frozen final `output/` from the DWF collect snapshot.
- `runs/<task_id>/run_receipt.json`: evaluator-compatible integrity receipt.
- `runs/<task_id>/export_evidence.json`: source receipt hashes, selected snapshot, model counts, and release checks.
- `scripts/export_completed.py`: one-time, no-overwrite adapter from the original harness.

## Validate without a judge call

Requires Python 3 and Node.js 20 or newer. From the repository root:

```bash
python3 -B scripts/validate_exports.py
```

This invokes the original `evaluate.py --dry-run` for all 27 runs in a temporary work directory. It verifies the run receipts, registered task input bytes, output hashes, submission format, and scorer cohort construction. It does **not** produce scientific scores.

## Score with a judge

Install the evaluator's Python dependencies from `TruthInsightBench/evaluation/evaluator/requirements.txt`, configure an authorized judge endpoint, and run:

```bash
python3 -B scripts/score_exports.py --judge-endpoint "$JUDGE_ENDPOINT" --judge-model "$JUDGE_MODEL" --work-dir scoring-work
```

The scoring work directory must be empty. The final results are written to `scoring-work/final/final_scoring_result.json`. The default judge model in the original evaluator is `GLM-5.1-W4A8`; set the endpoint and model explicitly for reproducibility. Judge credentials belong in the local environment, never in Git.

## Receipt semantics and limits

The original evaluator expects an Agent-run `platform_exit_code=0`. SAP DWF runs instead have release, validation, and frozen snapshot receipts rather than an Agent-run process exit code. The adapter sets the required zero **only as a compatibility normalization** after checking `RELEASED`, passing validation, unchanged registered inputs, and a frozen nonempty output. It is **not** evidence that a platform process emitted exit code zero. Original release flags, including any false or unavailable `jsonl_result_event`, are preserved in `export_evidence.json`. No result is represented as a measured score until the judge path actually runs.

The evaluator and task assets are copied unchanged from TruthInsightBench V1.0. The export adapter and documentation are new. The original Apache-2.0 license and third-party notices are retained under `TruthInsightBench/`.

## Privacy

This repository is designed to be publishable with only fake examples in its instructions and configuration. The scientific submissions remain byte-identical to the released snapshots, including generic sandbox paths embedded by the agents; changing those files would break their hash commitments and alter the submitted evidence. No local workstation paths, runtime credentials, tokens, or raw SDK logs are included.
