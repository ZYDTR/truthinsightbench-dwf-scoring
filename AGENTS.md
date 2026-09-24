# TruthInsightBench DWF scoring export

This is an independent, public scoring archive, not the live DWF harness.

- `TruthInsightBench/` contains the unmodified V1.0 evaluator, registry, integrity helper, license, and provenance notices.
- `runs/<task_id>/` contains immutable scoring inputs exported from released harness runs. Never rewrite these in place; publish a new version if a correction is required.
- `scripts/export_completed.py` adapts frozen harness snapshots to the evaluator's run receipt format. It must never read credentials or active runs and must not overwrite an export.
- `docs/working.md` records changes and verification. Update it whenever code or exports change.
- Use Python 3, Node.js 20+, and the dependencies in `TruthInsightBench/evaluation/evaluator/requirements.txt` for actual scoring.
- Keep this repository publishable. Do not add raw SDK logs, tokens, user-container state, live cron files, local absolute paths, or `.local/`.
- Commit coherent, tested changes. Never create a PR or force-push.
