#!/usr/bin/env python3
"""Run the original evaluator's complete no-judge input validation."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    manifest = json.loads((ROOT / "export_manifest.json").read_text(encoding="utf-8"))
    tasks = manifest["tasks"]
    if len(tasks) != manifest["task_count"] or len(tasks) != len(set(tasks)):
        raise SystemExit("export manifest task count or uniqueness mismatch")
    command = [sys.executable, "-B", str(ROOT / "TruthInsightBench" / "evaluation" / "evaluate.py")]
    for task_id in tasks:
        run_root = ROOT / "runs" / task_id
        if not (run_root / "run_receipt.json").is_file():
            raise SystemExit(f"missing exported run: {task_id}")
        command.extend(["--run-root", str(run_root)])
    with tempfile.TemporaryDirectory(prefix="tib-scoring-validate-") as directory:
        return subprocess.call([*command, "--work-dir", str(Path(directory) / "work"), "--dry-run"])


if __name__ == "__main__":
    raise SystemExit(main())
