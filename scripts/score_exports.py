#!/usr/bin/env python3
"""Score all frozen exports with the original evaluator and an explicit judge."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-endpoint", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    tasks = json.loads((ROOT / "export_manifest.json").read_text(encoding="utf-8"))["tasks"]
    command = [sys.executable, "-B", str(ROOT / "TruthInsightBench" / "evaluation" / "evaluate.py")]
    for task_id in tasks:
        command.extend(["--run-root", str(ROOT / "runs" / task_id)])
    command.extend([
        "--work-dir", str(args.work_dir),
        "--judge-endpoint", args.judge_endpoint,
        "--judge-model", args.judge_model,
    ])
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
