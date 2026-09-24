from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.export_completed import latest_complete_snapshot, model_label


class ExportTests(unittest.TestCase):
    def test_selects_latest_complete_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            snapshots = run / "snapshots"
            incomplete = snapshots / "later-incomplete"
            incomplete.mkdir(parents=True)
            (incomplete / "output").mkdir()
            older = snapshots / "older-complete"
            older.mkdir()
            (older / "output").mkdir()
            (older / "manifest.json").write_text(json.dumps({"at": "2026-09-23T10:00:00Z"}))
            latest = snapshots / "latest-complete"
            latest.mkdir()
            (latest / "output").mkdir()
            (latest / "manifest.json").write_text(json.dumps({"at": "2026-09-23T11:00:00Z"}))
            self.assertEqual(latest_complete_snapshot(run), latest)

    def test_model_label_does_not_hide_mixed_runs(self) -> None:
        self.assertEqual(model_label(Counter({"bonsai2-27b-pq2": 2})), "bonsai2-27b-pq2")
        self.assertEqual(
            model_label(Counter({"deepseek-v4.1-flash-max": 5, "bonsai2-27b-pq2": 2})),
            "mixed:bonsai2-27b-pq2+deepseek-v4.1-flash-max",
        )


if __name__ == "__main__":
    unittest.main()
