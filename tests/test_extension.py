from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from extend_completed import select_released_source, source_view


class ExtensionTests(unittest.TestCase):
    def release(self, path: Path, task: str, status: str = "RELEASED") -> None:
        (path / "private").mkdir(parents=True)
        (path / "snapshots/s1/output").mkdir(parents=True)
        (path / "workspace").mkdir()
        (path / "reviews").mkdir()
        (path / "run.json").write_text(json.dumps({"task_id": task, "status": status}))
        (path / "private/release.json").write_text(json.dumps({"task_id": task}))
        (path / "latest-validation.json").write_text(json.dumps({"passed": True}))
        (path / "snapshots/s1/manifest.json").write_text(json.dumps({"at": "2026-10-10T00:00:00Z"}))

    def test_selects_released_attempt_not_unreleased_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.release(root / "runs/T", "T", "READY")
            attempt = root / "runs/T/attempts/a1"
            self.release(attempt, "T")
            self.assertEqual(select_released_source(root, "T"), attempt)

    def test_rejects_multiple_released_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for attempt in ["a1", "a2"]:
                self.release(root / "runs/T/attempts" / attempt, "T")
            with self.assertRaisesRegex(ValueError, "found 2"):
                select_released_source(root, "T")

    def test_rejects_wrong_release_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.release(root / "runs/T/attempts/a1", "Other")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                select_released_source(root, "T")

    def test_view_keeps_task_identity_and_only_required_release(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source/attempts/a1"
            self.release(source, "T")
            views = root / "views"
            views.mkdir()
            view = source_view(source, "T", views)
            self.assertEqual(view.name, "T")
            self.assertEqual((view / "workspace").resolve(), (source / "workspace").resolve())
            self.assertEqual([p.name for p in (view / "private").iterdir()], ["release.json"])
            with self.assertRaises(FileExistsError):
                source_view(source, "T", views)


if __name__ == "__main__":
    unittest.main()
