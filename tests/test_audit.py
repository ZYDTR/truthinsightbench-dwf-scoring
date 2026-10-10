from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_exports import PATTERNS, looks_like_sdk_log


class AuditTests(unittest.TestCase):
    def test_rejects_sdk_rows_in_any_extension(self) -> None:
        row = {"type": "assistant", "sessionId": "example", "message": {"role": "assistant"}}
        self.assertTrue(looks_like_sdk_log(json.dumps(row).encode()))

    def test_rejects_wrapped_native_codex_rows(self) -> None:
        row = {"type": "response_item", "payload": {"type": "message"}}
        self.assertTrue(looks_like_sdk_log(b"Evidence\n```\n" + json.dumps(row).encode() + b"\n```"))

    def test_keeps_scientific_verification_note(self) -> None:
        self.assertFalse(looks_like_sdk_log(b"## Verdict\nIndependent scientific recomputation: confirmed.\n"))

    def test_keeps_scientific_json(self) -> None:
        self.assertFalse(looks_like_sdk_log(b'{"type":"prediction","r2":0.8}\n'))

    def test_rejects_native_workflow_journal(self) -> None:
        self.assertTrue(looks_like_sdk_log(b'{"type":"started","key":"example","agentId":"example"}\n'))

    def test_detects_windows_workstation_path(self) -> None:
        self.assertIsNotNone(PATTERNS["workstation_path"].search(b"C:\\Users\\example\\file.txt"))

    def test_detects_macos_workstation_path(self) -> None:
        self.assertIsNotNone(PATTERNS["workstation_path"].search(b"/" + b"Users/example/file.txt"))


if __name__ == "__main__":
    unittest.main()
