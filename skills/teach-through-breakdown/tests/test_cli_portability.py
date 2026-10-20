"""Exercise the CLI through a symlink, where argv and import URLs differ."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class CliPortabilityTests(unittest.TestCase):
    def test_symlink_entry_really_executes(self):
        skill = Path(__file__).resolve().parents[1]
        env = {k: v for k, v in os.environ.items() if k not in {"MARKED_MODULE", "NODE_PATH"}}
        with tempfile.TemporaryDirectory() as tmp:
            alias = Path(tmp) / "skill-link"
            alias.symlink_to(skill, target_is_directory=True)
            entry = alias / "scripts/build_term_card_reader.mjs"
            result = subprocess.run(
                ["node", str(entry), "--self-test"], cwd=tmp, env=env,
                text=True, capture_output=True, check=True,
            )
            self.assertEqual(json.loads(result.stdout)["status"], "PASS")
            missing = subprocess.run(
                ["node", str(entry)], cwd=tmp, env=env, text=True, capture_output=True,
            )
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("--markdown", missing.stderr)


if __name__ == "__main__":
    unittest.main()
