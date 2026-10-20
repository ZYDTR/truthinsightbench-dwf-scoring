#!/usr/bin/env python3
"""Run the bundled, unmodified writing-skill lint without installing a package."""

from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    source = Path(__file__).resolve().parents[1] / "vendor" / "writing-skill" / "src"
    sys.path.insert(0, str(source))
    runpy.run_module("writing_skill.external_prose_lint_cli", run_name="__main__")
