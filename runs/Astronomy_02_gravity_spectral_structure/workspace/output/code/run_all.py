#!/usr/bin/env python3
"""Reproduction harness: re-run every track script (track1 .. track5) in order.

Each track script is executed as a child process with ``sys.executable`` (the
interpreter running this harness), the exact command line is printed before
execution, and the full stdout/stderr of each child is streamed to this
process.  The harness exits non-zero if any child returns a non-zero status
(or is expected but missing, unless --allow-missing is given).

Discovery
---------
For track number N the harness looks, in output/code/, for files whose name
starts with ``trackN`` and ends with ``.py`` (e.g. ``track1_response_survey.py``,
``track5_figures.py``).  All matches are run in sorted order.  Override the
list entirely with the environment variable ``TRACK_SCRIPTS`` (whitespace- or
comma-separated absolute paths) or with repeated ``--only`` arguments.

Usage
-----
    python3 output/code/run_all.py                # run track1..track5
    python3 output/code/run_all.py --list         # show what would run
    python3 output/code/run_all.py --only track5  # run a single track
    python3 output/code/run_all.py --allow-missing  # missing tracks are warnings

Exit status
-----------
0  every discovered/expected track script exited 0
1  at least one track script failed or was expected but absent
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
PROJECT = CODE_DIR.parents[1]
RESULTS = PROJECT / "output" / "results"
TRACKS = [1, 2, 3, 4, 5]
DEFAULT_TIMEOUT_S = 3600


def discover(track: int, code_dir: Path) -> list[Path]:
    """All trackN*.py scripts in code_dir, sorted by name."""
    return sorted(p for p in code_dir.glob(f"track{track}*.py") if p.is_file())


def parse_only(values: list[str], code_dir: Path) -> list[Path]:
    """Resolve --only arguments (track numbers or explicit paths)."""
    scripts: list[Path] = []
    for v in values:
        p = Path(v)
        if p.suffix == ".py":
            scripts.append(p if p.is_absolute() else (code_dir / p))
        else:
            n = int(str(v).replace("track", ""))
            scripts.extend(discover(n, code_dir))
    return scripts


def build_plan(args) -> tuple[list[Path], list[int]]:
    if args.only:
        return parse_only(args.only, CODE_DIR), []

    env = os.environ.get("TRACK_SCRIPTS", "").strip()
    if env:
        items = [x for x in env.replace(",", " ").split() if x]
        scripts = [Path(x) if Path(x).is_absolute() else (CODE_DIR / x) for x in items]
        return scripts, []

    scripts, missing = [], []
    for t in TRACKS:
        found = discover(t, CODE_DIR)
        if found:
            scripts.extend(found)
        else:
            missing.append(t)
    return scripts, missing


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-run every track script in order.")
    ap.add_argument("--list", action="store_true", help="only list the plan, run nothing")
    ap.add_argument("--only", action="append", default=[],
                    help="run only this track (number like 5, or a .py path); repeatable")
    ap.add_argument("--allow-missing", action="store_true",
                    help="missing trackN scripts are warnings instead of failures")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                    help="per-script timeout in seconds (default %d)" % DEFAULT_TIMEOUT_S)
    args = ap.parse_args()

    plan, missing = build_plan(args)
    RESULTS.mkdir(parents=True, exist_ok=True)

    print(f"run_all: interpreter  {sys.executable}")
    print(f"run_all: python       {sys.version.split()[0]}")
    print(f"run_all: code dir     {CODE_DIR}")
    print(f"run_all: plan         {len(plan)} script(s)")

    if not plan:
        print("run_all: ERROR no track scripts discovered", file=sys.stderr)
        return 1
    for p in plan:
        print(f"  - {p}")
    if missing:
        print(f"run_all: tracks with no script: {missing}")
    if args.list:
        return 0

    results = []
    t_all = time.time()
    for p in plan:
        cmd = [sys.executable, str(p)]
        print("\n" + "=" * 78)
        print(f"run_all: RUN {' '.join(cmd)}")
        print(f"run_all: cwd {PROJECT}")
        print("=" * 78, flush=True)
        t0 = time.time()
        try:
            cp = subprocess.run(cmd, cwd=str(PROJECT), timeout=args.timeout)
            rc = cp.returncode
            note = "ok" if rc == 0 else f"exit {rc}"
        except subprocess.TimeoutExpired:
            rc = 124
            note = f"TIMEOUT after {args.timeout:.0f}s"
        except OSError as exc:  # e.g. missing file, not executable
            rc = 126
            note = f"OSError: {exc}"
        dt = time.time() - t0
        print(f"run_all: RESULT {p.name} -> {note} in {dt:.1f} s", flush=True)
        results.append({"script": p.name, "path": str(p), "command": " ".join(cmd),
                        "returncode": int(rc), "note": note, "elapsed_s": round(dt, 2)})

    print("\n" + "=" * 78)
    print("run_all: SUMMARY")
    for r in results:
        print(f"  {r['returncode']:>4}  {r['script']:<40} {r['note']:<24} {r['elapsed_s']:>8.1f} s")
    failed = [r for r in results if r["returncode"] != 0]
    if missing:
        status = "WARN" if args.allow_missing else "FAIL"
        print(f"  {status}  missing track scripts: {missing}")
    print(f"run_all: {len(results) - len(failed)}/{len(results)} scripts exited 0, "
          f"total {time.time() - t_all:.1f} s")

    import json
    out = RESULTS / "track5_run_all_summary.json"
    out.write_text(json.dumps({
        "interpreter": sys.executable,
        "python_version": sys.version.split()[0],
        "project": str(PROJECT),
        "plan": [str(p) for p in plan],
        "missing_tracks": missing,
        "allow_missing": bool(args.allow_missing),
        "results": results,
        "n_failed": len(failed),
        "total_elapsed_s": round(time.time() - t_all, 2),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"run_all: wrote {out}")

    if failed:
        return 1
    if missing and not args.allow_missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())