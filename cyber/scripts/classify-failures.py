"""Classify attempt endings into the supplemental failure taxonomy.

Single source of truth is rangebench.runner.classify_end_reason. Attempts
with scored=false are excluded from the per-class counts and reported
separately as invalid, so provider/environment outages never inflate or
deflate the scored failure breakdown.

Usage:
    python3 scripts/classify-failures.py results/<run>.json
    python3 scripts/classify-failures.py results/<run-id>      # run log dir
    python3 scripts/classify-failures.py results               # latest run
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rangebench.runner import FAIL_CLASSES, classify_end_reason  # noqa: E402


def _run_doc(path: Path) -> dict:
    """Accept a run JSON, a run log directory, or a results directory."""
    if path.is_dir():
        latest = path / "latest.json"
        path = latest if latest.is_file() else path.parent / f"{path.name}.json"
    doc = json.loads(path.read_text())
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: run document is not an object")
    return doc


def _fail_class(task: dict) -> str:
    stored = task.get("fail_class")
    if isinstance(stored, str) and stored in FAIL_CLASSES:
        return stored
    return classify_end_reason(str(task.get("end_reason", "")), bool(task.get("solved")))


def _counts(tasks: list[dict]) -> dict[str, int]:
    counts = dict.fromkeys(FAIL_CLASSES, 0)
    for task in tasks:
        counts[_fail_class(task)] += 1
    return counts


def _format(counts: dict[str, int]) -> str:
    return " ".join(f"{name}={counts[name]}" for name in FAIL_CLASSES)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: classify-failures.py results/<run>.json|results/<run>|results")
        return 2
    for arg in argv[1:]:
        doc = _run_doc(Path(arg))
        tasks = [task for task in doc.get("tasks", []) if isinstance(task, dict)]
        scored = [task for task in tasks if task.get("scored", True)]
        invalid = [task for task in tasks if not task.get("scored", True)]
        line = f"{arg}: tried={len(tasks)} scored={len(scored)} {_format(_counts(scored))}"
        if invalid:
            line += f" | invalid={len(invalid)} {_format(_counts(invalid))}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
