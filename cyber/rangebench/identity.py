"""Per-task identity: a content hash of one task's complete file closure.

The run manifest records whole-run provenance (harness commit, harness source
hash, task-set hash). A changed task-set hash cannot say which task changed.
The per-task identity answers exactly that, per task.

Closure rule: every file in the task directory is part of the identity —
task.json, docker-compose.yml, seed and entry scripts, app/ and src/ code,
solution/ oracles, gen.py, dind-entry.sh, and anything else the task ships.
Files are hashed in sorted relative-path order with NUL separators. Only
artifacts never shipped with a task are excluded: __pycache__/ and *.pyc
(Python caches) and .audit/ (CI gate metadata such as static-flag allowlists,
not execution content). Runtime flag values never appear in task files
because flags are generated inside the target at boot, so file content fully
determines task semantics. If a task's identity changes, old results for
that task are not comparable; other tasks are unaffected.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_IDENTITY_CHARS = 16  # same width as the whole-run task_set_hash in cli.py


def task_files(task_dir: Path) -> list[Path]:
    """Files in a task's identity closure, sorted by relative POSIX path."""
    files = [p for p in task_dir.rglob("*") if p.is_file() and _in_closure(p.relative_to(task_dir))]
    files.sort(key=lambda p: p.relative_to(task_dir).as_posix())
    return files


def _in_closure(path: Path) -> bool:
    if "__pycache__" in path.parts or path.suffix == ".pyc":
        return False
    return ".audit" not in path.parts


def task_identity(task_dir: Path) -> str:
    """Closure hash of one task's shipped files, 16 hex chars."""
    h = hashlib.sha256()
    for path in task_files(task_dir):
        h.update(path.relative_to(task_dir).as_posix().encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:_IDENTITY_CHARS]


def all_task_identities(tasks_dir: Path) -> dict[str, str]:
    """Identity for every task directory that ships a task.json, by task name."""
    return {
        p.name: task_identity(p) for p in sorted(tasks_dir.iterdir()) if (p / "task.json").exists()
    }
