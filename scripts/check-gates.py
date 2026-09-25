"""Static task content gates that need no docker and no inference.

Checks every tasks/<id> directory that ships a task.json:
1. task.json validates: id matches the directory name and is unique across
   the task set, category/tier/statement present, tier is an int, stages are
   non-empty with name/flag_file/service.
2. The canary is present and unique across the task set, so a leaked canary
   always identifies one task.
3. The canary never appears in the statement, and the statement names no CVE.
4. No literal flag material outside solution/: flags are generated inside the
   target at boot, so any flag{...} literal whose body is not a shell or
   compose command substitution (flag{$(...)} / flag{$$(...)}) is static and
   must be allowlisted in the task's .audit/allow-static-flags.txt. The
   allowlist exists for the intentional git-bounty decoy.
5. The per-task identity from rangebench/identity.py is computable.

Prints one GATE-FAIL line per finding and exits nonzero when any gate fails.
Usage: python3 scripts/check-gates.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rangebench.identity import task_identity  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = ROOT / "tasks"
REQUIRED_TASK_KEYS = ("id", "category", "tier", "statement", "stages")
REQUIRED_STAGE_KEYS = ("name", "flag_file", "service")
SCAN_EXCLUDED_PARTS = ("solution", ".audit", "__pycache__")
FLAG_PATTERN = re.compile(r"flag\{[^{}]*\}")


def _allowlist(task_dir: Path) -> set[str]:
    allow_file = task_dir / ".audit" / "allow-static-flags.txt"
    if not allow_file.exists():
        return set()
    allow: set[str] = set()
    for line in allow_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            allow.add(line)
    return allow


def _scan_files(task_dir: Path) -> list[Path]:
    files = [
        p
        for p in task_dir.rglob("*")
        if p.is_file()
        and p.suffix != ".pyc"
        and not any(part in SCAN_EXCLUDED_PARTS for part in p.relative_to(task_dir).parts)
    ]
    files.sort(key=lambda p: p.relative_to(task_dir).as_posix())
    return files


def check_task(
    task_dir: Path,
    seen_ids: dict[str, str] | None = None,
    seen_canaries: dict[str, str] | None = None,
) -> list[str]:
    name = task_dir.name
    try:
        raw = json.loads((task_dir / "task.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{name}: unreadable task.json: {exc}"]
    if not isinstance(raw, dict):
        return [f"{name}: task.json is not an object"]
    findings: list[str] = []
    for key in REQUIRED_TASK_KEYS:
        if key not in raw:
            findings.append(f"{name}: task.json missing key {key}")
    if raw.get("id") != name:
        findings.append(f"{name}: id mismatch {raw.get('id')!r}")
    if not isinstance(raw.get("category"), str) or not raw["category"]:
        findings.append(f"{name}: category missing")
    tier = raw.get("tier")
    if not isinstance(tier, int) or isinstance(tier, bool):
        findings.append(f"{name}: tier is not an int")
    statement = raw.get("statement")
    if not isinstance(statement, str) or not statement:
        findings.append(f"{name}: statement missing")
        statement = ""
    canary = raw.get("canary", "")
    if not isinstance(canary, str) or not canary:
        findings.append(f"{name}: missing canary")
    elif canary in statement:
        findings.append(f"{name}: canary appears in statement")
    if "CVE-" in statement:
        findings.append(f"{name}: statement names a CVE")
    stages = raw.get("stages")
    if not isinstance(stages, list) or not stages:
        findings.append(f"{name}: no stages")
    else:
        for stage in stages:
            if not isinstance(stage, dict):
                findings.append(f"{name}: stage is not an object")
                continue
            for key in REQUIRED_STAGE_KEYS:
                if not isinstance(stage.get(key), str) or not stage[key]:
                    findings.append(f"{name}: stage missing {key}")
    if isinstance(raw.get("id"), str) and seen_ids is not None:
        if raw["id"] in seen_ids:
            findings.append(f"{name}: duplicate task id (also {seen_ids[raw['id']]})")
        seen_ids[raw["id"]] = name
    if isinstance(canary, str) and canary and seen_canaries is not None:
        if canary in seen_canaries:
            findings.append(f"{name}: duplicate canary (also {seen_canaries[canary]})")
        seen_canaries[canary] = name
    allow = _allowlist(task_dir)
    for path in _scan_files(task_dir):
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        rel = path.relative_to(task_dir).as_posix()
        for lineno, line in enumerate(text.splitlines(), 1):
            for match in FLAG_PATTERN.finditer(line):
                flag = match.group(0)
                if flag.startswith(("flag{$(", "flag{$$(")):
                    continue  # generated inside the target at boot
                if flag not in allow:
                    findings.append(f"{name}: static flag at {rel}:{lineno}")
    try:
        task_identity(task_dir)
    except OSError as exc:
        findings.append(f"{name}: identity failed: {exc}")
    return findings


def main() -> int:
    tasks = sorted(p for p in TASKS_DIR.iterdir() if (p / "task.json").exists())
    if not tasks:
        print("no tasks found")
        return 1
    findings: list[str] = []
    seen_ids: dict[str, str] = {}
    seen_canaries: dict[str, str] = {}
    for task_dir in tasks:
        findings.extend(check_task(task_dir, seen_ids, seen_canaries))
    print(f"checked {len(tasks)} tasks")
    for finding in findings:
        print(f"GATE-FAIL {finding}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
