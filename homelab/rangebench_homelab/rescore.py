"""Re-score a finished run's reply-only checks against the current templates.

Usage: `python3 -m rangebench_homelab rescore runs/<id>/run.jsonl [-o out/run.jsonl]` (default
`runs/<id>-rescored/run.jsonl`). Per step, `reply` and `ask` checks are evaluated again from the
stored reply and rater labels; file, command, state and verified checks, forbids and follow-ups
keep their recorded results. pass, status and false_done are recomputed. A step whose recorded
check list (kinds and rules, in order) no longer matches the template's is kept as recorded
with a warning and `rescore_skipped`; `reply` and `ask` count as the same kind.

The resolved session needs the attempt's world params: `attempts/<session>-s<seed>/params.json`
next to the run file when present, else the current world.py is built for that seed in a
scratch dir under .work.
"""

from __future__ import annotations

import json
import random
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .checks import REPLY_ONLY, evaluate_reply
from .cli import tree_hash
from .gate import WORK_ROOT
from .report import Rec, is_false_done, load, status, step_status
from .sim import Simulator
from .template import PACKAGE_ROOT, Session, Spec, Step, load_template, template_dir


@dataclass
class Summary:
    rescored: int = 0
    to_pass: list[str] = field(default_factory=list)
    to_fail: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        out = [
            f"rescored {self.rescored} steps: {len(self.to_pass)} fail->pass, "
            f"{len(self.to_fail)} pass->fail, {len(self.skipped)} kept as recorded"
        ]
        out += [f"  fail->pass {s}" for s in self.to_pass]
        out += [f"  pass->fail {s}" for s in self.to_fail]
        out += [f"  kept {s}" for s in self.skipped]
        return out


def world_params(run_file: Path, template_id: str, seed: int) -> dict[str, str]:
    saved = run_file.parent / "attempts" / f"{template_id}-s{seed}" / "params.json"
    if saved.is_file():
        params: dict[str, str] = json.loads(saved.read_text(encoding="utf-8"))
        return params
    WORK_ROOT.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="rbrescore-", dir=WORK_ROOT))
    try:
        return load_template(template_id).build(Simulator(tmp / "world"), random.Random(seed))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def mismatch(recorded: list[Rec], checks: tuple[Spec, ...]) -> str:
    """Why the recorded checks cannot be paired with the template's, or "" when they can."""
    if len(recorded) != len(checks):
        return f"{len(recorded)} recorded checks, template has {len(checks)}"
    for i, (rec, spec) in enumerate(zip(recorded, checks, strict=True), 1):
        kinds_ok = rec.get("kind") == spec.kind or (
            rec.get("kind") in REPLY_ONLY and spec.kind in REPLY_ONLY
        )
        if not kinds_ok or rec.get("rule") != spec.rule:
            was = f"{rec.get('kind')}/{rec.get('rule')}"
            return f"check {i} was {was}, template has {spec.kind}/{spec.rule}"
    return ""


def rescore_step(rec: Rec, step: Step | None, n_raters: int) -> tuple[Rec, str]:
    """(new record, "" or the reason it is kept as recorded)."""
    recorded = rec.get("checks")
    if not isinstance(recorded, list):  # aborted or errored before checks ran
        return {**rec, "status": status(rec)}, ""
    if step is None:
        return {
            **rec,
            "status": status(rec),
            "rescore_skipped": "step not in template",
        }, "step not in template"
    checks = [c for c in recorded if isinstance(c, dict)]
    why = mismatch(checks, step.checks)
    if why:
        return {**rec, "status": status(rec), "rescore_skipped": why}, why
    raw_labels = rec.get("labels")
    labels = (
        {str(k): (v if isinstance(v, str) else None) for k, v in raw_labels.items()}
        if isinstance(raw_labels, dict)
        else {}
    )
    reply = str(rec.get("reply") or "")
    new_checks = [
        evaluate_reply(spec, reply, labels).as_dict() if spec.kind in REPLY_ONLY else old
        for old, spec in zip(checks, step.checks, strict=False)
    ]
    checks_ok = all(bool(c.get("ok")) for c in new_checks)
    passed = checks_ok and not rec.get("violations")
    return {
        **rec,
        "pass": passed,
        "status": step_status(passed, str(rec.get("end_class"))),
        "checks": new_checks,
        "false_done": is_false_done(labels, n_raters, checks_ok),
        "pass_recorded": bool(rec.get("pass")),
    }, ""


def rescore(run_file: Path) -> tuple[list[Rec], Summary]:
    records = load(run_file)
    header = next((r for r in records if r.get("type") == "run"), {})
    raters = header.get("raters")
    n_raters = len(raters) if isinstance(raters, list) else 0
    sessions: dict[tuple[str, int], Session] = {}
    summary = Summary()
    out: list[Rec] = []
    used: set[str] = set()
    for rec in records:
        if rec.get("type") == "run":
            continue
        if rec.get("type") != "step":
            out.append(rec)
            continue
        tid, seed = str(rec.get("session")), int(str(rec.get("seed") or 0))
        if (tid, seed) not in sessions:
            sessions[(tid, seed)] = load_template(tid).resolve(world_params(run_file, tid, seed))
        used.add(tid)
        step = next((s for s in sessions[(tid, seed)].steps if s.id == rec.get("step")), None)
        new, why = rescore_step(rec, step, n_raters)
        name = f"{tid}/{rec.get('step')} s{seed}"
        if why:
            summary.skipped.append(f"{name}: {why}")
            print(f"rescore: warning: {name} kept as recorded: {why}", file=sys.stderr)
        elif "pass_recorded" in new:
            summary.rescored += 1
            if new["pass"] and not new["pass_recorded"]:
                summary.to_pass.append(name)
            elif new["pass_recorded"] and not new["pass"]:
                summary.to_fail.append(name)
        out.append(new)
    stamp = {
        "at": datetime.now(UTC).isoformat(),
        "source": str(run_file),
        "harness_hash": tree_hash(PACKAGE_ROOT / "rangebench_homelab", "*.py"),
        "template_hashes": {tid: tree_hash(template_dir(tid)) for tid in sorted(used)},
    }
    return [{**header, "rescored": stamp}, *out], summary


def main(run: str, out: str | None) -> None:
    src = Path(run)
    if src.is_dir():
        src = src / "run.jsonl"
    dst = Path(out) if out else src.parent.with_name(src.parent.name + "-rescored") / "run.jsonl"
    if dst.resolve() == src.resolve():
        raise SystemExit("rescore: output would overwrite the source run file")
    records, summary = rescore(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    print("\n".join(summary.lines()))
    print(f"run file: {dst}")
