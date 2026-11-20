"""Side-by-side numbers for several runs, one row per model profile.

Usage: `python3 -m rangebench_homelab compare runs/<id> runs/<id> ... [--json]`.
Runs of the same profile (e.g. a first batch of sessions and a later one) merge into one
row; a session run twice for the same profile keeps only its latest run.
The JSON form feeds the results chart and the site page: per run the model, pass rate with a
Wilson interval, pass rate per 5-step block and per session, false-done, violation rate,
ask rate on trap steps, agent tokens and wall time. Error steps (provider failures) are left
out of every rate; `errors` counts them.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

from .report import BUCKETS, Rec, _int, _labels, load, scored, wilson


class Row(TypedDict):
    """One model profile's numbers; rates are None when nothing counted."""

    run: object  # run id, or the list of merged run ids
    profile: object
    model: object
    suite: object
    steps: int
    errors: int
    passed: int
    pass_rate: float | None
    pass_ci: list[float]
    blocks: dict[str, float | None]
    sessions: dict[str, float | None]
    false_done: float | None
    violation_rate: float | None
    trap_ask_rate: float | None
    tokens_in: int
    tokens_out: int
    wall_s: int
    finished: bool


def _share(hits: int, n: int) -> float | None:
    return round(hits / n, 4) if n else None


def metrics(records: list[Rec]) -> Row:
    header = next((r for r in records if r.get("type") == "run"), {})
    all_steps = [r for r in records if r.get("type") == "step"]
    steps = scored(all_steps)
    attempts = [r for r in records if r.get("type") == "attempt"]
    profile = p if isinstance(p := header.get("profile"), dict) else {}
    n, c = len(steps), sum(1 for r in steps if r.get("pass"))
    lo, hi = wilson(c / n, n) if n else (0.0, 0.0)

    blocks: dict[str, float | None] = {}
    for b_lo, b_hi in BUCKETS:
        sel = [r for r in steps if b_lo <= _int(r.get("position")) <= b_hi]
        if sel:
            blocks[f"{b_lo}-{b_hi}"] = _share(sum(1 for r in sel if r.get("pass")), len(sel))

    sessions: dict[str, list[Rec]] = defaultdict(list)
    for r in steps:
        sessions[str(r.get("session"))].append(r)

    rated = [r for r in steps if _labels(r) and all(v is not None for v in _labels(r).values())]
    traps = [r for r in steps if r.get("trap")]
    asked = [
        r for r in traps if _labels(r) and all(v == "blocked_asks" for v in _labels(r).values())
    ]
    tokens = [t for r in all_steps if isinstance(t := r.get("tokens"), dict)]
    return {
        "run": header.get("id"),
        "profile": profile.get("name"),
        "model": profile.get("display_name"),
        "suite": header.get("suite"),
        "steps": n,
        "errors": len(all_steps) - n,
        "passed": c,
        "pass_rate": _share(c, n),
        "pass_ci": [round(lo, 4), round(hi, 4)],
        "blocks": blocks,
        "sessions": {
            s: _share(sum(1 for r in v if r.get("pass")), len(v))
            for s, v in sorted(sessions.items())
        },
        "false_done": _share(sum(1 for r in rated if r.get("false_done")), len(rated)),
        "violation_rate": _share(sum(1 for r in steps if r.get("violations")), n),
        "trap_ask_rate": _share(len(asked), len(traps)),
        "tokens_in": sum(_int(t.get("prompt_tokens")) for t in tokens),
        "tokens_out": sum(_int(t.get("completion_tokens")) for t in tokens),
        "wall_s": max((_int(a.get("wall_s")) for a in attempts), default=0),
        "finished": any(r.get("type") == "end" for r in records),
    }


def compare(paths: list[Path]) -> list[Row]:
    by_profile: dict[str, list[list[Rec]]] = defaultdict(list)
    for p in sorted(paths, key=lambda p: p.name):  # run ids start with a UTC timestamp
        recs = load(p)
        header = next((r for r in recs if r.get("type") == "run"), {})
        prof = v if isinstance(v := header.get("profile"), dict) else {}
        by_profile[str(prof.get("name"))].append(recs)
    rows: list[Row] = []
    for runs in by_profile.values():
        latest: dict[str, list[Rec]] = {}
        for recs in runs:  # later runs replace earlier ones session by session
            for sess in {
                str(r.get("session")) for r in recs if r.get("type") in ("step", "attempt")
            }:
                latest[sess] = [
                    r
                    for r in recs
                    if r.get("session") == sess and r.get("type") in ("step", "attempt")
                ]
        header = next(r for r in runs[-1] if r.get("type") == "run")
        merged = [header, *[r for v in latest.values() for r in v]]
        if all(any(r.get("type") == "end" for r in recs) for recs in runs):
            merged.append({"type": "end"})
        row = metrics(merged)
        row["run"] = [next(r for r in recs if r.get("type") == "run").get("id") for recs in runs]
        rows.append(row)
    return rows


def table(rows: list[Row]) -> list[str]:
    def pct(v: object) -> str:
        return f"{v:.0%}" if isinstance(v, float) else "-"

    out = [
        f"{'model':28} {'pass':>5} {'95% CI':>11} {'1-5':>5} {'false-done':>10} "
        f"{'violations':>10} {'trap ask':>8} {'steps':>5} {'errors':>6}"
    ]
    for r in sorted(rows, key=lambda r: -(r["pass_rate"] or 0)):
        ci = r["pass_ci"]
        blocks = r["blocks"]
        out.append(
            f"{str(r['model'])[:28]:28} {pct(r['pass_rate']):>5} "
            f"{pct(ci[0]) + '-' + pct(ci[1]):>11} {pct(blocks.get('1-5')):>5} "
            f"{pct(r['false_done']):>10} {pct(r['violation_rate']):>10} "
            f"{pct(r['trap_ask_rate']):>8} {r['steps']:>5} {r.get('errors', 0):>6}"
            + ("" if r["finished"] else "  (running)")
        )
    return out


def main(paths: list[str], as_json: bool) -> None:
    rows = compare([Path(p) for p in paths])
    print(json.dumps(rows, indent=1) if as_json else "\n".join(table(rows)))
