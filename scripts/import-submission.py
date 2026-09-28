#!/usr/bin/env python3
"""Validate a submission and stage an updated leaderboard data file.

Run from the harness checkout. The input leaderboard is never overwritten.
The staged file still needs the site's normal build and publication process.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rangebench.submission import validate_submission  # noqa: E402


class ImportRejected(ValueError):
    """A fixed, safe explanation of an import policy failure."""


def stage_import(
    board: dict,
    submission: dict,
    *,
    allow_partial: bool = False,
    allow_source_change: bool = False,
) -> dict:
    """Return a new board, preserving the full submission and existing rows."""
    validate_submission(submission)
    models = board.get("models")
    if not isinstance(models, list) or not models:
        raise ImportRejected("leaderboard must contain an existing task catalog")
    expected = {t["task"]: t for t in models[0]["tasks"]}
    incoming = {t["task"]: t for t in submission["tasks"]}
    if not set(incoming) <= set(expected):
        raise ImportRejected("submission contains tasks outside this leaderboard")
    summary = submission["summary"]
    if not summary["scored"]:
        raise ImportRejected("submission has no scored attempts")
    if not allow_partial and (
        set(incoming) != set(expected)
        or summary["pass_at_1_tasks"] != len(expected)
        or summary.get("missing_slots", 0) > 0
    ):
        raise ImportRejected("partial coverage; review it and pass --allow-partial explicitly")
    for name, task in incoming.items():
        for key in ("category", "tier"):
            if task.get(key) != expected[name].get(key):
                raise ImportRejected("task metadata differs from the leaderboard")
    source = submission["benchmark"]
    known = [v for k, v in board.items() if k.endswith("source") and isinstance(v, dict)]
    identities = ("harness_source_hash", "task_set_hash")
    matches = any(
        all(s.get(k) == source.get(k) and source.get(k) for k in identities) for s in known
    )
    if not matches and not allow_source_change:
        raise ImportRejected(
            "source revision differs or is unknown; review it and pass --allow-source-change"
        )
    run_id = source["run_id"]
    for model in models:
        if any(r.get("id") == run_id for r in model.get("runs", [])):
            raise ImportRejected("run already appears in this leaderboard")
        old = model.get("submission", {}).get("benchmark", {})
        if old.get("run_id") == run_id:
            raise ImportRejected("submission already imported")
    model = submission["model"]
    usage = summary["usage"]
    scored = [a for a in submission["attempts"] if a["kind"] == "task" and a["scored"]]
    tasks = []
    for name, ref in expected.items():
        task = incoming.get(name)
        tasks.append(
            {
                "task": name,
                "category": ref["category"],
                "tier": ref["tier"],
                "scored": task["scored"] if task else 0,
                "solved": task["solved"] if task else 0,
                "pass1": task["pass_at_1"] if task else None,
                "pass3": task["pass_at_3"] if task else None,
            }
        )
    output_values = [a["usage"]["output_tokens"] for a in scored]
    suffix = hashlib.sha256(run_id.encode()).hexdigest()[:12]
    row_id = f"{model['id']}:{suffix}"
    if any(m["id"] == row_id for m in models):
        raise ImportRejected("submission row already exists")
    row = {
        "id": row_id,
        "label": model["display_name"],
        "short_label": model["display_name"],
        "tasks": tasks,
        "runs": [{"id": run_id, "finished": True, "aborted": False, "rows": summary["recorded"]}],
        "recorded": summary["recorded"],
        "scored": summary["scored"],
        "solved": summary["solved"],
        "pass1": summary["pass_at_1"],
        "pass1_tasks": summary["pass_at_1_tasks"],
        "pass3": summary["pass_at_3"],
        "pass3_tasks": summary["pass_at_3_tasks"],
        "wall_min_per_attempt": sum(a["wall_s"] for a in scored) / len(scored) / 60,
        "turns_per_attempt": sum(a["turns_used"] for a in scored) / len(scored),
        "output_k_per_attempt": (
            sum(output_values) / len(scored) / 1000
            if all(v is not None for v in output_values)
            else None
        ),
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "cache_read_tokens": usage["cache_read_tokens"],
        "cache_read_reported_calls": usage["cache_read_reported_calls"],
        "api_calls": usage["api_calls"],
        "api_cost": submission["pricing"]["estimated_usd"],
        "reasoning_effort": submission["policy"]["reasoning_effort"],
        "submission": copy.deepcopy(submission),
    }
    result = copy.deepcopy(board)
    result["models"].append(row)
    result["models"].sort(
        key=lambda item: (item.get("pass1") is not None, item.get("pass1") or 0), reverse=True
    )
    rates = submission["pricing"]["usd_per_million"]
    result.setdefault("pricing", {})[row_id] = {
        target: str(rates[key] / 1e6) if rates[key] is not None else None
        for key, target in (
            ("input", "prompt"),
            ("output", "completion"),
            ("cache_read", "input_cache_read"),
            ("cache_write", "input_cache_write"),
        )
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("leaderboard", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--allow-source-change", action="store_true")
    args = parser.parse_args()
    try:
        result = stage_import(
            json.loads(args.leaderboard.read_text()),
            json.loads(args.submission.read_text()),
            allow_partial=args.allow_partial,
            allow_source_change=args.allow_source_change,
        )
        # Exclusive creation prevents clobbering either input, including symlinks.
        with args.output.open("x", encoding="utf-8") as output:
            output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except ImportRejected as exc:
        raise SystemExit(str(exc)) from None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # Avoid echoing untrusted JSON, paths or provider content on failures.
        raise SystemExit(f"import failed ({type(exc).__name__}); inputs unchanged") from None
    print("Validated submission staged. Review source revisions and coverage before publication.")


if __name__ == "__main__":
    main()
