"""CLI: list / identities / check (oracle) / run (agent) / smoke."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from .agent import AnthropicChatClient, ChatClient, ChatClientProtocol, Usage
from .env import TASKS_DIR, EnvError, Task, load_all, load_task
from .identity import all_task_identities
from .launch import cmd_launch
from .progress import Progress
from .runner import (
    DEFAULT_CTX_WINDOW,
    DEFAULT_KEEP_TAIL,
    DEFAULT_RESERVE,
    DEFAULT_THRESHOLD,
    FAIL_CLASSES,
    FAIL_SKIPPED,
    MAX_CTX_WINDOW,
    MAX_WALL_CLOCK_SCALE,
    AttemptResult,
    classify_end_reason,
    run_attempt,
    run_oracle,
    transcript_tps,
    wall_clock_scale_for,
)
from .submission import build_submission, public_price_lookup, validate_submission

RESULTS = Path(__file__).resolve().parent.parent / "results"
PROBE_MESSAGES = [{"role": "user", "content": "Reply with exactly: COMMAND:\necho ok"}]
PROBE_MAX_TOKENS = 32768
# Unscored tier-1 task whose agent attempt measures tokens/sec in reference mode.
PROBE_TASK = "jwt-none"


def _probe_once(client: ChatClientProtocol) -> tuple[float, str, Usage, str | None]:
    """One probe call: (elapsed seconds, content, usage, error)."""
    start = time.perf_counter()
    content, usage, err = client.chat(PROBE_MESSAGES, PROBE_MAX_TOKENS)
    return time.perf_counter() - start, content, usage, err


def _as_number(value: object) -> float | None:
    """The value as a float if it is a JSON number (not a bool), else None."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _load_wall_clock_reference(path: str) -> dict:
    """Read and validate a --wall-clock-reference file; SystemExit on any problem."""
    try:
        reference = json.loads(Path(path).read_text())
    except FileNotFoundError:
        raise SystemExit(f"--wall-clock-reference {path}: file not found") from None
    except (OSError, ValueError) as exc:
        raise SystemExit(f"--wall-clock-reference {path}: cannot read JSON: {exc}") from None
    if not isinstance(reference, dict):
        raise SystemExit(f"--wall-clock-reference {path}: expected a JSON object")
    tps = _as_number(reference.get("reference_tps"))
    if tps is None or not math.isfinite(tps) or tps <= 0:
        raise SystemExit(f"--wall-clock-reference {path}: reference_tps must be a number above 0")
    by_tier = reference.setdefault("tool_share_by_tier", {})
    if not isinstance(by_tier, dict):
        raise SystemExit(f"--wall-clock-reference {path}: tool_share_by_tier must be an object")
    shares = {"tool_share_default": reference.get("tool_share_default")}
    shares.update({f"tool_share_by_tier[{k!r}]": v for k, v in by_tier.items()})
    for name, raw in shares.items():
        share = _as_number(raw)
        if share is None or not 0 <= share <= 1:
            raise SystemExit(
                f"--wall-clock-reference {path}: {name} must be a number between 0 and 1"
            )
    return reference


def _wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    delta = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - delta) / denom), min(1.0, (centre + delta) / denom)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n trials with c solved: 1 - C(n-c, k) / C(n, k)."""
    if c <= 0 or k <= 0 or k > n:
        return 0.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _print_pass_at_k(attempts: list[dict], trials: int) -> None:
    """Print per-task pass@1..3 and overall means over scored attempts only.

    A task enters the pass@k mean only if it has at least k scored attempts.
    """
    counts: dict[str, list[int]] = {}
    for t in attempts:
        if t["scored"]:
            n_c = counts.setdefault(t["task"], [0, 0])
            n_c[0] += 1
            n_c[1] += int(t["solved"])
    ks = range(1, min(3, trials) + 1)
    print("\npass@k (scored attempts only):")
    for tid, (n, c) in counts.items():
        cols = " ".join(f"pass@{k}={pass_at_k(n, c, k):.2%}" for k in ks if k <= n)
        print(f"  {tid:20} {c}/{n} {cols}")
    for k in sorted({ks[-1], 1}, reverse=True):
        scores = [pass_at_k(n, c, k) for n, c in counts.values() if n >= k]
        if not scores:
            continue
        mean = statistics.mean(scores)
        lo, hi = _wilson(mean, len(scores))
        print(f"overall pass@{k}: {mean:.2%} [{lo:.2%}, {hi:.2%}] mean over {len(scores)} tasks")


def _get_git_commit() -> str | None:
    root = TASKS_DIR.parent
    # Extracted kits must not inherit an unrelated enclosing repository's HEAD.
    if not (root / ".git").exists():
        try:
            manifest = json.loads((root / "suite-manifest.json").read_text(encoding="utf-8"))
            commit = manifest.get("harness_commit")
            if (
                manifest.get("harness_source_hash") == _get_harness_hash()
                and isinstance(commit, str)
                and re.fullmatch(r"[0-9a-f]{40}", commit)
            ):
                return commit
        except (OSError, ValueError, AttributeError):
            pass
        return None
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", out.stdout.strip()):
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _get_attacker_digest() -> str | None:
    try:
        out = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", "rb-attacker:latest"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip().startswith("sha256:"):
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _get_task_set_hash() -> str:
    h = hashlib.sha256()
    for path in sorted(TASKS_DIR.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        h.update(str(path.relative_to(TASKS_DIR)).encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:16]


def _get_harness_hash() -> str:
    root = TASKS_DIR.parent
    paths = sorted((root / "rangebench").glob("*.py")) + [root / "pyproject.toml"]
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path.relative_to(root)).encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:16]


def _source_fingerprint() -> dict[str, str | None]:
    return {
        "harness_commit": _get_git_commit(),
        "harness_source_hash": _get_harness_hash(),
        "task_set_hash": _get_task_set_hash(),
    }


def _write_manifest(log_dir: Path, doc: dict, extra: dict | None = None) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "manifest_version": 2,
        "id": doc.get("id"),
        "model": doc.get("model"),
        "base_url": doc.get("base_url"),
        "provider": doc.get("provider", "openai"),
        "ctx_window": doc.get("ctx_window"),
        "reserve": doc.get("reserve"),
        "keep_tail": doc.get("keep_tail"),
        "threshold": doc.get("threshold"),
        "compact": doc.get("compact"),
        "selected_tasks": doc.get("selected_tasks"),
        "trials": doc.get("trials"),
        "wall_clock_reference": doc.get("wall_clock_reference"),
        "wall_clock_scale_mode": doc.get("wall_clock_scale_mode"),
        "model_tps": doc.get("model_tps"),
        "wall_clock_scale": doc.get("wall_clock_scale"),
        "started": doc.get("started"),
        "finished": doc.get("finished"),
        "harness_commit": doc.get("harness_commit"),
        "harness_source_hash": doc.get("harness_source_hash"),
        "attacker_digest": doc.get("attacker_digest"),
        "service_image_ids": [
            {"task": task["task"], "trial": task["trial"], "images": task["service_image_ids"]}
            for task in doc.get("tasks", [])
            if task.get("fail_class") != FAIL_SKIPPED
        ],
        "service_image_fingerprints": [
            {
                "task": task["task"],
                "trial": task["trial"],
                "images": task["service_image_fingerprints"],
            }
            for task in doc.get("tasks", [])
            if task.get("fail_class") != FAIL_SKIPPED
        ],
        "task_set_hash": doc.get("task_set_hash"),
        "task_count": sum(t.get("fail_class") != FAIL_SKIPPED for t in doc.get("tasks", [])),
        "output_tokens_include_reasoning": True,
        "usage_coverage": {
            key: sum(int(task.get(key) or 0) for task in doc.get("tasks", []))
            for key in (
                "api_calls",
                "api_requests",
                "usage_reported_calls",
                "input_reported_calls",
                "output_reported_calls",
            )
        },
    }
    if extra:
        manifest.update(extra)
    tmp = log_dir / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, log_dir / "manifest.json")


def _write_report_html(log_dir: Path, doc: dict) -> None:
    try:
        tasks = doc.get("tasks", [])
        rows = []
        for t in tasks:
            out_tok = t.get("completion_tokens", 0)
            status = (
                "skipped"
                if t.get("fail_class") == FAIL_SKIPPED
                else "error"
                if not t.get("scored", True)
                else "pass"
                if t.get("solved")
                else "fail"
            )
            color = "#10b981" if status == "pass" else "#ef4444" if status == "error" else "#9ca3af"
            rows.append(
                f"<tr><td>{html.escape(t.get('task', ''))}</td><td>{html.escape(t.get('category', ''))}</td><td>T{t.get('tier', '')}</td><td style='color:{color}'>{status}</td><td>{t.get('turns_used', 0)}/{t.get('turns_budget') if t.get('turns_budget') is not None else '∞'}</td><td>{out_tok}</td><td>{t.get('wall_s', 0)}</td><td>{html.escape(t.get('end_reason', ''))}</td></tr>"
            )
        scored = [t for t in tasks if t.get("scored", True) and t.get("fail_class") != FAIL_SKIPPED]
        solved = sum(1 for t in scored if t.get("solved"))
        total = len(scored)
        body = f"<h1>rangebench {html.escape(doc.get('id', ''))}</h1><p>model {html.escape(doc.get('model', ''))} - {solved}/{total} - {html.escape(doc.get('started', ''))}</p><table border=1 cellpadding=6><tr><th>task</th><th>cat</th><th>tier</th><th>result</th><th>turns</th><th>out tok</th><th>wall</th><th>end</th></tr>{''.join(rows)}</table>"
        html_doc = f"<!doctype html><meta charset=utf-8><title>rangebench {html.escape(doc.get('id', ''))}</title><style>body{{font-family:system-ui,sans-serif;margin:2rem}}table{{border-collapse:collapse}}th{{background:#f3f4f6}}</style>{body}"
        tmp = log_dir / "report.html.tmp"
        tmp.write_text(html_doc)
        os.replace(tmp, log_dir / "report.html")
    except Exception:
        pass


def cmd_list(_args: argparse.Namespace) -> None:
    tasks = load_all()
    print(
        f"{'id':24} {'cat':10} {'tier':4} {'stages':6} {'turns':5} {'infra':6} {'out-budget':10} statement"
    )
    for t in tasks:
        print(
            f"{t.id:24} {t.category:10} T{t.tier:<3} {len(t.stages):<6} {str(t.turns) if t.turns is not None else '∞':<5} {t.infra_timeout:<6} {t.max_output_tokens:<10} {t.statement[:60]}"
        )


def cmd_identities(_args: argparse.Namespace) -> None:
    """Print each task's per-task identity hash, sorted by task name."""
    print(f"{'task':24} identity")
    for tid, ident in all_task_identities(TASKS_DIR).items():
        print(f"{tid:24} {ident}")


def cmd_check(args: argparse.Namespace) -> None:
    for tid in args.tasks:
        run_oracle(load_task(tid), project=f"rb-oracle-{uuid.uuid4().hex[:10]}")


def _base_url(args: argparse.Namespace) -> str:
    """--base-url, else $OPENAI_BASE_URL, else the local default."""
    return args.base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")


def _make_client(args: argparse.Namespace, base: str) -> ChatClientProtocol:
    """The chat client for --provider, --model and (openai only) --reasoning-effort."""
    key_env = getattr(args, "api_key_env", None)
    if key_env and not os.environ.get(key_env):
        raise SystemExit(f"{key_env} is unset")
    key = os.environ.get(key_env) if key_env else None
    if getattr(args, "provider", "openai") == "anthropic":
        return AnthropicChatClient(
            base_url=base,
            api_key=key
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("OPENAI_API_KEY", "dummy"),
            model=args.model,
        )
    return ChatClient(
        base_url=base,
        api_key=key or os.environ.get("OPENAI_API_KEY", "dummy"),
        model=args.model,
        reasoning_effort=getattr(args, "reasoning_effort", None),
    )


def _validate_run_args(args: argparse.Namespace) -> tuple[float | None, dict | None]:
    """Check run flags; return (manual wall-clock scale, loaded reference). SystemExit on error."""
    if args.trials < 1:
        raise SystemExit("--trials must be at least 1")
    if getattr(args, "max_attempts", None) is not None and args.max_attempts < 1:
        raise SystemExit("--max-attempts must be at least 1")
    if getattr(args, "infra_retries", 1) < 0:
        raise SystemExit("--infra-retries must be nonnegative")
    if (
        not 0 < args.ctx_window <= MAX_CTX_WINDOW
        or args.ctx_window <= args.reserve
        or args.reserve < 0
    ):
        raise SystemExit(f"--ctx-window must be above --reserve and at most {MAX_CTX_WINDOW}")
    if args.keep_tail < 0 or not 0 < args.threshold < 1:
        raise SystemExit("--keep-tail must be nonnegative and --threshold must be between 0 and 1")
    wall_clock_scale = getattr(args, "wall_clock_scale", None)
    reference_path = getattr(args, "wall_clock_reference", None)
    if reference_path is not None:
        if wall_clock_scale is not None:
            raise SystemExit("--wall-clock-scale and --wall-clock-reference are mutually exclusive")
        return wall_clock_scale, _load_wall_clock_reference(reference_path)
    if wall_clock_scale is None:
        wall_clock_scale = 1.0
    if not 0 < wall_clock_scale <= MAX_WALL_CLOCK_SCALE:
        raise SystemExit(f"--wall-clock-scale must be above 0 and at most {MAX_WALL_CLOCK_SCALE}")
    return wall_clock_scale, None


def _write_run_doc(out: Path, doc: dict) -> None:
    """Write the run doc to its own file and to latest.json from one serialization."""
    text = json.dumps(doc, indent=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    (RESULTS / "latest.json").write_text(text)


def _verify_source(
    doc: dict,
    expected: dict[str, str | None],
    out: Path,
    log_dir: Path,
    completed_attempt: bool = False,
) -> None:
    """Stop the run if the source fingerprint moved, unscoring the attempt just finished."""
    current = _source_fingerprint()
    if current == expected:
        return
    if completed_attempt:
        doc["tasks"][-1]["scored"] = False
        doc["tasks"][-1]["end_reason"] = "source changed"
        doc["tasks"][-1]["fail_class"] = classify_end_reason("source changed", False)
    doc["finished"] = datetime.now(UTC).isoformat()
    if completed_attempt:
        _write_run_doc(out, doc)
    _write_manifest(log_dir, doc, extra={"status": "source_changed", "observed_source": current})
    raise SystemExit("benchmark source changed during run; rerun from a frozen checkout")


def _attempt(
    args: argparse.Namespace,
    client: ChatClientProtocol,
    task: Task,
    trial: int,
    project: str,
    log_dir: Path,
    attacker_digest: str,
    wall_clock_scale: float,
    model_tps: float | None,
    on_event: Callable[[str, dict], None] | None = None,
    log_suffix: str = "",
) -> AttemptResult:
    """run_attempt with the run's compaction and --keep settings."""
    return run_attempt(
        client,
        task,
        trial,
        project,
        log_dir,
        keep=args.keep,
        ctx_window=args.ctx_window,
        reserve=args.reserve,
        keep_tail=args.keep_tail,
        threshold=args.threshold,
        use_llm_compact=getattr(args, "compact", "llm") == "llm",
        attacker_image=attacker_digest,
        wall_clock_scale=wall_clock_scale,
        model_tps=model_tps,
        on_event=on_event,
        log_suffix=log_suffix,
    )


def _measure_model_tps(
    args: argparse.Namespace,
    client: ChatClientProtocol,
    doc: dict,
    log_dir: Path,
    attacker_digest: str,
    verify_source: Callable[[], None],
    progress: Progress | None = None,
) -> float:
    """Run one unscored probe attempt and return the model's tokens/sec.

    Records the probe in doc; on any failure writes a probe_failed manifest and
    exits before the run starts. The probe transcript lives in log_dir/probe so
    the probe task may also be a run task without overwriting trial transcripts.
    """
    probe_task_id = getattr(args, "wall_clock_probe_task", None) or PROBE_TASK
    try:
        probe_task = load_task(probe_task_id)
    except EnvError as exc:
        _write_manifest(log_dir, doc, extra={"status": "probe_failed"})
        raise SystemExit(f"--wall-clock-probe-task {probe_task_id}: {exc}") from None
    verify_source()
    probe_dir = log_dir / "probe"
    if progress is not None:
        progress.start_probe()
    try:
        probe = _attempt(
            args,
            client,
            probe_task,
            1,
            f"rb-{probe_task.id}-probe-{uuid.uuid4().hex[:6]}",
            probe_dir,
            attacker_digest,
            1.0,
            None,
        )
    except Exception as exc:
        _write_manifest(log_dir, doc, extra={"status": "probe_failed"})
        raise SystemExit(
            f"wall-clock reference probe attempt failed ({exc or type(exc).__name__}); "
            "run not started"
        ) from None
    finally:
        if progress is not None:
            progress.end_probe()
    try:
        tokens, llm_s, calls = transcript_tps(probe_dir / f"{probe_task.id}-t1.jsonl")
    except OSError:
        tokens, llm_s, calls = 0, 0.0, 0
    execution_failed = probe.end_reason.startswith("env:") or probe.end_reason in {
        "infra timeout",
        "llm error",
        "context window exhausted",
        "model refusal",
        "empty provider response",
        "model produced no content 11x",
    }
    if execution_failed or tokens < 500 or llm_s < 1.0 or calls < 3:
        _write_manifest(log_dir, doc, extra={"status": "probe_failed"})
        raise SystemExit(
            f"wall-clock reference probe failed ({tokens} tokens over {llm_s:.1f}s across "
            f"{calls} calls; need >=500 tokens over >=1s across >=3 calls; attempt ended: "
            f"{probe.end_reason or 'unknown'}); run not started"
        )
    model_tps = tokens / llm_s
    doc["probe_task"] = probe_task_id
    doc["probe_usage"] = probe.total_usage().as_dict()
    doc["probe_turns_used"] = probe.turns_used
    doc["probe_attempt"] = {
        "task": probe_task_id,
        "solved": sorted(probe.solved) == sorted(s.name for s in probe_task.stages),
        "end_reason": probe.end_reason,
        "wall_s": probe.wall_s,
        "completion_tokens": probe.completion_tokens,
        "llm_s": round(llm_s, 1),
        "calls": calls,
        "model_tps": model_tps,
    }
    doc["model_tps"] = model_tps
    return model_tps


def _attempt_record(task: Task, trial: int, res: AttemptResult, task_scale: float) -> dict:
    """The run doc's tasks[] entry for one finished attempt."""
    usage = res.total_usage()
    solved = sorted(res.solved) == sorted(s.name for s in task.stages)
    return {
        "task": task.id,
        "category": task.category,
        "tier": task.tier,
        "trial": trial,
        "solved_stages": res.solved,
        "stages_total": [s.name for s in task.stages],
        "solved": solved,
        "scored": not (
            res.end_reason.startswith("env:")
            or res.end_reason
            in {
                "infra timeout",
                "llm error",
                "context window exhausted",
                "model refusal",
                "empty provider response",
            }
        ),
        "fail_class": classify_end_reason(res.end_reason, solved),
        "wrong": res.wrong,
        "refusals": res.refusals,
        "turns_used": res.turns_used,
        "turns_budget": task.turns,
        "effective_ctx_window": res.effective_ctx_window,
        "commands": res.commands,
        "service_image_ids": res.service_image_ids,
        "service_image_fingerprints": res.service_image_fingerprints,
        "prompt_tokens": res.prompt_tokens,
        "completion_tokens": res.completion_tokens,
        "reasoning_tokens": res.reasoning_tokens,
        "compaction_tokens": res.compaction_tokens,
        "compaction_fallbacks": res.compaction_fallbacks,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "api_calls": usage.calls,
        "api_requests": usage.requests,
        "usage_reported_calls": usage.reported_calls,
        "input_reported_calls": usage.input_reported_calls,
        "output_reported_calls": usage.output_reported_calls,
        "cache_read_reported_calls": usage.cache_read_reported_calls,
        "cache_write_reported_calls": usage.cache_write_reported_calls,
        "compaction_input_tokens": res.compaction_usage.input_tokens,
        "compaction_output_tokens": res.compaction_usage.output_tokens,
        "compaction_cache_read_tokens": res.compaction_usage.cache_read_tokens,
        "compaction_cache_write_tokens": res.compaction_usage.cache_write_tokens,
        "wall_s": res.wall_s,
        "wall_clock_scale": task_scale,
        "wall_clock_seconds": res.wall_clock_seconds,
        "wall_clock_exceeded": res.wall_clock_exceeded,
        "end_reason": res.end_reason,
        "max_output_tokens": task.max_output_tokens,
    }


def _skipped_record(task: Task, trial: int, task_scale: float) -> dict:
    """A placeholder with no score, environment, usage, or wall time."""
    record = _attempt_record(task, trial, AttemptResult(task.id, trial), task_scale)
    record.update(
        scored=False,
        fail_class=FAIL_SKIPPED,
        end_reason="skipped: repeat after no-progress wall cap",
        wall_clock_scale=task_scale,
    )
    for key, value in record.items():
        if key != "max_output_tokens" and "tokens" in key and value is None:
            record[key] = 0
    return record


def _print_run_summary(tasks: list[dict], trials: int, invalid: int) -> None:
    """Print solve counts, per-category/tier splits, failure classes, and per-task rows."""
    attempted = [t for t in tasks if t["fail_class"] != FAIL_SKIPPED]
    scored = [t for t in tasks if t["scored"]]
    solved_count = sum(1 for t in scored if t["solved"])
    total = len(scored)
    print(f"tasks solved: {solved_count}/{total} scored runs ({invalid} invalid)")
    by_cat = defaultdict(list)
    for t in attempted:
        by_cat[t["category"]].append(t)
    print("\nper-category:")
    for cat, lst in sorted(by_cat.items()):
        valid = [x for x in lst if x["scored"]]
        s = sum(1 for x in valid if x["solved"])
        print(f"  {cat:10} {s}/{len(valid)} ({len(lst) - len(valid)} invalid)")
    by_tier = defaultdict(list)
    for t in attempted:
        by_tier[t["tier"]].append(t)
    print("per-tier:")
    for tier in sorted(by_tier):
        lst = by_tier[tier]
        valid = [x for x in lst if x["scored"]]
        s = sum(1 for x in valid if x["solved"])
        print(f"  T{tier} {s}/{len(valid)} ({len(lst) - len(valid)} invalid)")
    fail_counts = {name: sum(1 for t in tasks if t["fail_class"] == name) for name in FAIL_CLASSES}
    print(
        "failure classes: "
        + " ".join(f"{name}={fail_counts[name]}" for name in FAIL_CLASSES)
        + f" {FAIL_SKIPPED}={len(tasks) - len(attempted)}"
    )
    refusal_turns = sum(int(t["refusals"]) for t in attempted)
    refusing = sum(1 for t in attempted if t["refusals"])
    print(f"refusals: {refusal_turns} turns in {refusing}/{len(attempted)} attempts")
    if trials > 1:
        p = solved_count / total if total else 0
        lo, hi = _wilson(p, total)
        print(f"overall Wilson 95%: {p:.2%} [{lo:.2%}, {hi:.2%}] n={total}")
        _print_pass_at_k(tasks, trials)
    toks = [t["completion_tokens"] for t in attempted]
    if toks:
        print(
            f"output tokens: mean {statistics.mean(toks):.0f} median {statistics.median(toks):.0f} max {max(toks)}"
        )
    print("\nper-task:")
    print(f"{'task':20} {'solved':6} {'turns':10} {'out_tok':10} {'wall':8} end")
    for t in tasks:
        out_tok = t["completion_tokens"]
        print(
            f"{t['task']:20} {str(t['solved']):6} {t['turns_used']}/{str(t['turns_budget']) if t['turns_budget'] is not None else '∞':<6} {out_tok:<10} {t['wall_s']:<8} {t['end_reason']}"
        )


def cmd_run(args: argparse.Namespace) -> None:
    wall_clock_scale, reference = _validate_run_args(args)
    max_attempts = getattr(args, "max_attempts", None)
    infra_retries = getattr(args, "infra_retries", 1)
    reference_path = getattr(args, "wall_clock_reference", None)
    task_ids = args.tasks if args.tasks else [t.id for t in load_all()]
    base = _base_url(args)
    if not args.base_url and "OPENAI_BASE_URL" not in os.environ:
        print(f"[warn] OPENAI_BASE_URL not set, using default {base}", flush=True)
    attacker_digest = _get_attacker_digest()
    if not attacker_digest:
        raise SystemExit("rb-attacker image ID unavailable; run preflight first")
    provider = getattr(args, "provider", "openai")
    client = _make_client(args, base)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    log_dir = RESULTS / run_id
    out = RESULTS / f"{run_id}.json"
    source_fingerprint = _source_fingerprint()
    doc = {
        "id": run_id,
        "model": args.model,
        "base_url": base,
        "provider": provider,
        "api_key_env": getattr(args, "api_key_env", None),
        "attacker_digest": attacker_digest,
        **source_fingerprint,
        "ctx_window": args.ctx_window,
        "reserve": args.reserve,
        "keep_tail": args.keep_tail,
        "threshold": args.threshold,
        "compact": args.compact,
        "reasoning_effort": getattr(args, "reasoning_effort", None),
        "display_name": getattr(args, "display_name", None) or args.model,
        "route_name": getattr(args, "route_name", None),
        "upstream_provider": getattr(args, "upstream_provider", None),
        "repeat_caps": getattr(args, "repeat_caps", False),
        "infra_retries": infra_retries,
        "max_attempts": max_attempts,
        "selected_tasks": task_ids,
        "task_identities": {
            k: v for k, v in all_task_identities(TASKS_DIR).items() if k in task_ids
        },
        "task_catalog": [
            {"task": task.id, "category": task.category, "tier": task.tier}
            for task in load_all()
            if task.id in task_ids
        ],
        "trials": args.trials,
        "wall_clock_scale_mode": "manual" if reference is None else "reference",
        "wall_clock_scale": wall_clock_scale,
        "started": datetime.now(UTC).isoformat(),
        "tasks": [],
    }
    if reference is not None:
        doc["wall_clock_reference"] = reference_path
        doc["wall_clock_reference_hash"] = hashlib.sha256(
            json.dumps(reference, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        doc["model_tps"] = None
    # Fail before spending model calls if the manifest cannot be recorded.
    _write_manifest(log_dir, doc, extra={"status": "running"})
    status_path = getattr(args, "status_json", None)
    progress = Progress(
        log_dir,
        Path(status_path) if status_path else None,
        task_ids,
        args.trials,
        max_attempts,
        infra_retries,
        probe_attempts=int(reference is not None),
    )
    doc["infra_attempts"] = []

    def verify_source(completed_attempt: bool = False) -> None:
        try:
            _verify_source(doc, source_fingerprint, out, log_dir, completed_attempt)
        except SystemExit:
            progress.finish("source_changed")
            raise

    def save_progress() -> None:
        _write_run_doc(out, doc)
        _write_manifest(
            log_dir,
            doc,
            extra={"status": "running", "progress": f"{progress.used}/{progress.limit}"},
        )

    model_tps: float | None = None
    if reference is not None:
        # One unscored agent attempt measures generation speed; fail before the run starts.
        try:
            model_tps = _measure_model_tps(
                args, client, doc, log_dir, attacker_digest, verify_source, progress
            )
        except SystemExit:
            progress.finish("probe_failed")
            raise
        _write_manifest(log_dir, doc, extra={"status": "running"})

    capped = False
    for tid in task_ids:
        if progress.used >= progress.limit:
            capped = True
            break
        verify_source()
        task = load_task(tid)
        if not any(item["task"] == tid for item in doc["task_catalog"]):
            doc["task_catalog"].append({"task": tid, "category": task.category, "tier": task.tier})
        task_scale = 1.0 if wall_clock_scale is None else wall_clock_scale
        if reference is not None and model_tps is not None:
            try:
                task_scale = wall_clock_scale_for(reference, model_tps, task.tier)
            except ValueError as exc:
                raise SystemExit(f"--wall-clock-reference {reference_path}: {exc}") from None
        for trial in range(1, args.trials + 1):
            verify_source()
            prior = [
                t for t in doc["tasks"] if t["task"] == tid and t["fail_class"] != FAIL_SKIPPED
            ]
            if any(
                t.get("refusals", 0) or t.get("end_reason") == "empty provider response"
                for t in prior
            ):
                skipped = _skipped_record(task, trial, task_scale)
                skipped["end_reason"] = "skipped: prior refusal or empty provider response"
                doc["tasks"].append(skipped)
                progress.skip(tid, trial, "prior refusal or empty provider response")
                save_progress()
                continue
            if (
                trial > 1
                and not getattr(args, "repeat_caps", False)
                and prior
                and all(
                    not t["solved"] and t["wall_clock_exceeded"] and not t["solved_stages"]
                    for t in prior
                )
            ):
                doc["tasks"].append(_skipped_record(task, trial, task_scale))
                progress.skip(tid, trial, "repeat after no-progress wall cap")
                save_progress()
                continue
            if progress.used >= progress.limit:
                capped = True
                break
            retry = 0
            while True:
                project = f"rb-{task.id}-{trial}-{uuid.uuid4().hex[:6]}"
                progress.start(tid, trial, retry)

                def on_event(
                    kind: str, data: dict, task_id: str = tid, task_trial: int = trial
                ) -> None:
                    progress.event(task_id, task_trial, kind, data)

                res = _attempt(
                    args,
                    client,
                    task,
                    trial,
                    project,
                    log_dir,
                    attacker_digest,
                    task_scale,
                    model_tps,
                    on_event=on_event,
                    log_suffix=f"-infra{retry}" if retry else "",
                )
                infra_failure = res.end_reason.startswith("env:") or res.end_reason in {
                    "llm error",
                    "infra timeout",
                }
                retrying = (
                    infra_failure and retry < infra_retries and progress.used < progress.limit
                )
                scored = not (
                    infra_failure
                    or res.end_reason
                    in {"context window exhausted", "model refusal", "empty provider response"}
                )
                solved = sorted(res.solved) == sorted(s.name for s in task.stages)
                progress.end(tid, trial, res, scored, solved, retrying)
                if infra_failure:
                    doc["infra_attempts"].append(
                        {
                            "task": tid,
                            "trial": trial,
                            "retry": retry,
                            "fail_class": classify_end_reason(res.end_reason, False),
                            "scored": False,
                            "wall_s": res.wall_s,
                            "prompt_tokens": res.prompt_tokens,
                            "completion_tokens": res.completion_tokens,
                            "usage": res.total_usage().as_dict(),
                            "turns_used": res.turns_used,
                            "terminal": not retrying,
                        }
                    )
                    save_progress()
                if not retrying:
                    break
                verify_source()
                retry += 1
            doc["tasks"].append(_attempt_record(task, trial, res, task_scale))
            verify_source(completed_attempt=True)
            save_progress()
    if max_attempts is not None and progress.used >= progress.limit:
        capped = capped or any(slot["state"] == "pending" for slot in progress.tasks.values())
    verify_source()
    doc["finished"] = datetime.now(UTC).isoformat()
    _write_run_doc(out, doc)
    invalid = sum(1 for t in doc["tasks"] if not t["scored"] and t["fail_class"] != FAIL_SKIPPED)
    state = "attempt_limit" if capped else "completed_with_errors" if invalid else "completed"
    _write_manifest(log_dir, doc, extra={"status": state})
    progress.finish(state)
    _write_report_html(log_dir, doc)
    try:
        key_names = (getattr(args, "api_key_env", None), "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
        credentials = tuple(os.environ[name] for name in key_names if name and os.environ.get(name))
        submission = build_submission(doc, sensitive_values=credentials)
        submission_path = log_dir / "submission.json"
        submission_path.write_text(json.dumps(submission, indent=2, allow_nan=False) + "\n")
        print(f"wrote {submission_path}")
    except (OSError, ValueError, KeyError, TypeError):
        print("[warn] safe submission could not be generated; raw run was preserved")
    print(f"wrote {out}")
    print(f"wrote {log_dir / 'manifest.json'} and {log_dir / 'report.html'}")
    _print_run_summary(doc["tasks"], args.trials, invalid)
    if invalid:
        raise SystemExit(1)


def cmd_probe(args: argparse.Namespace) -> None:
    """Send the probe prompt --samples times and print latency and tokens/sec per call.

    With --json, print one JSON summary object and nothing else.
    """
    if args.samples < 1:
        raise SystemExit("--samples must be at least 1")
    client = _make_client(args, _base_url(args))
    as_json = getattr(args, "json", False)
    latencies: list[float] = []
    rates: list[float] = []
    for i in range(1, args.samples + 1):
        elapsed, content, usage, err = _probe_once(client)
        if err:
            if not as_json:
                print(f"sample {i}: err={err}")
            continue
        # completion_tokens already includes reasoning_tokens (see Usage.add).
        tokens = usage.completion_tokens
        tps = tokens / elapsed if elapsed > 0 else 0.0
        if not as_json:
            print(
                f"sample {i}: latency {elapsed:.2f}s tokens {tokens} "
                f"(reasoning {usage.reasoning_tokens}) {tps:.1f} tok/s content={content!r}"
            )
        latencies.append(elapsed)
        if tokens > 0 and elapsed > 0:
            rates.append(tps)
    if as_json:
        summary: dict[str, object] = {"samples": args.samples, "ok": len(latencies)}
        if latencies:
            summary["mean_latency_s"] = statistics.mean(latencies)
            summary["median_latency_s"] = statistics.median(latencies)
            summary["mean_tokens_per_sec"] = statistics.mean(rates) if rates else None
        print(json.dumps(summary))
        return
    if not latencies:
        return
    print(
        f"mean latency {statistics.mean(latencies):.2f}s "
        f"median latency {statistics.median(latencies):.2f}s"
    )
    mean_rate = f"{statistics.mean(rates):.1f}" if rates else "n/a"
    print(f"mean tokens/sec {mean_rate} n={len(latencies)}/{args.samples}")


def cmd_preflight(_args: argparse.Namespace) -> None:
    """Check Docker and prepare every target image without starting tasks."""
    if shutil.which("docker") is None:
        print("docker not found", flush=True)
        raise SystemExit(1)
    subprocess.run(["docker", "version"], check=True)
    res = subprocess.run(["docker", "compose", "version"], capture_output=True, text=True)
    if res.returncode != 0:
        print("docker compose not found, need docker compose plugin", flush=True)
        raise SystemExit(1)
    print(res.stdout.strip())
    root = TASKS_DIR.parent
    print("[preflight] building rb-attacker from current source...", flush=True)
    subprocess.run(
        ["docker", "build", "-t", "rb-attacker:latest", str(root / "attacker")],
        check=True,
    )
    digest = _get_attacker_digest()
    if not digest:
        raise SystemExit("rb-attacker image ID unavailable after build")
    print(f"attacker digest: {digest}")
    for task_dir in sorted(TASKS_DIR.iterdir()):
        compose = task_dir / "docker-compose.yml"
        if not compose.exists():
            continue
        print(f"[preflight] pulling {task_dir.name} ...", flush=True)
        subprocess.run(
            ["docker", "compose", "-f", str(compose), "pull", "--ignore-buildable", "--quiet"],
            check=True,
            timeout=600,
        )
        print(f"[preflight] building {task_dir.name} ...", flush=True)
        subprocess.run(
            ["docker", "compose", "-f", str(compose), "build", "--quiet"],
            check=True,
            timeout=1800,
        )
    print(f"task set hash: {_get_task_set_hash()}")
    commit = _get_git_commit()
    if commit:
        print(f"harness commit: {commit}")
    print("preflight done")


def cmd_export(args: argparse.Namespace) -> None:
    try:
        if args.output.resolve() == args.run_file.resolve():
            raise ValueError("output must differ from input")
        run = json.loads(args.run_file.read_text(encoding="utf-8"))
        if not isinstance(run, dict):
            raise ValueError("run must be an object")
        route = args.route_name or run.get("route_name")
        pricing = None
        if args.lookup_pricing:
            from .submission import _public_route

            pricing = public_price_lookup(
                args.price_model_id or run["model"],
                route or _public_route(run.get("base_url")),
                args.upstream_provider or run.get("upstream_provider"),
            )
        manual = (
            args.input_price,
            args.output_price,
            args.cache_read_price,
            args.cache_write_price,
        )
        if any(p is not None for p in manual):
            if (
                args.lookup_pricing
                or args.input_price is None
                or args.output_price is None
                or not args.price_source
                or not args.price_date
            ):
                raise ValueError(
                    "manual prices require input/output rates, source and date; do not combine with lookup"
                )
            from .submission import _unknown_pricing

            pricing = _unknown_pricing()
            pricing.update(
                status="estimated",
                source=args.price_source,
                as_of=args.price_date,
                usd_per_million=dict(
                    zip(("input", "output", "cache_read", "cache_write"), manual, strict=True)
                ),
            )
        key_names = (run.get("api_key_env"), "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
        credentials = tuple(
            os.environ[name] for name in key_names if isinstance(name, str) and os.environ.get(name)
        )
        submission = build_submission(
            run,
            pricing=pricing,
            display_name=args.display_name,
            route_name=args.route_name,
            upstream_provider=args.upstream_provider,
            sensitive_values=credentials,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            output.write(json.dumps(submission, indent=2, allow_nan=False) + "\n")
        print(f"wrote {args.output}")
    except (OSError, ValueError, KeyError, TypeError):
        raise SystemExit(
            "export failed: invalid input, unsafe metadata or output unavailable"
        ) from None


def cmd_validate_submission(args: argparse.Namespace) -> None:
    try:
        data = json.loads(args.submission.read_text(encoding="utf-8"))
        validate_submission(data)
    except (OSError, ValueError, KeyError, TypeError):
        raise SystemExit("invalid submission") from None
    print("valid self-reported submission")


def main() -> None:
    ap = argparse.ArgumentParser(prog="rangebench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(func=cmd_list)
    ident = sub.add_parser("identities", help="print per-task identity hashes")
    ident.set_defaults(func=cmd_identities)
    chk = sub.add_parser("check", help="run oracle solutions against live envs")
    chk.add_argument("tasks", nargs="+")
    chk.set_defaults(func=cmd_check)
    run = sub.add_parser("run", help="run agent against tasks")
    run.add_argument("--model", required=True)
    run.add_argument("--display-name", help="public model display name")
    run.add_argument("--route-name", help="public router or service name")
    run.add_argument("--upstream-provider", help="public upstream model provider")
    run.add_argument(
        "--base-url",
        default=None,
        help="OpenAI base url, default $OPENAI_BASE_URL or http://localhost:8000/v1",
    )
    run.add_argument(
        "--provider", choices=["openai", "anthropic"], default="openai", help="wire format"
    )
    run.add_argument("tasks", nargs="*")
    run.add_argument("--trials", type=int, default=1)
    run.add_argument(
        "--max-attempts",
        type=int,
        help="hard cap on all environment starts, including infra retries",
    )
    run.add_argument(
        "--infra-retries",
        type=int,
        default=1,
        help="fresh reruns after env/LLM errors per task/trial",
    )
    run.add_argument(
        "--status-json", type=Path, help="atomically updated machine-readable status file"
    )
    run.add_argument("--api-key-env", help="name of API key environment variable")
    run.add_argument(
        "--repeat-caps", action="store_true", help="run all trials after no-progress wall caps"
    )
    run.add_argument(
        "--wall-clock-scale",
        type=float,
        default=None,
        help="multiply each attempt's wall-clock cap so slow-inference models get "
        f"proportionally more time for the same turn budget, default 1, "
        f"at most {MAX_WALL_CLOCK_SCALE:g}",
    )
    run.add_argument(
        "--wall-clock-reference",
        default=None,
        metavar="PATH",
        help="reference JSON (reference_tps, tool_share_default, tool_share_by_tier); "
        "runs one unscored probe attempt to measure tokens/sec and scales each task's "
        "cap by its tier's generation share",
    )
    run.add_argument(
        "--wall-clock-probe-task",
        default=PROBE_TASK,
        metavar="TASK",
        help=f"unscored task attempted once to measure tokens/sec in reference mode, "
        f"default {PROBE_TASK}; its transcript is isolated so it may overlap run tasks",
    )
    run.add_argument("--keep", action="store_true", help="skip teardown (debug)")
    run.add_argument(
        "--ctx-window",
        type=int,
        default=DEFAULT_CTX_WINDOW,
        help=f"model context window for auto compaction, maximum {MAX_CTX_WINDOW}",
    )
    run.add_argument(
        "--reserve", type=int, default=DEFAULT_RESERVE, help="reserve tokens for compaction output"
    )
    run.add_argument(
        "--keep-tail",
        type=int,
        default=DEFAULT_KEEP_TAIL,
        help="tail turns kept verbatim after compaction",
    )
    run.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="compaction threshold fraction of ctx window",
    )
    run.add_argument(
        "--compact",
        choices=["deterministic", "llm"],
        default="llm",
        help="compaction mode, same-model llm is the default",
    )
    run.add_argument(
        "--reasoning-effort",
        default=None,
        help="openai-compatible reasoning effort knob sent to the provider (e.g. high, max)",
    )
    run.set_defaults(func=cmd_run)
    export = sub.add_parser(
        "export", help="export a safe standalone submission from one completed run"
    )
    export.add_argument("run_file", type=Path)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--display-name")
    export.add_argument("--route-name")
    export.add_argument("--upstream-provider")
    export.add_argument("--lookup-pricing", action="store_true")
    export.add_argument(
        "--price-model-id", help="exact catalog model ID when run model is an alias"
    )
    export.add_argument("--input-price", type=float, help="USD per million input tokens")
    export.add_argument("--output-price", type=float, help="USD per million output tokens")
    export.add_argument("--cache-read-price", type=float)
    export.add_argument("--cache-write-price", type=float)
    export.add_argument("--price-source", help="public source name for manually entered prices")
    export.add_argument("--price-date", help="date of manually entered public prices (YYYY-MM-DD)")
    export.set_defaults(func=cmd_export)
    validate = sub.add_parser("validate-submission", help="validate a standalone submission")
    validate.add_argument("submission", type=Path)
    validate.set_defaults(func=cmd_validate_submission)
    launch = sub.add_parser("launch", help="probe profiles and start one run per model")
    launch.add_argument("configs", nargs="*", help="names in configs/ (prompted when omitted)")
    launch.add_argument("--list", action="store_true", help="list available profiles")
    launch.add_argument("--parallel", type=int, help="number of concurrent run units (default 1)")
    launch.add_argument(
        "--dry-run", action="store_true", help="probe then print commands without spawning"
    )
    launch.add_argument("--max-attempts", type=int)
    launch.add_argument("--infra-retries", type=int)
    launch.add_argument("--status-dir", type=Path, help="write one status JSON per config name")
    launch.set_defaults(func=cmd_launch)
    probe = sub.add_parser("probe")
    probe.add_argument("--model", required=True)
    probe.add_argument("--base-url", default=None)
    probe.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    probe.add_argument("--reasoning-effort", default=None)
    probe.add_argument(
        "--samples", type=int, default=1, help="sequential calls for latency and tokens/sec"
    )
    probe.add_argument("--json", action="store_true", help="print one JSON summary object")
    probe.set_defaults(func=cmd_probe)
    pf = sub.add_parser("preflight", help="pull all images and check docker setup")
    pf.set_defaults(func=cmd_preflight)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
