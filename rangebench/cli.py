"""CLI: list / identities / check (oracle) / run (agent) / smoke."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
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
from .runner import (
    DEFAULT_CTX_WINDOW,
    DEFAULT_KEEP_TAIL,
    DEFAULT_RESERVE,
    DEFAULT_THRESHOLD,
    FAIL_CLASSES,
    MAX_CTX_WINDOW,
    MAX_WALL_CLOCK_SCALE,
    AttemptResult,
    classify_end_reason,
    run_attempt,
    run_oracle,
    transcript_tps,
    wall_clock_scale_for,
)

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
    for cmd in (["git", "rev-parse", "HEAD"], ["git", "rev-parse", "--short", "HEAD"]):
        try:
            out = subprocess.run(
                cmd, cwd=TASKS_DIR.parent, capture_output=True, text=True, timeout=5
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip().splitlines()[0][:40]
        except Exception:
            continue
    return os.environ.get("RANGEBENCH_COMMIT")


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
        ],
        "service_image_fingerprints": [
            {
                "task": task["task"],
                "trial": task["trial"],
                "images": task["service_image_fingerprints"],
            }
            for task in doc.get("tasks", [])
        ],
        "task_set_hash": doc.get("task_set_hash"),
        "task_count": len(doc.get("tasks", [])),
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
            status = "error" if not t.get("scored", True) else "pass" if t.get("solved") else "fail"
            color = "#10b981" if status == "pass" else "#ef4444" if status == "error" else "#9ca3af"
            rows.append(
                f"<tr><td>{html.escape(t.get('task', ''))}</td><td>{html.escape(t.get('category', ''))}</td><td>T{t.get('tier', '')}</td><td style='color:{color}'>{status}</td><td>{t.get('turns_used', 0)}/{t.get('turns_budget') if t.get('turns_budget') is not None else '∞'}</td><td>{out_tok}</td><td>{t.get('wall_s', 0)}</td><td>{html.escape(t.get('end_reason', ''))}</td></tr>"
            )
        scored = [t for t in tasks if t.get("scored", True)]
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
    if getattr(args, "provider", "openai") == "anthropic":
        return AnthropicChatClient(
            base_url=base,
            api_key=os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("OPENAI_API_KEY", "dummy"),
            model=args.model,
        )
    return ChatClient(
        base_url=base,
        api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
        model=args.model,
        reasoning_effort=getattr(args, "reasoning_effort", None),
    )


def _validate_run_args(args: argparse.Namespace) -> tuple[float | None, dict | None]:
    """Check run flags; return (manual wall-clock scale, loaded reference). SystemExit on error."""
    if args.trials < 1:
        raise SystemExit("--trials must be at least 1")
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
    )


def _measure_model_tps(
    args: argparse.Namespace,
    client: ChatClientProtocol,
    doc: dict,
    log_dir: Path,
    attacker_digest: str,
    verify_source: Callable[[], None],
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
    try:
        tokens, llm_s, calls = transcript_tps(probe_dir / f"{probe_task.id}-t1.jsonl")
    except OSError:
        tokens, llm_s, calls = 0, 0.0, 0
    execution_failed = probe.end_reason.startswith("env:") or probe.end_reason in {
        "infra timeout",
        "llm error",
        "context window exhausted",
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
            or res.end_reason in {"infra timeout", "llm error", "context window exhausted"}
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


def _print_run_summary(tasks: list[dict], trials: int, invalid: int) -> None:
    """Print solve counts, per-category/tier splits, failure classes, and per-task rows."""
    scored = [t for t in tasks if t["scored"]]
    solved_count = sum(1 for t in scored if t["solved"])
    total = len(scored)
    print(f"tasks solved: {solved_count}/{total} scored runs ({invalid} invalid)")
    by_cat = defaultdict(list)
    for t in tasks:
        by_cat[t["category"]].append(t)
    print("\nper-category:")
    for cat, lst in sorted(by_cat.items()):
        valid = [x for x in lst if x["scored"]]
        s = sum(1 for x in valid if x["solved"])
        print(f"  {cat:10} {s}/{len(valid)} ({len(lst) - len(valid)} invalid)")
    by_tier = defaultdict(list)
    for t in tasks:
        by_tier[t["tier"]].append(t)
    print("per-tier:")
    for tier in sorted(by_tier):
        lst = by_tier[tier]
        valid = [x for x in lst if x["scored"]]
        s = sum(1 for x in valid if x["solved"])
        print(f"  T{tier} {s}/{len(valid)} ({len(lst) - len(valid)} invalid)")
    fail_counts = {name: sum(1 for t in tasks if t["fail_class"] == name) for name in FAIL_CLASSES}
    print("failure classes: " + " ".join(f"{name}={fail_counts[name]}" for name in FAIL_CLASSES))
    refusal_turns = sum(int(t["refusals"]) for t in tasks)
    refusing = sum(1 for t in tasks if t["refusals"])
    print(f"refusals: {refusal_turns} turns in {refusing}/{len(tasks)} attempts")
    if trials > 1:
        p = solved_count / total if total else 0
        lo, hi = _wilson(p, total)
        print(f"overall Wilson 95%: {p:.2%} [{lo:.2%}, {hi:.2%}] n={total}")
        _print_pass_at_k(tasks, trials)
    toks = [t["completion_tokens"] for t in tasks]
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
        "attacker_digest": attacker_digest,
        **source_fingerprint,
        "ctx_window": args.ctx_window,
        "reserve": args.reserve,
        "keep_tail": args.keep_tail,
        "threshold": args.threshold,
        "compact": args.compact,
        "selected_tasks": task_ids,
        "trials": args.trials,
        "wall_clock_scale_mode": "manual" if reference is None else "reference",
        "wall_clock_scale": wall_clock_scale,
        "started": datetime.now(UTC).isoformat(),
        "tasks": [],
    }
    if reference is not None:
        doc["wall_clock_reference"] = reference_path
        doc["model_tps"] = None
    # Fail before spending model calls if the manifest cannot be recorded.
    _write_manifest(log_dir, doc, extra={"status": "running"})

    def verify_source(completed_attempt: bool = False) -> None:
        _verify_source(doc, source_fingerprint, out, log_dir, completed_attempt)

    model_tps: float | None = None
    if reference is not None:
        # One unscored agent attempt measures generation speed; fail before the run starts.
        model_tps = _measure_model_tps(args, client, doc, log_dir, attacker_digest, verify_source)
        _write_manifest(log_dir, doc, extra={"status": "running"})

    for tid in task_ids:
        verify_source()
        task = load_task(tid)
        task_scale = 1.0 if wall_clock_scale is None else wall_clock_scale
        if reference is not None and model_tps is not None:
            try:
                task_scale = wall_clock_scale_for(reference, model_tps, task.tier)
            except ValueError as exc:
                raise SystemExit(f"--wall-clock-reference {reference_path}: {exc}") from None
        for trial in range(1, args.trials + 1):
            verify_source()
            project = f"rb-{task.id}-{trial}-{uuid.uuid4().hex[:6]}"
            res = _attempt(
                args, client, task, trial, project, log_dir, attacker_digest, task_scale, model_tps
            )
            doc["tasks"].append(_attempt_record(task, trial, res, task_scale))
            verify_source(completed_attempt=True)
            _write_run_doc(out, doc)
            _write_manifest(
                log_dir,
                doc,
                extra={
                    "status": "running",
                    "progress": f"{len(doc['tasks'])}/{len(task_ids) * args.trials}",
                },
            )
    verify_source()
    doc["finished"] = datetime.now(UTC).isoformat()
    _write_run_doc(out, doc)
    invalid = sum(1 for t in doc["tasks"] if not t["scored"])
    _write_manifest(
        log_dir, doc, extra={"status": "completed_with_errors" if invalid else "completed"}
    )
    _write_report_html(log_dir, doc)
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
    """Check docker, compose, and pull all task images without running tasks."""
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
    print(f"task set hash: {_get_task_set_hash()}")
    commit = _get_git_commit()
    if commit:
        print(f"harness commit: {commit}")
    print("preflight done")


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
