"""Command line: run, gate, report.

    python3 -m rangebench_homelab run --profile example-api --suite medium
    python3 -m rangebench_homelab run --profile example-api --session example --seed 0
    python3 -m rangebench_homelab gate example --seeds 0,1,2
    python3 -m rangebench_homelab report <run-id>
    python3 -m rangebench_homelab rescore <run-id> [-o out/run.jsonl]
    python3 -m rangebench_homelab test [tests.test_sim ...]

`run` writes <suite>/runs/<id>/run.jsonl: a run header (profile, harness and template
hashes), one record per step as it finishes, one per attempt, and an end record.
Each session attempt runs in its own process, up to the profile's concurrency.
World dirs are removed when an attempt ends (unless --keep-world); `run` and `gate`
first sweep the worlds of runs and gates that were killed (see sweep()).
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import sys
import time
import unittest
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

from .agent import Message
from .gate import WORK_ROOT, gate
from .profiles import Profile, load_profile
from .rater import Rater
from .report import load, summarize
from .session import Agent, LLMAgent, Record, run_session
from .template import (
    PACKAGE_ROOT,
    RULES_FILE,
    SUITE_DIR,
    Session,
    load_template,
    suite_ids,
    template_dir,
)

RUNS_DIR = SUITE_DIR / "runs"
SUITES = {"medium": (5, (0,)), "max": (20, (0, 1, 2))}  # steps per session, seeds
DEFAULT_RATERS = os.environ.get("RANGEBENCH_HOMELAB_RATERS", "")  # comma-separated rater profiles


def tree_hash(root: Path, pattern: str = "**/*") -> str:
    h = hashlib.sha256()
    for path in sorted(
        p for p in root.glob(pattern) if p.is_file() and "__pycache__" not in p.parts
    ):
        h.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()[:16]


def hold(d: Path) -> IO[str]:
    """Lock d/.lock for as long as this process (and its forked workers) live; sweep() skips d
    meanwhile."""
    f = (d / ".lock").open("w")
    fcntl.flock(f, fcntl.LOCK_EX)
    return f


def _dead(d: Path) -> bool:
    """d has a .lock nobody holds. Dirs without one (made by older harness versions) count as
    live."""
    try:
        f = (d / ".lock").open()
    except OSError:
        return False
    with f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True


def _kept(run_dir: Path) -> bool:
    try:
        with (run_dir / "run.jsonl").open() as f:
            return bool(json.loads(f.readline()).get("keep_world"))
    except (OSError, ValueError, AttributeError):
        return False


def sweep() -> None:
    """Remove world and stub dirs that killed runs and gates left behind. run.jsonl, logs,
    commands, transcripts stay; --keep-world runs and runs still going are left alone."""
    for run_dir in RUNS_DIR.glob("*/"):
        left = [
            p for a in run_dir.glob("attempts/*") for p in (a / "world", a / "bin") if p.exists()
        ]
        if left and not _kept(run_dir) and _dead(run_dir):
            for p in left:
                shutil.rmtree(p, ignore_errors=True)
    for d in WORK_ROOT.glob("rbgate-*"):
        if _dead(d):
            shutil.rmtree(d, ignore_errors=True)


def _exit_on_sigterm() -> None:
    """systemctl stop sends SIGTERM to every process of the scope; exit through finally blocks."""
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))


def append(path: Path, doc: Mapping[str, object]) -> None:
    """Append one JSON line; flock keeps concurrent attempt processes from interleaving."""
    with path.open("a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        f.flush()


def _attempt(
    profile_name: str,
    rater_names: list[str],
    template_id: str,
    seed: int,
    max_steps: int | None,
    run_dir: str,
    keep_world: bool,
) -> dict[str, object]:
    """One session attempt (runs in a worker process)."""
    _exit_on_sigterm()
    stopped = False
    profile = load_profile(profile_name)
    template = load_template(template_id)
    raters = [Rater(load_profile(n)) for n in rater_names]
    run_file = Path(run_dir) / "run.jsonl"
    workdir = Path(run_dir) / "attempts" / f"{template_id}-s{seed}"

    def make_agent(session: Session, params: dict[str, str], tools: list[Message]) -> Agent:
        return LLMAgent(profile.client(), session.system, tools, profile.ctx_window)

    def emit(rec: Record) -> None:
        append(run_file, rec)
        print(
            f"[{datetime.now(UTC):%H:%M:%S}] {template_id} s{seed} step {rec['position']} "
            f"{rec['step']}: "
            f"{'PASS' if rec['pass'] else 'fail'} · {rec.get('end_reason')} · turns "
            f"{rec.get('turns')} · labels {rec.get('labels')}",
            flush=True,
        )

    try:
        res = run_session(
            template, seed, make_agent, workdir, max_steps, raters, profile.wall_clock_scale, emit
        )
    except (
        SystemExit
    ):  # SIGTERM: the scope is being stopped; clean up, then leave the pool for good
        stopped = True
    finally:
        if not keep_world:
            shutil.rmtree(workdir / "world", ignore_errors=True)
            shutil.rmtree(workdir / "bin", ignore_errors=True)
    if stopped:
        os._exit(143)
    doc: dict[str, object] = {
        "type": "attempt",
        "session": res.session,
        "seed": seed,
        "steps": len(res.steps),
        "passed": sum(1 for r in res.steps if r["pass"]),
        "usage": res.usage.as_dict(),
        "compaction_usage": res.compaction_usage.as_dict(),
        "compactions": res.compactions,
        "gaps": res.gaps,
        "rater_usage": {r.name: r.usage.as_dict() for r in raters},
        "wall_s": round(res.wall_s, 1),
    }
    append(run_file, doc)
    return doc


def cmd_run(args: argparse.Namespace) -> None:
    profile: Profile = load_profile(args.profile)
    if profile.api_key_env and not os.environ.get(profile.api_key_env):
        raise SystemExit(f"run: {profile.api_key_env} is unset")
    raters = [] if args.no_rater else [r for r in args.raters.split(",") if r]
    for name in raters:
        load_profile(name)
    if args.session:
        jobs = [(args.session, seed, args.steps) for seed in (args.seed or [0])]
        suite = None
    else:
        steps, seeds = SUITES[args.suite]
        only = {t for t in (args.only or "").split(",") if t}
        ids = [tid for tid in suite_ids(args.suite) if not only or tid in only]
        if only - set(ids):
            raise SystemExit(
                f"run: not in suite {args.suite}: {', '.join(sorted(only - set(ids)))}"
            )
        jobs = [(tid, seed, steps) for tid in ids for seed in seeds]
        if not jobs:
            raise SystemExit(f"run: no templates in suite {args.suite}")
        suite = args.suite
    run_id = f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{profile.name}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True)
    lock = hold(run_dir)  # noqa: F841 (held until exit)
    append(
        run_dir / "run.jsonl",
        {
            "type": "run",
            "id": run_id,
            "started": datetime.now(UTC).isoformat(),
            "profile": profile.public(),
            "raters": raters,
            "suite": suite,
            "jobs": [list(j) for j in jobs],
            "harness_hash": tree_hash(PACKAGE_ROOT / "rangebench_homelab", "*.py"),
            "rules_hash": tree_hash(RULES_FILE.parent),
            "template_hashes": {
                tid: tree_hash(template_dir(tid)) for tid in sorted({j[0] for j in jobs})
            },
            "keep_world": args.keep_world,
        },
    )
    print(f"run {run_id}: {len(jobs)} attempts, concurrency {profile.concurrency}", flush=True)
    rater_usage: dict[str, dict[str, int]] = {}
    with ProcessPoolExecutor(max_workers=profile.concurrency) as pool:
        futures = [
            pool.submit(
                _attempt, profile.name, raters, tid, seed, steps, str(run_dir), args.keep_world
            )
            for tid, seed, steps in jobs
        ]
        for fut in as_completed(futures):
            try:
                doc = fut.result()
            except Exception as exc:
                print(f"attempt crashed: {exc!r}", file=sys.stderr, flush=True)
                continue
            usage = doc.get("rater_usage")
            for name, u in usage.items() if isinstance(usage, dict) else []:
                total = rater_usage.setdefault(name, {})
                for k, v in u.items():
                    total[k] = total.get(k, 0) + v
    append(
        run_dir / "run.jsonl",
        {"type": "end", "finished": datetime.now(UTC).isoformat(), "rater_usage": rater_usage},
    )
    print("\n".join(summarize(load(run_dir))))
    print(f"run file: {run_dir / 'run.jsonl'}")


def cmd_gate(args: argparse.Namespace) -> None:
    seeds = [int(s) for s in args.seeds.split(",")]
    template = load_template(args.template)
    started = time.time()
    failures = gate(template, seeds)
    for f in failures:
        print(f"FAIL {f}")
    print(
        f"gate {template.id} seeds {seeds}: {'PASS' if not failures else 'FAIL'} "
        f"({time.time() - started:.1f}s)"
    )
    if failures:
        raise SystemExit(1)


def cmd_report(args: argparse.Namespace) -> None:
    path = Path(args.run)
    if not path.exists():
        path = RUNS_DIR / args.run
    print("\n".join(summarize(load(path))))


def cmd_rescore(args: argparse.Namespace) -> None:
    from .rescore import main as rescore_main

    rescore_main(args.run if Path(args.run).exists() else str(RUNS_DIR / args.run), args.out)


def cmd_compare(args: argparse.Namespace) -> None:
    from .compare import main as compare_main

    compare_main([r if Path(r).exists() else str(RUNS_DIR / r) for r in args.runs], args.json)


def cmd_chart(args: argparse.Namespace) -> None:
    from .chart import main as chart_main

    chart_main(args.rows, args.out, args.title, args.note or [], args.label or [], args.embed)


def cmd_test(args: argparse.Namespace) -> None:
    """Unit tests; the per-command memory test needs a delegated scope (see cgroup.py)."""
    loader = unittest.defaultTestLoader
    root = str(PACKAGE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    suite = (
        loader.loadTestsFromNames(args.names)
        if args.names
        else loader.discover(str(PACKAGE_ROOT / "tests"), top_level_dir=root)
    )
    if not unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful():
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m rangebench_homelab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run a suite or one session")
    run.add_argument("--profile", required=True)
    group = run.add_mutually_exclusive_group(required=True)
    group.add_argument("--suite", choices=sorted(SUITES))
    group.add_argument("--session")
    run.add_argument("--seed", type=int, action="append", help="with --session; repeatable")
    run.add_argument("--only", help="with --suite: comma-separated template ids to run")
    run.add_argument("--steps", type=int, help="with --session: run only the first N steps")
    run.add_argument(
        "--raters",
        default=DEFAULT_RATERS,
        help="comma-separated rater profiles (default $RANGEBENCH_HOMELAB_RATERS, else none)",
    )
    run.add_argument("--no-rater", action="store_true")
    run.add_argument("--keep-world", action="store_true", help="keep each attempt's world dir")
    run.set_defaults(func=cmd_run)
    g = sub.add_parser("gate", help="authoring gates for one template")
    g.add_argument("template")
    g.add_argument("--seeds", default="0,1,2")
    g.set_defaults(func=cmd_gate)
    rep = sub.add_parser("report", help="summarize a run")
    rep.add_argument("run")
    rep.set_defaults(func=cmd_report)
    rs = sub.add_parser(
        "rescore", help="re-score reply-only checks of a run with the current templates"
    )
    rs.add_argument("run", help="run.jsonl or its run dir")
    rs.add_argument("-o", "--out", help="output run.jsonl (default runs/<id>-rescored/run.jsonl)")
    rs.set_defaults(func=cmd_rescore)
    cmp = sub.add_parser("compare", help="side-by-side numbers for several runs")
    cmp.add_argument("runs", nargs="+")
    cmp.add_argument("--json", action="store_true")
    cmp.set_defaults(func=cmd_compare)
    ch = sub.add_parser("chart", help="static results page from `compare --json` output")
    ch.add_argument("rows")
    ch.add_argument("out")
    ch.add_argument("--title", default="RangeBench homelab results")
    ch.add_argument("--note", action="append", help="line under the title; repeatable")
    ch.add_argument(
        "--label", action="append", help="profile=suffix added to that model's label; repeatable"
    )
    ch.add_argument(
        "--embed",
        action="store_true",
        help="host-themed, no page background (for an embedding frame)",
    )
    ch.set_defaults(func=cmd_chart)
    t = sub.add_parser("test", help="run the unit tests")
    t.add_argument("names", nargs="*", help="test modules, classes or methods (default: all)")
    t.set_defaults(func=cmd_test)
    args = ap.parse_args(argv)
    _exit_on_sigterm()
    if args.cmd in ("run", "gate"):
        sweep()
    args.func(args)
