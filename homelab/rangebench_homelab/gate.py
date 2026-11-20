"""Authoring gates for one template, per seed.

Usage: `python3 -m rangebench_homelab gate <template> --seeds 0,1,2`. Passes when, for every
seed: each step's checks fail on the state before the step, solve.py passes every
step without violations, a no-op agent and a restart-everything agent fail every
step, and the reference run leaves the sim-gap log empty.
"""

from __future__ import annotations

import fcntl
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .agent import Message
from .checks import StepState, evaluate, split_path
from .session import (
    Agent,
    AgentFactory,
    Budget,
    Env,
    NoopAgent,
    RestartAllAgent,
    SolveAgent,
    Turn,
    _quiet,
    run_session,
)
from .template import Session, Step, Template

# On disk next to the package, never /tmp: where /tmp is tmpfs every world file would count
# as RAM against the run's memory cap.
WORK_ROOT = Path(__file__).resolve().parents[1] / ".work"


@dataclass
class PrecheckAgent:
    """SolveAgent that first records which steps already pass on the pre-step state."""

    inner: SolveAgent
    passing_before: list[str] = field(default_factory=list)

    def respond(self, step: Step, message: str, env: Env, budget: Budget) -> Turn:
        if message == step.user:
            sim = env.sim
            before = {
                c.path: sim.read(*split_path(c.path))
                for c in step.checks
                if c.kind == "file" and c.added
            }
            st = StepState(
                sim, [], "", before, sim.local_host, lambda h, c: env.sandbox.run(h, c, "/root")
            )
            with _quiet(sim):
                if all(evaluate(c, st).ok for c in step.checks):
                    self.passing_before.append(step.id)
        return self.inner.respond(step, message, env, budget)


def _always(make: Callable[[], Agent]) -> AgentFactory:
    """An AgentFactory that ignores the session and builds the same kind of agent every time."""
    return lambda session, params, tools: make()


def gate(template: Template, seeds: list[int], workroot: Path | None = None) -> list[str]:
    """Returns failure messages; empty means the template passes."""
    failures: list[str] = []
    workroot = workroot or WORK_ROOT
    workroot.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="rbgate-", dir=workroot))
    lock = (root / ".lock").open("w")  # cli.sweep() removes gate dirs whose lock is free
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        for seed in seeds:
            pre: PrecheckAgent | None = None

            def make_solve(session: Session, params: dict[str, str], tools: list[Message]) -> Agent:
                nonlocal pre
                pre = PrecheckAgent(SolveAgent(template, params))
                return pre

            try:
                ref = run_session(template, seed, make_solve, root / f"solve-{seed}")
            except Exception as exc:  # a solver crash is a gate failure, not a harness crash
                failures.append(f"seed {seed}: solve.py crashed: {exc!r}")
                continue
            assert pre is not None
            for step_id in pre.passing_before:
                failures.append(f"seed {seed}: {step_id}: checks already pass before the step")
            for rec in ref.steps:
                if not rec["pass"]:
                    checks = rec.get("checks")
                    bad = [
                        c
                        for c in (checks if isinstance(checks, list) else [])
                        if isinstance(c, dict) and not c.get("ok")
                    ]
                    failures.append(
                        f"seed {seed}: {rec['step']}: solve.py fails {bad} {rec.get('violations')}"
                    )
            for gap in ref.gaps:
                failures.append(
                    f"seed {seed}: sim gap in reference run: {gap['argv']} ({gap['detail']})"
                )
            for name, factory in (("noop", NoopAgent), ("restart-all", RestartAllAgent)):
                res = run_session(template, seed, _always(factory), root / f"{name}-{seed}")
                for rec in res.steps:
                    if rec["pass"]:
                        failures.append(f"seed {seed}: {rec['step']}: {name} agent passes")
    finally:
        shutil.rmtree(root, ignore_errors=True)
        lock.close()
    return failures
