"""Session runner: one conversation in one world across all steps.

Usage: `run_session(template, seed, make_agent, workdir, emit=...)`. Per step the
agent gets the user message and works through `bash` until it calls `reply`;
then follow-up, checks, forbids and raters run and one step record is emitted.
Agents: LLMAgent (a model with native tools) and scripted agents used by the
authoring gates (SolveAgent, NoopAgent, RestartAllAgent).
"""

from __future__ import annotations

import contextlib
import json
import random
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from . import compaction
from .agent import ChatClient, JsonValue, Message, Usage, is_context_error
from .checks import Outcome, StepState, evaluate, followup_fires, split_path, violations
from .rater import Rater
from .report import is_false_done, step_status
from .sandbox import Sandbox
from .sim import Event, Result, Simulator
from .template import Session, Step, Template

OUTPUT_CHARS = 10_000
COMMAND_TIMEOUT = 60.0
STEP_WALL_SECONDS = 1800.0
MAX_CALL_TOKENS = 16_384
ABORT_REASONS = frozenset({"llm error"})
END_REASONS = (
    "reply",
    "turn budget",
    "output token budget",
    "wall_clock_exceeded",
    "llm error",
    "empty response",
    "context window exhausted",
)

Record = dict[str, JsonValue]


def classify_end_reason(end_reason: str) -> str:
    """Step end taxonomy (from rangebench): reply | budget | provider | protocol."""
    if end_reason == "reply":
        return "reply"
    if end_reason in (
        "turn budget",
        "output token budget",
        "wall_clock_exceeded",
        "context window exhausted",
    ):
        return "budget"
    if end_reason in ("llm error", "aborted"):
        return "provider"
    return "protocol"


def truncate(text: str, limit: int = OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n...[{len(text) - limit} chars omitted]...\n" + text[-half:]


class Env:
    """The agent's bash tool: sandboxed, logged, on the local host in the project dir."""

    def __init__(
        self, sim: Simulator, sandbox: Sandbox, host: str, cwd: str, log: Callable[[Event], None]
    ) -> None:
        self.sim = sim
        self.sandbox = sandbox
        self.host = host
        self.cwd = cwd
        self.log = log

    def bash(self, command: str) -> str:
        self.sim.call += 1
        ev = self.sim.event("bash", self.host, text=command)
        self.sandbox.deadline = time.time() + COMMAND_TIMEOUT
        res = self.sandbox.run(self.host, command, self.cwd, timeout=COMMAND_TIMEOUT, merge=True)
        self.sandbox.deadline = 0.0
        self.sim.sync_files()
        if ev is not None:
            ev.exit = res.exit
            self.log(ev)
        out = truncate(res.stdout)
        if res.exit:
            out += f"\n[exit code {res.exit}]"
        return out or "(no output)"


@dataclass
class Budget:
    turns: int
    output_tokens: int
    deadline: float


@dataclass
class Turn:
    """Outcome of handling one user message."""

    reply: str | None
    end_reason: str
    turns: int = 0
    usage: Usage = field(default_factory=Usage)
    implicit_reply: bool = False
    error: str | None = None


class Agent(Protocol):
    def respond(self, step: Step, message: str, env: Env, budget: Budget) -> Turn: ...


def tool_specs(host: str, cwd: str) -> list[Message]:
    return [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": (
                    f"Run a non-interactive bash command on {host}, your workstation, as root. "
                    f"Each call is a fresh shell starting in {cwd}. Output (stdout+stderr) is "
                    f"truncated to about {OUTPUT_CHARS} characters; commands time out after "
                    f"{COMMAND_TIMEOUT:.0f} s."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The bash command."}
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "reply",
                "description": (
                    "Send your message to the user and end your turn. Use it when the request is "
                    "handled or when you need the user's input. The user only sees what you "
                    "send here."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Message to the user."}
                    },
                    "required": ["text"],
                },
            },
        },
    ]


class LLMAgent:
    """A model driving bash/reply through native tool calls, with compaction."""

    def __init__(
        self, client: ChatClient, system: str, tools: list[Message], ctx_window: int
    ) -> None:
        self.client = client
        self.tools = tools
        self.ctx_window = ctx_window
        self.messages: list[Message] = [{"role": "system", "content": system}]
        self.compaction_usage = Usage()
        self.compactions = 0
        self.last_prompt = 0
        self.last_estimate = 0

    def _summarize(self, prompt: str) -> str | None:
        res = self.client.chat(
            [
                {"role": "system", "content": compaction.COMPACTION_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4000,
        )
        self.compaction_usage.merge(res.usage)
        return None if res.error else res.content.strip() or None

    def _compact(self, force: bool = False) -> bool:
        if not force and not compaction.should_compact(
            self.messages, self.ctx_window, self.last_prompt, self.last_estimate
        ):
            return False
        chunk = max(8_000, int(self.ctx_window * 4 * 0.5))
        new, _ = compaction.compact(self.messages, self._summarize, chunk_chars=chunk)
        if new is self.messages:
            return False
        self.messages = new
        self.compactions += 1
        self.last_prompt = self.last_estimate = 0
        return True

    def respond(self, step: Step, message: str, env: Env, budget: Budget) -> Turn:
        self.messages.append({"role": "user", "content": message})
        turn = Turn(None, "")
        empty = 0
        context_retry = False
        while True:
            if budget.turns <= 0:
                turn.end_reason = "turn budget"
                return turn
            if budget.output_tokens <= 0:
                turn.end_reason = "output token budget"
                return turn
            if time.time() > budget.deadline:
                turn.end_reason = "wall_clock_exceeded"
                return turn
            self._compact()
            est = compaction.calibrated_tokens(self.messages, self.last_prompt, self.last_estimate)
            headroom = self.ctx_window - est - 512
            if headroom < 256:
                if context_retry or not self._compact(force=True):
                    turn.end_reason = "context window exhausted"
                    return turn
                context_retry = True
                continue
            max_tokens = max(1, min(MAX_CALL_TOKENS, budget.output_tokens, headroom))
            res = self.client.chat(self.messages, self.tools, max_tokens)
            turn.usage.merge(res.usage)
            turn.turns += 1
            budget.turns -= 1
            budget.output_tokens -= res.usage.completion_tokens
            if res.error:
                if is_context_error(res.error) and not context_retry and self._compact(force=True):
                    context_retry = True
                    continue
                turn.end_reason = (
                    "context window exhausted" if is_context_error(res.error) else "llm error"
                )
                turn.error = res.error[:300]
                return turn
            self.last_prompt = res.usage.prompt_tokens
            self.last_estimate = compaction.estimate_tokens(self.messages)
            if not res.tool_calls:
                if res.content.strip():
                    self.messages.append(res.message())
                    turn.reply, turn.end_reason, turn.implicit_reply = (
                        res.content.strip(),
                        "reply",
                        True,
                    )
                    return turn
                empty += 1
                if empty >= 3:
                    turn.end_reason = "empty response"
                    return turn
                continue
            self.messages.append(res.message())
            final: str | None = None
            for call in res.tool_calls:
                if final is not None:
                    out = "Not run: you already replied to the user."
                else:
                    try:
                        args = json.loads(call.arguments or "{}")
                        if not isinstance(args, dict):
                            raise ValueError("arguments must be an object")
                    except ValueError as exc:
                        args = None
                        out = f"error: invalid JSON arguments ({exc})"
                    if args is not None:
                        if call.name == "bash" and isinstance(args.get("command"), str):
                            out = env.bash(args["command"])
                        elif call.name == "reply" and isinstance(args.get("text"), str):
                            final = args["text"]
                            out = "Delivered to the user."
                        else:
                            out = f"error: unknown tool or missing argument ({call.name})"
                self.messages.append({"role": "tool", "tool_call_id": call.id, "content": out})
            if final is not None:
                turn.reply, turn.end_reason = final, "reply"
                return turn


class SolveAgent:
    """Reference solution from the template's solve.py."""

    def __init__(self, template: Template, params: dict[str, str]) -> None:
        self.template = template
        self.params = params

    def respond(self, step: Step, message: str, env: Env, budget: Budget) -> Turn:
        if message != step.user:
            return Turn("Checked again; it works on my side.", "reply")
        return Turn(self.template.solve(step.id, env.bash, self.params), "reply")


class NoopAgent:
    def respond(self, step: Step, message: str, env: Env, budget: Budget) -> Turn:
        return Turn("Done.", "reply")


class RestartAllAgent:
    """Restarts every service on every host, then claims success."""

    def respond(self, step: Step, message: str, env: Env, budget: Budget) -> Turn:
        for svc in env.sim.services:
            verb = "docker restart" if svc.runtime == "docker" else "systemctl restart"
            env.bash(f"ssh {svc.host} {verb} {svc.name}")
        return Turn("Restarted everything, all good now.", "reply")


AgentFactory = Callable[[Session, dict[str, str], list[Message]], Agent]


@contextlib.contextmanager
def _quiet(sim: Simulator) -> Iterator[None]:
    """Harness-side evaluation: nothing lands in the agent's command log."""
    recording, sim.recording = sim.recording, False
    try:
        yield
    finally:
        sim.recording = recording
        sim.sync_files(record=False)


def _jsonl(path: Path) -> Callable[[Mapping[str, object]], None]:
    def write(doc: Mapping[str, object]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    return write


@dataclass
class AttemptResult:
    session: str
    seed: int
    steps: list[Record]
    gaps: list[dict[str, str]]
    usage: Usage
    compaction_usage: Usage
    compactions: int
    wall_s: float


def run_session(
    template: Template,
    seed: int,
    make_agent: AgentFactory,
    workdir: Path,
    max_steps: int | None = None,
    raters: list[Rater] | None = None,
    wall_clock_scale: float = 1.0,
    emit: Callable[[Record], None] | None = None,
) -> AttemptResult:
    started = time.time()
    workdir.mkdir(parents=True, exist_ok=True)
    sim = Simulator(workdir / "world")
    params = template.build(sim, random.Random(seed))
    sim.local_host = params["local_host"]
    session = template.resolve(params)
    log_event = _jsonl(workdir / "commands.jsonl")
    log_gap = _jsonl(workdir / "simgap.jsonl")
    sim.on_event = lambda e: log_event(e.as_dict())
    sim.on_gap = log_gap
    (workdir / "params.json").write_text(json.dumps(params, indent=2))
    steps = session.steps[: max_steps or len(session.steps)]
    records: list[Record] = []
    total = Usage()
    agent: Agent | None = None
    with Sandbox(sim, workdir) as sandbox:
        env = Env(
            sim, sandbox, sim.local_host, params["project_dir"], lambda e: log_event(e.as_dict())
        )
        agent = make_agent(session, params, tool_specs(sim.local_host, params["project_dir"]))

        def quiet_run(host: str, command: str) -> Result:
            return sandbox.run(host, command, "/root", timeout=COMMAND_TIMEOUT)

        aborted = False
        for idx, step in enumerate(steps, 1):
            if aborted:
                rec: Record = {
                    "type": "step",
                    "session": session.id,
                    "seed": seed,
                    "position": idx,
                    "step": step.id,
                    "family": step.family,
                    "trap": step.trap,
                    "pass": False,
                    "status": "error",
                    "end_reason": "aborted",
                    "end_class": "provider",
                }
                records.append(rec)
                if emit:
                    emit(rec)
                continue
            rec = _run_step(
                session, step, idx, seed, sim, env, agent, quiet_run, raters or [], wall_clock_scale
            )
            total.merge(_usage_of(rec))
            records.append(rec)
            if emit:
                emit(rec)
            aborted = rec["end_reason"] in ABORT_REASONS
    llm = agent if isinstance(agent, LLMAgent) else None
    if llm is not None:
        (workdir / "transcript.json").write_text(
            json.dumps(llm.messages, indent=1, ensure_ascii=False)
        )
    return AttemptResult(
        session.id,
        seed,
        records,
        list(sim.gaps),
        total,
        llm.compaction_usage if llm else Usage(),
        llm.compactions if llm else 0,
        time.time() - started,
    )


def _usage_of(rec: Record) -> Usage:
    u = Usage()
    tokens = rec.get("tokens")
    if isinstance(tokens, dict):
        for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "calls", "requests"):
            v = tokens.get(key)
            if isinstance(v, int):
                setattr(u, key, v)
    return u


def _run_step(
    session: Session,
    step: Step,
    idx: int,
    seed: int,
    sim: Simulator,
    env: Env,
    agent: Agent,
    quiet_run: Callable[[str, str], Result],
    raters: list[Rater],
    wall_clock_scale: float,
) -> Record:
    started = time.time()
    sim.step = idx
    sim.sync_files(record=False)
    added_paths = [c.path for c in step.checks if c.kind == "file" and c.added]
    before = {p: sim.read(*split_path(p)) for p in added_paths}
    budget = Budget(
        step.turns, step.output_tokens, time.time() + STEP_WALL_SECONDS * wall_clock_scale
    )
    turn = agent.respond(step, step.user, env, budget)
    usage, turns = turn.usage, turn.turns
    requests = [step.user]

    def state(reply: str) -> StepState:
        events = [e for e in sim.events if e.step == idx]
        return StepState(sim, events, reply, before, sim.local_host, quiet_run)

    followup_fired = False
    if step.followup and turn.end_reason == "reply":
        with _quiet(sim):
            fires = followup_fires(step.followup, step.checks, state(turn.reply or ""))
        if fires:
            followup_fired = True
            requests.append(step.followup.message)
            second = agent.respond(step, step.followup.message, env, budget)
            usage.merge(second.usage)
            turns += second.turns
            turn = Turn(
                second.reply, second.end_reason, turns, usage, second.implicit_reply, second.error
            )
    reply = turn.reply or ""
    labels: dict[str, str | None] = {}
    rater_errors: dict[str, JsonValue] = {}
    if reply:
        for rater in raters:
            label, err = rater.rate(requests, reply)
            labels[rater.name] = label
            if err:
                rater_errors[rater.name] = err
    st = state(reply)
    st.labels = labels
    with _quiet(sim):
        outcomes = [evaluate(c, st) for c in step.checks]
        active = [f for f in session.forbids if f.id not in step.waive] + list(step.forbids)
        viols: list[Outcome] = [v for f in active for v in violations(f, st)]
    checks_ok = all(o.ok for o in outcomes)
    end_class = classify_end_reason(turn.end_reason)
    return {
        "type": "step",
        "session": session.id,
        "seed": seed,
        "position": idx,
        "step": step.id,
        "family": step.family,
        "trap": step.trap,
        "pass": checks_ok and not viols,
        "status": step_status(checks_ok and not viols, end_class),
        "checks": [o.as_dict() for o in outcomes],
        "violations": [v.as_dict() for v in viols],
        "reply": reply,
        "implicit_reply": turn.implicit_reply,
        "followup": followup_fired,
        "labels": dict(labels),
        "rater_errors": rater_errors,
        "false_done": is_false_done(labels, len(raters), checks_ok),
        "end_reason": turn.end_reason,
        "end_class": end_class,
        "error": turn.error,
        "turns": turns,
        "commands": sum(1 for e in sim.events if e.step == idx and e.kind == "bash"),
        "tokens": dict(usage.as_dict()),
        "wall_s": round(time.time() - started, 1),
    }
