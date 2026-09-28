"""Live, secret-free progress for a single run unit."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from .runner import AttemptResult


def _duration(seconds: float) -> str:
    minutes, rest = divmod(round(seconds), 60)
    return f"{minutes}m{rest:02d}s"


class Progress:
    def __init__(
        self,
        log_dir: Path,
        status_path: Path | None,
        task_ids: list[str],
        trials: int,
        max_attempts: int | None,
        infra_retries: int = 1,
        probe_attempts: int = 0,
    ):
        self.log_dir = log_dir
        self.status_path = status_path
        self.limit = (
            max_attempts
            if max_attempts is not None
            else len(task_ids) * trials * (1 + infra_retries) + probe_attempts
        )
        self.max_attempts = max_attempts
        self.used = 0
        self.model_attempts = 0
        self.infra_retries = 0
        self.tokens = 0
        self.started = datetime.now(UTC).isoformat()
        self.current: dict | None = None
        self.finished: str | None = None
        self.state = "running"
        self.tasks = {
            f"{tid}:t{trial}": {
                "task": tid,
                "trial": trial,
                "state": "pending",
                "attempts_used": 0,
                "infra_retries": 0,
                "skip_reason": None,
            }
            for tid in task_ids
            for trial in range(1, trials + 1)
        }
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log = (log_dir / "live.log").open("w", encoding="utf-8")
        self.update()

    def update(self) -> None:
        if self.status_path is None:
            return
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        status = {
            "state": self.state,
            "started": self.started,
            "finished": self.finished,
            "attempts_used": self.used,
            "attempts_remaining": max(0, self.limit - self.used),
            "attempts_limit": self.limit,
            "max_attempts": self.max_attempts,
            "model_attempts_used": self.model_attempts,
            "infra_retries": self.infra_retries,
            "tokens": self.tokens,
            "current": self.current,
            "tasks": self.tasks,
        }
        tmp = self.status_path.with_name(self.status_path.name + ".tmp")
        tmp.write_text(json.dumps(status, indent=2))
        os.replace(tmp, self.status_path)

    def line(self, text: str) -> None:
        line = f"[{datetime.now(UTC):%H:%M:%S}] {text}"
        print(line, flush=True)
        self.log.write(line + "\n")
        self.log.flush()
        self.update()

    def start(self, tid: str, trial: int, retry: int) -> None:
        slot = self.tasks[f"{tid}:t{trial}"]
        slot["state"] = "running"
        self.current = {"task": tid, "trial": trial, "retry": retry, "event": "started"}
        self.used += 1
        if retry:
            self.infra_retries += 1
            slot["infra_retries"] = retry
        label = f" · infra retry {retry}" if retry else ""
        self.line(f"attempts {self.used}/{self.limit} · {tid} t{trial} · started{label}")

    def start_probe(self) -> None:
        """The reference calibration probe starts an environment and consumes the same budget."""
        self.used += 1
        self.current = {"event": "reference-probe"}
        self.line(f"attempts {self.used}/{self.limit} · reference probe started")

    def end_probe(self) -> None:
        self.current = None
        self.line("reference probe finished")

    def event(self, tid: str, trial: int, kind: str, data: dict) -> None:
        text = {
            "submit": "result submitted",
            "stage-completed": f"stage completed: {data.get('stage')}",
            "incorrect-submission": f"incorrect submission ({data.get('wrong')})",
            "compaction": "compaction fired",
            "compaction-fallback": "compaction fired (fallback)",
            "context-retry": f"context adaptation retry (window {data.get('ctx_window')})",
            "generation-retry": f"generation adaptation retry (cap {data.get('max_tokens')})",
            "compaction-retry": f"compaction adaptation retry (window {data.get('ctx_window')})",
        }.get(kind)
        if text:
            self.current = {"task": tid, "trial": trial, "event": kind}
            self.line(f"{tid} t{trial} · {text}")

    def end(
        self, tid: str, trial: int, res: AttemptResult, scored: bool, solved: bool, retrying: bool
    ) -> None:
        slot = self.tasks[f"{tid}:t{trial}"]
        if scored:
            self.model_attempts += 1
            slot["attempts_used"] = 1
        slot["state"] = (
            "retrying"
            if retrying
            else "solved"
            if scored and solved
            else "unsolved"
            if scored
            else "invalid"
        )
        self.current = None
        self.tokens += res.prompt_tokens + res.completion_tokens + res.compaction_tokens
        outcome = (
            "SOLVED" if slot["state"] == "solved" else "UNSCORED" if not scored else "unsolved"
        )
        self.line(
            f"{tid} t{trial} · {outcome} in {_duration(res.wall_s)} · "
            f"turns {res.turns_used} · "
            f"prompt/completion {res.prompt_tokens}/{res.completion_tokens} tok · "
            f"run {self.tokens} tok" + (" · retry pending" if retrying else "")
        )

    def skip(self, tid: str, trial: int, reason: str) -> None:
        slot = self.tasks[f"{tid}:t{trial}"]
        slot["state"] = "skipped"
        slot["skip_reason"] = reason
        self.current = None
        self.line(f"{tid} t{trial} · skipped: {reason}")

    def finish(self, state: str) -> None:
        self.state = state
        self.finished = datetime.now(UTC).isoformat()
        self.line(f"run {state} · attempts {self.used}/{self.limit} · {self.tokens} tok")
        self.log.close()
