"""Durable, thread-safe budget authority for paid model requests and attempts.

Every request reserves a conservative input (the configured context window)
and output bound before the provider call. Successful calls settle to reported
usage; a transport failure, missing usage, or malformed usage keeps the full
reservation charged and marks the authority closed. Closed authorities reject
all new paid work, so a crash or ambiguous provider outcome cannot silently
spend and rerun.

The ledger is append-only and fsynced. A fresh authority replays it; open
request reservations become conservative unknown charges. Attempt grants are
consumed immediately and never refunded.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent import ChatClientProtocol, ChatResult, ResponseMetadata, Usage


class BudgetError(RuntimeError):
    """New paid work cannot be admitted under the frozen budget."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class BudgetLimits:
    max_attempts: int
    max_requests: int
    max_requests_per_attempt: int
    max_tokens: int
    max_spend_usd: float | None
    input_usd_per_million: float | None
    output_usd_per_million: float | None
    deadline_epoch: float
    ctx_window: int


@dataclass
class BudgetSnapshot:
    attempts: int = 0
    requests: int = 0
    rejected_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    spend_usd: float | None = None
    unknown: bool = False
    open_requests: int = 0
    open_input_tokens: int = 0
    open_output_tokens: int = 0

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "requests": self.requests,
            "rejected_requests": self.rejected_requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tokens": self.tokens,
            "spend_usd": self.spend_usd,
            "unknown": self.unknown,
            "open_requests": self.open_requests,
            "open_input_tokens": self.open_input_tokens,
            "open_output_tokens": self.open_output_tokens,
        }


@dataclass
class _Used:
    attempts: int = 0
    requests: int = 0
    rejected_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    spend_usd: float = 0.0


def _iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class BudgetAuthority:
    def __init__(
        self,
        ledger_path: Path,
        limits: BudgetLimits,
        *,
        clock: Callable[[], float] = time.time,
        recover: bool = True,
    ):
        self.ledger_path = ledger_path
        self.limits = limits
        self._clock = clock
        self._lock = threading.Lock()
        self._used = _Used()
        self._open: dict[str, dict[str, Any]] = {}
        self._attempt_requests: dict[str, int] = {}
        self._attempt_open: dict[str, int] = {}
        self._attempted_cells: set[str] = set()
        self._unknown = False
        self._closed_reason = ""
        self._recover = recover
        self._load()

    def _append(self, record: dict[str, Any]) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        existed = self.ledger_path.exists()
        line = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
        fd = os.open(
            self.ledger_path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            view = memoryview(line)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        if not existed:
            try:
                dir_fd = os.open(self.ledger_path.parent, os.O_RDONLY | os.O_DIRECTORY)
            except OSError:
                return
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

    def _spend(self, input_tokens: int, output_tokens: int) -> float:
        if self.limits.input_usd_per_million is None or self.limits.output_usd_per_million is None:
            return 0.0
        return (
            input_tokens * self.limits.input_usd_per_million
            + output_tokens * self.limits.output_usd_per_million
        ) / 1_000_000

    def _apply_settle(self, record: dict[str, Any]) -> None:
        self._used.requests += 1
        self._used.input_tokens += int(record.get("input") or 0)
        self._used.output_tokens += int(record.get("output") or 0)
        self._used.spend_usd += float(record.get("spend") or 0.0)

    def _apply_unknown(self, record: dict[str, Any]) -> None:
        self._used.requests += 1
        self._used.input_tokens += int(record.get("input_reserve") or 0)
        self._used.output_tokens += int(record.get("output_reserve") or 0)
        self._used.spend_usd += float(record.get("spend_reserve") or 0.0)
        self._unknown = True

    def _corrupt(self, line: int, message: str) -> BudgetError:
        return BudgetError("corrupt_ledger", f"line {line}: {message}")

    def _int_field(self, record: dict[str, Any], key: str, line: int) -> int:
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise self._corrupt(line, f"{key} must be a non-negative integer")
        return value

    def _number_field(self, record: dict[str, Any], key: str, line: int) -> float:
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
            raise self._corrupt(line, f"{key} must be a finite non-negative number")
        try:
            number = float(value)
        except OverflowError:
            raise self._corrupt(line, f"{key} is too large") from None
        if not math.isfinite(number):
            raise self._corrupt(line, f"{key} must be a finite non-negative number")
        return number

    def _text_field(self, record: dict[str, Any], key: str, line: int) -> str:
        value = record.get(key)
        if not isinstance(value, str) or not value:
            raise self._corrupt(line, f"{key} must be a nonempty string")
        return value

    def _load(self) -> None:
        if not self.ledger_path.is_file():
            return
        seen_ids: set[str] = set()
        closed = False
        try:
            ledger_text = self.ledger_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise BudgetError("corrupt_ledger", "ledger is not UTF-8") from None
        except OSError as exc:
            raise BudgetError("corrupt_ledger", f"ledger is unreadable: {exc}") from None
        for line_no, line in enumerate(ledger_text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (ValueError, RecursionError):
                raise self._corrupt(line_no, "not valid JSON") from None
            if not isinstance(record, dict):
                raise self._corrupt(line_no, "record is not an object")
            kind = record.get("kind")
            if kind == "reserve":
                request_id = self._text_field(record, "id", line_no)
                cell = self._text_field(record, "cell", line_no)
                input_reserve = self._int_field(record, "input_reserve", line_no)
                output_reserve = self._int_field(record, "output_reserve", line_no)
                spend_reserve = self._number_field(record, "spend_reserve", line_no)
                if closed:
                    raise self._corrupt(line_no, "reserve after close")
                if request_id in seen_ids or request_id in self._open:
                    raise self._corrupt(line_no, f"duplicate reservation id {request_id}")
                seen_ids.add(request_id)
                stored = {
                    "t": record.get("t"),
                    "kind": "reserve",
                    "id": request_id,
                    "cell": cell,
                    "input_reserve": input_reserve,
                    "output_reserve": output_reserve,
                    "spend_reserve": spend_reserve,
                }
                self._open[request_id] = stored
                self._attempt_requests[cell] = self._attempt_requests.get(cell, 0) + 1
                self._attempt_open[cell] = self._attempt_open.get(cell, 0) + 1
            elif kind in {"settle", "unknown", "reject"}:
                request_id = self._text_field(record, "id", line_no)
                cell = self._text_field(record, "cell", line_no)
                reservation = self._open.get(request_id)
                if reservation is None:
                    raise self._corrupt(line_no, f"terminal event without reservation {request_id}")
                if reservation.get("cell") != cell:
                    raise self._corrupt(line_no, "terminal event cell mismatch")
                if "reason" in record and not isinstance(record.get("reason"), str):
                    raise self._corrupt(line_no, "reason must be a string")
                if self._attempt_open.get(cell, 0) <= 0:
                    raise self._corrupt(line_no, "terminal event without an open attempt request")
                if kind == "settle":
                    input_tokens = self._int_field(record, "input", line_no)
                    output_tokens = self._int_field(record, "output", line_no)
                    spend = self._number_field(record, "spend", line_no)
                    if input_tokens > int(reservation["input_reserve"]):
                        raise self._corrupt(line_no, "settled input exceeds reservation")
                    if output_tokens > int(reservation["output_reserve"]):
                        raise self._corrupt(line_no, "settled output exceeds reservation")
                    if spend > float(reservation["spend_reserve"]) + 1e-12:
                        raise self._corrupt(line_no, "settled spend exceeds reservation")
                    expected_spend = self._spend(input_tokens, output_tokens)
                    if abs(spend - expected_spend) > 1e-9:
                        raise self._corrupt(line_no, "settled spend does not match token rates")
                    applied = {
                        "kind": "settle",
                        "input": input_tokens,
                        "output": output_tokens,
                        "spend": spend,
                    }
                    self._open.pop(request_id, None)
                    self._attempt_open[cell] -= 1
                    self._apply_settle(applied)
                elif kind == "unknown":
                    input_reserve = self._int_field(record, "input_reserve", line_no)
                    output_reserve = self._int_field(record, "output_reserve", line_no)
                    spend_reserve = self._number_field(record, "spend_reserve", line_no)
                    if (
                        input_reserve != int(reservation["input_reserve"])
                        or output_reserve != int(reservation["output_reserve"])
                        or spend_reserve != float(reservation["spend_reserve"])
                    ):
                        raise self._corrupt(line_no, "unknown event does not match reservation")
                    self._open.pop(request_id, None)
                    self._attempt_open[cell] -= 1
                    self._apply_unknown(reservation)
                else:
                    self._open.pop(request_id, None)
                    self._attempt_open[cell] -= 1
                    self._used.requests += 1
                    self._used.rejected_requests += 1
            elif kind == "attempt":
                cell = self._text_field(record, "cell", line_no)
                if closed:
                    raise self._corrupt(line_no, "attempt after close")
                self._used.attempts += 1
                self._attempted_cells.add(cell)
            elif kind == "close":
                reason = self._text_field(record, "reason", line_no)
                if closed:
                    raise self._corrupt(line_no, "duplicate close event")
                closed = True
                self._unknown = True
                self._closed_reason = reason
            else:
                raise self._corrupt(line_no, f"unknown event kind {kind!r}")
        for cell, count in self._attempt_requests.items():
            if count > self.limits.max_requests_per_attempt:
                raise BudgetError(
                    "corrupt_ledger",
                    f"cell {cell} replayed {count} requests over per-attempt cap",
                )
        if not self._recover:
            open_input, open_output, open_spend = self._open_tokens_locked()
            if self._used.requests + len(self._open) > self.limits.max_requests:
                raise BudgetError("corrupt_ledger", "replayed request total exceeds cap")
            if (
                self._used.input_tokens + self._used.output_tokens + open_input + open_output
                > self.limits.max_tokens
            ):
                raise BudgetError("corrupt_ledger", "replayed token total exceeds cap")
            limit = self.limits.max_spend_usd
            if limit is not None and self._used.spend_usd + open_spend > limit:
                raise BudgetError("corrupt_ledger", "replayed spend total exceeds cap")
            return
        for request_id, record in list(self._open.items()):
            self._append(
                {
                    "t": _iso(),
                    "kind": "unknown",
                    "id": request_id,
                    "cell": record.get("cell"),
                    "reason": "unsettled reservation recovered after restart",
                    "input_reserve": record.get("input_reserve"),
                    "output_reserve": record.get("output_reserve"),
                    "spend_reserve": record.get("spend_reserve"),
                }
            )
            self._open.pop(request_id, None)
            cell = str(record.get("cell"))
            if self._attempt_open.get(cell, 0) <= 0:
                raise BudgetError("corrupt_ledger", "recovery without an open attempt request")
            self._attempt_open[cell] -= 1
            self._apply_unknown(record)
        if self._used.attempts > self.limits.max_attempts:
            raise BudgetError("corrupt_ledger", "replayed attempt total exceeds cap")
        if self._used.requests > self.limits.max_requests:
            raise BudgetError("corrupt_ledger", "replayed request total exceeds cap")
        if self._used.input_tokens + self._used.output_tokens > self.limits.max_tokens:
            raise BudgetError("corrupt_ledger", "replayed token total exceeds cap")
        if (
            self.limits.max_spend_usd is not None
            and self._used.spend_usd > self.limits.max_spend_usd
        ):
            raise BudgetError("corrupt_ledger", "replayed spend total exceeds cap")

    def _check_open_locked(self) -> None:
        if self._unknown:
            detail = self._closed_reason or "uncertain outcome"
            raise BudgetError("usage_unknown", f"budget is closed: {detail}")
        if self._clock() >= self.limits.deadline_epoch:
            raise BudgetError("runtime_limit", "experiment clock expired")

    def _open_tokens_locked(self) -> tuple[int, int, float]:
        input_tokens = sum(int(r.get("input_reserve") or 0) for r in self._open.values())
        output_tokens = sum(int(r.get("output_reserve") or 0) for r in self._open.values())
        spend = sum(float(r.get("spend_reserve") or 0.0) for r in self._open.values())
        return input_tokens, output_tokens, spend

    def reserve_request(self, cell: str, max_output_tokens: int) -> str:
        with self._lock:
            self._check_open_locked()
            input_reserve = self.limits.ctx_window
            output_reserve = max_output_tokens
            spend_reserve = self._spend(input_reserve, output_reserve)
            if self._attempt_requests.get(cell, 0) + 1 > self.limits.max_requests_per_attempt:
                raise BudgetError(
                    "attempt_request_limit",
                    f"attempt {cell} reached {self.limits.max_requests_per_attempt} requests",
                )
            open_input, open_output, open_spend = self._open_tokens_locked()
            if self._used.requests + len(self._open) + 1 > self.limits.max_requests:
                raise BudgetError("request_limit", "request budget exhausted")
            if (
                self._used.input_tokens
                + self._used.output_tokens
                + open_input
                + open_output
                + input_reserve
                + output_reserve
                > self.limits.max_tokens
            ):
                raise BudgetError("token_limit", "token budget cannot cover the next request")
            limit = self.limits.max_spend_usd
            if limit is not None and self._used.spend_usd + open_spend + spend_reserve > limit:
                raise BudgetError("spend_limit", "spend budget cannot cover the next request")
            request_id = uuid.uuid4().hex
            record = {
                "t": _iso(),
                "kind": "reserve",
                "id": request_id,
                "cell": cell,
                "input_reserve": input_reserve,
                "output_reserve": output_reserve,
                "spend_reserve": spend_reserve,
            }
            self._append(record)
            self._open[request_id] = record
            self._attempt_requests[cell] = self._attempt_requests.get(cell, 0) + 1
            self._attempt_open[cell] = self._attempt_open.get(cell, 0) + 1
            return request_id

    def settle_request(self, request_id: str, usage: Usage) -> None:
        with self._lock:
            record = self._open.get(request_id)
            if record is None:
                raise BudgetError("unknown_reservation", "request reservation is missing")
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens
            if input_tokens > int(record.get("input_reserve") or 0) or output_tokens > int(
                record.get("output_reserve") or 0
            ):
                self._append(
                    {
                        "t": _iso(),
                        "kind": "unknown",
                        "id": request_id,
                        "cell": record.get("cell"),
                        "reason": "reported usage exceeded reservation",
                        "input_reserve": record.get("input_reserve"),
                        "output_reserve": record.get("output_reserve"),
                        "spend_reserve": record.get("spend_reserve"),
                    }
                )
                self._open.pop(request_id, None)
                cell = str(record.get("cell"))
                self._attempt_open[cell] = max(0, self._attempt_open.get(cell, 1) - 1)
                self._apply_unknown(record)
                raise BudgetError("usage_unknown", "reported usage exceeded reservation")
            settle = {
                "t": _iso(),
                "kind": "settle",
                "id": request_id,
                "cell": record.get("cell"),
                "input": input_tokens,
                "output": output_tokens,
                "spend": self._spend(input_tokens, output_tokens),
            }
            self._append(settle)
            self._open.pop(request_id, None)
            cell = str(record.get("cell"))
            self._attempt_open[cell] = max(0, self._attempt_open.get(cell, 1) - 1)
            self._apply_settle(settle)

    def reject_request(self, request_id: str, reason: str) -> None:
        """A definitive provider rejection (HTTP 4xx) with no completion usage."""
        with self._lock:
            record = self._open.get(request_id)
            if record is None:
                raise BudgetError("unknown_reservation", "request reservation is missing")
            self._append(
                {
                    "t": _iso(),
                    "kind": "reject",
                    "id": request_id,
                    "cell": record.get("cell"),
                    "reason": reason,
                }
            )
            self._open.pop(request_id, None)
            cell = str(record.get("cell"))
            self._attempt_open[cell] = max(0, self._attempt_open.get(cell, 1) - 1)
            self._used.requests += 1
            self._used.rejected_requests += 1

    def fail_request(self, request_id: str, reason: str) -> None:
        with self._lock:
            record = self._open.get(request_id)
            if record is None:
                raise BudgetError("unknown_reservation", "request reservation is missing")
            self._append(
                {
                    "t": _iso(),
                    "kind": "unknown",
                    "id": request_id,
                    "cell": record.get("cell"),
                    "reason": reason,
                    "input_reserve": record.get("input_reserve"),
                    "output_reserve": record.get("output_reserve"),
                    "spend_reserve": record.get("spend_reserve"),
                }
            )
            self._open.pop(request_id, None)
            cell = str(record.get("cell"))
            if self._attempt_open.get(cell, 0) <= 0:
                raise BudgetError("corrupt_ledger", "recovery without an open attempt request")
            self._attempt_open[cell] -= 1
            self._apply_unknown(record)

    def reserve_attempt(self, cell: str) -> None:
        """Charge one logical attempt exactly once across restarts."""
        with self._lock:
            self._check_open_locked()
            if cell in self._attempted_cells:
                return
            if self._used.attempts + 1 > self.limits.max_attempts:
                raise BudgetError("attempt_limit", "attempt budget exhausted")
            self._append({"t": _iso(), "kind": "attempt", "cell": cell})
            self._used.attempts += 1
            self._attempted_cells.add(cell)

    def attempt_reserved(self, cell: str) -> bool:
        with self._lock:
            return cell in self._attempted_cells

    def ensure_attempts(self, count: int) -> None:
        with self._lock:
            self._check_open_locked()
            if self._used.attempts + count > self.limits.max_attempts:
                raise BudgetError(
                    "attempt_limit",
                    f"attempt budget cannot cover {count} more cells",
                )

    def check_deadline(self) -> None:
        with self._lock:
            self._check_open_locked()

    def close(self, reason: str) -> None:
        with self._lock:
            if self._unknown:
                return
            self._append({"t": _iso(), "kind": "close", "reason": reason})
            self._unknown = True
            self._closed_reason = reason

    @staticmethod
    def _matches_cell(candidate: object, cell: str) -> bool:
        text = str(candidate)
        return text == cell or text.startswith(cell + "/")

    def cell_has_open_request(self, cell: str) -> bool:
        with self._lock:
            return any(
                self._matches_cell(record.get("cell"), cell) for record in self._open.values()
            )

    def cell_request_count(self, cell: str) -> int:
        with self._lock:
            return sum(
                count
                for key, count in self._attempt_requests.items()
                if self._matches_cell(key, cell)
            )

    def cell_request_count_exact(self, cell: str) -> int:
        with self._lock:
            return self._attempt_requests.get(cell, 0)

    @property
    def closed_reason(self) -> str:
        with self._lock:
            return self._closed_reason

    def remaining_seconds(self) -> float:
        return max(0.0, self.limits.deadline_epoch - self._clock())

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            open_input, open_output, _ = self._open_tokens_locked()
            spend = (
                self._used.spend_usd
                if self.limits.input_usd_per_million is not None
                and self.limits.output_usd_per_million is not None
                else None
            )
            return BudgetSnapshot(
                attempts=self._used.attempts,
                requests=self._used.requests,
                rejected_requests=self._used.rejected_requests,
                input_tokens=self._used.input_tokens,
                output_tokens=self._used.output_tokens,
                spend_usd=spend,
                unknown=self._unknown,
                open_requests=len(self._open),
                open_input_tokens=open_input,
                open_output_tokens=open_output,
            )


def _http_code(error: str) -> int | None:
    text = error[5:] if error.startswith("HTTP ") else error
    if text.startswith("HTTP "):
        text = text[5:]
    code_text = text.split(":", 1)[0].strip()
    try:
        return int(code_text)
    except ValueError:
        return None


def _definitive_rejection(error: str) -> bool:
    """HTTP 400/413/429 are request-level rejections; auth codes are handled globally."""
    return error.startswith("HTTP ") and _http_code(error) in {400, 413, 429}


def _definitive_auth(error: str) -> bool:
    return error.startswith("HTTP ") and _http_code(error) in {401, 403, 404}


def _uncertain_transport(error: str) -> bool:
    return error.startswith("transport:")


class BudgetedChatClient:
    """Route every chat call through the shared budget authority."""

    def __init__(
        self,
        inner: ChatClientProtocol,
        authority: BudgetAuthority,
        cell: str,
    ):
        self.inner = inner
        self.authority = authority
        self.cell = cell

    def _finish(
        self,
        request_id: str,
        content: str,
        usage: Usage,
        error: str | None,
        metadata: ResponseMetadata,
    ) -> ChatResult:
        complete = (
            usage.calls == 1
            and usage.requests == 1
            and usage.input_reported_calls == 1
            and usage.output_reported_calls == 1
        )
        if error is None and complete:
            self.authority.settle_request(request_id, usage)
            return ChatResult(content, usage, None, metadata)
        if error is not None and complete:
            self.authority.settle_request(request_id, usage)
            if _definitive_auth(error):
                self.authority.close("provider_rejected")
                raise BudgetError("provider_rejected", "provider rejected credentials/model")
            if _uncertain_transport(error):
                self.authority.close("uncertain_transport")
                raise BudgetError("usage_unknown", "transport outcome is uncertain")
            if _definitive_rejection(error):
                return ChatResult(content, usage, error, metadata)
            self.authority.close("provider_error")
            raise BudgetError("provider_error", "provider call failed")
        if error is not None and _definitive_auth(error):
            self.authority.reject_request(request_id, "provider authentication rejection")
            self.authority.close("provider_rejected")
            raise BudgetError("provider_rejected", "provider rejected credentials/model")
        if error is not None and _uncertain_transport(error):
            self.authority.fail_request(request_id, "uncertain transport")
            self.authority.close("uncertain_transport")
            raise BudgetError("usage_unknown", "transport outcome is uncertain")
        if error is not None and _definitive_rejection(error):
            self.authority.reject_request(request_id, "provider rejection")
            return ChatResult(content, usage, error, metadata)
        self.authority.fail_request(request_id, "missing or uncertain provider usage")
        raise BudgetError("usage_unknown", "provider usage is missing or uncertain")

    def chat_result(
        self, messages: list[dict], max_tokens: int, temperature: float = 0.2
    ) -> ChatResult:
        request_id = self.authority.reserve_request(self.cell, max_tokens)
        try:
            rich = getattr(self.inner, "chat_result", None)
            if rich is not None:
                result: ChatResult = rich(messages, max_tokens=max_tokens, temperature=temperature)
            else:
                content, usage, error = self.inner.chat(
                    messages, max_tokens=max_tokens, temperature=temperature
                )
                result = ChatResult(
                    content,
                    usage,
                    error,
                    ResponseMetadata(None, None, None, max_tokens),
                )
        except BaseException:
            self.authority.fail_request(request_id, "transport exception")
            self.authority.close("uncertain_transport")
            raise
        return self._finish(request_id, result.content, result.usage, result.error, result.metadata)

    def chat(
        self, messages: list[dict], max_tokens: int, temperature: float = 0.2
    ) -> tuple[str, Usage, str | None]:
        result = self.chat_result(messages, max_tokens, temperature)
        return result.content, result.usage, result.error
