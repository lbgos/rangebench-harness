"""OpenAI-compatible chat client with native tool calling, retries and usage accounting.

Usage: `client = ChatClient(base_url, key, model, extra={"reasoning_effort": "high"})`,
then `client.chat(messages, tools, max_tokens)` -> Reply. Transient failures
(429, 5xx, timeouts, connection errors) retry with exponential backoff, and a 4xx refusal
from the proxy (e.g. "claude refusal (reasoning_extraction)") retries up to REFUSAL_RETRIES
times; other HTTP errors return immediately in Reply.error. The session marks a step whose
call still fails `error`, and reports leave it out of the pass rate.
"""

from __future__ import annotations

import contextlib
import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
Message = dict[str, JsonValue]
RETRY_STATUS = frozenset({408, 409, 425, 429})  # plus every 5xx
# A 4xx whose body reads as a refusal or content block by the provider or proxy;
# often passes on retry.
REFUSAL = re.compile(
    r"refusal|refused|content.?(filter|policy|management)|safety|blocked", re.IGNORECASE
)
REFUSAL_RETRIES = 2


@dataclass
class Usage:
    """Token accounting across calls. completion_tokens includes reasoning tokens."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    calls: int = 0
    requests: int = 0

    def add(self, raw: JsonValue) -> None:
        self.calls += 1
        if not isinstance(raw, dict):
            return

        def num(d: JsonValue, key: str) -> int:
            v = d.get(key) if isinstance(d, dict) else None
            return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else 0

        prompt = num(raw, "prompt_tokens") or num(raw, "input_tokens")
        completion = num(raw, "completion_tokens") or num(raw, "output_tokens")
        total = num(raw, "total_tokens")
        if total > prompt + completion:
            completion = total - prompt
        details = raw.get("completion_tokens_details") or raw.get("output_tokens_details")
        reasoning = num(details, "reasoning_tokens")
        self.prompt_tokens += prompt
        self.completion_tokens += max(completion, reasoning)
        self.reasoning_tokens += reasoning
        self.cache_read_tokens += num(raw.get("prompt_tokens_details"), "cached_tokens")

    def merge(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.calls += other.calls
        self.requests += other.requests

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "calls": self.calls,
            "requests": self.requests,
        }


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class Reply:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning: str = ""
    finish_reason: str = ""
    usage: Usage = field(default_factory=Usage)
    error: str | None = None

    def message(self) -> Message:
        """The assistant message to append to the history."""
        msg: Message = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments},
                }
                for c in self.tool_calls
            ]
        if self.reasoning:
            msg["reasoning_content"] = self.reasoning
        return msg


def is_context_error(error: str) -> bool:
    lower = error.lower()
    return any(
        m in lower
        for m in (
            "context_length_exceeded",
            "context length",
            "context window",
            "maximum context",
            "prompt is too long",
            "http 413",
        )
    )


class ChatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        extra: dict[str, JsonValue] | None = None,
        timeout: float = 600,
        retries: int = 6,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.extra = dict(extra or {})
        self.timeout = timeout
        self.retries = retries
        self.sleep = sleep

    def chat(
        self,
        messages: list[Message],
        tools: list[Message] | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> Reply:
        payload: dict[str, JsonValue] = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "stream": False,
            **self.extra,
        }
        if tools:
            payload["tools"] = list(tools)
        if temperature is not None:
            payload["temperature"] = temperature
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(payload).encode()
        usage = Usage()
        error = "no attempt"
        refusals = 0
        for attempt in range(self.retries + 1):
            if attempt:
                self.sleep(min(60.0, 2.0 * 2 ** (attempt - 1)))
            usage.requests += 1
            req = urllib.request.Request(self.url, data=data, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode())
            except urllib.error.HTTPError as exc:
                detail = ""
                with contextlib.suppress(Exception):
                    detail = exc.read().decode(errors="replace")[:500]
                error = f"HTTP {exc.code}: {detail}"
                if is_context_error(error):
                    return Reply(usage=usage, error=error)
                if exc.code in RETRY_STATUS or exc.code >= 500:
                    continue
                if REFUSAL.search(detail) and refusals < REFUSAL_RETRIES:
                    refusals += 1
                    continue
                return Reply(usage=usage, error=error)
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                OSError,
                ValueError,
            ) as exc:
                error = f"transport: {exc}"
                continue
            return self._parse(body, usage)
        return Reply(usage=usage, error=error)

    def _parse(self, body: JsonValue, usage: Usage) -> Reply:
        if not isinstance(body, dict):
            return Reply(usage=usage, error="malformed response")
        usage.add(body.get("usage"))
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return Reply(usage=usage, error=f"no choices: {str(body)[:300]}")
        choice = choices[0]
        msg = choice.get("message")
        if not isinstance(msg, dict):
            return Reply(usage=usage, error="no message")
        content = msg.get("content")
        if isinstance(content, list):
            content = "".join(
                str(p.get("text", ""))
                for p in content
                if isinstance(p, dict) and p.get("type") not in ("reasoning", "thinking")
            )
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        calls: list[ToolCall] = []
        raw_calls = msg.get("tool_calls")
        for i, c in enumerate(raw_calls if isinstance(raw_calls, list) else []):
            fn = c.get("function") if isinstance(c, dict) else None
            if not isinstance(c, dict) or not isinstance(fn, dict):
                continue
            args = fn.get("arguments")
            calls.append(
                ToolCall(
                    str(c.get("id") or f"call_{i}"),
                    str(fn.get("name") or ""),
                    args if isinstance(args, str) else json.dumps(args),
                )
            )
        finish = choice.get("finish_reason")
        return Reply(
            content if isinstance(content, str) else "",
            calls,
            reasoning if isinstance(reasoning, str) else "",
            finish if isinstance(finish, str) else "",
            usage,
        )
