"""Durable budget authority and budgeted chat client tests."""

import io
import json
import tempfile
import threading
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from rangebench.agent import AnthropicChatClient, ChatClient, ChatResult, ResponseMetadata, Usage
from rangebench.budget import BudgetAuthority, BudgetedChatClient, BudgetError, BudgetLimits
from rangebench.runner import _summary_call


class FakeResponse:
    def __init__(self, body: dict):
        self.body = body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.body).encode()


def _http_error(code: int, body: bytes = b'{"error":"provider"}') -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://example.test/v1",
        code,
        "provider",
        {},
        io.BytesIO(body),
    )


def _limits(
    *,
    attempts: int = 10,
    requests: int = 10,
    tokens: int = 10_000,
    ctx: int = 100,
    spend: float | None = None,
) -> BudgetLimits:
    return BudgetLimits(
        max_attempts=attempts,
        max_requests=requests,
        max_requests_per_attempt=min(requests, 8),
        max_tokens=tokens,
        max_spend_usd=spend,
        input_usd_per_million=1.0 if spend is not None else None,
        output_usd_per_million=1.0 if spend is not None else None,
        deadline_epoch=10**12,
        ctx_window=ctx,
    )


class _CompleteClient:
    def __init__(self, input_tokens: int = 10, output_tokens: int = 5):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.calls = 0

    def chat(
        self, _messages: list[dict], max_tokens: int, temperature: float = 0.2
    ) -> tuple[str, Usage, None]:
        self.calls += 1
        return (
            "ok",
            Usage(
                prompt_tokens=self.input_tokens,
                completion_tokens=self.output_tokens,
                calls=1,
                requests=1,
                input_reported_calls=1,
                output_reported_calls=1,
            ),
            None,
        )


class BudgetAuthorityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "budget.jsonl"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_reservation_prevents_mid_request_overshoot(self) -> None:
        authority = BudgetAuthority(self.path, _limits(tokens=250))
        first = authority.reserve_request("cell", 100)
        authority.settle_request(
            first,
            Usage(
                prompt_tokens=100,
                completion_tokens=0,
                calls=1,
                requests=1,
                input_reported_calls=1,
                output_reported_calls=1,
            ),
        )
        with self.assertRaisesRegex(BudgetError, "token budget"):
            authority.reserve_request("cell", 100)
        snapshot = authority.snapshot()
        self.assertLessEqual(snapshot.tokens, 250)
        self.assertEqual(snapshot.requests, 1)

    def test_attempt_boundary_and_retries_are_counted(self) -> None:
        authority = BudgetAuthority(self.path, _limits(attempts=2))
        authority.reserve_attempt("cell-a")
        authority.reserve_attempt("cell-b")
        with self.assertRaisesRegex(BudgetError, "attempt budget"):
            authority.reserve_attempt("cell-c")
        self.assertEqual(authority.snapshot().attempts, 2)

    def test_attempt_reservation_is_idempotent_per_cell(self) -> None:
        authority = BudgetAuthority(self.path, _limits(attempts=2))
        authority.reserve_attempt("cell-a")
        authority.reserve_attempt("cell-a")
        self.assertEqual(authority.snapshot().attempts, 1)
        authority.ensure_attempts(0)
        authority.ensure_attempts(1)
        authority.reserve_attempt("cell-b")
        with self.assertRaisesRegex(BudgetError, "attempt budget"):
            authority.reserve_attempt("cell-c")
        replayed = BudgetAuthority(self.path, _limits(attempts=2))
        self.assertEqual(replayed.snapshot().attempts, 2)
        replayed.reserve_attempt("cell-a")
        self.assertEqual(replayed.snapshot().attempts, 2)
        self.assertTrue(replayed.attempt_reserved("cell-b"))
        self.assertFalse(replayed.attempt_reserved("cell-c"))

    def test_concurrent_reservations_do_not_oversubscribe(self) -> None:
        authority = BudgetAuthority(self.path, _limits(tokens=300))
        results: list[str] = []

        def reserve() -> None:
            try:
                authority.reserve_request("cell", 100)
                results.append("ok")
            except BudgetError:
                results.append("stop")

        threads = [threading.Thread(target=reserve) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sorted(results), ["ok", "stop"])
        self.assertEqual(authority.snapshot().open_requests, 1)

    def test_missing_usage_closes_budget_with_full_charge(self) -> None:
        class NoUsage:
            def chat(
                self, _messages: list[dict], max_tokens: int, temperature: float = 0.2
            ) -> tuple[str, Usage, None]:
                return "", Usage(), None

        authority = BudgetAuthority(self.path, _limits(tokens=10_000))
        client = BudgetedChatClient(NoUsage(), authority, "cell")
        with self.assertRaisesRegex(BudgetError, "usage"):
            client.chat([{"role": "user", "content": "x"}], 100)
        snapshot = authority.snapshot()
        self.assertTrue(snapshot.unknown)
        self.assertEqual(snapshot.tokens, 200)
        self.assertEqual(snapshot.requests, 1)
        with self.assertRaisesRegex(BudgetError, "closed"):
            authority.reserve_request("cell", 100)

    def test_transport_failure_closes_budget(self) -> None:
        class Broken:
            def chat(
                self, _messages: list[dict], max_tokens: int, temperature: float = 0.2
            ) -> tuple[str, Usage, None]:
                raise OSError("offline")

        authority = BudgetAuthority(self.path, _limits(tokens=10_000))
        client = BudgetedChatClient(Broken(), authority, "cell")
        with self.assertRaises(OSError):
            client.chat([{"role": "user", "content": "x"}], 100)
        snapshot = authority.snapshot()
        self.assertTrue(snapshot.unknown)
        self.assertEqual(snapshot.tokens, 200)

    def test_definitive_rejection_counts_request_without_usage(self) -> None:
        class Rejected:
            def chat(
                self, _messages: list[dict], max_tokens: int, temperature: float = 0.2
            ) -> tuple[str, Usage, str]:
                return "", Usage(calls=1, requests=1), "HTTP 400: context length"

        authority = BudgetAuthority(self.path, _limits(tokens=10_000))
        client = BudgetedChatClient(Rejected(), authority, "cell")
        _, _, error = client.chat([{"role": "user", "content": "x"}], 100)
        self.assertIn("HTTP 400", error)
        snapshot = authority.snapshot()
        self.assertFalse(snapshot.unknown)
        self.assertEqual(snapshot.requests, 1)
        self.assertEqual(snapshot.tokens, 0)

    def test_open_reservation_is_charged_after_restart(self) -> None:
        authority = BudgetAuthority(self.path, _limits(tokens=10_000))
        authority.reserve_request("cell", 100)
        recovered = BudgetAuthority(self.path, _limits(tokens=10_000))
        snapshot = recovered.snapshot()
        self.assertTrue(snapshot.unknown)
        self.assertEqual(snapshot.tokens, 200)
        with self.assertRaisesRegex(BudgetError, "closed"):
            recovered.reserve_request("cell", 100)

    def test_chat_result_metadata_is_preserved(self) -> None:
        class Rich:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, *_args: object, **_kwargs: object) -> None:
                raise AssertionError("chat should not be called")

            def chat_result(
                self,
                _messages: list[dict],
                max_tokens: int,
                temperature: float = 0.2,
            ) -> ChatResult:
                self.calls += 1
                return ChatResult(
                    "ANSWER: x",
                    Usage(
                        prompt_tokens=10,
                        completion_tokens=5,
                        calls=1,
                        requests=1,
                        input_reported_calls=1,
                        output_reported_calls=1,
                    ),
                    None,
                    ResponseMetadata("stop", False, True, max_tokens),
                )

        inner = Rich()
        authority = BudgetAuthority(self.path, _limits(tokens=10_000))
        client = BudgetedChatClient(inner, authority, "cell")
        result = client.chat_result([{"role": "user", "content": "x"}], 50)
        self.assertEqual(result.metadata.finish_reason, "stop")
        self.assertTrue(result.metadata.reasoning_content_present)
        self.assertEqual(inner.calls, 1)
        self.assertEqual(authority.snapshot().requests, 1)

    def test_http_error_with_usage_is_charged_and_classified(self) -> None:
        cases = (
            ("HTTP 401: nope", True, 1, True),
            ("transport: HTTP 429: slow down", True, 1, False),
            ("transport: HTTP 500: boom", True, 1, False),
            ("HTTP 400: bad request", False, 1, False),
            ("HTTP 413: too large", False, 1, False),
            ("HTTP 429: slow down", False, 1, False),
        )
        for error, raises, requests, unknown in cases:
            with self.subTest(error=error):

                class ErrorClient:
                    def chat(
                        self,
                        _messages: list[dict],
                        max_tokens: int,
                        temperature: float = 0.2,
                    ):
                        return (
                            "",
                            Usage(
                                prompt_tokens=10,
                                completion_tokens=0,
                                calls=1,
                                requests=1,
                                input_reported_calls=1,
                                output_reported_calls=1,
                            ),
                            error,
                        )

                path = Path(self.temp.name) / f"http-{abs(hash(error))}.jsonl"
                authority = BudgetAuthority(path, _limits(tokens=10_000))
                client = BudgetedChatClient(ErrorClient(), authority, "cell")
                if raises:
                    with self.assertRaises(BudgetError):
                        client.chat([{"role": "user", "content": "x"}], 100)
                else:
                    _, _, returned = client.chat([{"role": "user", "content": "x"}], 100)
                    self.assertEqual(returned, error)
                snapshot = authority.snapshot()
                self.assertEqual(snapshot.tokens, 10)
                self.assertEqual(snapshot.requests, requests)
                self.assertEqual(snapshot.unknown, raises)

    def test_http_400_without_usage_counts_request_only(self) -> None:
        class Rejected:
            def chat(self, _messages: list[dict], max_tokens: int, temperature: float = 0.2):
                return "", Usage(calls=1, requests=1), "HTTP 400: bad"

        authority = BudgetAuthority(self.path, _limits(tokens=10_000))
        client = BudgetedChatClient(Rejected(), authority, "cell")
        _, _, error = client.chat([{"role": "user", "content": "x"}], 100)
        self.assertIn("HTTP 400", error)
        snapshot = authority.snapshot()
        self.assertFalse(snapshot.unknown)
        self.assertEqual((snapshot.requests, snapshot.tokens), (1, 0))

    def test_real_openai_429_then_200_rejects_then_settles(self) -> None:
        authority = BudgetAuthority(self.path, _limits(tokens=10_000))
        client = BudgetedChatClient(
            ChatClient("https://example.test/v1", "placeholder", "model"),
            authority,
            "cell-429",
        )
        body = {
            "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        with patch(
            "rangebench.agent.urllib.request.urlopen",
            side_effect=[_http_error(429), FakeResponse(body)],
        ):
            _, _, error = client.chat([{"role": "user", "content": "x"}], 100)
            self.assertIn("HTTP 429", error or "")
            self.assertFalse(authority.snapshot().unknown)
            content, _, second_error = client.chat([{"role": "user", "content": "x"}], 100)
        self.assertEqual(content, "ok")
        self.assertIsNone(second_error)
        snapshot = authority.snapshot()
        self.assertEqual(snapshot.requests, 2)
        self.assertEqual(snapshot.rejected_requests, 1)
        self.assertEqual(snapshot.tokens, 15)
        self.assertFalse(snapshot.unknown)
        events = [json.loads(line) for line in self.path.read_text().splitlines()]
        self.assertEqual(
            [event["kind"] for event in events],
            ["reserve", "reject", "reserve", "settle"],
        )

    def test_real_anthropic_429_then_200_rejects_then_settles(self) -> None:
        authority = BudgetAuthority(self.path, _limits(tokens=10_000))
        client = BudgetedChatClient(
            AnthropicChatClient("https://example.test", "placeholder", "model"),
            authority,
            "cell-anthropic-429",
        )
        body = {
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        with patch(
            "rangebench.agent.urllib.request.urlopen",
            side_effect=[_http_error(429), FakeResponse(body)],
        ):
            _, _, error = client.chat([{"role": "user", "content": "x"}], 100)
            self.assertIn("HTTP 429", error or "")
            self.assertFalse(authority.snapshot().unknown)
            content, _, second_error = client.chat([{"role": "user", "content": "x"}], 100)
        self.assertEqual(content, "ok")
        self.assertIsNone(second_error)
        snapshot = authority.snapshot()
        self.assertEqual(snapshot.requests, 2)
        self.assertEqual(snapshot.rejected_requests, 1)
        self.assertEqual(snapshot.tokens, 15)
        self.assertFalse(snapshot.unknown)

    def test_real_503_closes_with_full_reservation(self) -> None:
        authority = BudgetAuthority(self.path, _limits(tokens=10_000))
        client = BudgetedChatClient(
            ChatClient("https://example.test/v1", "placeholder", "model"),
            authority,
            "cell-503",
        )
        with (
            patch(
                "rangebench.agent.urllib.request.urlopen",
                side_effect=_http_error(503),
            ),
            self.assertRaises(BudgetError),
        ):
            client.chat([{"role": "user", "content": "x"}], 100)
        snapshot = authority.snapshot()
        self.assertTrue(snapshot.unknown)
        self.assertEqual(snapshot.requests, 1)
        self.assertEqual(snapshot.tokens, 200)
        replayed = BudgetAuthority(self.path, _limits(tokens=10_000))
        self.assertTrue(replayed.snapshot().unknown)

    def test_repeated_real_429_is_bounded_by_attempt_cap(self) -> None:
        authority = BudgetAuthority(self.path, _limits(requests=20))
        client = BudgetedChatClient(
            ChatClient("https://example.test/v1", "placeholder", "model"),
            authority,
            "cell-repeat-429",
        )
        for _ in range(8):
            with patch(
                "rangebench.agent.urllib.request.urlopen",
                side_effect=_http_error(429),
            ):
                _, _, error = client.chat([{"role": "user", "content": "x"}], 100)
            self.assertIn("HTTP 429", error or "")
        with (
            patch(
                "rangebench.agent.urllib.request.urlopen",
                side_effect=_http_error(429),
            ),
            self.assertRaisesRegex(BudgetError, "attempt cell-repeat-429"),
        ):
            client.chat([{"role": "user", "content": "x"}], 100)
        snapshot = authority.snapshot()
        self.assertFalse(snapshot.unknown)
        self.assertEqual(snapshot.requests, 8)
        self.assertEqual(snapshot.rejected_requests, 8)
        self.assertEqual(snapshot.tokens, 0)

    def test_repeated_real_413_is_bounded_by_attempt_cap(self) -> None:
        authority = BudgetAuthority(self.path, _limits(requests=20))
        client = BudgetedChatClient(
            ChatClient("https://example.test/v1", "placeholder", "model"),
            authority,
            "cell-repeat-413",
        )
        with patch(
            "rangebench.agent.urllib.request.urlopen",
            side_effect=_http_error(413),
        ) as urlopen:
            for _ in range(8):
                _, _, error = client.chat([{"role": "user", "content": "x"}], 100)
                self.assertIn("HTTP 413", error or "")
            with self.assertRaisesRegex(BudgetError, "attempt cell-repeat-413"):
                client.chat([{"role": "user", "content": "x"}], 100)
        self.assertEqual(urlopen.call_count, 8)
        snapshot = authority.snapshot()
        self.assertFalse(snapshot.unknown)
        self.assertEqual(snapshot.requests, 8)
        self.assertEqual(snapshot.rejected_requests, 8)
        self.assertEqual(snapshot.tokens, 0)

    def test_compaction_summary_calls_are_budgeted(self) -> None:
        authority = BudgetAuthority(self.path, _limits(tokens=10_000))
        inner = _CompleteClient(input_tokens=20, output_tokens=10)
        client = BudgetedChatClient(inner, authority, "cell")
        _summary_call(client, [{"role": "user", "content": "summarize"}], 50)
        snapshot = authority.snapshot()
        self.assertEqual(snapshot.requests, 1)
        self.assertEqual(snapshot.tokens, 30)
        self.assertEqual(inner.calls, 1)

    def test_per_attempt_request_cap(self) -> None:
        authority = BudgetAuthority(self.path, _limits(requests=20))
        for _ in range(8):
            request = authority.reserve_request("cell-a", 10)
            authority.reject_request(request, "test")
        with self.assertRaisesRegex(BudgetError, "attempt cell-a"):
            authority.reserve_request("cell-a", 10)

    def test_corrupt_ledger_lines_fail_closed(self) -> None:
        cases = {
            "not-json": "not-json\n",
            "non-object": "[]\n",
            "unknown-kind": '{"kind":"mystery","cell":"c"}\n',
            "fieldless-reserve": '{"kind":"reserve","id":"r1","cell":"c"}\n',
            "negative": (
                '{"kind":"reserve","id":"r1","cell":"c","input_reserve":-1,'
                '"output_reserve":1,"spend_reserve":0}\n'
            ),
            "missing-id": (
                '{"kind":"reserve","cell":"c","input_reserve":1,'
                '"output_reserve":1,"spend_reserve":0}\n'
            ),
            "duplicate-id": (
                '{"kind":"reserve","id":"r1","cell":"c","input_reserve":1,'
                '"output_reserve":1,"spend_reserve":0}\n'
                '{"kind":"reserve","id":"r1","cell":"c","input_reserve":1,'
                '"output_reserve":1,"spend_reserve":0}\n'
            ),
            "unmatched-terminal": (
                '{"kind":"settle","id":"r1","cell":"c","input":0,"output":0,"spend":0}\n'
            ),
            "over-settle": (
                '{"kind":"reserve","id":"r1","cell":"c","input_reserve":1,'
                '"output_reserve":1,"spend_reserve":0}\n'
                '{"kind":"settle","id":"r1","cell":"c","input":2,"output":0,"spend":0}\n'
            ),
            "overflow": (
                '{"kind":"reserve","id":"r1","cell":"c","input_reserve":1,'
                '"output_reserve":1,"spend_reserve":1e999999}\n'
            ),
        }
        for name, text in cases.items():
            with self.subTest(name=name):
                path = Path(self.temp.name) / f"{name}.jsonl"
                path.write_text(text)
                with self.assertRaisesRegex(BudgetError, "corrupt_ledger|line"):
                    BudgetAuthority(path, _limits())

    def test_non_utf8_ledger_fails_closed(self) -> None:
        path = Path(self.temp.name) / "binary.jsonl"
        path.write_bytes(b"\xff\xfe\n")
        with self.assertRaisesRegex(BudgetError, "UTF-8"):
            BudgetAuthority(path, _limits())

    def test_over_cap_replay_fails_closed(self) -> None:
        path = Path(self.temp.name) / "over.jsonl"
        path.write_text('{"kind":"attempt","cell":"a"}\n{"kind":"attempt","cell":"b"}\n')
        with self.assertRaisesRegex(BudgetError, "attempt total exceeds"):
            BudgetAuthority(path, _limits(attempts=1))

    def test_success_settles_actual_usage(self) -> None:
        authority = BudgetAuthority(self.path, _limits(tokens=10_000))
        inner = _CompleteClient(input_tokens=50, output_tokens=20)
        client = BudgetedChatClient(inner, authority, "cell")
        content, _, error = client.chat([{"role": "user", "content": "x"}], 100)
        self.assertEqual(content, "ok")
        self.assertIsNone(error)
        snapshot = authority.snapshot()
        self.assertEqual(snapshot.tokens, 70)
        self.assertEqual(snapshot.requests, 1)
        self.assertFalse(snapshot.unknown)


if __name__ == "__main__":
    unittest.main()
