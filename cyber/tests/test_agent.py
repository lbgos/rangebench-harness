import io
import json
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

from rangebench.agent import (
    AnthropicChatClient,
    ChatClient,
    ChatResult,
    ResponseMetadata,
    _CredentialSafeRedirectHandler,
    parse_turn,
)


class FakeResponse:
    def __init__(self, body: dict):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return json.dumps(self.body).encode()


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://example.test/v1",
        code,
        "provider",
        {},
        io.BytesIO(b'{"error":"provider"}'),
    )


class ResponseMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.messages = [{"role": "user", "content": "test"}]

    def test_openai_reasoning_only_reply_keeps_visible_content_empty(self) -> None:
        body = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": None, "reasoning_content": "private reasoning"},
                }
            ],
            "usage": {"prompt_tokens": 13, "completion_tokens": 64},
        }
        client = ChatClient("http://localhost/v1", "secret", "model")
        with patch("rangebench.agent.urllib.request.urlopen", return_value=FakeResponse(body)):
            result = client.chat_result(self.messages, max_tokens=64)

        self.assertIsInstance(result, ChatResult)
        self.assertEqual(result.content, "")
        self.assertIsNone(result.error)
        self.assertEqual(result.usage.completion_tokens, 64)
        self.assertEqual(result.metadata, ResponseMetadata("length", True, True, 64))
        self.assertNotIn("private reasoning", repr(result))
        self.assertNotIn("secret", repr(result))

    def test_reasoning_effort_is_sent_only_when_configured(self) -> None:
        body = {
            "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }
        payloads: list[dict] = []

        def capture(req: urllib.request.Request, timeout: int | None = None) -> FakeResponse:
            payloads.append(json.loads(req.data.decode()))
            return FakeResponse(body)

        with patch("rangebench.agent.urllib.request.urlopen", side_effect=capture):
            ChatClient("http://localhost/v1", "k", "m").chat(self.messages, 64)
            ChatClient("http://localhost/v1", "k", "m", reasoning_effort="max").chat(
                self.messages, 64
            )

        self.assertNotIn("reasoning_effort", payloads[0])
        self.assertEqual(payloads[1]["reasoning_effort"], "max")

    def test_openai_visible_reply_and_existing_tuple_interface(self) -> None:
        body = {
            "choices": [
                {"finish_reason": "stop", "message": {"content": "I should inspect the host."}}
            ],
            "usage": {"prompt_tokens": 13, "completion_tokens": 9},
        }
        client = ChatClient("http://localhost/v1", "secret", "model")
        with patch("rangebench.agent.urllib.request.urlopen", return_value=FakeResponse(body)):
            result = client.chat_result(self.messages, max_tokens=120)
        self.assertEqual(result.metadata, ResponseMetadata("stop", False, False, 120))
        with patch("rangebench.agent.urllib.request.urlopen", return_value=FakeResponse(body)):
            legacy = client.chat(self.messages, max_tokens=120)
        self.assertIsInstance(legacy, tuple)
        self.assertEqual(len(legacy), 3)
        self.assertEqual(legacy, (result.content, result.usage, result.error))

    def test_untrusted_finish_reason_is_bounded(self) -> None:
        body = {
            "choices": [
                {
                    "finish_reason": "secret provider value",
                    "message": {"content": [{"type": "reasoning", "text": "hidden"}]},
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        client = ChatClient("http://localhost/v1", "key", "model")
        with patch("rangebench.agent.urllib.request.urlopen", return_value=FakeResponse(body)):
            result = client.chat_result(self.messages, max_tokens=4)
        self.assertEqual(result.metadata.finish_reason, "other")
        self.assertTrue(result.metadata.visible_content_empty)
        self.assertTrue(result.metadata.reasoning_content_present)
        self.assertNotIn("secret provider value", repr(result))

    def test_anthropic_thinking_only_and_tuple_interface(self) -> None:
        body = {
            "stop_reason": "max_tokens",
            "content": [{"type": "thinking", "thinking": "private reasoning"}],
            "usage": {"input_tokens": 3, "output_tokens": 12},
        }
        client = AnthropicChatClient("http://localhost", "secret", "model")
        with patch("rangebench.agent.urllib.request.urlopen", return_value=FakeResponse(body)):
            result = client.chat_result(self.messages, max_tokens=12)
        self.assertEqual(result.metadata, ResponseMetadata("max_tokens", True, True, 12))
        with patch("rangebench.agent.urllib.request.urlopen", return_value=FakeResponse(body)):
            legacy = client.chat(self.messages, max_tokens=12)
        self.assertEqual(legacy, (result.content, result.usage, result.error))

    def test_missing_usage_does_not_imply_empty_provider_reply(self) -> None:
        body = {"choices": [{"message": {"content": "ANSWER: value"}}]}
        client = ChatClient("http://localhost/v1", "key", "model")
        with patch("rangebench.agent.urllib.request.urlopen", return_value=FakeResponse(body)):
            result = client.chat_result(self.messages, max_tokens=8)
        self.assertEqual(result.error, "missing input or output token usage")
        self.assertEqual(result.metadata, ResponseMetadata(None, None, None, 8))

    def test_redirect_handler_blocks_key_leak_vectors(self) -> None:
        handler = _CredentialSafeRedirectHandler()
        origin = urllib.request.Request("https://api.example.test/v1", method="POST")
        for target in (
            "https://evil.test/v1",
            "http://api.example.test/v1",
            "https://api.example.test:8443/v1",
        ):
            with self.subTest(target=target), self.assertRaises(urllib.error.URLError):
                handler.redirect_request(origin, None, 302, "Found", {}, target)
        allowed = handler.redirect_request(
            origin, None, 302, "Found", {}, "https://api.example.test/other"
        )
        self.assertIsInstance(allowed, urllib.request.Request)
        self.assertEqual(allowed and allowed.full_url, "https://api.example.test/other")


class HTTPErrorClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.messages = [{"role": "user", "content": "test"}]

    def _clients(self) -> tuple[tuple[str, ChatClient], tuple[str, AnthropicChatClient]]:
        return (
            ("openai", ChatClient("https://example.test/v1", "placeholder", "model")),
            ("anthropic", AnthropicChatClient("https://example.test", "placeholder", "model")),
        )

    def test_429_is_a_definitive_rejection_for_both_handlers(self) -> None:
        for name, client in self._clients():
            with (
                self.subTest(name),
                patch(
                    "rangebench.agent.urllib.request.urlopen",
                    side_effect=_http_error(429),
                ),
            ):
                result = client.chat_result(self.messages, max_tokens=16)
            self.assertTrue((result.error or "").startswith("HTTP 429:"), result.error)
            self.assertFalse((result.error or "").startswith("transport:"))
            self.assertEqual(result.usage.requests, 1)
            self.assertEqual(result.usage.calls, 0)
            self.assertEqual(result.usage.input_tokens + result.usage.output_tokens, 0)

    def test_413_is_a_definitive_rejection_for_both_handlers(self) -> None:
        for name, client in self._clients():
            with (
                self.subTest(name),
                patch(
                    "rangebench.agent.urllib.request.urlopen",
                    side_effect=_http_error(413),
                ),
            ):
                result = client.chat_result(self.messages, max_tokens=16)
            self.assertTrue((result.error or "").startswith("HTTP 413:"), result.error)
            self.assertFalse((result.error or "").startswith("transport:"))
            self.assertEqual(result.usage.requests, 1)
            self.assertEqual(result.usage.calls, 0)
            self.assertEqual(result.usage.input_tokens + result.usage.output_tokens, 0)

    def test_503_remains_transport_for_both_handlers(self) -> None:
        for name, client in self._clients():
            with (
                self.subTest(name),
                patch(
                    "rangebench.agent.urllib.request.urlopen",
                    side_effect=_http_error(503),
                ),
            ):
                result = client.chat_result(self.messages, max_tokens=16)
            self.assertTrue((result.error or "").startswith("transport: HTTP 503:"), result.error)


class ParseTurnTests(unittest.TestCase):
    def test_command_block_with_duplicate_dsml_keeps_one_clean_command(self) -> None:
        text = (
            "\n\nCOMMAND:\nls -la /work/quote/ && file /work/quote/*\n\n"
            "<\uff5cDSML\uff5c calls>\n"
            '<\uff5cDSML\uff5c invoke name="bash">\n'
            '<\uff5cDSML\uff5c parameter name="command" string="true">'
            "ls -la /work/quote/ && file /work/quote/*"
            "</\uff5cDSML\uff5c parameter>\n"
            "</\uff5cDSML\uff5c invoke>\n"
            "</\uff5cDSML\uff5c calls>"
        )
        commands, answers = parse_turn(text)
        self.assertEqual(commands, ["ls -la /work/quote/ && file /work/quote/*"])
        self.assertEqual(answers, [])

    def test_dsml_only_turn_is_a_command(self) -> None:
        text = (
            "<\uff5cDSML\uff5c calls>\n"
            '<\uff5cDSML\uff5c invoke name="bash">\n'
            '<\uff5cDSML\uff5c parameter name="command" string="true">'
            "strings -n 8 /work/quote/quote | grep -i seccomp"
            "</\uff5cDSML\uff5c parameter>\n"
            "</\uff5cDSML\uff5c invoke>\n"
            "</\uff5cDSML\uff5c calls>"
        )
        commands, _ = parse_turn(text)
        self.assertEqual(commands, ["strings -n 8 /work/quote/quote | grep -i seccomp"])

    def test_privacy_event_is_not_part_of_the_command(self) -> None:
        text = (
            "COMMAND:\ncurl -s http://portal:8080/\n\n"
            "<privacy_event>I cannot see the output.</privacy_event>"
        )
        commands, _ = parse_turn(text)
        self.assertEqual(commands, ["curl -s http://portal:8080/"])

    def test_answer_is_still_parsed(self) -> None:
        commands, answers = parse_turn("ANSWER: flag{abc}")
        self.assertEqual(commands, [])
        self.assertEqual(answers, ["flag{abc}"])


class CommandCleanupTests(unittest.TestCase):
    def test_trailing_prose_is_not_part_of_the_command(self) -> None:
        commands, _ = parse_turn("COMMAND:\nls -la /work\nLet me inspect the binary.")
        self.assertEqual(commands, ["ls -la /work"])

    def test_fabricated_observation_is_cut(self) -> None:
        commands, _ = parse_turn("COMMAND:\ncat /flag\nOBSERVATION:\nflag{deadbeef}")
        self.assertEqual(commands, ["cat /flag"])

    def test_think_tag_is_removed(self) -> None:
        commands, _ = parse_turn("COMMAND:\nls /work</think>")
        self.assertEqual(commands, ["ls /work"])


if __name__ == "__main__":
    unittest.main()
