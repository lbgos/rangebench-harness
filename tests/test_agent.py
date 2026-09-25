import json
import unittest
from unittest.mock import patch

from rangebench.agent import AnthropicChatClient, ChatClient, ChatResult, ResponseMetadata


class FakeResponse:
    def __init__(self, body: dict):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return json.dumps(self.body).encode()


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


if __name__ == "__main__":
    unittest.main()
