"""Compaction triggers near the context window and keeps the conversation usable."""

import unittest

from rangebench_homelab import compaction
from rangebench_homelab.agent import Message, Reply, ToolCall, Usage
from rangebench_homelab.session import Budget, LLMAgent
from rangebench_homelab.template import Step


class FakeClient:
    """Answers summary requests with a note and agent requests with a reply tool call."""

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    def chat(
        self, messages: list[Message], tools: list[Message] | None = None, max_tokens: int = 4096
    ) -> Reply:
        self.calls.append(list(messages))
        usage = Usage(
            prompt_tokens=compaction.estimate_tokens(messages), completion_tokens=10, calls=1
        )
        if tools is None:
            return Reply("memory: api on app fixed", usage=usage)
        return Reply("", [ToolCall("c1", "reply", '{"text": "ok"}')], usage=usage)


def history(turns: int, size: int) -> list[Message]:
    msgs: list[Message] = [{"role": "system", "content": "rules"}]
    for i in range(turns):
        msgs.append({"role": "user", "content": f"request {i}"})
        msgs.append({"role": "assistant", "content": "x" * size})
    return msgs


STEP = Step("s", "go", "f", False, (), (), (), None, 10, 10_000)


class CompactionTest(unittest.TestCase):
    def test_should_compact_threshold(self) -> None:
        self.assertFalse(compaction.should_compact(history(10, 1000), 32_000))
        self.assertTrue(compaction.should_compact(history(10, 12_000), 32_000))
        # Calibration: the provider counted twice as many tokens as the estimate.
        msgs = history(10, 6_000)
        est = compaction.estimate_tokens(msgs)
        self.assertFalse(compaction.should_compact(msgs, 32_000))
        self.assertTrue(compaction.should_compact(msgs, 32_000, 2 * est, est))

    def test_compact_keeps_system_tail_and_request(self) -> None:
        msgs = history(10, 100)
        new, used_llm = compaction.compact(msgs, lambda prompt: "note", keep_tail=2)
        self.assertTrue(used_llm)
        self.assertEqual(new[0], msgs[0])
        self.assertTrue(str(new[1]["content"]).startswith(compaction.MEMORY_TAG))
        self.assertEqual(new[2:], msgs[-3:])  # tail starts at the 2nd-last assistant turn

    def test_compact_falls_back_without_summary(self) -> None:
        new, used_llm = compaction.compact(history(10, 100), lambda prompt: None, keep_tail=2)
        self.assertFalse(used_llm)
        self.assertIn("summary unavailable", str(new[1]["content"]))

    def test_agent_compacts_before_call_near_window(self) -> None:
        client = FakeClient()
        agent = LLMAgent(client, "rules", [], ctx_window=32_000)  # type: ignore[arg-type]
        agent.messages = history(10, 12_000)
        turn = agent.respond(STEP, "next", None, Budget(5, 10_000, float("inf")))  # type: ignore[arg-type]
        self.assertEqual(turn.reply, "ok")
        self.assertEqual(agent.compactions, 1)
        self.assertLess(compaction.estimate_tokens(agent.messages), 20_000)
        self.assertIn("next", str(agent.messages))


if __name__ == "__main__":
    unittest.main()
