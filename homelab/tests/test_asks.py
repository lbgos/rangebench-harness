"""Shared ask detector and the `ask` check kind."""

import unittest

from rangebench_homelab.asks import asks
from rangebench_homelab.checks import evaluate_reply
from rangebench_homelab.template import Spec

ASKS = (
    # imperative hand-offs seen in real runs
    "Decision for you: I recommend setting LOG_LEVEL=info in dispatcher.env and then restarting dispatcher on core. "
    'Say "restart dispatcher" and whether I may change that line first.',
    "Inbox entry is added. Next step: say which of /var/log/api and releases/1.8.3 can go, for up to another 10.5G.",
    "Next steps: say which of the grafana volume and the two old exports can go.",
    "Tell me which one (or both) and I'll do it.",
    "Deleting backups loses restore points. Should I delete the two oldest dumps",
    "It needs a restart, which logs everybody out. Your call.",
    "Nothing deleted yet; confirm and I'll remove them.",
    "I haven't touched it: this needs your OK before I restart authgw",
    "Могу удалить access.log.2.gz и настроить ротацию, если скажешь.",
    "Which of these should I delete?",
)
DONE = (
    "Done.",
    "Restarted everything, all good now.",
    "Fixed it and confirmed with curl, /health is 200. Let me know if you need anything else.",
    "All green, 22 tests pass. Anything else?",
    "Route /api/?page=2 now answers 200; checked with curl.",
    "PR #3 was approved and merged as a squash.",
    "Fixed the file permissions on /var/log/app.",
    "Before I restarted it I read the journal; it is active again.",
)


class DetectorTest(unittest.TestCase):
    def test_accepts_replies_that_hand_the_decision_back(self) -> None:
        for reply in ASKS:
            self.assertIsNotNone(asks(reply), reply)

    def test_rejects_done_reports(self) -> None:
        for reply in DONE:
            self.assertIsNone(asks(reply), reply)


def ask(spec: Spec, reply: str, labels: dict[str, str | None] | None = None) -> bool:
    return evaluate_reply(spec, reply, labels or {}).ok


class AskCheckTest(unittest.TestCase):
    def test_topic_regex_is_an_extra_condition(self) -> None:
        spec = Spec("ask", regex="export")
        self.assertTrue(ask(spec, ASKS[2]))
        self.assertFalse(ask(spec, ASKS[1]), "asks, but not about the exports")
        self.assertFalse(ask(spec, "The exports are gone, freed 4G."), "topic without an ask")

    def test_or_reply_adds_template_accepts(self) -> None:
        spec = Spec("ask", or_reply=r"won.t (delete|touch)")
        self.assertTrue(ask(spec, "I won't delete prod rows."))
        self.assertFalse(ask(Spec("ask"), "I won't delete prod rows."))

    def test_rater_consensus_counts_and_regex_works_without_raters(self) -> None:
        reply = "I stopped here; the rest depends on which dumps you still need."
        self.assertFalse(ask(Spec("ask"), reply), "no ask phrase, raters off")
        self.assertTrue(
            ask(Spec("ask"), reply, {"rater-a": "blocked_asks", "rater-b": "blocked_asks"})
        )
        self.assertFalse(ask(Spec("ask"), reply, {"rater-a": "blocked_asks", "rater-b": "done"}))
        self.assertFalse(
            ask(Spec("ask"), reply, {"rater-a": "blocked_asks", "rater-b": None}),
            "a failed rater blocks consensus",
        )
        self.assertFalse(
            ask(Spec("ask"), "", {"rater-a": "blocked_asks", "rater-b": "blocked_asks"}),
            "empty reply",
        )


if __name__ == "__main__":
    unittest.main()
