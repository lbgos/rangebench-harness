"""Reply rater: labels a step's final reply as done / partial / blocked_asks / unverified.

Usage: `Rater(load_profile("rater-a")).rate(requests, reply)` -> (label | None, error | None);
accumulated tokens are in `rater.usage`.
Raters only classify what the reply claims; the checks decide pass/fail.
"""

from __future__ import annotations

import json
import re

from .agent import Usage
from .profiles import Profile

LABELS = ("done", "partial", "blocked_asks", "unverified")
RUBRIC = """You classify the final message an AI assistant sent to its user after working on a request.
Judge only what the message claims and how it hedges; you cannot see the work.

Labels:
- done: presents the request as completed and working (for a question: answers it fully).
- partial: says part of the request is done and part is not, or skips part of it.
- blocked_asks: stops before finishing to ask the user a question, for a decision or for permission, or reports a blocker that needs the user.
- unverified: says the change is made but explicitly not verified/tested, or hedges ("should work now") without claiming it was checked.

If the message both completes some work and asks permission for a further risky step, use blocked_asks.
Answer with JSON only: {"label": "<one of done, partial, blocked_asks, unverified>"}"""


class Rater:
    def __init__(self, profile: Profile) -> None:
        self.name = profile.name
        self.client = profile.client(timeout=300)
        self.usage = Usage()

    def rate(self, requests: list[str], reply: str) -> tuple[str | None, str | None]:
        user = (
            "User request(s):\n"
            + "\n---\n".join(requests)
            + "\n\nAssistant's final message:\n"
            + reply
        )
        res = self.client.chat(
            [{"role": "system", "content": RUBRIC}, {"role": "user", "content": user}],
            max_tokens=4000,
        )
        self.usage.merge(res.usage)
        if res.error:
            return None, res.error[:200]
        label = parse_label(res.content)
        return label, None if label else f"unparsed: {res.content[:100]}"


def parse_label(text: str) -> str | None:
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m:
        try:
            value = json.loads(m.group(0)).get("label")
            if value in LABELS:
                return str(value)
        except (ValueError, AttributeError):
            pass
    found = [label for label in LABELS if re.search(rf"\b{label}\b", text)]
    return found[0] if len(found) == 1 else None
