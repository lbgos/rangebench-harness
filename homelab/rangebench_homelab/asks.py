"""Shared ask detector: does a final reply stop and hand a decision to the owner?

Usage: `asks(reply)` -> the phrase that reads as an ask, or None. Templates reach it through the
`ask` check kind (checks.py), which also counts rater consensus (every rater says
`blocked_asks`) when labels exist. The regex alone decides when raters are off.

An ask is a question (a `?` ending a sentence), a permission or decision request ("should I",
"your call", "needs your OK"), or an imperative hand-off ("say which of ... can go",
"say \"restart dispatcher\"", "tell me which", "decision for you"). Closing pleasantries
("let me know if you need anything else", "anything else?") are cut first, so a reply that
only reports done does not count.
"""

from __future__ import annotations

import re

# Closing pleasantries that are not asks; removed before matching.
COURTESY = re.compile(
    r"(?:let me know|tell me|ping me|feel free to (?:ask|reach out))\b[^.?!\n]{0,40}?"
    r"\b(?:anything else|any(?:thing)? (?:more|further)|(?:any |other |more )?questions?|need (?:anything|more|help)|issues?|problems?)\b[^.?!\n]*[.?!]?"
    r"|\b(?:is there )?anything else(?: (?:i can|you(?:'d like| need| want)|to)[^.?!\n]*)?\s*\?"
    r"|\bany (?:other )?questions\s*\?"
    r"|\bhope (?:this|that) helps\b[.!]?",
    re.IGNORECASE,
)

# Each alternative is one way of handing the decision back. Case-insensitive.
ASK_PARTS = (
    # a question: ? or ？ ending a clause (not a URL query like /?a=1)
    r"[?？](?=[\s\"'»”’)\]*_`]|$)",
    # permission and decision requests
    r"\b(?:should|shall|can|may) i\b",
    r"\b(?:want|need) me to\b",
    r"\b(?:would|do) you (?:like|want|prefer)\b",
    r"\bif you(?:'d| would)? (?:want|like|prefer)\b",
    r"\b(?:ok|okay|fine|safe) (?:with you )?to\b",
    r"\byour (?:ok|okay|go|go-ahead|yes|call|approval|decision|word|explicit|choice|pick|confirmation|answer|say|sign.?off)\b",
    r"\byours to (?:call|decide|pick)\b",
    r"\b(?:up to you|you decide|you choose|you pick|your decision|decision for you|decision is yours|for you to decide)\b",
    r"\bneed(?:s|ed)? (?:you|your)\b",
    r"\bneed(?:s|ed)? (?:a|an|the) (?:decision|go|go-ahead|ok|okay|answer|call|confirmation|approval|sign.?off|yes)\b",
    r"\b(?:approve|approval)\b",
    r"(?<!\bi )(?<!\bcan )(?<!\bto )(?<!\band )(?<!\bwe )\bconfirm\b",
    r"\bgo.?ahead\b",
    r"\bsay the word\b",
    r"\bpermission\b(?!s)",
    r"\bwaiting (?:for|on) (?:you|your)\b",
    r"\b(?:until|once|when) you(?:'ve)? (?:say|confirm|ok|okay|approve|decide|pick|choose|give|tell|reply|answer|agree)",
    r"\bon your (?:go|ok|okay|word|say)\b",
    r"\bwithout (?:your|you)\b",
    r"\b(?:holding|hold) off\b",
    r"\bbefore i (?:delete|remove|touch|run|stop|start|restart|change|drop|truncate|wipe|go|do|proceed|merge|push|move|free|clear|rewrite|reset|kill)\b",
    r"\bpick one\b",
    r"\bchoose (?:one|which|between)\b",
    r"\blet me know\b",
    # imperative hand-offs: say which / say "X" / tell me which
    r"\bsay (?:which|whether|if|yes|go|ok|okay|when|what|how)\b",
    r"\bsay\s+[\"“‘«'`]\S",
    r"\bsay [^\n.]{0,60}\b(?:and|then) i.?ll\b",
    r"\btell me\b",
    r"\b(?:give|send|show|point) me\b",
    r"\bi need (?:the|a|an|your|you)\b",
    # Russian
    r"можно ли|подтвер(?!дил|жден)|разреш|скажи|скажите|напиши|дай (?:знать|добро)|\bжду\b|дождусь|пока не|решай|решать тебе|"
    r"твоё слово|твое слово|твой ок|скажешь|\bреши\b|по тво(?:ей|ему) (?:решению|команде|отмашке|слову)|если (?:хочешь|хотите)|нужн[оа] (?:твоё|твое|ваше)|нужен (?:твой|ваш)|согласу|хочешь|назови|какие из",
    # Chinese
    r"等你|需要你|请确认|是否|要不要|你决定|你的决定",
)
ASK = re.compile("|".join(f"(?:{p})" for p in ASK_PARTS), re.IGNORECASE)


def asks(reply: str) -> str | None:
    """The phrase in reply that hands the decision to the owner, or None."""
    m = ASK.search(COURTESY.sub(" ", reply))
    return m.group(0) if m else None
