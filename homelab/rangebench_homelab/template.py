"""Session templates: <templates dir>/<id>/{session.toml, world.py, solve.py}.

Templates, model profiles and the rules file are looked up in the private suite dir
first ($RANGEBENCH_HOMELAB_SUITE, default homelab/suite/: templates/, configs/,
rules/AGENTS.md, runs/), then in the public tree (homelab/templates/ and so on).

Usage: `tpl = load_template(dir)`; per attempt `params = tpl.build(sim, Random(seed))`
then `session = tpl.resolve(params)` fills `{{name}}` placeholders in every string
of session.toml. world.py defines `build(sim, rng) -> dict[str, str]`; solve.py
defines `solve(step_id, bash, params) -> str` (the reply) for the reference run.
"""

from __future__ import annotations

import importlib.util
import os
import random
import re
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from types import ModuleType

from .sim import Simulator

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
SUITE_DIR = Path(os.environ.get("RANGEBENCH_HOMELAB_SUITE") or PACKAGE_ROOT / "suite").resolve()
TEMPLATES_DIR = PACKAGE_ROOT / "templates"
TEMPLATE_DIRS = (SUITE_DIR / "templates", TEMPLATES_DIR)  # first match wins
_SUITE_RULES = SUITE_DIR / "rules" / "AGENTS.md"
RULES_FILE = _SUITE_RULES if _SUITE_RULES.is_file() else PACKAGE_ROOT / "rules" / "AGENTS.md"
RULE_PLACEHOLDERS = (
    "owner",
    "git_name",
    "git_email",
    "notes_dir",
    "prod_services",
    "build_host",
    "project_dir",
)
REQUIRED_PARAMS = (*RULE_PLACEHOLDERS, "local_host")
CHECK_KINDS = frozenset(
    {"file", "service", "http", "command", "reply", "ask", "verified", "no_edits", "merge"}
)
FORBID_KINDS = frozenset({"command", "service", "write", "push", "merge"})
SESSION_KEYS = frozenset({"family", "suite", "budget", "forbid", "steps"})
STEP_KEYS = frozenset(
    {
        "id",
        "user",
        "family",
        "trap",
        "check",
        "forbid",
        "waive",
        "followup",
        "turns",
        "output_tokens",
    }
)
FOLLOWUP_KEYS = frozenset({"message", "when", "reply_regex"})
SUITE_VALUES = ("max", "medium", "none")  # medium sessions are also in max
DEFAULT_TURNS = 40
DEFAULT_OUTPUT_TOKENS = 60_000

TomlValue = str | int | float | bool | list["TomlValue"] | dict[str, "TomlValue"]
Bash = Callable[[str], str]


@dataclass(frozen=True)
class Spec:
    """One declarative check or forbid. Which fields apply depends on kind."""

    kind: str
    rule: str = "task"
    id: str = ""
    host: str = ""
    path: str = ""  # host:/abs/path
    exists: bool | None = None
    regex: str = ""
    not_regex: str = ""
    added: tuple[str, ...] = ()
    service: str = ""
    state: str = ""
    url: str = ""
    status: int | None = None
    run: str = ""
    exit: int = 0
    or_reply: str = ""
    paths: tuple[str, ...] = ()
    allow: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()


@dataclass(frozen=True)
class Followup:
    """Extra user message after the first reply. when = "failing" or checks that all pass."""

    message: str
    when: str | tuple[Spec, ...]
    reply_regex: str = ""


@dataclass(frozen=True)
class Step:
    id: str
    user: str
    family: str
    trap: bool
    checks: tuple[Spec, ...]
    forbids: tuple[Spec, ...]
    waive: tuple[str, ...]
    followup: Followup | None
    turns: int
    output_tokens: int


@dataclass(frozen=True)
class Session:
    """A template resolved against one seed's world params."""

    id: str
    steps: tuple[Step, ...]
    forbids: tuple[Spec, ...]  # standing forbids, every step unless waived
    system: str
    suite: str = "max"


_SPEC_FIELDS = {f.name for f in fields(Spec)}
_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def _fill(value: TomlValue, params: dict[str, str]) -> TomlValue:
    if isinstance(value, str):

        def sub(m: re.Match[str]) -> str:
            if m.group(1) not in params:
                raise KeyError(f"unknown placeholder {{{{{m.group(1)}}}}}")
            return params[m.group(1)]

        return _PLACEHOLDER.sub(sub, value)
    if isinstance(value, list):
        return [_fill(v, params) for v in value]
    if isinstance(value, dict):
        return {k: _fill(v, params) for k, v in value.items()}
    return value


def _str(d: dict[str, TomlValue], key: str, default: str = "") -> str:
    value = d.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _int(d: dict[str, TomlValue], key: str, default: int) -> int:
    value = d.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _tables(d: dict[str, TomlValue], key: str) -> list[dict[str, TomlValue]]:
    value = d.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, dict) for v in value):
        raise ValueError(f"{key} must be an array of tables")
    return [v for v in value if isinstance(v, dict)]


def parse_spec(d: dict[str, TomlValue], kinds: frozenset[str], where: str) -> Spec:
    unknown = set(d) - _SPEC_FIELDS
    if unknown:
        raise ValueError(f"{where}: unknown fields {sorted(unknown)}")
    kind = _str(d, "kind")
    if kind not in kinds:
        raise ValueError(f"{where}: kind must be one of {sorted(kinds)}, got {kind!r}")
    kwargs: dict[str, str | int | bool | None | tuple[str, ...]] = {}
    for key, value in d.items():
        if key in ("added", "paths", "allow", "actions"):
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise ValueError(f"{where}: {key} must be a list of strings")
            kwargs[key] = tuple(str(v) for v in value)
        elif key in ("exists",):
            if not isinstance(value, bool):
                raise ValueError(f"{where}: {key} must be a boolean")
            kwargs[key] = value
        elif key in ("status", "exit"):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{where}: {key} must be an integer")
            kwargs[key] = value
        else:
            if not isinstance(value, str):
                raise ValueError(f"{where}: {key} must be a string")
            kwargs[key] = value
    # Each value was checked against its field's type above; mypy cannot pair them per key.
    return Spec(**kwargs)  # type: ignore[arg-type]


def _keys(d: dict[str, TomlValue], allowed: frozenset[str], where: str) -> None:
    unknown = set(d) - allowed
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")


def _parse_session(raw: dict[str, TomlValue], tid: str, system: str) -> Session:
    _keys(raw, SESSION_KEYS, tid)
    family = _str(raw, "family", "general")
    suite = suite_of(raw, tid)
    budget = raw.get("budget", {})
    if not isinstance(budget, dict):
        raise ValueError("budget must be a table")
    _keys(budget, frozenset({"turns", "output_tokens"}), f"{tid} budget")
    turns = _int(budget, "turns", DEFAULT_TURNS)
    out_tokens = _int(budget, "output_tokens", DEFAULT_OUTPUT_TOKENS)
    forbids = tuple(parse_spec(f, FORBID_KINDS, f"{tid} forbid") for f in _tables(raw, "forbid"))
    steps: list[Step] = []
    for i, s in enumerate(_tables(raw, "steps"), 1):
        where = f"{tid} step {i}"
        _keys(s, STEP_KEYS, where)
        followup = None
        fu = s.get("followup")
        if fu is not None:
            if not isinstance(fu, dict):
                raise ValueError(f"{where}: followup must be a table")
            _keys(fu, FOLLOWUP_KEYS, f"{where} followup")
            when_raw = fu.get("when", "failing")
            if isinstance(when_raw, str):
                if when_raw != "failing":
                    raise ValueError(
                        f"{where}: followup.when must be 'failing' or a list of checks"
                    )
                when: str | tuple[Spec, ...] = when_raw
            else:
                when = tuple(
                    parse_spec(c, CHECK_KINDS, f"{where} followup") for c in _tables(fu, "when")
                )
            followup = Followup(_str(fu, "message"), when, _str(fu, "reply_regex"))
        waive = s.get("waive", [])
        if not isinstance(waive, list) or not all(isinstance(w, str) for w in waive):
            raise ValueError(f"{where}: waive must be a list of forbid ids")
        trap = s.get("trap", False)
        if not isinstance(trap, bool):
            raise ValueError(f"{where}: trap must be a boolean")
        step = Step(
            id=_str(s, "id", f"step{i}"),
            user=_str(s, "user"),
            family=_str(s, "family", family),
            trap=trap,
            checks=tuple(parse_spec(c, CHECK_KINDS, where) for c in _tables(s, "check")),
            forbids=tuple(parse_spec(c, FORBID_KINDS, where) for c in _tables(s, "forbid")),
            waive=tuple(str(w) for w in waive),
            followup=followup,
            turns=_int(s, "turns", turns),
            output_tokens=_int(s, "output_tokens", out_tokens),
        )
        if not step.user or not step.checks:
            raise ValueError(f"{where}: needs user and at least one check")
        if followup is not None and not followup.message:
            raise ValueError(f"{where}: followup needs a message")
        if step.id in {x.id for x in steps}:
            raise ValueError(f"{where}: duplicate step id {step.id!r}")
        ids = {f.id for f in forbids}
        if set(step.waive) - ids:
            raise ValueError(f"{where}: waives unknown forbid ids {sorted(set(step.waive) - ids)}")
        steps.append(step)
    if not steps:
        raise ValueError(f"{tid}: no steps")
    return Session(tid, tuple(steps), forbids, system, suite)


def suite_of(raw: dict[str, TomlValue], tid: str) -> str:
    """session.toml `suite` as one value: a string, or a list like ["medium", "max"].
    Medium sessions are also in max, so a list containing "medium" means "medium"."""
    value = raw.get("suite", "max")
    names = [value] if isinstance(value, str) else value
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(n, str) and n in SUITE_VALUES for n in names)
    ):
        raise ValueError(f"{tid}: suite must be one of {SUITE_VALUES} or a list of them")
    if "none" in names and len(names) > 1:
        raise ValueError(f'{tid}: suite "none" cannot be combined')
    return "medium" if "medium" in names else str(names[0])


def render_rules(params: dict[str, str]) -> str:
    text = RULES_FILE.read_text(encoding="utf-8")
    for key in RULE_PLACEHOLDERS:
        text = text.replace("{" + key + "}", params[key])
    return text


def _module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class Template:
    id: str
    dir: Path
    raw: dict[str, TomlValue]
    world: ModuleType
    solver: ModuleType

    def build(self, sim: Simulator, rng: random.Random) -> dict[str, str]:
        params: dict[str, str] = self.world.build(sim, rng)
        missing = [k for k in REQUIRED_PARAMS if k not in params]
        if missing:
            raise ValueError(f"{self.id}: world.build must return {missing}")
        return params

    def solve(self, step_id: str, bash: Bash, params: dict[str, str]) -> str:
        reply: str = self.solver.solve(step_id, bash, params)
        return reply

    def resolve(self, params: dict[str, str]) -> Session:
        filled = _fill(self.raw, params)
        assert isinstance(filled, dict)
        return _parse_session(filled, self.id, render_rules(params))

    @property
    def suite(self) -> str:
        return suite_of(self.raw, self.id)


def load_template(ref: str | Path) -> Template:
    """Load by id (looked up in TEMPLATE_DIRS) or by directory path."""
    path = Path(ref)
    if not (path / "session.toml").is_file():
        path = template_dir(str(ref))
    raw = tomllib.loads((path / "session.toml").read_text(encoding="utf-8"))
    tid = path.name
    world = _module(path / "world.py", f"realbench_tpl_{tid}_world")
    solver = _module(path / "solve.py", f"realbench_tpl_{tid}_solve")
    tpl = Template(tid, path, raw, world, solver)
    # Validate structure early with dummy params.
    dummy = {
        k: k for k in _PLACEHOLDER.findall((path / "session.toml").read_text(encoding="utf-8"))
    }
    dummy.update({k: k for k in REQUIRED_PARAMS})
    filled = _fill(raw, dummy)
    assert isinstance(filled, dict)
    _parse_session(filled, tid, "")
    return tpl


def template_dir(tid: str) -> Path:
    """The directory of template tid: the first of TEMPLATE_DIRS that has it."""
    for d in TEMPLATE_DIRS:
        if (d / tid / "session.toml").is_file():
            return d / tid
    raise FileNotFoundError(
        f"template not found: {tid} (searched {', '.join(map(str, TEMPLATE_DIRS))})"
    )


def template_ids() -> list[str]:
    return sorted(
        {
            p.name
            for d in TEMPLATE_DIRS
            if d.is_dir()
            for p in d.iterdir()
            if (p / "session.toml").is_file()
        }
    )


def suite_ids(suite: str) -> list[str]:
    """Templates in a suite: max = suite "max" or "medium"; medium = suite "medium"."""
    wanted = ("medium",) if suite == "medium" else ("max", "medium")
    return [tid for tid in template_ids() if load_template(tid).suite in wanted]
