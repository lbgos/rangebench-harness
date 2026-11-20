"""Model profiles: <suite>/configs/<name>.toml, else homelab/configs/<name>.toml.

Usage: `profile = load_profile("example-api")`; `profile.client()` builds a ChatClient
with the key read from the env var named by api_key_env (never stored). Fields:
display_name, base_url, api_key_env, model, ctx_window, concurrency,
wall_clock_scale, and an optional [extra] table merged into every request
(e.g. reasoning_effort).
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .agent import ChatClient, JsonValue
from .template import PACKAGE_ROOT, SUITE_DIR

CONFIG_DIRS = (SUITE_DIR / "configs", PACKAGE_ROOT / "configs")  # first match wins
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")
_FIELDS = {
    "display_name",
    "base_url",
    "api_key_env",
    "model",
    "ctx_window",
    "concurrency",
    "wall_clock_scale",
    "extra",
}


@dataclass(frozen=True)
class Profile:
    name: str
    display_name: str
    base_url: str
    api_key_env: str
    model: str
    ctx_window: int
    concurrency: int = 1
    wall_clock_scale: float = 1.0
    extra: dict[str, JsonValue] = field(default_factory=dict)

    def client(self, timeout: float = 600) -> ChatClient:
        key = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
        return ChatClient(self.base_url, key, self.model, self.extra, timeout=timeout)

    def public(self) -> dict[str, JsonValue]:
        """Profile fields safe to record in a run file."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "model": self.model,
            "base_url": self.base_url,
            "ctx_window": self.ctx_window,
            "concurrency": self.concurrency,
            "wall_clock_scale": self.wall_clock_scale,
            "extra": dict(self.extra),
        }


def load_profile(name: str) -> Profile:
    path = (
        next((p for d in CONFIG_DIRS if (p := d / f"{name}.toml").is_file()), None)
        if Path(name).name == name
        else None
    )
    if path is None:
        raise ValueError(f"profile not found: {name}")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    if "api_key" in data:
        raise ValueError(f"{name}: use api_key_env, never a literal key")
    unknown = set(data) - _FIELDS
    if unknown:
        raise ValueError(f"{name}: unknown fields {sorted(unknown)}")
    for key in ("display_name", "base_url", "model"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ValueError(f"{name}: {key} must be a nonempty string")
    env = data.get("api_key_env", "")
    if not isinstance(env, str) or (env and not _ENV_NAME.fullmatch(env)):
        raise ValueError(f"{name}: api_key_env must be an environment variable name")
    for key in ("ctx_window", "concurrency"):
        value = data.get(key, 1)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name}: {key} must be a positive integer")
    if "ctx_window" not in data:
        raise ValueError(f"{name}: missing ctx_window")
    scale = data.get("wall_clock_scale", 1.0)
    if not isinstance(scale, int | float) or scale <= 0:
        raise ValueError(f"{name}: wall_clock_scale must be positive")
    extra = data.get("extra", {})
    if not isinstance(extra, dict):
        raise ValueError(f"{name}: extra must be a table")
    return Profile(
        name,
        data["display_name"],
        data["base_url"],
        env,
        data["model"],
        data["ctx_window"],
        data.get("concurrency", 1),
        float(scale),
        extra,
    )
