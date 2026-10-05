"""Load competitor profiles and launch independent run units."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .agent import AnthropicChatClient, ChatClient, ChatClientProtocol
from .runner import MAX_CTX_WINDOW

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")


def _is_loopback(host: str) -> bool:
    """Loopback hosts where cleartext HTTP never leaves the machine."""
    return host == "localhost" or host == "::1" or host.startswith("127.")


@dataclass(frozen=True)
class ModelConfig:
    name: str
    display_name: str
    base_url: str
    api_key_env: str
    model: str
    ctx_window: int
    trials: int
    provider: str = "openai"
    route_name: str | None = None
    upstream_provider: str | None = None
    reasoning_effort: str | None = None
    wall_clock_reference: str | None = None
    notes: str = ""


def load_config(path: Path) -> ModelConfig:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"{path.name}: malformed TOML") from exc
    allowed = set(ModelConfig.__dataclass_fields__) - {"name"}
    if "api_key" in data:
        raise ValueError(f"{path.name}: use api_key_env, never a literal key")
    if set(data) - allowed:
        raise ValueError(f"{path.name}: unknown config fields: {sorted(set(data) - allowed)}")
    required = {"display_name", "base_url", "api_key_env", "model", "ctx_window", "trials"}
    for key in sorted(required - set(data)):
        raise ValueError(f"{path.name}: missing {key}")
    for key in required | (set(data) - required):
        value = data.get(key)
        if key in {"ctx_window", "trials"}:
            if type(value) is not int or value < 1:
                raise ValueError(f"{path.name}: {key} must be a positive integer")
        elif value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{path.name}: {key} must be a nonempty string")
    if data["ctx_window"] > MAX_CTX_WINDOW:
        raise ValueError(f"{path.name}: ctx_window exceeds {MAX_CTX_WINDOW}")
    try:
        url = urlsplit(data["base_url"])
    except ValueError as exc:
        raise ValueError(f"{path.name}: invalid base_url") from exc
    if (
        url.scheme not in {"http", "https"}
        or not url.netloc
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError(f"{path.name}: base_url must be an HTTP(S) URL without credentials")
    # API keys travel in request headers; remote HTTP would expose them on the
    # wire, so only loopback endpoints may use cleartext HTTP.
    if url.scheme == "http" and not _is_loopback(url.hostname or ""):
        raise ValueError(f"{path.name}: remote base_url must use https")
    if not _ENV_NAME.fullmatch(data["api_key_env"]):
        raise ValueError(f"{path.name}: api_key_env must be an environment variable name")
    if data.get("provider", "openai") not in {"openai", "anthropic"}:
        raise ValueError(f"{path.name}: unsupported provider")
    if data.get("provider") == "anthropic" and data.get("reasoning_effort"):
        raise ValueError(f"{path.name}: reasoning_effort is OpenAI-only")
    return ModelConfig(name=path.stem, **data)


def _probe(config: ModelConfig, key: str) -> None:
    client: ChatClientProtocol
    if config.provider == "anthropic":
        client = AnthropicChatClient(config.base_url, key, config.model, timeout=20)
    else:
        client = ChatClient(
            config.base_url, key, config.model, timeout=20, reasoning_effort=config.reasoning_effort
        )
    content, _, error = client.chat([{"role": "user", "content": "Reply with: ok"}], 32)
    if error or not content.strip():
        # Provider errors can contain URLs or echoed headers. Never print them.
        raise ValueError(f"{config.name}: model endpoint probe failed")


def cmd_launch(args: argparse.Namespace) -> None:
    paths = sorted(CONFIGS.glob("*.toml"))
    if args.list:
        for path in paths:
            try:
                config = load_config(path)
            except ValueError as exc:
                raise SystemExit(f"launch: {exc}") from exc
            print(f"{config.name}: {config.display_name} ({config.model}, {config.trials} trials)")
        return
    names = args.configs
    if not names:
        for path in paths:
            print(path.stem)
        if not sys.stdin.isatty():
            raise SystemExit("launch: specify config names (or use --list)")
        names = input("Configs (space separated): ").split()
    if len(set(names)) != len(names):
        raise SystemExit("launch: duplicate config names")
    parallel = args.parallel
    if parallel is None:
        parallel = 1 if not sys.stdin.isatty() else int(input("Concurrent runs [1]: ") or "1")
    if parallel < 1:
        raise SystemExit("launch: --parallel must be at least 1")
    if args.max_attempts is not None and args.max_attempts < 1:
        raise SystemExit("launch: --max-attempts must be at least 1")
    if args.infra_retries is not None and args.infra_retries < 0:
        raise SystemExit("launch: --infra-retries must be nonnegative")
    configs = []
    for name in names:
        if not name or Path(name).name != name or name.endswith(".toml"):
            raise SystemExit(f"launch: invalid config name: {name}")
        path = CONFIGS / f"{name}.toml"
        if not path.is_file():
            raise SystemExit(f"launch: config not found: {name}")
        try:
            config = load_config(path)
        except Exception as exc:
            raise SystemExit(f"launch: {exc}") from exc
        key = os.environ.get(config.api_key_env)
        if not key:
            raise SystemExit(f"launch: {name}: {config.api_key_env} is unset")
        configs.append((config, key))
    for config, key in configs:
        try:
            _probe(config, key)
        except ValueError as exc:
            raise SystemExit(f"launch: {exc}") from exc
    commands = []
    for config, _ in configs:
        cmd = [
            sys.executable,
            "-m",
            "rangebench",
            "run",
            "--model",
            config.model,
            "--display-name",
            config.display_name,
            "--base-url",
            config.base_url,
            "--provider",
            config.provider,
            "--api-key-env",
            config.api_key_env,
            "--ctx-window",
            str(config.ctx_window),
            "--trials",
            str(config.trials),
        ]
        if config.reasoning_effort:
            cmd += ["--reasoning-effort", config.reasoning_effort]
        if config.route_name:
            cmd += ["--route-name", config.route_name]
        if config.upstream_provider:
            cmd += ["--upstream-provider", config.upstream_provider]
        if config.wall_clock_reference:
            cmd += ["--wall-clock-reference", config.wall_clock_reference]
        if args.max_attempts is not None:
            cmd += ["--max-attempts", str(args.max_attempts)]
        if args.infra_retries is not None:
            cmd += ["--infra-retries", str(args.infra_retries)]
        if args.status_dir:
            cmd += ["--status-json", str(args.status_dir / f"{config.name}.json")]
        commands.append(cmd)
        print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    # Each process has its own run ID and artifacts. A failure does not prevent
    # the other selected competitors from completing.
    active: list[subprocess.Popen] = []
    failed = False
    for cmd in commands:
        # Wait for any slot to free up, not the oldest process: a slow first
        # run must not idle capacity freed by a faster later run.
        while len(active) >= parallel:
            done = [process for process in active if process.poll() is not None]
            if done:
                for process in done:
                    active.remove(process)
                    failed |= process.wait() != 0
            else:
                time.sleep(0.05)
        active.append(subprocess.Popen(cmd))
    for process in active:
        failed |= process.wait() != 0
    if failed:
        raise SystemExit(1)
