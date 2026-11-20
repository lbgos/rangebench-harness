"""Ops simulator: hosts are real directories, services have state, stubs talk to it.

One Simulator per session attempt. A template's world.py builds it (hosts, files,
services, then boot()), the state server in sandbox.py feeds it stub commands from
inside the sandbox, and the checker reads its state and event log. Nothing here
runs inside the sandbox.

Service behaviour comes from config files on disk: start/restart/reload re-parse
them, so editing a config and restarting really fixes or breaks a service.
"""

from __future__ import annotations

import bisect
import dataclasses
import fnmatch
import hashlib
import json
import os
import posixpath
import re
import shlex
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlsplit

if TYPE_CHECKING:
    from .gh import Gh

GAP_EXIT = 64
ENOSPC = "No space left on device"
READ_LIMIT = 4 * 1024 * 1024
SYSTEM_KIB = 2_400 * 1024  # what du reports for /usr; part of every host's used space
# Top-level dirs that come from the real system (read-only in the sandbox), not the world.
SYSTEM_DIRS = ("usr", "bin", "sbin", "lib", "lib32", "lib64", "libx32")
MOUNT_POINTS = ("usr", "proc", "dev", "run")


@dataclass
class Result:
    """Output of one simulated command."""

    stdout: str = ""
    stderr: str = ""
    exit: int = 0
    probe: bool = False  # read-only check of live state (counts for `verified`)


@dataclass
class Event:
    """One command-log entry. kind: bash | stub | files | transition | push | merge | gh.

    transition actions: start stop restart reload, plus create/remove for containers.
    push: action create|update|force|delete, text "<host>:<repo> <ref>", paths [host:repo].
    merge: action squash|merge|rebase, text "<owner/repo>#<n> <method> author=<email>[ admin]".
    gh: action create|comment, text "<owner/repo>#<n> ...".
    """

    seq: int
    step: int
    call: int
    kind: str
    host: str
    text: str = ""
    exit: int | None = None
    probe: bool = False
    mutation: bool = False
    paths: list[str] = field(default_factory=list)
    service: str = ""
    action: str = ""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class Gap(Exception):
    """A command, subcommand or flag the simulator does not implement."""


class ConfigError(Exception):
    """A service config that fails to load; args are log lines."""


@dataclass
class Host:
    name: str
    ip: str
    aliases: list[str] = field(default_factory=list)
    vmid: int | None = None
    hypervisor: bool = False


# --- service configs -------------------------------------------------------


@dataclass
class Location:
    modifier: str  # "", "=", "^~", "~", "~*"
    path: str
    ret: tuple[int, str] | None = None
    proxy: str | None = None
    root: str | None = None
    alias: str | None = None
    index: list[str] = field(default_factory=list)
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class Server:
    ports: list[int]
    names: list[str]
    locations: list[Location]
    root: str | None = None
    index: list[str] = field(default_factory=list)


@dataclass
class ProxyConf:
    servers: list[Server]
    upstreams: dict[str, str]
    files: list[str]  # every file read, for nginx -T


@dataclass
class AppConf:
    port: int
    env: dict[str, str]


@dataclass
class DbConf:
    port: int
    settings: dict[str, str]


Conf = ProxyConf | AppConf | DbConf


@dataclass
class Service:
    """A unit (systemd) or container (docker) on one host."""

    host: str
    name: str
    kind: str  # proxy | app | db
    config: str  # absolute path on the host
    runtime: str = "systemd"
    description: str = ""
    routes: dict[str, str] = field(default_factory=dict)  # app: path -> JSON body
    image: str = ""
    enabled: bool = True
    state: str = "inactive"  # active | failed | inactive
    conf: Conf | None = None
    pid: int = 0
    since: datetime | None = None
    exit_status: int = 0
    # docker only
    present: bool = True  # the container exists (docker rm / compose down clear it)
    env: dict[str, str] = field(default_factory=dict)  # docker run -e, applied over the config file
    network: str = "host"  # "host", or "bridge": only published ports are reachable
    publish: dict[int, int] = field(default_factory=dict)  # -p: container port -> host port
    volumes: list[str] = field(default_factory=list)
    compose: str = ""  # compose project dir
    declared: tuple[str, str] = (
        "",
        "",
    )  # (config, image) from add_service; compose up restores them
    generation: int = 0  # bumps on every docker re-create (new container id)
    handler: AppHandler | None = None  # app: answers requests before routes and 404
    started_env: list[str] | None = None  # docker: Config.Env as of the last create/start
    workdir: str = ""  # systemd WorkingDirectory=, the process cwd (default /)
    user: str = ""  # systemd User= (default root)

    @property
    def unit(self) -> str:
        return f"{self.name}.service"

    def listen_ports(self) -> list[int]:
        """Ports the process listens on (inside its own network namespace)."""
        if self.conf is None or self.state != "active":
            return []
        if isinstance(self.conf, ProxyConf):
            return sorted({p for s in self.conf.servers for p in s.ports})
        return [self.conf.port]

    def ports(self) -> list[int]:
        """Ports reachable on the host's address."""
        own = self.listen_ports()
        if self.runtime != "docker" or self.network == "host":
            return own
        return sorted(self.publish[p] for p in own if p in self.publish)


@dataclass
class Image:
    """A docker image `docker run` can start on a host."""

    kind: str
    routes: dict[str, str] = field(default_factory=dict)
    config: str = ""  # config path inside the container; a -v bind onto it supplies the file


@dataclass
class Response:
    status: int
    body: str
    headers: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class Request:
    """What an app service's handler sees (see Simulator.add_service)."""

    method: str
    path: str
    query: str
    headers: dict[str, str]
    body: str
    client: str  # host the request came from


@dataclass
class ConnError:
    code: int  # curl exit code
    message: str


REASONS = {
    200: "OK",
    201: "Created",
    204: "No Content",
    301: "Moved Permanently",
    302: "Found",
    400: "Bad Request",
    403: "Forbidden",
    404: "Not Found",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
}
NGINX_VERSION = "1.24.0"
DOCKER_VERSION = "27.5.1"
KIND_DEFAULTS = {
    "proxy": ("A high performance web server and a reverse proxy server", "/usr/sbin/nginx"),
    "app": ("Application service", "/usr/bin/python3 -m app"),
    "db": ("PostgreSQL database server", "/usr/lib/postgresql/16/bin/postgres"),
}
SYSTEMD_VERSION = (
    "systemd 255 (255.4-1ubuntu8.6)\n"
    "+PAM +AUDIT +SELINUX +APPARMOR +IMA +SMACK +SECCOMP +GCRYPT -GNUTLS +OPENSSL +ACL +BLKID +CURL +ELFUTILS "
    "+FIDO2 +IDN2 -IDN +IPTC +KMOD +LIBCRYPTSETUP +LIBFDISK +PCRE2 -PWQUALITY +P11KIT +QRENCODE +TPM2 +BZIP2 "
    "+LZ4 +XZ +ZLIB +ZSTD -BPF_FRAMEWORK -XKBCOMMON +UTMP +SYSVINIT default-hierarchy=unified\n"
)
WELL_KNOWN_PORTS = {22: "ssh", 80: "http", 443: "https", 5432: "postgresql"}
MASK_DIR = "/etc/systemd/system"
# Names people pass to `systemctl show -p` that are not unit properties: systemd prints nothing.
NOT_UNIT_PROPERTIES = frozenset({"EnvironmentFile", "PID", "Pid", "Status", "State", "Loaded"})
# stdin of a stub call: text, or bytes when it is not UTF-8 (tar streams through ssh)
Stdin = str | bytes | None
EXEC_HOOK = Callable[[str, str, Stdin], Result]
# (sim, service, request) -> response, or None to fall through to /health, routes and 404.
AppHandler = Callable[["Simulator", "Service", Request], Response | None]


def resolve_in_root(root: Path, path: str) -> Path:
    """Map an absolute host path to the real path under root, following symlinks inside root."""
    parts = [p for p in path.split("/") if p not in ("", ".")]
    cur: list[str] = []
    hops = 0
    while parts:
        part = parts.pop(0)
        if part == "..":
            if cur:
                cur.pop()
            continue
        candidate = root.joinpath(*cur, part)
        if candidate.is_symlink():
            hops += 1
            if hops > 40:
                raise OSError(f"too many levels of symbolic links: {path}")
            target = os.readlink(candidate)
            if target.startswith("/"):
                cur = []
            parts = [p for p in target.split("/") if p not in ("", ".")] + parts
            continue
        cur.append(part)
    return root.joinpath(*cur)


def getopt(
    prog: str,
    args: list[str],
    short: str = "",
    short_valued: str = "",
    long: frozenset[str] = frozenset(),
    long_valued: frozenset[str] = frozenset(),
    stop_at_positional: bool = False,
    multi: frozenset[str] = frozenset(),
) -> tuple[dict[str, str], list[str]]:
    """GNU-style option parsing; unknown options raise Gap. Flags map to "" or their value;
    values of options in `multi` accumulate, newline-separated."""
    opts: dict[str, str] = {}
    pos: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--":
            pos.extend(args[i + 1 :])
            break
        if a.startswith("--") and len(a) > 2:
            name, eq, value = a[2:].partition("=")
            if name in long_valued:
                if not eq:
                    i += 1
                    if i >= len(args):
                        raise Gap(f"{prog} --{name} needs a value")
                    value = args[i]
                opts[name] = opts[name] + "\n" + value if name in multi and name in opts else value
            elif name in long and not eq:
                opts[name] = ""
            else:
                raise Gap(f"{prog} --{name}")
        elif a.startswith("-") and len(a) > 1:
            j = 1
            while j < len(a):
                c = a[j]
                if c in short_valued:
                    value = a[j + 1 :]
                    if not value:
                        i += 1
                        if i >= len(args):
                            raise Gap(f"{prog} -{c} needs a value")
                        value = args[i]
                    opts[c] = opts[c] + "\n" + value if c in multi and c in opts else value
                    break
                if c not in short:
                    raise Gap(f"{prog} -{c}")
                opts[c] = ""
                j += 1
        else:
            if stop_at_positional:
                pos.extend(args[i:])
                break
            pos.append(a)
        i += 1
    return opts, pos


# --- nginx-subset config parser --------------------------------------------

NGINX_BLOCKS = frozenset({"events", "http", "server", "location", "upstream", "types"})
NGINX_SIMPLE = frozenset(
    """user worker_processes pid error_log access_log include worker_connections
    sendfile tcp_nopush tcp_nodelay keepalive_timeout types_hash_max_size server_tokens
    default_type gzip gzip_vary gzip_proxied gzip_comp_level gzip_types gzip_min_length
    gzip_disable gzip_buffers gzip_http_version listen server_name root index try_files
    return proxy_pass proxy_set_header proxy_http_version proxy_read_timeout
    proxy_connect_timeout proxy_send_timeout proxy_redirect proxy_buffering
    client_max_body_size add_header ssl_certificate ssl_certificate_key ssl_protocols
    ssl_ciphers ssl_prefer_server_ciphers server_names_hash_bucket_size rewrite deny
    allow expires charset log_format real_ip_header set_real_ip_from limit_req
    limit_req_zone alias autoindex error_page internal multi_accept""".split()
)
NGINX_ALLOWED_IN = {
    "http": {"main"},
    "events": {"main"},
    "server": {"http", "upstream"},
    "upstream": {"http"},
    "location": {"server", "location"},
    "listen": {"server"},
    "server_name": {"server"},
    "proxy_pass": {"location"},
}


@dataclass
class Directive:
    name: str
    args: list[str]
    block: list[Directive] | None
    file: str
    line: int


def _nginx_tokens(text: str, fname: str) -> list[tuple[str, int, bool]]:
    """(token, line, quoted) triples; braces and semicolons are their own tokens."""
    out: list[tuple[str, int, bool]] = []
    i, line, n = 0, 1, len(text)
    while i < n:
        c = text[i]
        if c == "\n":
            line += 1
            i += 1
        elif c.isspace():
            i += 1
        elif c == "#":
            while i < n and text[i] != "\n":
                i += 1
        elif c in "{};":
            out.append((c, line, False))
            i += 1
        elif c in "\"'":
            j = i + 1
            buf = []
            while j < n and text[j] != c:
                if text[j] == "\\" and j + 1 < n:
                    j += 1
                if text[j] == "\n":
                    line += 1
                buf.append(text[j])
                j += 1
            if j >= n:
                raise ConfigError(f"nginx: [emerg] unexpected end of file in {fname}:{line}")
            out.append(("".join(buf), line, True))
            i = j + 1
        else:
            j = i
            while j < n and not text[j].isspace() and text[j] not in "{};\"'#":
                j += 1
            out.append((text[i:j], line, False))
            i = j
    return out


class _NginxParser:
    def __init__(self, sim: Simulator, host: str) -> None:
        self.sim = sim
        self.host = host
        self.files: list[str] = []

    def emerg(self, msg: str, fname: str, line: int) -> ConfigError:
        return ConfigError(f"nginx: [emerg] {msg} in {fname}:{line}")

    def parse_file(self, path: str, context: str) -> list[Directive]:
        text = self.sim.read(self.host, path)
        if text is None:
            raise ConfigError(
                f'nginx: [emerg] open() "{path}" failed (2: No such file or directory)'
            )
        self.files.append(path)
        tokens = _nginx_tokens(text, path)
        block, pos = self.parse_block(tokens, 0, path, context, top=True)
        return block

    def include(self, pattern: str, fname: str, line: int, context: str) -> list[Directive]:
        if not pattern.startswith("/"):
            pattern = "/etc/nginx/" + pattern
        if not any(ch in pattern for ch in "*?["):
            return self.parse_file(pattern, context)
        directory, base = posixpath.split(pattern)
        real = self.sim.real_path(self.host, directory)
        names = sorted(os.listdir(real)) if real.is_dir() else []
        out: list[Directive] = []
        for name in names:
            if fnmatch.fnmatch(name, base):
                out.extend(self.parse_file(posixpath.join(directory, name), context))
        return out

    def parse_block(
        self,
        tokens: list[tuple[str, int, bool]],
        pos: int,
        fname: str,
        context: str,
        top: bool,
    ) -> tuple[list[Directive], int]:
        out: list[Directive] = []
        words: list[str] = []
        start_line = 0
        last_line = tokens[-1][1] if tokens else 1
        while pos < len(tokens):
            tok, line, quoted = tokens[pos]
            pos += 1
            if not quoted and tok == "}":
                if words:
                    raise self.emerg('unexpected "}"', fname, line)
                if top:
                    raise self.emerg('unexpected "}"', fname, line)
                return out, pos
            if not quoted and tok in (";", "{"):
                if not words:
                    raise self.emerg(f'unexpected "{tok}"', fname, line)
                name, args = words[0], words[1:]
                words = []
                self.validate(name, context, fname, start_line, tok == "{")
                if tok == "{":
                    child_ctx = name
                    if name == "types":
                        depth = 1
                        while pos < len(tokens) and depth:
                            depth += {"{": 1, "}": -1}.get(tokens[pos][0], 0)
                            pos += 1
                        out.append(Directive(name, args, [], fname, start_line))
                        continue
                    block, pos = self.parse_block(tokens, pos, fname, child_ctx, top=False)
                    out.append(Directive(name, args, block, fname, start_line))
                elif name == "include":
                    if len(args) != 1:
                        raise self.emerg(
                            'invalid number of arguments in "include" directive', fname, start_line
                        )
                    out.extend(self.include(args[0], fname, start_line, context))
                else:
                    out.append(Directive(name, args, None, fname, start_line))
                continue
            if not words:
                start_line = line
            words.append(tok)
        if words:
            raise self.emerg('unexpected end of file, expecting ";" or "}"', fname, last_line)
        if not top:
            raise self.emerg('unexpected end of file, expecting "}"', fname, last_line)
        return out, pos

    def validate(self, name: str, context: str, fname: str, line: int, has_block: bool) -> None:
        if name not in NGINX_BLOCKS and name not in NGINX_SIMPLE:
            raise self.emerg(f'unknown directive "{name}"', fname, line)
        if (
            name in NGINX_BLOCKS
            and not has_block
            and not (name == "server" and context == "upstream")
        ):
            raise self.emerg(f'directive "{name}" has no opening "{{"', fname, line)
        if name in NGINX_SIMPLE and has_block:
            raise self.emerg(f'directive "{name}" is not terminated by ";"', fname, line)
        allowed = NGINX_ALLOWED_IN.get(name)
        if allowed is not None and context not in allowed:
            raise self.emerg(f'"{name}" directive is not allowed here', fname, line)


LISTEN_PARAMS = frozenset(
    "default_server ssl http2 quic proxy_protocol reuseport deferred bind ipv6only backlog rcvbuf sndbuf so_keepalive".split()
)


def _parse_listen(arg: str, fname: str, line: int) -> int | None:
    if arg.startswith("[::]"):
        return None
    port_s = arg.rsplit(":", 1)[-1]
    if not port_s.isdigit() or not 0 < int(port_s) < 65536:
        raise ConfigError(
            f'nginx: [emerg] invalid port in "{arg}" of the "listen" directive in {fname}:{line}'
        )
    return int(port_s)


def _location(d: Directive) -> Location:
    if not d.args:
        raise ConfigError(
            f'nginx: [emerg] invalid number of arguments in "location" directive in {d.file}:{d.line}'
        )
    modifier, path = ("", d.args[0]) if len(d.args) == 1 else (d.args[0], d.args[1])
    loc = Location(modifier=modifier, path=path)
    for c in d.block or []:
        if c.name == "return":
            code_s = c.args[0] if c.args else ""
            if not code_s.isdigit():
                raise ConfigError(
                    f'nginx: [emerg] invalid return code "{code_s}" in {c.file}:{c.line}'
                )
            loc.ret = (int(code_s), " ".join(c.args[1:]))
        elif c.name == "proxy_pass":
            url = c.args[0] if c.args else ""
            if not re.match(r"https?://", url):
                raise ConfigError(f"nginx: [emerg] invalid URL prefix in {c.file}:{c.line}")
            loc.proxy = url
        elif c.name == "root":
            loc.root = c.args[0] if c.args else None
        elif c.name == "alias":
            loc.alias = c.args[0] if c.args else None
        elif c.name == "index":
            loc.index = list(c.args)
        elif c.name == "proxy_set_header" and len(c.args) >= 2:
            loc.headers[c.args[0]] = c.args[1]
    return loc


def load_proxy(sim: Simulator, svc: Service) -> ProxyConf:
    parser = _NginxParser(sim, svc.host)
    top = parser.parse_file(svc.config, "main")
    servers: list[Server] = []
    upstreams: dict[str, str] = {}
    for d in top:
        if d.name != "http":
            continue
        for s in d.block or []:
            if s.name == "upstream" and s.args:
                first = next((u for u in s.block or [] if u.name == "server" and u.args), None)
                upstreams[s.args[0]] = first.args[0] if first else ""
            if s.name != "server":
                continue
            ports: list[int] = []
            names: list[str] = []
            locations: list[Location] = []
            server = Server(ports, names, locations)
            for c in s.block or []:
                if c.name == "listen" and c.args:
                    for extra in c.args[1:]:
                        if extra.split("=", 1)[0] not in LISTEN_PARAMS:
                            raise ConfigError(
                                f'nginx: [emerg] invalid parameter "{extra}" in {c.file}:{c.line}'
                            )
                    port = _parse_listen(c.args[0], c.file, c.line)
                    if port is not None:
                        ports.append(port)
                elif c.name == "server_name":
                    names.extend(c.args)
                elif c.name == "location":
                    loc = _location(c)
                    if any(x.path == loc.path and x.modifier == loc.modifier for x in locations):
                        raise ConfigError(
                            f'nginx: [emerg] duplicate location "{loc.path}" in {c.file}:{c.line}'
                        )
                    locations.append(loc)
                elif c.name == "root" and c.args:
                    server.root = c.args[0]
                elif c.name == "index":
                    server.index = list(c.args)
            if not ports:
                ports.append(80)
            servers.append(server)
    return ProxyConf(servers, upstreams, parser.files)


ENV_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
LOG_LEVELS = frozenset({"debug", "info", "warning", "error", "critical"})


def load_app(sim: Simulator, svc: Service) -> AppConf:
    """Env file at svc.config (none for `docker run` containers) plus svc.env on top."""
    text = sim.read(svc.host, svc.config) if svc.config else ""
    if text is None:
        raise ConfigError(f"ERROR: config file {svc.config} not found")
    env: dict[str, str] = {}
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = ENV_LINE.match(line)
        if not m:
            raise ConfigError(f"ERROR: config error: cannot parse {svc.config} line {n}: {line!r}")
        env[m.group(1)] = m.group(2).strip().strip("\"'")
    env.update(svc.env)
    port_s = env.get("PORT", "")
    if not port_s.isdigit():
        raise ConfigError(f"ERROR: config error: PORT must be an integer, got {port_s!r}")
    if "DB_PORT" in env and not env["DB_PORT"].isdigit():
        raise ConfigError(
            f"ERROR: config error: DB_PORT must be an integer, got {env['DB_PORT']!r}"
        )
    level = env.get("LOG_LEVEL", "info")
    if level.lower() not in LOG_LEVELS:
        raise ConfigError(
            f"ERROR: config error: invalid LOG_LEVEL {level!r} "
            f"(expected one of {', '.join(sorted(LOG_LEVELS))})"
        )
    return AppConf(int(port_s), env)


PG_SETTINGS = frozenset(
    """port listen_addresses max_connections shared_buffers data_directory hba_file
    ident_file external_pid_file unix_socket_directories ssl ssl_cert_file ssl_key_file
    work_mem maintenance_work_mem effective_cache_size wal_level max_wal_size
    min_wal_size checkpoint_completion_target log_line_prefix log_timezone timezone
    datestyle lc_messages lc_monetary lc_numeric lc_time default_text_search_config
    cluster_name dynamic_shared_memory_type include_dir log_min_duration_statement
    random_page_cost effective_io_concurrency archive_mode archive_command""".split()
)


def load_db(sim: Simulator, svc: Service) -> DbConf:
    text = sim.read(svc.host, svc.config)
    if text is None:
        raise ConfigError(f'FATAL:  could not open configuration file "{svc.config}"')
    settings: dict[str, str] = {}
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key, eq, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if not eq or not key:
            raise ConfigError(
                f'LOG:  syntax error in file "{svc.config}" line {n}, near token "{line.split()[0]}"'
            )
        if key not in PG_SETTINGS:
            raise ConfigError(
                f'LOG:  unrecognized configuration parameter "{key}" in file "{svc.config}" line {n}',
            )
        settings[key] = value
    for key in ("port", "max_connections"):
        if key in settings and not settings[key].isdigit():
            raise ConfigError(f'FATAL:  invalid value for parameter "{key}": "{settings[key]}"')
    return DbConf(int(settings.get("port", "5432")), settings)


LOADERS: dict[str, Callable[[Simulator, Service], Conf]] = {
    "proxy": load_proxy,
    "app": load_app,
    "db": load_db,
}


_TPL_ACTION = re.compile(r"\{\{(-?)\s*(.*?)\s*(-?)\}\}", re.S)
_TPL_WORD = re.compile(r'"(?:[^"\\]|\\.)*"|`[^`]*`|\(|\)|\||[^\s()|]+')
_TPL_VERB = re.compile(r"%[-+# 0-9.]*([vsdqf%])")


@dataclass
class _TplNode:
    kind: str  # text | expr | range | if | with
    arg: str = ""
    body: list[_TplNode] = field(default_factory=list)
    other: list[_TplNode] = field(default_factory=list)  # {{else}} branch


def _tpl_parse(fmt: str) -> list[_TplNode]:
    root: list[_TplNode] = []
    stack: list[tuple[_TplNode | None, list[_TplNode], bool]] = [
        (None, root, False)
    ]  # node, open list, else-if
    pos, trim = 0, False
    for m in _TPL_ACTION.finditer(fmt):
        text = fmt[pos : m.start()]
        text = text.lstrip() if trim else text
        text = text.rstrip() if m.group(1) else text
        pos, trim = m.end(), bool(m.group(3))
        node, items, chained = stack[-1]
        if text:
            items.append(_TplNode("text", text))
        word, _, rest = m.group(2).partition(" ")
        rest = rest.strip()
        if word in ("range", "if", "with"):
            new = _TplNode(word, rest)
            items.append(new)
            stack.append((new, new.body, False))
        elif word == "else":
            if node is None:
                raise Gap("template {{else}} outside a block")
            if rest.startswith("if "):
                inner = _TplNode("if", rest[3:].strip())
                node.other.append(inner)
                stack[-1] = (node, node.other, chained)
                stack.append((inner, inner.body, True))
            else:
                stack[-1] = (node, node.other, chained)
        elif word == "end":
            if node is None:
                raise Gap("template {{end}} without a block")
            while stack.pop()[2]:
                pass
        elif not word.startswith("/*"):
            items.append(_TplNode("expr", m.group(2)))
    tail = fmt[pos:]
    if trim:
        tail = tail.lstrip()
    if tail:
        stack[-1][1].append(_TplNode("text", tail))
    if len(stack) != 1:
        raise Gap("template block without {{end}}")
    return root


def _tpl_text(v: object) -> str:
    """A value as Go's fmt prints it."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "<nil>"
    if isinstance(v, list):
        return "[" + " ".join(_tpl_text(x) for x in v) + "]"
    if isinstance(v, dict):
        return "map[" + " ".join(f"{k}:{_tpl_text(v[k])}" for k in sorted(v)) + "]"
    return str(v)


def _tpl_truthy(v: object) -> bool:
    return not (v is None or v is False or v == 0 or v == "" or v == [] or v == {})


def _tpl_printf(fmt: str, *args: object) -> str:
    queue = list(args)

    def verb(m: re.Match[str]) -> str:
        if m.group(1) == "%":
            return "%"
        v = queue.pop(0) if queue else None
        return json.dumps(v) if m.group(1) == "q" else _tpl_text(v)

    return _TPL_VERB.sub(verb, fmt)


def _tpl_string(w: str) -> str:
    """A Go string literal; its escapes are JSON's (raw newlines from `\\n` expansion allowed)."""
    try:
        return str(json.loads(w, strict=False))
    except json.JSONDecodeError:
        raise Gap(f"template string {w}") from None


TPL_FUNCS: dict[str, Callable[..., object]] = {
    "json": lambda v: json.dumps(v, separators=(",", ":")),
    "println": lambda *a: " ".join(_tpl_text(x) for x in a) + "\n",
    "print": lambda *a: "".join(_tpl_text(x) for x in a),
    "printf": _tpl_printf,
    "len": lambda v: len(v) if isinstance(v, (list, dict, str)) else 0,
    "index": lambda v, *keys: _tpl_index(v, keys),
    "join": lambda v, sep: sep.join(_tpl_text(x) for x in v),
    "split": lambda s, sep: str(s).split(sep),
    "upper": lambda s: str(s).upper(),
    "lower": lambda s: str(s).lower(),
    "title": lambda s: str(s).title(),
    "eq": lambda a, *bs: any(a == b for b in bs),
    "ne": lambda a, b: a != b,
    "not": lambda a: not _tpl_truthy(a),
    "and": lambda *a: next((x for x in a if not _tpl_truthy(x)), a[-1]),
    "or": lambda *a: next((x for x in a if _tpl_truthy(x)), a[-1]),
}


def _tpl_index(v: object, keys: tuple[object, ...]) -> object:
    for k in keys:
        if isinstance(v, dict):
            v = v.get(str(k))
        elif isinstance(v, list) and isinstance(k, int) and 0 <= k < len(v):
            v = v[k]
        else:
            raise Gap(f"template index {k!r}")
    return v


_NOVALUE = object()


class _Tpl:
    """Evaluates a parsed template against one document."""

    def __init__(self, root: object) -> None:
        self.root = root

    def render(self, nodes: list[_TplNode], dot: object, env: dict[str, object]) -> str:
        out: list[str] = []
        for n in nodes:
            if n.kind == "text":
                out.append(n.arg)
            elif n.kind == "expr":
                out.append(_tpl_text(self.pipeline(n.arg, dot, env)))
            elif n.kind in ("if", "with"):
                v = self.pipeline(n.arg, dot, env)
                if _tpl_truthy(v):
                    out.append(self.render(n.body, v if n.kind == "with" else dot, env))
                else:
                    out.append(self.render(n.other, dot, env))
            else:
                out.append(self.range(n, dot, env))
        return "".join(out)

    def range(self, n: _TplNode, dot: object, env: dict[str, object]) -> str:
        names: list[str] = []
        expr = n.arg
        if ":=" in expr:
            lhs, expr = expr.split(":=", 1)
            names = [x.strip() for x in lhs.split(",")]
        v = self.pipeline(expr, dot, env)
        pairs: list[tuple[object, object]] = (
            [(k, v[k]) for k in sorted(v)]
            if isinstance(v, dict)
            else list(enumerate(v))
            if isinstance(v, list)
            else []
        )
        if not pairs:
            return self.render(n.other, dot, env)
        out = []
        for k, item in pairs:
            scope = dict(env)
            if len(names) == 2:
                scope[names[0]], scope[names[1]] = k, item
            elif names:
                scope[names[0]] = item
            out.append(self.render(n.body, item, scope))
        return "".join(out)

    def pipeline(self, src: str, dot: object, env: dict[str, object]) -> object:
        words = _TPL_WORD.findall(src)
        tree, end = self.group(words, 0)
        if end != len(words):
            raise Gap(f"template expression {src}")
        value: object = _NOVALUE
        cmd: list[object] = []
        for w in [*tree, "|"]:
            if w == "|":
                value = self.command(cmd, dot, env, value)
                cmd = []
            else:
                cmd.append(w)
        return value

    def group(self, words: list[str], i: int) -> tuple[list[object], int]:
        """Nest parenthesized words into sub-lists."""
        out: list[object] = []
        while i < len(words) and words[i] != ")":
            if words[i] == "(":
                sub, i = self.group(words, i + 1)
                if i >= len(words):
                    raise Gap("template: unclosed (")
                out.append(sub)
            else:
                out.append(words[i])
            i += 1
        return out, i

    def command(
        self, words: list[object], dot: object, env: dict[str, object], piped: object
    ) -> object:
        if not words:
            raise Gap("template: empty command")
        head = words[0]
        if isinstance(head, str) and head in TPL_FUNCS:
            args = [self.operand(w, dot, env) for w in words[1:]]
            if piped is not _NOVALUE:
                args.append(piped)
            try:
                return TPL_FUNCS[head](*args)
            except (TypeError, IndexError, AttributeError) as exc:
                raise Gap(f"template {head}: {exc}") from None
        if len(words) == 1 and piped is _NOVALUE:
            return self.operand(head, dot, env)
        raise Gap(f"template command {head}")

    def operand(self, w: object, dot: object, env: dict[str, object]) -> object:
        if isinstance(w, list):
            return self.command(w, dot, env, _NOVALUE)
        assert isinstance(w, str)
        if w.startswith('"'):
            return _tpl_string(w)
        if w.startswith("`"):
            return w[1:-1]
        if re.fullmatch(r"-?\d+", w):
            return int(w)
        if w in ("true", "false", "nil"):
            return {"true": True, "false": False, "nil": None}[w]
        if w in TPL_FUNCS:
            return TPL_FUNCS[w]()
        base, path = dot, w
        if w.startswith("$"):
            name, _, rest = w.partition(".")
            if name == "$":
                base = self.root
            elif name in env:
                base = env[name]
            else:
                raise Gap(f"template variable {name}")
            path = "." + rest if rest else "."
        if not path.startswith("."):
            raise Gap(f"template word {w}")
        for key in [k for k in path.split(".") if k]:
            if not isinstance(base, dict) or key not in base:
                raise Gap(f"template field {w}")
            base = base[key]
        return base


def go_template(fmt: str, doc: object) -> str:
    """Go templates as docker --format takes them: fields (`.A.B`, `$`, `$var`), `range`
    (with `$k, $v :=`), `if`/`else if`/`with`, pipes, and json, println, print, printf,
    index, len, join, split, upper, lower, title, eq, ne, not, and, or."""
    text = fmt.replace("\\n", "\n").replace("\\t", "\t")
    return _Tpl(doc).render(_tpl_parse(text), doc, {})


def text_table(rows: list[list[str]]) -> str:
    """Tab-separated cells aligned like Go's tabwriter (docker `table` formats)."""
    widths = [max(len(r[i]) for r in rows if i < len(r)) for i in range(max(map(len, rows)))]
    return "".join(
        "   ".join(c.ljust(widths[i]) if i < len(r) - 1 else c for i, c in enumerate(r)).rstrip()
        + "\n"
        for r in rows
    )


def docker_format(fmt: str, docs: Sequence[Mapping[str, object]], headers: dict[str, str]) -> str:
    """docker ls-style --format: a Go template per row; `table` adds a header row from
    `headers` (field -> column title) and aligns tab-separated columns."""
    if not fmt.startswith("table"):
        return "".join(go_template(fmt, d) + "\n" for d in docs)
    body = fmt.removeprefix("table").strip() or "\t".join("{{." + k + "}}" for k in headers)
    head = re.sub(
        r"\{\{\s*\.(\w+)\s*\}\}", lambda m: headers.get(m.group(1), m.group(1).upper()), body
    )
    lines = [head.replace("\\t", "\t")] + [go_template(body, d) for d in docs]
    return text_table([line.split("\t") for line in lines])


@dataclass
class Proc:
    pid: int
    ppid: int
    cmd: str
    start: datetime
    cwd: str = "/"
    user: str = "root"

    @property
    def comm(self) -> str:
        first = self.cmd.split()[0].rstrip(":")
        return os.path.basename(first)[:15]


# ps -o keyword -> (header, right-aligned)
PS_FIELDS = {
    "pid": ("PID", True),
    "ppid": ("PPID", True),
    "user": ("USER", False),
    "euser": ("EUSER", False),
    "uname": ("USER", False),
    "uid": ("UID", True),
    "cmd": ("CMD", False),
    "args": ("COMMAND", False),
    "command": ("COMMAND", False),
    "comm": ("COMMAND", False),
    "ucmd": ("CMD", False),
    "etime": ("ELAPSED", True),
    "etimes": ("ELAPSED", True),
    "lstart": ("STARTED", False),
    "start": ("STARTED", False),
    "start_time": ("START", False),
    "bsdstart": ("START", True),
    "stime": ("STIME", False),
    "stat": ("STAT", False),
    "state": ("S", False),
    "%cpu": ("%CPU", True),
    "pcpu": ("%CPU", True),
    "%mem": ("%MEM", True),
    "pmem": ("%MEM", True),
    "rss": ("RSS", True),
    "rsz": ("RSZ", True),
    "vsz": ("VSZ", True),
    "time": ("TIME", True),
    "cputime": ("TIME", True),
    "tty": ("TT", False),
    "tt": ("TT", False),
    "ni": ("NI", True),
    "pri": ("PRI", True),
    "c": ("C", True),
    "cwd": ("CWD", False),
    "pgid": ("PGID", True),
    "pgrp": ("PGRP", True),
    "sid": ("SID", True),
    "sess": ("SESS", True),
    "tpgid": ("TPGID", True),
    "nlwp": ("NLWP", True),
    "thcount": ("THCNT", True),
    "ruser": ("RUSER", False),
    "group": ("GROUP", False),
    "wchan": ("WCHAN", False),
    "f": ("F", True),
    "psr": ("PSR", True),
}


class PsError(Exception):
    """A procps usage error: printed with the usage hint, exit 1."""


PS_USAGE = (
    "\nUsage:\n ps [options]\n\n Try 'ps --help <simple|list|output|threads|misc|all>'\n"
    "  or 'ps --help <s|l|o|t|m|a>'\n for additional help text.\n\nFor more details see ps(1).\n"
)


def _ps_fields(value: str) -> list[str]:
    """-o list: comma or space separated; after `key=` the rest of the item is the header."""
    out: list[str] = []
    for item in value.split():
        head, eq, label = item.partition("=")
        keys = [f for f in head.split(",") if f]
        if eq and keys:
            keys[-1] += "=" + label
        out += keys
    return out


def _pid_list(value: str) -> set[int]:
    pids = value.replace(",", " ").split()
    if not pids or not all(p.isdigit() for p in pids):
        raise PsError("error: process ID list syntax error")
    return {int(p) for p in pids}


def _ps_sort(procs: list[Proc], spec: str, now: datetime) -> list[Proc]:
    """--sort=[+-]key[,...]: numeric keys by value, the rest as text; ties keep pid order."""
    for key in reversed([k for k in spec.split(",") if k]):
        desc = key.startswith("-")
        name = key.lstrip("+-")
        if name not in PS_FIELDS:
            raise PsError("error: unknown sort specifier")
        if name in ("lstart", "start", "start_time", "bsdstart", "stime"):
            procs = sorted(procs, key=lambda p: p.start, reverse=desc)
        elif name in ("etime", "etimes"):
            procs = sorted(procs, key=lambda p: p.start, reverse=not desc)
        else:
            values = {p.pid: _ps_value(name, p, now) for p in procs}
            numeric = all(re.fullmatch(r"[\d.]+", v) for v in values.values())
            procs = sorted(
                procs,
                key=lambda p: float(values[p.pid]) if numeric else values[p.pid],
                reverse=desc,
            )
    return procs


def _ps_value(key: str, p: Proc, now: datetime) -> str:
    secs = max(0, int((now - p.start).total_seconds()))
    if key == "etime":
        d, rest = divmod(secs, 86400)
        h, rest = divmod(rest, 3600)
        m, sec = divmod(rest, 60)
        return (
            f"{d}-{h:02}:{m:02}:{sec:02}"
            if d
            else f"{h:02}:{m:02}:{sec:02}"
            if h
            else f"{m:02}:{sec:02}"
        )
    if key in ("start", "start_time", "bsdstart", "stime"):
        return f"{p.start:%H:%M}" if secs < 86400 else f"{p.start:%b%d}"
    simple = {
        "pid": str(p.pid),
        "ppid": str(p.ppid),
        "uid": "0",
        "etimes": str(secs),
        "lstart": f"{p.start:%a %b} {p.start.day:>2} {p.start:%H:%M:%S %Y}",
        "stat": "Ss" if p.ppid <= 1 else "S",
        "state": "S",
        "%cpu": "0.0",
        "pcpu": "0.0",
        "%mem": "0.4",
        "pmem": "0.4",
        "rss": "14000",
        "rsz": "14000",
        "vsz": "98000",
        "time": "00:00:01",
        "cputime": "00:00:01",
        "tty": "?",
        "tt": "?",
        "ni": "0",
        "pri": "19",
        "c": "0",
    }
    if key in simple:
        return simple[key]
    if key in ("user", "euser", "uname", "ruser", "group"):
        return p.user if key != "group" else "root"
    if key == "cwd":
        return p.cwd
    if key in ("pgid", "pgrp", "sid", "sess"):
        return str(p.pid if p.ppid <= 1 else p.ppid)
    if key in ("tpgid",):
        return "-1"
    if key in ("nlwp", "thcount"):
        return "1"
    if key in ("wchan",):
        return "-"
    if key in ("f", "psr"):
        return "4" if key == "f" else "0"
    return p.comm if key in ("comm", "ucmd") else p.cmd


def _ps_row(cells: list[tuple[str, bool]], cols: list[tuple[str, bool, list[str]]]) -> str:
    parts = []
    for n, ((text, right), (head, _, values)) in enumerate(zip(cells, cols, strict=True)):
        width = max([len(head), *map(len, values)])
        last = n == len(cols) - 1
        parts.append(text.rjust(width) if right else text if last else text.ljust(width))
    return " ".join(parts).rstrip()


def _block_size(spec: str) -> tuple[int, str, str]:
    """df -B: (bytes per block, suffix printed after each count, header label)."""
    m = re.fullmatch(r"(\d*)([KMGT]?)(i?B?)", spec.strip())
    if not m or not (m.group(1) or m.group(2)):
        raise Gap(f"df -B {spec}")
    num = int(m.group(1) or "1")
    unit = m.group(2)
    base = 1000 if m.group(3) == "B" else 1024
    size = num * base ** "_KMGT".index(unit) if unit else num
    suffix = unit + m.group(3) if unit and not m.group(1) else ""
    if unit:
        label = f"{num}{unit}{m.group(3)}"
    else:  # GNU names a plain byte count by its largest exact binary unit
        k = max((i for i in range(5) if num % 1024**i == 0), default=0)
        label = f"{num // 1024**k}{'BKMGT'[k]}"
    return size, suffix, label


def _blocks(n: int, block: tuple[int, str, str]) -> str:
    return f"{-(-n // block[0])}{block[1]}"


def _human(n: int) -> str:
    """Bytes as du/df -h print them."""
    value = float(n)
    for suffix in ("", "K", "M", "G", "T"):
        if value < 1024 or suffix == "T":
            if suffix == "":
                return f"{int(value)}"
            return f"{value:.1f}{suffix}" if value < 10 else f"{value:.0f}{suffix}"
        value /= 1024
    return str(n)


def _image_ref(image: str) -> str:
    """Normalize an image reference: no tag means :latest."""
    last = image.rsplit("/", 1)[-1]
    return image if ":" in last or "@" in last else image + ":latest"


TIMESPAN_UNITS = {
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hour": 3600, "hours": 3600,
    "d": 86400, "day": 86400, "days": 86400,
    "w": 604800, "week": 604800, "weeks": 604800,
}  # fmt: skip
_SPAN_PART = re.compile(r"(\d+)\s*([a-z]+)")
_ABS_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d")


def parse_journal_time(text: str, now: datetime) -> datetime:
    """journalctl --since/--until values (UTC): absolute dates, HH:MM[:SS] today,
    now/today/yesterday/tomorrow, -1h / +30min / "2 hours ago", @epoch. Raises Gap otherwise."""
    s = text.strip().lower().removesuffix(" utc")
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    words = {
        "now": now,
        "today": midnight,
        "yesterday": midnight - timedelta(days=1),
        "tomorrow": midnight + timedelta(days=1),
    }
    if s in words:
        return words[s]
    day, _, clock = s.partition(" ")
    if day in ("today", "yesterday", "tomorrow") and clock:
        t = parse_journal_time(clock, now)
        return words[day].replace(hour=t.hour, minute=t.minute, second=t.second)
    if s.startswith("@") and s[1:].isdigit():
        return datetime.fromtimestamp(int(s[1:]), UTC)
    sign = 0
    body = s
    if body[:1] in "+-":
        sign, body = (1 if body[0] == "+" else -1), body[1:].strip()
    elif body.endswith(" ago"):
        sign, body = -1, body.removesuffix(" ago").strip()
    elif body.endswith(" left"):
        sign, body = 1, body.removesuffix(" left").strip()
    if sign:
        parts = _SPAN_PART.findall(body)
        if (
            parts
            and _SPAN_PART.sub("", body).strip() == ""
            and all(u in TIMESPAN_UNITS for _, u in parts)
        ):
            return now + sign * timedelta(seconds=sum(int(n) * TIMESPAN_UNITS[u] for n, u in parts))
        raise Gap(f"journalctl time {text!r}")
    for fmt in _ABS_FORMATS:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=UTC)
        except ValueError:
            pass
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            t = datetime.strptime(s, fmt)
        except ValueError:
            continue
        return midnight.replace(hour=t.hour, minute=t.minute, second=t.second)
    raise Gap(f"journalctl time {text!r}")


CURL_VERSION = (
    "curl 8.5.0 (x86_64-pc-linux-gnu) libcurl/8.5.0 OpenSSL/3.0.13 zlib/1.3 brotli/1.1.0 zstd/1.5.5 libidn2/2.4.0 nghttp2/1.59.0\n"
    "Release-Date: 2023-12-06\n"
    "Protocols: dict file ftp ftps gopher gophers http https imap imaps ldap ldaps mqtt pop3 pop3s rtmp rtsp scp sftp smb smbs smtp smtps telnet tftp\n"
    "Features: alt-svc AsynchDNS brotli GSS-API HSTS HTTP2 HTTPS-proxy IDN IPv6 Kerberos Largefile libz NTLM PSL SPNEGO SSL threadsafe TLS-SRP UnixSockets zstd\n"
)
CURL_BOUNDARY = "------------------------d74496d66958873e"
_WRITE_OUT = re.compile(r"%\{(\w+)\}|%header\{([^}]+)\}|%%")


def _header(headers: list[tuple[str, str]], name: str) -> str:
    return next((v for k, v in headers if k.lower() == name), "")


def _squash_dots(url: str) -> str:
    """Remove ./ and ../ path segments the way curl does unless --path-as-is."""
    parts = urlsplit(url if "://" in url else "http://" + url)
    if "/." not in parts.path:
        return url
    path = "/" + posixpath.normpath(parts.path).lstrip("/")
    if parts.path.endswith(("/", "/.", "/..")) and path != "/":
        path += "/"
    fixed = parts._replace(path=path).geturl()
    return fixed if "://" in url else fixed.removeprefix("http://")


SS_STATES = frozenset(
    {
        "established",
        "syn-sent",
        "syn-recv",
        "fin-wait-1",
        "fin-wait-2",
        "time-wait",
        "closed",
        "close-wait",
        "last-ack",
        "listening",
        "listen",
        "closing",
        "all",
        "connected",
        "synchronized",
        "bucket",
        "big",
    }
)


def _ss_filter(words: list[str]) -> tuple[set[str], set[int] | None]:
    """ss `state X` and `sport = :N` / `dport = :N` filters: (states, local ports or None).
    Every socket here listens, so a dport filter matches nothing."""
    tokens = re.findall(r"\(|\)|[^\s()]+", " ".join(words))
    states: set[str] = set()
    sports: set[int] | None = None
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "state" and i + 1 < len(tokens) and tokens[i + 1] in SS_STATES:
            states.add(tokens[i + 1])
            i += 2
            continue
        if t in ("sport", "dport") and i + 2 < len(tokens) and tokens[i + 1] in ("=", "==", "eq"):
            port_s = tokens[i + 2].rsplit(":", 1)[-1]
            port = (
                int(port_s)
                if port_s.isdigit()
                else next((p for p, n in WELL_KNOWN_PORTS.items() if n == port_s), None)
            )
            if port is None:
                raise Gap(f"ss filter {tokens[i + 2]}")
            sports = (sports or set()) | ({port} if t == "sport" else set())
            i += 3
            continue
        if t in ("(", ")", "or", "||", "and", "&&"):
            i += 1
            continue
        raise Gap("ss filter " + " ".join(words))
    return states, sports


PRIORITIES = ("emerg", "alert", "crit", "err", "warning", "notice", "info", "debug")


def _priority(name: str) -> int:
    """journalctl -p level: a name or 0-7."""
    n = {"error": "err", "warn": "warning", "critical": "crit", "emergency": "emerg"}.get(
        name.lower(), name.lower()
    )
    if n.isdigit() and int(n) < 8:
        return int(n)
    if n not in PRIORITIES:
        raise Gap(f"journalctl -p {name}")
    return PRIORITIES.index(n)


def _line_priority(line: str) -> int:
    """Syslog priority of a journal line, guessed from its words (lines carry none of their own)."""
    low = line.lower()
    for words, level in (
        (("emerg", "panic"), 0),
        (("alert",), 1),
        (("crit", "fatal"), 2),
        (("error", "[err", "failed", "failure", "traceback", "exception", "refused"), 3),
        (("warn",), 4),
        (("notice",), 5),
        (("debug",), 7),
    ):
        if any(w in low for w in words):
            return level
    return 6


NGINX_HELP = f"""nginx version: nginx/{NGINX_VERSION}
Usage: nginx [-?hvVtTq] [-s signal] [-p prefix]
             [-e filename] [-c filename] [-g directives]

Options:
  -?,-h         : this help
  -v            : show version and exit
  -V            : show version and configure options then exit
  -t            : test configuration and exit
  -T            : test configuration, dump it and exit
  -q            : suppress non-error messages during configuration testing
  -s signal     : send signal to a master process: stop, quit, reopen, reload
  -p prefix     : set prefix path (default: /usr/share/nginx/)
  -e filename   : set error log file (default: /var/log/nginx/error.log)
  -c filename   : set configuration file (default: /etc/nginx/nginx.conf)
  -g directives : set global directives out of configuration file
"""


SSH_G = """host {host}
user {user}
hostname {hostname}
port {port}
addressfamily any
batchmode no
checkhostip no
compression no
forwardagent no
hashknownhosts yes
identitiesonly no
passwordauthentication yes
pubkeyauthentication true
stricthostkeychecking ask
connecttimeout none
identityfile ~/.ssh/id_rsa
identityfile ~/.ssh/id_ecdsa
identityfile ~/.ssh/id_ed25519
serveraliveinterval 0
serveralivecountmax 3
userknownhostsfile ~/.ssh/known_hosts ~/.ssh/known_hosts2
"""


def _nginx_page(status: int) -> str:
    title = f"{status} {REASONS.get(status, '')}".strip()
    return (
        f"<html>\n<head><title>{title}</title></head>\n<body>\n"
        f"<center><h1>{title}</h1></center>\n<hr><center>nginx/{NGINX_VERSION}</center>\n"
        "</body>\n</html>\n"
    )


def host_git(
    repo: Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run git on a world repo from the simulator (outside the sandbox) with hooks,
    fsmonitor and global/system config disabled, so repo contents cannot run code."""
    base = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/nonexistent",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    return subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            "-C",
            str(repo),
            *args,
        ],
        env={**base, **(env or {})},
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )


GIT_PACK_PROGS = ("git-upload-pack", "git-receive-pack", "git-upload-archive")
STREAM = Callable[[str, str], int]  # (host, command) -> exit code, stdio streamed to the client

DOCKER_HELP = """
Usage:  docker [OPTIONS] COMMAND

A self-sufficient runtime for containers

Common Commands:
  run         Create and run a new container from an image
  exec        Execute a command in a running container
  ps          List containers
  images      List images
  pull        Download an image from a registry
  version     Show the Docker version information

Management Commands:
  compose     Docker Compose
  container   Manage containers

Commands:
  create      Create a new container
  inspect     Return low-level information on Docker objects
  logs        Fetch the logs of a container
  port        List port mappings or a specific mapping for the container
  restart     Restart one or more containers
  rm          Remove one or more containers
  start       Start one or more stopped containers
  stop        Stop one or more running containers

Run 'docker COMMAND --help' for more information on a command.
"""
DOCKER_USAGE = {
    "run": "docker run [OPTIONS] IMAGE\n\nOptions: -d, --name, -p host:container, -e KEY=VAL, --env-file FILE, -v src:dst[:ro], --restart, --network",
    "create": "docker create [OPTIONS] IMAGE\n\nOptions: --name, -p host:container, -e KEY=VAL, --env-file FILE, -v src:dst[:ro], --restart, --network",
    "exec": "docker exec [OPTIONS] CONTAINER COMMAND [ARG...]\n\nOptions: -i, -t, -e KEY=VAL, -w DIR, -u USER",
    "rm": "docker rm [OPTIONS] CONTAINER [CONTAINER...]\n\nOptions: -f, --force, -v, --volumes",
    "ps": "docker ps [OPTIONS]\n\nOptions: -a, --all, -q, --quiet, --format",
    "logs": "docker logs [OPTIONS] CONTAINER\n\nOptions: -n, --tail N, -t, --timestamps",
    "compose": "docker compose [-f FILE] [-p NAME] [--project-directory DIR] COMMAND\n\nCommands: up -d, down, ps, logs, start, stop, restart",
}


# --- the simulator ---------------------------------------------------------


class Simulator:
    """World state plus stub-command handlers. Thread-safe; one per session attempt."""

    def __init__(self, root: Path, local_host: str = "") -> None:
        self.root = root
        self.hosts_dir = root / "hosts"
        self.hosts_dir.mkdir(parents=True, exist_ok=True)
        self.local_host = local_host
        self.hosts: dict[str, Host] = {}
        self.services: list[Service] = []
        self.journal: dict[str, list[tuple[datetime, str, str]]] = {}  # host -> (ts, unit, line)
        self.container_logs: dict[tuple[str, str], list[str]] = {}
        self.events: list[Event] = []
        self.gaps: list[dict[str, str]] = []
        self.disks: dict[str, int] = {}  # host -> size of / in KiB; used = SYSTEM_KIB + world files
        self.tls = threading.local()  # per stub call: cwd
        self.on_event: Callable[[Event], None] | None = None
        self.on_gap: Callable[[dict[str, str]], None] | None = None
        self.exec_hook: EXEC_HOOK | None = None  # set by the session: run a command on a host
        self.images: dict[tuple[str, str], Image] = {}  # (host, image) -> what it runs
        self.gh: Gh | None = None  # GitHub mock, created by github()
        self.lock = threading.RLock()
        self.recording = True
        self.step = 0
        self.call = 0
        self._seq = 0
        self._pid = 900
        self._snapshot: dict[str, tuple[int, int, str]] = {}

    # world building (used by world.py) -------------------------------------

    def add_host(
        self,
        name: str,
        ip: str,
        aliases: list[str] | None = None,
        vmid: int | None = None,
        hypervisor: bool = False,
    ) -> Host:
        host = Host(name, ip, aliases or [], vmid, hypervisor)
        self.hosts[name] = host
        self.journal[name] = []
        base = self.hosts_dir / name
        for d in ("etc", "root", "tmp", "var/log", "srv", *MOUNT_POINTS):
            (base / d).mkdir(parents=True, exist_ok=True)
        for d in SYSTEM_DIRS:
            src = Path("/") / d
            if src.is_symlink():
                (base / d).symlink_to(os.readlink(src))
            elif src.is_dir():
                (base / d).mkdir(exist_ok=True)
        self.write(name, "/etc/hostname", name + "\n")
        self.write(name, "/etc/passwd", "root:x:0:0:root:/root:/bin/bash\n")
        self.write(name, "/etc/group", "root:x:0:\n")
        alt = Path("/etc/alternatives")
        if alt.is_dir():  # Debian resolves awk, editor, ... through these links
            (base / "etc/alternatives").mkdir(exist_ok=True)
            for entry in alt.iterdir():
                if entry.is_symlink():
                    (base / "etc/alternatives" / entry.name).symlink_to(os.readlink(entry))
        self.disks[name] = 20 * 1024 * 1024
        return host

    def sparse(self, host: str, path: str, size: int) -> None:
        """A file with a large apparent size and no real disk use."""
        real = self.real_path(host, path)
        real.parent.mkdir(parents=True, exist_ok=True)
        with real.open("wb") as f:
            f.truncate(size)

    def write(self, host: str, path: str, text: str) -> None:
        real = self.real_path(host, path)
        real.parent.mkdir(parents=True, exist_ok=True)
        real.write_text(text, encoding="utf-8")

    def git_repo(
        self, host: str, path: str, name: str, email: str, message: str = "Initial commit"
    ) -> None:
        """git init the directory at path and commit everything in it as name <email>."""
        real = str(self.real_path(host, path))
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": name,
            "GIT_AUTHOR_EMAIL": email,
            "GIT_COMMITTER_NAME": name,
            "GIT_COMMITTER_EMAIL": email,
            "GIT_CONFIG_GLOBAL": "/dev/null",
        }
        for argv in (["init", "-q", "-b", "main"], ["add", "-A"], ["commit", "-q", "-m", message]):
            subprocess.run(["git", "-C", real, *argv], env=env, check=True, capture_output=True)

    def symlink(self, host: str, path: str, target: str) -> None:
        link = self.hosts_dir / host / path.lstrip("/")
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)

    def add_service(
        self,
        host: str,
        name: str,
        kind: str,
        config: str,
        runtime: str = "systemd",
        description: str = "",
        routes: dict[str, str] | None = None,
        image: str = "",
        enabled: bool = True,
        compose: str = "",
        handler: AppHandler | None = None,
        workdir: str = "",
        user: str = "",
    ) -> Service:
        """compose (docker only): the dir holding the compose file that owns this container.
        handler (app only): called for every request while the service runs, see AppHandler.
        workdir, user (systemd): WorkingDirectory= and User= of the unit; ps shows them as cwd and user."""
        if kind not in LOADERS:
            raise ValueError(f"unknown service kind {kind!r}")
        svc = Service(
            host,
            name,
            kind,
            config,
            runtime,
            description or KIND_DEFAULTS[kind][0],
            dict(routes or {}),
            image,
            enabled,
            compose=compose.rstrip("/"),
            declared=(config, image),
            handler=handler,
            workdir=workdir,
            user=user,
        )
        self.services.append(svc)
        return svc

    def add_image(
        self,
        host: str,
        image: str,
        kind: str,
        routes: dict[str, str] | None = None,
        config: str = "",
    ) -> None:
        """An image `docker run`/`create` can start on host. Images of declared containers
        are known without this. config: path inside the container a -v bind supplies."""
        if kind not in LOADERS:
            raise ValueError(f"unknown service kind {kind!r}")
        self.images[(host, _image_ref(image))] = Image(kind, dict(routes or {}), config)

    def github(self, login: str, email: str, name: str = "", host: str = "github.com") -> Gh:
        """The GitHub mock behind `gh` and `git@<host>:owner/repo.git` remotes (see gh.py)."""
        from .gh import Gh

        if self.gh is None:
            if host not in self.hosts:
                self.add_host(host, "140.82.121.4")
            self.gh = Gh(self, host, login, email, name or login)
        return self.gh

    def boot(self) -> None:
        """Start every enabled service from its config on disk, unrecorded."""
        recording, self.recording = self.recording, False
        try:
            for svc in self.services:
                if svc.enabled:
                    self.start(svc)
        finally:
            self.recording = recording
        self.sync_files(record=False)

    # filesystem ------------------------------------------------------------

    def real_path(self, host: str, path: str) -> Path:
        return resolve_in_root(self.hosts_dir / host, path)

    def read(self, host: str, path: str, limit: int = READ_LIMIT) -> str | None:
        """Text of a file (first `limit` bytes; sparse dumps can be gigabytes), None if absent."""
        try:
            real = self.real_path(host, path)
            if not real.is_file():
                return None
            with real.open("rb") as f:
                return f.read(limit).decode("utf-8", errors="replace")
        except OSError:
            return None

    def disk_used(self, host: str) -> int:
        """Used KiB of /: the OS (/usr etc.) plus the apparent size of every world file."""
        return min(self.disks[host], self._used_kib(host))

    def _used_kib(self, host: str) -> int:
        files = sum(
            size for key, (size, _, _) in self.snapshot().items() if key.startswith(host + ":")
        )
        return SYSTEM_KIB + files // 1024

    def disk_free(self, host: str) -> int:
        """Bytes a write on host may still add before its simulated / is full."""
        return max(0, self.disks[host] - self._used_kib(host)) * 1024

    def snapshot(self) -> dict[str, tuple[int, int, str]]:
        """Stat of every file and symlink as host:/path -> (size, mtime_ns, link target)."""
        snap: dict[str, tuple[int, int, str]] = {}
        for host in self.hosts:
            base = self.hosts_dir / host
            for dirpath, dirnames, filenames in os.walk(base):
                if dirpath == str(base):
                    dirnames[:] = [
                        d for d in dirnames if d not in SYSTEM_DIRS and d not in MOUNT_POINTS
                    ]
                    filenames = [f for f in filenames if f not in SYSTEM_DIRS]
                entries = filenames + [
                    d for d in dirnames if os.path.islink(os.path.join(dirpath, d))
                ]
                for name in entries:
                    full = os.path.join(dirpath, name)
                    try:
                        st = os.lstat(full)
                    except OSError:
                        continue
                    target = os.readlink(full) if os.path.islink(full) else ""
                    rel = "/" + os.path.relpath(full, base)
                    snap[f"{host}:{rel}"] = (st.st_size, st.st_mtime_ns, target)
        return snap

    def sync_files(self, record: bool = True) -> list[str]:
        """Diff the world against the last snapshot; log a files event for changed paths."""
        with self.lock:
            snap = self.snapshot()
            old = self._snapshot
            changed = sorted(p for p in set(snap) | set(old) if snap.get(p) != old.get(p))
            self._snapshot = snap
            if changed and record and self.recording:
                host = changed[0].split(":", 1)[0]
                self.event("files", host, paths=changed, mutation=True)
            return changed

    # event log -------------------------------------------------------------

    def event(self, kind: str, host: str, **fields: object) -> Event | None:
        if not self.recording:
            return None
        with self.lock:
            self._seq += 1
            ev = Event(self._seq, self.step, self.call, kind, host)
            for key, value in fields.items():
                setattr(ev, key, value)
            self.events.append(ev)
        if self.on_event and kind != "bash":
            self.on_event(ev)
        return ev

    def gap(self, host: str, argv: list[str], detail: str) -> Result:
        entry = {"host": host, "argv": shlex.join(argv), "detail": detail}
        with self.lock:
            self.gaps.append(entry)
        if self.on_gap:
            self.on_gap(entry)
        prog = os.path.basename(argv[0]) if argv else "?"
        return Result(stderr=f"{prog}: unsupported here: {detail}\n", exit=GAP_EXIT)

    # services --------------------------------------------------------------

    def find(self, host: str, name: str, runtime: str | None = None) -> Service | None:
        name = name.removesuffix(".service")
        for svc in self.services:
            if (
                svc.host == host
                and svc.name == name
                and (runtime is None or svc.runtime == runtime)
            ):
                return svc
        return None

    def service(self, host: str, name: str) -> Service:
        svc = self.find(host, name)
        if svc is None:
            raise KeyError(f"no service {name} on {host}")
        return svc

    def now(self) -> datetime:
        return datetime.now(UTC)

    def log(
        self, svc: Service, line: str, ident: str | None = None, at: datetime | None = None
    ) -> None:
        """Journal (systemd) or docker logs line. at: timestamp for pre-seeded history
        (e.g. sim.now() - timedelta(hours=3)), so journalctl --since/--until can select it."""
        if svc.runtime == "docker":
            self.container_logs.setdefault((svc.host, svc.name), []).append(line)
            return
        pid = svc.pid or self._pid
        entry = (at or self.now(), svc.name, f"{ident or svc.name}[{pid}]: {line}")
        bisect.insort(self.journal[svc.host], entry, key=lambda e: e[0])

    def syslog(self, svc: Service, line: str) -> None:
        if svc.runtime == "docker":
            return
        self.journal[svc.host].append((self.now(), svc.name, f"systemd[1]: {line}"))

    def start(self, svc: Service) -> str | None:
        """Start from config on disk. Returns the error line on failure."""
        if svc.state == "active":
            return None
        self._pid += 7
        svc.pid = self._pid
        if svc.runtime == "docker":
            svc.started_env = self._env_file_lines(svc)
        self.syslog(svc, f"Starting {svc.unit} - {svc.description}...")
        try:
            conf = LOADERS[svc.kind](self, svc)
            busy = {
                p: other
                for other in self.services
                if other is not svc and other.host == svc.host
                for p in other.ports()
            }
            ports = dataclasses.replace(svc, state="active", conf=conf).ports()
            for p in ports:
                if p in busy:
                    raise ConfigError(self._bind_error(svc, p))
        except ConfigError as exc:
            for line in str(exc.args[0]).splitlines():
                self.log(svc, line)
            svc.state, svc.conf, svc.exit_status = "failed", None, 1
            svc.since = self.now()
            self.syslog(svc, f"{svc.unit}: Control process exited, code=exited, status=1/FAILURE")
            self.syslog(svc, f"{svc.unit}: Failed with result 'exit-code'.")
            self.syslog(svc, f"Failed to start {svc.unit} - {svc.description}.")
            return str(exc.args[0])
        svc.state, svc.conf, svc.exit_status, svc.since = "active", conf, 0, self.now()
        self._log_started(svc, conf)
        self.syslog(svc, f"Started {svc.unit} - {svc.description}.")
        return None

    def _bind_error(self, svc: Service, port: int) -> str:
        if svc.kind == "proxy":
            return f"nginx: [emerg] bind() to 0.0.0.0:{port} failed (98: Address already in use)"
        if svc.kind == "db":
            return f"FATAL:  could not create any TCP/IP sockets (port {port} in use)"
        return f"ERROR: [Errno 98] error while attempting to bind on address ('0.0.0.0', {port}): address already in use"

    def _log_started(self, svc: Service, conf: Conf) -> None:
        if isinstance(conf, AppConf):
            self.log(svc, "INFO:     Started server process [1]")
            self.log(
                svc,
                f"INFO:     Uvicorn running on http://0.0.0.0:{conf.port} (Press CTRL+C to quit)",
            )
        elif isinstance(conf, DbConf):
            self.log(
                svc, f'LOG:  listening on IPv4 address "0.0.0.0", port {conf.port}', "postgres"
            )
            self.log(svc, "LOG:  database system is ready to accept connections", "postgres")

    def stop(self, svc: Service) -> None:
        if svc.state == "active":
            self.syslog(svc, f"Stopping {svc.unit} - {svc.description}...")
            self.syslog(svc, f"{svc.unit}: Deactivated successfully.")
            self.syslog(svc, f"Stopped {svc.unit} - {svc.description}.")
            if svc.runtime == "docker":
                self.log(svc, "INFO:     Shutting down")
        svc.state, svc.conf, svc.exit_status = "inactive", None, 0
        svc.since = self.now()

    def reload(self, svc: Service) -> str | None:
        """Re-read config; on error keep running the old one (nginx semantics)."""
        if svc.state != "active":
            return f"{svc.unit} is not active, cannot reload."
        try:
            conf = LOADERS[svc.kind](self, svc)
        except ConfigError as exc:
            for line in str(exc.args[0]).splitlines():
                self.log(svc, line)
            self.syslog(svc, f"{svc.unit}: Control process exited, code=exited, status=1/FAILURE")
            self.syslog(svc, f"Reload failed for {svc.unit} - {svc.description}.")
            return str(exc.args[0])
        svc.conf = conf
        self.syslog(svc, f"Reloaded {svc.unit} - {svc.description}.")
        return None

    def transition(self, svc: Service, action: str) -> str | None:
        """start | stop | restart | reload, recorded as a mutation. Returns an error line."""
        error: str | None = None
        if action == "start":
            error = self.start(svc)
        elif action == "stop":
            self.stop(svc)
        elif action == "restart":
            self.stop(svc)
            error = self.start(svc)
        elif action == "reload":
            error = self.reload(svc)
        else:
            raise ValueError(action)
        self.event(
            "transition",
            svc.host,
            service=svc.name,
            action=action,
            mutation=True,
            exit=1 if error else 0,
            text=f"{action} {svc.name}",
        )
        return error

    # HTTP --------------------------------------------------------------------

    def resolve_host(self, name: str, from_host: str) -> str | None:
        if name in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
            return from_host
        for host in self.hosts.values():
            if name == host.name or name == host.ip or name in host.aliases:
                return host.name
        return None

    def request(
        self,
        from_host: str,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        depth: int = 0,
        body: str = "",
    ) -> Response | ConnError:
        parts = urlsplit(url if "://" in url else "http://" + url)
        hostname = parts.hostname or ""
        target = self.resolve_host(hostname, from_host)
        if target is None:
            return ConnError(6, f"Could not resolve host: {hostname}")
        try:
            port = parts.port or (443 if parts.scheme == "https" else 80)
        except ValueError:
            return ConnError(3, "URL rejected: Port number was not a decimal number")
        svc = next((s for s in self.services if s.host == target and port in s.ports()), None)
        if svc is None or svc.conf is None:
            return ConnError(
                7,
                f"Failed to connect to {hostname} port {port} after 0 ms: Couldn't connect to server",
            )
        path = parts.path or "/"
        hdrs = {"Host": hostname, **(headers or {})}
        if isinstance(svc.conf, ProxyConf):
            return self._serve_proxy(
                svc, svc.conf, port, method, path, parts.query, hdrs, depth, body
            )
        if isinstance(svc.conf, AppConf):
            return self._serve_app(
                svc, svc.conf, Request(method, path, parts.query, hdrs, body, from_host)
            )
        return ConnError(52, "Empty reply from server")

    def _serve_app(self, svc: Service, conf: AppConf, req: Request) -> Response:
        ip = self.hosts[req.client].ip
        method, path = req.method, req.path
        json_hdr = [("content-type", "application/json")]
        custom = svc.handler(self, svc, req) if svc.handler else None
        if custom is not None:
            resp = custom
        elif path == "/health":
            db_host = conf.env.get("DB_HOST")
            if db_host is None:
                resp = Response(200, '{"status":"ok"}', json_hdr)
            else:
                ok = self._db_reachable(svc.host, db_host, int(conf.env.get("DB_PORT", "5432")))
                if ok:
                    resp = Response(200, '{"status":"ok","db":"up"}', json_hdr)
                else:
                    self.log(
                        svc,
                        f"ERROR:    database check failed: could not connect to {db_host}:{conf.env.get('DB_PORT', '5432')}",
                    )
                    resp = Response(503, '{"status":"error","db":"down"}', json_hdr)
        elif path in svc.routes:
            resp = Response(200, svc.routes[path], json_hdr)
        else:
            resp = Response(404, '{"detail":"Not Found"}', json_hdr)
        self.log(
            svc,
            f'INFO:     {ip}:51234 - "{method} {path} HTTP/1.1" {resp.status} {REASONS.get(resp.status, "")}',
        )
        return resp

    def _db_reachable(self, from_host: str, db_host: str, port: int) -> bool:
        target = self.resolve_host(db_host, from_host)
        if target is None:
            return False
        return any(s.host == target and s.kind == "db" and port in s.ports() for s in self.services)

    def _serve_proxy(
        self,
        svc: Service,
        conf: ProxyConf,
        port: int,
        method: str,
        path: str,
        query: str,
        headers: dict[str, str],
        depth: int,
        body: str = "",
    ) -> Response:
        nginx_hdr = [("Server", f"nginx/{NGINX_VERSION}"), ("Content-Type", "text/html")]
        candidates = [s for s in conf.servers if port in s.ports]
        host_name = headers.get("Host", "")
        server = next((s for s in candidates if host_name in s.names), candidates[0])
        loc = self._match_location(server.locations, path)
        if loc is None:
            return self._static(svc, server.root, None, server.index, path, "", nginx_hdr)
        if loc.ret is not None:
            code, text = loc.ret
            if code in (301, 302):
                return Response(code, _nginx_page(code), nginx_hdr + [("Location", text)])
            if text:
                return Response(
                    code,
                    text,
                    [
                        ("Server", f"nginx/{NGINX_VERSION}"),
                        ("Content-Type", "application/octet-stream"),
                    ],
                )
            return Response(code, _nginx_page(code), nginx_hdr)
        if loc.proxy is not None:
            if depth > 5:
                return Response(508, _nginx_page(508), nginx_hdr)
            up = urlsplit(loc.proxy)
            netloc = conf.upstreams.get(up.netloc, up.netloc)
            if up.path:
                rest = path[len(loc.path) :] if path.startswith(loc.path) else path
                new_path = up.path + rest
            else:
                new_path = path
            url = f"{up.scheme}://{netloc}{new_path}" + (f"?{query}" if query else "")
            # nginx passes the client's headers (Host becomes the upstream's) and body on
            forwarded = {k: v for k, v in headers.items() if k.lower() != "host"} | dict(
                loc.headers
            )
            resp = self.request(svc.host, method, url, forwarded, depth + 1, body)
            if isinstance(resp, ConnError):
                self.log(
                    svc,
                    f"[error] connect() failed (111: Connection refused) while connecting to upstream, "
                    f'request: "{method} {path} HTTP/1.1", upstream: "{url}"',
                    "nginx",
                )
                return Response(502, _nginx_page(502), nginx_hdr)
            return Response(
                resp.status, resp.body, [("Server", f"nginx/{NGINX_VERSION}")] + resp.headers
            )
        return self._static(
            svc,
            loc.root or server.root,
            loc if loc.alias else None,
            loc.index or server.index,
            path,
            loc.path,
            nginx_hdr,
        )

    def _match_location(self, locations: list[Location], path: str) -> Location | None:
        for loc in locations:
            if loc.modifier == "=" and loc.path == path:
                return loc
        prefix = [
            loc for loc in locations if loc.modifier in ("", "^~") and path.startswith(loc.path)
        ]
        best = max(prefix, key=lambda loc: len(loc.path), default=None)
        if best is not None and best.modifier == "^~":
            return best
        for loc in locations:
            if loc.modifier in ("~", "~*"):
                flags = re.IGNORECASE if loc.modifier == "~*" else 0
                try:
                    if re.search(loc.path, path, flags):
                        return loc
                except re.error:
                    continue
        return best

    def _static(
        self,
        svc: Service,
        root: str | None,
        alias_loc: Location | None,
        index: list[str],
        path: str,
        loc_path: str,
        hdrs: list[tuple[str, str]],
    ) -> Response:
        if alias_loc is not None and alias_loc.alias:
            fs_path = posixpath.join(alias_loc.alias, path[len(loc_path) :])
        elif root:
            fs_path = root.rstrip("/") + posixpath.normpath("/" + path)
        else:
            return Response(404, _nginx_page(404), hdrs)
        candidates = [fs_path]
        if path.endswith("/"):
            candidates = [posixpath.join(fs_path, i) for i in (index or ["index.html"])]
        for cand in candidates:
            text = self.read(svc.host, cand)
            if text is not None:
                return Response(
                    200, text, [("Server", f"nginx/{NGINX_VERSION}"), ("Content-Type", "text/html")]
                )
        self.log(
            svc, f'[error] open() "{candidates[0]}" failed (2: No such file or directory)', "nginx"
        )
        return Response(404, _nginx_page(404), hdrs)

    # stub dispatch -----------------------------------------------------------

    def handle(self, host: str, argv: list[str], cwd: str = "/root", stdin: Stdin = None) -> Result:
        """Run one stub command (argv[0] is the program name) on host and log it. Binary stdin
        reaches only commands that pass it to another host (ssh, pct exec, docker exec)."""
        prog = os.path.basename(argv[0]) if argv else ""
        # Commands that run other commands in a sandbox get stdin as sent and must not hold
        # the lock: what they run may call stubs, which are served on other threads.
        relay = prog in ("ssh", "pct") or (prog == "docker" and argv[1:2] == ["exec"])
        text = stdin.decode(errors="replace") if isinstance(stdin, bytes) else stdin
        relay_handler = self.RELAY_HANDLERS.get(prog)
        text_handler = self.TEXT_HANDLERS.get(prog)
        self.tls.cwd = cwd
        self.sync_files()
        ev = self.event("stub", host, text=shlex.join(argv))
        if relay_handler is None and text_handler is None:
            res = Result(stderr=f"bash: {prog}: command not found\n", exit=127)
        else:

            def call() -> Result:
                if relay_handler is not None:
                    return relay_handler(self, host, argv[1:], stdin if relay else text)
                assert text_handler is not None
                return text_handler(self, host, argv[1:], text)

            try:
                if relay:
                    res = call()
                else:
                    with self.lock:
                        res = call()
            except Gap as gap:
                res = self.gap(host, argv, str(gap))
            except Exception as exc:  # a simulator bug: log it as a gap, answer like a broken tool
                self.gap(host, argv, f"internal error {type(exc).__name__}: {exc}")
                res = Result(stderr=f"{prog}: internal error\n", exit=GAP_EXIT)
        if ev is not None:
            ev.exit = res.exit
            ev.probe = res.probe
        return res

    def handle_stream(self, host: str, argv: list[str], cwd: str, pump: STREAM) -> Result:
        """ssh carrying git's pack protocol (push/fetch/clone with host:/path remotes).
        pump(target, command) runs the command on target with stdio streamed to the client;
        the returned Result holds only what was not streamed, plus the exit code."""
        prog = os.path.basename(argv[0]) if argv else ""
        self.tls.cwd = cwd
        self.sync_files()
        ev = self.event("stub", host, text=shlex.join(argv))
        try:
            if prog != "ssh":
                raise Gap(f"{prog} streaming")
            res = self._ssh_git(host, argv[1:], pump)
        except Gap as gap:
            res = self.gap(host, argv, str(gap))
        except Exception as exc:
            self.gap(host, argv, f"internal error {type(exc).__name__}: {exc}")
            res = Result(stderr=f"{prog}: internal error\n", exit=GAP_EXIT)
        if ev is not None:
            ev.exit = res.exit
        return res

    def _ssh_target(self, host: str, args: list[str]) -> tuple[str, str] | Result:
        """(target host, command) of an ssh call, or the error Result."""
        opts, pos = getopt(
            "ssh", args, short="tTqnANvxCG4V", short_valued="oipljFJ", stop_at_positional=True
        )
        if "V" in opts:
            return Result(
                stderr="OpenSSH_9.6p1 Ubuntu-3ubuntu13.5, OpenSSL 3.0.13 30 Jan 2024\n", probe=True
            )
        if not pos:
            raise Gap("ssh without a host")
        if "G" in opts:  # resolved client config; no ssh_config in these worlds
            user, _, dest = pos[0].rpartition("@")
            user = user or opts.get("l", "root")
            return Result(
                SSH_G.format(host=dest.lower(), hostname=dest, user=user, port=opts.get("p", "22")),
                probe=True,
            )
        dest = pos[0].rsplit("@", 1)[-1]
        target = self.resolve_host(dest, host)
        if target is None:
            return Result(
                stderr=f"ssh: Could not resolve hostname {dest}: Name or service not known\n",
                exit=255,
            )
        return target, " ".join(pos[1:])

    def _no_shell(self, target: str) -> Result | None:
        """GitHub's answer to anything but git over ssh."""
        if self.gh is None or target != self.gh.host:
            return None
        return Result(
            stderr=f"PTY allocation request failed on channel 0\nHi {self.gh.login}! "
            "You've successfully authenticated, but GitHub does not provide shell access.\n",
            exit=1,
        )

    def git_repo_path(self, host: str, path: str) -> str | None:
        """The repo a git-over-ssh path names: relative to /root, `.git` suffix optional."""
        base = path if path.startswith("/") else "/root/" + path
        for cand in (base, base + ".git", "/root" + base, "/root" + base + ".git"):
            real = self.real_path(host, cand)
            if (real / "HEAD").is_file() or (real / ".git").exists():
                return posixpath.normpath(cand)
        return None

    def git_refs(self, host: str, repo: str) -> dict[str, str]:
        real = self.real_path(host, repo)
        out = host_git(real, "for-each-ref", "--format=%(refname) %(objectname)").stdout
        return dict(line.split(" ", 1) for line in out.splitlines() if " " in line)

    def _ssh_git(self, host: str, args: list[str], pump: STREAM) -> Result:
        found = self._ssh_target(host, args)
        if isinstance(found, Result):
            return found
        target, command = found
        words = shlex.split(command)
        if words[:1] == ["git"] and len(words) > 1:
            words = [f"git-{words[1]}", *words[2:]]
        if not words or words[0] not in GIT_PACK_PROGS or len(words) != 2:
            return self._no_shell(target) or Result(
                stderr="ssh: interactive git transport needs a git pack command\n", exit=255
            )
        repo = self.git_repo_path(target, words[1])
        if repo is None:
            return Result(
                stderr=f"fatal: '{words[1]}' does not appear to be a git repository\n", exit=128
            )
        before = self.git_refs(target, repo) if words[0] == "git-receive-pack" else {}
        code = pump(target, f"{words[0]} {shlex.quote(repo)}")
        if words[0] == "git-receive-pack":
            after = self.git_refs(target, repo)
            self._record_push(target, repo, before, after)
            if self.gh is not None and target == self.gh.host:
                with self.lock:
                    self.gh.on_push(repo, before, after)
        return Result(exit=code)

    def _record_push(
        self, host: str, repo: str, before: dict[str, str], after: dict[str, str]
    ) -> None:
        real = self.real_path(host, repo)
        for ref in sorted(set(before) | set(after)):
            old, new = before.get(ref), after.get(ref)
            if old == new:
                continue
            if old is None:
                action = "create"
            elif new is None:
                action = "delete"
            else:
                ff = host_git(real, "merge-base", "--is-ancestor", old, new).returncode == 0
                action = "update" if ff else "force"
            self.event(
                "push",
                host,
                text=f"{host}:{repo} {ref}",
                action=action,
                mutation=True,
                paths=[f"{host}:{repo}"],
            )

    # ssh / pct -------------------------------------------------------------

    def _run_on(self, target: str, command: str, stdin: Stdin) -> Result:
        if self.exec_hook is None:
            return Result(stderr="ssh: remote execution unavailable\n", exit=255)
        return self.exec_hook(target, command, stdin)

    def _ssh(self, host: str, args: list[str], stdin: Stdin) -> Result:
        found = self._ssh_target(host, args)
        if isinstance(found, Result):
            return found
        target, command = found
        refused = self._no_shell(target)
        if refused is not None:
            return refused
        if not command:
            if not stdin:
                raise Gap("interactive ssh session (give a command)")
            command = "bash -s"
        return self._run_on(target, command, stdin)

    def _pct(self, host: str, args: list[str], stdin: Stdin) -> Result:
        if not self.hosts[host].hypervisor:
            return Result(stderr="bash: pct: command not found\n", exit=127)
        if not args:
            raise Gap("pct without a subcommand")
        sub, rest = args[0], args[1:]
        guests = {h.vmid: h for h in self.hosts.values() if h.vmid is not None}
        if sub == "list":
            lines = ["VMID       Status     Lock         Name"]
            for vmid in sorted(guests):
                lines.append(f"{vmid:<10} {'running':<10} {'':<12} {guests[vmid].name}")
            return Result("\n".join(lines) + "\n", probe=True)
        if sub in ("status", "exec"):
            if not rest or not rest[0].isdigit():
                raise Gap(f"pct {sub} without a vmid")
            guest = guests.get(int(rest[0]))
            if guest is None:
                return Result(
                    stderr=f"Configuration file 'nodes/{host}/lxc/{rest[0]}.conf' does not exist\n",
                    exit=2,
                )
            if sub == "status":
                return Result("status: running\n", probe=True)
            cmd = rest[1:]
            if cmd and cmd[0] == "--":
                cmd = cmd[1:]
            if not cmd:
                raise Gap("pct exec without a command")
            command = cmd[0] if len(cmd) == 1 else shlex.join(cmd)
            return self._run_on(guest.name, command, stdin)
        raise Gap(f"pct {sub}")

    # systemctl / service ---------------------------------------------------

    def _units(self, host: str) -> list[Service]:
        return [s for s in self.services if s.host == host and s.runtime == "systemd"]

    def _has_docker(self, host: str) -> bool:
        return any(s.host == host and s.runtime == "docker" for s in self.services) or any(
            h == host for h, _ in self.images
        )

    def _systemctl(self, host: str, args: list[str], stdin: str | None = None) -> Result:
        opts, pos = getopt(
            "systemctl",
            args,
            short="laqf",
            short_valued="tnps",
            long=frozenset(
                {
                    "no-pager",
                    "full",
                    "quiet",
                    "no-block",
                    "failed",
                    "all",
                    "plain",
                    "no-legend",
                    "now",
                    "system",
                    "value",
                    "version",
                    "force",
                }
            ),
            long_valued=frozenset(
                {"type", "state", "lines", "property", "signal", "kill-whom", "kill-who"}
            ),
            multi=frozenset({"p", "property"}),
        )
        if "version" in opts:
            return Result(SYSTEMD_VERSION, probe=True)
        if not pos:
            return self._list_units(host, opts)
        sub, names = pos[0], pos[1:]
        if sub in ("list-units",):
            return self._list_units(host, opts)
        if sub == "list-timers":  # timers are not simulated: none are loaded
            if "no-legend" in opts:
                return Result(probe=True)
            return Result(
                "NEXT LEFT LAST PASSED UNIT ACTIVATES\n\n0 timers listed.\nPass --all to see loaded but inactive timers, too.\n",
                probe=True,
            )
        if sub in ("daemon-reload", "daemon-reexec"):
            return Result()
        if sub == "list-unit-files":
            return self._list_unit_files(host, names, opts)
        if sub == "list-jobs":
            return Result("" if "no-legend" in opts else "No jobs running.\n", probe=True)
        if sub == "is-system-running":
            degraded = any(s.state == "failed" for s in self._units(host))
            return Result(
                "degraded\n" if degraded else "running\n", exit=1 if degraded else 0, probe=True
            )
        if sub in ("mask", "unmask"):
            if not names:
                return Result(stderr="Too few arguments.\n", exit=1)
            return self._mask(host, sub, names, "now" in opts)
        if sub == "kill":
            if not names:
                return Result(stderr="Too few arguments.\n", exit=1)
            return self._systemctl_kill(host, names, opts.get("s", opts.get("signal", "SIGTERM")))
        if sub in (
            "start",
            "stop",
            "restart",
            "reload",
            "try-restart",
            "reload-or-restart",
            "enable",
            "disable",
            "reset-failed",
        ):
            if not names and sub != "reset-failed":
                return Result(stderr="Too few arguments.\n", exit=1)
            return self._systemctl_action(host, sub, names, "now" in opts)
        if sub == "show":
            if not names:
                raise Gap("systemctl show without a unit (manager properties)")
            return Result("\n".join(self._systemctl_show(host, n, opts) for n in names), probe=True)
        if sub in ("status", "is-active", "is-failed", "is-enabled", "cat"):
            if not names:
                if sub == "status":
                    raise Gap("systemctl status without a unit")
                return Result(stderr="Too few arguments.\n", exit=1)
            return self._systemctl_query(host, sub, names, "q" in opts or "quiet" in opts)
        raise Gap(f"systemctl {sub}")

    def _systemctl_show(self, host: str, name: str, opts: dict[str, str]) -> str:
        """Property lines of one unit; unknown units show as not-found (or masked), like systemd."""
        props = self._unit_props(host, name)
        wanted = [
            w
            for v in (opts.get("p", ""), opts.get("property", ""))
            for w in v.replace(",", "\n").split()
            if w
        ]
        for key in wanted:
            if key not in props and key not in NOT_UNIT_PROPERTIES:
                raise Gap(f"systemctl show -p {key}")
        keys = [k for k in props if k in wanted] or list(props)  # systemd's order, not the -p order
        return "".join((props[k] if "value" in opts else f"{k}={props[k]}") + "\n" for k in keys)

    def _unit_props(self, host: str, name: str) -> dict[str, str]:
        unit = self._lookup_unit(host, name)
        uname = name if "." in name else f"{name}.service"
        masked = self._masked(host, uname)
        if isinstance(unit, Service):
            return self._service_props(unit, masked)
        # docker.service, or a unit that does not exist: same property set, systemd's values for it
        blank = dict.fromkeys(
            self._service_props(Service(host, uname.removesuffix(".service"), "app", ""), masked),
            "",
        )
        known = unit == "docker"
        return blank | {
            "Type": "notify" if known else "simple",
            "Restart": "always" if known else "no",
            "RestartUSec": "2s" if known else "100ms",
            "MainPID": "2211" if known else "0",
            "ControlPID": "0",
            "ExecMainPID": "2211" if known else "0",
            "ExecMainCode": "0",
            "ExecMainStatus": "0",
            "Result": "success",
            "NRestarts": "0",
            "Slice": "system.slice" if known else "",
            "ControlGroup": "/system.slice/docker.service" if known else "",
            "MemoryCurrent": "88000000" if known else "[not set]",
            "TasksCurrent": "21" if known else "[not set]",
            "UID": "[not set]",
            "GID": "[not set]",
            "KillMode": "process" if known else "control-group",
            "KillSignal": "15",
            "TimeoutStartUSec": "infinity" if known else "1min 30s",
            "TimeoutStopUSec": "1min 30s",
            "Id": uname,
            "Names": uname,
            "Description": "Docker Application Container Engine" if known else uname,
            "LoadState": "masked" if masked else "loaded" if known else "not-found",
            "ActiveState": "active" if known else "inactive",
            "SubState": "running" if known else "dead",
            "FragmentPath": "/lib/systemd/system/docker.service" if known and not masked else "",
            "UnitFileState": "masked" if masked else "enabled" if known else "",
            "UnitFilePreset": "enabled" if known else "",
            "TriggeredBy": "docker.socket" if known else "",
            "WantedBy": "multi-user.target" if known else "",
            "CanStart": "yes" if known and not masked else "no",
            "CanStop": "yes",
            "CanReload": "yes" if known else "no",
            "NeedDaemonReload": "no",
            "ConditionResult": "yes" if known else "no",
            "RefuseManualStart": "no",
            "RefuseManualStop": "no",
            "PrivateTmp": "no",
            "ProtectSystem": "no",
            "ProtectHome": "no",
            "NoNewPrivileges": "no",
            "LimitNOFILE": "524288",
            "MemoryMax": "infinity",
            "StandardInput": "null",
            "StandardOutput": "journal",
            "StandardError": "inherit",
        }

    def _service_props(self, svc: Service, masked: bool) -> dict[str, str]:
        active = svc.state == "active"

        def ts(t: datetime | None) -> str:
            return t.strftime("%a %Y-%m-%d %H:%M:%S UTC") if t else ""

        started = ts(svc.since) if active else ""
        ended = ts(svc.since) if not active else ""
        binary = KIND_DEFAULTS[svc.kind][1].split()[0]
        exec_start = f"{{ path={binary} ; argv[]={self._exec_line(svc)} ; ignore_errors=no ; start_time=[{started or 'n/a'}] ; stop_time=[n/a] ; pid={svc.pid if active else 0} ; code=(null) ; status=0/0 }}"
        return {
            "Type": {"proxy": "forking", "db": "notify"}.get(svc.kind, "simple"),
            "Restart": "on-failure",
            "RestartUSec": "100ms",
            "TimeoutStartUSec": "1min 30s",
            "TimeoutStopUSec": "1min 30s",
            "MainPID": str(svc.pid if active else 0),
            "ControlPID": "0",
            "Result": "exit-code" if svc.state == "failed" else "success",
            "NRestarts": "0",
            "ExecMainStartTimestamp": started,
            "ExecMainExitTimestamp": ended,
            "ExecMainPID": str(svc.pid if active else 0),
            "ExecMainCode": "0" if active else "1",
            "ExecMainStatus": str(svc.exit_status),
            "ExecStart": exec_start,
            "ExecStartPre": "",
            "ExecStartPost": "",
            "ExecReload": f"{{ path={binary} ; argv[]={binary} -s reload ; ignore_errors=no }}"
            if svc.kind == "proxy"
            else "",
            "ExecStop": "",
            "Slice": "system.slice",
            "ControlGroup": f"/system.slice/{svc.unit}" if active else "",
            "MemoryCurrent": "14336000" if active else "[not set]",
            "TasksCurrent": "3" if active else "[not set]",
            "Environment": "",
            "EnvironmentFiles": "",
            "WorkingDirectory": svc.workdir,
            "RootDirectory": "",
            "RootImage": "",
            "User": svc.user,
            "Group": "",
            "UID": "[not set]",
            "GID": "[not set]",
            "PrivateTmp": "no",
            "ProtectSystem": "no",
            "ProtectHome": "no",
            "NoNewPrivileges": "no",
            "StandardInput": "null",
            "StandardOutput": "journal",
            "StandardError": "inherit",
            "KillMode": "control-group",
            "KillSignal": "15",
            "LimitNOFILE": "524288",
            "MemoryMax": "infinity",
            "Id": svc.unit,
            "Names": svc.unit,
            "Requires": "system.slice sysinit.target",
            "Wants": "",
            "WantedBy": "multi-user.target" if svc.enabled and not masked else "",
            "After": "network.target system.slice sysinit.target basic.target systemd-journald.socket",
            "Before": "multi-user.target shutdown.target" if svc.enabled else "shutdown.target",
            "TriggeredBy": "",
            "Triggers": "",
            "Description": svc.description,
            "LoadState": "masked" if masked else "loaded",
            "ActiveState": svc.state,
            "SubState": {"active": "running", "failed": "failed", "inactive": "dead"}[svc.state],
            "FragmentPath": "" if masked else f"/lib/systemd/system/{svc.unit}",
            "DropInPaths": "",
            "SourcePath": "",
            "UnitFileState": "masked" if masked else "enabled" if svc.enabled else "disabled",
            "UnitFilePreset": "enabled",
            "StateChangeTimestamp": ts(svc.since),
            "ActiveEnterTimestamp": started,
            "ActiveExitTimestamp": ended,
            "InactiveEnterTimestamp": ended,
            "InactiveExitTimestamp": started,
            "CanStart": "no" if masked else "yes",
            "CanStop": "yes",
            "CanReload": "yes" if svc.kind in ("proxy", "db") else "no",
            "NeedDaemonReload": "no",
            "ConditionResult": "yes",
            "RefuseManualStart": "no",
            "RefuseManualStop": "no",
        }

    def _masked(self, host: str, unit: str) -> bool:
        """systemctl mask: a /dev/null symlink in /etc/systemd/system (the model can also make or remove it)."""
        link = self.hosts_dir / host / MASK_DIR.lstrip("/") / unit
        return link.is_symlink() and os.readlink(link) == "/dev/null"

    def _mask(self, host: str, sub: str, names: list[str], now: bool) -> Result:
        out = Result()
        for name in names:
            uname = name if "." in name else f"{name}.service"
            link = self.hosts_dir / host / MASK_DIR.lstrip("/") / uname
            if sub == "mask":
                if not self._masked(host, uname):
                    if os.path.lexists(link):
                        out.stderr += (
                            f"Failed to mask unit: File {MASK_DIR}/{uname} already exists.\n"
                        )
                        out.exit = 1
                        continue
                    link.parent.mkdir(parents=True, exist_ok=True)
                    link.symlink_to("/dev/null")
                    out.stderr += f"Created symlink {MASK_DIR}/{uname} → /dev/null.\n"
                unit = self._lookup_unit(host, name)
                if now and isinstance(unit, Service) and unit.state == "active":
                    self.transition(unit, "stop")
            elif self._masked(host, uname):
                link.unlink()
                out.stderr += f'Removed "{MASK_DIR}/{uname}".\n'
        return out

    def _systemctl_kill(self, host: str, names: list[str], signal: str) -> Result:
        """Signals as systemd's Restart=on-failure sees them: TERM/INT end the service cleanly,
        KILL/QUIT fail it and it is restarted, HUP reloads nginx and postgres and ends an app."""
        sig = signal.upper().removeprefix("SIG")
        sig = {"1": "HUP", "2": "INT", "3": "QUIT", "9": "KILL", "15": "TERM"}.get(sig, sig)
        if sig not in ("HUP", "INT", "QUIT", "KILL", "TERM"):
            raise Gap(f"systemctl kill -s {signal}")
        out = Result()
        for name in names:
            unit = self._lookup_unit(host, name)
            uname = name if "." in name else f"{name}.service"
            if not isinstance(unit, Service):
                if unit is None:
                    out.stderr += f"Failed to kill unit {uname}: Unit {uname} not loaded.\n"
                    out.exit = 1
                continue
            if unit.state != "active":
                continue
            if sig == "HUP" and unit.kind in ("proxy", "db"):
                self.transition(unit, "reload")
            elif sig in ("TERM", "INT", "HUP"):
                self.transition(unit, "stop")
            else:
                self.syslog(
                    unit,
                    f"{unit.unit}: Main process exited, code=killed, status={ {'KILL': 9, 'QUIT': 3}[sig] }/{sig}",
                )
                self.syslog(unit, f"{unit.unit}: Failed with result 'signal'.")
                self.syslog(unit, f"{unit.unit}: Scheduled restart job, restart counter is at 1.")
                self.transition(unit, "restart")
        return out

    def _list_unit_files(self, host: str, patterns: list[str], opts: dict[str, str]) -> Result:
        typ = opts.get("type", opts.get("t", "service")).removesuffix("s")
        if typ not in ("service", "timer", "socket", "target", "path", "mount"):
            raise Gap(f"systemctl list-unit-files --type={typ}")
        states = [x for x in opts.get("state", "").split(",") if x]
        rows: dict[str, str] = {}
        if typ == "service":
            for svc in self._units(host):
                rows[svc.unit] = "enabled" if svc.enabled else "disabled"
            if self._has_docker(host):
                rows["docker.service"] = "enabled"
            mask_dir = self.hosts_dir / host / MASK_DIR.lstrip("/")
            if mask_dir.is_dir():
                for entry in mask_dir.iterdir():
                    if entry.name.endswith(".service") and self._masked(host, entry.name):
                        rows[entry.name] = "masked"
        picked = sorted(
            (u, st)
            for u, st in rows.items()
            if (
                not patterns
                or any(
                    fnmatch.fnmatch(u, pat) or fnmatch.fnmatch(u, pat + ".service")
                    for pat in patterns
                )
            )
            and (not states or st in states)
        )
        width = max([len(u) for u, _ in picked] + [9]) + 1
        lines = [] if "no-legend" in opts else [f"{'UNIT FILE':<{width}}STATE    PRESET"]
        lines += [f"{u:<{width}}{st:<9}enabled" for u, st in picked]
        if "no-legend" not in opts:
            lines += ["", f"{len(picked)} unit files listed."]
        return Result("".join(line + "\n" for line in lines), probe=True)

    def _lookup_unit(self, host: str, name: str) -> Service | str | None:
        base = name.removesuffix(".service")
        if base == "docker" and self._has_docker(host):
            return "docker"
        return self.find(host, base, "systemd")

    def _systemctl_action(self, host: str, sub: str, names: list[str], now: bool) -> Result:
        out = Result()
        for name in names:
            unit = self._lookup_unit(host, name)
            uname = name if name.endswith(".service") else f"{name}.service"
            if unit is None:
                verb = {"try-restart": "restart", "reload-or-restart": "restart"}.get(sub, sub)
                out.stderr += f"Failed to {verb} {uname}: Unit {uname} not found.\n"
                out.exit = 5
                continue
            if isinstance(unit, str):  # "docker"
                if sub in ("restart", "start", "stop", "try-restart", "reload-or-restart"):
                    action = "stop" if sub == "stop" else "start" if sub == "start" else "restart"
                    for svc in self._containers(host):
                        self.transition(svc, action)
                continue
            svc = unit
            if self._masked(host, svc.unit) and sub not in ("disable", "reset-failed", "stop"):
                verb = {"try-restart": "restart", "reload-or-restart": "restart"}.get(sub, sub)
                out.stderr += (
                    f"Failed to enable unit: Unit file {MASK_DIR}/{svc.unit} is masked.\n"
                    if sub == "enable"
                    else f"Failed to {verb} {svc.unit}: Unit {svc.unit} is masked.\n"
                )
                out.exit = 1
                continue
            if sub in ("enable", "disable"):
                svc.enabled = sub == "enable"
                if now:
                    self.transition(svc, "start" if sub == "enable" else "stop")
                continue
            if sub == "reset-failed":
                if svc.state == "failed":
                    svc.state = "inactive"
                continue
            action = {"try-restart": "restart", "reload-or-restart": "restart"}.get(sub, sub)
            if sub == "try-restart" and svc.state != "active":
                continue
            if sub == "reload-or-restart" and svc.state == "active":
                action = "reload"
            error = self.transition(svc, action)
            if error:
                job = "reload" if action == "reload" else action
                if action == "reload" and svc.state != "active":
                    out.stderr += f"{svc.unit} is not active, cannot reload.\n"
                else:
                    out.stderr += (
                        f"Job for {svc.unit} failed because the control process exited with error code.\n"
                        f'See "systemctl status {svc.unit}" and "journalctl -xeu {svc.unit}" for details.\n'
                    )
                _ = job
                out.exit = 1
        return out

    def _ago(self, since: datetime | None) -> str:
        if since is None:
            return ""
        secs = int((self.now() - since).total_seconds())
        if secs < 60:
            return f"{secs}s ago"
        if secs < 3600:
            return f"{secs // 60}min {secs % 60}s ago"
        return f"{secs // 3600}h {secs % 3600 // 60}min ago"

    def _exec_line(self, svc: Service) -> str:
        base = KIND_DEFAULTS[svc.kind][1]
        if svc.kind == "proxy":
            return f"{base} -g 'daemon on; master_process on;'"
        if svc.kind == "db":
            return f"{base} -c config_file={svc.config}"
        return f"{base} --env-file {svc.config}"

    def _status_text(self, svc: Service) -> str:
        enabled = "enabled" if svc.enabled else "disabled"
        since = svc.since.strftime("%a %Y-%m-%d %H:%M:%S UTC") if svc.since else ""
        if svc.state == "active":
            active = f"active (running) since {since}; {self._ago(svc.since)}"
            dot = "●"
            pid = f"   Main PID: {svc.pid} ({os.path.basename(KIND_DEFAULTS[svc.kind][1].split()[0])})"
        elif svc.state == "failed":
            active = f"failed (Result: exit-code) since {since}; {self._ago(svc.since)}"
            dot = "×"
            pid = f"   Main PID: {svc.pid} (code=exited, status={svc.exit_status}/FAILURE)"
        else:
            active = "inactive (dead)" + (
                f" since {since}; {self._ago(svc.since)}" if since else ""
            )
            dot = "○"
            pid = ""
        loaded = (
            f"masked (Reason: Unit {svc.unit} is masked.)"
            if self._masked(svc.host, svc.unit)
            else f"loaded (/lib/systemd/system/{svc.unit}; {enabled}; preset: enabled)"
        )
        lines = [
            f"{dot} {svc.unit} - {svc.description}",
            f"     Loaded: {loaded}",
            f"     Active: {active}",
        ]
        if pid:
            lines.append(pid)
        tail = [line for ts, unit, line in self.journal[svc.host] if unit == svc.name][-10:]
        stamp = [ts for ts, unit, _ in self.journal[svc.host] if unit == svc.name][-10:]
        if tail:
            lines.append("")
            for ts, line in zip(stamp, tail, strict=True):
                lines.append(f"{ts:%b %d %H:%M:%S} {svc.host} {line}")
        return "\n".join(lines) + "\n"

    def _systemctl_query(self, host: str, sub: str, names: list[str], quiet: bool) -> Result:
        out = Result(probe=True)
        worst = 0
        for name in names:
            unit = self._lookup_unit(host, name)
            uname = name if name.endswith(".service") else f"{name}.service"
            if isinstance(unit, str):  # "docker"
                if sub == "status":
                    out.stdout += "● docker.service - Docker Application Container Engine\n     Loaded: loaded (/lib/systemd/system/docker.service; enabled; preset: enabled)\n     Active: active (running)\n"
                elif sub == "is-active":
                    out.stdout += "active\n"
                elif sub == "is-enabled":
                    out.stdout += "enabled\n"
                elif sub == "is-failed":
                    out.stdout += "active\n"
                    worst = max(worst, 1)
                continue
            if unit is None:
                if sub == "status":
                    out.stderr += f"Unit {uname} could not be found.\n"
                    worst = 4
                elif sub == "cat":
                    out.stderr += f"No files found for {uname}.\n"
                    worst = 1
                else:
                    out.stdout += "inactive\n" if sub != "is-enabled" else ""
                    if sub == "is-enabled":
                        out.stderr += f"Failed to get unit file state for {uname}: No such file or directory\n"
                    worst = max(worst, 4 if sub == "is-enabled" else 3 if sub == "is-active" else 1)
                continue
            svc = unit
            if sub == "status":
                out.stdout += self._status_text(svc)
                if svc.state != "active":
                    worst = max(worst, 3)
            elif sub == "is-active":
                out.stdout += f"{svc.state}\n"
                if svc.state != "active":
                    worst = max(worst, 3)
            elif sub == "is-failed":
                out.stdout += f"{svc.state}\n"
                if svc.state != "failed":
                    worst = max(worst, 1)
            elif sub == "is-enabled":
                masked = self._masked(host, svc.unit)
                out.stdout += "masked\n" if masked else "enabled\n" if svc.enabled else "disabled\n"
                if masked or not svc.enabled:
                    worst = max(worst, 1)
            elif sub == "cat":
                extra = (f"User={svc.user}\n" if svc.user else "") + (
                    f"WorkingDirectory={svc.workdir}\n" if svc.workdir else ""
                )
                out.stdout += (
                    f"# /lib/systemd/system/{svc.unit}\n[Unit]\nDescription={svc.description}\n"
                    f"After=network.target\n\n[Service]\n{extra}ExecStart={self._exec_line(svc)}\n"
                    "Restart=on-failure\n\n[Install]\nWantedBy=multi-user.target\n"
                )
        if quiet:
            out.stdout = ""
        out.exit = worst
        return out

    def _list_units(self, host: str, opts: dict[str, str]) -> Result:
        types = {t.removesuffix("s") for t in opts.get("type", opts.get("t", "service")).split(",")}
        if not types <= {"service", "timer", "socket", "target", "path", "mount"}:
            raise Gap(f"systemctl --type={','.join(sorted(types))}")
        if "service" not in types:  # only services are simulated
            return Result(
                ""
                if "no-legend" in opts
                else "  UNIT LOAD ACTIVE SUB DESCRIPTION\n\n0 loaded units listed.\n",
                probe=True,
            )
        state = opts.get("state", "")
        rows: list[tuple[str, str, str, str, str]] = []
        units = list(self._units(host))
        for svc in units:
            sub_state = {"active": "running", "failed": "failed", "inactive": "dead"}[svc.state]
            if "failed" in opts and svc.state != "failed":
                continue
            if state and state not in (svc.state, sub_state):
                continue
            if svc.state == "inactive" and not ("all" in opts or "a" in opts or state):
                continue
            rows.append(
                (
                    svc.unit,
                    "masked" if self._masked(host, svc.unit) else "loaded",
                    svc.state,
                    sub_state,
                    svc.description,
                )
            )
        if (
            self._has_docker(host)
            and "failed" not in opts
            and (not state or state in ("active", "running"))
        ):
            rows.append(
                (
                    "docker.service",
                    "loaded",
                    "active",
                    "running",
                    "Docker Application Container Engine",
                )
            )
        rows.sort()
        lines = []
        legend = "no-legend" not in opts
        width = max([len(r[0]) for r in rows] + [4]) + 2
        if legend:
            lines.append(f"  {'UNIT':<{width}}LOAD   ACTIVE SUB     DESCRIPTION")
        for unit, load, active, sub, desc in rows:
            mark = "●" if active == "failed" and "plain" not in opts else " "
            lines.append(f"{mark} {unit:<{width}}{load:<7}{active:<7}{sub:<8}{desc}")
        if legend:
            lines.append("")
            lines.append(f"{len(rows)} loaded units listed.")
        return Result("\n".join(lines) + "\n", probe=True)

    def _service(self, host: str, args: list[str], stdin: str | None = None) -> Result:
        if args == ["--status-all"]:
            units = sorted((s.name, s.state == "active") for s in self._units(host))
            return Result(
                "".join(f" [ {'+' if up else '-'} ]  {name}\n" for name, up in units), probe=True
            )
        if len(args) != 2:
            raise Gap("service " + " ".join(args))
        name, action = args
        if action not in ("start", "stop", "restart", "reload", "status"):
            raise Gap(f"service {action}")
        return self._systemctl(host, [action, name])

    # journalctl --------------------------------------------------------------

    def _journalctl(self, host: str, args: list[str], stdin: str | None = None) -> Result:
        boot = "0"
        rest: list[str] = []
        for i, a in enumerate(args):  # -b takes an optional value: -b, -b 0, -b -1, --boot=-1
            prev = args[i - 1] if i else ""
            if prev in ("-b", "--boot") and re.fullmatch(r"[-+]?\d+|all|[0-9a-f]{32}", a):
                boot = a
            elif a.startswith("--boot="):
                boot = a.partition("=")[2]
            else:
                rest.append(a)
        opts, pos = getopt(
            "journalctl",
            rest,
            short="exbqlram",
            short_valued="unotSUpgt",
            long=frozenset(
                {
                    "no-pager",
                    "pager-end",
                    "catalog",
                    "boot",
                    "quiet",
                    "full",
                    "reverse",
                    "no-hostname",
                    "utc",
                    "disk-usage",
                    "list-boots",
                    "system",
                    "all",
                    "merge",
                    "case-sensitive",
                }
            ),
            long_valued=frozenset(
                {
                    "unit",
                    "lines",
                    "output",
                    "since",
                    "until",
                    "identifier",
                    "priority",
                    "grep",
                    "vacuum-size",
                    "vacuum-time",
                    "vacuum-files",
                }
            ),
        )
        if pos:
            raise Gap("journalctl match arguments " + " ".join(pos))
        if "disk-usage" in opts:
            return Result(
                "Archived and active journals take up 48.0M in the file system.\n", probe=True
            )
        if "list-boots" in opts:
            first = self.journal[host][0][0] if self.journal[host] else self.now()
            boot_id = hashlib.sha256(host.encode()).hexdigest()[:32]
            return Result(
                f"IDX BOOT ID                          FIRST ENTRY                 LAST ENTRY\n  0 {boot_id} {first:%a %Y-%m-%d %H:%M:%S} UTC {self.now():%a %Y-%m-%d %H:%M:%S} UTC\n",
                probe=True,
            )
        vacuum = next(
            (k for k in ("vacuum-size", "vacuum-time", "vacuum-files") if k in opts), None
        )
        if vacuum:
            return Result(
                stderr="Vacuuming done, freed 0B of archived journals from /var/log/journal/"
                + hashlib.sha256(host.encode()).hexdigest()[:32]
                + ".\n"
            )
        if boot not in ("0", "all") and not re.fullmatch(r"[0-9a-f]{32}", boot):
            return Result(
                stderr=f"Data from the specified boot ({boot}) is not available: No such boot ID in journal\n",
                exit=1,
            )
        unit = opts.get("u", opts.get("unit", "")).removesuffix(".service")
        output = opts.get("o", opts.get("output", "short"))
        if output not in ("short", "cat", "short-iso"):
            raise Gap(f"journalctl -o {output}")
        entries = self.journal[host]
        since_s, until_s = opts.get("S", opts.get("since")), opts.get("U", opts.get("until"))
        if since_s is not None:
            since = parse_journal_time(since_s, self.now())
            entries = [e for e in entries if e[0] >= since]
        if until_s is not None:
            until = parse_journal_time(until_s, self.now())
            entries = [e for e in entries if e[0] <= until]
        prio = opts.get("p", opts.get("priority"))
        if prio is not None:
            lo, _, hi = prio.partition("..")
            top, bottom = _priority(hi or lo), (_priority(lo) if hi else 0)
            entries = [
                e
                for e in entries
                if bottom <= _line_priority(e[2]) <= top or top <= _line_priority(e[2]) <= bottom
            ]
        pattern = opts.get("g", opts.get("grep"))
        if pattern is not None:
            flags = (
                0
                if any(c.isupper() for c in pattern) or "case-sensitive" in opts
                else re.IGNORECASE
            )
            try:
                rx = re.compile(pattern, flags)
            except re.error:
                return Result(stderr=f'Failed to compile pattern "{pattern}"\n', exit=1)
            entries = [e for e in entries if rx.search(e[2].split(": ", 1)[-1])]
        if unit:
            if unit == "docker" and self._has_docker(host):
                entries = []
            elif self.find(host, unit, "systemd") is None:
                return Result("-- No entries --\n", probe=True)
            else:
                entries = [e for e in entries if e[1] == unit]
        n_s = opts.get("n", opts.get("lines", ""))
        if "e" in opts or "pager-end" in opts:
            n_s = n_s or "1000"
        if n_s:
            if not n_s.isdigit():
                raise Gap(f"journalctl -n {n_s}")
            entries = entries[-int(n_s) :] if int(n_s) else []
        if "r" in opts or "reverse" in opts:
            entries = list(reversed(entries))
        if not entries:
            return Result("-- No entries --\n", probe=True)
        lines = []
        for ts, _, line in entries:
            if output == "cat":
                lines.append(line.split(": ", 1)[-1])
            elif output == "short-iso":
                lines.append(f"{ts:%Y-%m-%dT%H:%M:%S+0000} {host} {line}")
            else:
                lines.append(f"{ts:%b %d %H:%M:%S} {host} {line}")
        return Result("\n".join(lines) + "\n", probe=True)

    # docker ----------------------------------------------------------------

    def _containers(self, host: str) -> list[Service]:
        """Existing containers on host (removed ones are kept as services but not listed)."""
        return [s for s in self.services if s.host == host and s.runtime == "docker" and s.present]

    def _cid(self, svc: Service, full: bool = False) -> str:
        key = f"{svc.host}/{svc.name}" + (f"/{svc.generation}" if svc.generation else "")
        digest = hashlib.sha256(key.encode()).hexdigest()
        return digest if full else digest[:12]

    def _docker_status(self, svc: Service) -> str:
        if svc.since is None:
            return "Created"
        ago = self._ago(svc.since).replace("min ", " minutes ").replace("s ago", " seconds ago")
        if svc.state == "active":
            return f"Up {ago.removesuffix(' ago')}"
        return f"Exited ({svc.exit_status}) {ago}"

    def _docker_ports(self, svc: Service) -> str:
        if svc.network == "host":
            return ", ".join(f"0.0.0.0:{p}->{p}/tcp" for p in svc.ports())
        if svc.state != "active":
            return ""
        return ", ".join(f"0.0.0.0:{h}->{c}/tcp" for c, h in sorted(svc.publish.items()))

    def _docker(self, host: str, args: list[str], stdin: Stdin = None) -> Result:
        if not self._has_docker(host):
            return Result(stderr="bash: docker: command not found\n", exit=127)
        if args in (["--version"], ["-v"]):
            return Result(f"Docker version {DOCKER_VERSION}, build 4c9b3b0\n", probe=True)
        if not args or args in (["--help"], ["-h"], ["help"]):
            return Result(DOCKER_HELP.lstrip("\n"), probe=True)
        sub, rest = args[0], args[1:]
        if "--help" in rest or (sub == "help" and rest):
            name = rest[0] if sub == "help" else sub
            if name not in DOCKER_USAGE:
                raise Gap(f"docker {name} --help")
            return Result(f"\nUsage:  {DOCKER_USAGE[name]}\n", probe=True)
        if sub in ("run", "create"):
            return self._docker_run(host, sub, rest)
        if sub == "rm" or (sub == "container" and rest[:1] == ["rm"]):
            return self._docker_rm(host, rest[1:] if sub == "container" else rest)
        if sub == "exec":
            return self._docker_exec(host, rest, stdin)
        if sub == "compose":
            return self._compose(host, rest)
        if sub == "container" and rest and rest[0] in ("ls", "ps", "list", "inspect"):
            sub, rest = ("ps" if rest[0] != "inspect" else "inspect"), rest[1:]
        if sub == "image" and rest and rest[0] in ("ls", "list", "inspect", "pull"):
            sub, rest = (
                {"ls": "images", "list": "images", "pull": "pull"}.get(rest[0], "image-inspect"),
                rest[1:],
            )
        if sub == "pull":
            return Result(
                stderr='Error response from daemon: Get "https://registry-1.docker.io/v2/": dial tcp: '
                "lookup registry-1.docker.io on 127.0.0.53:53: no such host\n",
                exit=1,
            )
        if sub == "images":
            return self._docker_images(host, rest)
        if sub == "image-inspect":
            return self._docker_image_inspect(host, rest)
        if sub in ("volume", "network"):
            return self._docker_objects(host, sub, rest)
        if sub == "system" and rest[:1] == ["df"] and len(rest) == 1:
            return self._docker_system_df(host)
        if sub == "info" or (sub == "system" and rest[:1] == ["info"]):
            return self._docker_info(host, rest[1:] if sub == "system" else rest)
        if sub == "context" and rest in (["ls"], ["list"], ["show"]):
            if rest == ["show"]:
                return Result("default\n", probe=True)
            return Result(
                "NAME        DESCRIPTION                               DOCKER ENDPOINT               ERROR\n"
                "default *   Current DOCKER_HOST based configuration   unix:///var/run/docker.sock\n",
                probe=True,
            )
        if sub == "ps":
            opts, pos = getopt(
                "docker ps",
                rest,
                short="aqs",
                long=frozenset({"all", "quiet", "no-trunc", "size"}),
                long_valued=frozenset({"format", "filter"}),
                short_valued="f",
                multi=frozenset({"f", "filter"}),
            )
            if pos:
                raise Gap("docker ps " + " ".join(pos))
            show_all = "a" in opts or "all" in opts
            rows = [s for s in self._containers(host) if show_all or s.state == "active"]
            rows = self._docker_filter(rows, opts.get("f", opts.get("filter", "")))
            if "q" in opts or "quiet" in opts:
                return Result("".join(self._cid(s) + "\n" for s in rows), probe=True)
            fmt = opts.get("format")
            if fmt:
                headers = {
                    "ID": "CONTAINER ID",
                    "Names": "NAMES",
                    "Image": "IMAGE",
                    "Status": "STATUS",
                    "Ports": "PORTS",
                    "State": "STATE",
                    "Command": "COMMAND",
                    "CreatedAt": "CREATED AT",
                    "RunningFor": "CREATED",
                    "Size": "SIZE",
                }
                return Result(
                    docker_format(fmt, [self._ps_doc(s) for s in rows], headers), probe=True
                )
            lines = [
                f"{'CONTAINER ID':<15}{'IMAGE':<28}{'COMMAND':<24}{'STATUS':<26}{'PORTS':<28}NAMES"
            ]
            for s in rows:
                ports = self._docker_ports(s)
                lines.append(
                    f"{self._cid(s):<15}{s.image:<28}{'"uvicorn app:app"':<24}{self._docker_status(s):<26}{ports:<28}{s.name}"
                )
            return Result("\n".join(lines) + "\n", probe=True)
        if sub == "logs":
            opts, pos = getopt(
                "docker logs",
                rest,
                short="tf",
                short_valued="n",
                long=frozenset({"timestamps", "follow"}),
                long_valued=frozenset({"tail", "since"}),
            )
            if "f" in opts or "follow" in opts:
                raise Gap("docker logs --follow")
            if len(pos) != 1:
                raise Gap("docker logs needs one container")
            svc = self._container(host, pos[0])
            if svc is None:
                return Result(
                    stderr=f"Error response from daemon: No such container: {pos[0]}\n", exit=1
                )
            lines = self.container_logs.get((host, svc.name), [])
            tail = opts.get("n", opts.get("tail", "all"))
            if tail != "all":
                if not tail.isdigit():
                    raise Gap(f"docker logs --tail {tail}")
                lines = lines[-int(tail) :] if int(tail) else []
            return Result("".join(line + "\n" for line in lines), probe=True)
        if sub in ("restart", "start", "stop"):
            opts, pos = getopt(
                f"docker {sub}", rest, short_valued="t", long_valued=frozenset({"time"})
            )
            out = Result()
            for name in pos:
                svc = self._container(host, name)
                if svc is None:
                    out.stderr += f"Error response from daemon: No such container: {name}\n"
                    out.exit = 1
                    continue
                self.transition(svc, sub)
                out.stdout += name + "\n"
            return out
        if sub == "version":
            return Result(
                f"Client: Docker Engine - Community\n Version:           {DOCKER_VERSION}\n\nServer: Docker Engine - Community\n Engine:\n  Version:          {DOCKER_VERSION}\n",
                probe=True,
            )
        if sub == "port":
            if len(rest) != 1:
                raise Gap("docker port with a port filter")
            if self._container(host, rest[0]) is None:
                return Result(
                    stderr=f"Error response from daemon: No such container: {rest[0]}\n", exit=1
                )
            return Result(probe=True)  # containers run with network_mode: host
        if sub == "inspect":
            opts, pos = getopt(
                "docker inspect",
                rest,
                long_valued=frozenset({"format", "type"}),
                short_valued="f",
                short="s",
                long=frozenset({"size"}),
            )
            fmt = opts.get("f", opts.get("format"))
            docs: list[dict[str, object]] = []
            for name in pos:
                svc = self._container(host, name)
                if svc is None:
                    return Result("[]\n", stderr=f"Error: No such object: {name}\n", exit=1)
                docs.append(self._inspect_doc(svc))
            if fmt is not None:
                return Result("".join(go_template(fmt, d) + "\n" for d in docs), probe=True)
            return Result(json.dumps(docs, indent=4) + "\n", probe=True)
        raise Gap(f"docker {sub}")

    def _container_env(self, svc: Service) -> list[str]:
        """Config.Env: the env as of the last create/start; edits to the file show after a restart."""
        return svc.started_env if svc.started_env is not None else self._env_file_lines(svc)

    def _env_file_lines(self, svc: Service) -> list[str]:
        """KEY=VALUE lines of the container's env file as it is on disk now, then -e values."""
        text = (self.read(svc.host, svc.config) or "") if svc.config and svc.kind == "app" else ""
        lines = [line.strip() for line in text.splitlines() if ENV_LINE.match(line.strip())]
        return lines + [f"{k}={v}" for k, v in svc.env.items()]

    def _image(self, host: str, image: str) -> Image | None:
        ref = _image_ref(image)
        if (host, ref) in self.images:
            return self.images[(host, ref)]
        for s in self.services:
            if (
                s.host == host
                and s.runtime == "docker"
                and s.declared[1]
                and _image_ref(s.declared[1]) == ref
            ):
                return Image(s.kind, dict(s.routes), s.declared[0])
        return None

    @staticmethod
    def _iso(t: datetime | None) -> str:
        return f"{t:%Y-%m-%dT%H:%M:%S}.{t.microsecond:06d}000Z" if t else "0001-01-01T00:00:00Z"

    def _container_cmd(self, svc: Service) -> tuple[list[str], list[str]]:
        """(Config.Cmd, Config.Entrypoint) of the image's default command."""
        if svc.kind == "proxy":
            return ["nginx", "-g", "daemon off;"], ["/docker-entrypoint.sh"]
        if svc.kind == "db":
            return ["postgres"], ["docker-entrypoint.sh"]
        port = svc.listen_ports()[0] if svc.listen_ports() else 8000
        return ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", str(port)], []

    def _labels(self, svc: Service) -> dict[str, str]:
        if not svc.compose:
            return {}
        return {
            "com.docker.compose.config-hash": hashlib.sha256(svc.name.encode()).hexdigest(),
            "com.docker.compose.project": posixpath.basename(svc.compose),
            "com.docker.compose.project.config_files": f"{svc.compose}/docker-compose.yml",
            "com.docker.compose.project.working_dir": svc.compose,
            "com.docker.compose.service": svc.name,
        }

    def _mounts(self, svc: Service) -> list[dict[str, object]]:
        binds = svc.volumes or ([f"{svc.config}:{svc.config}:ro"] if svc.config else [])
        out: list[dict[str, object]] = []
        for b in binds:
            src, _, rest = b.partition(":")
            dst, _, mode = rest.partition(":")
            if not dst:
                continue
            named = not src.startswith("/")
            out.append(
                {
                    "Type": "volume" if named else "bind",
                    **({"Name": src} if named else {}),
                    "Source": f"/var/lib/docker/volumes/{src}/_data" if named else src,
                    "Destination": dst,
                    "Driver": "local" if named else "",
                    "Mode": mode if mode else ("z" if named else ""),
                    "RW": "ro" not in mode.split(","),
                    "Propagation": "" if named else "rprivate",
                }
            )
        return out

    def _inspect_doc(self, svc: Service) -> dict[str, object]:
        """docker inspect of a container (the fields people format, Docker's names and types)."""
        binds = svc.volumes or ([f"{svc.config}:{svc.config}:ro"] if svc.config else [])
        ports = {
            f"{c}/tcp": [{"HostIp": "", "HostPort": str(h)}] for c, h in sorted(svc.publish.items())
        }
        cmd, entry = self._container_cmd(svc)
        cid = self._cid(svc, full=True)
        running = svc.state == "active"
        created = (
            min(svc.since, self.now() - timedelta(days=2))
            if svc.since
            else self.now() - timedelta(days=2)
        )
        return {
            "Id": cid,
            "Created": self._iso(created),
            "Path": (entry or cmd)[0],
            "Args": (entry[1:] + cmd) if entry else cmd[1:],
            "State": {
                "Status": "running" if running else "exited",
                "Running": running,
                "Paused": False,
                "Restarting": False,
                "OOMKilled": False,
                "Dead": False,
                "Pid": svc.pid if running else 0,
                "ExitCode": svc.exit_status,
                "Error": "",
                "StartedAt": self._iso(svc.since),
                "FinishedAt": "0001-01-01T00:00:00Z" if running else self._iso(svc.since),
            },
            "Image": "sha256:" + hashlib.sha256(_image_ref(svc.image).encode()).hexdigest(),
            "LogPath": f"/var/lib/docker/containers/{cid}/{cid}-json.log",
            "Name": "/" + svc.name,
            "RestartCount": 0,
            "Driver": "overlay2",
            "Platform": "linux",
            "Config": {
                "Hostname": cid[:12],
                "User": "",
                "Env": self._container_env(svc),
                "Cmd": cmd,
                "Entrypoint": entry,
                "Image": svc.image,
                "WorkingDir": "/app" if svc.kind == "app" else "",
                "Labels": self._labels(svc),
                "ExposedPorts": {f"{p}/tcp": {} for p in self._container_cmd_ports(svc)},
            },
            "HostConfig": {
                "Binds": binds,
                "NetworkMode": svc.network,
                "PortBindings": ports,
                "RestartPolicy": {"Name": "unless-stopped", "MaximumRetryCount": 0},
                "LogConfig": {"Type": "json-file", "Config": {}},
                "Privileged": False,
                "Memory": 0,
            },
            "NetworkSettings": {
                "Networks": {svc.network: {"IPAddress": ""}},
                "Ports": ports,
                "IPAddress": "",
            },
            "Mounts": self._mounts(svc),
        }

    def _container_cmd_ports(self, svc: Service) -> list[int]:
        return sorted(set(svc.publish) | set(svc.listen_ports())) if svc.network != "host" else []

    def _ps_doc(self, svc: Service) -> dict[str, object]:
        """Fields of one `docker ps --format` row."""
        cmd, entry = self._container_cmd(svc)
        command = " ".join(entry + cmd)
        created = svc.since or self.now()
        return {
            "ID": self._cid(svc),
            "Names": svc.name,
            "Image": svc.image,
            "Command": json.dumps(
                command if len(command) <= 20 else command[:19] + "…", ensure_ascii=False
            ),
            "CreatedAt": f"{created:%Y-%m-%d %H:%M:%S} +0000 UTC",
            "RunningFor": self._ago(created)
            .replace("min ", " minutes ")
            .replace("s ago", " seconds ago"),
            "Status": self._docker_status(svc),
            "State": "running" if svc.state == "active" else "exited",
            "Ports": self._docker_ports(svc),
            "Size": "0B (virtual 187MB)",
            "Labels": ",".join(f"{k}={v}" for k, v in self._labels(svc).items()),
            "Mounts": ",".join(str(m.get("Name", m["Source"])) for m in self._mounts(svc)),
            "Networks": svc.network,
            "LocalVolumes": str(sum(1 for m in self._mounts(svc) if m["Type"] == "volume")),
        }

    def _docker_filter(self, rows: list[Service], filters: str) -> list[Service]:
        """docker ps --filter key=value (name, id, status, ancestor, label): AND across keys, OR within one."""
        wanted: dict[str, list[str]] = {}
        for f in filter(None, filters.split("\n")):
            key, eq, value = f.partition("=")
            if not eq or key not in ("name", "id", "status", "ancestor", "label"):
                raise Gap(f"docker ps --filter {f}")
            wanted.setdefault(key, []).append(value)

        def match(s: Service, key: str, value: str) -> bool:
            doc = self._ps_doc(s)
            if key == "name":
                try:
                    return re.search(value, s.name) is not None
                except re.error:
                    return value in s.name
            if key == "id":
                return self._cid(s, full=True).startswith(value)
            if key == "status":
                return doc["State"] == value
            if key == "ancestor":
                return _image_ref(s.image) == _image_ref(value) or s.image.split(":")[0] == value
            k, _, v = value.partition("=")
            return k in self._labels(s) and (not v or self._labels(s)[k] == v)

        return [s for s in rows if all(any(match(s, k, v) for v in vs) for k, vs in wanted.items())]

    def _image_refs(self, host: str) -> list[str]:
        return sorted(
            {ref for h, ref in self.images if h == host}
            | {
                _image_ref(s.declared[1])
                for s in self.services
                if s.host == host and s.runtime == "docker" and s.declared[1]
            }
        )

    def _image_doc(self, ref: str) -> dict[str, object]:
        repo, tag = ref.rsplit(":", 1)
        digest = hashlib.sha256(ref.encode()).hexdigest()
        return {
            "ID": digest[:12],
            "Repository": repo,
            "Tag": tag,
            "Digest": "<none>",
            "Size": "187MB",
            "CreatedSince": "3 weeks ago",
            "CreatedAt": f"{self.now() - timedelta(weeks=3):%Y-%m-%d %H:%M:%S} +0000 UTC",
            "Containers": "N/A",
            "SharedSize": "N/A",
            "UniqueSize": "N/A",
            "VirtualSize": "187MB",
        }

    def _docker_images(self, host: str, args: list[str]) -> Result:
        opts, pos = getopt(
            "docker images",
            args,
            short="aqf",
            long=frozenset({"all", "quiet", "no-trunc", "digests"}),
            long_valued=frozenset({"format", "filter"}),
            multi=frozenset({"filter"}),
        )
        if "f" in opts:
            raise Gap("docker images -f")
        refs = self._image_refs(host)
        if pos:
            refs = [r for r in refs if r.rsplit(":", 1)[0] == pos[0] or r == _image_ref(pos[0])]
        for f in filter(None, opts.get("filter", "").split("\n")):
            key, _, value = f.partition("=")
            if key == "dangling":
                refs = [] if value == "true" else refs
            elif key == "reference":
                refs = [
                    r
                    for r in refs
                    if fnmatch.fnmatch(r, value) or fnmatch.fnmatch(r.rsplit(":", 1)[0], value)
                ]
            else:
                raise Gap(f"docker images --filter {f}")
        docs = [self._image_doc(r) for r in refs]
        if "q" in opts or "quiet" in opts:
            return Result("".join(f"{d['ID']}\n" for d in docs), probe=True)
        if "format" in opts:
            headers = {
                "Repository": "REPOSITORY",
                "Tag": "TAG",
                "ID": "IMAGE ID",
                "CreatedSince": "CREATED",
                "CreatedAt": "CREATED AT",
                "Size": "SIZE",
                "Digest": "DIGEST",
            }
            return Result(docker_format(opts["format"], docs, headers), probe=True)
        lines = [f"{'REPOSITORY':<32}{'TAG':<12}{'IMAGE ID':<15}{'CREATED':<15}SIZE"]
        for d in docs:
            lines.append(
                f"{d['Repository']:<32}{d['Tag']:<12}{d['ID']:<15}{d['CreatedSince']:<15}{d['Size']}"
            )
        return Result("\n".join(lines) + "\n", probe=True)

    def _docker_image_inspect(self, host: str, args: list[str]) -> Result:
        opts, pos = getopt(
            "docker image inspect", args, short_valued="f", long_valued=frozenset({"format"})
        )
        fmt = opts.get("f", opts.get("format"))
        docs: list[dict[str, object]] = []
        for name in pos:
            ref = _image_ref(name)
            image = self._image(host, name)
            if image is None:
                return Result(
                    "[]\n", stderr=f"Error response from daemon: No such image: {ref}\n", exit=1
                )
            probe = Service(host, ref, image.kind, image.config, "docker", image=ref)
            cmd, entry = self._container_cmd(probe)
            docs.append(
                {
                    "Id": "sha256:" + hashlib.sha256(ref.encode()).hexdigest(),
                    "RepoTags": [ref],
                    "RepoDigests": [],
                    "Created": self._iso(self.now() - timedelta(weeks=3)),
                    "Size": 187_000_000,
                    "Architecture": "amd64",
                    "Os": "linux",
                    "Config": {
                        "Cmd": cmd,
                        "Entrypoint": entry,
                        "Env": [
                            "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
                        ],
                        "ExposedPorts": {},
                        "Labels": {},
                        "WorkingDir": "/app" if image.kind == "app" else "",
                    },
                }
            )
        if fmt is not None:
            return Result("".join(go_template(fmt, d) + "\n" for d in docs), probe=True)
        return Result(json.dumps(docs, indent=4) + "\n", probe=True)

    def _docker_objects(self, host: str, kind: str, args: list[str]) -> Result:
        """docker volume|network ls / inspect: named volumes of the host's containers;
        bridge, host, none plus networks containers name."""
        sub, rest = (args[0], args[1:]) if args else ("", [])
        containers = self._containers(host)
        if kind == "volume":
            names = sorted(
                {
                    str(m["Name"])
                    for s in containers
                    for m in self._mounts(s)
                    if m["Type"] == "volume"
                }
            )
            docs = [
                {
                    "Name": n,
                    "Driver": "local",
                    "Scope": "local",
                    "Mountpoint": f"/var/lib/docker/volumes/{n}/_data",
                    "Labels": {},
                    "Options": {},
                    "CreatedAt": self._iso(self.now() - timedelta(days=30)),
                }
                for n in names
            ]
            headers = {"Driver": "DRIVER", "Name": "VOLUME NAME"}
        else:
            names = ["bridge", "host", "none"] + sorted(
                {s.network for s in containers} - {"bridge", "host", "none"}
            )
            docs = []
            for n in names:
                members = [s for s in containers if s.network == n and s.state == "active"]
                docs.append(
                    {
                        "Name": n,
                        "ID": hashlib.sha256(f"{host}/net/{n}".encode()).hexdigest()[:12],
                        "Id": hashlib.sha256(f"{host}/net/{n}".encode()).hexdigest(),
                        "Driver": {"host": "host", "none": "null"}.get(n, "bridge"),
                        "Scope": "local",
                        "IPAM": {
                            "Driver": "default",
                            "Config": [{"Subnet": "172.17.0.0/16", "Gateway": "172.17.0.1"}]
                            if n == "bridge"
                            else [],
                        },
                        "Containers": {
                            self._cid(s, full=True): {"Name": s.name, "IPv4Address": ""}
                            for s in members
                        },
                    }
                )
            headers = {"ID": "NETWORK ID", "Name": "NAME", "Driver": "DRIVER", "Scope": "SCOPE"}
        if sub in ("ls", "list"):
            opts, pos = getopt(
                f"docker {kind} ls",
                rest,
                short="q",
                long=frozenset({"quiet", "no-trunc"}),
                long_valued=frozenset({"format", "filter"}),
            )
            if pos or "filter" in opts:
                raise Gap(f"docker {kind} ls " + " ".join(rest))
            if "q" in opts or "quiet" in opts:
                return Result(
                    "".join(f"{d['Name' if kind == 'volume' else 'ID']}\n" for d in docs),
                    probe=True,
                )
            fmt = opts.get("format") or "table " + "\t".join("{{." + k + "}}" for k in headers)
            return Result(docker_format(fmt, docs, headers), probe=True)
        if sub == "inspect":
            opts, pos = getopt(
                f"docker {kind} inspect", rest, short_valued="f", long_valued=frozenset({"format"})
            )
            by_name = {str(d["Name"]): d for d in docs}
            found = []
            for n in pos:
                if n not in by_name:
                    what = (
                        f"get {n}: no such volume" if kind == "volume" else f"network {n} not found"
                    )
                    return Result("[]\n", stderr=f"Error response from daemon: {what}\n", exit=1)
                found.append(by_name[n])
            tmpl = opts.get("f", opts.get("format"))
            if tmpl is not None:
                return Result("".join(go_template(tmpl, d) + "\n" for d in found), probe=True)
            return Result(json.dumps(found, indent=4) + "\n", probe=True)
        raise Gap(f"docker {kind} {sub}".strip())

    def _docker_info(self, host: str, args: list[str]) -> Result:
        opts, pos = getopt("docker info", args, short_valued="f", long_valued=frozenset({"format"}))
        if pos:
            raise Gap("docker info " + " ".join(pos))
        containers = self._containers(host)
        running = sum(1 for s in containers if s.state == "active")
        doc: dict[str, object] = {
            "ID": hashlib.sha256(f"{host}/docker".encode()).hexdigest()[:36],
            "Name": host,
            "ServerVersion": DOCKER_VERSION,
            "Containers": len(containers),
            "ContainersRunning": running,
            "ContainersPaused": 0,
            "ContainersStopped": len(containers) - running,
            "Images": len(self._image_refs(host)),
            "Driver": "overlay2",
            "DockerRootDir": "/var/lib/docker",
            "LoggingDriver": "json-file",
            "CgroupDriver": "systemd",
            "CgroupVersion": "2",
            "OperatingSystem": "Ubuntu 24.04.1 LTS",
            "OSType": "linux",
            "Architecture": "x86_64",
            "NCPU": 4,
            "MemTotal": 8_331_448_320,
            "KernelVersion": "6.8.0-49-generic",
        }
        fmt = opts.get("f", opts.get("format"))
        if fmt is not None:
            return Result(go_template(fmt, doc) + "\n", probe=True)
        lines = [
            "Client: Docker Engine - Community",
            f" Version:    {DOCKER_VERSION}",
            "",
            "Server:",
        ]
        labels = {
            "Containers": "Containers",
            "ContainersRunning": " Running",
            "ContainersPaused": " Paused",
            "ContainersStopped": " Stopped",
            "Images": "Images",
            "ServerVersion": "Server Version",
            "Driver": "Storage Driver",
            "LoggingDriver": "Logging Driver",
            "CgroupDriver": "Cgroup Driver",
            "CgroupVersion": "Cgroup Version",
            "KernelVersion": "Kernel Version",
            "OperatingSystem": "Operating System",
            "OSType": "OSType",
            "Architecture": "Architecture",
            "NCPU": "CPUs",
            "Name": "Name",
            "ID": "ID",
            "DockerRootDir": "Docker Root Dir",
        }
        lines += [f" {label}: {doc[key]}" for key, label in labels.items()]
        return Result("\n".join(lines) + "\n", probe=True)

    def _docker_system_df(self, host: str) -> Result:
        containers = self._containers(host)
        images = self._image_refs(host)
        used = {_image_ref(s.image) for s in containers}
        volumes = {
            str(m["Name"]) for s in containers for m in self._mounts(s) if m["Type"] == "volume"
        }
        active = sum(1 for s in containers if s.state == "active")
        rows = [
            ["TYPE", "TOTAL", "ACTIVE", "SIZE", "RECLAIMABLE"],
            [
                "Images",
                str(len(images)),
                str(len(used & set(images))),
                f"{187 * len(images)}MB",
                f"{187 * (len(images) - len(used & set(images)))}MB ({100 * (len(images) - len(used & set(images))) // max(1, len(images))}%)",
            ],
            ["Containers", str(len(containers)), str(active), "0B", "0B (0%)"],
            ["Local Volumes", str(len(volumes)), str(len(volumes)), "0B", "0B (0%)"],
            ["Build Cache", "0", "0", "0B", "0B"],
        ]
        return Result(text_table(rows), probe=True)

    def _docker_run(self, host: str, sub: str, args: list[str]) -> Result:
        """docker run -d / docker create: a (re)created container of a known image."""
        multi = frozenset(
            {
                "p",
                "publish",
                "e",
                "env",
                "env-file",
                "v",
                "volume",
                "l",
                "label",
                "log-opt",
                "add-host",
            }
        )
        opts, pos = getopt(
            f"docker {sub}",
            args,
            short="dit",
            short_valued="pevlw",
            long=frozenset({"detach", "interactive", "tty", "init", "read-only"}),
            long_valued=frozenset(
                {
                    "name",
                    "publish",
                    "env",
                    "env-file",
                    "volume",
                    "restart",
                    "network",
                    "net",
                    "hostname",
                    "label",
                    "workdir",
                    "pull",
                    "user",
                    "memory",
                    "cpus",
                    "log-driver",
                    "log-opt",
                    "add-host",
                }
            ),
            stop_at_positional=True,
            multi=multi,
        )
        if sub == "run" and not ("d" in opts or "detach" in opts):
            raise Gap("docker run without -d (foreground containers)")
        if not pos:
            return Result(stderr=f'"docker {sub}" requires at least 1 argument.\n', exit=1)
        if len(pos) > 1:
            raise Gap(f"docker {sub} with a command")
        image_name = pos[0]
        image = self._image(host, image_name)
        if image is None:
            return Result(
                stderr=f"Unable to find image '{_image_ref(image_name)}' locally\n"
                'docker: Error response from daemon: Get "https://registry-1.docker.io/v2/": dial tcp: '
                "lookup registry-1.docker.io on 127.0.0.53:53: no such host.\n"
                f"See 'docker {sub} --help'.\n",
                exit=125,
            )
        cwd = getattr(self.tls, "cwd", "/root")

        def values(*keys: str) -> list[str]:
            return [v for k in keys if k in opts for v in opts[k].split("\n")]

        env: dict[str, str] = {}
        for path in values("env-file"):
            text = self.read(host, posixpath.join(cwd, path))
            if text is None:
                return Result(stderr=f"docker: open {path}: no such file or directory\n", exit=125)
            for line in text.splitlines():
                m = ENV_LINE.match(line.strip())
                if m:
                    env[m.group(1)] = m.group(2)
        for kv in values("e", "env"):
            key, eq, value = kv.partition("=")
            if eq:
                env[key] = value
        publish: dict[int, int] = {}
        for spec in values("p", "publish"):
            parts = spec.split("/")[0].split(":")
            if len(parts) < 2 or not parts[-1].isdigit() or not parts[-2].isdigit():
                raise Gap(f"docker {sub} -p {spec}")
            publish[int(parts[-1])] = int(parts[-2])
        config = ""
        volumes: list[str] = []
        for spec in values("v", "volume"):
            src, _, rest = spec.partition(":")
            dst = rest.split(":")[0]
            if src.startswith((".", "/")):
                src = posixpath.normpath(posixpath.join(cwd, src))
            volumes.append(":".join([src, *rest.split(":")]))
            if image.config and dst == image.config:
                config = src
        network = opts.get("network", opts.get("net", "bridge"))
        name = (
            opts.get("name")
            or f"{image_name.rsplit('/', 1)[-1].split(':')[0]}-{len(self.services)}"
        )
        svc = self.find(host, name, "docker")
        if svc is not None and svc.present:
            return Result(
                stderr=f'docker: Error response from daemon: Conflict. The container name "/{name}" is already in use '
                f'by container "{self._cid(svc, full=True)}". You have to remove (or rename) that container to be '
                "able to reuse that name.\nSee 'docker run --help'.\n",
                exit=125,
            )
        if svc is None:
            svc = self.add_service(
                host, name, image.kind, config, "docker", routes=image.routes, image=image_name
            )
            svc.declared = ("", "")
        else:
            svc.generation += 1
            svc.kind, svc.routes = image.kind, dict(image.routes)
        svc.description = svc.description or KIND_DEFAULTS[image.kind][0]
        svc.config, svc.image, svc.env, svc.publish, svc.volumes = (
            config,
            image_name,
            env,
            publish,
            volumes,
        )
        svc.network = "host" if network == "host" else "bridge"
        svc.present, svc.state, svc.since, svc.exit_status = True, "inactive", None, 0
        svc.started_env = self._env_file_lines(svc)
        self.container_logs.pop((host, name), None)
        self.event(
            "transition",
            host,
            service=name,
            action="create",
            mutation=True,
            exit=0,
            text=f"create {name}",
        )
        if sub == "run":
            self.transition(svc, "start")
        return Result(self._cid(svc, full=True) + "\n")

    def _remove_container(self, svc: Service) -> None:
        if svc.state == "active":
            self.transition(svc, "stop")
        svc.present, svc.state, svc.since = False, "inactive", None
        self.container_logs.pop((svc.host, svc.name), None)
        self.event(
            "transition",
            svc.host,
            service=svc.name,
            action="remove",
            mutation=True,
            exit=0,
            text=f"remove {svc.name}",
        )

    def _docker_rm(self, host: str, args: list[str]) -> Result:
        opts, pos = getopt("docker rm", args, short="fv", long=frozenset({"force", "volumes"}))
        if not pos:
            return Result(stderr='"docker rm" requires at least 1 argument.\n', exit=1)
        out = Result()
        for name in pos:
            svc = self._container(host, name)
            if svc is None:
                out.stderr += f"Error response from daemon: No such container: {name}\n"
                out.exit = 1
                continue
            if svc.state == "active" and not ("f" in opts or "force" in opts):
                out.stderr += (
                    f'Error response from daemon: cannot remove container "/{svc.name}": container is running: '
                    "stop the container before removing or force remove\n"
                )
                out.exit = 1
                continue
            self._remove_container(svc)
            out.stdout += name + "\n"
        return out

    def _docker_exec(self, host: str, args: list[str], stdin: Stdin) -> Result:
        """Runs in the host's filesystem with the container's env (containers share the host fs here)."""
        opts, pos = getopt(
            "docker exec",
            args,
            short="itd",
            short_valued="euw",
            long=frozenset({"interactive", "tty", "detach", "privileged"}),
            long_valued=frozenset({"env", "user", "workdir"}),
            stop_at_positional=True,
            multi=frozenset({"e", "env"}),
        )
        if len(pos) < 2:
            return Result(stderr='"docker exec" requires at least 2 arguments.\n', exit=1)
        with self.lock:
            svc = self._container(host, pos[0])
            if svc is None:
                return Result(
                    stderr=f"Error response from daemon: No such container: {pos[0]}\n", exit=1
                )
            if svc.state != "active":
                return Result(
                    stderr=f"Error response from daemon: container {self._cid(svc, full=True)} is not running\n",
                    exit=1,
                )
            env = [*self._container_env(svc), f"HOSTNAME={self._cid(svc)}"]
        env += [v for k in ("e", "env") if k in opts for v in opts[k].split("\n")]
        workdir = opts.get("w", opts.get("workdir", "/"))
        command = f"cd {shlex.quote(workdir)} && exec env {shlex.join(env)} {shlex.join(pos[1:])}"
        interactive = "i" in opts or "interactive" in opts
        return self._run_on(host, command, stdin if interactive else None)

    # docker compose ---------------------------------------------------------

    COMPOSE_FILES = ("compose.yaml", "compose.yml", "docker-compose.yaml", "docker-compose.yml")

    def _compose(self, host: str, args: list[str]) -> Result:
        """Compose files are not parsed: the project's containers are the services declared
        with compose=<dir>. `up -d` starts missing ones and re-creates those whose config changed."""
        opts, pos = getopt(
            "docker compose",
            args,
            short_valued="fp",
            long_valued=frozenset(
                {"file", "project-name", "project-directory", "ansi", "progress"}
            ),
            stop_at_positional=True,
        )
        cwd = getattr(self.tls, "cwd", "/root")
        file_opt = opts.get("f", opts.get("file"))
        directory: str | None = None
        if file_opt is not None:
            path = posixpath.normpath(posixpath.join(cwd, file_opt))
            if self.read(host, path) is None:
                return Result(stderr=f"open {path}: no such file or directory\n", exit=1)
            directory = posixpath.dirname(path)
        elif "project-directory" in opts:
            directory = posixpath.normpath(posixpath.join(cwd, opts["project-directory"]))
        else:
            d = posixpath.normpath(cwd)
            while True:
                if any(
                    self.read(host, posixpath.join(d, n)) is not None for n in self.COMPOSE_FILES
                ):
                    directory = d
                    break
                if d == "/":
                    break
                d = posixpath.dirname(d)
        if directory is None:
            return Result(stderr="no configuration file provided: not found\n", exit=1)
        project = [
            s
            for s in self.services
            if s.host == host and s.runtime == "docker" and s.compose == directory
        ]
        if not pos:
            return Result(DOCKER_USAGE["compose"] + "\n", probe=True)
        sub, rest = pos[0], pos[1:]
        if sub == "version":
            return Result("Docker Compose version v2.32.4\n", probe=True)
        flags: dict[str, tuple[str, str, frozenset[str], frozenset[str]]] = {
            "up": (
                "d",
                "",
                frozenset(
                    {
                        "detach",
                        "build",
                        "force-recreate",
                        "no-deps",
                        "remove-orphans",
                        "wait",
                        "quiet-pull",
                    }
                ),
                frozenset(),
            ),
            "down": ("v", "t", frozenset({"volumes", "remove-orphans"}), frozenset({"timeout"})),
            "ps": ("aq", "", frozenset({"all", "quiet", "services"}), frozenset()),
            "logs": (
                "t",
                "n",
                frozenset({"no-color", "timestamps", "no-log-prefix"}),
                frozenset({"tail", "since"}),
            ),
            "start": ("", "", frozenset(), frozenset()),
            "stop": ("", "t", frozenset(), frozenset({"timeout"})),
            "restart": ("", "t", frozenset(), frozenset({"timeout"})),
        }
        if sub not in flags:
            raise Gap(f"docker compose {sub}")
        short, short_valued, long, long_valued = flags[sub]
        sopts, names = getopt(
            f"docker compose {sub}",
            rest,
            short=short,
            short_valued=short_valued,
            long=long,
            long_valued=long_valued,
        )
        unknown = [n for n in names if n not in {s.name for s in project}]
        if unknown:
            return Result(stderr=f"no such service: {unknown[0]}\n", exit=1)
        chosen = [s for s in project if not names or s.name in names]
        if sub == "up":
            if not ("d" in sopts or "detach" in sopts):
                raise Gap("docker compose up without -d (attached)")
            return self._compose_up(chosen, "force-recreate" in sopts)
        if sub == "down":
            err = ""
            for svc in project:
                if svc.present:
                    self._remove_container(svc)
                    err += f" Container {svc.name}  Removed\n"
            return Result(
                stderr=err + f" Network {posixpath.basename(directory)}_default  Removed\n"
            )
        if sub == "ps":
            rows = [
                s
                for s in chosen
                if s.present and (s.state == "active" or "a" in sopts or "all" in sopts)
            ]
            if "q" in sopts or "quiet" in sopts:
                return Result("".join(self._cid(s) + "\n" for s in rows), probe=True)
            if "services" in sopts:
                return Result("".join(s.name + "\n" for s in rows), probe=True)
            lines = [f"{'NAME':<20}{'IMAGE':<28}{'SERVICE':<16}{'STATUS':<26}PORTS"]
            lines += [
                f"{s.name:<20}{s.image:<28}{s.name:<16}{self._docker_status(s):<26}{self._docker_ports(s)}"
                for s in rows
            ]
            return Result("\n".join(lines) + "\n", probe=True)
        if sub == "logs":
            tail = sopts.get("n", sopts.get("tail", "all"))
            if tail != "all" and not tail.isdigit():
                raise Gap(f"docker compose logs --tail {tail}")
            logs = ""
            for s in chosen:
                lines = self.container_logs.get((host, s.name), [])
                if tail != "all":
                    lines = lines[-int(tail) :] if int(tail) else []
                prefix = "" if "no-log-prefix" in sopts else f"{s.name:<12}| "
                logs += "".join(prefix + line + "\n" for line in lines)
            return Result(logs, probe=True)
        out = Result()
        for svc in chosen:
            if not svc.present:
                continue
            error = self.transition(svc, sub)
            out.stderr += (
                f" Container {svc.name}  {sub.capitalize()}{'ed' if sub != 'stop' else 'ped'}\n"
            )
            if error:
                out.exit = 1
        return out

    def _compose_up(self, chosen: list[Service], force: bool) -> Result:
        out = Result()
        for svc in chosen:
            if not svc.present or force:
                if svc.present:
                    self._remove_container(svc)
                svc.generation += 1
                svc.config, svc.image = svc.declared
                svc.env, svc.publish, svc.volumes, svc.network = {}, {}, [], "host"
                svc.present, svc.state, svc.since = True, "inactive", None
                self.event(
                    "transition",
                    svc.host,
                    service=svc.name,
                    action="create",
                    mutation=True,
                    exit=0,
                    text=f"create {svc.name}",
                )
                error, status = self.transition(svc, "start"), "Started"
            elif svc.state != "active":
                error, status = self.transition(svc, "start"), "Started"
            else:
                try:
                    changed = LOADERS[svc.kind](self, svc) != svc.conf
                except ConfigError:
                    changed = True
                if not changed:
                    out.stderr += f" Container {svc.name}  Running\n"
                    continue
                error, status = self.transition(svc, "restart"), "Recreated"
            out.stderr += f" Container {svc.name}  {status}\n"
            if error:
                out.stderr += f"Error response from daemon: container {svc.name} exited (1)\n"
                out.exit = 1
        return out

    def _container(self, host: str, name: str) -> Service | None:
        for s in self._containers(host):
            if name in (s.name, self._cid(s)) or (len(name) >= 4 and self._cid(s).startswith(name)):
                return s
        return None

    # nginx -----------------------------------------------------------------

    def _nginx(self, host: str, args: list[str], stdin: str | None = None) -> Result:
        svc = next((s for s in self.services if s.host == host and s.kind == "proxy"), None)
        if svc is None:
            return Result(stderr="bash: nginx: command not found\n", exit=127)
        if args in (["-h"], ["-?"], ["--help"]):
            return Result(stderr=NGINX_HELP, probe=True)
        opts, pos = getopt("nginx", args, short="tTvVq", short_valued="sc")
        if pos:
            raise Gap("nginx " + " ".join(pos))
        if "v" in opts or "V" in opts:
            return Result(stderr=f"nginx version: nginx/{NGINX_VERSION}\n", probe=True)
        config = opts.get("c", svc.config)
        if not config.startswith("/"):
            config = "/etc/nginx/" + config
        if "s" in opts:
            if config != svc.config:
                raise Gap("nginx -s with -c of another config")
            signal = opts["s"]
            if signal in ("reload", "reopen") and svc.state != "active":
                return Result(
                    stderr='nginx: [error] invalid PID number "" in "/run/nginx.pid"\n', exit=1
                )
            if signal == "reload":
                error = self.transition(svc, "reload")
                return Result(stderr=(error + "\n") if error else "", exit=1 if error else 0)
            if signal == "reopen":
                return Result()
            if signal in ("stop", "quit"):
                self.transition(svc, "stop")
                return Result()
            raise Gap(f"nginx -s {signal}")
        if "t" in opts or "T" in opts:
            try:
                conf = load_proxy(self, dataclasses.replace(svc, config=config))
            except ConfigError as exc:
                return Result(
                    stderr=f"{exc.args[0]}\nnginx: configuration file {config} test failed\n",
                    exit=1,
                    probe=True,
                )
            err = (
                f"nginx: the configuration file {config} syntax is ok\n"
                f"nginx: configuration file {config} test is successful\n"
            )
            dump = ""
            if "T" in opts:
                for f in conf.files:
                    dump += f"# configuration file {f}:\n{self.read(host, f) or ''}\n"
            return Result(dump, err, probe=True)
        if config != svc.config:
            raise Gap("nginx -c to start another config")
        if svc.state == "active":  # a second master cannot bind the ports
            port = svc.listen_ports()[0] if svc.listen_ports() else 80
            line = f"nginx: [emerg] bind() to 0.0.0.0:{port} failed (98: Address already in use)\n"
            return Result(stderr=line * 5 + "nginx: [emerg] still could not bind()\n", exit=1)
        error = self.transition(svc, "start")
        return Result(stderr=(error + "\n") if error else "", exit=1 if error else 0)

    # curl ------------------------------------------------------------------

    def _curl(self, host: str, args: list[str], stdin: str | None = None) -> Result:
        if args in (["--version"], ["-V"]):
            return Result(CURL_VERSION, probe=True)
        opts, pos = getopt(
            "curl",
            args,
            short="sSfiIvLkg0",
            short_valued="owmXHdAeDrF",
            long=frozenset(
                {
                    "silent",
                    "show-error",
                    "fail",
                    "include",
                    "head",
                    "verbose",
                    "location",
                    "insecure",
                    "fail-with-body",
                    "compressed",
                    "globoff",
                    "http1.0",
                    "http1.1",
                    "http2",
                    "path-as-is",
                    "no-buffer",
                    "no-progress-meter",
                    "no-keepalive",
                }
            ),
            long_valued=frozenset(
                {
                    "output",
                    "write-out",
                    "max-time",
                    "connect-timeout",
                    "request",
                    "header",
                    "data",
                    "data-raw",
                    "data-binary",
                    "data-urlencode",
                    "json",
                    "user-agent",
                    "retry",
                    "retry-delay",
                    "referer",
                    "resolve",
                    "dump-header",
                    "range",
                    "form",
                    "form-string",
                    "noproxy",
                    "user",
                }
            ),
            multi=frozenset({"F", "form", "form-string"}),
        )
        if not pos:
            return Result(
                stderr="curl: no URL specified\ncurl: try 'curl --help' or 'curl --manual' for more information\n",
                exit=2,
            )
        cwd = getattr(self.tls, "cwd", "/root")
        data_key: str | None = next(
            (
                k
                for k in ("d", "data", "data-raw", "data-binary", "data-urlencode", "json")
                if k in opts
            ),
            None,
        )
        data = opts[data_key] if data_key else ""
        if data_key and data_key not in ("data-raw", "data-urlencode") and data.startswith("@"):
            src = data[1:]
            text = (stdin or "") if src == "-" else self.read(host, posixpath.join(cwd, src))
            if text is None:
                return Result(
                    stderr=f'Warning: Couldn\'t read data from file "{src}", this makes an empty POST.\n',
                    probe=True,
                )
            data = (
                text
                if data_key in ("data-binary", "json")
                else text.replace("\r", "").replace("\n", "")
            )
        forms = [
            f
            for k in ("F", "form", "form-string")
            for f in opts.get(k, "").split("\n")
            if f and k in opts
        ]
        headers: dict[str, str] = {}
        if forms:
            if data_key:
                return Result(
                    stderr="Warning: You can only select one HTTP request method! You asked for both POST\nWarning: (-d, --data) and multipart formpost (-F, --form).\n",
                    exit=2,
                )
            body = self._curl_form(host, cwd, forms)
            if isinstance(body, Result):
                return body
            data, data_key = body, "form"
            headers["Content-Type"] = f"multipart/form-data; boundary={CURL_BOUNDARY}"
        elif data_key == "json":
            headers |= {"Content-Type": "application/json", "Accept": "application/json"}
        elif data_key:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        if "r" in opts or "range" in opts:
            headers["Range"] = "bytes=" + opts.get("r", opts.get("range", ""))
        for i, a in enumerate(args):  # -H repeats; getopt keeps only the last value
            hdr = None
            if a in ("-H", "--header") and i + 1 < len(args):
                hdr = args[i + 1]
            elif a.startswith("--header="):
                hdr = a.split("=", 1)[1]
            elif a.startswith("-H") and len(a) > 2:
                hdr = a[2:]
            if hdr and ":" in hdr:
                k, v = hdr.split(":", 1)
                headers[k.strip()] = v.strip()
        last = opts.get("H", opts.get("header"))  # also catches combined flags like -sH
        if last and ":" in last:
            k, v = last.split(":", 1)
            headers.setdefault(k.strip(), v.strip())
        head = "I" in opts or "head" in opts
        method = opts.get("X", opts.get("request", "HEAD" if head else "GET"))
        if data_key and "X" not in opts and "request" not in opts:
            method = "POST"
        out = Result(probe=True)
        for n, url in enumerate(
            pos
        ):  # -o and -D name the first transfer's files; later ones go to stdout
            one = self._curl_url(host, opts, url, method, dict(headers), data, first=n == 0)
            out.stdout += one.stdout
            out.stderr += one.stderr
            out.exit = one.exit or out.exit
        return out

    def _curl_form(self, host: str, cwd: str, forms: list[str]) -> str | Result:
        """-F name=value / name=@file / name=<file as a multipart body."""
        parts = []
        for f in forms:
            name, eq, value = f.partition("=")
            if not eq:
                return Result(
                    stderr=f"Warning: Illegally formatted input field: '{f}'!\ncurl: option -F: is badly used here\n",
                    exit=2,
                )
            filename = ""
            if value[:1] in ("@", "<"):
                path = value[1:].split(";", 1)[0]
                text = self.read(host, posixpath.join(cwd, path))
                if text is None:
                    return Result(
                        stderr="curl: (26) Failed to open/read local data from file/application\n",
                        exit=26,
                    )
                filename = posixpath.basename(path) if value[0] == "@" else ""
                value = text
            disp = f'Content-Disposition: form-data; name="{name}"' + (
                f'; filename="{filename}"\r\nContent-Type: application/octet-stream'
                if filename
                else ""
            )
            parts.append(f"--{CURL_BOUNDARY}\r\n{disp}\r\n\r\n{value}\r\n")
        return "".join(parts) + f"--{CURL_BOUNDARY}--\r\n"

    def _curl_url(
        self,
        host: str,
        opts: dict[str, str],
        url: str,
        method: str,
        headers: dict[str, str],
        data: str,
        first: bool,
    ) -> Result:
        """One transfer of a curl command line."""
        cwd = getattr(self.tls, "cwd", "/root")
        output = opts.get("o", opts.get("output")) if first else None
        dump = opts.get("D", opts.get("dump-header")) if first else None
        silent = "s" in opts or "silent" in opts
        show_err = "S" in opts or "show-error" in opts or not silent
        fail = "f" in opts or "fail" in opts
        head = "I" in opts or "head" in opts
        include = "i" in opts or "include" in opts
        verbose = "v" in opts or "verbose" in opts
        fmt = opts.get("w", opts.get("write-out"))
        original = url
        if "path-as-is" not in opts:
            url = _squash_dots(url)
        if "resolve" in opts:  # name:port:addr -> connect to addr, keep Host: name
            name, _, rest_ = opts["resolve"].partition(":")
            port_s, _, addr = rest_.partition(":")
            parts = urlsplit(url if "://" in url else "http://" + url)
            if (
                parts.hostname == name
                and str(parts.port or (443 if parts.scheme == "https" else 80)) == port_s
                and addr
            ):
                headers.setdefault("Host", name)
                url = parts._replace(netloc=f"{addr}:{port_s}").geturl()
        resp: Response | ConnError = ConnError(0, "")
        follow = "L" in opts or "location" in opts
        redirects = 0
        while True:
            resp = self.request(host, method, url, headers, body=data)
            location = _header(resp.headers, "location") if isinstance(resp, Response) else ""
            if (
                not (
                    follow
                    and isinstance(resp, Response)
                    and resp.status in (301, 302, 303, 307, 308)
                    and location
                )
                or redirects >= 5
            ):
                break
            url = urljoin(url if "://" in url else "http://" + url, location)
            redirects += 1
        res = Result(probe=True)
        if isinstance(resp, ConnError):
            if show_err:
                res.stderr = f"curl: ({resp.code}) {resp.message}\n"
            res.exit = resp.code
            if fmt:
                res.stdout = self._write_out(
                    fmt, self._curl_info(host, original, url, resp, method, data, redirects, output)
                )
            return res
        status_line = f"HTTP/1.1 {resp.status} {REASONS.get(resp.status, '')}".rstrip()
        all_headers = resp.headers + [("Content-Length", str(len(resp.body.encode())))]
        header_text = (
            status_line + "\r\n" + "".join(f"{k}: {v}\r\n" for k, v in all_headers) + "\r\n"
        )
        if verbose:
            parts = urlsplit(url if "://" in url else "http://" + url)
            res.stderr += f"* Connected to {parts.hostname} port {parts.port or 80}\n> {method} {parts.path or '/'} HTTP/1.1\n> Host: {parts.hostname}\n>\n"
            res.stderr += (
                "".join("< " + line + "\n" for line in header_text.strip().split("\r\n")) + "<\n"
            )
        info = self._curl_info(
            host, original, url, resp, method, data, redirects, output, len(header_text)
        )
        if fail and resp.status >= 400:
            if show_err:
                res.stderr += f"curl: (22) The requested URL returned error: {resp.status}\n"
            res.exit = 22
            if "fail-with-body" not in opts:
                if fmt:
                    res.stdout = self._write_out(fmt, info)
                return res
        body = "" if head else resp.body
        shown_headers = header_text if head or include else ""
        if output not in (None, "-"):
            saved = (
                ""
                if output == "/dev/null"
                else self._curl_save(host, posixpath.join(cwd, output), shown_headers + body)
            )
            if saved:
                warn = (
                    f"Warning: Failed to open the file {output}: {saved}\n"
                    if saved != ENOSPC
                    else ""
                )
                res.stderr += f"{warn}curl: (23) Failure writing output to destination\n"
                res.exit = 23
                return res
            shown_headers = body = ""
        if dump == "-":
            shown_headers = header_text
        elif dump is not None and self._curl_save(host, posixpath.join(cwd, dump), header_text):
            res.stderr += f"Warning: Failed to open {dump}\n"
        res.stdout = shown_headers + body
        if fmt:
            res.stdout += self._write_out(fmt, info)
        return res

    def _curl_save(self, host: str, path: str, text: str) -> str:
        """curl -o/-D to a file on host; the error ("" when written)."""
        try:
            real = self.real_path(host, path)
            if not real.parent.is_dir():
                return "No such file or directory"
            data = text.encode()
            old = real.stat().st_size if real.is_file() else 0
            if len(data) - old > self.disk_free(host):
                return ENOSPC
            real.write_bytes(data)
        except OSError as exc:
            return exc.strerror or ENOSPC
        return ""

    def _curl_info(
        self,
        host: str,
        original: str,
        url: str,
        resp: Response | ConnError,
        method: str,
        data: str,
        redirects: int,
        output: str | None,
        header_size: int = 0,
    ) -> dict[str, str]:
        """curl -w variables of one transfer. Times are small fixed values."""
        parts = urlsplit(url if "://" in url else "http://" + url)
        name = parts.hostname or ""
        try:
            port = parts.port or (443 if parts.scheme == "https" else 80)
        except ValueError:
            port = 0
        target = self.resolve_host(name, host)
        loopback = name in ("localhost", "127.0.0.1", "0.0.0.0")
        ok = isinstance(resp, Response)
        connected = ok or (isinstance(resp, ConnError) and resp.code == 52)
        status = resp.status if isinstance(resp, Response) else 0
        size = len(resp.body.encode()) if isinstance(resp, Response) and method != "HEAD" else 0
        location = _header(resp.headers, "location") if isinstance(resp, Response) else ""
        tls = parts.scheme == "https"
        times = {
            "time_namelookup": 0.000412,
            "time_connect": 0.000533,
            "time_appconnect": 0.004112 if tls else 0.0,
            "time_pretransfer": 0.004201 if tls else 0.000611,
            "time_starttransfer": 0.006734,
            "time_total": 0.006913,
            "time_redirect": 0.0,
        }
        if not connected:
            times = {
                k: (0.000412 if k in ("time_namelookup", "time_total") and target else 0.0)
                for k in times
            }
        info = {
            "http_code": f"{status:03d}",
            "response_code": f"{status:03d}",
            "http_connect": "000",
            "http_version": "1.1" if ok else "0",
            "method": method,
            "content_type": _header(resp.headers, "content-type")
            if isinstance(resp, Response)
            else "",
            "redirect_url": urljoin(url if "://" in url else "http://" + url, location)
            if location and status in (301, 302, 303, 307, 308)
            else "",
            "num_redirects": str(redirects),
            "num_connects": "1" if connected else "0",
            "size_download": str(size),
            "size_header": str(header_size),
            "size_request": str(len(method) + len(parts.path or "/") + 60),
            "size_upload": str(len(data.encode())),
            "speed_download": str(int(size / times["time_total"])) if connected else "0",
            "speed_upload": "0",
            "remote_ip": ("127.0.0.1" if loopback else self.hosts[target].ip)
            if connected and target
            else "",
            "remote_port": str(port) if connected else "0",
            "local_ip": ("127.0.0.1" if loopback else self.hosts[host].ip) if connected else "",
            "local_port": "51234" if connected else "0",
            "ssl_verify_result": "0",
            "url": original,
            "url_effective": url,
            "filename_effective": output if output not in (None, "-", "/dev/null") else "",
            "exitcode": str(resp.code if isinstance(resp, ConnError) else 0),
            "errormsg": resp.message if isinstance(resp, ConnError) else "",
        }
        info |= {k: f"{v:.6f}" for k, v in times.items()}
        if isinstance(resp, Response):
            info |= {f"header:{k.lower()}": v for k, v in resp.headers}
        return info

    def _write_out(self, fmt: str, info: dict[str, str]) -> str:
        def var(m: re.Match[str]) -> str:
            if m.group(0) == "%%":
                return "%"
            if m.group(2) is not None:
                return info.get("header:" + m.group(2).lower(), "")
            if m.group(1) not in info:
                raise Gap(f"curl -w %{{{m.group(1)}}}")
            return info[m.group(1)]

        return _WRITE_OUT.sub(
            var, fmt.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
        )

    # ss / ps ---------------------------------------------------------------

    def _ss(self, host: str, args: list[str], stdin: str | None = None) -> Result:
        opts, pos = getopt(
            "ss",
            args,
            short="tulnpa46Hexo",
            long=frozenset(
                {
                    "tcp",
                    "udp",
                    "listening",
                    "numeric",
                    "processes",
                    "all",
                    "unix",
                    "no-header",
                    "extended",
                    "options",
                }
            ),
        )
        states, sports = _ss_filter(pos)
        tcp = "t" in opts or "tcp" in opts
        unix = "x" in opts or "unix" in opts
        if ("u" in opts or "udp" in opts) and not tcp and not unix:
            return Result(
                "State  Recv-Q Send-Q Local Address:Port Peer Address:Port Process\n", probe=True
            )
        numeric = "n" in opts or "numeric" in opts
        procs = "p" in opts or "processes" in opts
        listening = not states or bool(states & {"listening", "listen", "all", "bucket"})
        rows: list[tuple[int, str, int]] = [(22, "sshd", 600)]
        socks: list[tuple[str, int, str, int]] = [
            ("/run/systemd/journal/stdout", 13021, "systemd-journal", 312),
            ("/run/dbus/system_bus_socket", 15211, "dbus-daemon", 590),
        ]
        if self._has_docker(host):
            socks.append(("/run/docker.sock", 23140, "dockerd", 2211))
        for svc in self.services:
            if svc.host != host:
                continue
            proc = {
                "proxy": "nginx",
                "db": "postgres",
                "app": "uvicorn" if svc.runtime == "docker" else "python3",
            }[svc.kind]
            if svc.runtime == "docker":
                proc = "docker-proxy"
            for port in svc.ports():
                rows.append((port, proc, svc.pid))
            if svc.kind == "db" and svc.runtime == "systemd":
                socks += [
                    (f"/var/run/postgresql/.s.PGSQL.{port}", 30000 + port, "postgres", svc.pid)
                    for port in svc.listen_ports()
                ]
        rows = [r for r in rows if listening and (sports is None or r[0] in sports)]
        socks = sorted(socks) if listening and sports is None else []
        single = len(states) == 1 and not states & {
            "all",
            "connected",
            "synchronized",
            "bucket",
            "big",
        }
        both = unix and tcp
        head = (
            ("Netid " if unix else "")
            + ("" if single else "State  ")
            + "Recv-Q Send-Q Local Address:Port Peer Address:Port Process"
        )
        lines = [] if "H" in opts or "no-header" in opts else [head]
        state_col = "" if single else "LISTEN "
        if unix:
            for path, inode, proc, pid in socks:
                line = f"u_str {state_col}0      4096   {path} {inode}            * 0"
                lines.append(line + (f'     users:(("{proc}",pid={pid},fd=3))' if procs else ""))
        if tcp or not unix:
            for port, proc, pid in sorted(rows):
                label = str(port) if numeric else WELL_KNOWN_PORTS.get(port, str(port))
                line = (
                    "tcp   " if both else ""
                ) + f"{state_col}0      511          0.0.0.0:{label:<9}      0.0.0.0:*"
                lines.append(line + (f'     users:(("{proc}",pid={pid},fd=6))' if procs else ""))
        return Result("\n".join(lines) + "\n", probe=True)

    def _procs(self, host: str) -> list[Proc]:
        """The host's process table: a few system daemons plus the active services."""
        now = self.now()
        boot = min((s.since for s in self.services if s.since), default=now) - timedelta(
            days=3, minutes=17
        )
        procs = [
            Proc(1, 0, "/sbin/init", boot),
            Proc(420, 1, "/usr/sbin/cron -f", boot),
            Proc(600, 1, "sshd: /usr/sbin/sshd -D [listener] 0 of 10-100 startups", boot),
        ]
        for svc in self.services:
            if svc.host != host or svc.state != "active":
                continue
            at = svc.since or now
            if svc.kind == "proxy":
                procs.append(
                    Proc(
                        svc.pid,
                        1,
                        "nginx: master process /usr/sbin/nginx -g daemon on; master_process on;",
                        at,
                    )
                )
                procs.append(Proc(svc.pid + 1, svc.pid, "nginx: worker process", at))
            elif svc.kind == "db":
                procs.append(
                    Proc(
                        svc.pid,
                        1,
                        f"/usr/lib/postgresql/16/bin/postgres -c config_file={svc.config}",
                        at,
                    )
                )
                procs.append(Proc(svc.pid + 1, svc.pid, "postgres: 16/main: checkpointer", at))
            elif svc.runtime == "docker":
                procs.append(
                    Proc(
                        svc.pid,
                        1,
                        f"/usr/bin/containerd-shim-runc-v2 -namespace moby -id {self._cid(svc)}",
                        at,
                    )
                )
                port = svc.listen_ports()[0]
                procs.append(
                    Proc(
                        svc.pid + 1,
                        svc.pid,
                        f"/usr/local/bin/python /usr/local/bin/uvicorn app:app --host 0.0.0.0 --port {port}",
                        at,
                    )
                )
            else:
                procs.append(
                    Proc(
                        svc.pid, 1, self._exec_line(svc), at, svc.workdir or "/", svc.user or "root"
                    )
                )
        return sorted(procs, key=lambda p: p.pid)

    def _ps(self, host: str, args: list[str], stdin: str | None = None) -> Result:
        sort = ""
        rest: list[str] = []
        i = 0
        while i < len(args):
            a = args[i]
            if a.startswith("--sort"):
                if a in ("--sort",):
                    sort, i = (args[i + 1] if i + 1 < len(args) else ""), i + 2
                    continue
                sort = a.partition("=")[2]
            else:
                rest.append(a)
            i += 1
        try:
            style = " ".join(rest)
            if style not in (
                "aux",
                "auxf",
                "auxww",
                "-ef",
                "-e",
                "-A",
                "ax",
                "-aux",
                "-efww",
                "-eF",
                "faux",
                "axu",
                "-ely",
                "-elf",
                "aux -ww",
            ):
                return self._ps_select(host, rest, sort)
            procs = self._procs(host)
            if sort:
                procs = _ps_sort(procs, sort, self.now())
        except PsError as exc:
            return Result(stderr=f"{exc}\n{PS_USAGE}", exit=1)
        if style.startswith("-e") or style == "-A":
            lines = ["UID          PID    PPID  C STIME TTY          TIME CMD"]
            lines += [
                f"{p.user:<8}{p.pid:>8} {p.ppid:>7}  0 09:00 ?        00:00:01 {p.cmd}"
                for p in procs
            ]
        else:
            lines = ["USER         PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND"]
            lines += [
                f"{p.user:<8}{p.pid:>8}  0.0  0.4  98000 14000 ?        Ss   09:00   0:01 {p.cmd}"
                for p in procs
            ]
        return Result("\n".join(lines) + "\n", probe=True)

    def _ps_select(self, host: str, args: list[str], sort: str = "") -> Result:
        """`ps -o FIELDS -p PIDS`, `ps -fp PID`, `ps -eo pid,etime,cmd`, `ps -C nginx`, `ps 1 2`."""
        fields: list[str] = []
        pids: set[int] | None = None
        names: set[str] | None = None
        every = full = False
        headers = True
        i = 0
        while i < len(args):
            a = args[i]
            i += 1
            if a.isdigit() or (a.replace(",", "").isdigit() and a):
                pids = (pids or set()) | _pid_list(a)
                continue
            if a.startswith("--"):
                name, eq, value = a[2:].partition("=")
                if name in ("no-headers", "no-heading"):
                    headers = False
                    continue
                if name not in ("pid", "format"):
                    raise Gap(f"ps {a}")
                if not eq:
                    if i >= len(args):
                        raise PsError(f"error: option --{name} requires an argument")
                    value, i = args[i], i + 1
                if name == "pid":
                    pids = (pids or set()) | _pid_list(value)
                else:
                    fields += _ps_fields(value)
                continue
            if not a.startswith("-") or len(a) < 2:
                raise Gap(f"ps {' '.join(args)}")
            for j, c in enumerate(a[1:], 1):
                if c in "eA":
                    every = True
                elif c == "f":
                    full = True
                elif c == "w":
                    pass
                elif c in "opC":
                    value = a[j + 1 :]
                    if not value:
                        if i >= len(args) or (c == "p" and args[i].startswith("-")):
                            what = {
                                "o": "error: improper format list",
                                "p": "error: list of process IDs must follow -p",
                                "C": "error: list of command names must follow -C",
                            }
                            raise PsError(what[c])
                        value, i = args[i], i + 1
                    if c == "o":
                        fields += _ps_fields(value)
                    elif c == "p":
                        pids = (pids or set()) | _pid_list(value)
                    else:
                        names = (names or set()) | set(value.replace(",", " ").split())
                    break
                else:
                    raise Gap(f"ps {' '.join(args)}")
        if not every and pids is None and names is None:
            raise Gap(f"ps {' '.join(args)} (no -e, -p or -C)")
        procs = self._procs(host)
        if not every:
            procs = [
                p
                for p in procs
                if (pids is not None and p.pid in pids) or (names is not None and p.comm in names)
            ]
        now = self.now()
        if sort:
            procs = _ps_sort(procs, sort, now)
        if not fields:
            fields = (
                ["uid=UID", "pid", "ppid", "c", "stime", "tty", "time", "cmd"]
                if full
                else ["pid", "tty", "time", "comm=CMD"]
            )
        cols: list[tuple[str, bool, list[str]]] = []  # header, right-aligned, values
        for spec in fields:
            key, eq, label = spec.partition("=")
            if key not in PS_FIELDS:
                raise PsError(f'error: unknown user-defined format specifier "{key}"')
            default, right = PS_FIELDS[key]
            cols.append((label if eq else default, right, [_ps_value(key, p, now) for p in procs]))
        lines: list[str] = []
        if headers and any(h for h, _, _ in cols):
            lines.append(_ps_row([(h, r) for h, r, _ in cols], cols))
        for n in range(len(procs)):
            lines.append(_ps_row([(v[n], r) for _, r, v in cols], cols))
        return Result("".join(line + "\n" for line in lines), exit=0 if procs else 1, probe=True)

    # df / du ---------------------------------------------------------------

    def _df(self, host: str, args: list[str], stdin: str | None = None) -> Result:
        if args == ["--version"]:
            return Result("df (GNU coreutils) 9.4\n", probe=True)
        opts, pos = getopt(
            "df",
            args,
            short="hHTkPlaxim",
            short_valued="Bt",
            long=frozenset(
                {
                    "human-readable",
                    "si",
                    "print-type",
                    "local",
                    "all",
                    "portability",
                    "inodes",
                    "total",
                    "sync",
                    "no-sync",
                }
            ),
            long_valued=frozenset({"output", "block-size", "type", "exclude-type"}),
        )
        files = sum(1 for key in self.snapshot() if key.startswith(host + ":"))
        size, used = self.disks[host], self.disk_used(host)
        itotal = size // 16
        iused = 61_000 + files
        human = "h" in opts or "human-readable" in opts or "H" in opts or "si" in opts
        spec = "1M" if "m" in opts else opts.get("B", opts.get("block-size", "1K"))
        block = _block_size(spec)
        rows: list[dict[str, int | str]] = [
            {
                "source": "/dev/mapper/vg0-root",
                "fstype": "ext4",
                "size": size,
                "used": used,
                "itotal": itotal,
                "iused": iused,
                "target": "/",
            },
            {
                "source": "tmpfs",
                "fstype": "tmpfs",
                "size": 1024 * 1024,
                "used": 0,
                "itotal": 262144,
                "iused": 1,
                "target": "/run",
            },
        ]
        if pos:
            rows = rows[:1]
        if "output" in opts:
            cols = [c for c in opts["output"].split(",") if c] or [
                "source",
                "fstype",
                "itotal",
                "iused",
                "iavail",
                "ipcent",
                "size",
                "used",
                "avail",
                "pcent",
                "file",
                "target",
            ]
        elif "i" in opts or "inodes" in opts:
            cols = ["source", "itotal", "iused", "iavail", "ipcent", "target"]
        else:
            cols = (
                ["source"]
                + (["fstype"] if "T" in opts or "print-type" in opts else [])
                + ["size", "used", "avail", "pcent", "target"]
            )
        posix = "P" in opts or "portability" in opts
        heads = {
            "source": "Filesystem",
            "fstype": "Type",
            "itotal": "Inodes",
            "iused": "IUsed",
            "iavail": "IFree",
            "ipcent": "IUse%",
            "size": "Size"
            if human
            else f"{'1024' if posix and block[2] == '1K' else block[2]}-blocks",
            "used": "Used",
            "avail": "Avail" if human or "output" in opts else "Available",
            "pcent": "Capacity" if posix else "Use%",
            "file": "File",
            "target": "Mounted on",
        }
        table: list[list[str]] = [[heads.get(c, "") for c in cols]]
        for c in cols:
            if c not in heads:
                return Result(
                    stderr=f"df: option --output: field '{c}' unknown\nTry 'df --help' for more information.\n",
                    exit=1,
                )
        for r in rows:
            sz, u = int(r["size"]), int(r["used"])
            it, iu = int(r["itotal"]), int(r["iused"])

            def amount(n: int) -> str:
                return _human(n * 1024) if human else _blocks(n * 1024, block)

            cell = {
                "source": str(r["source"]),
                "fstype": str(r["fstype"]),
                "target": str(r["target"]),
                "file": pos[0] if pos else "-",
                "size": amount(sz),
                "used": amount(u),
                "avail": amount(sz - u),
                "pcent": f"{(u * 100 + sz - 1) // sz}%",
                "itotal": str(it),
                "iused": str(iu),
                "iavail": str(it - iu),
                "ipcent": f"{(iu * 100 + it - 1) // it}%",
            }
            table.append([cell[c] for c in cols])
        widths = [max(len(row[i]) for row in table) for i in range(len(cols))]
        left = {"source", "fstype", "file", "target"}
        lines = [
            " ".join(
                v.ljust(widths[i])
                if cols[i] in left and i < len(cols) - 1
                else v
                if cols[i] in left
                else v.rjust(widths[i])
                for i, v in enumerate(row)
            ).rstrip()
            for row in table
        ]
        return Result("\n".join(lines) + "\n", probe=True)

    def _du(self, host: str, args: list[str], stdin: str | None = None) -> Result:
        if args == ["--version"]:
            return Result("du (GNU coreutils) 9.4\n", probe=True)
        opts, pos = getopt(
            "du",
            args,
            short="shacxbkmLPD",
            short_valued="dB",
            long=frozenset(
                {
                    "summarize",
                    "human-readable",
                    "all",
                    "total",
                    "apparent-size",
                    "one-file-system",
                    "si",
                    "bytes",
                    "dereference",
                    "no-dereference",
                }
            ),
            long_valued=frozenset(
                {"max-depth", "exclude", "files0-from", "block-size", "threshold"}
            ),
            multi=frozenset({"exclude"}),
        )
        excludes = [x for x in opts.get("exclude", "").split("\n") if x]
        if "threshold" in opts:
            raise Gap("du --threshold")
        if "files0-from" in opts:
            if pos:
                return Result(
                    stderr=f"du: extra operand '{pos[0]}'\nfile operands cannot be combined with --files0-from\n",
                    exit=1,
                )
            src = opts["files0-from"]
            names = (
                (stdin or "")
                if src == "-"
                else self.read(host, posixpath.join(getattr(self.tls, "cwd", "/root"), src))
            )
            if names is None:
                return Result(
                    stderr=f"du: cannot open '{src}' for reading: No such file or directory\n",
                    exit=1,
                )
            pos = [n for n in names.split("\0") if n]
        # the last of -h / -b / -k / -m / -B wins, as in GNU du
        unit_flags = [
            a
            for a in args
            if a in ("-h", "--human-readable", "--si", "-b", "--bytes", "-k", "-m")
            or a.startswith(("-B", "--block-size"))
            or (a.startswith("-") and not a.startswith("--") and any(c in a for c in "hbkmB"))
        ]
        last = unit_flags[-1] if unit_flags else ""
        human = bool(last) and (
            last in ("--human-readable", "--si")
            or (not last.startswith("--") and "h" in last and "B" not in last.split("h")[-1])
        )
        block: tuple[int, str, str] | None = None
        if (
            not human
            and ("B" in opts or "block-size" in opts)
            and ("B" in last or "block-size" in last)
        ):
            block = _block_size(opts.get("B", opts.get("block-size", "1K")))
        depth_s = opts.get("d", opts.get("max-depth", ""))
        if depth_s and not depth_s.isdigit():
            raise Gap(f"du -d {depth_s}")
        max_depth = 0 if ("s" in opts or "summarize" in opts) else int(depth_s) if depth_s else None
        show_files = "a" in opts or "all" in opts
        unit = 1 if "b" in opts or "bytes" in opts else 1024 * 1024 if "m" in opts else 1024
        cwd = getattr(self.tls, "cwd", "/root")
        out: list[str] = []
        err = ""
        grand = 0

        def fmt(n: int) -> str:
            if human:
                return _human(n)
            if block is not None:
                return _blocks(n, block)
            return str(n if unit == 1 else -(-n // unit))

        def walk(path: str, shown: str, depth: int) -> int:
            top = path.strip("/").split("/", 1)[0]
            if (
                top in MOUNT_POINTS or top in SYSTEM_DIRS
            ):  # the real system is opaque; /usr has a fixed size
                if path.strip("/") != top:
                    raise Gap(f"du inside /{top}")
                size = SYSTEM_KIB * 1024 if top == "usr" else 0
                if max_depth is None or depth <= max_depth:
                    out.append(f"{fmt(size)}\t{shown}")
                return size
            if depth and any(
                fnmatch.fnmatch(posixpath.basename(path), x) or fnmatch.fnmatch(path, x)
                for x in excludes
            ):
                return 0
            real = self.real_path(host, path)
            if not real.exists():  # dangling symlink
                real = self.hosts_dir / host / path.lstrip("/")
            if real.is_symlink() or not real.is_dir():
                size = (
                    real.lstat().st_size
                    if real.is_symlink()
                    else max(4096, -(-real.stat().st_size // 4096) * 4096)
                )
                if show_files or depth == 0:
                    out.append(f"{fmt(size)}\t{shown}")
                return size
            total = 4096
            for name in sorted(os.listdir(real)):
                total += walk(posixpath.join(path, name), posixpath.join(shown, name), depth + 1)
            if max_depth is None or depth <= max_depth:
                out.append(f"{fmt(total)}\t{shown}")
            return total

        for target in pos or ["."]:
            abs_path = posixpath.normpath(posixpath.join(cwd, target))
            real = self.real_path(host, abs_path)
            lexists = os.path.lexists(self.hosts_dir / host / abs_path.lstrip("/"))
            if not real.exists() and not lexists:
                err += f"du: cannot access '{target}': No such file or directory\n"
                continue
            grand += walk(abs_path, target.rstrip("/") or "/", 0)
        if "c" in opts or "total" in opts:
            out.append(f"{fmt(grand)}\ttotal")
        return Result("".join(line + "\n" for line in out), err, 1 if err else 0, probe=True)

    def _docker_compose(self, host: str, args: list[str], stdin: str | None = None) -> Result:
        return self._docker(host, ["compose", *args], stdin)

    def _gh(self, host: str, args: list[str], stdin: str | None = None) -> Result:
        if self.gh is None:
            return Result(stderr="bash: gh: command not found\n", exit=127)
        return self.gh.handle(host, args, stdin)

    # Stubs by program name. Relay handlers get binary stdin for what they pass to another
    # host (ssh, pct exec, docker exec); text handlers get it decoded.
    RELAY_HANDLERS: dict[str, Callable[[Simulator, str, list[str], Stdin], Result]] = {}
    TEXT_HANDLERS: dict[str, Callable[[Simulator, str, list[str], str | None], Result]] = {}


Simulator.RELAY_HANDLERS = {
    "ssh": Simulator._ssh,
    "pct": Simulator._pct,
    "docker": Simulator._docker,
}
Simulator.TEXT_HANDLERS = {
    "systemctl": Simulator._systemctl,
    "service": Simulator._service,
    "journalctl": Simulator._journalctl,
    "nginx": Simulator._nginx,
    "curl": Simulator._curl,
    "ss": Simulator._ss,
    "ps": Simulator._ps,
    "df": Simulator._df,
    "du": Simulator._du,
    "docker-compose": Simulator._docker_compose,
    "gh": Simulator._gh,
}
STUB_NAMES = (*Simulator.RELAY_HANDLERS, *Simulator.TEXT_HANDLERS)
