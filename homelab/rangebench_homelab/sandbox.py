"""bwrap sandbox per host plus the state server that answers stub commands.

Usage: `with Sandbox(sim, workdir) as sb: sb.run(host, "cmd", timeout=60)`.
Every command runs in a fresh bwrap (--unshare-all, so no network) whose root is
the host's world directory, with the real /usr and lib dirs read-only on top. Stubs on PATH
reach the state server through /run/rb/sock; one socket per host tells the server which
host a stub runs on. The server lives in this process, outside the sandbox.
Each command also gets its own memory limit (cgroup.py, default CMD_MEMORY_MB) when the
run is in a delegated scope; over it, the command is killed and its result says so.
Disk: a DiskGuard per world (disk.py) kills running commands when the world passes
WORLD_CAP_MB of real data or a host passes its simulated disk; while one is passed, new
commands there run with RLIMIT_FSIZE 0, so writes fail until files are removed.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import shutil
import signal
import socket
import socketserver
import struct
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import TracebackType

from .cgroup import CMD_MEMORY_MB, limiter
from .disk import WORLD_CAP_MB, DiskGuard, no_writes
from .sim import STUB_NAMES, SYSTEM_DIRS, Result, Simulator, Stdin

STUB_SOURCE = Path(__file__).with_name("stub.py")
PATH = "/run/rb/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
MAX_STDIN = 1_000_000
MAX_OUTPUT = (
    4_000_000  # bytes kept per stream of one command; the agent sees far less (session.truncate)
)
# /usr/local is masked (it often holds host-specific tools); only these come through.
LOCAL_TOOLS = ("node", "npm", "npx", "pnpm", "corepack")


def local_tool_args() -> list[str]:
    """bwrap args exposing LOCAL_TOOLS (symlinks into /opt/<pkg>) on an empty /usr/local."""
    args = ["--tmpfs", "/usr/local"]
    roots: set[str] = set()
    for tool in LOCAL_TOOLS:
        link = Path("/usr/local/bin") / tool
        if not link.is_symlink():
            continue
        target = os.readlink(link)
        args += ["--symlink", target, str(link)]
        parts = Path(target).parts
        if len(parts) > 2 and parts[1] == "opt":
            roots.add(str(Path(*parts[:3])))
    for root in sorted(roots):
        if Path(root).exists():
            args += ["--ro-bind", os.path.realpath(root), root]
    return args


def _stdin(req: dict[str, object]) -> Stdin:
    """A request's stdin: `stdin_b64` (bytes, as text when it is UTF-8) or a plain `stdin`
    string."""
    raw = req.get("stdin_b64")
    if isinstance(raw, str):
        data = base64.b64decode(raw)[:MAX_STDIN]
        try:
            return data.decode()
        except UnicodeDecodeError:
            return data
    stdin = req.get("stdin")
    return stdin if isinstance(stdin, str) else None


class _Handler(socketserver.StreamRequestHandler):
    server: _HostServer

    def handle(self) -> None:
        try:
            line = self.rfile.readline(MAX_STDIN * 4 // 3 + 65536)
        except OSError:
            return  # the stub was killed while sending
        try:
            req = json.loads(line)
            argv = [str(a) for a in req["argv"]]
            if req.get("stream"):
                self._stream(argv, str(req.get("cwd") or "/root"))
                return
            res = self.server.sim.handle(
                self.server.host, argv, str(req.get("cwd") or "/root"), _stdin(req)
            )
        except Exception as exc:  # a stub must always get an answer
            res = Result(stderr=f"internal simulator error: {exc}\n", exit=70)
        reply = {"stdout": res.stdout, "stderr": res.stderr, "exit": res.exit}
        with contextlib.suppress(
            OSError
        ):  # the stub was killed (timeout, output cap) before its answer came
            self.wfile.write(json.dumps(reply).encode())

    def _stream(self, argv: list[str], cwd: str) -> None:
        """git pack protocol over ssh: raw stdin in, framed stdout/stderr/exit out (see stub.py)."""
        lock = threading.Lock()

        def frame(channel: int, data: bytes) -> None:
            with lock:
                try:
                    self.wfile.write(bytes([channel]) + struct.pack(">I", len(data)) + data)
                    self.wfile.flush()
                except OSError:
                    pass

        def pump(target: str, command: str) -> int:
            return self.server.sandbox.pipe(target, command, self.rfile, frame)

        res = self.server.sim.handle_stream(self.server.host, argv, cwd, pump)
        if res.stdout:
            frame(1, res.stdout.encode())
        if res.stderr:
            frame(2, res.stderr.encode())
        frame(0, str(res.exit).encode())


class _HostServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

    def handle_error(
        self, request: socket.socket | tuple[bytes, socket.socket], client_address: object
    ) -> None:
        """A client that hung up is normal (killed commands); anything else still prints."""
        if not isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            super().handle_error(request, client_address)

    def __init__(self, path: str, sandbox: Sandbox, host: str) -> None:
        self.sim = sandbox.sim
        self.sandbox = sandbox
        self.host = host
        super().__init__(path, _Handler)


class Sandbox:
    """Runs commands on simulated hosts and serves their stub calls."""

    def __init__(
        self,
        sim: Simulator,
        workdir: Path,
        timeout: float = 60.0,
        memory_mb: int = CMD_MEMORY_MB,
        disk_mb: int = WORLD_CAP_MB,
    ) -> None:
        self.sim = sim
        self.timeout = timeout
        self.memory_mb = memory_mb
        self.guard = DiskGuard(sim, disk_mb)
        self.limiter = limiter()
        self.deadline = 0.0  # absolute deadline of the current agent command
        self.sockdir = Path(tempfile.mkdtemp(prefix="rb-"))  # short path: sun_path limit
        self.bindir = workdir / "bin"
        self.local_tools = local_tool_args()
        self.servers: list[_HostServer] = []
        self.threads: list[threading.Thread] = []

    def __enter__(self) -> Sandbox:
        self.bindir.mkdir(parents=True, exist_ok=True)
        shutil.copy(STUB_SOURCE, self.bindir / "_stub")
        os.chmod(self.bindir / "_stub", 0o755)
        for name in (*STUB_NAMES, "sudo"):
            link = self.bindir / name
            if not link.exists():
                link.symlink_to("_stub")
        for host in self.sim.hosts:
            server = _HostServer(str(self.sockdir / f"{host}.sock"), self, host)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.servers.append(server)
            self.threads.append(thread)
        self.sim.exec_hook = self._remote
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.sim.exec_hook = None
        self.guard.close()
        for server in self.servers:
            server.shutdown()
            server.server_close()
        shutil.rmtree(self.sockdir, ignore_errors=True)

    def argv(self, host: str, cwd: str, command: str) -> list[str]:
        base = self.sim.hosts_dir / host
        # The host's world dir is / (one mount, so find -xdev and du -x see it all);
        # the real system is mounted read-only on the skeleton sim.add_host made.
        args = [
            "bwrap",
            "--die-with-parent",
            "--unshare-all",
            "--hostname",
            host,
            "--bind",
            str(base),
            "/",
        ]
        for name in SYSTEM_DIRS:  # includes usr
            src = Path("/") / name
            if src.is_dir() and not src.is_symlink():
                args += ["--ro-bind", str(src), f"/{name}"]
        args += self.local_tools
        args += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/run"]
        args += [
            "--ro-bind",
            str(self.bindir),
            "/run/rb/bin",
            "--ro-bind",
            str(self.sockdir / f"{host}.sock"),
            "/run/rb/sock",
            "--clearenv",
            "--setenv",
            "PATH",
            PATH,
            "--setenv",
            "HOME",
            "/root",
            "--setenv",
            "USER",
            "root",
            "--setenv",
            "LANG",
            "C.UTF-8",
            "--setenv",
            "TERM",
            "dumb",
            "--setenv",
            "HOSTNAME",
            host,
            *self.git_env(),
            "--chdir",
            cwd if self.sim.real_path(host, cwd).is_dir() else "/",
            "bash",
            "-c",
            command,
        ]
        return args

    def _spawn(
        self, host: str, argv: list[str], stdout: int, stderr: int
    ) -> tuple[subprocess.Popen[bytes], Path | None]:
        """Start argv in its own memory cgroup when the run has one; no writes while host's
        disk is full."""
        cg = self.limiter.create(self.memory_mb) if self.limiter else None
        if self.guard.full(host):
            argv = no_writes(argv)
        if cg is not None and self.limiter:
            argv = self.limiter.wrap(cg, argv)
        proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, start_new_session=True
        )
        return proc, cg

    def _killer(self, proc: subprocess.Popen[bytes], cg: Path | None) -> Callable[[], None]:
        """Kills the command's process group and, with a cgroup, everything that left it."""

        def kill() -> None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            if cg is not None and self.limiter:
                self.limiter.kill(cg)

        return kill

    def _release(self, cg: Path | None) -> str:
        """Remove the command's cgroup; returns the note for the agent if it was OOM-killed."""
        if cg is None or not self.limiter:
            return ""
        killed = self.limiter.oom_killed(cg)
        self.limiter.remove(cg)
        return f"\n[killed: command over {self.memory_mb} MB]\n" if killed else ""

    def git_env(self) -> list[str]:
        """bwrap args that send https remotes of the GitHub mock over ssh (no network here)."""
        if self.sim.gh is None:
            return []
        gh = self.sim.gh.host
        pairs = {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": f"url.git@{gh}:.insteadOf",
            "GIT_CONFIG_VALUE_0": f"https://{gh}/",
        }
        return [arg for k, v in pairs.items() for arg in ("--setenv", k, v)]

    def run(
        self,
        host: str,
        command: str,
        cwd: str = "/root",
        stdin: Stdin = None,
        timeout: float | None = None,
        merge: bool = False,
    ) -> Result:
        """Run bash -c command on host. merge puts stderr into stdout (agent view)."""
        limit = timeout if timeout is not None else self.timeout
        proc, cg = self._spawn(
            host,
            self.argv(host, cwd, command),
            subprocess.PIPE,
            subprocess.STDOUT if merge else subprocess.PIPE,
        )
        assert proc.stdin and proc.stdout
        overflow = threading.Event()
        kill = self._killer(proc, cg)

        def collect(stream: io.BufferedReader, buf: bytearray) -> None:
            # Bounded: `cat` of a sparse multi-GB dump must not land in this process's memory.
            while data := stream.read1(65536):
                if len(buf) + len(data) > MAX_OUTPUT:
                    buf += data[: MAX_OUTPUT - len(buf)]
                    overflow.set()
                    kill()
                    return
                buf += data

        out, err = bytearray(), bytearray()
        readers = [threading.Thread(target=collect, args=(proc.stdout, out), daemon=True)]
        if proc.stderr:
            readers.append(threading.Thread(target=collect, args=(proc.stderr, err), daemon=True))
        for t in readers:
            t.start()
        timed_out = False
        with self.guard.watch(kill) as watch:
            try:
                data = stdin.encode() if isinstance(stdin, str) else stdin or b""
                proc.stdin.write(data[:MAX_STDIN])
                proc.stdin.close()
            except OSError:
                pass
            try:
                code = proc.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                kill()
                code = proc.wait()
                timed_out = True
        for t in readers:
            t.join(5)
        notes = self._release(cg) + watch.note
        if overflow.is_set():
            notes += f"\n[output over {MAX_OUTPUT // 1_000_000} MB; command killed]\n"
        if timed_out:
            notes += f"\n[timed out after {limit:.0f}s]\n"
        res = Result(
            out.decode(errors="replace"), err.decode(errors="replace"), 124 if timed_out else code
        )
        if merge:  # the agent sees stdout only
            res.stdout += notes
        else:
            res.stderr += notes
        return res

    def pipe(
        self,
        host: str,
        command: str,
        source: io.BufferedIOBase,
        frame: Callable[[int, bytes], None],
    ) -> int:
        """Run command on host with stdin from source and stdout/stderr sent as frames
        (channel 1/2) while it runs; returns the exit code. Same deadline as _remote."""
        remaining = self.deadline - time.time() if self.deadline else self.timeout
        proc, cg = self._spawn(
            host, self.argv(host, "/root", command), subprocess.PIPE, subprocess.PIPE
        )
        assert proc.stdin and isinstance(proc.stdout, io.BufferedReader) and proc.stderr
        sink = proc.stdin

        def feed() -> None:
            try:
                while data := source.read1(65536):
                    sink.write(data)
                    sink.flush()
            except (OSError, ValueError):
                pass
            with contextlib.suppress(OSError):
                sink.close()

        def drain(stream: io.BufferedReader, channel: int) -> None:
            while data := stream.read1(65536):
                frame(channel, data)

        threads = [
            threading.Thread(target=feed, daemon=True),
            threading.Thread(target=drain, args=(proc.stderr, 2), daemon=True),
        ]
        for t in threads:
            t.start()
        kill = self._killer(proc, cg)
        timer = threading.Timer(max(1.0, remaining), kill)
        timer.start()
        with self.guard.watch(kill) as watch:
            try:
                drain(proc.stdout, 1)
                code = proc.wait()
                threads[1].join()
            finally:
                timer.cancel()
                proc.stdout.close()
                proc.stderr.close()
        if note := self._release(cg) + watch.note:
            frame(2, note.encode())
        return code

    def _remote(self, host: str, command: str, stdin: Stdin) -> Result:
        """exec_hook for ssh/pct exec: same deadline as the agent command that issued it."""
        remaining = self.deadline - time.time() if self.deadline else self.timeout
        return self.run(host, command, "/root", stdin, max(1.0, remaining))
