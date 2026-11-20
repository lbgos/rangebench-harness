#!/usr/bin/python3 -I
"""Stub command client, run inside the sandbox under the names in sim.STUB_NAMES.

Sends argv, cwd and (for ssh/pct) stdin, base64 so binary streams survive, to the state
server over /run/rb/sock and prints its answer. `sudo` runs the rest of its argv directly.

ssh carrying git's pack protocol (git push/fetch/clone with host:/path remotes) streams:
after the request line, stdin flows raw to the server (half-closed at EOF) and the server
answers with frames of 1 channel byte (1 stdout, 2 stderr, 0 exit code) + 4-byte length.
"""

import base64
import contextlib
import json
import os
import re
import signal
import socket
import struct
import sys
import threading

SOCKET = "/run/rb/sock"
MAX_STDIN = (
    1_000_000  # same as sandbox.MAX_STDIN; read bounded so `tar c big | ssh h ...` can't eat RAM
)
GIT_PACK = re.compile(r"\bgit[- ](upload-pack|receive-pack|upload-archive)\b")


def stream(sock: socket.socket, request: dict) -> int:
    sock.sendall(json.dumps(request).encode() + b"\n")

    def pump_stdin() -> None:
        try:
            while data := os.read(0, 65536):
                sock.sendall(data)
        except OSError:
            pass
        with contextlib.suppress(OSError):
            sock.shutdown(socket.SHUT_WR)

    threading.Thread(target=pump_stdin, daemon=True).start()
    reader = sock.makefile("rb")
    while True:
        head = reader.read(5)
        if len(head) < 5:
            return 255
        channel, size = head[0], struct.unpack(">I", head[1:])[0]
        payload = reader.read(size)
        if channel == 0:
            return int(payload)
        fd = 1 if channel == 1 else 2
        while payload:
            payload = payload[os.write(fd, payload) :]


def wants_stdin(prog: str, args: list) -> bool:
    """ssh/pct always; `docker exec -i`; `gh ... --body-file -` (others must not eat a loop's
    stdin)."""
    if prog in ("ssh", "pct"):
        return True
    if prog == "docker" and args[:1] == ["exec"]:
        for a in args[1:]:
            if not a.startswith("-"):
                return False
            if a == "--interactive" or re.fullmatch(r"-[a-z]*i[a-z]*", a):
                return True
    return prog == "gh" and "-" in args


def main() -> int:
    signal.signal(
        signal.SIGPIPE, signal.SIG_DFL
    )  # `ssh h cat big | head` ends quietly, like real ssh
    prog = os.path.basename(sys.argv[0])
    args = sys.argv[1:]
    if prog == "sudo":
        while args and args[0].startswith("-"):
            flag = args.pop(0)
            if flag in ("-u", "-g") and args:
                args.pop(0)
        if not args:
            return 0
        os.execvp(args[0], args)
    request = {"argv": [prog, *args], "cwd": os.getcwd(), "stdin": None}
    if prog == "ssh" and GIT_PACK.search(" ".join(args)):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(SOCKET)
            return stream(sock, {**request, "stream": True})
    if wants_stdin(prog, args) and not sys.stdin.isatty():
        try:
            data = sys.stdin.buffer.read(MAX_STDIN + 1)
        except (OSError, ValueError):
            data = b""
        if len(data) > MAX_STDIN:
            sys.stderr.write(
                f"{prog}: input over {MAX_STDIN // 1_000_000} MB is not supported here; send "
                "smaller pieces\n"
            )
            return 1
        request["stdin_b64"] = base64.b64encode(data).decode()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(SOCKET)
        sock.sendall(json.dumps(request).encode() + b"\n")
        chunks = []
        while True:
            data = sock.recv(65536)
            if not data:
                break
            chunks.append(data)
    reply = json.loads(b"".join(chunks))
    sys.stdout.write(reply["stdout"])
    sys.stdout.flush()
    sys.stderr.write(reply["stderr"])
    return int(reply["exit"])


if __name__ == "__main__":
    sys.exit(main())
