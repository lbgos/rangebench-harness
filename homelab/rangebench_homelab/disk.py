"""Disk bounds for one world: real bytes on the machine running the bench and each host's
simulated `df`.

The sandbox runs commands on real directories, so nothing else stops a model from filling
that machine's disk. A DiskGuard per world holds two limits:

- the world's real usage (all hosts, git remotes, gh state; ENTRY_BYTES minimum per entry)
  stays under WORLD_CAP_MB;
- each host's simulated used space (what `df` shows: SYSTEM_KIB plus apparent file sizes)
  stays under its `sim.disks` size, or under what the template built if that is more.

While commands run, the guard walks the world every POLL_S and kills every running command
of the world (agent command, ssh/pct hops, git streams) once a value grows past its limit;
the command's output ends with a note, like the memory and output caps. A walk at the end of
each batch of commands catches commands too quick for the poll. While the world is over its
cap or a host over its disk, commands there start with RLIMIT_FSIZE 0 (`no_writes`, SIGXFSZ
ignored), so every write that would add data fails with "File too large" until a command that
removes files has ended. Reads and deletes still work.

    guard = DiskGuard(sim)
    if guard.full(host): argv = no_writes(argv)
    with guard.watch(kill) as w:     # kill(): stop this command and its children
        ... run the command ...
    notes += w.note                  # "" unless the command pushed a value past its limit
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .sim import MOUNT_POINTS, SYSTEM_DIRS, SYSTEM_KIB

if TYPE_CHECKING:
    from .sim import Simulator

# Real bytes one world may hold: a run with concurrency 4 needs up to 2 GB free.
WORLD_CAP_MB = 512
ENTRY_BYTES = 4096  # each file, dir or link costs at least an inode and a block
# Walk interval while commands run (overshoot ~ write speed x POLL_S);
# 4x the walk time on big worlds.
POLL_S = 0.05


def no_writes(argv: list[str]) -> list[str]:
    """argv run under RLIMIT_FSIZE 0 with SIGXFSZ ignored: writes that add data fail with EFBIG."""
    return ["sh", "-c", 'trap "" XFSZ; ulimit -f 0 && exec "$@"', "sh", *argv]


@dataclass
class Usage:
    real: int  # bytes on the real disk
    hosts: dict[str, int] = field(default_factory=dict)  # simulated used KiB of each host's /


def measure(sim: Simulator) -> Usage:
    """Real usage of the world dir and each host's simulated `df` used (sim.disk_used,
    unclamped)."""
    real = 0
    apparent: dict[str, int] = {}
    seen: set[int] = set()
    hosts_dir = str(sim.hosts_dir)
    stack: list[tuple[str, str, bool]] = [(str(sim.root), "", False)]  # (dir, host, is host root)
    while stack:
        path, host, top = stack.pop()
        try:
            entries = os.scandir(path)
        except OSError:
            continue
        with entries:
            for e in entries:
                try:
                    st = e.stat(follow_symlinks=False)
                    is_dir = e.is_dir(follow_symlinks=False)
                except OSError:
                    continue
                if st.st_nlink > 1 and not is_dir:
                    if st.st_ino in seen:
                        continue
                    seen.add(st.st_ino)
                real += max(st.st_blocks * 512, ENTRY_BYTES)
                if is_dir:
                    if path == hosts_dir:
                        stack.append((e.path, e.name, True))
                    elif not (top and e.name in (*SYSTEM_DIRS, *MOUNT_POINTS)):
                        stack.append((e.path, host, False))
                elif host and not (top and e.name in SYSTEM_DIRS):
                    apparent[host] = apparent.get(host, 0) + st.st_size
    return Usage(real, {h: SYSTEM_KIB + apparent.get(h, 0) // 1024 for h in sim.hosts})


@dataclass
class Watch:
    kill: Callable[[], None]
    note: str = ""


class DiskGuard:
    """Holds one world under its disk limits (see module doc). close() stops its thread."""

    def __init__(self, sim: Simulator, cap_mb: int = WORLD_CAP_MB) -> None:
        self.sim = sim
        self.cap = cap_mb * 2**20
        self.usage = measure(sim)  # latest walk
        self.limits = {h: max(sim.disks.get(h, used), used) for h, used in self.usage.hosts.items()}
        self.floor = self.usage  # lowest usage seen since the current batch began
        self.cond = threading.Condition()
        self.active: list[Watch] = []
        self.closed = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def full(self, host: str) -> bool:
        """The latest walk has the world over its cap or host over its disk."""
        u = self.usage
        return u.real > self.cap or u.hosts.get(host, 0) > self.limits.get(
            host, u.hosts.get(host, 0)
        )

    @contextlib.contextmanager
    def watch(self, kill: Callable[[], None]) -> Iterator[Watch]:
        w = Watch(kill)
        with self.cond:
            if not self.active:  # a new batch: the latest walk is its baseline
                self.floor = self.usage
            self.active.append(w)
            self.cond.notify()
        try:
            yield w
        finally:
            with self.cond:
                self.active.remove(w)
                last = not self.active
            if last:  # catch what was written after the last poll
                self._check([w], kill_too=False)

    def close(self) -> None:
        with self.cond:
            self.closed = True
            self.cond.notify()

    def _over(self, u: Usage) -> str:
        """The note for a world that grew past a limit during this batch, else ""."""
        f = self.floor
        if u.real > max(self.cap, f.real):
            cap_mb = self.cap // 2**20
            return f"\n[no space left on device: this world holds at most {cap_mb} MB of files]\n"
        for host, used in u.hosts.items():
            if used > max(self.limits.get(host, used), f.hosts.get(host, 0)):
                return f"\n[no space left on device: / on {host} is full]\n"
        return ""

    def _check(self, watches: list[Watch] | None, kill_too: bool = True) -> None:
        """Walk the world; note (and kill) the given watches, or all active ones, on growth
        past a limit."""
        u = measure(self.sim)
        with self.cond:
            self.usage = u
            note = self._over(u)
            for w in self.active if watches is None else watches:
                if note and not w.note:
                    w.note = note + ("[command killed]\n" if kill_too else "")
                if note and kill_too:
                    w.kill()
            self.floor = Usage(
                min(self.floor.real, u.real),
                {h: min(used, self.floor.hosts.get(h, used)) for h, used in u.hosts.items()},
            )

    def _loop(self) -> None:
        while True:
            with self.cond:
                while not self.active and not self.closed:
                    self.cond.wait()
                if self.closed:
                    return
            started = time.monotonic()
            self._check(None)
            time.sleep(max(POLL_S, 4 * (time.monotonic() - started)))
