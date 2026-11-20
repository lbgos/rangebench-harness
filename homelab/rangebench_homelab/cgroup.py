"""Per-command memory limits through cgroup v2 children of the run's delegated scope.

Start a run in a delegated systemd scope (`systemd-run --scope -p MemoryMax=2G -p Delegate=yes
python3 -m rangebench_homelab ...`). `limiter()` moves the
processes of that scope into `<scope>/harness`, enables the memory controller for its
children, and returns a Limiter; the sandbox then starts each command in its own
`<scope>/cmd-<pid>-<n>` with memory.max set, so a command that allocates past its limit
is OOM-killed alone and the run goes on. memory.max counts resident memory and page
cache, so node's large virtual mappings are fine. Outside a delegated scope (no
systemd-run, or a scope without Delegate=yes) `limiter()` returns None and commands
run without their own limit.

    lim = limiter()
    cg = lim.create(768) if lim else None
    argv = lim.wrap(cg, argv)        # the command joins cg before it execs
    ...
    lim.oom_killed(cg); lim.remove(cg)   # remove also reclaims scope cache above SCOPE_SOFT
"""

from __future__ import annotations

import contextlib
import itertools
import os
import threading
import time
from pathlib import Path

CGROOT = Path("/sys/fs/cgroup")
CMD_MEMORY_MB = 768  # per sandboxed command; the run's scope has 2G
# Page cache a command leaves above this is reclaimed when it ends. Reading a sparse
# multi-GB dump fills the cache with zero pages; without the trim they would move to the
# scope with the removed cgroup and pile up there until the 2G cap forces reclaim.
TRIM_ABOVE = 64 * 2**20
# What each removed command leaves (up to TRIM_ABOVE) is charged to the scope, and
# parallel seeds each read their own dumps; when a command ends with the whole scope above
# SCOPE_SOFT, its cache is reclaimed down to SCOPE_FLOOR so peaks stay off the 2G cap.
# Swap is off, so reclaim takes only cache, never the harness's or a command's memory.
SCOPE_SOFT = 1024 * 2**20
SCOPE_FLOOR = 640 * 2**20

_lock = threading.Lock()
_cached: list[Limiter | None] = []  # one entry once set up; forked workers share it


def own_cgroup() -> Path | None:
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("0::"):
            return CGROOT / line[3:].lstrip("/")
    return None


def delegated(cg: Path) -> bool:
    """systemd marks a Delegate=yes unit's cgroup with a delegate xattr."""
    for attr in ("trusted.delegate", "user.delegate"):
        try:
            if os.getxattr(cg, attr) == b"1":
                return True
        except OSError:
            pass
    return False


class Limiter:
    def __init__(self, scope: Path) -> None:
        self.scope = scope
        self.counter = itertools.count(1)

    def create(self, limit_mb: int) -> Path:
        cg = self.scope / f"cmd-{os.getpid()}-{next(self.counter)}"
        cg.mkdir()
        (cg / "memory.max").write_text(f"{limit_mb}M")
        swap = cg / "memory.swap.max"
        if swap.exists():
            swap.write_text("0")
        (cg / "memory.oom.group").write_text("1")  # kill the whole command, not one process of it
        return cg

    @staticmethod
    def wrap(cg: Path, argv: list[str]) -> list[str]:
        """argv that moves itself into cg, then execs (same pid, so killpg still works)."""
        return ["sh", "-c", 'echo $$ > "$0/cgroup.procs" && exec "$@"', str(cg), *argv]

    @staticmethod
    def oom_killed(cg: Path) -> bool:
        try:
            text = (cg / "memory.events").read_text()
        except OSError:
            return False
        events = dict(line.split() for line in text.splitlines() if line.strip())
        return int(events.get("oom_kill", "0")) > 0 or int(events.get("oom_group_kill", "0")) > 0

    @staticmethod
    def kill(cg: Path) -> None:
        with contextlib.suppress(OSError):
            (cg / "cgroup.kill").write_text("1")

    @staticmethod
    def trim(cg: Path, above: int = TRIM_ABOVE, down_to: int = TRIM_ABOVE // 2) -> int:
        """Reclaim the cgroup's memory down to down_to once it is above `above`; returns
        memory.current after."""
        try:
            current = int((cg / "memory.current").read_text())
            if current > above:
                (cg / "memory.reclaim").write_text(str(current - down_to))
        except OSError:
            pass  # EAGAIN: reclaimed less than asked
        try:
            return int((cg / "memory.current").read_text())
        except OSError:
            return 0

    def remove(self, cg: Path) -> None:
        """Kill what is left, trim its cache and remove the cgroup; a stuck one is left for the
        scope's end."""
        self.kill(cg)
        self.trim(cg)
        for _ in range(50):
            try:
                cg.rmdir()
                break
            except FileNotFoundError:
                break
            except OSError:
                time.sleep(0.02)
        self.relieve()

    def relieve(self, soft: int = SCOPE_SOFT, floor: int = SCOPE_FLOOR) -> int:
        """Reclaim the scope's cache down to floor once it is above soft; returns
        memory.current after."""
        return self.trim(self.scope, above=soft, down_to=floor)


def _setup() -> Limiter | None:
    cg = own_cgroup()
    if cg is None:
        return None
    scope = cg.parent if cg.name == "harness" else cg
    if not delegated(scope):
        return None
    harness = scope / "harness"
    try:
        harness.mkdir(exist_ok=True)
        if cg != harness:
            # A cgroup that hands controllers to children may hold no processes itself.
            for pid in (scope / "cgroup.procs").read_text().split():
                with contextlib.suppress(OSError):  # exited meanwhile
                    (harness / "cgroup.procs").write_text(pid)
        if "memory" not in (scope / "cgroup.subtree_control").read_text().split():
            (scope / "cgroup.subtree_control").write_text("+memory")
    except OSError:
        return None
    return Limiter(scope)


def limiter() -> Limiter | None:
    """The process-wide Limiter, set up on first use; None without a delegated scope."""
    with _lock:
        if not _cached:
            _cached.append(_setup())
        return _cached[0]
