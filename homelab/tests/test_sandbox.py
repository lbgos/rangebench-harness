import contextlib
import io
import json
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path

from rangebench_homelab.cgroup import TRIM_ABOVE, limiter
from rangebench_homelab.disk import measure
from rangebench_homelab.sandbox import MAX_OUTPUT, Sandbox
from rangebench_homelab.sim import SYSTEM_KIB, Simulator

ON_DISK = (
    Path(__file__).resolve().parents[1]
)  # worlds that write real data stay on disk: /tmp may be RAM (tmpfs)


@unittest.skipUnless(shutil.which("bwrap"), "needs bwrap")
class OutputBoundTest(unittest.TestCase):
    def test_cat_of_huge_sparse_file_is_cut_and_killed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sim = Simulator(Path(tmp) / "world")
            sim.add_host("ws", "10.0.0.2")
            sim.sparse("ws", "/var/dump.bin", 20_000_000_000)
            sim.boot()
            with Sandbox(sim, Path(tmp)) as sb:
                res = sb.run("ws", "cat /var/dump.bin", "/root", timeout=60)
                self.assertLessEqual(len(res.stdout), MAX_OUTPUT)
                self.assertIn("command killed", res.stderr)
                self.assertEqual(sb.run("ws", "echo ok", "/root").stdout, "ok\n")

    def test_ssh_stdin_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sim = Simulator(Path(tmp) / "world")
            sim.add_host("ws", "10.0.0.2")
            sim.add_host("bk", "10.0.0.3")
            sim.boot()
            with Sandbox(sim, Path(tmp)) as sb:
                big = sb.run(
                    "ws",
                    "head -c 3000000 /dev/zero | tr '\\0' a | ssh bk 'wc -c'",
                    "/root",
                    merge=True,
                )
                self.assertNotEqual(big.exit, 0)
                self.assertIn("not supported here", big.stdout)
                small = sb.run("ws", "printf abc | ssh bk 'wc -c'", "/root")
                self.assertEqual(small.stdout.strip(), "3", small.stderr)

    def test_binary_stdin_survives_ssh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sim = Simulator(Path(tmp) / "world")
            sim.add_host("ws", "10.0.0.2")
            sim.add_host("bk", "10.0.0.3")
            sim.write("ws", "/srv/data/a.txt", "payload\n")
            sim.boot()
            with Sandbox(sim, Path(tmp)) as sb:
                cmd = "tar czf - -C /srv data | ssh bk 'mkdir -p /restore && tar xzf - -C /restore && cat /restore/data/a.txt'"
                res = sb.run("ws", cmd, "/root")
                self.assertEqual((res.exit, res.stdout), (0, "payload\n"), res.stderr)
                raw = sb.run(
                    "ws",
                    "head -c 256 /dev/urandom > /tmp/r && ssh bk 'sha256sum' < /tmp/r && sha256sum < /tmp/r",
                    "/root",
                )
                remote, local = raw.stdout.splitlines()
                self.assertEqual(remote, local, raw.stderr)


@unittest.skipUnless(shutil.which("bwrap"), "needs bwrap")
class CommandMemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        if limiter() is None:
            self.skipTest(
                "needs a delegated scope: systemd-run --scope -p Delegate=yes python3 -m rangebench_homelab test"
            )

    def test_command_over_limit_is_killed_and_run_goes_on(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sim = Simulator(Path(tmp) / "world")
            sim.add_host("ws", "10.0.0.2")
            sim.add_host("bk", "10.0.0.3")
            sim.boot()
            hog = "python3 -c 'x = bytearray(400 * 2**20); print(len(x))'"
            with Sandbox(sim, Path(tmp), memory_mb=128) as sb:
                res = sb.run("ws", hog, "/root", merge=True)
                self.assertNotEqual(res.exit, 0)
                self.assertIn("[killed: command over 128 MB]", res.stdout)
                remote = sb.run("ws", f'ssh bk "{hog}"; echo after', "/root", merge=True)
                self.assertIn("[killed: command over 128 MB]", remote.stdout)
                self.assertIn("after", remote.stdout)
                self.assertEqual(sb.run("ws", "echo ok", "/root").stdout, "ok\n")
                if shutil.which(
                    "node"
                ):  # large virtual mappings stay under a resident-memory limit
                    self.assertEqual(
                        sb.run("ws", "node -e 'console.log(42)'", "/root").stdout, "42\n"
                    )
                assert sb.limiter
                self.assertEqual(list(sb.limiter.scope.glob("cmd-*")), [])

    def test_cache_left_by_a_command_is_trimmed(self) -> None:
        lim = limiter()
        assert lim
        with tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parents[1]
        ) as tmp:  # on disk, /tmp is RAM
            dump = Path(tmp) / "dump"
            with dump.open("wb") as f:
                f.truncate(300 * 2**20)  # sparse: reading it caches zero pages
            cg = lim.create(768)
            subprocess.run(lim.wrap(cg, ["cat", str(dump)]), stdout=subprocess.DEVNULL, check=True)
            before = int((cg / "memory.current").read_text())
            after = lim.trim(cg)
            lim.remove(cg)
        self.assertGreater(before, 200 * 2**20)
        self.assertLessEqual(after, TRIM_ABOVE)

    def test_scope_cache_is_reclaimed_past_the_soft_limit(self) -> None:
        lim = limiter()
        assert lim
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            dump = Path(tmp) / "dump"
            with dump.open("wb") as f:
                f.truncate(300 * 2**20)
            cg = lim.create(768)
            subprocess.run(lim.wrap(cg, ["cat", str(dump)]), stdout=subprocess.DEVNULL, check=True)
            cg.rmdir()  # untrimmed: its cache now counts against the scope
            before = int((lim.scope / "memory.current").read_text())
            untouched = lim.relieve(soft=before + 2**30)
            after = lim.relieve(soft=before - 100 * 2**20, floor=before - 250 * 2**20)
        self.assertGreater(untouched, before - 50 * 2**20, "below soft: nothing reclaimed")
        self.assertLess(after, before - 150 * 2**20)


@unittest.skipUnless(shutil.which("bwrap"), "needs bwrap")
class DiskTest(unittest.TestCase):
    """The world cap (16 MB here) holds against one command writing many files, ssh hops and
    single big files; the sim's own df holds for apparent sizes. The run goes on after each."""

    def world(self, tmp: str) -> Simulator:
        sim = Simulator(Path(tmp) / "world")
        sim.add_host("ws", "10.0.0.2")
        sim.add_host("bk", "10.0.0.3")
        sim.boot()
        return sim

    def test_loop_writing_files_past_the_cap_is_killed(self) -> None:
        with tempfile.TemporaryDirectory(dir=ON_DISK) as tmp:
            sim = self.world(tmp)
            with Sandbox(sim, Path(tmp), disk_mb=16) as sb:
                loop = (
                    "for i in $(seq 300); do head -c 500000 /dev/zero > /tmp/f$i; done; echo done"
                )
                res = sb.run("ws", loop, "/root", merge=True)
                self.assertIn("no space left on device", res.stdout)
                self.assertNotIn("done", res.stdout)
                self.assertLess(measure(sim).real, 100 * 2**20)
                full = sb.run(
                    "ws", "echo x > /tmp/more; cat /tmp/more; ls /tmp | wc -l", "/root", merge=True
                )
                self.assertIn("File too large", full.stdout)
                sb.run("ws", "rm -f /tmp/f*", "/root")
                freed = sb.run(
                    "ws", "head -c 1000000 /dev/zero > /tmp/after && echo ok", "/root", merge=True
                )
                self.assertEqual(freed.stdout, "ok\n")
                self.assertEqual(
                    sb.run("ws", "truncate -s 4G /tmp/sparse && echo ok", "/root").stdout,
                    "ok\n",
                    "sparse: no real bytes",
                )
                big = sb.run(
                    "ws", "head -c 300000000 /dev/zero > /tmp/big; echo done", "/root", merge=True
                )
                self.assertIn("no space left on device", big.stdout)
                self.assertNotIn("done", big.stdout)

    def test_ssh_dd_loop_past_the_cap_is_killed(self) -> None:
        with tempfile.TemporaryDirectory(dir=ON_DISK) as tmp:
            sim = self.world(tmp)
            with Sandbox(sim, Path(tmp), disk_mb=16) as sb:
                loop = 'for i in $(seq 60); do head -c 1000000 /dev/zero | ssh bk "dd of=/srv/w.$i bs=1M status=none"; done; echo done'
                res = sb.run("ws", loop, "/root", merge=True)
                self.assertIn("no space left on device", res.stdout)
                self.assertNotIn("done", res.stdout)
                self.assertLess(len(list((sim.hosts_dir / "bk/srv").iterdir())), 40)
                self.assertEqual(sb.run("bk", "rm /srv/w.*; echo ok", "/root").stdout, "ok\n")

    def test_host_over_its_simulated_disk(self) -> None:
        with tempfile.TemporaryDirectory(dir=ON_DISK) as tmp:
            sim = self.world(tmp)
            sim.disks["ws"] = SYSTEM_KIB + 50 * 1024  # 50 MB free in df
            with Sandbox(sim, Path(tmp)) as sb:
                res = sb.run("ws", "truncate -s 200M /srv/sparse", "/root", merge=True)
                self.assertIn("no space left on device: / on ws is full", res.stdout)
                self.assertIn(
                    "File too large", sb.run("ws", "echo x > /srv/new", "/root", merge=True).stdout
                )
                self.assertEqual(
                    sb.run("bk", "echo x > /srv/new && echo ok", "/root").stdout, "ok\n"
                )
                sb.run(
                    "ws", "rm /srv/sparse", "/root"
                )  # writes work again once the command that freed space ended
                self.assertEqual(
                    sb.run("ws", "echo x > /srv/new && echo ok", "/root").stdout, "ok\n"
                )


class ServerTest(unittest.TestCase):
    def test_client_that_hung_up_is_not_a_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sim = Simulator(Path(tmp) / "world")
            sim.add_host("ws", "10.0.0.2")
            sim.boot()
            answered = threading.Event()
            real = sim.handle

            def slow(*args: object, **kwargs: object) -> object:
                time.sleep(0.3)  # the stub is gone by now
                answered.set()
                return real(*args, **kwargs)  # type: ignore[arg-type]

            sim.handle = slow  # type: ignore[method-assign]
            err = io.StringIO()
            with Sandbox(sim, Path(tmp)) as sb, contextlib.redirect_stderr(err):
                client = socket.socket(socket.AF_UNIX)
                client.connect(str(sb.sockdir / "ws.sock"))
                client.sendall(json.dumps({"argv": ["ps", "aux"]}).encode() + b"\n")
                client.close()
                self.assertTrue(answered.wait(5))
                time.sleep(0.3)
        self.assertNotIn("Traceback", err.getvalue())


if __name__ == "__main__":
    unittest.main()
