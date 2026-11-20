import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rangebench_homelab import cli


class SweepTest(unittest.TestCase):
    def test_removes_worlds_of_dead_runs_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs, work = Path(tmp) / "runs", Path(tmp) / ".work"

            def run(name: str, keep: bool = False) -> Path:
                d = runs / name
                (d / "attempts/x-s0/world/hosts").mkdir(parents=True)
                (d / "attempts/x-s0/bin").mkdir()
                (d / "run.jsonl").write_text(json.dumps({"type": "run", "keep_world": keep}) + "\n")
                (d / ".lock").touch()
                return d

            dead, kept, legacy = run("dead"), run("kept", keep=True), run("legacy")
            (legacy / ".lock").unlink()
            live = run("live")
            gate_dead = work / "rbgate-a"
            gate_dead.mkdir(parents=True)
            (gate_dead / ".lock").touch()
            with (
                mock.patch.object(cli, "RUNS_DIR", runs),
                mock.patch.object(cli, "WORK_ROOT", work),
            ):
                held = cli.hold(live)
                cli.sweep()
                held.close()
            self.assertFalse((dead / "attempts/x-s0/world").exists())
            self.assertFalse((dead / "attempts/x-s0/bin").exists())
            self.assertTrue((dead / "run.jsonl").exists())
            for d in (kept, legacy, live):
                self.assertTrue((d / "attempts/x-s0/world").exists(), d.name)
            self.assertFalse(gate_dead.exists())


if __name__ == "__main__":
    unittest.main()
