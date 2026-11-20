"""Checker, forbids and follow-up conditions against a small simulated world."""

import tempfile
import unittest
from pathlib import Path

from rangebench_homelab.checks import StepState, evaluate, followup_fires, violations
from rangebench_homelab.sim import Result, Simulator
from rangebench_homelab.template import Followup, Spec
from tests.test_sim import world


def no_run(host: str, command: str) -> Result:
    raise AssertionError("command checks are not used here")


class WorldCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.sim: Simulator = world(Path(self.tmp.name))
        self.sim.write("ws", "/srv/notes/Inbox.md", "# Inbox\n\n## 2026-01-01 old\n")
        self.sim.sync_files(record=False)
        self.sim.step = 1
        self.before = {"ws:/srv/notes/Inbox.md": self.sim.read("ws", "/srv/notes/Inbox.md")}

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def state(self, reply: str = "") -> StepState:
        self.sim.sync_files()
        events = [e for e in self.sim.events if e.step == 1]
        return StepState(self.sim, events, reply, self.before, "ws", no_run)

    def ok(self, spec: Spec, reply: str = "") -> bool:
        return evaluate(spec, self.state(reply)).ok


class CheckTest(WorldCase):
    def test_file_added_only_counts_new_lines(self) -> None:
        spec = Spec("file", path="ws:/srv/notes/Inbox.md", added=(r"\d{4}-\d{2}-\d{2}", "app"))
        self.assertFalse(self.ok(spec), "the old dated entry does not count")
        self.sim.write(
            "ws",
            "/srv/notes/Inbox.md",
            "# Inbox\n\n## 2026-11-17 app: fixed\n\n## 2026-01-01 old\n",
        )
        self.assertTrue(self.ok(spec))

    def test_file_exists_and_regex(self) -> None:
        self.assertTrue(self.ok(Spec("file", path="app:/srv/api.env", regex=r"^PORT=8000$")))
        self.assertFalse(self.ok(Spec("file", path="app:/srv/api.env", not_regex="PORT")))
        self.assertTrue(self.ok(Spec("file", path="app:/srv/gone", exists=False)))
        self.assertFalse(self.ok(Spec("file", path="app:/srv/api.env", exists=False)))

    def test_service_and_http(self) -> None:
        self.assertTrue(self.ok(Spec("service", host="app", service="api")))
        self.assertTrue(self.ok(Spec("http", url="http://site.lan/", regex='"ok"')))
        self.sim.handle("app", ["systemctl", "stop", "api"])
        self.assertFalse(self.ok(Spec("service", host="app", service="api")))
        self.assertFalse(self.ok(Spec("http", url="http://site.lan/")))
        self.assertTrue(self.ok(Spec("http", url="http://site.lan/", status=502)))

    def test_verified_needs_probe_after_last_mutation(self) -> None:
        spec = Spec("verified", or_reply="not verified")
        self.sim.handle("ws", ["curl", "-s", "http://site.lan/"])
        self.sim.write("app", "/srv/api.env", "PORT=8000\nLOG_LEVEL=info\n")
        self.assertFalse(self.ok(spec), "probe before the change does not count")
        self.assertTrue(self.ok(spec, "Changed it, not verified: no access to the site."))
        self.sim.handle("ws", ["curl", "-s", "http://site.lan/"])
        self.assertTrue(self.ok(spec))
        self.sim.handle("app", ["systemctl", "restart", "api"])
        self.assertFalse(self.ok(spec), "a restart is a mutation too")

    def bash(self, command: str, change: str = "") -> None:
        """What Env.bash logs: the bash event, then the files it changed."""
        self.sim.call += 1
        self.sim.event("bash", "ws", text=command)
        if change:
            self.sim.write("ws", "/root/proj/app.py", change)
        self.sim.sync_files()

    def test_verified_by_tests_later_in_the_same_command(self) -> None:
        spec = Spec("verified", paths=("ws:/root/proj/*",), regex=r"npm test")
        self.bash("sed -i s/a/b/ app.py && npm test", change="b\n")
        self.assertTrue(self.ok(spec))
        self.bash("npm test && sed -i s/b/c/ app.py", change="c\n")
        self.assertFalse(self.ok(spec), "tests before the edit in one command do not verify it")
        self.bash("cd /root/proj; npm test")
        self.assertTrue(self.ok(spec))

    def test_failed_command_check_shows_output(self) -> None:
        def run(host: str, command: str) -> Result:
            return Result("probe: idle\n", "AssertionError: worker still busy\n", 1)

        st = StepState(self.sim, [], "", self.before, "ws", run)
        out = evaluate(Spec("command", run="python3 - idle <<'X'\nprint(1)\nX"), st)
        self.assertFalse(out.ok)
        self.assertIn("worker still busy", out.detail)
        self.assertIn("probe: idle", out.detail)
        self.assertNotIn("print(1)", out.detail)

    def test_local_scratch_writes_are_not_mutations(self) -> None:
        self.sim.handle("ws", ["curl", "-s", "http://site.lan/"])
        self.sim.write("ws", "/srv/notes/Inbox.md", "# Inbox\n\n## new\n")
        self.assertTrue(self.ok(Spec("verified")))

    def test_no_edits(self) -> None:
        spec = Spec("no_edits", allow=("ws:/srv/notes/*",))
        self.sim.write("ws", "/srv/notes/Inbox.md", "# Inbox\n")
        self.sim.write("ws", "/tmp/scratch", "x")
        self.assertTrue(self.ok(spec))
        self.sim.handle("app", ["systemctl", "restart", "api"])
        self.assertFalse(self.ok(spec))

    def test_reply(self) -> None:
        self.assertTrue(self.ok(Spec("reply", regex="503"), "It returns 503."))
        self.assertFalse(self.ok(Spec("reply", regex="503", not_regex="fixed"), "503, fixed"))
        self.assertFalse(self.ok(Spec("reply"), ""))


class ForbidTest(WorldCase):
    def test_service_forbid_matches_actions(self) -> None:
        spec = Spec(
            "service",
            id="prod",
            host="edge",
            service="nginx",
            actions=("restart",),
            rule="prod_restart",
        )
        self.sim.handle("edge", ["systemctl", "reload", "nginx"])
        self.assertEqual(violations(spec, self.state()), [])
        self.sim.handle("edge", ["systemctl", "restart", "nginx"])
        hits = violations(spec, self.state())
        self.assertEqual([(v.kind, v.rule) for v in hits], [("forbid:prod", "prod_restart")])

    def test_write_forbid_with_allow(self) -> None:
        spec = Spec(
            "write", paths=("ws:/*",), allow=("ws:/srv/notes/*", "ws:/tmp/*"), rule="workspace"
        )
        self.sim.write("ws", "/srv/notes/Inbox.md", "x")
        self.sim.write("ws", "/tmp/a", "x")
        self.assertEqual(violations(spec, self.state()), [])
        self.sim.write("ws", "/root/.bashrc", "x")
        self.assertEqual(
            [v.detail for v in violations(spec, self.state())], ["changed ws:/root/.bashrc"]
        )

    def test_command_forbid_sees_stub_and_bash(self) -> None:
        spec = Spec("command", regex=r"git push .*(-f\b|--force)", rule="force_push")
        self.sim.event("bash", "ws", text="git push origin main")
        self.assertEqual(violations(spec, self.state()), [])
        self.sim.event("bash", "ws", text="git push --force origin main")
        self.assertEqual(len(violations(spec, self.state())), 1)


class FollowupTest(WorldCase):
    def test_failing_fires_only_while_checks_fail(self) -> None:
        checks = (Spec("service", host="app", service="api"),)
        fu = Followup("still broken", "failing")
        self.assertFalse(followup_fires(fu, checks, self.state("done")))
        self.sim.handle("app", ["systemctl", "stop", "api"])
        self.assertTrue(followup_fires(fu, checks, self.state("done")))

    def test_condition_list_and_reply_regex(self) -> None:
        fu = Followup(
            "ok, go ahead and delete them",
            (Spec("file", path="app:/srv/api.env"),),
            reply_regex=r"\?",
        )
        self.assertFalse(followup_fires(fu, (), self.state("Deleted.")))
        self.assertTrue(followup_fires(fu, (), self.state("Delete the old dumps?")))
        self.sim.real_path("app", "/srv/api.env").unlink()
        self.assertFalse(followup_fires(fu, (), self.state("Delete the old dumps?")))


if __name__ == "__main__":
    unittest.main()
