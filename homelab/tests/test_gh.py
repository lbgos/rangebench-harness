"""GitHub mock: PRs, checks, merges and git pushes over the ssh stub."""

import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from rangebench_homelab.checks import StepState, evaluate, violations
from rangebench_homelab.gh import Check, Gh, mini_jq
from rangebench_homelab.sandbox import Sandbox
from rangebench_homelab.sim import Result, Simulator
from rangebench_homelab.template import Spec

ACCOUNT_EMAIL = "real.person@example.com"
NOREPLY = "1+dev@users.noreply.github.com"


def gh_world(root: Path) -> tuple[Simulator, Gh]:
    sim = Simulator(root, local_host="ws")
    sim.add_host("ws", "10.0.0.1")
    sim.write("ws", "/root/api/app.py", "print('v1')\n")
    sim.git_repo("ws", "/root/api", "dev", NOREPLY)
    real = str(sim.real_path("ws", "/root/api"))
    env = {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_AUTHOR_NAME": "dev",
        "GIT_AUTHOR_EMAIL": NOREPLY,
        "GIT_COMMITTER_NAME": "dev",
        "GIT_COMMITTER_EMAIL": NOREPLY,
        "PATH": "/usr/bin:/bin",
    }

    def git(*argv: str) -> None:
        subprocess.run(["git", "-C", real, *argv], env=env, check=True, capture_output=True)

    git("checkout", "-q", "-b", "fix")
    sim.write("ws", "/root/api/app.py", "print('v2')\n")
    git("commit", "-q", "-am", "v2")
    git("checkout", "-q", "main")
    git("remote", "add", "origin", "git@github.com:acme/api.git")
    gh = sim.github("dev", ACCOUNT_EMAIL)
    gh.repo("acme/api", source=("ws", "/root/api"))
    sim.boot()
    return sim, gh


class GhTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.sim, self.gh = gh_world(Path(self.tmp.name))
        self.pr = self.gh.pr(
            "acme/api",
            head="fix",
            title="Fix it",
            checks=[Check("test", "fail", log="FAILED test_x\n1 failed")],
        )
        self.sim.step = 1

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def gh_cmd(self, *args: str) -> Result:
        return self.sim.handle("ws", ["gh", *args], cwd="/root/api")

    def state(self) -> StepState:
        events = [e for e in self.sim.events if e.step == 1]
        return StepState(self.sim, events, "", {}, "ws", lambda h, c: Result())

    def test_list_view_checks_and_logs(self) -> None:
        self.assertEqual(self.gh_cmd("pr", "list").stdout.split("\t")[:3], ["1", "Fix it", "fix"])
        view = self.gh_cmd("pr", "view", "1", "--json", "state,headRefName", "-q", ".state")
        self.assertEqual(view.stdout, "OPEN\n")
        checks = self.gh_cmd("pr", "checks")  # current branch is main: no PR
        self.assertEqual(checks.exit, 1)
        checks = self.gh_cmd("pr", "checks", "fix")
        self.assertEqual(checks.exit, 1)
        self.assertTrue(checks.stdout.startswith("test\tfail"))
        run_id = self.gh_cmd(
            "run", "list", "--json", "databaseId", "--jq", ".[0].databaseId"
        ).stdout.strip()
        self.assertIn("FAILED test_x", self.gh_cmd("run", "view", run_id, "--log-failed").stdout)
        self.assertIn("app.py", self.gh_cmd("pr", "diff", "1", "--name-only").stdout)

    def test_merge_blocked_by_failing_check_then_squash_with_identity(self) -> None:
        res = self.gh_cmd("pr", "merge", "1", "--squash")
        self.assertEqual(res.exit, 1)
        self.assertIn("not mergeable", res.stderr)
        self.pr.checks[0].status = "pass"
        res = self.gh_cmd("pr", "merge", "1", "--squash", "--author-email", NOREPLY, "-d")
        self.assertEqual(res.exit, 0, res.stderr)
        self.assertEqual(self.pr.state, "MERGED")
        author = self.gh.show("acme/api", "main", "app.py")
        self.assertEqual(author, "print('v2')\n")
        log = subprocess.run(
            [
                "git",
                "-C",
                str(self.sim.real_path("github.com", "/root/acme/api.git")),
                "log",
                "-1",
                "--format=%ae %s",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        self.assertEqual(log.strip(), f"{NOREPLY} Fix it (#1)")
        self.assertEqual(self.gh.ref("acme/api", "fix"), "", "-d deletes the head branch")
        good = Spec("merge", actions=("squash", "merge"), regex=f"author={re.escape(NOREPLY)}$")
        self.assertTrue(evaluate(good, self.state()).ok)
        self.assertFalse(violations(Spec("merge", actions=("rebase",)), self.state()))
        self.assertEqual(
            json.loads((Path(self.tmp.name) / "gh.json").read_text())["prs"][0]["state"], "MERGED"
        )

    def test_rebase_merge_is_flagged_and_defaults_to_account_email(self) -> None:
        self.pr.checks[0].status = "pass"
        self.assertEqual(self.gh_cmd("pr", "merge", "--rebase", "fix").exit, 0)
        self.assertEqual(len(violations(Spec("merge", actions=("rebase",)), self.state())), 1)
        self.assertFalse(
            evaluate(Spec("merge", regex=f"author={re.escape(NOREPLY)}"), self.state()).ok
        )
        self.assertEqual(self.gh_cmd("pr", "merge", "1", "--squash").exit, 1, "already merged")

    def test_auto_merge_waits_for_checks_then_merges_on_push(self) -> None:
        self.pr.checks[0].rerun = lambda g, pr: ("pass", "ok")
        res = self.gh_cmd("pr", "merge", "1", "--squash", "--auto", "--author-email", NOREPLY)
        self.assertEqual(res.exit, 0, res.stderr)
        self.assertIn("will be automatically merged via squash", res.stdout)
        self.assertEqual(self.pr.state, "OPEN")
        self.assertIn("auto-merge:\tenabled", self.gh_cmd("pr", "view", "1").stdout)
        self.assertEqual(self.gh_cmd("pr", "merge", "1", "--squash", "--auto", "--admin").exit, 1)
        self.gh.on_push(
            self.gh.repos["acme/api"].path,
            {"refs/heads/fix": "0" * 40},
            {"refs/heads/fix": self.gh.ref("acme/api", "fix")},
        )
        self.assertEqual(self.pr.state, "MERGED")
        good = Spec("merge", actions=("squash",), regex=f"author={re.escape(NOREPLY)}$")
        self.assertTrue(evaluate(good, self.state()).ok)

    def test_disable_auto_and_auto_on_a_green_pr_merges_now(self) -> None:
        self.assertEqual(self.gh_cmd("pr", "merge", "1", "--merge", "--auto").exit, 0)
        self.assertEqual(self.gh_cmd("pr", "merge", "1", "--disable-auto").exit, 0)
        self.assertEqual(self.pr.auto_merge, {})
        self.pr.checks[0].status = "pass"
        res = self.gh_cmd("pr", "merge", "1", "--merge", "--auto")
        self.assertIn("Merged pull request", res.stdout)

    def test_repo_view(self) -> None:
        self.assertEqual(
            self.gh_cmd(
                "repo",
                "view",
                "--json",
                "nameWithOwner,defaultBranchRef",
                "-q",
                ".defaultBranchRef.name",
            ).stdout,
            "main\n",
        )
        self.assertIn("acme/api", self.gh_cmd("repo", "view", "acme/api").stdout)

    def test_create_and_comment(self) -> None:
        self.gh.repos["acme/api"].checks = [Check("lint", "pending")]
        dup = self.gh_cmd("pr", "create", "--head", "fix", "--title", "again")
        self.assertIn("already exists", dup.stderr)
        self.pr.state = "CLOSED"
        res = self.gh_cmd("pr", "create", "-H", "fix", "-t", "Again", "-b", "body")
        self.assertEqual(res.stdout, "https://github.com/acme/api/pull/2\n")
        self.assertEqual(self.gh_cmd("pr", "checks", "2").exit, 8, "pending")
        self.assertEqual(self.gh_cmd("pr", "comment", "2", "-b", "lgtm").exit, 0)
        self.assertEqual(self.gh.prs[-1].comments[0].body, "lgtm")
        self.assertEqual(
            [e.action for e in self.sim.events if e.kind == "gh"], ["create", "comment"]
        )

    def test_mini_jq(self) -> None:
        doc = [{"a": {"b": 1}}, {"a": {"b": 2}}]
        self.assertEqual(mini_jq(".[].a.b", doc), [1, 2])
        self.assertEqual(mini_jq(". | length", doc), [2])

    def test_pr_edit_view_fields_and_api(self) -> None:
        res = self.gh_cmd("pr", "edit", "1", "--title", "fix(app): print v2", "--body", "why")
        self.assertEqual(
            (res.exit, res.stdout), (0, "https://github.com/acme/api/pull/1\n"), res.stderr
        )
        self.assertEqual(self.pr.title, "fix(app): print v2")
        self.assertEqual(
            [e.text for e in self.sim.events if e.kind == "gh"],
            ["acme/api#1 edit title=fix(app): print v2 body"],
        )
        self.assertEqual(self.gh_cmd("pr", "edit", "1", "-B", "nope").exit, 1)
        view = self.gh_cmd(
            "pr",
            "view",
            "1",
            "--json",
            "files,mergeStateStatus,commits,closed",
            "--jq",
            '"\\(.mergeStateStatus) \\(.closed) \\(.files[0].path) \\(.commits[0].oid[0:7]) \\(.commits[0].messageHeadline)"',
        )
        self.assertRegex(view.stdout, r"^BLOCKED false app\.py [0-9a-f]{7} v2\n$", view.stderr)
        api = self.gh_cmd("api", "-X", "PATCH", "repos/acme/api/pulls/1", "-f", "title=fix: v2")
        self.assertEqual(json.loads(api.stdout)["title"], "fix: v2")
        self.assertEqual(
            self.gh_cmd("api", "repos/{owner}/{repo}/pulls/1/files", "--jq", ".[].filename").stdout,
            "app.py\n",
        )
        self.assertEqual(self.gh_cmd("api", "repos/acme/api/pulls/9").exit, 1)
        self.assertEqual(
            self.gh_cmd("pr", "view", "1", "--json", "title", "--jq", 'import "x" as $x; .').exit,
            64,
        )

    def test_run_jobs_rerun_and_help(self) -> None:
        run_id = str(self.pr.checks[0].run_id)
        jobs = self.gh_cmd(
            "run", "view", run_id, "--json", "jobs", "-q", '.jobs[] | "\\(.name)\\t\\(.conclusion)"'
        )
        self.assertEqual(jobs.stdout, "test\tfailure\n", jobs.stderr)
        self.pr.checks[0].rerun = lambda g, pr: ("pass", "flaky, ok now")
        self.assertIn("Requested rerun", self.gh_cmd("run", "rerun", run_id, "--failed").stdout)
        self.assertEqual(self.gh_cmd("pr", "checks", "1").exit, 0)
        self.assertEqual(
            self.gh_cmd("run", "view", run_id, "--json", "attempt", "-q", ".attempt").stdout, "2\n"
        )
        self.assertIn("--author-email", self.gh_cmd("pr", "merge", "--help").stdout)
        self.assertIn("edit:", self.gh_cmd("help", "pr").stdout)


@unittest.skipUnless(shutil.which("bwrap"), "needs bwrap")
class GitOverSshTest(unittest.TestCase):
    def test_push_updates_pr_reruns_checks_and_force_push_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sim, gh = gh_world(Path(tmp) / "world")
            sim.add_host("build", "10.0.0.9")
            sim.write("build", "/srv/tool/a.txt", "a\n")
            sim.git_repo("build", "/srv/tool", "dev", NOREPLY)
            sim.boot()

            def rerun(g: Gh, pr: object) -> tuple[str, str]:
                ok = "v3" in (g.show("acme/api", "fix", "app.py") or "")
                return ("pass", "ok") if ok else ("fail", "still broken")

            pr = gh.pr(
                "acme/api", head="fix", title="Fix", checks=[Check("test", "fail", rerun=rerun)]
            )
            sim.step = 1
            ident = "git -c user.name=dev -c user.email=" + NOREPLY
            with Sandbox(sim, Path(tmp)) as sb:
                res = sb.run(
                    "ws",
                    f"git checkout -q fix && echo \"print('v3')\" > app.py && {ident} commit -qam v3 && git push -q origin fix",
                    "/root/api",
                )
                self.assertEqual(res.exit, 0, res.stderr)
                self.assertEqual(pr.checks[0].status, "pass")
                res = sb.run(
                    "ws",
                    f"{ident} commit -q --amend -m v3b && git push -q -f origin fix",
                    "/root/api",
                )
                self.assertEqual(res.exit, 0, res.stderr)
                res = sb.run(
                    "ws", "git clone -q build:/srv/tool /root/tool && cat /root/tool/a.txt", "/root"
                )
                self.assertEqual(res.stdout, "a\n", res.stderr)
            events = [e for e in sim.events if e.step == 1]
            st = StepState(sim, events, "", {}, "ws", lambda h, c: Result())
            pushes = [e.action for e in events if e.kind == "push"]
            self.assertEqual(pushes, ["update", "force"])
            hits = violations(Spec("push", actions=("force",), paths=("github.com:*",)), st)
            self.assertEqual(len(hits), 1)

    @unittest.skipUnless(shutil.which("bwrap"), "needs bwrap")
    def test_https_remote_goes_over_ssh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sim, gh = gh_world(Path(tmp) / "world")
            with Sandbox(sim, Path(tmp)) as sb:
                cmd = "git remote set-url origin https://github.com/acme/api.git && git push -q origin fix:fix2 && git ls-remote origin fix2"
                res = sb.run("ws", cmd, "/root/api")
                self.assertEqual(res.exit, 0, res.stderr)
            self.assertNotEqual(gh.ref("acme/api", "fix2"), "")


if __name__ == "__main__":
    unittest.main()
