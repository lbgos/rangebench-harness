"""Profile validation and the run-unit attempt limit."""

import argparse
import contextlib
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from rangebench.agent import ChatClient
from rangebench.cli import cmd_run
from rangebench.env import Stage, Task
from rangebench.launch import cmd_launch, load_config
from rangebench.progress import Progress
from rangebench.runner import AttemptResult


class LaunchProgressTests(unittest.TestCase):
    def test_provider_transport_failure_does_not_hide_retries(self):
        client = ChatClient("https://example.test/v1", "placeholder", "fake")
        with patch(
            "rangebench.agent.urllib.request.urlopen", side_effect=urllib.error.URLError("offline")
        ) as request:
            _, usage, error = client.chat([{"role": "user", "content": "hello"}], 10)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(usage.requests, 1)
        self.assertTrue(error.startswith("transport:"))

    def test_launch_rejects_unset_key_without_probing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fake.toml"
            path.write_text(
                'display_name="Fake"\nbase_url="https://example.test/v1"\n'
                'api_key_env="MISSING_TEST_KEY"\nmodel="one"\nctx_window=8000\ntrials=2\n'
            )
            args = argparse.Namespace(
                configs=["fake"],
                list=False,
                parallel=1,
                max_attempts=None,
                infra_retries=None,
                status_dir=None,
                dry_run=True,
            )
            with (
                patch("rangebench.launch.CONFIGS", Path(tmp)),
                patch.dict("os.environ", {}, clear=True),
                patch("rangebench.launch._probe") as probe,
                self.assertRaisesRegex(SystemExit, "MISSING_TEST_KEY is unset"),
            ):
                cmd_launch(args)
            probe.assert_not_called()

    def test_launch_probes_then_builds_commands_without_keys(self):
        with tempfile.TemporaryDirectory() as tmp, io.StringIO() as stdout:
            path = Path(tmp) / "fake.toml"
            path.write_text(
                'display_name="Fake"\nbase_url="https://example.test/v1"\n'
                'api_key_env="FAKE_KEY"\nmodel="one"\nctx_window=8000\ntrials=2\n'
                'wall_clock_reference="reference.json"\n'
            )
            args = argparse.Namespace(
                configs=["fake"],
                list=False,
                parallel=1,
                max_attempts=2,
                infra_retries=0,
                status_dir=Path(tmp) / "status",
                dry_run=True,
            )
            with (
                patch("rangebench.launch.CONFIGS", Path(tmp)),
                patch.dict("os.environ", {"FAKE_KEY": "private-placeholder"}),
                patch("rangebench.launch._probe") as probe,
                patch("rangebench.launch.subprocess.Popen") as popen,
                contextlib.redirect_stdout(stdout),
            ):
                cmd_launch(args)
            probe.assert_called_once()
            popen.assert_not_called()
            command = stdout.getvalue()
            self.assertIn("--max-attempts 2", command)
            self.assertIn("--infra-retries 0", command)
            self.assertIn("--status-json", command)
            self.assertIn("--wall-clock-reference reference.json", command)
            self.assertIn("FAKE_KEY", command)
            self.assertNotIn("private-placeholder", command)

    def test_live_log_keeps_only_safe_event_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            progress = Progress(root, root / "status.json", ["sample"], 1, None)
            progress.start("sample", 1, 0)
            progress.event("sample", 1, "submit", {})
            progress.event("sample", 1, "stage-completed", {"stage": "one"})
            progress.event("sample", 1, "incorrect-submission", {"wrong": 1})
            progress.event("sample", 1, "compaction", {})
            progress.finish("completed")
            log = (root / "live.log").read_text()
            self.assertIn("result submitted", log)
            self.assertIn("stage completed: one", log)
            self.assertIn("incorrect submission (1)", log)
            self.assertIn("compaction fired", log)
            self.assertEqual(json.loads((root / "status.json").read_text())["state"], "completed")

    def test_config_validates_required_fields_and_key_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.toml"
            path.write_text(
                'display_name="Model"\nbase_url="https://example.test/v1"\n'
                'api_key_env="MODEL_KEY"\nmodel="one"\nctx_window=8000\ntrials=2\n'
            )
            self.assertEqual(load_config(path).api_key_env, "MODEL_KEY")
            path.write_text(path.read_text().replace("MODEL_KEY", "key=value"))
            with self.assertRaisesRegex(ValueError, "api_key_env"):
                load_config(path)
            path.write_text(
                'display_name="Model"\nmodel="one"\nctx_window=8000\ntrials=2\n'
                'api_key_env="MODEL_KEY"\n'
            )
            with self.assertRaisesRegex(ValueError, "base_url"):
                load_config(path)

    def test_cap_includes_unscored_retries_and_status_is_final(self):
        task = Task(
            "sample", Path("/tmp"), "web", 1, "Find the flag", stages=[Stage("one", "/flag", "web")]
        )
        args = argparse.Namespace(
            trials=3,
            tasks=[task.id],
            model="fake",
            base_url="http://localhost:8000/v1",
            provider="openai",
            ctx_window=128000,
            reserve=12000,
            keep_tail=12,
            threshold=0.82,
            compact="deterministic",
            keep=False,
            max_attempts=2,
            infra_retries=5,
        )
        with tempfile.TemporaryDirectory() as tmp, io.StringIO() as stdout:
            root = Path(tmp)
            args.status_json = root / "status.json"
            with (
                patch("rangebench.cli.RESULTS", root),
                patch("rangebench.cli.load_task", return_value=task),
                patch("rangebench.cli.ChatClient"),
                patch("rangebench.cli._get_attacker_digest", return_value="sha256:test"),
                patch(
                    "rangebench.cli.run_attempt",
                    return_value=AttemptResult(task.id, 1, end_reason="llm error"),
                ) as attempt,
                contextlib.redirect_stdout(stdout),
                self.assertRaises(SystemExit),
            ):
                cmd_run(args)
            status = json.loads(args.status_json.read_text())
            result = json.loads((root / "latest.json").read_text())
            live = next(root.glob("*/live.log")).read_text()
            self.assertEqual(attempt.call_count, 2)
            self.assertEqual(status["attempts_used"], 2)
            self.assertEqual(status["attempts_remaining"], 0)
            self.assertEqual(status["infra_retries"], 1)
            self.assertEqual(status["state"], "attempt_limit")
            self.assertEqual(len(result["infra_attempts"]), 2)
            self.assertEqual(len(result["tasks"]), 1)
            self.assertIn("attempts 1/2 · sample t1 · started", live)
            self.assertIn("attempts 2/2 · sample t1 · started · infra retry 1", live)
            self.assertIn("run attempt_limit", live)
            self.assertIn("prompt/completion 0/0 tok", live)
            self.assertIn("attempts 1/2", stdout.getvalue())

    def test_refusal_is_unscored_and_skips_repeat_without_retry(self):
        task = Task("sample", Path("/tmp"), "web", 1, "Example", stages=[Stage("one", "/flag", "web")])
        args = argparse.Namespace(
            trials=2, tasks=[task.id], model="fake", base_url="http://localhost:8000/v1",
            provider="openai", ctx_window=128000, reserve=12000, keep_tail=12,
            threshold=0.82, compact="deterministic", keep=False, infra_retries=3,
            repeat_caps=True,
        )
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            args.status_json = root / "status.json"
            with (
                patch("rangebench.cli.RESULTS", root),
                patch("rangebench.cli.load_task", return_value=task),
                patch("rangebench.cli.load_all", return_value=[task]),
                patch("rangebench.cli.ChatClient"),
                patch("rangebench.cli._get_attacker_digest", return_value="sha256:" + "a" * 64),
                patch("rangebench.cli.run_attempt", return_value=AttemptResult(
                    task.id, 1, end_reason="model refusal", refusals=1,
                )) as attempt,
            ):
                with self.assertRaises(SystemExit):
                    cmd_run(args)
            self.assertEqual(attempt.call_count, 1)
            run = json.loads((root / "latest.json").read_text())
            self.assertEqual([t["fail_class"] for t in run["tasks"]], ["refusal", "skipped"])
            self.assertTrue(all(not t["scored"] for t in run["tasks"]))
            self.assertEqual(run["infra_attempts"], [])
            submission = json.loads(next(root.glob("*/submission.json")).read_text())
            self.assertEqual(submission["summary"]["scored"], 0)
            self.assertIsNone(submission["summary"]["pass_at_1"])

    def test_cap_stops_before_next_scored_trial(self):
        task = Task(
            "sample", Path("/tmp"), "web", 1, "Find the flag", stages=[Stage("one", "/flag", "web")]
        )
        args = argparse.Namespace(
            trials=3,
            tasks=[task.id],
            model="fake",
            base_url="http://localhost:8000/v1",
            provider="openai",
            ctx_window=128000,
            reserve=12000,
            keep_tail=12,
            threshold=0.82,
            compact="deterministic",
            keep=False,
            max_attempts=1,
            infra_retries=1,
        )
        with tempfile.TemporaryDirectory() as tmp, io.StringIO() as stdout:
            root = Path(tmp)
            args.status_json = root / "status.json"
            with (
                patch("rangebench.cli.RESULTS", root),
                patch("rangebench.cli.load_task", return_value=task),
                patch("rangebench.cli.ChatClient"),
                patch("rangebench.cli._get_attacker_digest", return_value="sha256:test"),
                patch(
                    "rangebench.cli.run_attempt",
                    return_value=AttemptResult(task.id, 1, end_reason="turn budget"),
                ) as attempt,
                contextlib.redirect_stdout(stdout),
            ):
                cmd_run(args)
            self.assertEqual(attempt.call_count, 1)
            status = json.loads(args.status_json.read_text())
            self.assertEqual(status["model_attempts_used"], 1)
            self.assertEqual(status["tasks"]["sample:t2"]["state"], "pending")
            self.assertEqual(status["state"], "attempt_limit")
            self.assertEqual(len(json.loads((root / "latest.json").read_text())["tasks"]), 1)

    def test_config_rejects_remote_http_but_allows_loopback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.toml"
            base = (
                'display_name="Model"\napi_key_env="MODEL_KEY"\nmodel="one"\n'
                "ctx_window=8000\ntrials=2\n"
            )
            for url in (
                "https://example.test/v1",
                "http://localhost:8000/v1",
                "http://127.0.0.1:8000/v1",
                "http://[::1]:8000/v1",
            ):
                path.write_text(base + f'base_url="{url}"\n')
                self.assertEqual(load_config(path).base_url, url)
            path.write_text(base + 'base_url="http://example.test/v1"\n')
            with self.assertRaisesRegex(ValueError, "https"):
                load_config(path)

    def test_finish_is_idempotent_and_keeps_first_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            progress = Progress(root, root / "status.json", ["sample"], 1, None)
            progress.finish("completed")
            progress.finish("probe_failed")
            self.assertEqual(progress.state, "completed")
            self.assertEqual(json.loads((root / "status.json").read_text())["state"], "completed")

    def test_launch_waits_for_any_free_slot(self):
        class FakeProc:
            def __init__(self, events: list[str], tag: str, polls_before_done: int):
                self.events = events
                self.tag = tag
                self.polls = polls_before_done

            def poll(self) -> int | None:
                if self.polls > 0:
                    self.polls -= 1
                    return None
                return 0

            def wait(self) -> int:
                self.events.append(f"wait-{self.tag}")
                return 0

        with tempfile.TemporaryDirectory() as tmp:
            for name in ("one", "two", "three"):
                (Path(tmp) / f"{name}.toml").write_text(
                    'display_name="Fake"\nbase_url="https://example.test/v1"\n'
                    'api_key_env="FAKE_KEY"\nmodel="one"\nctx_window=8000\ntrials=1\n'
                )
            args = argparse.Namespace(
                configs=["one", "two", "three"],
                list=False,
                parallel=2,
                max_attempts=None,
                infra_retries=None,
                status_dir=None,
                dry_run=False,
            )
            events: list[str] = []
            procs = [FakeProc(events, str(i), polls) for i, polls in enumerate((1, 0, 0))]

            def fake_popen(cmd: list[str]) -> FakeProc:
                proc = procs[len([e for e in events if e.startswith("start-")])]
                events.append(f"start-{proc.tag}")
                return proc

            with (
                patch("rangebench.launch.CONFIGS", Path(tmp)),
                patch.dict("os.environ", {"FAKE_KEY": "placeholder"}),
                patch("rangebench.launch._probe"),
                patch("rangebench.launch.subprocess.Popen", side_effect=fake_popen),
                patch("rangebench.launch.time.sleep"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                cmd_launch(args)
        # The fast second run frees its slot while the first is still going,
        # so the third command starts before the first process is reaped.
        self.assertEqual(events, ["start-0", "start-1", "wait-1", "start-2", "wait-0", "wait-2"])


if __name__ == "__main__":
    unittest.main()
