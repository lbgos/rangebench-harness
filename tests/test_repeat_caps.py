"""Run-level handling of repeated no-progress wall caps."""

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rangebench.cli import cmd_run
from rangebench.env import Stage, Task
from rangebench.runner import FAIL_SKIPPED, AttemptResult


class RepeatCapsTests(unittest.TestCase):
    def run_trials(
        self, first: AttemptResult, repeat_caps: bool = False
    ) -> tuple[list[dict], str, int, dict]:
        task = Task(
            "sample",
            Path(tempfile.gettempdir()),
            "web",
            1,
            "Find the flag",
            stages=[Stage("one", "/flag", "target"), Stage("two", "/flag2", "target")],
        )
        args = argparse.Namespace(
            trials=3,
            repeat_caps=repeat_caps,
            ctx_window=128000,
            reserve=12000,
            keep_tail=12,
            threshold=0.82,
            tasks=[task.id],
            base_url="http://localhost:8000/v1",
            provider="openai",
            model="test",
            compact="deterministic",
            keep=False,
        )

        def attempt(
            _client: object, _task: Task, trial: int, *_args: object, **_kwargs: object
        ) -> AttemptResult:
            if trial == 1:
                return first
            return AttemptResult(task.id, trial, end_reason="turn budget")

        with tempfile.TemporaryDirectory() as tmp, io.StringIO() as stdout:
            with (
                patch("rangebench.cli.RESULTS", Path(tmp)),
                patch("rangebench.cli.load_task", return_value=task),
                patch("rangebench.cli.ChatClient"),
                patch("rangebench.cli.run_attempt", side_effect=attempt) as mock_attempt,
                patch("rangebench.cli._get_attacker_digest", return_value="sha256:test"),
                contextlib.redirect_stdout(stdout),
            ):
                cmd_run(args)
            records = json.loads((Path(tmp) / "latest.json").read_text())["tasks"]
            manifest = json.loads(next(Path(tmp).glob("*/manifest.json")).read_text())
            return records, stdout.getvalue(), mock_attempt.call_count, manifest

    def test_skips_remaining_trials_after_no_progress_cap(self) -> None:
        first = AttemptResult(
            "sample", 1, end_reason="wall_clock_exceeded", wall_clock_exceeded=True
        )
        records, output, calls, manifest = self.run_trials(first)
        self.assertEqual(calls, 1)
        self.assertEqual([t["trial"] for t in records], [1, 2, 3])
        self.assertTrue(records[0]["scored"])
        self.assertEqual(manifest["task_count"], 1)
        self.assertIn("refusals: 0 turns in 0/1 attempts", output)
        self.assertIn("0/1 scored runs (0 invalid)", output)
        self.assertIn("sample               0/1", output)

    def test_stage_progress_keeps_running(self) -> None:
        first = AttemptResult(
            "sample", 1, solved=["one"], end_reason="wall_clock_exceeded", wall_clock_exceeded=True
        )
        records, _, calls, _ = self.run_trials(first)
        self.assertEqual(calls, 3)
        self.assertTrue(all(t["fail_class"] != FAIL_SKIPPED for t in records))

    def test_non_wall_cap_keeps_running(self) -> None:
        first = AttemptResult("sample", 1, end_reason="turn budget")
        records, _, calls, _ = self.run_trials(first)
        self.assertEqual(calls, 3)
        self.assertTrue(all(t["fail_class"] != FAIL_SKIPPED for t in records))

    def test_skipped_record_has_no_usage_or_score(self) -> None:
        first = AttemptResult(
            "sample", 1, end_reason="wall_clock_exceeded", wall_clock_exceeded=True
        )
        records, _, _, manifest = self.run_trials(first)
        for record in records[1:]:
            self.assertFalse(record["scored"])
            self.assertEqual(record["fail_class"], FAIL_SKIPPED)
            self.assertEqual(record["solved_stages"], [])
            self.assertFalse(record["wall_clock_exceeded"])
            self.assertEqual(record["end_reason"], "skipped: repeat after no-progress wall cap")
            self.assertEqual(record["turns_used"], 0)
            self.assertEqual(record["wall_s"], 0)
            for key, value in record.items():
                if key != "max_output_tokens" and ("tokens" in key or key in {
                    "api_calls",
                    "api_requests",
                    "usage_reported_calls",
                    "input_reported_calls",
                    "output_reported_calls",
                    "cache_read_reported_calls",
                    "cache_write_reported_calls",
                }):
                    self.assertEqual(value, 0, key)
        self.assertEqual([item["trial"] for item in manifest["service_image_ids"]], [1])

    def test_opt_out_repeats_wall_caps(self) -> None:
        first = AttemptResult(
            "sample", 1, end_reason="wall_clock_exceeded", wall_clock_exceeded=True
        )
        records, _, calls, _ = self.run_trials(first, repeat_caps=True)
        self.assertEqual(calls, 3)
        self.assertTrue(all(t["fail_class"] != FAIL_SKIPPED for t in records))


if __name__ == "__main__":
    unittest.main()
