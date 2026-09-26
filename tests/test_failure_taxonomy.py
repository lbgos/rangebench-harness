"""Failure taxonomy, refusal counting, and classification reporting."""

import argparse
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rangebench.agent import ChatResult, ResponseMetadata, Usage, is_refusal
from rangebench.cli import cmd_run
from rangebench.env import ATTACKER_IMAGE, Stage, Task
from rangebench.runner import (
    FAIL_BUDGET,
    FAIL_ENV,
    FAIL_NORMAL,
    FAIL_PROTOCOL,
    FAIL_PROVIDER,
    FAIL_SOLVED,
    AttemptResult,
    classify_end_reason,
    run_attempt,
)


class FakeEnv:
    def __init__(self, task: Task, project: str, attacker_image: str = ATTACKER_IMAGE):
        self.attacker = f"{project}-atk"
        self.attacker_image = attacker_image
        self.service_image_ids = {"target": "sha256:" + "a" * 64}
        self.service_image_fingerprints = {"target": "sha256:" + "b" * 64}

    def up(self) -> None:
        pass

    def read_flag(self, stage: Stage) -> str:
        return "flag{test}"

    def exec(self, cmd: str, **_kwargs: object) -> tuple[int, str]:
        raise AssertionError("no command expected")

    def down(self) -> tuple[bool, str]:
        return True, ""


class ScriptedClient:
    """Legacy chat interface returning one scripted content per call."""

    def __init__(self, turns: list[str]):
        self.turns = turns
        self.calls = 0

    def chat(
        self, _messages: list[dict], max_tokens: int, temperature: float = 0.2
    ) -> tuple[str, Usage, None]:
        content = self.turns[self.calls]
        self.calls += 1
        return content, Usage(prompt_tokens=20, completion_tokens=10), None


class ScriptedResultClient:
    """chat_result interface; each turn is (content, finish_reason)."""

    def __init__(self, turns: list[tuple[str, str | None]]):
        self.turns = turns
        self.calls = 0

    def chat(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("legacy chat should not be used")

    def chat_result(
        self, _messages: list[dict], max_tokens: int, temperature: float = 0.2
    ) -> ChatResult:
        content, finish = self.turns[self.calls]
        self.calls += 1
        return ChatResult(
            content,
            Usage(prompt_tokens=20, completion_tokens=10),
            None,
            ResponseMetadata(finish, not bool(content.strip()), False, max_tokens),
        )


def single_stage_task(tmp: Path) -> Task:
    return Task(
        "sample",
        tmp,
        "web",
        1,
        "Find the flag",
        stages=[Stage("one", "/flag", "target")],
    )


class TaxonomyTests(unittest.TestCase):
    def test_typed_endings_map_to_their_buckets(self) -> None:
        endings = [
            ("all stages captured", True, FAIL_SOLVED),
            ("env: compose up failed", False, FAIL_ENV),
            ("env: no container for service target", False, FAIL_ENV),
            ("source changed", False, FAIL_ENV),
            ("llm error", False, FAIL_PROVIDER),
            ("too many wrong submissions", False, FAIL_PROTOCOL),
            ("model produced no content 11x", False, FAIL_PROTOCOL),
            ("turn budget", False, FAIL_BUDGET),
            ("output token budget", False, FAIL_BUDGET),
            ("infra timeout", False, FAIL_BUDGET),
            ("context window exhausted", False, FAIL_BUDGET),
            ("wall_clock_exceeded", False, FAIL_BUDGET),
            ("", False, FAIL_NORMAL),
            ("a future, unknown ending", False, FAIL_NORMAL),
        ]
        for end_reason, solved, expected in endings:
            with self.subTest(end_reason=end_reason, solved=solved):
                self.assertEqual(classify_end_reason(end_reason, solved), expected)

    def test_solved_wins_over_any_end_reason(self) -> None:
        for end_reason in ("turn budget", "llm error", "env: compose failed", ""):
            with self.subTest(end_reason=end_reason):
                self.assertEqual(classify_end_reason(end_reason, True), FAIL_SOLVED)

    def test_free_text_is_never_substring_matched(self) -> None:
        coarse_hits = [
            "recovered after turn budget pressure",
            "task notes mention a compose network",
            "model answered near the readiness probe",
        ]
        for end_reason in coarse_hits:
            with self.subTest(end_reason=end_reason):
                self.assertEqual(classify_end_reason(end_reason, False), FAIL_NORMAL)

    def test_turn_budget_is_not_the_gave_up_bucket(self) -> None:
        self.assertEqual(classify_end_reason("turn budget", False), FAIL_BUDGET)
        self.assertNotEqual(FAIL_BUDGET, FAIL_NORMAL)
        self.assertEqual(classify_end_reason("", False), FAIL_NORMAL)


class RefusalTests(unittest.TestCase):
    def run_attempt_with(self, client: object) -> tuple[AttemptResult, list[dict]]:
        with tempfile.TemporaryDirectory() as tmp, patch("rangebench.runner.TaskEnv", FakeEnv):
            result = run_attempt(
                client,
                single_stage_task(Path(tmp)),
                1,
                "rb-test",
                Path(tmp),
                verbose=False,
            )
            records = [
                json.loads(line)
                for line in (Path(tmp) / "sample-t1.jsonl").read_text().splitlines()
            ]
        return result, records

    def test_refusal_text_counts_and_content_is_not_logged(self) -> None:
        client = ScriptedClient(
            [
                "I'm sorry, but I cannot help with that request.",
                "ANSWER: flag{test}",
            ]
        )
        result, records = self.run_attempt_with(client)
        self.assertEqual(result.end_reason, "all stages captured")
        self.assertEqual(result.refusals, 1)
        refusal = next(record for record in records if record["kind"] == "refusal")
        self.assertNotIn("content", refusal)
        end = records[-1]
        self.assertEqual(end["kind"], "end")
        self.assertEqual(end["refusals"], 1)

    def test_typed_finish_reason_counts_without_refusal_text(self) -> None:
        client = ScriptedResultClient(
            [
                ("Let me consider the next step for a moment.", "refusal"),
                ("ANSWER: flag{test}", "stop"),
            ]
        )
        result, records = self.run_attempt_with(client)
        self.assertEqual(result.end_reason, "all stages captured")
        self.assertEqual(result.refusals, 1)
        refusal = next(record for record in records if record["kind"] == "refusal")
        self.assertNotIn("content", refusal)
        self.assertNotIn("Let me consider", json.dumps(refusal))
        self.assertEqual(records[-1]["refusals"], 1)

    def test_plain_stall_is_not_counted_as_refusal(self) -> None:
        client = ScriptedClient(
            [
                "Thinking about the services on the host.",
                "ANSWER: flag{test}",
            ]
        )
        result, records = self.run_attempt_with(client)
        self.assertEqual(result.refusals, 0)
        self.assertFalse(any(record["kind"] == "refusal" for record in records))

    def test_empty_typed_refusals_count_while_retry_is_retained(self) -> None:
        client = ScriptedResultClient([("", "refusal")] * 11)
        result, records = self.run_attempt_with(client)
        self.assertEqual(result.end_reason, "model produced no content 11x")
        self.assertEqual(result.refusals, 11)
        self.assertEqual(sum(1 for record in records if record["kind"] == "refusal"), 11)

    def test_is_refusal_prefers_typed_reason_and_narrow_text(self) -> None:
        self.assertTrue(is_refusal("I can’t assist with that request."))
        self.assertTrue(is_refusal("I'm sorry, but I cannot comply with that."))
        self.assertFalse(is_refusal("I cannot reach the host from this box."))
        self.assertFalse(is_refusal("I'm sorry the scan failed, retrying with nmap."))
        self.assertFalse(is_refusal("I apologize for the delay; running the scan now."))
        self.assertFalse(is_refusal("scan the target", ResponseMetadata("stop", False, False, 8)))
        self.assertTrue(is_refusal("scan the target", ResponseMetadata("refusal", False, False, 8)))
        self.assertFalse(is_refusal("", ResponseMetadata(None, True, False, 8)))


class RunSummaryTests(unittest.TestCase):
    PREEXISTING_FIELDS = {
        "task",
        "category",
        "tier",
        "trial",
        "solved_stages",
        "stages_total",
        "solved",
        "scored",
        "wrong",
        "turns_used",
        "turns_budget",
        "effective_ctx_window",
        "commands",
        "service_image_ids",
        "service_image_fingerprints",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "compaction_tokens",
        "compaction_fallbacks",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "api_calls",
        "api_requests",
        "usage_reported_calls",
        "input_reported_calls",
        "output_reported_calls",
        "cache_read_reported_calls",
        "cache_write_reported_calls",
        "compaction_input_tokens",
        "compaction_output_tokens",
        "compaction_cache_read_tokens",
        "compaction_cache_write_tokens",
        "wall_s",
        "wall_clock_seconds",
        "wall_clock_exceeded",
        "end_reason",
        "max_output_tokens",
    }

    def test_run_json_and_summary_report_fail_class_and_refusals(self) -> None:
        task = Task(
            "sample",
            Path(tempfile.gettempdir()),
            "web",
            1,
            "Find the flag",
            stages=[Stage("one", "/flag", "web")],
        )
        result = AttemptResult(task.id, 1, effective_ctx_window=128000, end_reason="turn budget")
        result.refusals = 2
        args = argparse.Namespace(
            trials=1,
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
        with tempfile.TemporaryDirectory() as tmp, io.StringIO() as stdout:
            with (
                patch("rangebench.cli.RESULTS", Path(tmp)),
                patch("rangebench.cli.load_task", return_value=task),
                patch("rangebench.cli.ChatClient"),
                patch("rangebench.cli.run_attempt", return_value=result),
                patch("rangebench.cli._get_attacker_digest", return_value="sha256:test"),
                contextlib.redirect_stdout(stdout),
            ):
                cmd_run(args)
            saved = json.loads((Path(tmp) / "latest.json").read_text())["tasks"][0]
            output = stdout.getvalue()

        self.assertEqual(
            set(saved),
            self.PREEXISTING_FIELDS | {"fail_class", "refusals", "wall_clock_scale"},
        )
        self.assertEqual(saved["fail_class"], FAIL_BUDGET)
        self.assertEqual(saved["refusals"], 2)
        self.assertTrue(saved["scored"])
        self.assertEqual(saved["end_reason"], "turn budget")
        self.assertIn(
            "failure classes: solved=0 provider_error=0 env_error=0"
            " protocol_error=0 budget_exhausted=1 normal=0",
            output,
        )
        self.assertIn("refusals: 2 turns in 1/1 attempts", output)


class ClassifyScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.script = Path(__file__).resolve().parents[1] / "scripts" / "classify-failures.py"

    def run_script(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(self.script), *args],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_taxonomy_breakdown_respects_scored_false(self) -> None:
        doc = {
            "id": "run1",
            "tasks": [
                {"task": "a", "end_reason": "all stages captured", "solved": True, "scored": True},
                {"task": "b", "end_reason": "turn budget", "solved": False, "scored": True},
                {"task": "c", "end_reason": "llm error", "solved": False, "scored": False},
                {
                    "task": "d",
                    "end_reason": "env: compose down failed",
                    "solved": False,
                    "scored": False,
                },
                {
                    "task": "e",
                    "end_reason": "a future unknown ending",
                    "solved": False,
                    "scored": True,
                    "fail_class": FAIL_PROTOCOL,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            (results / "run1.json").write_text(json.dumps(doc))
            completed = self.run_script(str(results / "run1.json"))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            completed.stdout.strip(),
            f"{results / 'run1.json'}: tried=5 scored=3 solved=1 provider_error=0"
            " env_error=0 protocol_error=1 budget_exhausted=1 normal=0"
            " | invalid=2 solved=0 provider_error=1 env_error=1 protocol_error=0"
            " budget_exhausted=0 normal=0",
        )

    def test_results_dir_reads_latest_and_run_dir_reads_sibling_doc(self) -> None:
        latest = {"tasks": [{"task": "x", "end_reason": "", "solved": False}]}
        sibling = {
            "tasks": [
                {"task": "y", "end_reason": "all stages captured", "solved": True},
                {"task": "z", "end_reason": "", "solved": False},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            (results / "latest.json").write_text(json.dumps(latest))
            run_dir = results / "run2"
            run_dir.mkdir()
            (run_dir / "manifest.json").write_text("{}")
            (results / "run2.json").write_text(json.dumps(sibling))

            from_results = self.run_script(str(results))
            from_run_dir = self.run_script(str(run_dir))

        self.assertEqual(from_results.returncode, 0, from_results.stderr)
        self.assertEqual(from_run_dir.returncode, 0, from_run_dir.stderr)
        self.assertIn("scored=1 solved=0", from_results.stdout)
        self.assertIn("scored=2 solved=1", from_run_dir.stdout)

    def test_missing_argument_prints_usage_and_exits_two(self) -> None:
        completed = self.run_script()
        self.assertEqual(completed.returncode, 2)
        self.assertIn("usage:", completed.stdout)


if __name__ == "__main__":
    unittest.main()
