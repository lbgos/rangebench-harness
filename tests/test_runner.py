import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rangebench.agent import ChatResult, ResponseMetadata, Usage
from rangebench.env import ATTACKER_IMAGE, EnvError, Stage, Task, TaskEnv, load_all
from rangebench.runner import run_attempt, run_oracle


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
        return 0, "oracle did not find the flag" if cmd.startswith("bash /oracle") else ""

    def down(self) -> tuple[bool, str]:
        return True, ""


class AnswerClient:
    def chat(self, _messages: list[dict], max_tokens: int, temperature: float = 0.2) -> tuple[str, Usage, None]:
        return "ANSWER: flag{test}", Usage(), None


class NoCallsClient:
    def chat(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("model was called")


class RunnerTests(unittest.TestCase):
    def test_llm_event_records_response_shape_without_response_text(self) -> None:
        class MetadataClient:
            def chat(self, *_args: object, **_kwargs: object) -> None:
                raise AssertionError("legacy chat should not be used")

            def chat_result(
                self, _messages: list[dict], max_tokens: int, temperature: float = 0.2
            ) -> ChatResult:
                return ChatResult(
                    "ANSWER: flag{test}",
                    Usage(prompt_tokens=20, completion_tokens=10),
                    None,
                    ResponseMetadata("stop", False, True, max_tokens),
                )

        with tempfile.TemporaryDirectory() as tmp, patch("rangebench.runner.TaskEnv", FakeEnv):
            task = Task(
                "sample", Path(tmp), "web", 1, "Find the flag",
                stages=[Stage("one", "/flag", "target")],
            )
            result = run_attempt(MetadataClient(), task, 1, "rb-test", Path(tmp), verbose=False)
            records = [
                json.loads(line)
                for line in (Path(tmp) / "sample-t1.jsonl").read_text().splitlines()
            ]

        self.assertEqual(result.end_reason, "all stages captured")
        call = next(record for record in records if record["kind"] == "llm-call")
        self.assertEqual(
            call["response_meta"],
            {
                "finish_reason": "stop",
                "visible_content_empty": False,
                "reasoning_content_present": True,
                "requested_max_tokens": task.max_tokens,
            },
        )
        self.assertNotIn("content", call)

    def test_release_tasks_allow_more_than_previous_turn_cap(self) -> None:
        class SlowClient:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, _messages: list[dict], max_tokens: int, temperature: float = 0.2) -> tuple[str, Usage, None]:
                self.calls += 1
                content = "ANSWER: flag{test}" if self.calls == 36 else "COMMAND:\ntrue"
                return content, Usage(prompt_tokens=20, completion_tokens=10), None

        self.assertTrue(all(task.turns is None for task in load_all()))
        client = SlowClient()
        with tempfile.TemporaryDirectory() as tmp, patch("rangebench.runner.TaskEnv", FakeEnv):
            task = Task("sample", Path(tmp), "web", 1, "Find the flag", stages=[Stage("one", "/flag", "target")])
            result = run_attempt(client, task, 1, "rb-test", Path(tmp), verbose=False)
        self.assertEqual(result.end_reason, "all stages captured")
        self.assertEqual(result.turns_used, 36)
        self.assertEqual(result.commands, 35)

    def test_unencodable_model_command_gets_observation_without_aborting(self) -> None:
        class GuardedEnv(TaskEnv):
            def up(self) -> None:
                self.service_image_ids = {"target": "sha256:" + "a" * 64}
                self.service_image_fingerprints = {"target": "sha256:" + "b" * 64}

            def read_flag(self, stage: Stage) -> str:
                return "flag{test}"

            def down(self) -> tuple[bool, str]:
                return True, ""

        class SurrogateClient:
            def __init__(self) -> None:
                self.calls = 0

            def chat(
                self, _messages: list[dict], max_tokens: int, temperature: float = 0.2
            ) -> tuple[str, Usage, None]:
                self.calls += 1
                content = "COMMAND:\necho \udcff" if self.calls == 1 else "ANSWER: flag{test}"
                return content, Usage(prompt_tokens=20, completion_tokens=10), None

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("rangebench.runner.TaskEnv", GuardedEnv),
            patch("rangebench.env.subprocess.run") as run,
        ):
            task = Task(
                "sample", Path(tmp), "web", 1, "Find the flag",
                stages=[Stage("one", "/flag", "target")], turns=2,
            )
            result = run_attempt(SurrogateClient(), task, 1, "rb-test", Path(tmp), verbose=False)
            raw = (Path(tmp) / "sample-t1.jsonl").read_text()
            records = [json.loads(line) for line in raw.splitlines()]

        self.assertEqual(result.end_reason, "all stages captured")
        self.assertEqual(result.commands, 1)
        self.assertIn("\\udcff", raw)
        self.assertEqual(
            next(record for record in records if record["kind"] == "exec")["out"],
            "[command contains invalid UTF-8]",
        )
        run.assert_not_called()

    def test_keep_writes_event_before_closing_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch("rangebench.runner.TaskEnv", FakeEnv):
            task = Task("sample", Path(tmp), "web", 1, "Find the flag", stages=[Stage("one", "/flag", "target")])
            result = run_attempt(AnswerClient(), task, 1, "rb-test", Path(tmp), verbose=False, keep=True)
            records = [json.loads(line) for line in (Path(tmp) / "sample-t1.jsonl").read_text().splitlines()]

        self.assertEqual(result.end_reason, "all stages captured")
        self.assertEqual(result.service_image_ids, {"target": "sha256:" + "a" * 64})
        self.assertEqual(result.service_image_fingerprints, {"target": "sha256:" + "b" * 64})
        env_up = next(record for record in records if record["kind"] == "env-up")
        self.assertEqual(env_up["service_image_fingerprints"], result.service_image_fingerprints)
        kinds = [record["kind"] for record in records]
        self.assertLess(kinds.index("keep"), kinds.index("end"))

    def test_missing_service_image_invalidates_attempt_before_model_call(self) -> None:
        class MissingImageEnv(FakeEnv):
            def up(self) -> None:
                self.service_image_ids = {}
                self.service_image_fingerprints = {}
                raise EnvError("no container for service target")

        with tempfile.TemporaryDirectory() as tmp, patch("rangebench.runner.TaskEnv", MissingImageEnv):
            task = Task("sample", Path(tmp), "web", 1, "Find the flag", stages=[Stage("one", "/flag", "target")])
            result = run_attempt(NoCallsClient(), task, 1, "rb-test", Path(tmp), verbose=False)
        self.assertEqual(result.end_reason, "env: no container for service target")
        self.assertEqual(result.service_image_ids, {})
        self.assertEqual(result.service_image_fingerprints, {})

    def test_late_startup_error_preserves_inspected_service_images(self) -> None:
        class LateFailureEnv(FakeEnv):
            def up(self) -> None:
                raise EnvError("attacker container failed")

        with tempfile.TemporaryDirectory() as tmp, patch("rangebench.runner.TaskEnv", LateFailureEnv):
            task = Task("sample", Path(tmp), "web", 1, "Find the flag", stages=[Stage("one", "/flag", "target")])
            result = run_attempt(NoCallsClient(), task, 1, "rb-test", Path(tmp), verbose=False)
            records = [json.loads(line) for line in (Path(tmp) / "sample-t1.jsonl").read_text().splitlines()]
        self.assertEqual(result.end_reason, "env: attacker container failed")
        expected = {"target": "sha256:" + "a" * 64}
        self.assertEqual(result.service_image_ids, expected)
        self.assertEqual(records[-1]["service_image_ids"], expected)
        self.assertEqual(result.service_image_fingerprints, {"target": "sha256:" + "b" * 64})
        self.assertEqual(records[-1]["service_image_fingerprints"], result.service_image_fingerprints)

    def test_oracle_failure_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            solution = Path(tmp) / "solution"
            solution.mkdir()
            (solution / "solve.sh").write_text("#!/bin/sh\nexit 0\n")
            task = Task("sample", Path(tmp), "web", 1, "Find the flag", stages=[Stage("one", "/flag", "target")])
            with patch("rangebench.runner.TaskEnv", FakeEnv), patch(
                "subprocess.run", return_value=SimpleNamespace(returncode=0, stderr="")
            ):
                with self.assertRaisesRegex(EnvError, "oracle failed"):
                    run_oracle(task)

    def test_oracle_dependency_copy_failure_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            solution = Path(tmp) / "solution"
            solution.mkdir()
            (solution / "solve.sh").write_text("#!/bin/sh\nexit 0\n")
            (solution / "helper.sh").write_text("#!/bin/sh\n")
            task = Task("sample", Path(tmp), "web", 1, "Find the flag", stages=[Stage("one", "/flag", "target")])
            copies = [
                SimpleNamespace(returncode=0, stderr=""),
                SimpleNamespace(returncode=1, stderr="copy failed"),
            ]
            with patch("rangebench.runner.TaskEnv", FakeEnv), patch(
                "subprocess.run", side_effect=copies
            ):
                with self.assertRaisesRegex(EnvError, "docker cp helper.sh"):
                    run_oracle(task)


if __name__ == "__main__":
    unittest.main()
