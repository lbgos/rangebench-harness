import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rangebench.agent import ChatResult, ResponseMetadata, Usage
from rangebench.env import ATTACKER_IMAGE, EnvError, Stage, Task, TaskEnv, load_all, truncate_output
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
    def chat(
        self, _messages: list[dict], max_tokens: int, temperature: float = 0.2
    ) -> tuple[str, Usage, None]:
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
                "sample",
                Path(tmp),
                "web",
                1,
                "Find the flag",
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

            def chat(
                self, _messages: list[dict], max_tokens: int, temperature: float = 0.2
            ) -> tuple[str, Usage, None]:
                self.calls += 1
                content = "ANSWER: flag{test}" if self.calls == 36 else "COMMAND:\ntrue"
                return content, Usage(prompt_tokens=20, completion_tokens=10), None

        self.assertTrue(all(task.turns is None for task in load_all()))
        client = SlowClient()
        with tempfile.TemporaryDirectory() as tmp, patch("rangebench.runner.TaskEnv", FakeEnv):
            task = Task(
                "sample",
                Path(tmp),
                "web",
                1,
                "Find the flag",
                stages=[Stage("one", "/flag", "target")],
            )
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
                "sample",
                Path(tmp),
                "web",
                1,
                "Find the flag",
                stages=[Stage("one", "/flag", "target")],
                turns=2,
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

    def _run_with_output(
        self, output: str, save_fails: bool = False
    ) -> tuple[list[dict], list[dict], list[str]]:
        """Run one command turn returning `output`; return (log records, messages, saved)."""
        saved: list[str] = []
        seen: list[list[dict]] = []

        class OutputEnv(FakeEnv):
            def exec(self, cmd: str, **_kwargs: object) -> tuple[int, str]:
                return 0, output

            def save_output(self, content: str) -> str:
                if save_fails:
                    raise EnvError("container gone")
                saved.append(content)
                return f"/work/obs/{len(saved):04d}.log"

        class CommandThenAnswer:
            def chat(
                self, messages: list[dict], max_tokens: int, temperature: float = 0.2
            ) -> tuple[str, Usage, None]:
                seen.append(list(messages))
                content = "COMMAND:\ncat big" if len(seen) == 1 else "ANSWER: flag{test}"
                return content, Usage(prompt_tokens=20, completion_tokens=10), None

        with tempfile.TemporaryDirectory() as tmp, patch("rangebench.runner.TaskEnv", OutputEnv):
            task = Task(
                "sample",
                Path(tmp),
                "web",
                1,
                "Find the flag",
                stages=[Stage("one", "/flag", "target")],
            )
            result = run_attempt(CommandThenAnswer(), task, 1, "rb-test", Path(tmp), verbose=False)
            records = [
                json.loads(line)
                for line in (Path(tmp) / "sample-t1.jsonl").read_text().splitlines()
            ]
        self.assertEqual(result.end_reason, "all stages captured")
        return records, seen[-1], saved

    def test_short_output_is_not_paged(self) -> None:
        output = "\n".join(f"line {i}" for i in range(400)) + "\n"
        records, messages, saved = self._run_with_output(output)
        self.assertEqual(saved, [])
        exec_rec = next(r for r in records if r["kind"] == "exec")
        self.assertNotIn("pager", exec_rec)
        self.assertEqual(exec_rec["out"], truncate_output(output))
        self.assertEqual(
            messages[-1]["content"], f"OBSERVATION (exit 0):\n{truncate_output(output)}"
        )

    def test_threshold_boundary_counts_final_line_without_newline(self) -> None:
        output = "\n".join(f"line {i}" for i in range(401))
        self.assertEqual(output.count("\n"), 400)  # old threshold would skip it
        records, _messages, saved = self._run_with_output(output)
        self.assertEqual(saved, [output])
        exec_rec = next(r for r in records if r["kind"] == "exec")
        self.assertEqual(exec_rec["pager"], {"path": "/work/obs/0001.log", "lines": 401})

    def test_long_output_is_saved_and_previewed(self) -> None:
        output = "\n".join(f"line {i}" for i in range(1, 1001)) + "\n"
        records, messages, saved = self._run_with_output(output)
        self.assertEqual(saved, [output])
        exec_rec = next(r for r in records if r["kind"] == "exec")
        self.assertEqual(exec_rec["pager"], {"path": "/work/obs/0001.log", "lines": 1000})
        self.assertNotIn("line 500\n", exec_rec["out"])
        obs = messages[-1]["content"].removeprefix("OBSERVATION (exit 0):\n")
        self.assertEqual(obs, exec_rec["out"])
        lines = obs.split("\n")
        self.assertEqual(len(lines), 120 + 1 + 40)
        self.assertEqual(lines[:120], [f"line {i}" for i in range(1, 121)])
        self.assertEqual(lines[121:], [f"line {i}" for i in range(961, 1001)])
        self.assertIn("840 lines omitted", lines[120])
        self.assertIn("sed -n '121,960p' /work/obs/0001.log", lines[120])

    def test_long_wide_output_preview_stays_bounded(self) -> None:
        output = "\n".join("x" * 500 for _ in range(1000))
        records, _messages, saved = self._run_with_output(output)
        exec_rec = next(r for r in records if r["kind"] == "exec")
        self.assertEqual(len(saved), 1)
        self.assertLessEqual(len(exec_rec["out"]), 6000)
        self.assertIn("/work/obs/0001.log", exec_rec["out"])

    def test_pager_save_failure_falls_back_to_truncate(self) -> None:
        output = "\n".join(f"line {i}" for i in range(1000))
        records, messages, _saved = self._run_with_output(output, save_fails=True)
        failed = next(r for r in records if r["kind"] == "pager-save-failed")
        self.assertIn("container gone", failed["error"])
        exec_rec = next(r for r in records if r["kind"] == "exec")
        self.assertNotIn("pager", exec_rec)
        self.assertEqual(exec_rec["out"], truncate_output(output))
        self.assertEqual(
            messages[-1]["content"], f"OBSERVATION (exit 0):\n{truncate_output(output)}"
        )

    def test_save_output_pipes_content_over_stdin(self) -> None:
        task = Task("sample", Path("/tmp"), "web", 1, "Find the flag")
        env = TaskEnv(task, "rb-test")
        ok = SimpleNamespace(returncode=0, stderr=b"")
        with patch("rangebench.env.subprocess.run", return_value=ok) as run:
            first = env.save_output("big \udcff output")
            second = env.save_output("more")
        self.assertEqual((first, second), ("/work/obs/0001.log", "/work/obs/0002.log"))
        argv = run.call_args_list[0].args[0]
        self.assertEqual(argv[:3], ["docker", "exec", "-i"])
        self.assertNotIn("big", " ".join(argv))
        self.assertEqual(
            run.call_args_list[0].kwargs["input"],
            "big \udcff output".encode("utf-8", "surrogatepass"),
        )

    def test_keep_writes_event_before_closing_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch("rangebench.runner.TaskEnv", FakeEnv):
            task = Task(
                "sample",
                Path(tmp),
                "web",
                1,
                "Find the flag",
                stages=[Stage("one", "/flag", "target")],
            )
            result = run_attempt(
                AnswerClient(), task, 1, "rb-test", Path(tmp), verbose=False, keep=True
            )
            records = [
                json.loads(line)
                for line in (Path(tmp) / "sample-t1.jsonl").read_text().splitlines()
            ]

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

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("rangebench.runner.TaskEnv", MissingImageEnv),
        ):
            task = Task(
                "sample",
                Path(tmp),
                "web",
                1,
                "Find the flag",
                stages=[Stage("one", "/flag", "target")],
            )
            result = run_attempt(NoCallsClient(), task, 1, "rb-test", Path(tmp), verbose=False)
        self.assertEqual(result.end_reason, "env: no container for service target")
        self.assertEqual(result.service_image_ids, {})
        self.assertEqual(result.service_image_fingerprints, {})

    def test_late_startup_error_preserves_inspected_service_images(self) -> None:
        class LateFailureEnv(FakeEnv):
            def up(self) -> None:
                raise EnvError("attacker container failed")

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("rangebench.runner.TaskEnv", LateFailureEnv),
        ):
            task = Task(
                "sample",
                Path(tmp),
                "web",
                1,
                "Find the flag",
                stages=[Stage("one", "/flag", "target")],
            )
            result = run_attempt(NoCallsClient(), task, 1, "rb-test", Path(tmp), verbose=False)
            records = [
                json.loads(line)
                for line in (Path(tmp) / "sample-t1.jsonl").read_text().splitlines()
            ]
        self.assertEqual(result.end_reason, "env: attacker container failed")
        expected = {"target": "sha256:" + "a" * 64}
        self.assertEqual(result.service_image_ids, expected)
        self.assertEqual(records[-1]["service_image_ids"], expected)
        self.assertEqual(result.service_image_fingerprints, {"target": "sha256:" + "b" * 64})
        self.assertEqual(
            records[-1]["service_image_fingerprints"], result.service_image_fingerprints
        )

    def test_oracle_failure_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            solution = Path(tmp) / "solution"
            solution.mkdir()
            (solution / "solve.sh").write_text("#!/bin/sh\nexit 0\n")
            task = Task(
                "sample",
                Path(tmp),
                "web",
                1,
                "Find the flag",
                stages=[Stage("one", "/flag", "target")],
            )
            with (
                patch("rangebench.runner.TaskEnv", FakeEnv),
                patch("subprocess.run", return_value=SimpleNamespace(returncode=0, stderr="")),
            ):
                with self.assertRaisesRegex(EnvError, "oracle failed"):
                    run_oracle(task)

    def test_oracle_dependency_copy_failure_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            solution = Path(tmp) / "solution"
            solution.mkdir()
            (solution / "solve.sh").write_text("#!/bin/sh\nexit 0\n")
            (solution / "helper.sh").write_text("#!/bin/sh\n")
            task = Task(
                "sample",
                Path(tmp),
                "web",
                1,
                "Find the flag",
                stages=[Stage("one", "/flag", "target")],
            )
            copies = [
                SimpleNamespace(returncode=0, stderr=""),
                SimpleNamespace(returncode=1, stderr="copy failed"),
            ]
            with (
                patch("rangebench.runner.TaskEnv", FakeEnv),
                patch("subprocess.run", side_effect=copies),
            ):
                with self.assertRaisesRegex(EnvError, "docker cp helper.sh"):
                    run_oracle(task)


if __name__ == "__main__":
    unittest.main()
