import argparse
import errno
import json
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from rangebench.agent import AnthropicChatClient, ChatClient, Usage
from rangebench.cli import (
    _get_git_commit,
    _get_task_set_hash,
    _write_manifest,
    cmd_preflight,
    cmd_run,
)
from rangebench.env import (
    MAX_COMMAND_BYTES,
    EnvError,
    Stage,
    Task,
    TaskEnv,
    _run,
    image_content_fingerprint,
)
from rangebench.runner import AttemptResult, _maybe_compact


class AccountingTests(unittest.TestCase):
    def test_missing_token_usage_invalidates_successful_response(self) -> None:
        class Response:
            def __init__(self, body: dict):
                self.body = body

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self) -> bytes:
                return json.dumps(self.body).encode()

        cases = (
            (
                ChatClient("http://localhost/v1", "key", "test"),
                "prompt_tokens",
                "completion_tokens",
                {"choices": [{"message": {"content": "ANSWER: flag{test}"}}]},
            ),
            (
                AnthropicChatClient("http://localhost", "key", "test"),
                "input_tokens",
                "output_tokens",
                {"content": [{"type": "text", "text": "ANSWER: flag{test}"}]},
            ),
        )
        for client, input_key, output_key, body in cases:
            for missing_key in (input_key, output_key):
                with self.subTest(client=type(client).__name__, missing=missing_key):
                    body["usage"] = {input_key: 0, output_key: 0}
                    del body["usage"][missing_key]
                    with patch(
                        "rangebench.agent.urllib.request.urlopen", return_value=Response(body)
                    ) as urlopen:
                        content, usage, error = client.chat(
                            [{"role": "user", "content": "test"}], 10
                        )
                    self.assertEqual(content, "")
                    self.assertEqual(error, "missing input or output token usage")
                    self.assertEqual((usage.calls, usage.requests, usage.reported_calls), (1, 1, 1))
                    self.assertEqual(usage.input_reported_calls, int(missing_key != input_key))
                    self.assertEqual(usage.output_reported_calls, int(missing_key != output_key))
                    urlopen.assert_called_once()

            if isinstance(client, AnthropicChatClient):
                with self.subTest(client="AnthropicChatClient", foreign_usage_fields=True):
                    body["usage"] = {"prompt_tokens": 10, "completion_tokens": 10}
                    with patch(
                        "rangebench.agent.urllib.request.urlopen", return_value=Response(body)
                    ):
                        content, usage, error = client.chat(
                            [{"role": "user", "content": "test"}], 10
                        )
                    self.assertEqual(content, "")
                    self.assertEqual(error, "missing input or output token usage")
                    self.assertEqual(
                        (usage.input_reported_calls, usage.output_reported_calls), (0, 0)
                    )

            with self.subTest(client=type(client).__name__, zero_usage=True):
                body["usage"] = {input_key: 0, output_key: 0}
                with patch("rangebench.agent.urllib.request.urlopen", return_value=Response(body)):
                    content, usage, error = client.chat([{"role": "user", "content": "test"}], 10)
                self.assertIn("ANSWER:", content)
                self.assertIsNone(error)
                self.assertEqual((usage.input_reported_calls, usage.output_reported_calls), (1, 1))

            for malformed in (True, -1, "12", 1.5, None):
                with self.subTest(client=type(client).__name__, malformed=malformed):
                    body["usage"] = {input_key: malformed, output_key: 0}
                    with patch(
                        "rangebench.agent.urllib.request.urlopen", return_value=Response(body)
                    ) as urlopen:
                        content, usage, error = client.chat(
                            [{"role": "user", "content": "test"}], 10
                        )
                    self.assertEqual(content, "")
                    self.assertEqual(
                        error,
                        f"malformed token usage: {input_key} must be a non-negative integer",
                    )
                    self.assertEqual((usage.calls, usage.requests, usage.reported_calls), (0, 1, 0))
                    urlopen.assert_called_once()

        usage = Usage()
        with self.assertRaisesRegex(ValueError, "prompt_tokens_details.cached_tokens"):
            usage.add(
                {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "prompt_tokens_details": {"cached_tokens": True},
                }
            )

    def test_manifest_keeps_source_fingerprint_from_run_start(self) -> None:
        fingerprint = {
            "harness_commit": "started-commit",
            "harness_source_hash": "started-harness",
            "task_set_hash": "started-tasks",
        }
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("rangebench.cli._get_git_commit", side_effect=AssertionError("recomputed")),
                patch("rangebench.cli._get_harness_hash", side_effect=AssertionError("recomputed")),
                patch(
                    "rangebench.cli._get_task_set_hash", side_effect=AssertionError("recomputed")
                ),
            ):
                _write_manifest(Path(tmp), {**fingerprint, "tasks": []}, {"status": "running"})
                _write_manifest(Path(tmp), {**fingerprint, "tasks": []}, {"status": "completed"})
            manifest = json.loads((Path(tmp) / "manifest.json").read_text())
        self.assertEqual({key: manifest[key] for key in fingerprint}, fingerprint)

    def test_source_drift_stops_run_before_next_trial(self) -> None:
        task = Task(
            "sample", Path("/tmp"), "web", 1, "Find the flag", stages=[Stage("one", "/flag", "web")]
        )
        args = argparse.Namespace(
            trials=2,
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
        first = {
            "harness_commit": "started-commit",
            "harness_source_hash": "started-harness",
            "task_set_hash": "started-tasks",
        }
        changed = {**first, "task_set_hash": "changed-tasks"}
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("rangebench.cli.RESULTS", Path(tmp)),
                patch("rangebench.cli.load_task", return_value=task),
                patch("rangebench.cli.ChatClient"),
                patch("rangebench.cli._get_attacker_digest", return_value="sha256:test"),
                patch(
                    "rangebench.cli._source_fingerprint", side_effect=[first, first, first, changed]
                ),
                patch(
                    "rangebench.cli.run_attempt",
                    return_value=AttemptResult(
                        task.id,
                        1,
                        service_image_ids={"api": "sha256:" + "a" * 64},
                        service_image_fingerprints={"api": "sha256:" + "b" * 64},
                    ),
                ) as attempt,
            ):
                with self.assertRaisesRegex(SystemExit, "source changed"):
                    cmd_run(args)
            self.assertEqual(attempt.call_count, 1)
            manifest = json.loads(next(Path(tmp).glob("*/manifest.json")).read_text())
            saved = json.loads((Path(tmp) / "latest.json").read_text())["tasks"][0]
        self.assertEqual(manifest["status"], "source_changed")
        self.assertEqual(manifest["task_set_hash"], first["task_set_hash"])
        self.assertEqual(manifest["observed_source"], changed)
        self.assertEqual(manifest["task_count"], 1)
        self.assertEqual(manifest["service_image_ids"][0]["images"], saved["service_image_ids"])
        self.assertEqual(manifest["service_image_fingerprints"][0]["images"], saved["service_image_fingerprints"])
        self.assertFalse(saved["scored"])
        self.assertEqual(saved["end_reason"], "source changed")
        self.assertEqual(saved["fail_class"], "env_error")

    def test_reasoning_is_part_of_completion_tokens(self) -> None:
        usage = Usage()
        usage.add(
            {
                "prompt_tokens": 10,
                "completion_tokens": 40,
                "completion_tokens_details": {"reasoning_tokens": 30},
            }
        )
        self.assertEqual(usage.completion_tokens, 40)
        self.assertEqual(usage.reasoning_tokens, 30)

    def test_total_tokens_covers_split_reasoning_output(self) -> None:
        usage = Usage()
        usage.add(
            {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 76,
                "completion_tokens_details": {"reasoning_tokens": 70},
            }
        )
        self.assertEqual(
            (usage.input_tokens, usage.output_tokens, usage.reasoning_tokens), (4, 72, 70)
        )

        nested = Usage()
        nested.add(
            {
                "prompt_tokens": 55,
                "completion_tokens": 37,
                "total_tokens": 92,
                "completion_tokens_details": {"reasoning_tokens": 34},
            }
        )
        self.assertEqual((nested.input_tokens, nested.output_tokens), (55, 37))

        for total in (True, -1, 5, "76"):
            with self.subTest(total=total):
                invalid = Usage()
                with self.assertRaisesRegex(ValueError, "total_tokens"):
                    invalid.add({"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": total})
                self.assertEqual(invalid.as_dict(), Usage().as_dict())

    def test_openai_cache_and_missing_metadata_are_distinct(self) -> None:
        usage = Usage()
        usage.add(
            {
                "prompt_tokens": 100,
                "completion_tokens": 40,
                "prompt_tokens_details": {"cached_tokens": 60},
                "completion_tokens_details": {"reasoning_tokens": 30},
            }
        )
        usage.add({"prompt_tokens": 20, "completion_tokens": 5})
        self.assertEqual((usage.input_tokens, usage.output_tokens), (120, 45))
        self.assertEqual(usage.cache_read_tokens, 60)
        self.assertIsNone(usage.cache_write_tokens)
        self.assertEqual((usage.cache_read_reported_calls, usage.calls), (1, 2))

        zero = Usage()
        zero.add(
            {
                "input_tokens": 10,
                "output_tokens": 4,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 3},
            }
        )
        self.assertEqual(zero.cache_read_tokens, 0)
        self.assertEqual(zero.reasoning_tokens, 3)

    def test_anthropic_input_includes_both_cache_buckets(self) -> None:
        usage = Usage()
        usage.add(
            {
                "input_tokens": 20,
                "cache_creation_input_tokens": 100,
                "cache_read_input_tokens": 200,
                "output_tokens": 30,
            },
            provider="anthropic",
        )
        self.assertEqual((usage.input_tokens, usage.output_tokens), (320, 30))
        self.assertEqual((usage.cache_read_tokens, usage.cache_write_tokens), (200, 100))
        self.assertEqual(
            (usage.cache_read_reported_calls, usage.cache_write_reported_calls), (1, 1)
        )

    def test_compaction_usage_is_separate_and_in_total(self) -> None:
        class Compactor:
            def chat(self, messages: list[dict], max_tokens: int, temperature: float = 0.2):
                usage = Usage()
                usage.add(
                    {
                        "prompt_tokens": 200,
                        "completion_tokens": 50,
                        "prompt_tokens_details": {"cached_tokens": 100},
                    }
                )
                return "state snapshot", usage, None

        res = AttemptResult("sample", 1)
        events: list[tuple[str, dict]] = []
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            *(
                message
                for _ in range(3)
                for message in (
                    {"role": "assistant", "content": "history " * 250},
                    {"role": "user", "content": "observation"},
                )
            ),
        ]
        compacted = _maybe_compact(
            Compactor(),
            messages,
            res,
            ctx_window=3000,
            reserve=100,
            keep_tail=1,
            threshold=0.3,
            use_llm=True,
            emit=lambda kind, **kv: events.append((kind, kv)),
        )
        self.assertLess(len(compacted), len(messages))
        self.assertEqual(res.compaction_tokens, 250)
        self.assertEqual(res.total_usage().input_tokens, 200)
        self.assertEqual(res.total_usage().cache_read_tokens, 100)
        self.assertEqual(events[0][0], "compaction-call")
        self.assertEqual(events[0][1]["usage"]["cache_read_tokens"], 100)

    def test_challenge_code_changes_task_set_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tasks = Path(tmp) / "tasks"
            challenge = tasks / "sample" / "app.py"
            challenge.parent.mkdir(parents=True)
            (challenge.parent / "task.json").write_text('{"id":"sample"}')
            challenge.write_text("print('old')")
            with patch("rangebench.cli.TASKS_DIR", tasks):
                original = _get_task_set_hash()
                challenge.write_text("print('new')")
                changed = _get_task_set_hash()
        self.assertNotEqual(original, changed)

    def test_docker_timeout_is_an_environment_error(self) -> None:
        with patch(
            "rangebench.env.subprocess.run", side_effect=subprocess.TimeoutExpired("docker", 1)
        ):
            with self.assertRaisesRegex(EnvError, "timed out after 1s"):
                _run(["docker", "compose", "up"], timeout=1)

    def test_docker_daemon_outage_is_not_a_model_command_failure(self) -> None:
        task = Task("sample", Path("/tmp"), "web", 1, "Find the flag")
        env = TaskEnv(task, "rb-test")
        exec_failure = subprocess.CompletedProcess([], 1, "", "Cannot connect to the Docker daemon")
        probe_failure = subprocess.CompletedProcess([], 1, "", "daemon unavailable")
        with patch("rangebench.env.subprocess.run", side_effect=[exec_failure, probe_failure]):
            with self.assertRaisesRegex(EnvError, "Docker exec failed"):
                env.exec("echo ok")

    def test_dead_attacker_container_is_not_a_model_command_failure(self) -> None:
        task = Task("sample", Path("/tmp"), "web", 1, "Find the flag")
        env = TaskEnv(task, "rb-test")
        failure = subprocess.CompletedProcess(
            [], 1, "", "Error response from daemon: Container rb-test-atk is not running"
        )
        probe_failure = subprocess.CompletedProcess([], 1, "", "container not running")
        with patch("rangebench.env.subprocess.run", side_effect=[failure, probe_failure]):
            with self.assertRaisesRegex(EnvError, "Docker exec failed"):
                env.exec("echo ok")

    def test_model_stderr_that_looks_like_docker_error_is_scored(self) -> None:
        task = Task("sample", Path("/tmp"), "web", 1, "Find the flag")
        env = TaskEnv(task, "rb-test")
        command = subprocess.CompletedProcess([], 1, "", "Error response from daemon: pretend")
        probe = subprocess.CompletedProcess([], 0, "", "")
        with patch("rangebench.env.subprocess.run", side_effect=[command, probe]):
            rc, out = env.exec("printf 'Error response from daemon: pretend' >&2; exit 1")
        self.assertEqual(rc, 1)
        self.assertIn("pretend", out)

    def test_binary_attacker_output_is_returned_with_replacement_characters(self) -> None:
        task = Task("sample", Path("/tmp"), "web", 1, "Find the flag")
        env = TaskEnv(task, "rb-test")
        actual_run = subprocess.run

        def binary_output(_cmd: list[str], **kwargs):
            return actual_run(
                [
                    sys.executable,
                    "-c",
                    "import sys; sys.stdout.buffer.write(b'out\\xff'); "
                    "sys.stderr.buffer.write(b'err\\xfe')",
                ],
                **kwargs,
            )

        with patch("rangebench.env.subprocess.run", side_effect=binary_output):
            self.assertEqual(env.exec("cat binary-file"), (0, "out\ufffd\n[stderr]\nerr\ufffd"))

    def test_oversized_utf8_command_is_scored_without_spawning_docker(self) -> None:
        task = Task("sample", Path("/tmp"), "web", 1, "Find the flag")
        env = TaskEnv(task, "rb-test")
        command = "é" * (MAX_COMMAND_BYTES // 2 + 1)
        with patch("rangebench.env.subprocess.run") as run:
            rc, output = env.exec(command)
        self.assertEqual(rc, 1)
        self.assertIn("exceeds", output)
        self.assertNotIn(command[:20], output)
        run.assert_not_called()

    def test_invalid_utf8_command_is_scored_without_spawning_docker(self) -> None:
        task = Task("sample", Path("/tmp"), "web", 1, "Find the flag")
        env = TaskEnv(task, "rb-test")
        with patch("rangebench.env.subprocess.run") as run:
            self.assertEqual(env.exec("echo \udcff"), (1, "[command contains invalid UTF-8]"))
        run.assert_not_called()

    def test_e2big_from_subprocess_is_scored_without_command_content(self) -> None:
        task = Task("sample", Path("/tmp"), "web", 1, "Find the flag")
        env = TaskEnv(task, "rb-test")
        with patch("rangebench.env.subprocess.run", side_effect=OSError(errno.E2BIG, "too long")):
            self.assertEqual(
                env.exec("private-model-command"),
                (1, "[command could not start: argument list too long]"),
            )

    def test_other_subprocess_start_error_invalidates_attempt(self) -> None:
        task = Task("sample", Path("/tmp"), "web", 1, "Find the flag")
        env = TaskEnv(task, "rb-test")
        with patch("rangebench.env.subprocess.run", side_effect=OSError(errno.ENOENT, "missing")):
            with self.assertRaisesRegex(EnvError, "Docker exec could not start"):
                env.exec("echo ok")

    def test_attacker_uses_recorded_image_id(self) -> None:
        task = Task("sample", Path("/tmp"), "web", 1, "Find the flag")
        env = TaskEnv(task, "rb-test", "sha256:recorded")
        with (
            patch("rangebench.env._run") as run,
            patch.object(env, "_verify_compose_config", return_value=["target"]),
            patch.object(env, "verify_isolation"),
            patch.object(env, "inspect_service_images", return_value={"target": "sha256:recorded"}),
            patch.object(env, "inspect_service_image_fingerprints", return_value={"target": "sha256:content"}),
        ):
            env.up()
        docker_run = next(
            call.args[0] for call in run.call_args_list if call.args[0][:2] == ["docker", "run"]
        )
        self.assertEqual(docker_run[-3:], ["sha256:recorded", "sleep", "infinity"])
        self.assertEqual(env.service_image_ids, {"target": "sha256:recorded"})
        self.assertEqual(env.service_image_fingerprints, {"target": "sha256:content"})

    def test_service_image_inspection_requires_each_service_id(self) -> None:
        task = Task("sample", Path("/tmp"), "web", 1, "Find the flag")
        env = TaskEnv(task, "rb-test")
        digest = "sha256:" + "a" * 64
        responses = [
            subprocess.CompletedProcess([], 0, "container-1\n", ""),
            subprocess.CompletedProcess([], 0, digest + "\n", ""),
            subprocess.CompletedProcess([], 0, "container-2\n", ""),
            subprocess.CompletedProcess([], 0, digest + "\n", ""),
        ]
        with patch("rangebench.env._run", side_effect=responses) as run:
            images = env.inspect_service_images(["docker", "compose"], ["api", "db"])
        self.assertEqual(images, {"api": digest, "db": digest})
        self.assertEqual(
            run.call_args.args[0], ["docker", "inspect", "--format", "{{.Image}}", "container-2"]
        )

        with patch("rangebench.env._run", return_value=subprocess.CompletedProcess([], 0, "", "")):
            with self.assertRaisesRegex(EnvError, "no container for service api"):
                env.inspect_service_images(["docker", "compose"], ["api"])

        bad = [
            subprocess.CompletedProcess([], 0, "container-1\n", ""),
            subprocess.CompletedProcess([], 0, "nginx:latest\n", ""),
        ]
        with patch("rangebench.env._run", side_effect=bad):
            with self.assertRaisesRegex(EnvError, "invalid image ID for service api"):
                env.inspect_service_images(["docker", "compose"], ["api"])

    def test_image_fingerprint_ignores_only_compose_project_label(self) -> None:
        image = {
            "Config": {
                "Env": ["PATH=/bin", "MODE=production"],
                "Cmd": ["run"],
                "Labels": {"com.docker.compose.project": "run-a", "app.version": "1"},
            },
            "RootFS": {"Type": "layers", "Layers": ["sha256:layer-a", "sha256:layer-b"]},
            "Os": "linux",
            "Architecture": "amd64",
        }
        changed_project = json.loads(json.dumps(image))
        changed_project["Config"]["Labels"]["com.docker.compose.project"] = "run-b"
        self.assertEqual(image_content_fingerprint(image), image_content_fingerprint(changed_project))
        self.assertEqual(image["Config"]["Labels"]["com.docker.compose.project"], "run-a")

        for field, change in (
            ("layer contents", lambda candidate: candidate["RootFS"]["Layers"].__setitem__(1, "sha256:other")),
            ("layers", lambda candidate: candidate["RootFS"]["Layers"].reverse()),
            ("config", lambda candidate: candidate["Config"]["Env"].append("OTHER=1")),
            ("other label", lambda candidate: candidate["Config"]["Labels"].update({"app.version": "2"})),
            ("os", lambda candidate: candidate.update({"Os": "windows"})),
            ("architecture", lambda candidate: candidate.update({"Architecture": "arm64"})),
        ):
            with self.subTest(field=field):
                changed = json.loads(json.dumps(image))
                change(changed)
                self.assertNotEqual(image_content_fingerprint(image), image_content_fingerprint(changed))

    def test_image_fingerprint_normalizes_inspect_api_defaults(self) -> None:
        image = {
            "Config": {"Cmd": ["run"], "Env": ["MODE=production"]},
            "RootFS": {"Layers": ["sha256:layer"]},
            "Os": "linux",
            "Architecture": "amd64",
        }
        older_api = json.loads(json.dumps(image))
        older_api["Config"].update({
            "Hostname": "",
            "Domainname": "",
            "AttachStdin": False,
            "Image": "",
            "Entrypoint": None,
            "Labels": {"com.docker.compose.project": "run-a"},
            "OnBuild": [],
            "User": "",
            "Volumes": {},
            "WorkingDir": "",
        })
        self.assertEqual(image_content_fingerprint(image), image_content_fingerprint(older_api))

        for field, value in (
            ("Cmd", ["different"]),
            ("Healthcheck", {"Test": ["CMD", "true"]}),
            ("ArgsEscaped", True),
            ("Shell", ["/bin/bash", "-c"]),
        ):
            with self.subTest(field=field):
                changed = json.loads(json.dumps(image))
                changed["Config"][field] = value
                self.assertNotEqual(image_content_fingerprint(image), image_content_fingerprint(changed))

    def test_service_fingerprints_inspect_exact_image_ids(self) -> None:
        task = Task("sample", Path("/tmp"), "web", 1, "Find the flag")
        env = TaskEnv(task, "rb-test")
        image_a = "sha256:" + "a" * 64
        image_b = "sha256:" + "b" * 64
        inspected = {
            "Config": {"Labels": {"com.docker.compose.project": "rb-test"}},
            "RootFS": {"Layers": ["sha256:layer"]},
            "Os": "linux",
            "Architecture": "amd64",
        }
        second = json.loads(json.dumps(inspected))
        second["Config"]["Labels"]["com.docker.compose.project"] = "other-run"
        responses = [
            subprocess.CompletedProcess([], 0, json.dumps([image]), "")
            for image in (inspected, second)
        ]
        with patch("rangebench.env._run", side_effect=responses) as run:
            fingerprints = env.inspect_service_image_fingerprints({"api": image_a, "worker": image_b})
        self.assertEqual(fingerprints["api"], fingerprints["worker"])
        self.assertEqual(fingerprints["api"], image_content_fingerprint(inspected))
        self.assertEqual(
            [call.args[0] for call in run.call_args_list],
            [["docker", "image", "inspect", image_a], ["docker", "image", "inspect", image_b]],
        )

    def test_attacker_command_timeout_is_enforced_inside_container(self) -> None:
        task = Task("sample", Path("/tmp"), "web", 1, "Find the flag")
        env = TaskEnv(task, "rb-test")
        timed_out_command = subprocess.CompletedProcess(
            [], 124, "", "timeout: sending signal TERM to command ‘bash’"
        )
        with patch(
            "rangebench.env.subprocess.run",
            return_value=timed_out_command,
        ) as run:
            rc, out = env.exec("sleep 10", timeout=1)
        self.assertEqual((rc, out), (124, "[command timed out after 1s]"))
        self.assertEqual(
            run.call_args.args[0][-9:],
            [
                "timeout",
                "--verbose",
                "--kill-after=5s",
                "1",
                "bash",
                "-lc",
                'exec 2>&1; exec bash -lc "$1"',
                "_",
                "sleep 10",
            ],
        )
        self.assertEqual(run.call_args.kwargs["timeout"], 16)

    def test_command_exit_124_keeps_its_output(self) -> None:
        task = Task("sample", Path("/tmp"), "web", 1, "Find the flag")
        env = TaskEnv(task, "rb-test")
        command = subprocess.CompletedProcess([], 124, "result", "")
        with patch("rangebench.env.subprocess.run", return_value=command):
            self.assertEqual(env.exec("printf result; exit 124"), (124, "result"))

    def test_kill_escalation_is_reported_as_command_timeout(self) -> None:
        task = Task("sample", Path("/tmp"), "web", 1, "Find the flag")
        env = TaskEnv(task, "rb-test")
        command = subprocess.CompletedProcess(
            [], 137, "", "timeout: sending signal TERM to command ‘bash’\n"
        )
        with patch("rangebench.env.subprocess.run", return_value=command):
            self.assertEqual(env.exec("sleep 10", timeout=1), (124, "[command timed out after 1s]"))

    def test_docker_timeout_is_not_scored_when_attacker_is_unavailable(self) -> None:
        task = Task("sample", Path("/tmp"), "web", 1, "Find the flag")
        env = TaskEnv(task, "rb-test")
        failed_probe = subprocess.CompletedProcess([], 1, "", "container not running")
        with patch(
            "rangebench.env.subprocess.run",
            side_effect=[subprocess.TimeoutExpired("docker exec", 1), failed_probe],
        ):
            with self.assertRaisesRegex(EnvError, "Docker exec unavailable"):
                env.exec("sleep 10", timeout=1)

    def test_outer_timeout_is_invalid_even_when_attacker_responds(self) -> None:
        task = Task("sample", Path("/tmp"), "web", 1, "Find the flag")
        env = TaskEnv(task, "rb-test")
        healthy_probe = subprocess.CompletedProcess([], 0, "", "")
        with patch(
            "rangebench.env.subprocess.run",
            side_effect=[subprocess.TimeoutExpired("docker exec", 16), healthy_probe],
        ):
            with self.assertRaisesRegex(EnvError, "did not finish"):
                env.exec("sleep 10", timeout=1)

    def test_invalid_trial_exits_nonzero_after_writing_artifacts(self) -> None:
        task = Task(
            "sample", Path("/tmp"), "web", 1, "Find the flag", stages=[Stage("one", "/flag", "web")]
        )
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
        result = AttemptResult(task.id, 1, effective_ctx_window=64000, end_reason="llm error")
        result.compaction_fallbacks = 1
        result.model_usage.add(
            {
                "prompt_tokens": 100,
                "completion_tokens": 40,
                "prompt_tokens_details": {"cached_tokens": 60},
            }
        )
        result.compaction_usage.add(
            {
                "prompt_tokens": 20,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 0},
            }
        )
        result.prompt_tokens = 100
        result.completion_tokens = 40
        result.service_image_ids = {"api": "sha256:" + "a" * 64}
        result.service_image_fingerprints = {"api": "sha256:" + "b" * 64}
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("rangebench.cli.RESULTS", Path(tmp)),
                patch("rangebench.cli.load_task", return_value=task),
                patch("rangebench.cli.ChatClient"),
                patch(
                    "rangebench.cli.run_attempt",
                    return_value=result,
                ) as run_attempt,
                patch("rangebench.cli._get_attacker_digest", return_value="sha256:test"),
            ):
                with self.assertRaises(SystemExit) as caught:
                    cmd_run(args)
            self.assertEqual(caught.exception.code, 1)
            self.assertEqual(run_attempt.call_args.kwargs["attacker_image"], "sha256:test")
            saved = json.loads((Path(tmp) / "latest.json").read_text())["tasks"][0]
            self.assertFalse(saved["scored"])
            self.assertEqual(saved["effective_ctx_window"], 64000)
            self.assertEqual(saved["compaction_fallbacks"], 1)
            self.assertEqual((saved["input_tokens"], saved["output_tokens"]), (120, 45))
            self.assertEqual(saved["cache_read_tokens"], 60)
            self.assertEqual(saved["compaction_cache_read_tokens"], 0)
            self.assertIsNone(saved["cache_write_tokens"])
            self.assertEqual(saved["cache_read_reported_calls"], 2)
            self.assertEqual(saved["service_image_ids"], result.service_image_ids)
            self.assertEqual(saved["service_image_fingerprints"], result.service_image_fingerprints)
            manifest = next(Path(tmp).glob("*/manifest.json"))
            manifest_data = json.loads(manifest.read_text())
            self.assertEqual(manifest_data["status"], "completed_with_errors")
            self.assertEqual(
                manifest_data["usage_coverage"],
                {
                    "api_calls": 2,
                    "api_requests": 0,
                    "usage_reported_calls": 2,
                    "input_reported_calls": 2,
                    "output_reported_calls": 2,
                },
            )
            self.assertEqual(
                manifest_data["service_image_ids"],
                [{"task": task.id, "trial": 1, "images": result.service_image_ids}],
            )
            self.assertEqual(
                manifest_data["service_image_fingerprints"],
                [{"task": task.id, "trial": 1, "images": result.service_image_fingerprints}],
            )

    def test_preflight_stops_if_attacker_build_fails(self) -> None:
        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch("rangebench.cli.subprocess.run") as run,
        ):
            run.side_effect = [
                subprocess.CompletedProcess([], 0, "", ""),
                subprocess.CompletedProcess([], 0, "Docker Compose version 2", ""),
                subprocess.CalledProcessError(1, "docker build"),
            ]
            with self.assertRaises(subprocess.CalledProcessError):
                cmd_preflight(argparse.Namespace())
            self.assertEqual(run.call_count, 3)
            self.assertIn("attacker", str(run.call_args.args[0]))

    def test_git_commit_is_read_from_benchmark_checkout(self) -> None:
        with patch("rangebench.cli.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, "deadbeef\n", "")
            self.assertEqual(_get_git_commit(), "deadbeef")
            self.assertEqual(run.call_args.kwargs["cwd"], Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    unittest.main()
