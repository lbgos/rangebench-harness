import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rangebench.agent import Usage
from rangebench.cli import cmd_probe, cmd_run, main
from rangebench.env import ATTACKER_IMAGE, EnvError, Stage, Task, load_task, wall_clock_default
from rangebench.runner import AttemptResult, run_attempt, wall_clock_scale_for


class FakeEnv:
    def __init__(self, task: Task, project: str, attacker_image: str = ATTACKER_IMAGE):
        self.task = task
        self.attacker = f"{project}-atk"
        self.attacker_image = attacker_image
        self.service_image_ids = {"target": "sha256:" + "a" * 64}
        self.service_image_fingerprints = {"target": "sha256:" + "b" * 64}
        self.seen_timeouts: list[int] = []

    def up(self) -> None:
        pass

    def read_flag(self, stage: Stage) -> str:
        return "flag{test}"

    def exec(self, cmd: str, **kwargs: object) -> tuple[int, str]:
        timeout = kwargs.get("timeout")
        assert isinstance(timeout, int)
        self.seen_timeouts.append(timeout)
        return 0, ""

    def down(self) -> tuple[bool, str]:
        return True, ""


class LoopClient:
    def chat(
        self, _messages: list[dict], max_tokens: int, temperature: float = 0.2
    ) -> tuple[str, Usage, None]:
        return "COMMAND:\ntrue", Usage(), None


class NoCallsClient:
    def chat(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("model was called")


def _write_task_json(
    base: Path, name: str, tier: int, extra: dict[str, object] | None = None
) -> None:
    raw: dict[str, object] = {
        "category": "web",
        "tier": tier,
        "statement": "Find the flag",
        "stages": [{"name": "one", "flag_file": "/flag", "service": "target"}],
    }
    if extra:
        raw.update(extra)
    task_dir = base / name
    task_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text(json.dumps(raw))


class WallClockTests(unittest.TestCase):
    def test_tier_defaults(self) -> None:
        self.assertEqual(
            {tier: wall_clock_default(tier) for tier in (1, 2, 3, 4, 5)},
            {1: 600, 2: 900, 3: 1500, 4: 1800, 5: 1800},
        )

    def test_load_task_defaults_by_tier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            for tier, expected in ((1, 600), (2, 900), (3, 1500), (4, 1800), (5, 1800)):
                name = f"sample-t{tier}"
                _write_task_json(base, name, tier)
                with (
                    self.subTest(tier=tier),
                    patch("rangebench.env.TASKS_DIR", base),
                ):
                    self.assertEqual(load_task(name).wall_clock, expected)

    def test_load_task_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_task_json(base, "sample", 1, {"wall_clock": 42})
            with patch("rangebench.env.TASKS_DIR", base):
                self.assertEqual(load_task("sample").wall_clock, 42)

    def test_load_task_rejects_non_positive_wall_clock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_task_json(base, "sample", 1, {"wall_clock": 0})
            with patch("rangebench.env.TASKS_DIR", base):
                with self.assertRaisesRegex(EnvError, "wall_clock"):
                    load_task("sample")

    def test_real_tasks_get_tier_defaults(self) -> None:
        self.assertEqual(load_task("jwt-none").wall_clock, 600)

    def test_expired_cap_ends_attempt_before_any_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch("rangebench.runner.TaskEnv", FakeEnv):
            task = Task(
                "sample",
                Path(tmp),
                "web",
                1,
                "Find the flag",
                stages=[Stage("one", "/flag", "target")],
                wall_clock=0,
            )
            result = run_attempt(NoCallsClient(), task, 1, "rb-test", Path(tmp), verbose=False)
            records = [
                json.loads(line)
                for line in (Path(tmp) / "sample-t1.jsonl").read_text().splitlines()
            ]
        self.assertEqual(result.end_reason, "wall_clock_exceeded")
        self.assertTrue(result.wall_clock_exceeded)
        self.assertEqual(result.wall_clock_seconds, 0)
        self.assertEqual(result.commands, 0)
        budget = [r for r in records if r["kind"] == "budget"]
        self.assertEqual(len(budget), 1)
        self.assertEqual(budget[0]["reason"], "wall_clock_exceeded")
        end = next(r for r in records if r["kind"] == "end")
        self.assertEqual(end["wall_clock_seconds"], 0)
        self.assertTrue(end["wall_clock_exceeded"])

    def test_in_flight_command_finishes_under_its_own_timeout(self) -> None:
        now = [1000.0]

        def fake_time() -> float:
            return now[0]

        envs: list[FakeEnv] = []
        orig_init = FakeEnv.__init__
        base_exec = FakeEnv.exec

        def tracking_init(
            self: FakeEnv, task: Task, project: str, attacker_image: str = ATTACKER_IMAGE
        ) -> None:
            orig_init(self, task, project, attacker_image)
            envs.append(self)

        def advancing_exec(self: FakeEnv, cmd: str, **kwargs: object) -> tuple[int, str]:
            now[0] += 120.0  # the command runs past the 60s wall clock
            return base_exec(self, cmd, **kwargs)

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("rangebench.runner.TaskEnv", FakeEnv),
            patch.object(FakeEnv, "__init__", tracking_init),
            patch.object(FakeEnv, "exec", advancing_exec),
            patch("rangebench.runner.time.time", side_effect=fake_time),
        ):
            task = Task(
                "sample",
                Path(tmp),
                "web",
                1,
                "Find the flag",
                stages=[Stage("one", "/flag", "target")],
                cmd_timeout=120,
                wall_clock=60,
            )
            result = run_attempt(LoopClient(), task, 1, "rb-test", Path(tmp), verbose=False)
        self.assertEqual(result.end_reason, "wall_clock_exceeded")
        self.assertTrue(result.wall_clock_exceeded)
        self.assertEqual(result.wall_clock_seconds, 60)
        # The in-flight command was not cut short: it ran once, to completion,
        # under its own per-command timeout, and the trip fired between turns.
        self.assertEqual(result.commands, 1)
        self.assertEqual(envs[-1].seen_timeouts, [120])

    def test_env_setup_failure_retains_wall_clock_cap(self) -> None:
        class FailingEnv(FakeEnv):
            def up(self) -> None:
                raise EnvError("boom")

        with tempfile.TemporaryDirectory() as tmp, patch("rangebench.runner.TaskEnv", FailingEnv):
            task = Task(
                "sample",
                Path(tmp),
                "web",
                1,
                "Find the flag",
                stages=[Stage("one", "/flag", "target")],
                wall_clock=42,
            )
            result = run_attempt(NoCallsClient(), task, 1, "rb-test", Path(tmp), verbose=False)
        self.assertTrue(result.end_reason.startswith("env:"))
        self.assertEqual(result.wall_clock_seconds, 42)
        self.assertFalse(result.wall_clock_exceeded)

    def test_final_finite_turn_past_deadline_reports_wall_clock(self) -> None:
        now = [1000.0]

        def fake_time() -> float:
            return now[0]

        base_exec = FakeEnv.exec

        def advancing_exec(self: FakeEnv, cmd: str, **kwargs: object) -> tuple[int, str]:
            now[0] += 120.0  # the only turn runs past the 60s wall clock
            return base_exec(self, cmd, **kwargs)

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("rangebench.runner.TaskEnv", FakeEnv),
            patch.object(FakeEnv, "exec", advancing_exec),
            patch("rangebench.runner.time.time", side_effect=fake_time),
        ):
            task = Task(
                "sample",
                Path(tmp),
                "web",
                1,
                "Find the flag",
                stages=[Stage("one", "/flag", "target")],
                turns=1,
                cmd_timeout=120,
                wall_clock=60,
            )
            result = run_attempt(LoopClient(), task, 1, "rb-test", Path(tmp), verbose=False)
        self.assertEqual(result.end_reason, "wall_clock_exceeded")
        self.assertTrue(result.wall_clock_exceeded)
        self.assertEqual(result.wall_clock_seconds, 60)

    def test_final_turn_past_both_deadlines_keeps_infra_precedence(self) -> None:
        now = [1000.0]

        def fake_time() -> float:
            return now[0]

        base_exec = FakeEnv.exec

        def advancing_exec(self: FakeEnv, cmd: str, **kwargs: object) -> tuple[int, str]:
            now[0] += 12000.0  # past the 60s wall clock and the 180min infra guard
            return base_exec(self, cmd, **kwargs)

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("rangebench.runner.TaskEnv", FakeEnv),
            patch.object(FakeEnv, "exec", advancing_exec),
            patch("rangebench.runner.time.time", side_effect=fake_time),
        ):
            task = Task(
                "sample",
                Path(tmp),
                "web",
                1,
                "Find the flag",
                stages=[Stage("one", "/flag", "target")],
                turns=1,
                cmd_timeout=120,
                wall_clock=60,
            )
            result = run_attempt(LoopClient(), task, 1, "rb-test", Path(tmp), verbose=False)
        # Mirrors the turn-start precedence: the non-scoring infra guard wins.
        self.assertEqual(result.end_reason, "infra timeout")
        self.assertFalse(result.wall_clock_exceeded)

    def test_wall_clock_trip_stays_scored(self) -> None:
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
        result = AttemptResult(task.id, 1, end_reason="wall_clock_exceeded")
        result.wall_clock_seconds = 600
        result.wall_clock_exceeded = True
        result.service_image_ids = {"api": "sha256:" + "a" * 64}
        result.service_image_fingerprints = {"api": "sha256:" + "b" * 64}
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("rangebench.cli.RESULTS", Path(tmp)),
                patch("rangebench.cli.load_task", return_value=task),
                patch("rangebench.cli.ChatClient"),
                patch("rangebench.cli.run_attempt", return_value=result),
                patch("rangebench.cli._get_attacker_digest", return_value="sha256:test"),
            ):
                cmd_run(args)  # must not exit nonzero: the trip is scored, not invalid
            saved = json.loads((Path(tmp) / "latest.json").read_text())["tasks"][0]
        self.assertTrue(saved["scored"])
        self.assertEqual(saved["end_reason"], "wall_clock_exceeded")
        self.assertEqual(saved["wall_clock_seconds"], 600)
        self.assertTrue(saved["wall_clock_exceeded"])
        # Existing summary fields are unchanged.
        self.assertEqual(saved["wall_s"], result.wall_s)
        self.assertEqual(saved["turns_used"], result.turns_used)

    def test_scale_multiplies_resolved_cap(self) -> None:
        for scale, expected in ((2.0, 60), (1.0, 30), (0.001, 1)):
            with (
                self.subTest(scale=scale),
                tempfile.TemporaryDirectory() as tmp,
                patch("rangebench.runner.TaskEnv", FakeEnv),
            ):
                task = Task(
                    "sample",
                    Path(tmp),
                    "web",
                    1,
                    "Find the flag",
                    stages=[Stage("one", "/flag", "target")],
                    turns=1,
                    wall_clock=30,
                )
                result = run_attempt(
                    LoopClient(),
                    task,
                    1,
                    "rb-test",
                    Path(tmp),
                    verbose=False,
                    wall_clock_scale=scale,
                )
                records = [
                    json.loads(line)
                    for line in (Path(tmp) / "sample-t1.jsonl").read_text().splitlines()
                ]
                self.assertEqual(result.wall_clock_seconds, expected)
                end = next(r for r in records if r["kind"] == "end")
                self.assertEqual(end["wall_clock_seconds"], expected)

    def _run_args(self, **overrides: object) -> argparse.Namespace:
        args = argparse.Namespace(
            trials=1,
            ctx_window=128000,
            reserve=12000,
            keep_tail=12,
            threshold=0.82,
            tasks=["sample"],
            base_url="http://localhost:8000/v1",
            provider="openai",
            model="test",
            compact="deterministic",
            keep=False,
            wall_clock_scale=1.0,
        )
        vars(args).update(overrides)
        return args

    def test_scale_validation(self) -> None:
        for scale in (0.0, -1.0, 100.5):
            with self.subTest(scale=scale), self.assertRaisesRegex(SystemExit, "wall-clock-scale"):
                cmd_run(self._run_args(wall_clock_scale=scale))

    def test_scale_recorded_in_doc_and_manifest(self) -> None:
        task = Task(
            "sample", Path("/tmp"), "web", 1, "Find the flag", stages=[Stage("one", "/flag", "web")]
        )
        result = AttemptResult(task.id, 1, end_reason="solved")
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("rangebench.cli.RESULTS", Path(tmp)),
                patch("rangebench.cli.load_task", return_value=task),
                patch("rangebench.cli.ChatClient"),
                patch("rangebench.cli.run_attempt", return_value=result) as attempt,
                patch("rangebench.cli._get_attacker_digest", return_value="sha256:test"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                cmd_run(self._run_args(wall_clock_scale=2.5))
            saved = json.loads((Path(tmp) / "latest.json").read_text())
            manifest = json.loads(next(Path(tmp).glob("*/manifest.json")).read_text())
        self.assertEqual(attempt.call_args.kwargs["wall_clock_scale"], 2.5)
        self.assertEqual(saved["wall_clock_scale"], 2.5)
        self.assertEqual(manifest["wall_clock_scale"], 2.5)


REFERENCE = {
    "reference_tps": 100.0,
    "tool_share_default": 0.55,
    "tool_share_by_tier": {"1": 0.5, "3": 0.6},
}


class WallClockReferenceTests(unittest.TestCase):
    def test_scale_formula(self) -> None:
        # tier 1: 0.5 + 0.5 * 100/50 = 1.5; tier 3: 0.6 + 0.4 * 100/200 = 0.8
        self.assertAlmostEqual(wall_clock_scale_for(REFERENCE, 50.0, 1), 1.5)
        self.assertAlmostEqual(wall_clock_scale_for(REFERENCE, 200.0, 3), 0.8)
        self.assertAlmostEqual(wall_clock_scale_for(REFERENCE, 100.0, 3), 1.0)

    def test_default_share_when_tier_missing(self) -> None:
        # tier 5 is absent: 0.55 + 0.45 * 100/25 = 2.35
        self.assertAlmostEqual(wall_clock_scale_for(REFERENCE, 25.0, 5), 2.35)
        no_tiers = {"reference_tps": 100.0, "tool_share_default": 0.25, "tool_share_by_tier": {}}
        self.assertAlmostEqual(wall_clock_scale_for(no_tiers, 50.0, 1), 1.75)

    def test_clamped_to_max(self) -> None:
        self.assertEqual(wall_clock_scale_for(REFERENCE, 0.001, 1), 100.0)

    def test_non_positive_tps_raises(self) -> None:
        for model_tps in (0.0, -5.0):
            with self.subTest(model_tps=model_tps), self.assertRaisesRegex(ValueError, "model"):
                wall_clock_scale_for(REFERENCE, model_tps, 1)
        for reference_tps in (0.0, -1.0):
            bad = dict(REFERENCE, reference_tps=reference_tps)
            with (
                self.subTest(reference_tps=reference_tps),
                self.assertRaisesRegex(ValueError, "reference_tps"),
            ):
                wall_clock_scale_for(bad, 50.0, 1)

    def _args(self, **overrides: object) -> argparse.Namespace:
        args = argparse.Namespace(
            trials=1,
            ctx_window=128000,
            reserve=12000,
            keep_tail=12,
            threshold=0.82,
            tasks=["sample"],
            base_url="http://localhost:8000/v1",
            provider="openai",
            model="test",
            compact="deterministic",
            keep=False,
            wall_clock_scale=None,
            wall_clock_reference=None,
        )
        vars(args).update(overrides)
        return args

    def _write_reference(self, tmp: str, content: object) -> str:
        path = Path(tmp) / "reference.json"
        path.write_text(content if isinstance(content, str) else json.dumps(content))
        return str(path)

    def test_scale_and_reference_are_mutually_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_reference(tmp, REFERENCE)
            with (
                patch("rangebench.cli.ChatClient", NoCallsClient),
                self.assertRaisesRegex(SystemExit, "mutually exclusive"),
            ):
                cmd_run(self._args(wall_clock_scale=2.0, wall_clock_reference=path))

    def test_parser_accepts_flags_and_rejects_both(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_reference(tmp, REFERENCE)
            argv = ["rangebench", "run", "--model", "m", "--wall-clock-scale", "2"]
            argv += ["--wall-clock-reference", path]
            with (
                patch("sys.argv", argv),
                patch("rangebench.cli.ChatClient", NoCallsClient),
                self.assertRaisesRegex(SystemExit, "mutually exclusive"),
            ):
                main()

    def test_reference_file_validation(self) -> None:
        cases: list[tuple[str, object, str]] = [
            ("missing", None, "not found"),
            ("bad json", "{not json", "JSON"),
            ("not object", [1, 2], "JSON object"),
            ("zero tps", dict(REFERENCE, reference_tps=0), "reference_tps"),
            ("negative tps", dict(REFERENCE, reference_tps=-3.0), "reference_tps"),
            ("no default", {"reference_tps": 50.0}, "tool_share_default"),
            ("bad default", dict(REFERENCE, tool_share_default=1.5), "tool_share_default"),
            ("bad tier", dict(REFERENCE, tool_share_by_tier={"2": "x"}), "tool_share_by_tier"),
        ]
        for name, content, message in cases:
            with (
                self.subTest(name),
                tempfile.TemporaryDirectory() as tmp,
                patch("rangebench.cli.ChatClient", NoCallsClient),
                self.assertRaisesRegex(SystemExit, message),
            ):
                path = (
                    str(Path(tmp) / "absent.json")
                    if content is None
                    else self._write_reference(tmp, content)
                )
                cmd_run(self._args(wall_clock_reference=path))

    def test_reference_run_scales_cap_per_task(self) -> None:
        calls: list[int] = []

        class CountingClient:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                pass

            def chat(
                self, _messages: list[dict], max_tokens: int, temperature: float = 0.2
            ) -> tuple[str, Usage, None]:
                calls.append(max_tokens)
                return "COMMAND:\ntrue", Usage(completion_tokens=50), None

        clock = iter([10.0, 11.0])  # the startup probe takes 1s: 50 tok/s
        with tempfile.TemporaryDirectory() as tmp:
            task = Task(
                "sample",
                Path(tmp),
                "web",
                1,
                "Find the flag",
                stages=[Stage("one", "/flag", "target")],
                turns=1,
                wall_clock=30,
            )
            path = self._write_reference(tmp, REFERENCE)
            with (
                patch("rangebench.cli.RESULTS", Path(tmp)),
                patch("rangebench.cli.load_task", return_value=task),
                patch("rangebench.cli.ChatClient", CountingClient),
                patch("rangebench.runner.TaskEnv", FakeEnv),
                patch("rangebench.cli.time.perf_counter", side_effect=lambda: next(clock)),
                patch("rangebench.cli._get_attacker_digest", return_value="sha256:test"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                cmd_run(self._args(wall_clock_reference=path))
            saved = json.loads((Path(tmp) / "latest.json").read_text())
            manifest = json.loads(next(Path(tmp).glob("*/manifest.json")).read_text())
            records = [
                json.loads(line)
                for line in next(Path(tmp).glob("*/sample-t1.jsonl")).read_text().splitlines()
            ]
        # One startup probe call plus the single agent turn.
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], 32768)
        # tier 1 share 0.5: 0.5 + 0.5 * 100/50 = 1.5, so the 30s cap becomes 45s.
        self.assertEqual(saved["tasks"][0]["wall_clock_scale"], 1.5)
        self.assertEqual(saved["model_tps"], 50.0)
        self.assertEqual(saved["wall_clock_reference"], path)
        self.assertEqual(saved["wall_clock_scale_mode"], "reference")
        self.assertEqual(manifest["wall_clock_reference"], path)
        end = next(r for r in records if r["kind"] == "end")
        self.assertEqual(end["wall_clock_seconds"], 45)
        config = [r for r in records if r["kind"] == "budget-config"]
        self.assertEqual(len(config), 1)
        self.assertEqual(config[0]["tier"], 1)
        self.assertEqual(config[0]["wall_clock_scale"], 1.5)
        self.assertEqual(config[0]["model_tps"], 50.0)

    def test_failed_probe_exits_before_run(self) -> None:
        for usage, err in (
            (Usage(completion_tokens=50), "HTTP 500"),
            (Usage(), None),
            (Usage(completion_tokens=50, requests=3), None),
        ):

            class FailingClient:
                def __init__(self, *_args: object, **_kwargs: object) -> None:
                    pass

                def chat(self, *_args: object, **_kwargs: object) -> tuple[str, Usage, str | None]:
                    return "", usage, err

            with (
                self.subTest(err=err),
                tempfile.TemporaryDirectory() as tmp,
                patch("rangebench.cli.RESULTS", Path(tmp)),
                patch("rangebench.cli.ChatClient", FailingClient),
                patch("rangebench.cli.run_attempt") as attempt,
                patch("rangebench.cli._get_attacker_digest", return_value="sha256:test"),
            ):
                path = self._write_reference(tmp, REFERENCE)
                with self.assertRaisesRegex(SystemExit, "probe failed"):
                    cmd_run(self._args(wall_clock_reference=path))
                manifest = json.loads(next(Path(tmp).glob("*/manifest.json")).read_text())
                attempt.assert_not_called()
                self.assertEqual(manifest["status"], "probe_failed")

    def test_manual_mode_recorded(self) -> None:
        task = Task(
            "sample", Path("/tmp"), "web", 1, "Find the flag", stages=[Stage("one", "/flag", "web")]
        )
        result = AttemptResult(task.id, 1, end_reason="solved")
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("rangebench.cli.RESULTS", Path(tmp)),
                patch("rangebench.cli.load_task", return_value=task),
                patch("rangebench.cli.ChatClient") as client_cls,
                patch("rangebench.cli.run_attempt", return_value=result),
                patch("rangebench.cli._get_attacker_digest", return_value="sha256:test"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                cmd_run(self._args(wall_clock_scale=2.0))
            saved = json.loads((Path(tmp) / "latest.json").read_text())
        client_cls.return_value.chat.assert_not_called()  # no startup probe in manual mode
        self.assertEqual(saved["wall_clock_scale_mode"], "manual")
        self.assertEqual(saved["wall_clock_scale"], 2.0)
        self.assertEqual(saved["tasks"][0]["wall_clock_scale"], 2.0)
        self.assertNotIn("model_tps", saved)


class ProbeTests(unittest.TestCase):
    def test_samples_report_throughput(self) -> None:
        calls: list[int] = []

        class FixedClient:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                pass

            def chat(self, _messages: list[dict], max_tokens: int) -> tuple[str, Usage, None]:
                calls.append(max_tokens)
                return "COMMAND:\necho ok", Usage(completion_tokens=50, reasoning_tokens=20), None

        args = argparse.Namespace(
            model="test", base_url=None, provider="openai", reasoning_effort=None, samples=3
        )
        clock = iter([0.0, 2.0, 10.0, 11.0, 20.0, 22.5])
        with (
            patch("rangebench.cli.ChatClient", FixedClient),
            patch("rangebench.cli.time.perf_counter", side_effect=lambda: next(clock)),
            contextlib.redirect_stdout(io.StringIO()) as out,
        ):
            cmd_probe(args)
        text = out.getvalue()
        self.assertEqual(len(calls), 3)
        self.assertIn("sample 2: latency 1.00s tokens 50 (reasoning 20) 50.0 tok/s", text)
        self.assertIn("mean latency 1.83s median latency 2.00s", text)
        self.assertIn("mean tokens/sec 31.7 n=3/3", text)

    def test_json_summary(self) -> None:
        replies = iter([(Usage(completion_tokens=50), None), (Usage(), "HTTP 500")] * 2)

        class MixedClient:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                pass

            def chat(self, _messages: list[dict], max_tokens: int) -> tuple[str, Usage, str | None]:
                usage, err = next(replies)
                return "COMMAND:\necho ok", usage, err

        args = argparse.Namespace(
            model="test", base_url=None, provider="openai", reasoning_effort=None, samples=3
        )
        args.json = True
        # samples: 2.0s ok, 1.0s error, 3.0s ok
        clock = iter([0.0, 2.0, 10.0, 11.0, 20.0, 23.0])
        with (
            patch("rangebench.cli.ChatClient", MixedClient),
            patch("rangebench.cli.time.perf_counter", side_effect=lambda: next(clock)),
            contextlib.redirect_stdout(io.StringIO()) as out,
        ):
            cmd_probe(args)
        self.assertEqual(
            json.loads(out.getvalue()),
            {
                "samples": 3,
                "ok": 2,
                "mean_latency_s": 2.5,
                "median_latency_s": 2.5,
                "mean_tokens_per_sec": (25.0 + 50.0 / 3.0) / 2,
            },
        )
        self.assertEqual(len(out.getvalue().strip().splitlines()), 1)

    def test_json_all_failed(self) -> None:
        class FailingClient:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                pass

            def chat(self, _messages: list[dict], max_tokens: int) -> tuple[str, Usage, str]:
                return "", Usage(), "HTTP 500"

        args = argparse.Namespace(
            model="test", base_url=None, provider="openai", reasoning_effort=None, samples=2
        )
        args.json = True
        with (
            patch("rangebench.cli.ChatClient", FailingClient),
            contextlib.redirect_stdout(io.StringIO()) as out,
        ):
            cmd_probe(args)
        self.assertEqual(json.loads(out.getvalue()), {"samples": 2, "ok": 0})


if __name__ == "__main__":
    unittest.main()
