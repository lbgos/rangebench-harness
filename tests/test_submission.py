import argparse
import contextlib
import copy
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rangebench.agent import Usage
from rangebench.cli import cmd_run, main
from rangebench.env import Stage, Task
from rangebench.runner import AttemptResult
from rangebench.submission import build_submission, public_price_lookup, validate_submission


def run_doc():
    def task(trial, solved):
        return {
            "task": "sample",
            "category": "web",
            "tier": 1,
            "trial": trial,
            "solved": solved,
            "scored": True,
            "fail_class": "solved" if solved else "normal",
            "wall_s": 2.0,
            "turns_used": 3,
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_tokens": 10,
            "cache_write_tokens": 0,
            "api_calls": 1,
            "api_requests": 1,
            "input_reported_calls": 1,
            "output_reported_calls": 1,
            "cache_read_reported_calls": 1,
            "cache_write_reported_calls": 1,
        }

    return {
        "id": "run-1",
        "model": "example/model",
        "provider": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "started": "2026-09-28T01:00:00+00:00",
        "finished": "2026-09-28T02:00:00+00:00",
        "selected_tasks": ["sample"],
        "task_catalog": [{"task": "sample", "category": "web", "tier": 1}],
        "task_identities": {"sample": "a" * 64},
        "trials": 3,
        "ctx_window": 1000,
        "reserve": 100,
        "keep_tail": 3,
        "threshold": 0.8,
        "compact": "llm",
        "wall_clock_scale_mode": "manual",
        "wall_clock_scale": 1.0,
        "tasks": [task(1, True), task(2, False), task(3, False)],
        "infra_attempts": [
            {
                "task": "sample",
                "trial": 2,
                "retry": 0,
                "terminal": False,
                "fail_class": "provider_error",
                "wall_s": 1.0,
                "usage": {
                    "input_tokens": 50,
                    "output_tokens": 5,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "api_calls": 1,
                    "input_reported_calls": 1,
                    "output_reported_calls": 1,
                    "cache_read_reported_calls": 1,
                    "cache_write_reported_calls": 1,
                },
            }
        ],
        "probe_attempt": {"task": "sample", "wall_s": 1.0},
        "probe_usage": {
            "input_tokens": 25,
            "output_tokens": 5,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "api_calls": 1,
            "input_reported_calls": 1,
            "output_reported_calls": 1,
            "cache_read_reported_calls": 1,
            "cache_write_reported_calls": 1,
        },
    }


class SubmissionTests(unittest.TestCase):
    def test_real_usage_dict_counts_probe_and_retry_calls(self):
        run = run_doc()
        usage = Usage()
        usage.add({"prompt_tokens": 25, "completion_tokens": 5})
        run["probe_usage"] = usage.as_dict()
        run["infra_attempts"][0]["usage"] = usage.as_dict()
        sub = build_submission(run)
        self.assertEqual(sub["summary"]["usage"]["api_calls"], 5)

    def test_automatic_export_then_cli_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            task = Task(
                "sample", root, "web", 1, "Example", stages=[Stage("one", "/flag", "target")]
            )
            args = argparse.Namespace(
                trials=1,
                ctx_window=128000,
                reserve=12000,
                keep_tail=12,
                threshold=0.82,
                tasks=["sample"],
                base_url="http://localhost:8000/v1",
                provider="openai",
                model="example",
                compact="llm",
                keep=False,
                reasoning_effort="high",
                route_name="Local",
                api_key_env="AUTH_HEADER",
            )
            with (
                patch.dict(os.environ, {"AUTH_HEADER": "synthetic-test-credential"}),
                patch("rangebench.cli.RESULTS", root),
                patch("rangebench.cli.load_task", return_value=task),
                patch("rangebench.cli.load_all", return_value=[task]),
                patch("rangebench.cli.ChatClient"),
                patch(
                    "rangebench.cli.run_attempt",
                    return_value=AttemptResult("sample", 1, end_reason="turn budget"),
                ),
                patch("rangebench.cli._get_attacker_digest", return_value="sha256:" + "a" * 64),
            ):
                cmd_run(args)
            path = next(root.glob("*/submission.json"))
            doc = json.loads(path.read_text())
            validate_submission(doc)
            self.assertEqual(doc["policy"]["reasoning_effort"], "high")
            self.assertEqual(doc["summary"]["missing_slots"], 0)
            exported = root / "shared.json"
            with patch(
                "sys.argv",
                ["rangebench", "export", str(root / "latest.json"), "--output", str(exported)],
            ):
                main()
            self.assertEqual(json.loads(exported.read_text()), doc)
            with patch("sys.argv", ["rangebench", "validate-submission", str(exported)]):
                main()
            before = exported.read_bytes()
            with (
                patch(
                    "sys.argv",
                    ["rangebench", "export", str(root / "latest.json"), "--output", str(exported)],
                ),
                self.assertRaises(SystemExit),
            ):
                main()
            self.assertEqual(exported.read_bytes(), before)

    def test_unknown_pricing_rejects_extra_data(self):
        doc = build_submission(run_doc())
        doc["pricing"]["source"] = {"api_key": "do-not-include"}
        with self.assertRaises(ValueError):
            validate_submission(doc)

    def test_counts_retries_probe_and_passk(self):
        sub = build_submission(run_doc())
        self.assertEqual(sub["summary"]["usage"]["input_tokens"], 375)
        self.assertEqual(sub["summary"]["usage"]["output_tokens"], 70)
        self.assertEqual(sub["summary"]["attempts_all"], 5)
        self.assertEqual(sub["summary"]["missing_slots"], 0)
        self.assertAlmostEqual(sub["tasks"][0]["pass_at_1"], 1 / 3)
        self.assertEqual(sub["tasks"][0]["pass_at_3"], 1.0)
        self.assertEqual(sub["model"]["route_name"], "OpenRouter")

    def test_allowlist_and_secret_rejection(self):
        run = run_doc()
        run["base_url"] = "https://user:password@example.com/v1"
        run["tasks"][0]["raw_output"] = "flag{private}"
        sub = build_submission(run)
        self.assertIsNone(sub["model"]["route_name"])
        self.assertNotIn("password", str(sub))
        self.assertNotIn("flag{private}", str(sub))
        with patch.dict(os.environ, {"MODEL_API_KEY": "known-secret-123"}):
            run["display_name"] = "known-secret-123"
            with self.assertRaises(ValueError):
                build_submission(run)
        run["display_name"] = "known-secret-123"
        with self.assertRaises(ValueError):
            build_submission(run, sensitive_values=("known-secret-123",))

    def test_validator_recomputes_and_rejects_nonfinite(self):
        sub = build_submission(run_doc())
        bad = copy.deepcopy(sub)
        bad["summary"]["solved"] = 99
        with self.assertRaises(ValueError):
            validate_submission(bad)
        bad = copy.deepcopy(sub)
        bad["attempts"][0]["fail_class"] = "provider_error"
        with self.assertRaises(ValueError):
            validate_submission(bad)
        run = run_doc()
        run["tasks"][0]["fail_class"] = "provider_error"
        with self.assertRaises(ValueError):
            build_submission(run)
        bad = copy.deepcopy(sub)
        bad["attempts"][0]["wall_s"] = float("nan")
        with self.assertRaises(ValueError):
            validate_submission(bad)
        bad = copy.deepcopy(sub)
        bad["policy"]["endpoint"] = "secret"
        with self.assertRaises(ValueError):
            validate_submission(bad)

    def test_missing_usage_remains_unknown(self):
        run = run_doc()
        del run["probe_usage"]
        sub = build_submission(run)
        self.assertIsNone(sub["summary"]["usage"]["input_tokens"])
        self.assertIsNone(sub["pricing"]["estimated_usd"])

    def test_terminal_retry_deduplicated_and_skip_counted(self):
        run = run_doc()
        run["infra_attempts"].append(
            {
                "task": "sample",
                "trial": 1,
                "retry": 1,
                "terminal": True,
                "fail_class": "solved",
                "wall_s": 2.0,
            }
        )
        sub = build_submission(run)
        self.assertEqual(sub["summary"]["attempts_all"], 5)
        run["tasks"][2].update(scored=False, solved=False, fail_class="skipped")
        sub = build_submission(run)
        self.assertEqual(sub["tasks"][0]["skipped"], 1)
        self.assertEqual(sub["summary"]["attempts_all"], 4)

    def test_price_cost_uses_uncached_input_and_recomputes(self):
        pricing = {
            "status": "estimated",
            "source": "OpenRouter",
            "as_of": "2026-09-28",
            "catalog_model_id": "example/model",
            "usd_per_million": {"input": 1.0, "output": 2.0, "cache_read": 0.5, "cache_write": 1.5},
            "estimated_usd": None,
        }
        sub = build_submission(run_doc(), pricing=pricing)
        self.assertAlmostEqual(sub["pricing"]["estimated_usd"], (345 * 1 + 70 * 2 + 30 * 0.5) / 1e6)
        sub["pricing"]["estimated_usd"] = 0
        with self.assertRaises(ValueError):
            validate_submission(sub)

    def test_ambiguous_catalog_stays_unknown(self):
        with patch("rangebench.submission.urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = (
                b'{"data":[{"id":"x"},{"id":"x"}]}'
            )
            self.assertEqual(public_price_lookup("x", "OpenRouter")["status"], "unknown")


if __name__ == "__main__":
    unittest.main()
