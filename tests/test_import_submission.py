"""The site handoff stages data without silently mixing suites or revisions."""

import copy
import importlib.util
import unittest
from pathlib import Path

from rangebench.submission import build_submission

_spec = importlib.util.spec_from_file_location(
    "import_submission", Path(__file__).resolve().parents[1] / "scripts" / "import-submission.py"
)
assert _spec and _spec.loader
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
stage_import = _module.stage_import


def fixture():
    run = {
        "id": "20260928T120000Z-abcdef",
        "model": "example/model",
        "provider": "openai",
        "started": "2026-09-28T12:00:00+00:00",
        "finished": "2026-09-28T12:10:00+00:00",
        "harness_source_hash": "a" * 16,
        "task_set_hash": "b" * 16,
        "selected_tasks": ["sample"],
        "trials": 1,
        "ctx_window": 128000,
        "reserve": 8000,
        "keep_tail": 4,
        "threshold": 0.8,
        "compact": "llm",
        "wall_clock_scale_mode": "manual",
        "wall_clock_scale": 1.0,
        "tasks": [
            {
                "task": "sample",
                "category": "web",
                "tier": 1,
                "trial": 1,
                "scored": True,
                "solved": True,
                "fail_class": "solved",
                "wall_s": 60,
                "turns_used": 2,
                "input_tokens": 1000,
                "output_tokens": 200,
                "cache_read_tokens": None,
                "cache_write_tokens": None,
                "api_calls": 2,
                "api_requests": 2,
                "usage_reported_calls": 2,
                "input_reported_calls": 2,
                "output_reported_calls": 2,
                "cache_read_reported_calls": 0,
                "cache_write_reported_calls": 0,
            }
        ],
    }
    board = {
        "version": "v2",
        "date": "2026-09-27",
        "source": {
            "harness_source_hash": "a" * 16,
            "task_set_hash": "b" * 16,
        },
        "pricing": {},
        "models": [
            {
                "id": "existing",
                "label": "Existing",
                "pass1": 0.0,
                "tasks": [
                    {
                        "task": "sample",
                        "category": "web",
                        "tier": 1,
                        "scored": 1,
                        "solved": 0,
                        "pass1": 0.0,
                        "pass3": None,
                    }
                ],
                "runs": [],
            }
        ],
    }
    return board, build_submission(run)


class ImportSubmissionTests(unittest.TestCase):
    def test_preserves_existing_rows_and_embeds_source(self):
        board, submission = fixture()
        before = copy.deepcopy(board)
        result = stage_import(board, submission)
        self.assertEqual(board, before)
        self.assertEqual([m for m in result["models"] if m["id"] == "existing"], before["models"])
        row = result["models"][0]
        self.assertEqual(row["submission"], submission)
        self.assertEqual(row["pass1"], 1.0)
        self.assertEqual(row["wall_min_per_attempt"], 1.0)
        self.assertIsNone(row["pass3"])
        self.assertIsNone(row["api_cost"])
        self.assertIsNone(result["pricing"][row["id"]]["prompt"])
        self.assertEqual(row["output_k_per_attempt"], 0.2)

    def test_duplicate_run_rejected(self):
        board, submission = fixture()
        result = stage_import(board, submission)
        with self.assertRaisesRegex(ValueError, "already"):
            stage_import(result, submission)

    def test_unknown_revision_requires_explicit_override(self):
        board, submission = fixture()
        board["source"]["harness_source_hash"] = "c" * 16
        with self.assertRaisesRegex(ValueError, "source revision"):
            stage_import(board, submission)
        self.assertEqual(
            len(stage_import(board, submission, allow_source_change=True)["models"]), 2
        )

    def test_partial_coverage_requires_override_and_fills_missing_cells(self):
        board, submission = fixture()
        board["models"][0]["tasks"].append({"task": "other", "category": "web", "tier": 2})
        with self.assertRaisesRegex(ValueError, "partial coverage"):
            stage_import(board, submission)
        row = stage_import(board, submission, allow_partial=True)["models"][0]
        self.assertEqual([t["task"] for t in row["tasks"]], ["sample", "other"])
        self.assertEqual(row["tasks"][1]["scored"], 0)
        self.assertIsNone(row["tasks"][1]["pass1"])

    def test_task_metadata_mismatch_rejected(self):
        board, submission = fixture()
        board["models"][0]["tasks"][0]["tier"] = 4
        with self.assertRaisesRegex(ValueError, "metadata"):
            stage_import(board, submission)

    def test_extra_task_rejected_even_with_overrides(self):
        board, submission = fixture()
        board["models"][0]["tasks"][0]["task"] = "other"
        with self.assertRaisesRegex(ValueError, "outside"):
            stage_import(board, submission, allow_partial=True, allow_source_change=True)

    def test_modified_score_cannot_be_imported(self):
        board, submission = fixture()
        submission["summary"]["solved"] += 1
        with self.assertRaises(ValueError):
            stage_import(board, submission)


if __name__ == "__main__":
    unittest.main()
