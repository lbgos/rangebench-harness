"""Per-task identity, static content gates, and the identities CLI output."""

import argparse
import contextlib
import importlib.util
import io
import json
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from rangebench.cli import cmd_identities
from rangebench.env import TASKS_DIR
from rangebench.identity import all_task_identities, task_identity

_GATE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check-gates.py"
_gate_spec = importlib.util.spec_from_file_location("check_gates", _GATE_PATH)
check_gates = importlib.util.module_from_spec(_gate_spec)
assert _gate_spec and _gate_spec.loader
_gate_spec.loader.exec_module(check_gates)

HEX16 = re.compile(r"[0-9a-f]{16}")


def make_task(
    tasks: Path,
    name: str,
    *,
    canary: str = "beef-cafe-test",
    statement: str = "Recover the secret.",
) -> Path:
    """A minimal but complete synthetic task directory."""
    d = tasks / name
    d.mkdir(parents=True)
    (d / "task.json").write_text(
        json.dumps(
            {
                "id": name,
                "category": "web",
                "tier": 1,
                "canary": canary,
                "statement": statement,
                "stages": [{"name": "flag", "flag_file": "/flag", "service": "target"}],
            }
        )
    )
    (d / "docker-compose.yml").write_text("services:\n  target: {}\n")
    (d / "seed.sh").write_text('#!/bin/sh\nFLAG="flag{$(head -c16 /dev/urandom)}"\n')
    app = d / "app"
    app.mkdir()
    (app / "main.py").write_text("print('service')\n")
    solution = d / "solution"
    solution.mkdir()
    (solution / "solve.sh").write_text("#!/bin/sh\ncat /flag\n")
    return d


class IdentityTests(unittest.TestCase):
    def test_identity_is_deterministic_and_location_independent(self) -> None:
        with TemporaryDirectory() as a, TemporaryDirectory() as b:
            first = make_task(Path(a), "sample")
            audit_parent = Path(b) / ".audit"
            audit_parent.mkdir()
            second = make_task(audit_parent, "sample")
            self.assertEqual(task_identity(first), task_identity(first))
            self.assertEqual(task_identity(first), task_identity(second))
            self.assertEqual(check_gates.check_task(second), [])

    def test_identity_changes_when_any_closure_file_changes(self) -> None:
        # Every shipped file is in the closure, including execution files a
        # whole-run task-set hash cannot attribute to one task.
        with TemporaryDirectory() as tmp:
            d = make_task(Path(tmp), "sample")
            original_contents = {
                "task.json": (d / "task.json").read_text(),
                "docker-compose.yml": (d / "docker-compose.yml").read_text(),
                "seed.sh": (d / "seed.sh").read_text(),
                "app/main.py": (d / "app/main.py").read_text(),
                "solution/solve.sh": (d / "solution/solve.sh").read_text(),
            }
            original = task_identity(d)
            changed_contents = {
                "task.json": '{"id": "sample", "tier": 2}',
                "docker-compose.yml": "services:\n  target: {image: other}\n",
                "seed.sh": "#!/bin/sh\necho changed\n",
                "app/main.py": "print('changed')\n",
                "solution/solve.sh": "#!/bin/sh\ncat /other\n",
            }
            for filename, content in changed_contents.items():
                with self.subTest(file=filename):
                    (d / filename).write_text(content)
                    self.assertNotEqual(task_identity(d), original)
                    (d / filename).write_text(original_contents[filename])

    def test_new_closure_file_changes_identity(self) -> None:
        with TemporaryDirectory() as tmp:
            d = make_task(Path(tmp), "sample")
            original = task_identity(d)
            (d / "gen.py").write_text("print('generate')\n")
            self.assertNotEqual(task_identity(d), original)

    def test_identity_ignores_caches_and_audit_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            d = make_task(Path(tmp), "sample")
            original = task_identity(d)
            cache = d / "app" / "__pycache__"
            cache.mkdir()
            (cache / "main.cpython-311.pyc").write_bytes(b"\x00stale")
            (d / "root.pyc").write_bytes(b"\x00stale")
            audit = d / ".audit"
            audit.mkdir()
            (audit / "allow-static-flags.txt").write_text("flag{decoy}\n")
            self.assertEqual(task_identity(d), original)


class GateTests(unittest.TestCase):
    def test_gate_passes_on_real_tasks(self) -> None:
        real = [p for p in TASKS_DIR.iterdir() if (p / "task.json").exists()]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(check_gates.main(), 0)
        self.assertIn(f"checked {len(real)} tasks", out.getvalue())
        self.assertNotIn("GATE-FAIL", out.getvalue())

    def test_gate_exit_nonzero_reports_findings(self) -> None:
        with TemporaryDirectory() as tmp:
            tasks = Path(tmp)
            make_task(tasks, "clean")
            planted = make_task(tasks, "planted")
            (planted / "seed.sh").write_text("echo flag{e2b7-static-plant}\n")
            with (
                patch.object(check_gates, "TASKS_DIR", tasks),
                contextlib.redirect_stdout(io.StringIO()) as out,
            ):
                self.assertEqual(check_gates.main(), 1)
        self.assertIn("GATE-FAIL planted: static flag at seed.sh:1", out.getvalue())

    def test_gate_catches_planted_literal_flag(self) -> None:
        with TemporaryDirectory() as tmp:
            d = make_task(Path(tmp), "sample")
            self.assertEqual(check_gates.check_task(d), [])
            (d / "seed.sh").write_text("printf 'DEPLOY_KEY=flag{ab12-static-literal}\n'\n")
            (d / "app/main.py").write_text("BANNER = 'flag{cd34-app-literal}'\n")
            findings = check_gates.check_task(d)
        self.assertEqual(len(findings), 2)
        self.assertIn("static flag at seed.sh:1", findings[1])
        self.assertIn("static flag at app/main.py:1", findings[0])

    def test_gate_understands_generated_flags(self) -> None:
        with TemporaryDirectory() as tmp:
            d = make_task(Path(tmp), "sample")
            compose = "services:\n  target:\n    command: sh -c 'FLAG=\"flag{$$(head -c16)}\"'\n"
            (d / "docker-compose.yml").write_text(compose)
            self.assertEqual(check_gates.check_task(d), [])

    def test_gate_allowlist_exempts_named_decoy_only(self) -> None:
        with TemporaryDirectory() as tmp:
            d = make_task(Path(tmp), "sample")
            (d / "seed.sh").write_text(
                "printf 'flag{0000-old-rotated-do-not-use-0000} "
                "flag{9999-not-allowlisted}' > .env\n"
            )
            audit = d / ".audit"
            audit.mkdir()
            (audit / "allow-static-flags.txt").write_text(
                "# comment\nflag{0000-old-rotated-do-not-use-0000}\n"
            )
            findings = check_gates.check_task(d)
        self.assertEqual(len(findings), 1)
        self.assertIn("static flag at seed.sh:1", findings[0])

    def test_gate_flags_hardcoded_flag_in_statement(self) -> None:
        with TemporaryDirectory() as tmp:
            d = make_task(Path(tmp), "sample")
            (d / "task.json").write_text(
                json.dumps(
                    {
                        "id": "sample",
                        "category": "web",
                        "tier": 1,
                        "canary": "beef-cafe-test",
                        "statement": "The flag is flag{deadbeef-statement}.",
                        "stages": [{"name": "flag", "flag_file": "/flag", "service": "target"}],
                    }
                )
            )
            findings = check_gates.check_task(d)
        self.assertEqual(len(findings), 1)
        self.assertIn("static flag at task.json", findings[0])

    def test_gate_checks_metadata(self) -> None:
        cases = {
            "canary in statement": {"canary": "leaked-canary", "statement": "leaked-canary"},
            "missing canary": {"canary": None},
            "CVE in statement": {"statement": "abuse CVE-2021-44228"},
            "tier not int": {"tier": "1"},
            "missing category": {"category": None},
            "missing statement": {"statement": None},
            "non-string statement": {"statement": 42},
            "no stages": {"stages": []},
            "stage missing service": {"stages": [{"name": "flag", "flag_file": "/flag"}]},
            "stage empty service": {
                "stages": [{"name": "flag", "flag_file": "/flag", "service": ""}]
            },
        }
        for label, edits in cases.items():
            with self.subTest(case=label), TemporaryDirectory() as tmp:
                d = make_task(Path(tmp), "sample")
                raw = json.loads((d / "task.json").read_text())
                for key, value in edits.items():
                    if value is None:
                        del raw[key]
                    else:
                        raw[key] = value
                (d / "task.json").write_text(json.dumps(raw))
                findings = check_gates.check_task(d)
            self.assertTrue(findings, "expected a finding")
            self.assertNotIn("static flag", "".join(findings))

    def test_gate_id_and_canary_uniqueness(self) -> None:
        with TemporaryDirectory() as tmp:
            tasks = Path(tmp)
            make_task(tasks, "alpha", canary="1111-alpha-test")
            make_task(tasks, "beta", canary="1111-alpha-test")
            make_task(tasks, "gamma", canary="3333-gamma-test")
            gamma = tasks / "gamma" / "task.json"
            gamma.write_text(gamma.read_text().replace('"id": "gamma"', '"id": "alpha"'))
            seen_ids: dict[str, str] = {}
            seen_canaries: dict[str, str] = {}
            findings: list[str] = []
            for d in sorted(tasks.iterdir()):
                findings.extend(check_gates.check_task(d, seen_ids, seen_canaries))
        self.assertEqual(len(findings), 3)
        self.assertIn("beta: duplicate canary (also alpha)", findings[0])
        self.assertIn("gamma: id mismatch", findings[1])
        self.assertIn("gamma: duplicate task id (also alpha)", findings[2])

    def test_gate_reports_unreadable_task_json(self) -> None:
        with TemporaryDirectory() as tmp:
            d = make_task(Path(tmp), "sample")
            (d / "task.json").write_text("{not json")
            findings = check_gates.check_task(d)
        self.assertEqual(len(findings), 1)
        self.assertIn("unreadable task.json", findings[0])


class IdentitiesCommandTests(unittest.TestCase):
    def test_cmd_identities_prints_sorted_16hex_hashes(self) -> None:
        with TemporaryDirectory() as tmp:
            tasks = Path(tmp)
            make_task(tasks, "zulu")
            make_task(tasks, "alpha")
            (tasks / "not-a-task").mkdir()
            with (
                patch("rangebench.cli.TASKS_DIR", tasks),
                contextlib.redirect_stdout(io.StringIO()) as out,
            ):
                cmd_identities(argparse.Namespace())
            lines = out.getvalue().splitlines()
        self.assertEqual(lines[0], f"{'task':24} identity")
        rows = [line.split() for line in lines[1:]]
        self.assertEqual([row[0] for row in rows], ["alpha", "zulu"])
        for _, ident in rows:
            self.assertRegex(ident, HEX16)

    def test_cli_identities_lists_all_real_tasks(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            cmd_identities(argparse.Namespace())
        rows = [line.split() for line in out.getvalue().splitlines()[1:]]
        expected = [t.name for t in sorted(TASKS_DIR.iterdir()) if (t / "task.json").exists()]
        self.assertEqual([row[0] for row in rows], expected)
        self.assertEqual(all_task_identities(TASKS_DIR), {row[0]: row[1] for row in rows})


if __name__ == "__main__":
    unittest.main()
