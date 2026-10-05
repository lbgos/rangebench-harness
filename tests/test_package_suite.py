"""Suite kits hold exactly the named tasks and run only those."""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "package_suite", Path(__file__).resolve().parents[1] / "scripts" / "package-suite.py"
)
assert _spec and _spec.loader
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
build = _module.build
PackageError = _module.PackageError


def _task(tasks_dir: Path, tid: str) -> Path:
    d = tasks_dir / tid
    (d / "app").mkdir(parents=True)
    (d / "solution").mkdir()
    spec = {
        "id": tid,
        "category": "web",
        "tier": 1,
        "statement": f"Sample statement for {tid}.",
        "stages": [{"name": "flag", "flag_file": "/flag", "service": "app"}],
    }
    (d / "task.json").write_text(json.dumps(spec))
    (d / "docker-compose.yml").write_text("services:\n  app:\n    build: app\n")
    (d / "app" / "app.py").write_text("print('sample')\n")
    solve = d / "solution" / "solve.sh"
    solve.write_text("#!/bin/sh\necho sample\n")
    solve.chmod(0o755)
    return d


class PackageSuiteTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.tasks = self.tmp / "suite"
        for tid in ("alpha", "beta", "other"):
            _task(self.tasks, tid)
        alpha = self.tasks / "alpha"
        for junk in (".git/config", "results/run.json", "logs/run.log", "app/__pycache__/a.pyc"):
            (alpha / junk).parent.mkdir(parents=True, exist_ok=True)
            (alpha / junk).write_text("local state\n")
        (alpha / "app" / ".DS_Store").write_text("x")
        self.out = self.tmp / "kit.tar.gz"

    def build(self, ids=("alpha", "beta"), **kwargs):
        with contextlib.redirect_stderr(io.StringIO()):
            return build(self.tasks, list(ids), self.out, **kwargs)

    def names(self):
        with tarfile.open(self.out) as tar:
            return {m.name.removeprefix("kit/") for m in tar.getmembers() if m.isfile()}

    def extract(self) -> Path:
        dest = self.tmp / "extracted"
        with tarfile.open(self.out) as tar:
            tar.extractall(dest, filter="data")
        return dest / "kit"

    def test_exact_selection_without_solutions_or_local_state(self):
        manifest = self.build()
        names = self.names()
        task_files = {n for n in names if n.startswith("tasks/")}
        self.assertEqual(
            task_files,
            {
                f"tasks/{t}/{f}"
                for t in ("alpha", "beta")
                for f in ("task.json", "docker-compose.yml", "app/app.py")
            },
        )
        self.assertIn("attacker/Dockerfile", names)
        self.assertIn("rangebench/__main__.py", names)
        self.assertFalse(any(n.startswith(("scripts/", "tests/", ".github/")) for n in names))
        self.assertEqual(manifest["task_ids"], ["alpha", "beta"])
        self.assertFalse(manifest["solutions_included"])
        self.assertEqual(set(manifest["tasks"]), {"alpha", "beta"})
        self.assertNotIn(str(self.tmp), json.dumps(manifest))

    def test_include_solutions_is_opt_in(self):
        manifest = self.build(ids=["beta"], include_solutions=True)
        self.assertIn("tasks/beta/solution/solve.sh", self.names())
        self.assertTrue(manifest["solutions_included"])
        beta = manifest["tasks"]["beta"]
        self.assertEqual(beta["identity"], beta["source_identity"])
        with tarfile.open(self.out) as tar:
            self.assertEqual(tar.getmember("kit/tasks/beta/solution/solve.sh").mode, 0o755)

    def test_env_file_is_rejected(self):
        for name in (".env", ".env.local", ".netrc"):
            path = self.tasks / "beta" / "app" / name
            path.write_text("TOKEN=example\n")
            with self.assertRaises(PackageError):
                self.build()
            path.unlink()
            self.assertFalse(self.out.exists())

    def test_task_ids_cannot_traverse_or_repeat(self):
        (self.tasks / "notask").mkdir()
        for ids in (["../suite"], ["alpha/app"], ["/etc"], [".git"], [""], ["missing"]):
            with self.subTest(ids=ids), self.assertRaises(PackageError):
                self.build(ids=ids)
        for ids in (["notask"], ["alpha", "alpha"], []):
            with self.subTest(ids=ids), self.assertRaises(PackageError):
                self.build(ids=ids)
        self.assertFalse(self.out.exists())

    def test_mismatched_task_json_id_is_rejected(self):
        (self.tasks / "beta" / "task.json").write_text(json.dumps({"id": "other"}))
        with self.assertRaises(PackageError):
            self.build()

    def test_symlinks_are_rejected(self):
        secret = self.tmp / "outside.txt"
        secret.write_text("private\n")
        link = self.tasks / "beta" / "app" / "data.txt"
        link.symlink_to(secret)
        with self.assertRaises(PackageError):
            self.build()
        link.unlink()
        (self.tasks / "beta" / "app" / "dir").symlink_to(self.tmp)
        with self.assertRaises(PackageError):
            self.build()
        (self.tasks / "linked").symlink_to(self.tasks / "other")
        with self.assertRaises(PackageError):
            self.build(ids=["linked"])
        self.assertFalse(self.out.exists())

    def test_existing_output_is_not_overwritten(self):
        self.out.write_bytes(b"keep")
        with self.assertRaises(PackageError):
            self.build()
        self.assertEqual(self.out.read_bytes(), b"keep")

    def test_extracted_kit_matches_manifest_and_runs_only_selected_tasks(self):
        manifest = self.build()
        kit = self.extract()
        on_disk = {
            p.relative_to(kit).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in kit.rglob("*")
            if p.is_file() and p.name != "suite-manifest.json"
        }
        self.assertEqual(on_disk, manifest["files"])
        self.assertEqual(json.loads((kit / "suite-manifest.json").read_text()), manifest)

        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        py = [sys.executable, "-m"]
        help_out = subprocess.run(
            [*py, "rangebench", "--help"], cwd=kit, env=env, capture_output=True, text=True
        )
        self.assertEqual(help_out.returncode, 0, help_out.stderr)
        listed = subprocess.run(
            [*py, "rangebench", "list"], cwd=kit, env=env, capture_output=True, text=True
        )
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertIn("alpha", listed.stdout)
        self.assertNotIn("other", listed.stdout)
        probe = (
            "import json, rangebench\n"
            "from rangebench import cli\n"
            "from rangebench.identity import all_task_identities\n"
            "print(json.dumps([rangebench.__file__, cli._get_harness_hash(),"
            " cli._get_task_set_hash(), all_task_identities(cli.TASKS_DIR)]))\n"
        )
        seen = subprocess.run(
            [sys.executable, "-P", "-c", probe],
            cwd=kit,
            env={**env, "PYTHONPATH": str(kit)},
            capture_output=True,
            text=True,
        )
        self.assertEqual(seen.returncode, 0, seen.stderr)
        module_file, harness, task_set, identities = json.loads(seen.stdout)
        self.assertTrue(Path(module_file).resolve().is_relative_to(kit.resolve()))
        self.assertEqual(harness, manifest["harness_source_hash"])
        self.assertEqual(task_set, manifest["task_set_hash"])
        self.assertEqual(identities, {t: v["identity"] for t, v in manifest["tasks"].items()})

        run_help = subprocess.run(
            [str(kit / "run-suite.sh"), "--help"], cwd=self.tmp, env=env, capture_output=True
        )
        self.assertEqual(run_help.returncode, 0, run_help.stderr)
        self.assertIn(b"usage: rangebench run", run_help.stdout)

        fake_bin = self.tmp / "bin"
        fake_bin.mkdir()
        fake = fake_bin / "python3"
        fake.write_text('#!/bin/sh\nprintf "%s\\n" "$PYTHONPATH" "$@"\n')
        fake.chmod(0o755)
        wrapped = subprocess.run(
            [str(kit / "run-suite.sh"), "--model", "m", "--api-key-env", "KEY_VAR"],
            cwd=self.tmp,
            env={**env, "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}"},
            capture_output=True,
            text=True,
        )
        self.assertEqual(wrapped.returncode, 0, wrapped.stderr)
        lines = wrapped.stdout.splitlines()
        self.assertEqual(Path(lines[0]).resolve(), kit.resolve())
        self.assertEqual(
            lines[1:],
            ["-P", "-m", "rangebench", "run", "alpha", "beta", "--model", "m"]
            + ["--api-key-env", "KEY_VAR"],
        )


if __name__ == "__main__":
    unittest.main()
