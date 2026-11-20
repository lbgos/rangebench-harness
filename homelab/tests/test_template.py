"""Template loading: the example parses, typos in session.toml fail loudly."""

import shutil
import tempfile
import unittest
from pathlib import Path

from rangebench_homelab.template import TEMPLATES_DIR, load_template, suite_of


class TemplateTest(unittest.TestCase):
    def test_example_loads(self) -> None:
        tpl = load_template("example")
        self.assertEqual(tpl.suite, "none")

    def test_suite_string_or_list(self) -> None:
        self.assertEqual(suite_of({}, "t"), "max")
        self.assertEqual(suite_of({"suite": "medium"}, "t"), "medium")
        self.assertEqual(suite_of({"suite": ["medium", "max"]}, "t"), "medium")
        self.assertEqual(suite_of({"suite": ["max"]}, "t"), "max")
        for bad in ("big", ["max", "big"], [], ["none", "max"]):
            with self.assertRaises(ValueError):
                suite_of({"suite": bad}, "t")

    def test_unknown_step_key_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "typo"
            shutil.copytree(TEMPLATES_DIR / "example", copy)
            toml = copy / "session.toml"
            toml.write_text(toml.read_text().replace("trap = true", "trapp = true"))
            with self.assertRaisesRegex(ValueError, "unknown keys .*trapp"):
                load_template(copy)


if __name__ == "__main__":
    unittest.main()
