import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from rangebench.env import EnvError, Task, TaskEnv


class NetworkIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        task = Task("sample", Path("/tmp"), "web", 1, "Find the flag")
        self.env = TaskEnv(task, "rb-test")

    def test_external_task_network_is_rejected(self) -> None:
        responses = [
            subprocess.CompletedProcess([], 0, "rb-test_default\nrb-test_core\n", ""),
            subprocess.CompletedProcess([], 0, "false\n", ""),
        ]
        with patch("rangebench.env._run", side_effect=responses):
            with self.assertRaisesRegex(EnvError, "allows external access"):
                self.env.verify_isolation(["rb-test_default"])

    def test_compose_escape_is_rejected_before_up(self) -> None:
        safe = {
            "networks": {"default": {"name": "rb-test_default", "internal": True}},
            "services": {"web": {"networks": {"default": {}}}},
        }
        unsafe = (
            ({"networks": {"default": {"name": "rb-test_default"}}}, "external access"),
            ({"networks": {"default": {"name": "rb-test_default", "internal": True, "external": True}}}, "is external"),
            ({"networks": {"default": {"name": "shared", "internal": True}}}, "project-scoped"),
            ({"services": {"web": {"network_mode": "host"}}}, "bypasses Compose networks"),
        )
        for change, error in unsafe:
            with self.subTest(error=error):
                config = safe | change
                rendered = subprocess.CompletedProcess([], 0, json.dumps(config), "")
                with patch("rangebench.env._run", return_value=rendered) as run:
                    with self.assertRaisesRegex(EnvError, error):
                        self.env.up()
                self.assertEqual(run.call_count, 1)
                self.assertEqual(run.call_args.args[0][-3:], ["config", "--format", "json"])

    def test_safe_compose_config_precedes_reset_and_up(self) -> None:
        config = {
            "networks": {"default": {"name": "rb-test_default", "internal": True}},
            "services": {"web": {"networks": {"default": {}}}},
        }
        rendered = subprocess.CompletedProcess([], 0, json.dumps(config), "")
        cleared = subprocess.CompletedProcess([], 0, "", "")
        with patch("rangebench.env._run", side_effect=[rendered, cleared, cleared, EnvError("stop before real Docker")]) as run:
            with self.assertRaisesRegex(EnvError, "stop before real Docker"):
                self.env.up()
        self.assertEqual(run.call_count, 4)
        self.assertEqual(run.call_args_list[0].args[0][-3:], ["config", "--format", "json"])
        self.assertEqual(run.call_args_list[1].args[0], ["docker", "rm", "-f", "rb-test-atk"])
        self.assertEqual(run.call_args_list[2].args[0][-3:], ["down", "-v", "--remove-orphans"])
        self.assertEqual(run.call_args_list[3].args[0][-3:], ["up", "-d", "--build"])

    def test_failed_volume_reset_prevents_up(self) -> None:
        config = {
            "networks": {"default": {"name": "rb-test_default", "internal": True}},
            "services": {"web": {"networks": {"default": {}}}},
        }
        rendered = subprocess.CompletedProcess([], 0, json.dumps(config), "")
        cleared = subprocess.CompletedProcess([], 0, "", "")
        with patch("rangebench.env._run", side_effect=[rendered, cleared, EnvError("volume in use")]) as run:
            with self.assertRaisesRegex(EnvError, "volume in use"):
                self.env.up()
        self.assertEqual(run.call_count, 3)
        self.assertEqual(run.call_args.args[0][-3:], ["down", "-v", "--remove-orphans"])


if __name__ == "__main__":
    unittest.main()
