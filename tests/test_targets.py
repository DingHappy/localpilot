from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from localpilot import cli
from localpilot.targets import (
    SSHTarget,
    TargetError,
    extract_target_options,
    run_on_target,
)


class TargetParsingTests(unittest.TestCase):
    def test_extracts_target_flags_in_any_position(self):
        target, command, remaining = extract_target_options(
            ["autopilot", "goal", "--target", "ssh://pilot@spark:2222", "--json"]
        )
        self.assertEqual(target, "ssh://pilot@spark:2222")
        self.assertEqual(command, "localpilot")
        self.assertEqual(remaining, ["autopilot", "goal", "--json"])

    def test_rejects_passwords_and_paths(self):
        with self.assertRaises(TargetError):
            SSHTarget.parse("ssh://pilot:secret@spark")
        with self.assertRaises(TargetError):
            SSHTarget.parse("ssh://pilot@spark/tmp")

    @patch("localpilot.targets.shutil.which", return_value="/usr/bin/ssh")
    def test_builds_a_quoted_remote_command(self, _which):
        target = SSHTarget.parse("ssh://pilot@spark:2222")
        command = target.command(
            ["autopilot", "发票识别，质量优先", "--json"], "localpilot"
        )
        self.assertEqual(command[:4], ["/usr/bin/ssh", "-p", "2222", "pilot@spark"])
        self.assertEqual(
            command[4], "localpilot autopilot '发票识别，质量优先' --json"
        )

    @patch("localpilot.targets.shutil.which", return_value="/usr/bin/ssh")
    def test_remote_command_may_name_a_virtualenv_python_module(self, _which):
        command = SSHTarget.parse("ssh://spark").command(
            ["doctor", "--json"], "python3 -m localpilot.cli"
        )
        self.assertEqual(
            command[-1], "python3 -m localpilot.cli doctor --json"
        )

    @patch("localpilot.targets.subprocess.run")
    @patch("localpilot.targets.shutil.which", return_value="/usr/bin/ssh")
    def test_returns_the_remote_exit_code(self, _which, mocked_run):
        mocked_run.return_value = subprocess.CompletedProcess([], 7)
        code = run_on_target("ssh://spark", ["doctor", "--json"], "localpilot")
        self.assertEqual(code, 7)
        mocked_run.assert_called_once_with(
            ["/usr/bin/ssh", "spark", "localpilot doctor --json"], check=False
        )

    @patch("localpilot.targets.shutil.which", return_value="/usr/bin/ssh")
    def test_remote_dashboard_requires_an_explicit_tunnel(self, _which):
        with self.assertRaises(TargetError):
            SSHTarget.parse("ssh://spark").command(["serve"], "localpilot")

    @patch("localpilot.cli.run_on_target", return_value=0)
    def test_cli_routes_to_remote_before_running_local_handlers(self, remote):
        code = cli.main(
            ["autopilot", "invoice task", "--target", "ssh://pilot@spark", "--json"]
        )
        self.assertEqual(code, 0)
        remote.assert_called_once_with(
            "ssh://pilot@spark",
            ["autopilot", "invoice task", "--json"],
            "localpilot",
        )

    @patch("localpilot.cli.run_on_target", return_value=0)
    def test_remote_version_check_is_proxied_to_the_node(self, remote):
        code = cli.main(["--target", "ssh://pilot@spark", "--version"])
        self.assertEqual(code, 0)
        remote.assert_called_once_with(
            "ssh://pilot@spark", ["--version"], "localpilot"
        )


if __name__ == "__main__":
    unittest.main()
