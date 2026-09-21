from __future__ import annotations

import contextlib
import io
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from localpilot import __version__
from localpilot.cli import build_parser, main


class CliContractTests(unittest.TestCase):
    def test_version_matches_package_version(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as raised:
                main(["--version"])
        self.assertEqual(0, raised.exception.code)
        self.assertEqual(f"localpilot {__version__}\n", output.getvalue())

    def test_package_version_matches_project_metadata(self):
        metadata = (
            Path(__file__).resolve().parents[1] / "pyproject.toml"
        ).read_text(encoding="utf-8")
        declared = re.search(
            r'^version = "([^"]+)"$', metadata, flags=re.MULTILINE
        )
        self.assertIsNotNone(declared)
        self.assertEqual(declared.group(1), __version__)

    def test_registry_uses_real_hardware_by_default(self):
        args = build_parser().parse_args(["registry"])
        self.assertFalse(args.simulate)

    def test_registry_simulation_is_explicit(self):
        args = build_parser().parse_args(["registry", "--simulate"])
        self.assertTrue(args.simulate)

    def test_registry_passes_default_hardware_mode_to_profiler(self):
        profile = type(
            "Profile",
            (),
            {
                "platform_id": "generic",
                "memory": {"total_gb": 32},
                "memory_bandwidth_gbps": None,
                "unified_memory": False,
            },
        )()
        with patch("localpilot.cli.HardwareProfiler.profile", return_value=profile) as call:
            with patch("localpilot.cli.ModelRegistry.all", return_value=[]):
                self.assertEqual(0, main(["registry", "--json"]))
        call.assert_called_once_with(simulate=False)


if __name__ == "__main__":
    unittest.main()
