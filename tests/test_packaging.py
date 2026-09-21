from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from localpilot.utils import config_file, project_home


class PackagingTests(unittest.TestCase):
    def test_packaged_configs_match_checkout_configs(self):
        root = Path(__file__).resolve().parents[1]
        packaged = root / "localpilot" / "resources" / "config"
        for source in sorted((root / "config").glob("*.yaml")):
            self.assertEqual(
                source.read_bytes(),
                (packaged / source.name).read_bytes(),
                source.name,
            )

    def test_config_falls_back_to_packaged_copy_outside_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {}, clear=True), patch(
                "pathlib.Path.cwd", return_value=Path(directory)
            ):
                path = config_file("models.yaml")
        self.assertEqual("models.yaml", path.name)
        self.assertIn("resources/config", path.as_posix())
        self.assertTrue(path.is_file())

    def test_installed_state_uses_explicit_state_home(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state"
            with patch.dict(
                os.environ,
                {"LOCALPILOT_STATE_HOME": str(state)},
                clear=True,
            ), patch("pathlib.Path.cwd", return_value=Path(directory)):
                self.assertEqual(state.resolve(), project_home())


if __name__ == "__main__":
    unittest.main()
