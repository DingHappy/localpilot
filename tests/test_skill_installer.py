from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "install_skill.py"
)
SPEC = importlib.util.spec_from_file_location("localpilot_skill_installer", SCRIPT)
installer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(installer)


class SkillInstallerTests(unittest.TestCase):
    def test_install_status_and_uninstall(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            installed = installer.install(target, "claude-code")
            self.assertEqual("installed", installed["status"])
            self.assertTrue(installed["discoverable"])
            destination = target / ".claude" / "skills" / installer.SKILL_NAME
            self.assertTrue((destination / "SKILL.md").is_file())
            self.assertTrue(
                (destination / "references" / "evidence.md").is_file()
            )
            self.assertTrue(
                (destination / "references" / "composition.md").is_file()
            )
            self.assertFalse((destination / "evals").exists())
            self.assertFalse((destination / "scripts").exists())

            state = installer.status(target, "claude-code")
            self.assertEqual("installed", state["status"])
            self.assertEqual([], state["modified"])

            removed = installer.uninstall(target, "claude-code")
            self.assertEqual("not_installed", removed["status"])
            self.assertFalse(Path(removed["destination"]).exists())

    def test_refuses_unmanaged_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            destination = (
                target / ".cursor" / "skills" / installer.SKILL_NAME
            )
            destination.mkdir(parents=True)
            (destination / "SKILL.md").write_text("unmanaged\n", encoding="utf-8")

            with self.assertRaises(installer.InstallError):
                installer.install(target, "cursor")

    def test_refuses_to_remove_modified_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            installer.install(target, "gemini")
            destination = target / ".gemini" / "skills" / installer.SKILL_NAME
            with (destination / "SKILL.md").open("a", encoding="utf-8") as handle:
                handle.write("\nlocal change\n")

            with self.assertRaises(installer.InstallError):
                installer.uninstall(target, "gemini")

    def test_repository_copy_is_directly_discoverable_for_codex(self):
        repository = Path(__file__).resolve().parents[1]
        state = installer.status(repository, "codex")
        self.assertEqual("discoverable_source", state["status"])
        self.assertTrue(state["discoverable"])

        removed = installer.uninstall(repository, "codex")
        self.assertEqual("canonical_preserved", removed["action"])
        self.assertTrue(removed["discoverable"])

    def test_project_copy_matches_installable_package(self):
        repository = Path(__file__).resolve().parents[1]
        package = repository / "skills" / installer.SKILL_NAME
        project_copy = repository / ".agents" / "skills" / installer.SKILL_NAME
        self.assertEqual(installer._hashes(package), installer._hashes(project_copy))


if __name__ == "__main__":
    unittest.main()
