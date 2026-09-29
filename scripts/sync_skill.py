#!/usr/bin/env python3
"""Keep Codex's project-local discovery copy equal to the installable Skill."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "skills" / "local-ai-autopilot"
DESTINATION = ROOT / ".agents" / "skills" / "local-ai-autopilot"


def hashes(root: Path) -> dict[str, str]:
    if not root.is_dir():
        return {}
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
        and not any(part == "__pycache__" for part in path.relative_to(root).parts)
        and path.suffix != ".pyc"
        and path.name not in {".DS_Store", ".localpilot-install.json"}
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="replace the project discovery copy")
    parser.add_argument("--force", action="store_true", help="replace a locally modified discovery copy")
    args = parser.parse_args()
    if not (SOURCE / "SKILL.md").is_file():
        parser.error(f"Missing canonical Skill: {SOURCE}")
    if args.write:
        if DESTINATION.exists() and not args.force:
            relative = DESTINATION.relative_to(ROOT)
            result = subprocess.run(
                ["git", "status", "--porcelain", "--", str(relative)],
                cwd=ROOT, capture_output=True, text=True, check=True,
            )
            if result.stdout.strip():
                parser.error(
                    "Project discovery copy has local changes; review them "
                    "before using --write --force"
                )
        if DESTINATION.exists():
            shutil.rmtree(DESTINATION)
        DESTINATION.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            SOURCE,
            DESTINATION,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
        )
    source_files = hashes(SOURCE)
    copy_files = hashes(DESTINATION)
    changed = sorted(
        name for name in set(source_files) | set(copy_files)
        if source_files.get(name) != copy_files.get(name)
    )
    if changed:
        print("Skill copy differs: " + ", ".join(changed))
        return 1
    print("Skill package and project discovery copy match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
