#!/usr/bin/env python3
"""Install the LocalPilot skill into a supported agent's project directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List


SKILL_NAME = "local-ai-autopilot"
MANIFEST_NAME = ".localpilot-install.json"
AGENT_PATHS = {
    "codex": Path(".agents/skills"),
    "claude-code": Path(".claude/skills"),
    "cursor": Path(".cursor/skills"),
    "gemini": Path(".gemini/skills"),
}


class InstallError(RuntimeError):
    pass


def skill_source() -> Path:
    return Path(__file__).resolve().parents[1]


def _included_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if (
            path.name in {MANIFEST_NAME, ".DS_Store"}
            or "__pycache__" in relative.parts
            or path.suffix == ".pyc"
        ):
            continue
        yield path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hashes(root: Path) -> Dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _digest(path)
        for path in _included_files(root)
    }


def _read_manifest(destination: Path) -> dict | None:
    path = destination / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InstallError(f"Invalid install manifest at {path}: {exc}") from exc
    if value.get("skill") != SKILL_NAME or not isinstance(value.get("files"), dict):
        raise InstallError(f"Unrecognized install manifest at {path}")
    return value


def _validate_skill(root: Path) -> None:
    entrypoint = root / "SKILL.md"
    if not entrypoint.is_file():
        raise InstallError(f"Missing skill entrypoint: {entrypoint}")
    text = entrypoint.read_text(encoding="utf-8")
    if f"name: {SKILL_NAME}" not in text:
        raise InstallError(f"SKILL.md does not declare name: {SKILL_NAME}")


def _modified(destination: Path, manifest: dict) -> List[str]:
    expected = manifest["files"]
    current = _hashes(destination)
    names = sorted(set(expected) | set(current))
    return [name for name in names if expected.get(name) != current.get(name)]


def _destination(target: Path, agent: str) -> Path:
    return target.resolve() / AGENT_PATHS[agent] / SKILL_NAME


def status(target: Path, agent: str) -> dict:
    source = skill_source()
    destination = _destination(target, agent)
    if destination.resolve() == source:
        _validate_skill(source)
        return {
            "agent": agent,
            "destination": str(destination),
            "status": "discoverable_source",
            "discoverable": True,
            "modified": [],
        }
    if not destination.exists():
        return {
            "agent": agent,
            "destination": str(destination),
            "status": "not_installed",
            "discoverable": False,
            "modified": [],
        }
    manifest = _read_manifest(destination)
    if manifest is None:
        return {
            "agent": agent,
            "destination": str(destination),
            "status": "unmanaged_collision",
            "discoverable": (destination / "SKILL.md").is_file(),
            "modified": [],
        }
    changed = _modified(destination, manifest)
    return {
        "agent": agent,
        "destination": str(destination),
        "status": "modified" if changed else "installed",
        "discoverable": (destination / "SKILL.md").is_file(),
        "modified": changed,
    }


def install(target: Path, agent: str, dry_run: bool = False) -> dict:
    source = skill_source()
    _validate_skill(source)
    destination = _destination(target, agent)
    if destination.resolve() == source:
        result = status(target, agent)
        result["action"] = "none"
        return result

    if destination.exists():
        manifest = _read_manifest(destination)
        if manifest is None:
            raise InstallError(
                f"Refusing to overwrite unmanaged skill directory: {destination}"
            )
        changed = _modified(destination, manifest)
        if changed:
            raise InstallError(
                "Refusing to overwrite a modified installation: "
                + ", ".join(changed)
            )

    if dry_run:
        return {
            "agent": agent,
            "destination": str(destination),
            "status": "would_install",
            "discoverable": False,
            "modified": [],
            "action": "preview",
        }

    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(MANIFEST_NAME, ".DS_Store", "__pycache__", "*.pyc"),
    )
    files = _hashes(destination)
    manifest = {
        "schema_version": 1,
        "skill": SKILL_NAME,
        "agent": agent,
        "source": str(source),
        "files": files,
    }
    (destination / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = status(target, agent)
    result["action"] = "installed"
    return result


def uninstall(target: Path, agent: str, dry_run: bool = False) -> dict:
    source = skill_source()
    destination = _destination(target, agent)
    if destination.resolve() == source:
        result = status(target, agent)
        result["action"] = "canonical_preserved"
        return result
    if not destination.exists():
        return {
            "agent": agent,
            "destination": str(destination),
            "status": "not_installed",
            "discoverable": False,
            "modified": [],
            "action": "none",
        }
    manifest = _read_manifest(destination)
    if manifest is None:
        raise InstallError(f"Refusing to remove unmanaged directory: {destination}")
    changed = _modified(destination, manifest)
    if changed:
        raise InstallError(
            "Refusing to remove a modified installation: " + ", ".join(changed)
        )
    if dry_run:
        return {
            "agent": agent,
            "destination": str(destination),
            "status": "would_uninstall",
            "discoverable": True,
            "modified": [],
            "action": "preview",
        }
    shutil.rmtree(destination)
    return {
        "agent": agent,
        "destination": str(destination),
        "status": "not_installed",
        "discoverable": False,
        "modified": [],
        "action": "uninstalled",
    }


def _agents(value: str) -> List[str]:
    return list(AGENT_PATHS) if value == "all" else [value]


def _print(results: List[dict], as_json: bool) -> None:
    if as_json:
        print(json.dumps({"skill": SKILL_NAME, "results": results}, indent=2))
        return
    for result in results:
        print(
            f"{result['agent']}: {result['status']} "
            f"({result['destination']})"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install or inspect the LocalPilot Agent Skill"
    )
    parser.add_argument("action", choices=("install", "status", "uninstall"))
    parser.add_argument(
        "--agent", choices=tuple(AGENT_PATHS) + ("all",), default="codex"
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path.cwd(),
        help="project root receiving the agent-specific skill directory",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        results = []
        for agent in _agents(args.agent):
            if args.action == "install":
                results.append(install(args.target, agent, args.dry_run))
            elif args.action == "uninstall":
                results.append(uninstall(args.target, agent, args.dry_run))
            else:
                results.append(status(args.target, agent))
        _print(results, args.json)
        return 0
    except InstallError as exc:
        if args.json:
            print(json.dumps({"skill": SKILL_NAME, "error": str(exc)}))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
