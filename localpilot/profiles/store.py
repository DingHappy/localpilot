from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from localpilot.schemas import SavedProfile
from localpilot.utils import atomic_write_json, project_home


class ProfileStore:
    def __init__(self, root: Path = None) -> None:
        self.root = root or (project_home() / "profiles")
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, profile: SavedProfile) -> Path:
        path = self.root / f"profile-{profile.profile_key}.json"
        atomic_write_json(path, profile.to_dict())
        self.set_current(
            {
                "status": "READY",
                "profile_key": profile.profile_key,
                "candidate": profile.candidate.to_dict(),
                "benchmark": profile.benchmark.to_dict(),
                "score": profile.score,
                "simulated": profile.simulated,
                "updated_at": profile.last_verified_at,
            }
        )
        return path

    def load(self, key: str) -> Optional[SavedProfile]:
        path = self.root / f"profile-{key}.json"
        if not path.exists():
            return None
        return SavedProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def find_match(
        self,
        hardware_fingerprint: str,
        task: str,
        priority: str,
        simulated: bool,
    ) -> Optional[SavedProfile]:
        matches = []
        for path in self.root.glob("profile-*.json"):
            try:
                profile = SavedProfile.from_dict(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if (
                profile.hardware_fingerprint == hardware_fingerprint
                and profile.task == task
                and profile.priority == priority
                and profile.simulated == simulated
            ):
                matches.append(profile)
        if not matches:
            return None
        return max(matches, key=lambda item: item.last_verified_at)

    def save_run(self, run_id: str, data: Dict[str, Any]) -> Path:
        path = self.root / "runs" / f"{run_id}.json"
        atomic_write_json(path, data)
        return path

    def current(self) -> Dict[str, Any]:
        path = self.root / "current.json"
        if not path.exists():
            return {"status": "STOPPED"}
        return json.loads(path.read_text(encoding="utf-8"))

    def set_current(self, state: Dict[str, Any]) -> None:
        atomic_write_json(self.root / "current.json", state)

    def stop(self) -> None:
        current = self.current()
        current["status"] = "STOPPED"
        self.set_current(current)

