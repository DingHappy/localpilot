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

    def save(self, profile: SavedProfile, activate: bool = True) -> Path:
        path = self.root / f"profile-{profile.profile_key}.json"
        atomic_write_json(path, profile.to_dict())
        if activate:
            self.set_current(self._current_from_profile(profile))
        return path

    def activate(self, key: str) -> SavedProfile:
        profile = self.load(key)
        if profile is None:
            raise ValueError(f"Unknown profile: {key}")
        self.set_current(self._current_from_profile(profile))
        return profile

    @staticmethod
    def _current_from_profile(profile: SavedProfile) -> Dict[str, Any]:
        return {
            "status": "READY",
            "profile_key": profile.profile_key,
            "candidate": profile.candidate.to_dict(),
            "benchmark": profile.benchmark.to_dict(),
            "score": profile.score,
            "simulated": profile.simulated,
            "updated_at": profile.last_verified_at,
        }

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
        requirements: Dict[str, Any],
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
                and profile.requirements == requirements
            ):
                matches.append(profile)
        if not matches:
            return None
        return max(matches, key=lambda item: item.last_verified_at)

    def list_profiles(self) -> list:
        """Saved profiles, newest verification first."""
        profiles = []
        for path in self.root.glob("profile-*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            candidate = data.get("candidate", {})
            benchmark = data.get("benchmark", {})
            profiles.append(
                {
                    "profile_key": data.get("profile_key"),
                    "task": data.get("task"),
                    "priority": data.get("priority"),
                    "platform_id": data.get("platform_id"),
                    "candidate_id": candidate.get("candidate_id"),
                    "model_id": candidate.get("model_id"),
                    "engine": candidate.get("engine"),
                    "precision": candidate.get("precision"),
                    "context_length": candidate.get("context_length"),
                    "concurrency": candidate.get("concurrency"),
                    "score": data.get("score"),
                    "simulated": data.get("simulated"),
                    "requirements": data.get("requirements", {}),
                    "ttft_ms": benchmark.get("ttft_ms"),
                    "throughput_tokens_s": benchmark.get("throughput_tokens_s"),
                    "peak_memory_gb": benchmark.get("peak_memory_gb"),
                    "created_at": data.get("created_at"),
                    "last_verified_at": data.get("last_verified_at"),
                }
            )
        profiles.sort(key=lambda item: item.get("last_verified_at") or "", reverse=True)
        return profiles

    def save_run(self, run_id: str, data: Dict[str, Any]) -> Path:
        path = self.root / "runs" / f"{run_id}.json"
        atomic_write_json(path, data)
        return path

    def load_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        path = self.root / "runs" / f"{run_id}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def list_runs(self, limit: int = 20) -> list:
        runs_dir = self.root / "runs"
        if not runs_dir.is_dir():
            return []
        paths = sorted(
            runs_dir.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        summaries = []
        for path in paths[:limit]:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            best = data.get("best_profile") or {}
            summaries.append(
                {
                    "run_id": data.get("run_id"),
                    "task": (data.get("intent") or {}).get("task"),
                    "priority": (data.get("intent") or {}).get("priority"),
                    "goal": (data.get("intent") or {}).get("raw_text", "")[:160],
                    "status": data.get("status"),
                    "profile_reused": data.get("profile_reused"),
                    "simulated": (data.get("hardware") or {}).get("simulated"),
                    "candidates": len(data.get("candidates") or []),
                    "winner": (best.get("candidate") or {}).get("candidate_id"),
                    "score": best.get("score"),
                    "started_at": data.get("started_at"),
                    "finished_at": data.get("finished_at"),
                }
            )
        return summaries

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
