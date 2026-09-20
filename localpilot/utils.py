from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def project_home() -> Path:
    configured = os.environ.get("LOCALPILOT_HOME")
    if configured:
        return Path(configured).expanduser().resolve()

    current = Path.cwd().resolve()
    if (current / "config" / "models.yaml").exists():
        return current

    package_root = Path(__file__).resolve().parent.parent
    return package_root


def load_data_file(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                f"{path} is not JSON-compatible YAML and PyYAML is not installed"
            ) from exc
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a mapping")
        return loaded


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False
    ) as handle:
        handle.write(payload)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def append_event(event: str, details: Dict[str, Any]) -> None:
    home = project_home()
    log_path = home / "logs" / "localpilot.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    safe_details = {
        key: value
        for key, value in details.items()
        if key not in {"prompt", "messages", "response", "code"}
    }
    record = {"timestamp": utc_now(), "event": event, "details": safe_details}
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

