"""Local JSON speech history, compatible with existing speeches/<user>/*.json files."""
import json
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent / "speeches"


def user_dir(user):
    name = re.sub(r"[^A-Za-z0-9_-]", "_", user.strip())[:50]
    if not name or name in (".", ".."):
        raise ValueError("Enter a valid local profile name")
    path = ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_record(user, record):
    folder = user_dir(user)
    name = datetime.now().strftime("%Y-%m-%d_%H%M%S_") + uuid4().hex[:8] + ".json"
    path = folder / name
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_records(user):
    rows = []
    for path in user_dir(user).glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(record, dict) and isinstance(record.get("summary"), dict):
            record["_filename"] = path.name
            rows.append(record)
    return sorted(rows, key=lambda row: row.get("when", ""))


def load_last(user):
    records = load_records(user)
    return records[-1] if records else None


def delete_record(user, filename):
    if Path(filename).name != filename or not filename.endswith(".json"):
        raise ValueError("Invalid filename")
    (user_dir(user) / filename).unlink()
