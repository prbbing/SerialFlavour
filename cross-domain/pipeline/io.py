"""Small, explicit artifact and identity helpers shared across domains."""

import hashlib
import json
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def artifact_record(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"stage artifact missing: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def artifacts_match(records):
    return bool(records) and all(
        Path(item["path"]).is_file()
        and Path(item["path"]).stat().st_size == item["bytes"]
        and sha256_file(item["path"]) == item["sha256"]
        for item in records
    )
