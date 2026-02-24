"""Storage module — session lifecycle, event persistence, and directory management."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MbStorageError(Exception):
    """Error raised for corrupt or unreadable storage files."""


MEMORY_BANK_DIR = ".memory-bank"
CONFIG_VERSION = "1.0"

DEFAULT_CONFIG = {
    "version": CONFIG_VERSION,
    "ollama": {
        "base_url": "http://localhost:11434",
        "embed_model": "nomic-embed-text",
        "chat_model": "gemma3:4b",
    },
    "chunking": {
        "max_tokens": 512,
        "overlap_tokens": 50,
    },
}


def _storage_root() -> Path:
    """Return the storage root for the current working directory."""
    return Path.cwd() / MEMORY_BANK_DIR


def generate_session_id() -> str:
    """Generate a session ID in YYYYMMDD-HHMMSS-XXXX format."""
    now = datetime.now(timezone.utc)
    hex_suffix = os.urandom(2).hex()
    return now.strftime("%Y%m%d-%H%M%S") + f"-{hex_suffix}"


def init_storage(root: Path | None = None) -> tuple[bool, Path]:
    """Initialize .memory-bank/ directory structure.

    Returns:
        Tuple of (created: bool, storage_path: Path).
        created is False if already initialized.
    """
    storage = root or _storage_root()

    if (storage / "config.json").exists():
        return False, storage

    storage.mkdir(exist_ok=True)
    (storage / "sessions").mkdir(exist_ok=True)
    (storage / "index").mkdir(exist_ok=True)
    (storage / "state").mkdir(exist_ok=True)

    config_path = storage / "config.json"
    config_path.write_text(
        json.dumps(DEFAULT_CONFIG, indent=2) + "\n", encoding="utf-8"
    )

    _ensure_gitignore(storage)

    return True, storage


def _ensure_gitignore(storage: Path) -> None:
    """Add .memory-bank/ to .gitignore in the project root."""
    project_root = storage.parent
    gitignore = project_root / ".gitignore"
    entry = MEMORY_BANK_DIR + "/"

    if gitignore.exists():
        content = gitignore.read_text(encoding="utf-8")
        if entry in content.splitlines():
            return
        # Ensure trailing newline before appending
        if content and not content.endswith("\n"):
            content += "\n"
        content += entry + "\n"
        gitignore.write_text(content, encoding="utf-8")
    else:
        gitignore.write_text(entry + "\n", encoding="utf-8")


def ensure_initialized(root: Path | None = None) -> Path:
    """Check that storage is initialized. Return storage path or raise FileNotFoundError."""
    storage = root or _storage_root()
    if not (storage / "config.json").exists():
        raise FileNotFoundError(
            f"Memory Bank not initialized. Run `mb init` first. (looked in {storage})"
        )
    return storage


def read_config(root: Path | None = None) -> dict[str, Any]:
    """Read config.json from storage."""
    storage = ensure_initialized(root)
    config_path = storage / "config.json"
    try:
        result: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise MbStorageError(f"Corrupt config.json: {e}") from e
    return result


def write_config(config: dict[str, Any], root: Path | None = None) -> None:
    """Write config.json to storage."""
    storage = ensure_initialized(root)
    config_path = storage / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def create_session(
    command: list[str],
    cwd: str | None = None,
    root: Path | None = None,
    source: str | None = None,
    create_events: bool = True,
) -> str:
    """Create a new session directory with meta.json.

    Args:
        command: Command that was executed.
        cwd: Working directory (defaults to cwd).
        root: Storage root path.
        source: Optional source tag (e.g. "hook").
        create_events: If True (default), create empty events.jsonl.

    Returns:
        The session_id.
    """
    storage = root or _storage_root()
    session_id = generate_session_id()
    session_dir = storage / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {
        "session_id": session_id,
        "command": command,
        "cwd": cwd or str(Path.cwd()),
        "started_at": time.time(),
        "ended_at": None,
        "exit_code": None,
    }
    if source is not None:
        meta["source"] = source

    (session_dir / "meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    if create_events:
        (session_dir / "events.jsonl").touch()

    return session_id


def write_event(
    session_id: str,
    stream: str,
    role: str,
    content: str,
    ts: float | None = None,
    root: Path | None = None,
) -> None:
    """Append an event to the session's events.jsonl."""
    storage = root or _storage_root()
    events_path = storage / "sessions" / session_id / "events.jsonl"

    event = {
        "ts": ts if ts is not None else time.monotonic(),
        "session_id": session_id,
        "stream": stream,
        "role": role,
        "content": content,
    }
    line = json.dumps(event, ensure_ascii=False) + "\n"

    with events_path.open("a", encoding="utf-8") as f:
        f.write(line)


def finalize_session(
    session_id: str,
    exit_code: int | None = None,
    root: Path | None = None,
) -> None:
    """Update meta.json with end time and optionally exit code."""
    storage = root or _storage_root()
    meta_path = storage / "sessions" / session_id / "meta.json"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["ended_at"] = time.time()
    if exit_code is not None:
        meta["exit_code"] = exit_code
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def delete_session(session_id: str, root: Path | None = None) -> bool:
    """Delete a session directory. Returns True if deleted, False if not found."""
    import shutil

    storage = root or _storage_root()
    session_dir = storage / "sessions" / session_id
    if not session_dir.exists():
        return False
    shutil.rmtree(session_dir)
    return True


def list_sessions(root: Path | None = None) -> list[dict[str, Any]]:
    """Read all session meta.json files, sorted by start time descending."""
    storage = root or _storage_root()
    sessions_dir = storage / "sessions"

    if not sessions_dir.exists():
        return []

    sessions = []
    for entry in sessions_dir.iterdir():
        if not entry.is_dir():
            continue
        meta_path = entry / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Skipping corrupt session %s", entry.name)
            continue
        sessions.append(meta)

    sessions.sort(key=lambda s: s.get("started_at", 0), reverse=True)
    return sessions
