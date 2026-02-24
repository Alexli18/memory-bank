"""Storage module — backward-compatible shim delegating to NdjsonStorage.

All public functions are preserved with identical signatures.
Internally, each call constructs NdjsonStorage(root) and delegates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mb.models import SessionMeta
from mb.store import (
    DEFAULT_CONFIG,
    MEMORY_BANK_DIR,
    CONFIG_VERSION,
    MbStorageError,
    NdjsonStorage,
    _generate_session_id as generate_session_id,
)

__all__ = [
    "MbStorageError",
    "MEMORY_BANK_DIR",
    "CONFIG_VERSION",
    "DEFAULT_CONFIG",
    "generate_session_id",
    "init_storage",
    "ensure_initialized",
    "read_config",
    "write_config",
    "create_session",
    "write_event",
    "finalize_session",
    "delete_session",
    "list_sessions",
]


def _storage_root() -> Path:
    """Return the storage root for the current working directory."""
    return Path.cwd() / MEMORY_BANK_DIR


def init_storage(root: Path | None = None) -> tuple[bool, Path]:
    """Initialize .memory-bank/ directory structure.

    Returns:
        Tuple of (created: bool, storage_path: Path).
        created is False if already initialized.
    """
    created, store = NdjsonStorage.init(root)
    return created, store.root


def ensure_initialized(root: Path | None = None) -> Path:
    """Check that storage is initialized. Return storage path or raise FileNotFoundError."""
    store = NdjsonStorage.open(root)
    return store.root


def read_config(root: Path | None = None) -> dict[str, Any]:
    """Read config.json from storage."""
    store = NdjsonStorage.open(root)
    return store.read_config()


def write_config(config: dict[str, Any], root: Path | None = None) -> None:
    """Write config.json to storage."""
    store = NdjsonStorage.open(root)
    store.write_config(config)


def create_session(
    command: list[str],
    cwd: str | None = None,
    root: Path | None = None,
    source: str | None = None,
    create_events: bool = True,
) -> SessionMeta:
    """Create a new session directory with meta.json.

    Returns:
        SessionMeta for the created session.
    """
    storage = root or _storage_root()
    store = NdjsonStorage(storage)
    return store.create_session(command, cwd=cwd, source=source, create_events=create_events)


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
    store = NdjsonStorage(storage)
    store.write_event(session_id, stream, role, content, ts=ts)


def finalize_session(
    session_id: str,
    exit_code: int | None = None,
    root: Path | None = None,
) -> None:
    """Update meta.json with end time and optionally exit code."""
    storage = root or _storage_root()
    store = NdjsonStorage(storage)
    store.finalize_session(session_id, exit_code=exit_code)


def delete_session(session_id: str, root: Path | None = None) -> bool:
    """Delete a session directory. Returns True if deleted, False if not found."""
    storage = root or _storage_root()
    store = NdjsonStorage(storage)
    return store.delete_session(session_id)


def list_sessions(root: Path | None = None) -> list[SessionMeta]:
    """Read all session meta.json files, sorted by start time descending."""
    storage = root or _storage_root()
    store = NdjsonStorage(storage)
    return store.list_sessions()
