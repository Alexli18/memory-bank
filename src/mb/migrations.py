"""Schema versioning and migration for Memory Bank storage."""

from __future__ import annotations

import json
import logging
from typing import Any

from mb.models import _generate_event_id
from mb.ollama_client import OllamaClient
from mb.search import VectorIndex
from mb.store import NdjsonStorage

logger = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION = 2


def detect_version(storage: NdjsonStorage) -> int:
    """Detect the schema version of a storage instance.

    Returns 1 if no schema_version field exists in config, otherwise
    returns the integer value of schema_version.
    """
    config = storage.read_config()
    return int(config.get("schema_version", 1))


def migrate(storage: NdjsonStorage) -> tuple[int, int]:
    """Run all pending migrations on the storage.

    Returns (old_version, new_version).
    """
    old_version = detect_version(storage)

    if old_version >= CURRENT_SCHEMA_VERSION:
        return old_version, old_version

    current = old_version

    if current == 1:
        _migrate_v1_to_v2(storage)
        current = 2

    return old_version, current


def _migrate_v1_to_v2(storage: NdjsonStorage) -> None:
    """Migrate from v1 to v2: add schema_version to config, add event_id to events."""
    # 1. Update config
    config = storage.read_config()
    config["schema_version"] = 2
    storage.write_config(config)

    # 2. Add event_id to events that don't have one
    sessions_dir = storage.root / "sessions"
    if not sessions_dir.exists():
        return

    for session_dir in sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue

        events_path = session_dir / "events.jsonl"
        if not events_path.exists():
            continue

        # Read all events
        lines: list[str] = []
        modified = False
        with events_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                event_data: dict[str, Any] = json.loads(stripped)
                if not event_data.get("event_id"):
                    event_data["event_id"] = _generate_event_id(
                        event_data.get("session_id", ""),
                        event_data.get("ts", 0),
                    )
                    modified = True
                lines.append(json.dumps(event_data, ensure_ascii=False))

        if modified:
            # Atomic write via tmp file
            tmp_path = events_path.with_suffix(".tmp")
            tmp_path.write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
            tmp_path.rename(events_path)

    logger.info("Migrated storage from v1 to v2")


def reindex(
    storage: NdjsonStorage,
    ollama_client: OllamaClient,
) -> dict[str, int]:
    """Clear the embedding index and rebuild from all chunks.

    Returns dict with 'chunks' and 'sessions' counts.
    """
    index_dir = storage.root / "index"
    index_dir.mkdir(exist_ok=True)
    index = VectorIndex(index_dir)

    # Clear existing index
    index.clear()

    total_chunks = 0
    session_set: set[str] = set()

    for chunk in storage.iter_all_chunks():
        session_set.add(chunk.session_id)
        vectors = ollama_client.embed([chunk.text])
        metadata = {
            "chunk_id": chunk.chunk_id,
            "session_id": chunk.session_id,
            "text": chunk.text[:500],
            "ts_start": chunk.ts_start,
            "ts_end": chunk.ts_end,
        }
        index.add(vectors[0], metadata)
        total_chunks += 1

    return {"chunks": total_chunks, "sessions": len(session_set)}
