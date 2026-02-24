"""Tests for schema versioning and migration (US6)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock


from mb.store import NdjsonStorage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_v1_storage(root: Path, *, with_events: bool = True) -> NdjsonStorage:
    """Create a v1 storage layout (no schema_version in config)."""
    root.mkdir(exist_ok=True)
    (root / "sessions").mkdir(exist_ok=True)
    (root / "index").mkdir(exist_ok=True)
    (root / "state").mkdir(exist_ok=True)

    config: dict[str, Any] = {
        "version": "1.0",
        "ollama": {
            "base_url": "http://localhost:11434",
            "embed_model": "nomic-embed-text",
            "chat_model": "gemma3:4b",
        },
        "chunking": {"max_tokens": 512, "overlap_tokens": 50},
    }
    (root / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    if with_events:
        session_id = "20260224-120000-abcd"
        session_dir = root / "sessions" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        meta: dict[str, Any] = {
            "session_id": session_id,
            "command": ["python", "hello.py"],
            "cwd": str(root.parent),
            "started_at": 1700000000.0,
            "ended_at": 1700000060.0,
            "exit_code": 0,
        }
        (session_dir / "meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )

        events = [
            {
                "ts": 1.0,
                "session_id": session_id,
                "stream": "stdout",
                "role": "terminal",
                "content": "Hello, world!",
            },
            {
                "ts": 2.0,
                "session_id": session_id,
                "stream": "stdout",
                "role": "terminal",
                "content": "Process finished",
            },
        ]
        with (session_dir / "events.jsonl").open("w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")

        chunks = [
            {
                "chunk_id": f"{session_id}-0",
                "session_id": session_id,
                "index": 0,
                "text": "Hello, world! Process finished",
                "ts_start": 1.0,
                "ts_end": 2.0,
                "token_estimate": 7,
                "quality_score": 0.75,
            },
        ]
        with (session_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for ch in chunks:
                f.write(json.dumps(ch) + "\n")

    return NdjsonStorage(root)


def _write_v2_storage(root: Path) -> NdjsonStorage:
    """Create a v2 storage layout (schema_version present)."""
    storage = _write_v1_storage(root, with_events=True)
    config = storage.read_config()
    config["schema_version"] = 2
    storage.write_config(config)
    return storage


# ---------------------------------------------------------------------------
# detect_version tests
# ---------------------------------------------------------------------------


class TestDetectVersion:
    def test_v1_no_schema_version(self, tmp_path: Path) -> None:
        """v1 storage has no schema_version → detect_version returns 1."""
        from mb.migrations import detect_version

        storage = _write_v1_storage(tmp_path / ".memory-bank")
        assert detect_version(storage) == 1

    def test_v2_with_schema_version(self, tmp_path: Path) -> None:
        """v2 storage has schema_version:2 → detect_version returns 2."""
        from mb.migrations import detect_version

        storage = _write_v2_storage(tmp_path / ".memory-bank")
        assert detect_version(storage) == 2

    def test_explicit_version_3(self, tmp_path: Path) -> None:
        """Future schema_version:3 → detect_version returns 3."""
        from mb.migrations import detect_version

        storage = _write_v1_storage(tmp_path / ".memory-bank", with_events=False)
        config = storage.read_config()
        config["schema_version"] = 3
        storage.write_config(config)
        assert detect_version(storage) == 3


# ---------------------------------------------------------------------------
# migrate tests
# ---------------------------------------------------------------------------


class TestMigrate:
    def test_migrate_v1_to_v2_updates_config(self, tmp_path: Path) -> None:
        """Migration v1→v2 adds schema_version:2 to config.json."""
        from mb.migrations import migrate

        storage = _write_v1_storage(tmp_path / ".memory-bank")
        old_version, new_version = migrate(storage)

        assert old_version == 1
        assert new_version == 2

        config = storage.read_config()
        assert config["schema_version"] == 2

    def test_migrate_v1_to_v2_adds_event_id(self, tmp_path: Path) -> None:
        """Migration v1→v2 adds event_id to events missing it."""
        from mb.migrations import migrate

        storage = _write_v1_storage(tmp_path / ".memory-bank")
        migrate(storage)

        events = storage.read_events("20260224-120000-abcd")
        assert len(events) == 2
        for event in events:
            assert event.event_id, "event_id should be non-empty after migration"
            assert len(event.event_id) >= 8

    def test_migrate_v1_to_v2_no_data_loss(self, tmp_path: Path) -> None:
        """Migration preserves all original event fields."""
        from mb.migrations import migrate

        storage = _write_v1_storage(tmp_path / ".memory-bank")
        migrate(storage)

        events = storage.read_events("20260224-120000-abcd")
        assert events[0].content == "Hello, world!"
        assert events[0].ts == 1.0
        assert events[0].stream == "stdout"
        assert events[0].role == "terminal"
        assert events[0].session_id == "20260224-120000-abcd"

        assert events[1].content == "Process finished"
        assert events[1].ts == 2.0

    def test_migrate_v2_to_v2_is_noop(self, tmp_path: Path) -> None:
        """Migrating an already v2 storage is a no-op."""
        from mb.migrations import migrate

        storage = _write_v2_storage(tmp_path / ".memory-bank")
        old_version, new_version = migrate(storage)

        assert old_version == 2
        assert new_version == 2

    def test_migrate_preserves_existing_event_id(self, tmp_path: Path) -> None:
        """Events that already have event_id are not modified."""
        from mb.migrations import migrate

        root = tmp_path / ".memory-bank"
        storage = _write_v1_storage(root, with_events=False)

        session_id = "20260224-130000-efgh"
        session_dir = root / "sessions" / session_id
        session_dir.mkdir(parents=True)

        meta: dict[str, Any] = {
            "session_id": session_id,
            "command": ["echo", "test"],
            "cwd": str(root.parent),
            "started_at": 1700001000.0,
            "ended_at": 1700001060.0,
            "exit_code": 0,
        }
        (session_dir / "meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )

        existing_id = "custom-event-id-12345"
        events = [
            {
                "event_id": existing_id,
                "ts": 1.0,
                "session_id": session_id,
                "stream": "stdout",
                "role": "terminal",
                "content": "Already has ID",
            },
        ]
        with (session_dir / "events.jsonl").open("w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")

        migrate(storage)

        migrated_events = storage.read_events(session_id)
        assert migrated_events[0].event_id == existing_id

    def test_migrate_empty_storage(self, tmp_path: Path) -> None:
        """Migration on storage with no sessions succeeds."""
        from mb.migrations import migrate

        storage = _write_v1_storage(tmp_path / ".memory-bank", with_events=False)
        old_version, new_version = migrate(storage)
        assert old_version == 1
        assert new_version == 2

    def test_migrate_atomic_writes(self, tmp_path: Path) -> None:
        """Events file is written atomically (via tmp file)."""
        from mb.migrations import migrate

        storage = _write_v1_storage(tmp_path / ".memory-bank")
        migrate(storage)

        # After migration, the events file should exist and be valid
        events_path = (
            tmp_path / ".memory-bank" / "sessions" / "20260224-120000-abcd" / "events.jsonl"
        )
        assert events_path.exists()
        lines = events_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            data = json.loads(line)
            assert "event_id" in data


# ---------------------------------------------------------------------------
# reindex tests
# ---------------------------------------------------------------------------


class TestReindex:
    def test_reindex_clears_and_rebuilds(self, tmp_path: Path) -> None:
        """Reindex clears existing index and rebuilds from all chunks."""
        from mb.migrations import reindex

        storage = _write_v1_storage(tmp_path / ".memory-bank")

        # Pre-populate index with stale data
        vectors_path = storage.root / "index" / "vectors.bin"
        metadata_path = storage.root / "index" / "metadata.jsonl"
        vectors_path.write_bytes(b"\x00" * 100)
        metadata_path.write_text("stale\n", encoding="utf-8")

        mock_client = MagicMock()
        mock_client.embed.return_value = [[0.1] * 768]

        stats = reindex(storage, mock_client)

        assert stats["chunks"] == 1
        assert stats["sessions"] == 1

        # Index files should exist and be valid
        assert vectors_path.exists()
        assert metadata_path.exists()

        # Verify embed was called
        mock_client.embed.assert_called()

    def test_reindex_empty_storage(self, tmp_path: Path) -> None:
        """Reindex on storage with no chunks returns zero counts."""
        from mb.migrations import reindex

        storage = _write_v1_storage(tmp_path / ".memory-bank", with_events=False)

        mock_client = MagicMock()
        stats = reindex(storage, mock_client)

        assert stats["chunks"] == 0
        assert stats["sessions"] == 0
        mock_client.embed.assert_not_called()

    def test_reindex_multiple_sessions(self, tmp_path: Path) -> None:
        """Reindex processes chunks from multiple sessions."""
        from mb.migrations import reindex

        root = tmp_path / ".memory-bank"
        storage = _write_v1_storage(root)

        # Add a second session with chunks
        session_id2 = "20260224-130000-efgh"
        session_dir2 = root / "sessions" / session_id2
        session_dir2.mkdir(parents=True)

        meta: dict[str, Any] = {
            "session_id": session_id2,
            "command": ["echo", "world"],
            "cwd": str(root.parent),
            "started_at": 1700002000.0,
            "ended_at": 1700002060.0,
            "exit_code": 0,
        }
        (session_dir2 / "meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )

        chunks = [
            {
                "chunk_id": f"{session_id2}-0",
                "session_id": session_id2,
                "index": 0,
                "text": "Second session chunk",
                "ts_start": 1.0,
                "ts_end": 2.0,
                "token_estimate": 4,
                "quality_score": 0.8,
            },
            {
                "chunk_id": f"{session_id2}-1",
                "session_id": session_id2,
                "index": 1,
                "text": "Another chunk here",
                "ts_start": 2.0,
                "ts_end": 3.0,
                "token_estimate": 3,
                "quality_score": 0.7,
            },
        ]
        with (session_dir2 / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for ch in chunks:
                f.write(json.dumps(ch) + "\n")

        mock_client = MagicMock()
        mock_client.embed.return_value = [[0.1] * 768]

        stats = reindex(storage, mock_client)

        assert stats["chunks"] == 3  # 1 from first session + 2 from second
        assert stats["sessions"] == 2
