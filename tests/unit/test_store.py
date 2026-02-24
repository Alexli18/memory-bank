"""Tests for NdjsonStorage — CRUD operations on sessions, events, chunks, state."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from mb.models import Chunk, Event, ProjectState, SessionMeta
from mb.store import NdjsonStorage, MbStorageError


@pytest.fixture()
def storage(tmp_path: Path) -> NdjsonStorage:
    """Initialized NdjsonStorage backed by tmp_path."""
    _, s = NdjsonStorage.init(tmp_path / ".memory-bank")
    return s


# -- Factory methods --------------------------------------------------------


class TestInit:
    def test_init_creates_structure(self, tmp_path: Path) -> None:
        root = tmp_path / ".memory-bank"
        created, s = NdjsonStorage.init(root)
        assert created is True
        assert (root / "config.json").exists()
        assert (root / "sessions").is_dir()
        assert (root / "index").is_dir()
        assert (root / "state").is_dir()

    def test_init_idempotent(self, tmp_path: Path) -> None:
        root = tmp_path / ".memory-bank"
        NdjsonStorage.init(root)
        created, _ = NdjsonStorage.init(root)
        assert created is False

    def test_open_raises_if_not_initialized(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not initialized"):
            NdjsonStorage.open(tmp_path / ".memory-bank")

    def test_open_success(self, tmp_path: Path) -> None:
        root = tmp_path / ".memory-bank"
        NdjsonStorage.init(root)
        s = NdjsonStorage.open(root)
        assert s.root == root


# -- SessionStore -----------------------------------------------------------


class TestSessionStore:
    def test_create_session_returns_meta(self, storage: NdjsonStorage) -> None:
        meta = storage.create_session(["python", "hello.py"], cwd="/tmp")
        assert isinstance(meta, SessionMeta)
        assert meta.command == ["python", "hello.py"]
        assert meta.cwd == "/tmp"
        assert meta.exit_code is None

    def test_create_session_writes_meta_json(self, storage: NdjsonStorage) -> None:
        meta = storage.create_session(["ls"])
        meta_path = storage.root / "sessions" / meta.session_id / "meta.json"
        assert meta_path.exists()
        data = json.loads(meta_path.read_text())
        assert data["session_id"] == meta.session_id

    def test_create_session_with_source(self, storage: NdjsonStorage) -> None:
        meta = storage.create_session(["claude"], source="hook", create_events=False)
        data = json.loads(
            (storage.root / "sessions" / meta.session_id / "meta.json").read_text()
        )
        assert data["source"] == "hook"

    def test_create_session_creates_events_file(self, storage: NdjsonStorage) -> None:
        meta = storage.create_session(["ls"])
        events_path = storage.root / "sessions" / meta.session_id / "events.jsonl"
        assert events_path.exists()

    def test_create_session_no_events(self, storage: NdjsonStorage) -> None:
        meta = storage.create_session(["claude"], create_events=False)
        events_path = storage.root / "sessions" / meta.session_id / "events.jsonl"
        assert not events_path.exists()

    def test_finalize_session(self, storage: NdjsonStorage) -> None:
        meta = storage.create_session(["ls"])
        storage.finalize_session(meta.session_id, exit_code=0)
        data = json.loads(
            (storage.root / "sessions" / meta.session_id / "meta.json").read_text()
        )
        assert data["exit_code"] == 0
        assert data["ended_at"] is not None

    def test_delete_session(self, storage: NdjsonStorage) -> None:
        meta = storage.create_session(["ls"])
        assert storage.delete_session(meta.session_id) is True
        assert not (storage.root / "sessions" / meta.session_id).exists()

    def test_delete_session_not_found(self, storage: NdjsonStorage) -> None:
        assert storage.delete_session("nonexistent") is False

    def test_list_sessions_empty(self, storage: NdjsonStorage) -> None:
        assert storage.list_sessions() == []

    def test_list_sessions_sorted(self, storage: NdjsonStorage) -> None:
        storage.create_session(["a"])
        time.sleep(0.01)
        storage.create_session(["b"])
        sessions = storage.list_sessions()
        assert len(sessions) == 2
        # Most recent first
        assert sessions[0].started_at >= sessions[1].started_at

    def test_read_meta(self, storage: NdjsonStorage) -> None:
        meta = storage.create_session(["ls"])
        result = storage.read_meta(meta.session_id)
        assert result is not None
        assert result.session_id == meta.session_id

    def test_read_meta_not_found(self, storage: NdjsonStorage) -> None:
        assert storage.read_meta("nonexistent") is None


# -- EventStore -------------------------------------------------------------


class TestEventStore:
    def test_write_and_read_events(self, storage: NdjsonStorage) -> None:
        meta = storage.create_session(["ls"])
        storage.write_event(meta.session_id, "stdout", "terminal", "hello", ts=1.0)
        storage.write_event(meta.session_id, "stdin", "user", "world", ts=2.0)
        events = storage.read_events(meta.session_id)
        assert len(events) == 2
        assert all(isinstance(e, Event) for e in events)
        assert events[0].content == "hello"
        assert events[1].content == "world"

    def test_read_events_empty(self, storage: NdjsonStorage) -> None:
        meta = storage.create_session(["ls"])
        events = storage.read_events(meta.session_id)
        assert events == []

    def test_read_events_no_file(self, storage: NdjsonStorage) -> None:
        meta = storage.create_session(["claude"], create_events=False)
        events = storage.read_events(meta.session_id)
        assert events == []


# -- ChunkStore -------------------------------------------------------------


class TestChunkStore:
    def _make_chunks(self, session_id: str, count: int = 2) -> list[Chunk]:
        return [
            Chunk(
                chunk_id=f"{session_id}-{i}",
                session_id=session_id,
                index=i,
                text=f"chunk text {i}",
                ts_start=float(i),
                ts_end=float(i + 1),
                token_estimate=10,
                quality_score=0.8,
            )
            for i in range(count)
        ]

    def test_write_and_read_chunks(self, storage: NdjsonStorage) -> None:
        meta = storage.create_session(["ls"])
        chunks = self._make_chunks(meta.session_id)
        storage.write_chunks(meta.session_id, chunks)
        result = storage.read_chunks(meta.session_id)
        assert len(result) == 2
        assert all(isinstance(c, Chunk) for c in result)
        assert result[0].text == "chunk text 0"
        assert result[1].text == "chunk text 1"

    def test_read_chunks_empty(self, storage: NdjsonStorage) -> None:
        meta = storage.create_session(["ls"])
        assert storage.read_chunks(meta.session_id) == []

    def test_has_chunks(self, storage: NdjsonStorage) -> None:
        meta = storage.create_session(["ls"])
        assert storage.has_chunks(meta.session_id) is False
        storage.write_chunks(meta.session_id, self._make_chunks(meta.session_id))
        assert storage.has_chunks(meta.session_id) is True

    def test_iter_all_chunks(self, storage: NdjsonStorage) -> None:
        m1 = storage.create_session(["a"])
        m2 = storage.create_session(["b"])
        storage.write_chunks(m1.session_id, self._make_chunks(m1.session_id, 2))
        storage.write_chunks(m2.session_id, self._make_chunks(m2.session_id, 3))
        all_chunks = list(storage.iter_all_chunks())
        assert len(all_chunks) == 5

    def test_write_chunks_round_trip(self, storage: NdjsonStorage) -> None:
        meta = storage.create_session(["ls"])
        original = self._make_chunks(meta.session_id)
        storage.write_chunks(meta.session_id, original)
        loaded = storage.read_chunks(meta.session_id)
        for orig, loaded_chunk in zip(original, loaded):
            assert orig.to_dict() == loaded_chunk.to_dict()


# -- StateStore -------------------------------------------------------------


class TestStateStore:
    def _make_state(self) -> ProjectState:
        return ProjectState(
            summary="Test project summary",
            decisions=[{"id": "D1", "statement": "Use Python", "rationale": "Simple"}],
            constraints=["No new deps"],
            tasks=[{"id": "T1", "status": "in_progress"}],
            updated_at=1700000000.0,
            source_sessions=["session-1"],
        )

    def test_save_and_load_state(self, storage: NdjsonStorage) -> None:
        state = self._make_state()
        storage.save_state(state)
        loaded = storage.load_state()
        assert loaded is not None
        assert loaded.summary == state.summary
        assert loaded.decisions == state.decisions
        assert loaded.constraints == state.constraints
        assert loaded.updated_at == state.updated_at

    def test_load_state_none(self, storage: NdjsonStorage) -> None:
        assert storage.load_state() is None

    def test_is_stale_no_state(self, storage: NdjsonStorage) -> None:
        assert storage.is_stale() is False

    def test_is_stale_after_new_chunks(self, storage: NdjsonStorage) -> None:
        state = self._make_state()
        storage.save_state(state)

        meta = storage.create_session(["ls"])
        chunks = [
            Chunk(
                chunk_id=f"{meta.session_id}-0",
                session_id=meta.session_id,
                index=0,
                text="test",
                ts_start=0.0,
                ts_end=1.0,
                token_estimate=1,
                quality_score=0.5,
            )
        ]
        time.sleep(0.01)
        storage.write_chunks(meta.session_id, chunks)
        assert storage.is_stale() is True

    def test_state_round_trip(self, storage: NdjsonStorage) -> None:
        state = self._make_state()
        storage.save_state(state)
        loaded = storage.load_state()
        assert loaded is not None
        assert state.to_dict() == loaded.to_dict()


# -- Config helpers ---------------------------------------------------------


class TestConfigHelpers:
    def test_read_config(self, storage: NdjsonStorage) -> None:
        config = storage.read_config()
        assert "version" in config
        assert "ollama" in config

    def test_write_config(self, storage: NdjsonStorage) -> None:
        config = storage.read_config()
        config["custom_key"] = "value"
        storage.write_config(config)
        loaded = storage.read_config()
        assert loaded["custom_key"] == "value"

    def test_read_config_corrupt(self, storage: NdjsonStorage) -> None:
        (storage.root / "config.json").write_text("not json", encoding="utf-8")
        with pytest.raises(MbStorageError, match="Corrupt"):
            storage.read_config()


# -- Hooks/Import state helpers ---------------------------------------------


class TestStateHelpers:
    def test_hooks_state_round_trip(self, storage: NdjsonStorage) -> None:
        state = storage.load_hooks_state()
        assert state == {"sessions": {}}
        state["sessions"]["abc"] = {"mb_session_id": "123"}
        storage.save_hooks_state(state)
        loaded = storage.load_hooks_state()
        assert loaded["sessions"]["abc"]["mb_session_id"] == "123"

    def test_import_state_round_trip(self, storage: NdjsonStorage) -> None:
        state = storage.load_import_state()
        assert state == {"imported": {}}
        state["imported"]["uuid1"] = "session-1"
        storage.save_import_state(state)
        loaded = storage.load_import_state()
        assert loaded["imported"]["uuid1"] == "session-1"


# -- On-disk format verification -------------------------------------------


class TestOnDiskFormat:
    def test_session_file_layout(self, storage: NdjsonStorage) -> None:
        meta = storage.create_session(["python", "hello.py"], source="pty")
        sid = meta.session_id
        session_dir = storage.root / "sessions" / sid

        assert (session_dir / "meta.json").exists()
        assert (session_dir / "events.jsonl").exists()

        storage.write_event(sid, "stdout", "terminal", "hello", ts=1.0)

        chunks = [
            Chunk(
                chunk_id=f"{sid}-0",
                session_id=sid,
                index=0,
                text="hello",
                ts_start=1.0,
                ts_end=1.0,
                token_estimate=1,
                quality_score=0.8,
            )
        ]
        storage.write_chunks(sid, chunks)
        assert (session_dir / "chunks.jsonl").exists()

        # Verify NDJSON format (one JSON per line)
        lines = (session_dir / "events.jsonl").read_text().strip().split("\n")
        for line in lines:
            json.loads(line)  # Should not raise

        chunk_lines = (session_dir / "chunks.jsonl").read_text().strip().split("\n")
        for line in chunk_lines:
            json.loads(line)  # Should not raise

    def test_state_file_layout(self, storage: NdjsonStorage) -> None:
        state = ProjectState(
            summary="test", decisions=[], constraints=[],
            tasks=[], updated_at=0.0, source_sessions=[],
        )
        storage.save_state(state)
        state_path = storage.root / "state" / "state.json"
        assert state_path.exists()
        data = json.loads(state_path.read_text())
        assert data["summary"] == "test"


# -- Protocol conformance --------------------------------------------------


class TestProtocolConformance:
    def test_session_store(self, storage: NdjsonStorage) -> None:
        from mb.store import SessionStore
        assert isinstance(storage, SessionStore)

    def test_event_store(self, storage: NdjsonStorage) -> None:
        from mb.store import EventStore
        assert isinstance(storage, EventStore)

    def test_chunk_store(self, storage: NdjsonStorage) -> None:
        from mb.store import ChunkStore
        assert isinstance(storage, ChunkStore)

    def test_state_store(self, storage: NdjsonStorage) -> None:
        from mb.store import StateStore
        assert isinstance(storage, StateStore)


# -- Redaction on write ----------------------------------------------------


class TestEventRedaction:
    def test_write_event_redacts_secrets(self, storage: NdjsonStorage) -> None:
        meta = storage.create_session(["ls"])
        storage.write_event(
            meta.session_id, "stdout", "terminal",
            "key is AKIAIOSFODNN7EXAMPLE here", ts=1.0,
        )
        events = storage.read_events(meta.session_id)
        assert len(events) == 1
        assert "AKIAIOSFODNN7EXAMPLE" not in events[0].content
        assert "[REDACTED:AWS_KEY]" in events[0].content

    def test_write_event_clean_text_unchanged(self, storage: NdjsonStorage) -> None:
        meta = storage.create_session(["ls"])
        storage.write_event(
            meta.session_id, "stdout", "terminal", "Hello, world!", ts=1.0,
        )
        events = storage.read_events(meta.session_id)
        assert events[0].content == "Hello, world!"

    def test_write_event_disabled_redactor(self, tmp_path: Path) -> None:
        from mb.redactor import Redactor, RedactorConfig
        _, s = NdjsonStorage.init(tmp_path / ".memory-bank")
        s = NdjsonStorage(s.root, redactor=Redactor(RedactorConfig(enabled=False)))
        meta = s.create_session(["ls"])
        s.write_event(
            meta.session_id, "stdout", "terminal",
            "AKIAIOSFODNN7EXAMPLE", ts=1.0,
        )
        events = s.read_events(meta.session_id)
        assert events[0].content == "AKIAIOSFODNN7EXAMPLE"
