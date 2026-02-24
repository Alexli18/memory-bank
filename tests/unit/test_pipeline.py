"""Tests for mb.pipeline — Source/Processor plugin system."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mb.store import NdjsonStorage


# ---- Protocol compliance tests ----


def test_source_protocol_compliance() -> None:
    """Any class with ingest(storage) -> list[str] satisfies Source."""
    from mb.pipeline import Source

    class MySource:
        def ingest(self, storage: NdjsonStorage) -> list[str]:
            return ["sid-1"]

    assert isinstance(MySource(), Source)


def test_processor_protocol_compliance() -> None:
    """Any class with process(storage, session_ids) -> None satisfies Processor."""
    from mb.pipeline import Processor

    class MyProcessor:
        def process(self, storage: NdjsonStorage, session_ids: list[str]) -> None:
            pass

    assert isinstance(MyProcessor(), Processor)


# ---- ProcessorPipeline tests ----


def test_pipeline_calls_processors_in_order() -> None:
    """ProcessorPipeline calls processors in the order they were added."""
    from mb.pipeline import ProcessorPipeline

    call_order: list[str] = []

    class ProcA:
        def process(self, storage: NdjsonStorage, session_ids: list[str]) -> None:
            call_order.append("A")

    class ProcB:
        def process(self, storage: NdjsonStorage, session_ids: list[str]) -> None:
            call_order.append("B")

    pipeline = ProcessorPipeline([ProcA(), ProcB()])
    mock_storage = MagicMock(spec=NdjsonStorage)
    pipeline.run(mock_storage, ["sid-1"])

    assert call_order == ["A", "B"]


def test_pipeline_passes_session_ids() -> None:
    """ProcessorPipeline passes session_ids to each processor."""
    from mb.pipeline import ProcessorPipeline

    received_ids: list[list[str]] = []

    class Recorder:
        def process(self, storage: NdjsonStorage, session_ids: list[str]) -> None:
            received_ids.append(list(session_ids))

    pipeline = ProcessorPipeline([Recorder()])
    mock_storage = MagicMock(spec=NdjsonStorage)
    pipeline.run(mock_storage, ["sid-1", "sid-2"])

    assert received_ids == [["sid-1", "sid-2"]]


def test_pipeline_empty_processors() -> None:
    """Pipeline with no processors runs without error."""
    from mb.pipeline import ProcessorPipeline

    pipeline = ProcessorPipeline()
    mock_storage = MagicMock(spec=NdjsonStorage)
    pipeline.run(mock_storage, ["sid-1"])  # no error


def test_pipeline_empty_session_ids() -> None:
    """Pipeline with empty session_ids still calls processors."""
    from mb.pipeline import ProcessorPipeline

    called = False

    class Marker:
        def process(self, storage: NdjsonStorage, session_ids: list[str]) -> None:
            nonlocal called
            called = True

    pipeline = ProcessorPipeline([Marker()])
    mock_storage = MagicMock(spec=NdjsonStorage)
    pipeline.run(mock_storage, [])

    assert called


# ---- ChunkProcessor tests ----


def test_chunk_processor_chunks_sessions(storage: NdjsonStorage, sample_session: str) -> None:
    """ChunkProcessor calls chunk_session for sessions without chunks."""
    from mb.pipeline import ChunkProcessor

    # Remove existing chunks to trigger processing
    chunks_path = storage.root / "sessions" / sample_session / "chunks.jsonl"
    chunks_path.unlink()

    with patch("mb.pipeline.chunk_session") as mock_chunk:
        proc = ChunkProcessor()
        proc.process(storage, [sample_session])

        mock_chunk.assert_called_once_with(storage, sample_session)


def test_chunk_processor_skips_chunked(storage: NdjsonStorage, sample_session: str) -> None:
    """ChunkProcessor skips sessions that already have chunks."""
    from mb.pipeline import ChunkProcessor

    with patch("mb.pipeline.chunk_session") as mock_chunk:
        proc = ChunkProcessor()
        proc.process(storage, [sample_session])

        mock_chunk.assert_not_called()


def test_chunk_processor_force(storage: NdjsonStorage, sample_session: str) -> None:
    """ChunkProcessor with force=True re-chunks even if chunks exist."""
    from mb.pipeline import ChunkProcessor

    with patch("mb.pipeline.chunk_session") as mock_chunk:
        proc = ChunkProcessor(force=True)
        proc.process(storage, [sample_session])

        mock_chunk.assert_called_once_with(storage, sample_session)


# ---- EmbedProcessor tests ----


def test_embed_processor_calls_build_index(storage: NdjsonStorage) -> None:
    """EmbedProcessor calls build_index with storage and ollama_client."""
    from mb.pipeline import EmbedProcessor

    mock_client = MagicMock()

    with patch("mb.pipeline.build_index") as mock_build:
        proc = EmbedProcessor(mock_client)
        proc.process(storage, ["sid-1"])

        mock_build.assert_called_once_with(storage, mock_client)


# ---- PtySource tests ----


def test_pty_source_calls_run_session() -> None:
    """PtySource calls run_session and captures exit_code and session_id."""
    from mb.pipeline import PtySource

    mock_storage = MagicMock(spec=NdjsonStorage)

    with patch("mb.pipeline.run_session", return_value=(0, "sid-pty-1")) as mock_run:
        source = PtySource(["echo", "hello"])
        session_ids = source.ingest(mock_storage)

        mock_run.assert_called_once_with(["echo", "hello"], mock_storage)
        assert session_ids == ["sid-pty-1"]
        assert source.exit_code == 0
        assert source.session_id == "sid-pty-1"


def test_pty_source_captures_nonzero_exit() -> None:
    """PtySource captures non-zero exit codes."""
    from mb.pipeline import PtySource

    mock_storage = MagicMock(spec=NdjsonStorage)

    with patch("mb.pipeline.run_session", return_value=(1, "sid-pty-2")):
        source = PtySource(["false"])
        source.ingest(mock_storage)

        assert source.exit_code == 1


# ---- HookSource tests ----


def test_hook_source_creates_session(tmp_path: Path) -> None:
    """HookSource creates a new session from transcript."""
    from mb.pipeline import HookSource

    storage_root = tmp_path / ".memory-bank"
    storage_root.mkdir()
    (storage_root / "sessions").mkdir()
    (storage_root / "config.json").write_text('{"version": "1.0"}')
    storage = NdjsonStorage(storage_root)

    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript)

    source = HookSource(str(transcript), str(tmp_path), "claude-uuid-1")
    session_ids = source.ingest(storage)

    assert len(session_ids) == 1
    sid = session_ids[0]

    # Verify chunks were written
    assert storage.has_chunks(sid)

    # Verify hooks_state was updated
    state = storage.load_hooks_state()
    assert "claude-uuid-1" in state["sessions"]
    assert state["sessions"]["claude-uuid-1"]["mb_session_id"] == sid


def test_hook_source_updates_on_change(tmp_path: Path) -> None:
    """HookSource updates session when transcript grows."""
    from mb.pipeline import HookSource

    storage_root = tmp_path / ".memory-bank"
    storage_root.mkdir()
    (storage_root / "sessions").mkdir()
    (storage_root / "config.json").write_text('{"version": "1.0"}')
    storage = NdjsonStorage(storage_root)

    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript)

    # First call
    source1 = HookSource(str(transcript), str(tmp_path), "claude-uuid-upd")
    ids1 = source1.ingest(storage)
    assert len(ids1) == 1

    # Append more content
    with transcript.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "type": "user",
            "message": {"role": "user", "content": "show more"},
        }) + "\n")
        f.write(json.dumps({
            "type": "assistant",
            "message": {"role": "assistant", "content": [
                {"type": "text", "text": "Here is more info."},
            ]},
        }) + "\n")

    # Second call with changed transcript
    source2 = HookSource(str(transcript), str(tmp_path), "claude-uuid-upd")
    ids2 = source2.ingest(storage)
    assert len(ids2) == 1
    assert ids2[0] == ids1[0]  # Same session_id


def test_hook_source_skips_unchanged(tmp_path: Path) -> None:
    """HookSource skips processing if transcript size unchanged."""
    from mb.pipeline import HookSource

    storage_root = tmp_path / ".memory-bank"
    storage_root.mkdir()
    (storage_root / "sessions").mkdir()
    (storage_root / "config.json").write_text('{"version": "1.0"}')
    storage = NdjsonStorage(storage_root)

    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript)

    # First call
    source1 = HookSource(str(transcript), str(tmp_path), "claude-uuid-2")
    ids1 = source1.ingest(storage)
    assert len(ids1) == 1

    # Second call with same transcript
    source2 = HookSource(str(transcript), str(tmp_path), "claude-uuid-2")
    ids2 = source2.ingest(storage)
    assert ids2 == []  # No-op


def test_hook_source_empty_transcript(tmp_path: Path) -> None:
    """HookSource returns empty list for empty transcript."""
    from mb.pipeline import HookSource

    storage_root = tmp_path / ".memory-bank"
    storage_root.mkdir()
    (storage_root / "sessions").mkdir()
    (storage_root / "config.json").write_text('{"version": "1.0"}')
    storage = NdjsonStorage(storage_root)

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("")

    source = HookSource(str(transcript), str(tmp_path), "claude-uuid-3")
    session_ids = source.ingest(storage)
    assert session_ids == []


def test_hook_source_missing_transcript(tmp_path: Path) -> None:
    """HookSource returns empty list for missing transcript."""
    from mb.pipeline import HookSource

    storage_root = tmp_path / ".memory-bank"
    storage_root.mkdir()
    (storage_root / "sessions").mkdir()
    (storage_root / "config.json").write_text('{"version": "1.0"}')
    storage = NdjsonStorage(storage_root)

    source = HookSource("/nonexistent/path.jsonl", str(tmp_path), "claude-uuid-4")
    session_ids = source.ingest(storage)
    assert session_ids == []


# ---- ImportSource tests ----


def test_import_source_imports_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ImportSource imports Claude sessions and returns session_ids."""
    from mb.pipeline import ImportSource

    storage_root = tmp_path / ".memory-bank"
    storage_root.mkdir()
    (storage_root / "sessions").mkdir()
    (storage_root / "config.json").write_text('{"version": "1.0"}')
    storage = NdjsonStorage(storage_root)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = str(storage_root.parent)

    from mb.claude_adapter import encode_project_dir

    project_dir = tmp_path / ".claude" / "projects" / encode_project_dir(cwd)
    project_dir.mkdir(parents=True)
    _write_claude_session(project_dir / "session-aaa.jsonl")

    source = ImportSource()
    session_ids = source.ingest(storage)

    assert len(session_ids) == 1
    assert source.imported == 1
    assert source.skipped == 0


def test_import_source_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ImportSource with dry_run=True doesn't create sessions."""
    from mb.pipeline import ImportSource

    storage_root = tmp_path / ".memory-bank"
    storage_root.mkdir()
    (storage_root / "sessions").mkdir()
    (storage_root / "config.json").write_text('{"version": "1.0"}')
    storage = NdjsonStorage(storage_root)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = str(storage_root.parent)

    from mb.claude_adapter import encode_project_dir

    project_dir = tmp_path / ".claude" / "projects" / encode_project_dir(cwd)
    project_dir.mkdir(parents=True)
    _write_claude_session(project_dir / "session-bbb.jsonl")

    source = ImportSource(dry_run=True)
    session_ids = source.ingest(storage)

    assert session_ids == []
    assert source.imported == 1
    assert source.skipped == 0

    # No sessions created
    sessions_dir = storage_root / "sessions"
    session_dirs = [d for d in sessions_dir.iterdir() if d.is_dir()]
    assert len(session_dirs) == 0


def test_import_source_skips_already_imported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ImportSource skips already-imported sessions."""
    from mb.pipeline import ImportSource

    storage_root = tmp_path / ".memory-bank"
    storage_root.mkdir()
    (storage_root / "sessions").mkdir()
    (storage_root / "config.json").write_text('{"version": "1.0"}')
    storage = NdjsonStorage(storage_root)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = str(storage_root.parent)

    from mb.claude_adapter import encode_project_dir

    project_dir = tmp_path / ".claude" / "projects" / encode_project_dir(cwd)
    project_dir.mkdir(parents=True)
    _write_claude_session(project_dir / "session-ccc.jsonl")

    # First import
    source1 = ImportSource()
    source1.ingest(storage)

    # Second import
    source2 = ImportSource()
    session_ids = source2.ingest(storage)

    assert session_ids == []
    assert source2.imported == 0
    assert source2.skipped == 1


# ---- Helpers ----


def _write_transcript(path: Path) -> None:
    """Write a minimal valid Claude transcript."""
    messages = [
        {"type": "user", "message": {"role": "user", "content": "explain decorators"}},
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Decorators wrap functions to extend behavior."},
                ],
            },
        },
    ]
    with path.open("w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")


def _write_claude_session(path: Path) -> None:
    """Write a minimal Claude Code session JSONL."""
    messages = [
        {
            "type": "user",
            "message": {"content": "Hello, help me"},
            "timestamp": "2026-01-15T10:00:00Z",
        },
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "Sure, I can help."}]},
            "timestamp": "2026-01-15T10:00:05Z",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")
