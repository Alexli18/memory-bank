"""Tests for mb.chunker."""

from __future__ import annotations

import json
from pathlib import Path

from mb.chunker import _quality_score, chunk_all_sessions, chunk_session
from mb.models import Chunk
from mb.store import NdjsonStorage


def _make_storage(tmp_path: Path) -> NdjsonStorage:
    """Create a minimal NdjsonStorage for tests."""
    root = tmp_path / ".memory-bank"
    root.mkdir(exist_ok=True)
    (root / "sessions").mkdir(exist_ok=True)
    (root / "config.json").write_text('{"version": "1.0"}', encoding="utf-8")
    return NdjsonStorage(root)


def _write_session_with_events(
    storage: NdjsonStorage, session_id: str, events: list[dict]
) -> None:
    """Write events.jsonl + meta.json for a session."""
    session_dir = storage.root / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    meta = {"session_id": session_id, "command": ["test"], "cwd": "/tmp",
            "started_at": 1.0, "ended_at": None, "exit_code": None}
    (session_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with (session_dir / "events.jsonl").open("w", encoding="utf-8") as f:
        for ev in events:
            ev.setdefault("session_id", session_id)
            ev.setdefault("role", "terminal")
            f.write(json.dumps(ev) + "\n")


def test_chunks_created_for_unchunked_session(tmp_path: Path) -> None:
    """chunk_all_sessions creates chunks.jsonl for sessions that lack it."""
    storage = _make_storage(tmp_path)
    _write_session_with_events(storage, "s1", [
        {"stream": "stdout", "content": "hello world", "ts": 1.0},
    ])

    chunk_all_sessions(storage)

    chunks_path = storage.root / "sessions" / "s1" / "chunks.jsonl"
    assert chunks_path.exists()
    lines = [json.loads(line) for line in chunks_path.read_text().strip().splitlines()]
    assert len(lines) >= 1
    assert "hello world" in lines[0]["text"]


def test_skips_already_chunked_session(tmp_path: Path) -> None:
    """chunk_all_sessions does not overwrite existing chunks.jsonl."""
    storage = _make_storage(tmp_path)
    _write_session_with_events(storage, "s1", [
        {"stream": "stdout", "content": "hello", "ts": 1.0},
    ])
    # Pre-create chunks.jsonl with known content
    chunks_path = storage.root / "sessions" / "s1" / "chunks.jsonl"
    chunks_path.write_text('{"chunk_id":"existing"}\n')

    chunk_all_sessions(storage)

    # Should not be overwritten
    content = chunks_path.read_text()
    assert '"existing"' in content


def test_skips_session_without_events(tmp_path: Path) -> None:
    """chunk_all_sessions skips directories without events.jsonl."""
    storage = _make_storage(tmp_path)
    session_dir = storage.root / "sessions" / "s1"
    session_dir.mkdir(parents=True)
    meta = {"session_id": "s1", "command": ["test"], "cwd": "/tmp",
            "started_at": 1.0, "ended_at": None, "exit_code": None}
    (session_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    chunk_all_sessions(storage)

    assert not (session_dir / "chunks.jsonl").exists()


def test_noop_when_no_sessions_dir(tmp_path: Path) -> None:
    """chunk_all_sessions is a no-op when sessions/ doesn't exist."""
    root = tmp_path / ".memory-bank"
    root.mkdir()
    (root / "config.json").write_text('{"version": "1.0"}')
    storage = NdjsonStorage(root)
    chunk_all_sessions(storage)  # should not raise


def test_multiple_sessions(tmp_path: Path) -> None:
    """chunk_all_sessions processes multiple sessions."""
    storage = _make_storage(tmp_path)
    for name in ("s1", "s2", "s3"):
        _write_session_with_events(storage, name, [
            {"stream": "stdout", "content": f"data from {name}", "ts": 1.0},
        ])

    chunk_all_sessions(storage)

    for name in ("s1", "s2", "s3"):
        assert (storage.root / "sessions" / name / "chunks.jsonl").exists()


# --- _quality_score tests ---


def test_quality_score_empty_string() -> None:
    assert _quality_score("") == 0.0


def test_quality_score_whitespace_only() -> None:
    assert _quality_score("   \n\t  ") == 0.0


def test_quality_score_noise_only() -> None:
    """Pure noise characters (after stripping) score very low."""
    assert _quality_score("---!!!...") == 0.0


def test_quality_score_normal_text() -> None:
    """Normal English text should score well above 0.5."""
    score = _quality_score("Hello world this is normal text")
    assert score > 0.5


def test_quality_score_mixed() -> None:
    """Text with some non-alnum characters gets a mid-range score."""
    score = _quality_score("a = b + c")
    assert 0.2 < score < 0.5


# --- chunk_session noise stripping & quality_score tests ---


def test_chunk_session_strips_noise(tmp_path: Path) -> None:
    """chunk_session strips terminal UI noise from event content."""
    storage = _make_storage(tmp_path)
    _write_session_with_events(storage, "s1", [
        {"stream": "stdout", "content": "hello\u2500\u2500\u2500world", "ts": 1.0},
    ])

    chunks = chunk_session(storage, "s1")

    assert len(chunks) >= 1
    assert isinstance(chunks[0], Chunk)
    assert "\u2500\u2500\u2500" not in chunks[0].text
    assert "helloworld" in chunks[0].text


def test_chunk_session_includes_quality_score(tmp_path: Path) -> None:
    """chunk_session adds quality_score field to each chunk."""
    storage = _make_storage(tmp_path)
    _write_session_with_events(storage, "s1", [
        {"stream": "stdout", "content": "normal readable text", "ts": 1.0},
    ])

    chunks = chunk_session(storage, "s1")

    assert len(chunks) >= 1
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].quality_score > 0.5
