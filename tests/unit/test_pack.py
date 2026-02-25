"""Tests for mb.pack excerpt collection."""

from __future__ import annotations

import json
from pathlib import Path

from mb.models import Chunk
from mb.pack import _collect_recent_excerpts
from mb.store import NdjsonStorage


def _make_storage(tmp_path: Path) -> NdjsonStorage:
    """Create a minimal NdjsonStorage for tests."""
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "sessions").mkdir(exist_ok=True)
    (tmp_path / "config.json").write_text('{"version": "1.0"}', encoding="utf-8")
    return NdjsonStorage(tmp_path)


def test_collect_recent_excerpts_returns_chunks(tmp_path: Path) -> None:
    """_collect_recent_excerpts returns data when chunks.jsonl exists."""
    root = tmp_path / ".mb"
    storage = _make_storage(root)
    session_dir = root / "sessions" / "s1"
    session_dir.mkdir(parents=True)
    chunks = [
        {"chunk_id": "s1-0", "session_id": "s1", "text": "The authentication module handles JWT token validation and refresh", "ts_start": 1.0, "ts_end": 2.0, "quality_score": 0.8},
        {"chunk_id": "s1-1", "session_id": "s1", "text": "Database migration scripts run automatically on deployment startup", "ts_start": 3.0, "ts_end": 4.0, "quality_score": 0.8},
    ]
    with (session_dir / "chunks.jsonl").open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    result = _collect_recent_excerpts(storage)

    assert len(result) == 2
    assert all(isinstance(c, Chunk) for c in result)
    # Most recent first (ts_end descending)
    assert result[0].chunk_id == "s1-1"
    assert result[1].chunk_id == "s1-0"


def test_collect_recent_excerpts_empty_without_sessions(tmp_path: Path) -> None:
    """_collect_recent_excerpts returns empty list when no sessions exist."""
    root = tmp_path / ".mb"
    storage = _make_storage(root)
    result = _collect_recent_excerpts(storage)
    assert result == []


def test_collect_recent_excerpts_multiple_sessions(tmp_path: Path) -> None:
    """_collect_recent_excerpts merges chunks from multiple sessions sorted by ts_end."""
    root = tmp_path / ".mb"
    storage = _make_storage(root)
    texts = {
        "s1": "Refactored the authentication middleware to use async request handlers",
        "s2": "Added PostgreSQL connection pooling with configurable maximum limits",
    }
    for sid, ts in [("s1", 1.0), ("s2", 5.0)]:
        session_dir = root / "sessions" / sid
        session_dir.mkdir(parents=True)
        chunk = {
            "chunk_id": f"{sid}-0", "session_id": sid,
            "text": texts[sid],
            "ts_end": ts, "quality_score": 0.8,
        }
        (session_dir / "chunks.jsonl").write_text(json.dumps(chunk) + "\n")

    result = _collect_recent_excerpts(storage)

    assert len(result) == 2
    assert result[0].session_id == "s2"  # ts_end=5.0 first
    assert result[1].session_id == "s1"  # ts_end=1.0 second


def test_collect_recent_excerpts_filters_low_quality(tmp_path: Path) -> None:
    """_collect_recent_excerpts filters out chunks with quality_score below threshold."""
    root = tmp_path / ".mb"
    storage = _make_storage(root)
    session_dir = root / "sessions" / "s1"
    session_dir.mkdir(parents=True)
    chunks = [
        {"chunk_id": "s1-0", "text": "good readable text that is long enough", "ts_end": 1.0, "quality_score": 0.8},
        {"chunk_id": "s1-1", "text": "---...!!!---...!!!---...!!!---", "ts_end": 2.0, "quality_score": 0.0},
        {"chunk_id": "s1-2", "text": "  ", "ts_end": 3.0, "quality_score": 0.0},
        {"chunk_id": "s1-3", "text": "mostly whitespace with a few words here", "ts_end": 4.0, "quality_score": 0.25},
    ]
    with (session_dir / "chunks.jsonl").open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    result = _collect_recent_excerpts(storage)

    assert len(result) == 1
    assert result[0].chunk_id == "s1-0"


def test_collect_recent_excerpts_filters_short_chunks(tmp_path: Path) -> None:
    """_collect_recent_excerpts filters out chunks shorter than min_length."""
    root = tmp_path / ".mb"
    storage = _make_storage(root)
    session_dir = root / "sessions" / "s1"
    session_dir.mkdir(parents=True)
    chunks = [
        {"chunk_id": "s1-0", "text": "This is a long enough chunk with real content", "ts_end": 2.0, "quality_score": 0.8},
        {"chunk_id": "s1-1", "text": "tiny", "ts_end": 3.0, "quality_score": 1.0},
        {"chunk_id": "s1-2", "text": "ab", "ts_end": 4.0, "quality_score": 1.0},
    ]
    with (session_dir / "chunks.jsonl").open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    result = _collect_recent_excerpts(storage)

    assert len(result) == 1
    assert result[0].chunk_id == "s1-0"


def test_collect_recent_excerpts_backward_compat(tmp_path: Path) -> None:
    """Chunks without quality_score field get scored on the fly."""
    root = tmp_path / ".mb"
    storage = _make_storage(root)
    session_dir = root / "sessions" / "s1"
    session_dir.mkdir(parents=True)
    chunks = [
        {"chunk_id": "s1-0", "text": "hello world readable content that passes both filters", "ts_end": 1.0},
        {"chunk_id": "s1-1", "text": "......!!!......!!!......!!!...", "ts_end": 2.0},
    ]
    with (session_dir / "chunks.jsonl").open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    result = _collect_recent_excerpts(storage)

    # Only the chunk with real text should pass
    assert len(result) == 1
    assert result[0].chunk_id == "s1-0"


def _make_chunk(i: int, ts_end: float) -> dict:
    """Helper: create a valid chunk dict with unique content."""
    import hashlib
    h = hashlib.sha256(str(i).encode()).hexdigest()
    return {
        "chunk_id": f"s1-{i}",
        "session_id": "s1",
        "text": f"unique content {h} end marker",
        "ts_end": ts_end,
        "quality_score": 0.8,
    }


def test_collect_recent_excerpts_max_excerpts(tmp_path: Path) -> None:
    """With 500 chunks and limit=10, only 10 are returned."""
    root = tmp_path / ".mb"
    storage = _make_storage(root)
    session_dir = root / "sessions" / "s1"
    session_dir.mkdir(parents=True)
    with (session_dir / "chunks.jsonl").open("w") as f:
        for i in range(500):
            f.write(json.dumps(_make_chunk(i, float(i))) + "\n")

    result = _collect_recent_excerpts(storage, max_excerpts=10)

    assert len(result) == 10


def test_collect_recent_excerpts_keeps_most_recent(tmp_path: Path) -> None:
    """Bounded collection keeps the most recent chunks."""
    root = tmp_path / ".mb"
    storage = _make_storage(root)
    session_dir = root / "sessions" / "s1"
    session_dir.mkdir(parents=True)
    with (session_dir / "chunks.jsonl").open("w") as f:
        for i in range(100):
            f.write(json.dumps(_make_chunk(i, float(i))) + "\n")

    result = _collect_recent_excerpts(storage, max_excerpts=5)

    assert len(result) == 5
    ts_ends = [c.ts_end for c in result]
    # Should be the 5 most recent (95..99), sorted descending
    assert ts_ends == [99.0, 98.0, 97.0, 96.0, 95.0]


def test_collect_recent_excerpts_fewer_than_max(tmp_path: Path) -> None:
    """When fewer chunks than limit exist, all are returned."""
    root = tmp_path / ".mb"
    storage = _make_storage(root)
    session_dir = root / "sessions" / "s1"
    session_dir.mkdir(parents=True)
    with (session_dir / "chunks.jsonl").open("w") as f:
        for i in range(3):
            f.write(json.dumps(_make_chunk(i, float(i))) + "\n")

    result = _collect_recent_excerpts(storage, max_excerpts=100)

    assert len(result) == 3
