"""Tests for mb.pack excerpt collection."""

from __future__ import annotations

import json
from pathlib import Path

from mb.pack import _collect_recent_excerpts


def test_collect_recent_excerpts_returns_chunks(tmp_path: Path) -> None:
    """_collect_recent_excerpts returns data when chunks.jsonl exists."""
    session_dir = tmp_path / "sessions" / "s1"
    session_dir.mkdir(parents=True)
    chunks = [
        {"chunk_id": "s1-0", "session_id": "s1", "text": "first chunk with enough text here", "ts_start": 1.0, "ts_end": 2.0, "quality_score": 0.8},
        {"chunk_id": "s1-1", "session_id": "s1", "text": "second chunk with enough text here", "ts_start": 3.0, "ts_end": 4.0, "quality_score": 0.8},
    ]
    with (session_dir / "chunks.jsonl").open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    result = _collect_recent_excerpts(tmp_path)

    assert len(result) == 2
    # Most recent first (ts_end descending)
    assert result[0]["chunk_id"] == "s1-1"
    assert result[1]["chunk_id"] == "s1-0"


def test_collect_recent_excerpts_empty_without_sessions(tmp_path: Path) -> None:
    """_collect_recent_excerpts returns empty list when no sessions exist."""
    result = _collect_recent_excerpts(tmp_path)
    assert result == []


def test_collect_recent_excerpts_multiple_sessions(tmp_path: Path) -> None:
    """_collect_recent_excerpts merges chunks from multiple sessions sorted by ts_end."""
    for sid, ts in [("s1", 1.0), ("s2", 5.0)]:
        session_dir = tmp_path / "sessions" / sid
        session_dir.mkdir(parents=True)
        chunk = {
            "chunk_id": f"{sid}-0", "session_id": sid,
            "text": "normal data here that is long enough to pass filter",
            "ts_end": ts, "quality_score": 0.8,
        }
        (session_dir / "chunks.jsonl").write_text(json.dumps(chunk) + "\n")

    result = _collect_recent_excerpts(tmp_path)

    assert len(result) == 2
    assert result[0]["session_id"] == "s2"  # ts_end=5.0 first
    assert result[1]["session_id"] == "s1"  # ts_end=1.0 second


def test_collect_recent_excerpts_filters_low_quality(tmp_path: Path) -> None:
    """_collect_recent_excerpts filters out chunks with quality_score below threshold."""
    session_dir = tmp_path / "sessions" / "s1"
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

    result = _collect_recent_excerpts(tmp_path)

    assert len(result) == 1
    assert result[0]["chunk_id"] == "s1-0"


def test_collect_recent_excerpts_filters_short_chunks(tmp_path: Path) -> None:
    """_collect_recent_excerpts filters out chunks shorter than min_length."""
    session_dir = tmp_path / "sessions" / "s1"
    session_dir.mkdir(parents=True)
    chunks = [
        {"chunk_id": "s1-0", "text": "This is a long enough chunk with real content", "ts_end": 2.0, "quality_score": 0.8},
        {"chunk_id": "s1-1", "text": "tiny", "ts_end": 3.0, "quality_score": 1.0},
        {"chunk_id": "s1-2", "text": "ab", "ts_end": 4.0, "quality_score": 1.0},
    ]
    with (session_dir / "chunks.jsonl").open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    result = _collect_recent_excerpts(tmp_path)

    assert len(result) == 1
    assert result[0]["chunk_id"] == "s1-0"


def test_collect_recent_excerpts_backward_compat(tmp_path: Path) -> None:
    """Chunks without quality_score field get scored on the fly."""
    session_dir = tmp_path / "sessions" / "s1"
    session_dir.mkdir(parents=True)
    chunks = [
        {"chunk_id": "s1-0", "text": "hello world readable content that passes both filters", "ts_end": 1.0},
        {"chunk_id": "s1-1", "text": "......!!!......!!!......!!!...", "ts_end": 2.0},
    ]
    with (session_dir / "chunks.jsonl").open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    result = _collect_recent_excerpts(tmp_path)

    # Only the chunk with real text should pass
    assert len(result) == 1
    assert result[0]["chunk_id"] == "s1-0"
