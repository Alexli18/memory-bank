"""Tests for mb.chunker."""

from __future__ import annotations

import json
from pathlib import Path

from mb.chunker import _quality_score, chunk_all_sessions, chunk_session


def _write_events(session_dir: Path, events: list[dict]) -> None:
    """Helper: write events.jsonl into a session directory."""
    session_dir.mkdir(parents=True, exist_ok=True)
    events_path = session_dir / "events.jsonl"
    with events_path.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def test_chunks_created_for_unchunked_session(tmp_path: Path) -> None:
    """chunk_all_sessions creates chunks.jsonl for sessions that lack it."""
    session_dir = tmp_path / "sessions" / "s1"
    _write_events(session_dir, [
        {"stream": "stdout", "content": "hello world", "ts": 1.0},
    ])

    chunk_all_sessions(tmp_path)

    chunks_path = session_dir / "chunks.jsonl"
    assert chunks_path.exists()
    lines = [json.loads(line) for line in chunks_path.read_text().strip().splitlines()]
    assert len(lines) >= 1
    assert "hello world" in lines[0]["text"]


def test_skips_already_chunked_session(tmp_path: Path) -> None:
    """chunk_all_sessions does not overwrite existing chunks.jsonl."""
    session_dir = tmp_path / "sessions" / "s1"
    _write_events(session_dir, [
        {"stream": "stdout", "content": "hello", "ts": 1.0},
    ])
    # Pre-create chunks.jsonl with known content
    chunks_path = session_dir / "chunks.jsonl"
    chunks_path.write_text('{"chunk_id":"existing"}\n')

    chunk_all_sessions(tmp_path)

    # Should not be overwritten
    content = chunks_path.read_text()
    assert '"existing"' in content


def test_skips_session_without_events(tmp_path: Path) -> None:
    """chunk_all_sessions skips directories without events.jsonl."""
    session_dir = tmp_path / "sessions" / "s1"
    session_dir.mkdir(parents=True)

    chunk_all_sessions(tmp_path)

    assert not (session_dir / "chunks.jsonl").exists()


def test_noop_when_no_sessions_dir(tmp_path: Path) -> None:
    """chunk_all_sessions is a no-op when sessions/ doesn't exist."""
    chunk_all_sessions(tmp_path)  # should not raise


def test_multiple_sessions(tmp_path: Path) -> None:
    """chunk_all_sessions processes multiple sessions."""
    for name in ("s1", "s2", "s3"):
        _write_events(tmp_path / "sessions" / name, [
            {"stream": "stdout", "content": f"data from {name}", "ts": 1.0},
        ])

    chunk_all_sessions(tmp_path)

    for name in ("s1", "s2", "s3"):
        assert (tmp_path / "sessions" / name / "chunks.jsonl").exists()


# --- _quality_score tests ---


def test_quality_score_empty_string() -> None:
    assert _quality_score("") == 0.0


def test_quality_score_whitespace_only() -> None:
    assert _quality_score("   \n\t  ") == 0.0


def test_quality_score_noise_only() -> None:
    """Pure noise characters (after stripping) score very low."""
    # Punctuation-only text has 0 alnum chars
    assert _quality_score("---!!!...") == 0.0


def test_quality_score_normal_text() -> None:
    """Normal English text should score well above 0.5."""
    score = _quality_score("Hello world this is normal text")
    assert score > 0.5


def test_quality_score_mixed() -> None:
    """Text with some non-alnum characters gets a mid-range score."""
    score = _quality_score("a = b + c")
    # 3 alnum out of 9 chars
    assert 0.2 < score < 0.5


# --- chunk_session noise stripping & quality_score tests ---


def test_chunk_session_strips_noise(tmp_path: Path) -> None:
    """chunk_session strips terminal UI noise from event content."""
    session_dir = tmp_path / "sessions" / "s1"
    _write_events(session_dir, [
        {"stream": "stdout", "content": "hello───world", "ts": 1.0},
    ])

    chunks = chunk_session(session_dir / "events.jsonl")

    assert len(chunks) >= 1
    assert "───" not in chunks[0]["text"]
    assert "helloworld" in chunks[0]["text"]


def test_chunk_session_includes_quality_score(tmp_path: Path) -> None:
    """chunk_session adds quality_score field to each chunk."""
    session_dir = tmp_path / "sessions" / "s1"
    _write_events(session_dir, [
        {"stream": "stdout", "content": "normal readable text", "ts": 1.0},
    ])

    chunks = chunk_session(session_dir / "events.jsonl")

    assert len(chunks) >= 1
    assert "quality_score" in chunks[0]
    assert chunks[0]["quality_score"] > 0.5
