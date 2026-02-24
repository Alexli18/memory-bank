"""Tests for mb.models — round-trip serialization and enum behavior."""

from __future__ import annotations

from mb.models import (
    Chunk,
    Event,
    EventSource,
    PackFormat,
    ProjectState,
    SearchResult,
    SessionMeta,
    quality_score,
)


# --- Enums ---


def test_event_source_values() -> None:
    assert EventSource.PTY.value == "pty"
    assert EventSource.HOOK.value == "hook"
    assert EventSource.IMPORT.value == "import"


def test_event_source_from_string() -> None:
    assert EventSource("pty") is EventSource.PTY
    assert EventSource("hook") is EventSource.HOOK
    assert EventSource("import") is EventSource.IMPORT


def test_pack_format_values() -> None:
    assert PackFormat.XML.value == "xml"
    assert PackFormat.JSON.value == "json"
    assert PackFormat.MARKDOWN.value == "md"


# --- quality_score ---


def test_quality_score_empty() -> None:
    assert quality_score("") == 0.0


def test_quality_score_whitespace() -> None:
    assert quality_score("   \n\t  ") == 0.0


def test_quality_score_normal_text() -> None:
    score = quality_score("Hello world this is normal text")
    assert score > 0.5


def test_quality_score_noise() -> None:
    assert quality_score("---!!!...") == 0.0


# --- Event ---


def test_event_round_trip() -> None:
    data = {
        "event_id": "abc123",
        "ts": 1.5,
        "session_id": "s1",
        "stream": "stdout",
        "role": "terminal",
        "content": "hello",
        "source": "pty",
        "meta": {"key": "val"},
    }
    event = Event.from_dict(data)
    assert event.event_id == "abc123"
    assert event.source == EventSource.PTY
    assert event.meta == {"key": "val"}
    roundtrip = Event.from_dict(event.to_dict())
    assert roundtrip == event


def test_event_tolerates_missing_event_id() -> None:
    data = {
        "ts": 2.0,
        "session_id": "s1",
        "stream": "stdout",
        "role": "terminal",
        "content": "data",
    }
    event = Event.from_dict(data)
    assert event.event_id  # generated, non-empty
    assert event.source is None


def test_event_to_dict_omits_none_source() -> None:
    data = {
        "event_id": "e1",
        "ts": 1.0,
        "session_id": "s1",
        "stream": "stdout",
        "role": "terminal",
        "content": "x",
    }
    event = Event.from_dict(data)
    d = event.to_dict()
    assert "source" not in d
    assert "meta" not in d  # empty meta omitted


def test_event_unknown_source_becomes_none() -> None:
    data = {
        "event_id": "e1",
        "ts": 1.0,
        "session_id": "s1",
        "stream": "stdout",
        "role": "terminal",
        "content": "x",
        "source": "unknown_source",
    }
    event = Event.from_dict(data)
    assert event.source is None


# --- SessionMeta ---


def test_session_meta_round_trip() -> None:
    data = {
        "session_id": "20260224-120000-abcd",
        "command": ["python", "hello.py"],
        "cwd": "/tmp",
        "started_at": 1700000000.0,
        "ended_at": 1700000060.0,
        "exit_code": 0,
        "source": "hook",
    }
    meta = SessionMeta.from_dict(data)
    assert meta.session_id == "20260224-120000-abcd"
    assert meta.source == EventSource.HOOK
    roundtrip = SessionMeta.from_dict(meta.to_dict())
    assert roundtrip == meta


def test_session_meta_defaults() -> None:
    meta = SessionMeta.from_dict({"session_id": "s1", "command": ["ls"], "cwd": "/"})
    assert meta.ended_at is None
    assert meta.exit_code is None
    assert meta.source is None


def test_session_meta_to_dict_omits_none_source() -> None:
    meta = SessionMeta.from_dict({
        "session_id": "s1",
        "command": ["ls"],
        "cwd": "/",
        "started_at": 1.0,
    })
    d = meta.to_dict()
    assert "source" not in d


def test_session_meta_string_command() -> None:
    """from_dict handles command as a string (legacy)."""
    meta = SessionMeta.from_dict({
        "session_id": "s1",
        "command": "ls",
        "cwd": "/",
    })
    assert meta.command == ["ls"]


# --- Chunk ---


def test_chunk_round_trip() -> None:
    data = {
        "chunk_id": "s1-0",
        "session_id": "s1",
        "index": 0,
        "text": "hello world",
        "ts_start": 1.0,
        "ts_end": 2.0,
        "token_estimate": 3,
        "quality_score": 0.85,
    }
    chunk = Chunk.from_dict(data)
    assert chunk.chunk_id == "s1-0"
    roundtrip = Chunk.from_dict(chunk.to_dict())
    assert roundtrip == chunk


def test_chunk_preserves_extra_fields() -> None:
    data = {
        "chunk_id": "s1-0",
        "session_id": "s1",
        "index": 0,
        "text": "text",
        "ts_start": 1.0,
        "ts_end": 2.0,
        "token_estimate": 1,
        "quality_score": 0.5,
        "source": "claude_native",
        "turn_number": 3,
    }
    chunk = Chunk.from_dict(data)
    d = chunk.to_dict()
    assert d["source"] == "claude_native"
    assert d["turn_number"] == 3
    roundtrip = Chunk.from_dict(d)
    assert roundtrip == chunk


# --- SearchResult ---


def test_search_result_round_trip() -> None:
    data = {
        "chunk_id": "s1-0",
        "session_id": "s1",
        "index": 0,
        "text": "result text",
        "ts_start": 1.0,
        "ts_end": 2.0,
        "token_estimate": 3,
        "quality_score": 0.8,
        "score": 0.95,
    }
    result = SearchResult.from_dict(data)
    assert result.score == 0.95
    roundtrip = SearchResult.from_dict(result.to_dict())
    assert roundtrip == result


# --- ProjectState ---


def test_project_state_round_trip() -> None:
    data = {
        "summary": "A test project.",
        "decisions": [{"id": "D1", "statement": "Use Python", "rationale": "Fast dev"}],
        "constraints": ["No new deps"],
        "tasks": [{"id": "T1", "status": "in_progress"}],
        "updated_at": 1700000000.0,
        "source_sessions": ["s1", "s2"],
    }
    state = ProjectState.from_dict(data)
    assert state.summary == "A test project."
    assert len(state.decisions) == 1
    roundtrip = ProjectState.from_dict(state.to_dict())
    assert roundtrip == state


def test_project_state_defaults() -> None:
    state = ProjectState.from_dict({})
    assert state.summary == ""
    assert state.decisions == []
    assert state.constraints == []
    assert state.tasks == []
    assert state.updated_at == 0.0
    assert state.source_sessions == []
