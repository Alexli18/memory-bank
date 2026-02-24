"""Tests for mb.importer — retroactive Claude Code session import."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _make_claude_jsonl(path: Path, turns: list[dict] | None = None) -> None:
    """Write a minimal Claude Code JSONL session file."""
    if turns is None:
        turns = [
            {
                "type": "user",
                "message": {"content": "Hello, help me with Python"},
                "timestamp": "2026-01-15T10:00:00Z",
            },
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Sure, I can help with Python."}]},
                "timestamp": "2026-01-15T10:00:05Z",
            },
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for turn in turns:
            f.write(json.dumps(turn) + "\n")


def _claude_project_dir(tmp_path: Path, cwd: str) -> Path:
    """Create and return the Claude projects dir for a given cwd."""
    from mb.claude_adapter import encode_project_dir

    project_dir = tmp_path / ".claude" / "projects" / encode_project_dir(cwd)
    project_dir.mkdir(parents=True)
    return project_dir


# --- discover_claude_sessions ---


def test_discover_no_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty project dir returns empty list."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = str(tmp_path / "my-project")
    _claude_project_dir(tmp_path, cwd)

    from mb.importer import discover_claude_sessions

    result = discover_claude_sessions(cwd)
    assert result == []


def test_discover_skips_agent_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """agent-*.jsonl files are excluded from discovery."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = str(tmp_path / "my-project")
    project_dir = _claude_project_dir(tmp_path, cwd)

    # Create regular session and agent session
    _make_claude_jsonl(project_dir / "abc-123.jsonl")
    _make_claude_jsonl(project_dir / "agent-sub1.jsonl")

    from mb.importer import discover_claude_sessions

    result = discover_claude_sessions(cwd)
    assert len(result) == 1
    assert result[0].name == "abc-123.jsonl"


def test_discover_no_claude_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing ~/.claude/projects/ returns empty list."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from mb.importer import discover_claude_sessions

    result = discover_claude_sessions(str(tmp_path / "no-project"))
    assert result == []


# --- import_claude_sessions ---


def test_import_creates_sessions(
    storage_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full round-trip: JSONL -> session -> chunks.jsonl."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = str(storage_root.parent)
    project_dir = _claude_project_dir(tmp_path, cwd)

    _make_claude_jsonl(project_dir / "session-aaa.jsonl")

    from mb.importer import import_claude_sessions

    imported, skipped = import_claude_sessions(storage_root)

    assert imported == 1
    assert skipped == 0

    # Verify session was created
    sessions_dir = storage_root / "sessions"
    session_dirs = [d for d in sessions_dir.iterdir() if d.is_dir()]
    assert len(session_dirs) == 1

    session_dir = session_dirs[0]

    # Check meta.json
    meta = json.loads((session_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["source"] == "import"
    assert meta["command"] == ["claude"]
    assert meta["started_at"] > 0

    # Check chunks.jsonl exists and has content
    chunks_path = session_dir / "chunks.jsonl"
    assert chunks_path.exists()
    chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(chunks) >= 1
    assert "Python" in chunks[0]["text"]


def test_import_dedup_skips_already_imported(
    storage_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Second import call skips already-imported sessions."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = str(storage_root.parent)
    project_dir = _claude_project_dir(tmp_path, cwd)

    _make_claude_jsonl(project_dir / "session-bbb.jsonl")

    from mb.importer import import_claude_sessions

    imported1, skipped1 = import_claude_sessions(storage_root)
    assert imported1 == 1
    assert skipped1 == 0

    imported2, skipped2 = import_claude_sessions(storage_root)
    assert imported2 == 0
    assert skipped2 == 1


def test_import_dry_run(
    storage_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dry run does not create sessions."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = str(storage_root.parent)
    project_dir = _claude_project_dir(tmp_path, cwd)

    _make_claude_jsonl(project_dir / "session-ccc.jsonl")

    from mb.importer import import_claude_sessions

    imported, skipped = import_claude_sessions(storage_root, dry_run=True)

    assert imported == 1
    assert skipped == 0

    # No sessions should be created
    sessions_dir = storage_root / "sessions"
    session_dirs = [d for d in sessions_dir.iterdir() if d.is_dir()]
    assert len(session_dirs) == 0


def test_import_empty_turns(
    storage_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """JSONL without user messages is skipped."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = str(storage_root.parent)
    project_dir = _claude_project_dir(tmp_path, cwd)

    # Write JSONL with only non-message lines
    empty_session = project_dir / "session-empty.jsonl"
    empty_session.parent.mkdir(parents=True, exist_ok=True)
    empty_session.write_text(
        json.dumps({"type": "system", "message": {}}) + "\n",
        encoding="utf-8",
    )

    from mb.importer import import_claude_sessions

    imported, skipped = import_claude_sessions(storage_root)

    assert imported == 0
    assert skipped == 1


def test_import_state_persistence(
    storage_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """import_state.json tracks imported session UUIDs."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = str(storage_root.parent)
    project_dir = _claude_project_dir(tmp_path, cwd)

    _make_claude_jsonl(project_dir / "session-ddd.jsonl")

    from mb.importer import import_claude_sessions, load_import_state

    import_claude_sessions(storage_root)

    state = load_import_state(storage_root)
    assert "session-ddd" in state["imported"]
