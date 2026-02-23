"""Tests for mb.claude_adapter."""

from __future__ import annotations

import json
from pathlib import Path

from mb.claude_adapter import (
    Turn,
    _extract_assistant_text,
    _extract_user_text,
    encode_project_dir,
    extract_turns,
    is_claude_session,
    chunk_claude_session,
)


# --- encode_project_dir ---


def test_encode_project_dir_standard() -> None:
    assert encode_project_dir("/Users/alex/Desktop/memory-bank") == "-Users-alex-Desktop-memory-bank"


def test_encode_project_dir_root() -> None:
    assert encode_project_dir("/") == "-"


def test_encode_project_dir_trailing_slash() -> None:
    assert encode_project_dir("/Users/alex/project/") == "-Users-alex-project"


# --- is_claude_session ---


def test_is_claude_session_true() -> None:
    assert is_claude_session({"command": ["claude"]}) is True


def test_is_claude_session_with_path() -> None:
    assert is_claude_session({"command": ["/usr/local/bin/claude"]}) is True


def test_is_claude_session_false() -> None:
    assert is_claude_session({"command": ["bash"]}) is False


def test_is_claude_session_empty() -> None:
    assert is_claude_session({"command": []}) is False
    assert is_claude_session({}) is False


# --- _extract_user_text ---


def test_extract_user_text_string() -> None:
    assert _extract_user_text("hello world") == "hello world"


def test_extract_user_text_list_with_text() -> None:
    content = [{"type": "text", "text": "search for nginx config"}]
    assert _extract_user_text(content) == "search for nginx config"


def test_extract_user_text_skips_tool_result() -> None:
    content = [
        {"type": "tool_result", "content": "file contents here"},
        {"type": "text", "text": "now fix the bug"},
    ]
    assert _extract_user_text(content) == "now fix the bug"


def test_extract_user_text_skips_command_wrapper() -> None:
    assert _extract_user_text("<command-name>/commit</command-name>") is None


def test_extract_user_text_skips_interrupt() -> None:
    content = [{"type": "text", "text": "[Request interrupted by user]"}]
    assert _extract_user_text(content) is None


def test_extract_user_text_tool_result_only() -> None:
    content = [{"type": "tool_result", "content": "some output"}]
    assert _extract_user_text(content) is None


def test_extract_user_text_skips_task_notification() -> None:
    content = "<task-notification>\n<task-id>abc</task-id>\n</task-notification>"
    assert _extract_user_text(content) is None


def test_extract_user_text_skips_local_command() -> None:
    assert _extract_user_text("<local-command-stdout>output</local-command-stdout>") is None


def test_extract_user_text_skips_system_reminder() -> None:
    assert _extract_user_text("<system-reminder>reminder text</system-reminder>") is None


def test_extract_user_text_skips_system_tags_in_list() -> None:
    content = [
        {"type": "text", "text": "<task-notification>stuff</task-notification>"},
        {"type": "text", "text": "actual question here"},
    ]
    assert _extract_user_text(content) == "actual question here"


def test_extract_user_text_skips_bash_input() -> None:
    assert _extract_user_text("<bash-input>uv run mb sessions</bash-input>") is None


def test_extract_user_text_skips_bash_stdout() -> None:
    assert _extract_user_text("<bash-stdout>SESSION  COMMAND</bash-stdout>") is None


def test_extract_user_text_skips_bash_stderr() -> None:
    assert _extract_user_text("<bash-stderr>error output</bash-stderr>") is None


def test_extract_user_text_skips_local_command_caveat() -> None:
    assert _extract_user_text("<local-command-caveat>Caveat: ...</local-command-caveat>") is None


def test_extract_user_text_skips_bash_tags_in_list() -> None:
    content = [
        {"type": "text", "text": "<bash-input>mb sessions</bash-input>"},
        {"type": "text", "text": "<bash-stdout>output here</bash-stdout>"},
        {"type": "text", "text": "actual user question"},
    ]
    assert _extract_user_text(content) == "actual user question"


def test_extract_user_text_none() -> None:
    assert _extract_user_text(None) is None


# --- _extract_assistant_text ---


def test_extract_assistant_text_from_list() -> None:
    content = [{"type": "text", "text": "Here is the fix"}]
    assert _extract_assistant_text(content) == "Here is the fix"


def test_extract_assistant_text_skips_tool_use() -> None:
    content = [
        {"type": "tool_use", "name": "Read", "input": {}},
        {"type": "text", "text": "I found the issue"},
    ]
    assert _extract_assistant_text(content) == "I found the issue"


def test_extract_assistant_text_skips_thinking() -> None:
    content = [{"type": "thinking", "thinking": "Let me think..."}]
    assert _extract_assistant_text(content) is None


def test_extract_assistant_text_string() -> None:
    assert _extract_assistant_text("plain response") == "plain response"


def test_extract_assistant_text_none() -> None:
    assert _extract_assistant_text(None) is None


# --- extract_turns ---


def _write_claude_jsonl(path: Path, messages: list[dict]) -> None:
    """Write Claude-style JSONL for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")


def test_extract_turns_basic(tmp_path: Path) -> None:
    """Single complete turn: user text + assistant text."""
    session_file = tmp_path / "session.jsonl"
    _write_claude_jsonl(session_file, [
        {
            "type": "user",
            "message": {"role": "user", "content": "explain decorators"},
            "timestamp": "2026-02-23T10:00:00Z",
        },
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": [
                {"type": "text", "text": "Decorators are functions that wrap other functions."},
            ]},
        },
    ])

    turns = extract_turns(session_file)
    assert len(turns) == 1
    assert turns[0].user_message == "explain decorators"
    assert turns[0].assistant_texts == ["Decorators are functions that wrap other functions."]


def test_extract_turns_skips_sidechain(tmp_path: Path) -> None:
    """Sidechain messages are excluded."""
    session_file = tmp_path / "session.jsonl"
    _write_claude_jsonl(session_file, [
        {
            "type": "user",
            "message": {"role": "user", "content": "main question"},
        },
        {
            "type": "assistant",
            "isSidechain": True,
            "message": {"role": "assistant", "content": [
                {"type": "text", "text": "subagent response"},
            ]},
        },
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": [
                {"type": "text", "text": "main response"},
            ]},
        },
    ])

    turns = extract_turns(session_file)
    assert len(turns) == 1
    assert turns[0].assistant_texts == ["main response"]


def test_extract_turns_skips_meta(tmp_path: Path) -> None:
    """isMeta messages (expanded skill prompts) are excluded."""
    session_file = tmp_path / "session.jsonl"
    _write_claude_jsonl(session_file, [
        {
            "type": "user",
            "message": {"role": "user", "content": "/commit"},
        },
        {
            "type": "user",
            "isMeta": True,
            "message": {"role": "user", "content": [
                {"type": "text", "text": "## Expanded commit instructions..."},
            ]},
        },
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": [
                {"type": "text", "text": "Committing changes..."},
            ]},
        },
    ])

    turns = extract_turns(session_file)
    assert len(turns) == 1
    assert turns[0].user_message == "/commit"


def test_extract_turns_multiple(tmp_path: Path) -> None:
    """Multiple turns are detected correctly."""
    session_file = tmp_path / "session.jsonl"
    _write_claude_jsonl(session_file, [
        {"type": "user", "message": {"role": "user", "content": "first question"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "first answer"},
        ]}},
        {"type": "user", "message": {"role": "user", "content": "second question"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "second answer"},
        ]}},
    ])

    turns = extract_turns(session_file)
    assert len(turns) == 2
    assert turns[0].user_message == "first question"
    assert turns[1].user_message == "second question"


def test_extract_turns_skips_tool_result_only_messages(tmp_path: Path) -> None:
    """Tool result messages don't start new turns."""
    session_file = tmp_path / "session.jsonl"
    _write_claude_jsonl(session_file, [
        {"type": "user", "message": {"role": "user", "content": "read the file"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Read", "input": {"path": "foo.py"}},
        ]}},
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "content": "file contents here"},
        ]}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "I see the file contains..."},
        ]}},
    ])

    turns = extract_turns(session_file)
    assert len(turns) == 1
    assert turns[0].assistant_texts == ["I see the file contains..."]


def test_extract_turns_handles_file_snapshot(tmp_path: Path) -> None:
    """file-history-snapshot lines are ignored."""
    session_file = tmp_path / "session.jsonl"
    _write_claude_jsonl(session_file, [
        {"type": "file-history-snapshot", "snapshot": {}},
        {"type": "user", "message": {"role": "user", "content": "hello"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "hi there"},
        ]}},
    ])

    turns = extract_turns(session_file)
    assert len(turns) == 1


# --- Turn.full_text ---


def test_turn_full_text() -> None:
    turn = Turn(turn_number=1, user_message="how?", assistant_texts=["like this", "also this"])
    text = turn.full_text
    assert "User: how?" in text
    assert "Assistant: like this" in text
    assert "Assistant: also this" in text


# --- chunk_claude_session ---


def test_chunk_claude_session_generates_chunks(tmp_path: Path) -> None:
    """End-to-end: meta.json + Claude JSONL -> chunks.jsonl."""
    # Create session dir with meta.json
    session_dir = tmp_path / "sessions" / "20260223-100000-abcd"
    session_dir.mkdir(parents=True)

    meta = {
        "session_id": "20260223-100000-abcd",
        "command": ["claude"],
        "cwd": "/tmp/test-project",
        "started_at": 1000000.0,
        "ended_at": 1000100.0,
    }
    (session_dir / "meta.json").write_text(json.dumps(meta))
    (session_dir / "events.jsonl").write_text("")  # empty, we use Claude JSONL

    # Create fake Claude project dir
    claude_dir = tmp_path / ".claude" / "projects" / "-tmp-test-project"
    claude_dir.mkdir(parents=True)
    claude_session = claude_dir / "test-session.jsonl"
    _write_claude_jsonl(claude_session, [
        {"type": "user", "message": {"role": "user", "content": "explain pytest fixtures"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "Pytest fixtures provide reusable test setup and teardown."},
        ]}},
        {"type": "user", "message": {"role": "user", "content": "show me an example"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "Here is an example:\n\n@pytest.fixture\ndef client():\n    return TestClient(app)"},
        ]}},
    ])

    # Monkey-patch home dir for find_claude_session_file
    original_home = Path.home

    def fake_home():
        return tmp_path

    try:
        Path.home = staticmethod(fake_home)
        chunks = chunk_claude_session(session_dir)
    finally:
        Path.home = original_home

    assert len(chunks) >= 2
    assert chunks[0]["source"] == "claude_native"
    assert "pytest fixtures" in chunks[0]["text"].lower()
    assert chunks[0]["quality_score"] > 0.3

    # Verify chunks.jsonl was written
    chunks_path = session_dir / "chunks.jsonl"
    assert chunks_path.exists()
    written = [json.loads(line) for line in chunks_path.read_text().strip().splitlines()]
    assert len(written) == len(chunks)
