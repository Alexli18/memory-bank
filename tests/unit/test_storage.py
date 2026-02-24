"""Tests for mb.storage — session lifecycle, config, and directory management."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pytest

from mb.storage import (
    DEFAULT_CONFIG,
    MEMORY_BANK_DIR,
    MbStorageError,
    create_session,
    delete_session,
    ensure_initialized,
    finalize_session,
    generate_session_id,
    init_storage,
    list_sessions,
    read_config,
    write_config,
    write_event,
)


# -- init_storage --


def test_init_storage_creates_structure(tmp_path: Path) -> None:
    """init_storage creates .memory-bank/ with config.json, subdirs, and .gitignore entry."""
    root = tmp_path / MEMORY_BANK_DIR

    created, storage_path = init_storage(root)

    assert created is True
    assert storage_path == root
    assert (root / "config.json").exists()
    assert (root / "sessions").is_dir()
    assert (root / "index").is_dir()
    assert (root / "state").is_dir()

    # config.json should contain the default config
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    assert config["version"] == DEFAULT_CONFIG["version"]
    assert config["ollama"] == DEFAULT_CONFIG["ollama"]
    assert config["chunking"] == DEFAULT_CONFIG["chunking"]

    # .gitignore entry should exist in the parent (project root)
    gitignore = tmp_path / ".gitignore"
    assert gitignore.exists()
    assert MEMORY_BANK_DIR + "/" in gitignore.read_text(encoding="utf-8").splitlines()


def test_init_storage_idempotent(tmp_path: Path) -> None:
    """Second call to init_storage returns (False, path) without overwriting config."""
    root = tmp_path / MEMORY_BANK_DIR

    created_first, path_first = init_storage(root)
    assert created_first is True

    # Mutate config to verify it is NOT overwritten
    config_path = root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["custom_key"] = "sentinel"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    created_second, path_second = init_storage(root)

    assert created_second is False
    assert path_second == path_first
    # Custom key should still be present (config was not overwritten)
    config_after = json.loads(config_path.read_text(encoding="utf-8"))
    assert config_after["custom_key"] == "sentinel"


# -- ensure_initialized --


def test_ensure_initialized_returns_path(storage_root: Path) -> None:
    """ensure_initialized returns the storage path when initialized."""
    result = ensure_initialized(storage_root)
    assert result == storage_root


def test_ensure_initialized_raises_when_not_initialized(tmp_path: Path) -> None:
    """ensure_initialized raises FileNotFoundError when .memory-bank/ is missing."""
    missing_root = tmp_path / MEMORY_BANK_DIR
    with pytest.raises(FileNotFoundError, match="not initialized"):
        ensure_initialized(missing_root)


# -- read_config / write_config --


def test_read_write_config_roundtrip(storage_root: Path) -> None:
    """write_config followed by read_config preserves all data."""
    custom_config = {
        "version": "2.0",
        "ollama": {"base_url": "http://custom:1234", "embed_model": "custom-model", "chat_model": "custom-chat"},
        "chunking": {"max_tokens": 1024, "overlap_tokens": 100},
        "extra": True,
    }

    write_config(custom_config, root=storage_root)
    result = read_config(root=storage_root)

    assert result == custom_config


# -- generate_session_id --


def test_generate_session_id_format_and_uniqueness() -> None:
    """generate_session_id returns YYYYMMDD-HHMMSS-XXXX format; two calls differ."""
    sid1 = generate_session_id()
    sid2 = generate_session_id()

    pattern = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")
    assert pattern.match(sid1), f"Unexpected format: {sid1}"
    assert pattern.match(sid2), f"Unexpected format: {sid2}"

    # The hex suffix is random so two consecutive IDs should differ
    # (extremely high probability, 1 in 65536 chance of collision)
    assert sid1 != sid2


# -- create_session --


def test_create_session_default(storage_root: Path) -> None:
    """create_session with defaults creates meta.json and events.jsonl."""
    session_meta = create_session(
        command=["python", "app.py"],
        cwd="/tmp/project",
        root=storage_root,
    )

    # create_session returns SessionMeta
    assert session_meta.command == ["python", "app.py"]
    assert session_meta.cwd == "/tmp/project"
    assert isinstance(session_meta.started_at, float)
    assert session_meta.ended_at is None
    assert session_meta.exit_code is None
    assert session_meta.source is None

    session_dir = storage_root / "sessions" / session_meta.session_id
    assert session_dir.is_dir()

    # events.jsonl should be created (empty)
    events_path = session_dir / "events.jsonl"
    assert events_path.exists()
    assert events_path.read_text(encoding="utf-8") == ""


def test_create_session_source_hook_no_events(storage_root: Path) -> None:
    """create_session with source='hook' and create_events=False."""
    from mb.models import EventSource

    session_meta = create_session(
        command=["git", "commit"],
        root=storage_root,
        source="hook",
        create_events=False,
    )

    assert session_meta.source == EventSource.HOOK

    session_dir = storage_root / "sessions" / session_meta.session_id
    # events.jsonl must NOT exist
    assert not (session_dir / "events.jsonl").exists()


# -- write_event --


def test_write_event_appends_jsonl(storage_root: Path) -> None:
    """write_event appends valid JSONL lines to events.jsonl."""
    session_meta = create_session(
        command=["echo", "hi"], root=storage_root
    )
    session_id = session_meta.session_id

    write_event(session_id, "stdout", "terminal", "Hello", ts=1.0, root=storage_root)
    write_event(session_id, "stderr", "terminal", "Warning", ts=2.0, root=storage_root)

    events_path = storage_root / "sessions" / session_id / "events.jsonl"
    lines = [json.loads(line) for line in events_path.read_text(encoding="utf-8").strip().splitlines()]

    assert len(lines) == 2

    assert lines[0]["stream"] == "stdout"
    assert lines[0]["role"] == "terminal"
    assert lines[0]["content"] == "Hello"
    assert lines[0]["ts"] == 1.0
    assert lines[0]["session_id"] == session_id

    assert lines[1]["stream"] == "stderr"
    assert lines[1]["content"] == "Warning"
    assert lines[1]["ts"] == 2.0


# -- finalize_session --


def test_finalize_session_with_exit_code(storage_root: Path) -> None:
    """finalize_session with exit_code sets both ended_at and exit_code."""
    session_meta = create_session(
        command=["pytest"], root=storage_root
    )

    before = time.time()
    finalize_session(session_meta.session_id, exit_code=1, root=storage_root)
    after = time.time()

    meta = json.loads(
        (storage_root / "sessions" / session_meta.session_id / "meta.json").read_text(encoding="utf-8")
    )
    assert meta["exit_code"] == 1
    assert isinstance(meta["ended_at"], float)
    assert before <= meta["ended_at"] <= after


def test_finalize_session_without_exit_code(storage_root: Path) -> None:
    """finalize_session with exit_code=None only sets ended_at, leaves exit_code as None."""
    session_meta = create_session(
        command=["ls"], root=storage_root
    )

    finalize_session(session_meta.session_id, exit_code=None, root=storage_root)

    meta = json.loads(
        (storage_root / "sessions" / session_meta.session_id / "meta.json").read_text(encoding="utf-8")
    )
    assert meta["ended_at"] is not None
    assert meta["exit_code"] is None


# -- delete_session --


def test_delete_session_removes_dir_and_returns_false_for_missing(
    storage_root: Path,
) -> None:
    """delete_session removes the directory; returns False for a missing session."""
    session_meta = create_session(
        command=["rm", "-rf", "/"], root=storage_root
    )
    session_id = session_meta.session_id
    session_dir = storage_root / "sessions" / session_id
    assert session_dir.exists()

    result = delete_session(session_id, root=storage_root)
    assert result is True
    assert not session_dir.exists()

    # Second delete returns False
    result_again = delete_session(session_id, root=storage_root)
    assert result_again is False

    # Non-existent id
    assert delete_session("nonexistent-id", root=storage_root) is False


# -- list_sessions --


def test_list_sessions_sorted_desc_and_empty(storage_root: Path) -> None:
    """list_sessions returns sessions sorted by started_at desc; empty when no sessions."""
    # Empty at first
    assert list_sessions(root=storage_root) == []

    # Create sessions with a small delay to get distinct started_at values
    meta1 = create_session(command=["first"], root=storage_root)
    # Ensure the second session has a later started_at
    time.sleep(0.01)
    meta2 = create_session(command=["second"], root=storage_root)

    sessions = list_sessions(root=storage_root)
    assert len(sessions) == 2
    # Most recent first — sessions are SessionMeta objects
    assert sessions[0].session_id == meta2.session_id
    assert sessions[1].session_id == meta1.session_id
    assert sessions[0].started_at >= sessions[1].started_at


# -- error handling (Phase 4G) --


def test_read_config_corrupt_json_raises_storage_error(storage_root: Path) -> None:
    """read_config raises MbStorageError for corrupt JSON."""
    (storage_root / "config.json").write_text("{broken", encoding="utf-8")

    with pytest.raises(MbStorageError, match="Corrupt config.json"):
        read_config(root=storage_root)


def test_list_sessions_skips_corrupt_meta(storage_root: Path) -> None:
    """list_sessions skips sessions with unreadable meta.json."""
    # Create a valid session
    meta_good = create_session(command=["ok"], root=storage_root)

    # Create a session with corrupt meta
    bad_dir = storage_root / "sessions" / "corrupt-session"
    bad_dir.mkdir()
    (bad_dir / "meta.json").write_text("{bad json", encoding="utf-8")

    sessions = list_sessions(root=storage_root)
    # Only the good session should be returned
    assert len(sessions) == 1
    assert sessions[0].session_id == meta_good.session_id
