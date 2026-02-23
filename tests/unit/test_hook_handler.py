"""Tests for mb.hook_handler — Claude Code Stop hook processing."""

from __future__ import annotations

import json
import time
from pathlib import Path

from mb.hook_handler import _process_hook, _load_hooks_state, main


def _write_claude_jsonl(path: Path, messages: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")


def _sample_transcript(path: Path) -> None:
    """Write a minimal valid Claude transcript."""
    _write_claude_jsonl(path, [
        {"type": "user", "message": {"role": "user", "content": "explain decorators"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "Decorators wrap functions to extend behavior."},
        ]}},
    ])


def _init_storage(storage_root: Path) -> None:
    """Minimal storage init for tests."""
    storage_root.mkdir(parents=True, exist_ok=True)
    (storage_root / "sessions").mkdir(exist_ok=True)
    (storage_root / "config.json").write_text('{"version": "1.0"}', encoding="utf-8")


def test_process_creates_session(tmp_path: Path) -> None:
    """First hook call creates meta.json + chunks.jsonl."""
    storage_root = tmp_path / ".memory-bank"
    _init_storage(storage_root)

    transcript = tmp_path / "transcript.jsonl"
    _sample_transcript(transcript)

    _process_hook(
        transcript_path=str(transcript),
        cwd=str(tmp_path),
        claude_session_id="test-uuid-1",
        storage_root=storage_root,
    )

    # Check hooks_state.json
    state = _load_hooks_state(storage_root)
    assert "test-uuid-1" in state["sessions"]
    session_id = state["sessions"]["test-uuid-1"]["mb_session_id"]

    # Check session dir
    session_dir = storage_root / "sessions" / session_id
    assert (session_dir / "meta.json").exists()
    assert (session_dir / "chunks.jsonl").exists()

    # No events.jsonl for hook sessions
    assert not (session_dir / "events.jsonl").exists()

    # Check chunks content
    chunks = [
        json.loads(line)
        for line in (session_dir / "chunks.jsonl").read_text().strip().splitlines()
    ]
    assert len(chunks) >= 1
    assert "decorators" in chunks[0]["text"].lower()


def test_process_updates_existing(tmp_path: Path) -> None:
    """Same claude_session_id with changed transcript updates chunks."""
    storage_root = tmp_path / ".memory-bank"
    _init_storage(storage_root)

    transcript = tmp_path / "transcript.jsonl"
    _sample_transcript(transcript)

    _process_hook(
        transcript_path=str(transcript),
        cwd=str(tmp_path),
        claude_session_id="test-uuid-2",
        storage_root=storage_root,
    )

    state = _load_hooks_state(storage_root)
    session_id = state["sessions"]["test-uuid-2"]["mb_session_id"]

    # Append more content to transcript
    with transcript.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "type": "user",
            "message": {"role": "user", "content": "show me an example"},
        }) + "\n")
        f.write(json.dumps({
            "type": "assistant",
            "message": {"role": "assistant", "content": [
                {"type": "text", "text": "Here is an example with @property."},
            ]},
        }) + "\n")

    _process_hook(
        transcript_path=str(transcript),
        cwd=str(tmp_path),
        claude_session_id="test-uuid-2",
        storage_root=storage_root,
    )

    # Same session_id, updated chunks
    state = _load_hooks_state(storage_root)
    assert state["sessions"]["test-uuid-2"]["mb_session_id"] == session_id

    session_dir = storage_root / "sessions" / session_id
    chunks = [
        json.loads(line)
        for line in (session_dir / "chunks.jsonl").read_text().strip().splitlines()
    ]
    assert len(chunks) >= 2


def test_process_skips_unchanged(tmp_path: Path) -> None:
    """Same transcript size results in no-op."""
    storage_root = tmp_path / ".memory-bank"
    _init_storage(storage_root)

    transcript = tmp_path / "transcript.jsonl"
    _sample_transcript(transcript)

    _process_hook(
        transcript_path=str(transcript),
        cwd=str(tmp_path),
        claude_session_id="test-uuid-3",
        storage_root=storage_root,
    )

    state_before = _load_hooks_state(storage_root)
    ts_before = state_before["sessions"]["test-uuid-3"]["last_processed"]

    # Small delay to detect timestamp change
    time.sleep(0.01)

    _process_hook(
        transcript_path=str(transcript),
        cwd=str(tmp_path),
        claude_session_id="test-uuid-3",
        storage_root=storage_root,
    )

    state_after = _load_hooks_state(storage_root)
    ts_after = state_after["sessions"]["test-uuid-3"]["last_processed"]

    # last_processed should NOT have changed — it was a no-op
    assert ts_after == ts_before


def test_process_auto_inits_storage(tmp_path: Path, monkeypatch) -> None:
    """main() auto-inits .memory-bank/ if missing."""
    transcript = tmp_path / "transcript.jsonl"
    _sample_transcript(transcript)

    payload = json.dumps({
        "session_id": "uuid-auto-init",
        "transcript_path": str(transcript),
        "cwd": str(tmp_path),
    })
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(payload))

    main()

    storage_root = tmp_path / ".memory-bank"
    assert (storage_root / "config.json").exists()
    assert (storage_root / "sessions").is_dir()


def test_process_handles_empty_transcript(tmp_path: Path) -> None:
    """Empty transcript file exits cleanly."""
    storage_root = tmp_path / ".memory-bank"
    _init_storage(storage_root)

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("", encoding="utf-8")

    _process_hook(
        transcript_path=str(transcript),
        cwd=str(tmp_path),
        claude_session_id="test-uuid-empty",
        storage_root=storage_root,
    )

    state = _load_hooks_state(storage_root)
    assert "test-uuid-empty" not in state.get("sessions", {})


def test_meta_has_source_hook(tmp_path: Path) -> None:
    """meta.json contains source=hook."""
    storage_root = tmp_path / ".memory-bank"
    _init_storage(storage_root)

    transcript = tmp_path / "transcript.jsonl"
    _sample_transcript(transcript)

    _process_hook(
        transcript_path=str(transcript),
        cwd=str(tmp_path),
        claude_session_id="test-uuid-source",
        storage_root=storage_root,
    )

    state = _load_hooks_state(storage_root)
    session_id = state["sessions"]["test-uuid-source"]["mb_session_id"]
    meta_path = storage_root / "sessions" / session_id / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["source"] == "hook"


def test_main_handles_bad_stdin(monkeypatch) -> None:
    """main() exits cleanly on malformed stdin."""
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("not json"))
    # Should not raise
    main()


def test_main_handles_missing_fields(monkeypatch) -> None:
    """main() exits cleanly when required fields are missing."""
    payload = json.dumps({"cwd": ""})
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(payload))
    main()
