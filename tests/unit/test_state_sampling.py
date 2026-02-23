"""Tests for _sample_chunks_for_state and _state_is_stale in mb.state."""

from __future__ import annotations

import os
import time
from pathlib import Path

from mb.state import _sample_chunks_for_state, _state_is_stale


def test_sample_all_fit() -> None:
    """When total text fits within budget, return all in chronological order."""
    chunks = [
        (0.5, 3.0, "third"),
        (0.8, 1.0, "first"),
        (0.6, 2.0, "second"),
    ]
    result = _sample_chunks_for_state(chunks, max_chars=1000)
    assert result == "first\n\nsecond\n\nthird"


def test_sample_empty() -> None:
    """Empty input produces empty string."""
    assert _sample_chunks_for_state([], max_chars=8000) == ""


def test_sample_single_chunk() -> None:
    """Single chunk is returned as-is."""
    chunks = [(0.9, 1.0, "only chunk")]
    result = _sample_chunks_for_state(chunks, max_chars=8000)
    assert result == "only chunk"


def test_sample_keeps_first_and_last() -> None:
    """First (oldest) and last (newest) chunks are always present."""
    chunks = [
        (0.1, 1.0, "oldest"),
        (0.9, 2.0, "middle-high-q"),
        (0.1, 3.0, "newest"),
    ]
    # Budget tight enough that not everything fits
    result = _sample_chunks_for_state(chunks, max_chars=35)
    assert "oldest" in result
    assert "newest" in result


def test_sample_prefers_high_quality() -> None:
    """Under budget pressure, high quality chunks are preferred over low quality."""
    chunks = [
        (0.1, 1.0, "first-low"),   # pinned as oldest
        (0.9, 2.0, "high-quality"),
        (0.1, 3.0, "low-quality"),
        (0.1, 4.0, "last-low"),    # pinned as newest
    ]
    # Budget: fits first + last + one more
    budget = len("first-low") + len("last-low") + len("high-quality") + 4 * 2
    result = _sample_chunks_for_state(chunks, max_chars=budget)
    assert "high-quality" in result
    assert "low-quality" not in result


def test_sample_respects_budget() -> None:
    """Result does not exceed max_chars."""
    chunks = [(0.8, float(i), f"chunk-{i:04d} " * 20) for i in range(100)]
    max_chars = 500
    result = _sample_chunks_for_state(chunks, max_chars=max_chars)
    assert len(result) <= max_chars


def test_sample_chronological_output() -> None:
    """Output chunks are sorted by ts_end regardless of selection order."""
    chunks = [
        (0.1, 1.0, "a"),
        (0.9, 5.0, "e"),
        (0.8, 3.0, "c"),
        (0.7, 4.0, "d"),
        (0.1, 2.0, "b"),
        (0.1, 6.0, "f"),
    ]
    result = _sample_chunks_for_state(chunks, max_chars=50)
    parts = result.split("\n\n")
    # Verify chronological order by checking that each part
    # appears in the original chunks with increasing ts_end
    ts_map = {text: ts for _, ts, text in chunks}
    timestamps = [ts_map[p] for p in parts]
    assert timestamps == sorted(timestamps)


# --- _state_is_stale ---


def _setup_storage(tmp_path: Path, *, with_state: bool = True, with_session: bool = True) -> Path:
    """Create a minimal storage_root with optional state.json and session chunks."""
    storage = tmp_path / ".memory-bank"
    if with_state:
        state_dir = storage / "state"
        state_dir.mkdir(parents=True)
        (state_dir / "state.json").write_text('{"summary":"test"}')
    if with_session:
        session_dir = storage / "sessions" / "sess1"
        session_dir.mkdir(parents=True)
        (session_dir / "chunks.jsonl").write_text('{"text":"chunk"}\n')
    return storage


def test_state_is_stale_when_chunks_newer(tmp_path: Path) -> None:
    """chunks.jsonl newer than state.json → stale."""
    storage = _setup_storage(tmp_path)
    # Ensure state.json is older than chunks.jsonl
    state_path = storage / "state" / "state.json"
    old_time = time.time() - 100
    os.utime(state_path, (old_time, old_time))
    assert _state_is_stale(storage) is True


def test_state_is_stale_returns_false_when_state_newer(tmp_path: Path) -> None:
    """state.json newer than chunks.jsonl → not stale."""
    storage = _setup_storage(tmp_path)
    # Ensure chunks.jsonl is older than state.json
    chunks_path = storage / "sessions" / "sess1" / "chunks.jsonl"
    old_time = time.time() - 100
    os.utime(chunks_path, (old_time, old_time))
    assert _state_is_stale(storage) is False


def test_state_is_stale_no_state_file(tmp_path: Path) -> None:
    """No state.json → not stale (nothing to invalidate)."""
    storage = _setup_storage(tmp_path, with_state=False)
    assert _state_is_stale(storage) is False


def test_state_is_stale_no_sessions(tmp_path: Path) -> None:
    """No sessions dir → not stale."""
    storage = _setup_storage(tmp_path, with_session=False)
    assert _state_is_stale(storage) is False
