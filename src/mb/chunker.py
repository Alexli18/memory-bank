"""Deterministic text chunker for session events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mb.sanitizer import strip_terminal_noise


def _quality_score(text: str) -> float:
    """Score chunk quality: ratio of alphanumeric content to total length."""
    if not text or not text.strip():
        return 0.0
    stripped = text.strip()
    alnum_count = sum(1 for c in stripped if c.isalnum())
    return round(alnum_count / len(stripped), 3) if stripped else 0.0


def chunk_session(events_path: Path) -> list[dict[str, Any]]:
    """Read events.jsonl and produce deterministic text chunks.

    For Claude Code sessions, delegates to the Claude adapter which reads
    Claude's native structured JSONL for much higher quality chunks.

    For other sessions, aggregates stdout events ordered by timestamp,
    segments at double newline or when token estimate exceeds 512.

    Returns list of chunk dicts and writes chunks.jsonl to the same directory.
    """
    session_dir = events_path.parent
    session_id = session_dir.name

    # Try Claude adapter for Claude Code sessions
    claude_chunks = _try_claude_adapter(session_dir)
    if claude_chunks:
        return claude_chunks

    # Read and filter stdout events, ordered by timestamp
    events: list[dict[str, Any]] = []
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("stream") == "stdout":
                events.append(event)

    events.sort(key=lambda e: e.get("ts", 0))

    # Strip terminal UI noise from event content before chunking
    for event in events:
        event["content"] = strip_terminal_noise(event.get("content", ""))

    if not events:
        return []

    # Build segments from aggregated text
    segments = _segment_events(events)

    # Build chunk dicts with overlap
    chunks: list[dict[str, Any]] = []
    overlap_text = ""

    for idx, seg in enumerate(segments):
        text = overlap_text + seg["text"] if overlap_text else seg["text"]
        # Re-apply noise stripping to assembled text (UI patterns may span events)
        text = strip_terminal_noise(text)

        chunk = {
            "chunk_id": f"{session_id}-{idx}",
            "session_id": session_id,
            "index": idx,
            "text": text,
            "ts_start": seg["ts_start"],
            "ts_end": seg["ts_end"],
            "token_estimate": len(text) // 4,
            "quality_score": _quality_score(text),
        }
        chunks.append(chunk)

        # Prepare overlap for next chunk: last 200 chars (~50 tokens)
        overlap_chars = 200
        if len(seg["text"]) > overlap_chars:
            overlap_text = seg["text"][-overlap_chars:]
        else:
            overlap_text = seg["text"]

    # Write chunks.jsonl
    chunks_path = session_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    return chunks


def chunk_all_sessions(storage_root: Path, force: bool = False) -> None:
    """Ensure all sessions have chunks.jsonl.

    Args:
        storage_root: Path to .memory-bank/ directory.
        force: If True, re-chunk sessions that already have chunks.jsonl.
    """
    sessions_dir = storage_root / "sessions"
    if not sessions_dir.exists():
        return
    for session_dir in sorted(sessions_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        chunks_path = session_dir / "chunks.jsonl"
        if chunks_path.exists() and not force:
            continue
        events_path = session_dir / "events.jsonl"
        if events_path.exists():
            chunk_session(events_path)
        else:
            # Hook-created sessions may lack events.jsonl;
            # try Claude adapter directly via meta.json
            _try_claude_adapter(session_dir)


def _segment_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Segment events into chunks by double newline or token limit."""
    max_tokens = 512
    max_chars = max_tokens * 4  # chars/4 heuristic

    segments: list[dict[str, Any]] = []
    current_text = ""
    current_ts_start: float | None = None
    current_ts_end: float = 0.0

    for event in events:
        content = event.get("content", "")
        ts = event.get("ts", 0.0)

        if current_ts_start is None:
            current_ts_start = ts
        current_ts_end = ts

        # Check for double newline split points within content
        parts = content.split("\n\n")

        for i, part in enumerate(parts):
            if i > 0:
                # There was a \n\n boundary — flush current segment if non-empty
                if current_text.strip():
                    segments.append(
                        {
                            "text": current_text,
                            "ts_start": current_ts_start,
                            "ts_end": current_ts_end,
                        }
                    )
                current_text = ""
                current_ts_start = ts

            current_text += part
            if i < len(parts) - 1:
                current_text += "\n\n"

            # Check token limit
            while len(current_text) > max_chars:
                # Split at max_chars boundary
                split_text = current_text[:max_chars]
                segments.append(
                    {
                        "text": split_text,
                        "ts_start": current_ts_start
                        if current_ts_start is not None
                        else ts,
                        "ts_end": ts,
                    }
                )
                current_text = current_text[max_chars:]
                current_ts_start = ts

    # Flush remaining
    if current_text.strip():
        segments.append(
            {
                "text": current_text,
                "ts_start": current_ts_start if current_ts_start is not None else 0.0,
                "ts_end": current_ts_end,
            }
        )

    return segments


def _try_claude_adapter(session_dir: Path) -> list[dict[str, Any]] | None:
    """Try to use Claude adapter for a session. Returns None if not applicable."""
    meta_path = session_dir / "meta.json"
    if not meta_path.exists():
        return None

    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    from mb.claude_adapter import is_claude_session, chunk_claude_session

    if not is_claude_session(meta):
        return None

    chunks = chunk_claude_session(session_dir)
    return chunks if chunks else None
