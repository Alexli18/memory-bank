"""Deterministic text chunker for session events."""

from __future__ import annotations

from typing import Any

from mb.models import Chunk, quality_score
from mb.sanitizer import strip_terminal_noise
from mb.store import NdjsonStorage


def _quality_score(text: str) -> float:
    """Score chunk quality. Delegates to models.quality_score for backward compat."""
    return quality_score(text)


def chunk_session(storage: NdjsonStorage, session_id: str) -> list[Chunk]:
    """Read events and produce deterministic text chunks.

    For Claude Code sessions, delegates to the Claude adapter which reads
    Claude's native structured JSONL for much higher quality chunks.

    For other sessions, aggregates stdout events ordered by timestamp,
    segments at double newline or when token estimate exceeds 512.

    Returns list of Chunk objects and writes chunks via storage.
    """
    # Try Claude adapter for Claude Code sessions
    claude_chunks = _try_claude_adapter(storage, session_id)
    if claude_chunks:
        return claude_chunks

    # Read events from storage
    all_events = storage.read_events(session_id)
    if not all_events:
        return []

    # Filter stdout events, ordered by timestamp
    stdout_events = [e for e in all_events if e.stream == "stdout"]
    stdout_events.sort(key=lambda e: e.ts)

    # Convert to dicts for segment processing with noise stripping
    events: list[dict[str, Any]] = []
    for e in stdout_events:
        events.append({
            "content": strip_terminal_noise(e.content),
            "ts": e.ts,
        })

    if not events:
        return []

    # Build segments from aggregated text
    segments = _segment_events(events)

    # Build chunks with overlap
    chunks: list[Chunk] = []
    overlap_text = ""

    for idx, seg in enumerate(segments):
        text = overlap_text + seg["text"] if overlap_text else seg["text"]
        # Re-apply noise stripping to assembled text (UI patterns may span events)
        text = strip_terminal_noise(text)

        chunk = Chunk(
            chunk_id=f"{session_id}-{idx}",
            session_id=session_id,
            index=idx,
            text=text,
            ts_start=seg["ts_start"],
            ts_end=seg["ts_end"],
            token_estimate=len(text) // 4,
            quality_score=quality_score(text),
        )
        chunks.append(chunk)

        # Prepare overlap for next chunk: last 200 chars (~50 tokens)
        overlap_chars = 200
        if len(seg["text"]) > overlap_chars:
            overlap_text = seg["text"][-overlap_chars:]
        else:
            overlap_text = seg["text"]

    # Write chunks via storage
    storage.write_chunks(session_id, chunks)

    return chunks


def chunk_all_sessions(storage: NdjsonStorage, force: bool = False) -> None:
    """Ensure all sessions have chunks.

    Args:
        storage: NdjsonStorage instance.
        force: If True, re-chunk sessions that already have chunks.
    """
    sessions_dir = storage.root / "sessions"
    if not sessions_dir.exists():
        return
    for session_dir in sorted(sessions_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        session_id = session_dir.name
        if storage.has_chunks(session_id) and not force:
            continue
        events_path = session_dir / "events.jsonl"
        if events_path.exists():
            chunk_session(storage, session_id)
        else:
            # Hook-created sessions may lack events.jsonl;
            # try Claude adapter directly via meta.json
            _try_claude_adapter(storage, session_id)


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


def _try_claude_adapter(storage: NdjsonStorage, session_id: str) -> list[Chunk] | None:
    """Try to use Claude adapter for a session. Returns None if not applicable."""
    meta = storage.read_meta(session_id)
    if meta is None:
        return None

    from mb.claude_adapter import is_claude_session, chunk_claude_session

    if not is_claude_session(meta.to_dict()):
        return None

    chunks = chunk_claude_session(storage, session_id)
    return chunks if chunks else None
