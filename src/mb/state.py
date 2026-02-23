"""ProjectState generation via Ollama LLM summarization."""

from __future__ import annotations

import heapq
import json
import time
from pathlib import Path

from mb.chunker import chunk_all_sessions
from mb.ollama_client import OllamaClient


def _sample_chunks_for_state(
    chunks: list[tuple[float, float, str]],
    max_chars: int = 8000,
) -> str:
    """Select a representative sample of chunks within a character budget.

    Each element in *chunks* is ``(quality_score, ts_end, text)``.

    Strategy:
    1. If everything fits — return all chunks in chronological order.
    2. Otherwise pin the first (oldest) and last (newest) chunk as temporal
       anchors, then fill the remaining budget with the highest-quality
       chunks.  The final output is sorted by ``ts_end`` for chronological
       coherence.
    """
    if not chunks:
        return ""

    total = sum(len(t) for _, _, t in chunks)
    separator_overhead = (len(chunks) - 1) * 2  # "\n\n" between chunks

    if total + separator_overhead <= max_chars:
        # Everything fits — chronological order
        ordered = sorted(chunks, key=lambda c: c[1])
        return "\n\n".join(t for _, _, t in ordered)

    # Pin first (min ts_end) and last (max ts_end)
    first = min(chunks, key=lambda c: c[1])
    last = max(chunks, key=lambda c: c[1])

    selected: list[tuple[float, float, str]] = [first]
    if last is not first:
        selected.append(last)

    budget = max_chars - sum(len(t) for _, _, t in selected)
    budget -= (len(selected) - 1) * 2  # separators for pinned

    # Remaining candidates sorted by quality descending
    pinned_ids = {id(first), id(last)}
    candidates = [c for c in chunks if id(c) not in pinned_ids]
    # Use a max-heap (negate quality) for top-quality selection
    quality_heap = [(-q, ts, t) for q, ts, t in candidates]
    heapq.heapify(quality_heap)

    while quality_heap and budget > 0:
        neg_q, ts, t = heapq.heappop(quality_heap)
        cost = len(t) + 2  # text + separator
        if cost <= budget + 2:  # +2: last chunk doesn't need trailing sep
            selected.append((-neg_q, ts, t))
            budget -= cost

    # Sort by ts_end for chronological output
    selected.sort(key=lambda c: c[1])
    return "\n\n".join(t for _, _, t in selected)


_SYSTEM_PROMPT = """\
You are a project analyst. Given a transcript of developer sessions with an LLM assistant, \
produce a structured JSON summary with these exact fields:
- "summary": A 2-3 sentence overview of the project and its current state.
- "decisions": A list of key decisions, each with "id" (D1, D2, ...), "statement", and "rationale".
- "constraints": A list of identified constraints (strings).
- "tasks": A list of active tasks, each with "id" (T1, T2, ...) and "status" (one of: pending, in_progress, done).

Output ONLY valid JSON, no markdown, no explanations."""


def generate_state(
    storage_root: Path,
    ollama_client: OllamaClient,
) -> dict:
    """Generate ProjectState from session chunks via LLM.

    Uses chunks (cleaned, quality-filtered text from chunker/claude_adapter)
    instead of raw events to avoid TUI noise. Sends concatenated chunk text
    to Ollama chat with deterministic settings (temperature=0.0, seed=42).

    Saves result to state/state.json.
    """
    # Ensure all sessions are chunked (triggers claude_adapter for Claude sessions)
    chunk_all_sessions(storage_root, force=True)

    sessions_dir = storage_root / "sessions"
    all_chunks: list[tuple[float, float, str]] = []
    source_sessions: list[str] = []

    if sessions_dir.exists():
        for session_dir in sorted(sessions_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            chunks_path = session_dir / "chunks.jsonl"
            if not chunks_path.exists():
                continue

            source_sessions.append(session_dir.name)
            with chunks_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    chunk = json.loads(line)
                    text = chunk.get("text", "").strip()
                    quality = chunk.get("quality_score", 0)
                    ts_end = chunk.get("ts_end", 0)
                    if text and quality >= 0.3:
                        all_chunks.append((quality, ts_end, text))

    combined_text = _sample_chunks_for_state(all_chunks)

    result = ollama_client.chat(
        user_prompt=combined_text or "(No session data available)",
        system_prompt=_SYSTEM_PROMPT,
        as_json=True,
        temperature=0.0,
        seed=42,
    )

    # Ensure result is a dict with expected fields
    if not isinstance(result, dict):
        result = {
            "summary": str(result),
            "decisions": [],
            "constraints": [],
            "tasks": [],
        }

    result["updated_at"] = time.time()
    result["source_sessions"] = source_sessions

    # Save to state/state.json
    state_dir = storage_root / "state"
    state_dir.mkdir(exist_ok=True)
    state_path = state_dir / "state.json"
    state_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    return result


def load_state(storage_root: Path) -> dict | None:
    """Load existing state.json if present."""
    state_path = storage_root / "state" / "state.json"
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text(encoding="utf-8"))


def _state_is_stale(storage_root: Path) -> bool:
    """Check if any session's chunks.jsonl is newer than state.json."""
    state_path = storage_root / "state" / "state.json"
    if not state_path.exists():
        return False
    sessions_dir = storage_root / "sessions"
    if not sessions_dir.exists():
        return False
    state_mtime = state_path.stat().st_mtime
    for session_dir in sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue
        chunks_path = session_dir / "chunks.jsonl"
        if chunks_path.exists() and chunks_path.stat().st_mtime > state_mtime:
            return True
    return False
