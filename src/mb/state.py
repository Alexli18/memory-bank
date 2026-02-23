"""ProjectState generation via Ollama LLM summarization."""

from __future__ import annotations

import json
import time
from pathlib import Path

from mb.chunker import chunk_all_sessions
from mb.ollama_client import OllamaClient


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
    all_text_parts: list[str] = []
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
                    if text and quality >= 0.3:
                        all_text_parts.append(text)

    combined_text = "\n\n".join(all_text_parts)

    # Truncate to avoid token limits (~8K chars for gemma3:4b context)
    max_input_chars = 8000
    if len(combined_text) > max_input_chars:
        combined_text = combined_text[:max_input_chars] + "\n[...truncated...]"

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
