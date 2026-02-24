"""Claude Code native session adapter.

Reads Claude Code's structured JSONL session files from ~/.claude/projects/
and extracts clean turn-based content for chunking, bypassing raw PTY output.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from mb.chunker import _quality_score

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    """A single user→assistant conversation turn."""

    turn_number: int
    user_message: str
    assistant_texts: list[str] = field(default_factory=list)
    timestamp: str | None = None

    @property
    def full_text(self) -> str:
        parts = []
        if self.user_message:
            parts.append(f"User: {self.user_message}")
        for text in self.assistant_texts:
            parts.append(f"Assistant: {text}")
        return "\n\n".join(parts)


def encode_project_dir(cwd: str) -> str:
    """Encode a project path to Claude Code's directory name format.

    /home/user/my-project -> -home-user-my-project
    /Users/alex/SG_prod   -> -Users-alex-SG-prod

    Claude Code replaces both '/' and '_' with '-'.
    """
    path = cwd.rstrip("/")
    if path.startswith("/"):
        path = path[1:]
    return "-" + path.replace("/", "-").replace("_", "-")


def find_claude_session_file(
    cwd: str, started_at: float, ended_at: float | None = None
) -> Path | None:
    """Find the Claude Code session JSONL that matches our mb session.

    Looks in ~/.claude/projects/<encoded-cwd>/ for .jsonl files
    modified during the mb session's time window.
    """
    claude_projects = Path.home() / ".claude" / "projects"
    if not claude_projects.exists():
        return None

    project_dir_name = encode_project_dir(cwd)
    project_dir = claude_projects / project_dir_name
    if not project_dir.exists():
        return None

    # Find .jsonl files (exclude agent-*.jsonl subchains)
    candidates: list[tuple[float, Path]] = []
    for f in project_dir.iterdir():
        if not f.suffix == ".jsonl" or f.name.startswith("agent-"):
            continue
        mtime = f.stat().st_mtime
        # Session file should have been modified during or after our session
        if mtime >= started_at - 60:  # 60s tolerance
            if ended_at is None or mtime <= ended_at + 300:  # 5min tolerance
                candidates.append((mtime, f))

    if not candidates:
        # Fallback: find most recently modified file
        all_files = [
            (f.stat().st_mtime, f)
            for f in project_dir.iterdir()
            if f.suffix == ".jsonl" and not f.name.startswith("agent-")
        ]
        if all_files:
            all_files.sort(key=lambda x: x[0], reverse=True)
            return all_files[0][1]
        return None

    # Pick the one closest to our session start
    candidates.sort(key=lambda x: abs(x[0] - started_at))
    return candidates[0][1]


def extract_turns(session_file: Path) -> list[Turn]:
    """Parse Claude Code JSONL and extract conversation turns.

    A turn starts with a user text message (not tool_result, not isMeta)
    and includes all subsequent assistant text blocks until the next user message.
    """
    turns: list[Turn] = []
    current_turn: Turn | None = None
    turn_number = 0

    with session_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Skipping malformed JSONL line in %s", session_file)
                continue

            # Skip non-message lines
            msg_type = data.get("type")
            if msg_type not in ("user", "assistant"):
                continue

            # Skip sidechain (subagent) messages
            if data.get("isSidechain"):
                continue

            # Skip expanded skill prompts
            if data.get("isMeta"):
                continue

            message = data.get("message", {})
            content = message.get("content")
            timestamp = data.get("timestamp")

            if msg_type == "user":
                user_text = _extract_user_text(content)
                if user_text:
                    # Start a new turn
                    if current_turn is not None:
                        turns.append(current_turn)
                    turn_number += 1
                    current_turn = Turn(
                        turn_number=turn_number,
                        user_message=user_text,
                        timestamp=timestamp,
                    )

            elif msg_type == "assistant" and current_turn is not None:
                assistant_text = _extract_assistant_text(content)
                if assistant_text:
                    current_turn.assistant_texts.append(assistant_text)

    # Don't forget the last turn
    if current_turn is not None:
        turns.append(current_turn)

    return turns


def _extract_user_text(content: str | list | None) -> str | None:
    """Extract user text from message content, skipping tool_results and commands."""
    if content is None:
        return None

    if isinstance(content, str):
        text = content.strip()
        # Skip system/command wrappers and local command output
        if text.startswith((
            "<command-", "<local-command-", "<task-notification>",
            "<system-reminder>", "<local-command-caveat>",
            "<bash-input>", "<bash-stdout>", "<bash-stderr>",
        )):
            return None
        # Skip interrupt placeholders
        if "request interrupted by user" in text.lower():
            return None
        return text if text else None

    if isinstance(content, list):
        texts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "tool_result":
                continue
            if item.get("type") == "text":
                text = item.get("text", "").strip()
                # Skip system/command wrappers and local command output
                if text.startswith((
                    "<command-", "<local-command-", "<task-notification>",
                    "<system-reminder>", "<local-command-caveat>",
                    "<bash-input>", "<bash-stdout>", "<bash-stderr>",
                )):
                    continue
                if "request interrupted by user" in text.lower():
                    continue
                if text:
                    texts.append(text)
        return "\n".join(texts) if texts else None

    return None


def _extract_assistant_text(content: str | list | None) -> str | None:
    """Extract assistant text from message content, skipping tool_use and thinking."""
    if content is None:
        return None

    if isinstance(content, str):
        return content.strip() or None

    if isinstance(content, list):
        texts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                text = item.get("text", "").strip()
                if text:
                    texts.append(text)
        return "\n".join(texts) if texts else None

    return None


def _parse_ts(iso_str: str | None) -> float:
    """Parse ISO 8601 timestamp string to epoch float.

    Returns 0.0 for None, empty string, or unparseable values.
    """
    if not iso_str:
        return 0.0
    try:
        # Python 3.10 fromisoformat doesn't handle 'Z' suffix
        normalized = iso_str.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).timestamp()
    except (ValueError, TypeError):
        return 0.0


def chunks_from_turns(
    turns: list[Turn],
    session_id: str,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
) -> list[dict[str, Any]]:
    """Generate chunk dicts from Turn objects.

    Shared by hook_handler (direct transcript) and chunk_claude_session (PTY path).

    Returns list of chunk dicts compatible with the standard chunker output.
    """
    max_chars = max_tokens * 4
    overlap_chars = overlap_tokens * 4
    chunks: list[dict[str, Any]] = []
    chunk_index = 0

    for turn in turns:
        text = turn.full_text
        if not text.strip():
            continue

        turn_ts = _parse_ts(turn.timestamp)
        segments = _split_turn_text(text, max_chars)
        overlap_text = ""

        for seg in segments:
            chunk_text = overlap_text + seg if overlap_text else seg
            token_estimate = len(chunk_text) // 4

            chunk = {
                "chunk_id": f"{session_id}-{chunk_index}",
                "session_id": session_id,
                "index": chunk_index,
                "text": chunk_text,
                "ts_start": turn_ts,
                "ts_end": turn_ts,
                "token_estimate": token_estimate,
                "quality_score": _quality_score(chunk_text),
                "source": "claude_native",
                "turn_number": turn.turn_number,
            }
            chunks.append(chunk)
            chunk_index += 1

            if len(seg) > overlap_chars:
                overlap_text = seg[-overlap_chars:]
            else:
                overlap_text = seg

    return chunks


def chunk_claude_session(
    session_dir: Path,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
) -> list[dict[str, Any]]:
    """Generate chunks from a Claude Code native session file.

    Reads meta.json to find session params, locates the corresponding
    Claude Code JSONL, extracts turns, and generates meaningful chunks.

    Returns list of chunk dicts compatible with the standard chunker output.
    """
    meta_path = session_dir / "meta.json"
    if not meta_path.exists():
        return []

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    cwd = meta.get("cwd", "")
    started_at = meta.get("started_at", 0)
    ended_at = meta.get("ended_at")
    session_id = meta.get("session_id", session_dir.name)

    session_file = find_claude_session_file(cwd, started_at, ended_at)
    if session_file is None:
        return []

    turns = extract_turns(session_file)
    if not turns:
        return []

    chunks = chunks_from_turns(turns, session_id, max_tokens, overlap_tokens)

    # Write chunks.jsonl
    chunks_path = session_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    return chunks


def _split_turn_text(text: str, max_chars: int) -> list[str]:
    """Split turn text into segments at paragraph boundaries."""
    if len(text) <= max_chars:
        return [text]

    segments: list[str] = []
    paragraphs = text.split("\n\n")
    current = ""

    for para in paragraphs:
        candidate = current + "\n\n" + para if current else para
        if len(candidate) > max_chars and current:
            segments.append(current)
            current = para
        else:
            current = candidate

    if current.strip():
        # If still too long, force-split
        while len(current) > max_chars:
            segments.append(current[:max_chars])
            current = current[max_chars:]
        if current.strip():
            segments.append(current)

    return segments


def is_claude_session(meta: dict[str, Any]) -> bool:
    """Check if a session was a Claude Code session based on meta.json."""
    command = meta.get("command", [])
    if not command:
        return False
    cmd = command[0] if isinstance(command, list) else str(command)
    # Match "claude", "/path/to/claude", etc.
    return os.path.basename(cmd) == "claude"
