"""Retroactive import of Claude Code sessions into Memory Bank.

Discovers historical sessions from ~/.claude/projects/<encoded-cwd>/
and imports them as MB sessions with chunks.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

IMPORT_STATE_FILE = "import_state.json"


def discover_claude_sessions(cwd: str) -> list[Path]:
    """Find all .jsonl files in ~/.claude/projects/<encoded-cwd>/, excluding agent-*."""
    from mb.claude_adapter import encode_project_dir

    claude_projects = Path.home() / ".claude" / "projects"
    if not claude_projects.exists():
        return []

    project_dir = claude_projects / encode_project_dir(cwd)
    if not project_dir.exists():
        return []

    sessions = [
        f
        for f in sorted(project_dir.iterdir())
        if f.suffix == ".jsonl" and not f.name.startswith("agent-")
    ]
    return sessions


def load_import_state(storage_root: Path) -> dict[str, Any]:
    """Load .memory-bank/import_state.json (tracks imported Claude session UUIDs)."""
    path = storage_root / IMPORT_STATE_FILE
    if not path.exists():
        return {"imported": {}}
    try:
        result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"imported": {}}
    return result


def save_import_state(storage_root: Path, state: dict[str, Any]) -> None:
    """Atomic write to import_state.json."""
    path = storage_root / IMPORT_STATE_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    tmp.rename(path)


def import_claude_sessions(
    storage_root: Path,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Import all unprocessed Claude sessions. Returns (imported, skipped)."""
    from mb.claude_adapter import _parse_ts, chunks_from_turns, extract_turns
    from mb.storage import create_session, finalize_session

    cwd = str(storage_root.parent)
    session_files = discover_claude_sessions(cwd)

    if not session_files:
        return 0, 0

    state = load_import_state(storage_root)
    imported_map = state.setdefault("imported", {})

    imported_count = 0
    skipped_count = 0

    for jsonl_file in session_files:
        claude_uuid = jsonl_file.stem

        if claude_uuid in imported_map:
            skipped_count += 1
            continue

        turns = extract_turns(jsonl_file)
        if not turns:
            skipped_count += 1
            continue

        if dry_run:
            imported_count += 1
            continue

        # Extract timestamps from turns
        started_at = _parse_ts(turns[0].timestamp)
        ended_at = _parse_ts(turns[-1].timestamp)

        session_id = create_session(
            ["claude"], cwd=cwd, root=storage_root,
            source="import", create_events=False,
        )

        # Update meta.json with original timestamps
        meta_path = storage_root / "sessions" / session_id / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if started_at:
            meta["started_at"] = started_at
        if ended_at:
            meta["ended_at"] = ended_at
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

        chunks = chunks_from_turns(turns, session_id)

        # Write chunks.jsonl
        session_dir = storage_root / "sessions" / session_id
        chunks_path = session_dir / "chunks.jsonl"
        with chunks_path.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        finalize_session(session_id, root=storage_root)

        # Restore ended_at from original (finalize_session overwrites with time.time())
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if ended_at:
            meta["ended_at"] = ended_at
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

        imported_map[claude_uuid] = session_id
        save_import_state(storage_root, state)

        imported_count += 1

    return imported_count, skipped_count
