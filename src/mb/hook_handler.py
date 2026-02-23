"""Claude Code Stop hook entry point.

Invoked by Claude Code on each Stop event via:
    python -m mb.hook_handler

Reads hook payload from stdin, processes the transcript into chunks.
Always exits 0 — never blocks Claude.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def _load_hooks_state(storage_root: Path) -> dict:
    path = storage_root / "hooks_state.json"
    if not path.exists():
        return {"sessions": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_hooks_state(storage_root: Path, state: dict) -> None:
    path = storage_root / "hooks_state.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    tmp.rename(path)


def _create_hook_session(storage_root: Path, cwd: str) -> str:
    """Create a new session directory with meta.json (source=hook, no events.jsonl)."""
    from mb.storage import generate_session_id

    session_id = generate_session_id()
    session_dir = storage_root / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "session_id": session_id,
        "command": ["claude"],
        "cwd": cwd,
        "started_at": time.time(),
        "ended_at": None,
        "exit_code": None,
        "source": "hook",
    }
    (session_dir / "meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return session_id


def _update_meta(storage_root: Path, session_id: str) -> None:
    """Update ended_at timestamp in meta.json."""
    meta_path = storage_root / "sessions" / session_id / "meta.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["ended_at"] = time.time()
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _process_hook(
    transcript_path: str,
    cwd: str,
    claude_session_id: str,
    storage_root: Path,
) -> None:
    """Core processing: create/update session, generate chunks from transcript."""
    from mb.claude_adapter import chunks_from_turns, extract_turns

    transcript = Path(transcript_path)
    if not transcript.exists():
        return

    transcript_size = transcript.stat().st_size
    if transcript_size == 0:
        return

    state = _load_hooks_state(storage_root)
    sessions = state.setdefault("sessions", {})
    mapping = sessions.get(claude_session_id)

    if mapping is None:
        # New session
        session_id = _create_hook_session(storage_root, cwd)
        sessions[claude_session_id] = {
            "mb_session_id": session_id,
            "transcript_path": transcript_path,
            "transcript_size": transcript_size,
            "last_processed": time.time(),
        }
    else:
        session_id = mapping["mb_session_id"]
        if mapping.get("transcript_size") == transcript_size:
            # Transcript unchanged — no-op
            return

    # Extract turns and generate chunks
    turns = extract_turns(transcript)
    if not turns:
        _save_hooks_state(storage_root, state)
        return

    chunks = chunks_from_turns(turns, session_id)
    if not chunks:
        _save_hooks_state(storage_root, state)
        return

    # Write chunks.jsonl
    session_dir = storage_root / "sessions" / session_id
    chunks_path = session_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Update meta ended_at
    _update_meta(storage_root, session_id)

    # Update state
    sessions[claude_session_id]["transcript_size"] = transcript_size
    sessions[claude_session_id]["last_processed"] = time.time()
    _save_hooks_state(storage_root, state)


def main() -> None:
    """Entry point: parse stdin JSON payload, resolve project, process."""
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return

    transcript_path = payload.get("transcript_path")
    claude_session_id = payload.get("session_id")
    cwd = payload.get("cwd", "")

    if not transcript_path or not claude_session_id or not cwd:
        return

    # Resolve storage root = {cwd}/.memory-bank/
    storage_root = Path(cwd) / ".memory-bank"

    # Auto-init if missing
    if not (storage_root / "config.json").exists():
        from mb.storage import init_storage
        init_storage(storage_root)

    _process_hook(
        transcript_path=transcript_path,
        cwd=cwd,
        claude_session_id=claude_session_id,
        storage_root=storage_root,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
    sys.exit(0)
