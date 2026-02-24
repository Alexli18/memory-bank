"""Claude Code Stop hook entry point.

Invoked by Claude Code on each Stop event via:
    python -m mb.hook_handler

Reads hook payload from stdin, processes the transcript into chunks.
Always exits 0 — never blocks Claude.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _process_hook(
    transcript_path: str,
    cwd: str,
    claude_session_id: str,
    storage_root: Path,
) -> None:
    """Core processing: create/update session, generate chunks from transcript."""
    from mb.pipeline import HookSource
    from mb.store import NdjsonStorage

    storage = NdjsonStorage(storage_root)
    source = HookSource(transcript_path, cwd, claude_session_id)
    source.ingest(storage)


def main() -> None:
    """Entry point: parse stdin JSON payload, resolve project, process."""
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        logger.debug("Failed to parse hook stdin payload")
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
        from mb.store import NdjsonStorage
        NdjsonStorage.init(storage_root)

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
        logger.debug("Hook handler failed", exc_info=True)
    sys.exit(0)
