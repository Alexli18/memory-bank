"""Claude Code SessionStart hook entry point.

Invoked by Claude Code on each SessionStart event via:
    python -m mb.session_start_hook

Reads hook payload from stdin, outputs lightweight context pack on startup.
Always exits 0 — never blocks Claude.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point: parse stdin JSON, check preconditions, output lightweight pack."""
    import json
    from pathlib import Path

    try:
        raw = sys.stdin.read()
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return

    source = payload.get("source")
    if source != "startup":
        return

    # Check .memory-bank/ exists in cwd
    storage_root = Path.cwd() / ".memory-bank"
    if not storage_root.is_dir():
        return

    from mb.store import NdjsonStorage

    storage = NdjsonStorage(storage_root)

    # Check data readiness: at least one session with chunks
    sessions = storage.list_sessions()
    if not sessions:
        return

    has_chunks = False
    for chunk in storage.iter_all_chunks():
        has_chunks = True
        break
    if not has_chunks:
        return

    from mb.pack import build_pack

    output = build_pack(budget=6000, storage=storage, lightweight=True, mode="auto")
    if output:
        print(output, end="")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.debug("SessionStart hook failed", exc_info=True)
    sys.exit(0)
