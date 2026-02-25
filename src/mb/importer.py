"""Retroactive import of Claude Code sessions into Memory Bank.

Discovers historical sessions from ~/.claude/projects/<encoded-cwd>/
and imports them as MB sessions with chunks.
"""

from __future__ import annotations

import logging
from pathlib import Path

from mb.store import NdjsonStorage

logger = logging.getLogger(__name__)


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


def import_claude_sessions(
    storage: NdjsonStorage,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Import all unprocessed Claude sessions. Returns (imported, skipped)."""
    from mb.pipeline import ImportSource

    source = ImportSource(dry_run=dry_run)
    source.ingest(storage)
    return source.imported, source.skipped


def import_claude_sessions_with_artifacts(
    storage: NdjsonStorage,
    dry_run: bool = False,
) -> dict:
    """Import sessions and artifacts. Returns detailed result dict."""
    from mb.pipeline import ImportSource

    source = ImportSource(dry_run=dry_run)
    source.ingest(storage)
    return {
        "imported": source.imported,
        "skipped": source.skipped,
        "plans_imported": source.plans_imported,
        "todos_imported": source.todos_imported,
        "tasks_imported": source.tasks_imported,
        "dry_run_todo_items": source.dry_run_todo_items,
        "dry_run_task_items": source.dry_run_task_items,
    }
