"""Source/Processor pipeline for unified ingestion.

Provides a plugin system where Source adapters produce sessions
and Processor classes post-process them in configurable order.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from mb.chunker import chunk_session
from mb.pty_runner import run_session
from mb.search import build_index
from mb.store import NdjsonStorage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class Source(Protocol):
    """Ingestion source that produces sessions."""

    def ingest(self, storage: NdjsonStorage) -> list[str]:
        """Create/update sessions and return their IDs."""
        ...


@runtime_checkable
class Processor(Protocol):
    """Post-processor that operates on ingested sessions."""

    def process(self, storage: NdjsonStorage, session_ids: list[str]) -> None:
        """Process the given sessions."""
        ...


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class ProcessorPipeline:
    """Runs a sequence of processors on ingested sessions."""

    def __init__(self, processors: list[Processor] | None = None) -> None:
        self._processors: list[Processor] = list(processors or [])

    def run(self, storage: NdjsonStorage, session_ids: list[str]) -> None:
        for proc in self._processors:
            proc.process(storage, session_ids)


# ---------------------------------------------------------------------------
# Processors
# ---------------------------------------------------------------------------


class ChunkProcessor:
    """Generates chunks from session events."""

    def __init__(self, force: bool = False) -> None:
        self._force = force

    def process(self, storage: NdjsonStorage, session_ids: list[str]) -> None:
        for sid in session_ids:
            if storage.has_chunks(sid) and not self._force:
                continue
            chunk_session(storage, sid)


class EmbedProcessor:
    """Builds embedding index for session chunks."""

    def __init__(self, ollama_client: Any) -> None:
        self._client = ollama_client

    def process(self, storage: NdjsonStorage, session_ids: list[str]) -> None:
        build_index(storage, self._client)


# ---------------------------------------------------------------------------
# Source Adapters
# ---------------------------------------------------------------------------


class PtySource:
    """PTY source — runs a command in a PTY and captures events."""

    def __init__(self, child_cmd: list[str]) -> None:
        self._child_cmd = child_cmd
        self.exit_code: int = 1
        self.session_id: str = ""

    def ingest(self, storage: NdjsonStorage) -> list[str]:
        exit_code, session_id = run_session(self._child_cmd, storage)
        self.exit_code = exit_code
        self.session_id = session_id
        return [session_id]


class HookSource:
    """Hook source — processes a Claude Code transcript."""

    def __init__(
        self, transcript_path: str, cwd: str, claude_session_id: str
    ) -> None:
        self._transcript_path = transcript_path
        self._cwd = cwd
        self._claude_session_id = claude_session_id

    def ingest(self, storage: NdjsonStorage) -> list[str]:
        from mb.claude_adapter import chunks_from_turns, extract_turns

        transcript = Path(self._transcript_path)
        if not transcript.exists():
            return []

        transcript_size = transcript.stat().st_size
        if transcript_size == 0:
            return []

        state = storage.load_hooks_state()
        sessions = state.setdefault("sessions", {})
        mapping = sessions.get(self._claude_session_id)

        if mapping is None:
            meta = storage.create_session(
                ["claude"], cwd=self._cwd, source="hook", create_events=False,
            )
            session_id = meta.session_id
            sessions[self._claude_session_id] = {
                "mb_session_id": session_id,
                "transcript_path": self._transcript_path,
                "transcript_size": transcript_size,
                "last_processed": time.time(),
            }
        else:
            session_id = mapping["mb_session_id"]
            if mapping.get("transcript_size") == transcript_size:
                return []

        turns = extract_turns(transcript)
        if not turns:
            storage.save_hooks_state(state)
            return []

        chunks = chunks_from_turns(turns, session_id)
        if not chunks:
            storage.save_hooks_state(state)
            return []

        storage.write_chunks(session_id, chunks)
        storage.finalize_session(session_id)

        sessions[self._claude_session_id]["transcript_size"] = transcript_size
        sessions[self._claude_session_id]["last_processed"] = time.time()
        storage.save_hooks_state(state)

        return [session_id]


class ImportSource:
    """Import source — imports historical Claude Code sessions and artifacts."""

    def __init__(self, dry_run: bool = False) -> None:
        self._dry_run = dry_run
        self.imported: int = 0
        self.skipped: int = 0
        self.plans_imported: int = 0
        self.todos_imported: int = 0
        self.tasks_imported: int = 0
        # Dry-run detail counts
        self.dry_run_todo_items: int = 0
        self.dry_run_task_items: int = 0

    def ingest(self, storage: NdjsonStorage) -> list[str]:
        from mb.claude_adapter import _parse_ts, chunks_from_turns, extract_turns
        from mb.importer import discover_claude_sessions

        cwd = str(storage.root.parent)
        session_files = discover_claude_sessions(cwd)

        if not session_files:
            return []

        state = storage.load_import_state()
        imported_map = state.setdefault("imported", {})

        session_ids: list[str] = []

        for jsonl_file in session_files:
            claude_uuid = jsonl_file.stem

            if claude_uuid in imported_map:
                self.skipped += 1
                continue

            turns = extract_turns(jsonl_file)
            if not turns:
                self.skipped += 1
                continue

            if self._dry_run:
                self.imported += 1
                continue

            started_at = _parse_ts(turns[0].timestamp)
            ended_at = _parse_ts(turns[-1].timestamp)

            session_meta = storage.create_session(
                ["claude"], cwd=cwd, source="import", create_events=False,
                started_at=started_at or None,
            )
            session_id = session_meta.session_id

            # Update meta.json with original end timestamp
            meta_path = storage.root / "sessions" / session_id / "meta.json"
            meta_dict = json.loads(meta_path.read_text(encoding="utf-8"))
            if ended_at:
                meta_dict["ended_at"] = ended_at
            meta_path.write_text(
                json.dumps(meta_dict, indent=2) + "\n", encoding="utf-8"
            )

            chunks = chunks_from_turns(turns, session_id)
            storage.write_chunks(session_id, chunks)
            storage.finalize_session(session_id)

            # Restore ended_at from original
            meta_dict = json.loads(meta_path.read_text(encoding="utf-8"))
            if ended_at:
                meta_dict["ended_at"] = ended_at
            meta_path.write_text(
                json.dumps(meta_dict, indent=2) + "\n", encoding="utf-8"
            )

            imported_map[claude_uuid] = session_id
            storage.save_import_state(state)

            session_ids.append(session_id)
            self.imported += 1

        # Import artifacts after sessions
        self._import_artifacts(storage, cwd)

        return session_ids

    def _import_artifacts(self, storage: NdjsonStorage, cwd: str) -> None:
        """Discover and import artifacts (todos, plans, tasks) for the project."""
        import sys

        from mb.artifact_chunker import chunk_plan, chunk_task, chunk_todo_list
        from mb.claude_adapter import (
            discover_plan_slugs,
            discover_plans,
            discover_task_dirs,
            discover_todos,
        )
        from mb.models import PlanMeta, TaskItem, TodoItem, TodoList

        artifact_state = storage.load_artifact_import_state()
        artifacts = artifact_state.setdefault("artifacts", {"todos": {}, "plans": {}, "tasks": {}})
        artifacts.setdefault("todos", {})
        artifacts.setdefault("plans", {})
        artifacts.setdefault("tasks", {})

        # --- Todos ---
        todo_files = discover_todos(cwd)
        for todo_file in todo_files:
            stem = todo_file.stem
            session_uuid = stem.split("-agent-")[0] if "-agent-" in stem else stem
            agent_id = stem.split("-agent-")[1] if "-agent-" in stem else None

            if session_uuid in artifacts["todos"]:
                continue

            try:
                raw = json.loads(todo_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                print(f"Warning: skipping malformed todo file {todo_file}: {exc}", file=sys.stderr)
                continue

            # Parse items — raw can be a list of items or a dict with items key
            raw_items: list[dict] = raw if isinstance(raw, list) else raw.get("items", [])
            if not raw_items:
                continue

            items = tuple(TodoItem.from_dict(i) for i in raw_items if isinstance(i, dict))
            if not items:
                continue

            if self._dry_run:
                self.todos_imported += 1
                self.dry_run_todo_items += len(items)
                continue

            mtime = todo_file.stat().st_mtime
            todo_list = TodoList(
                session_id=session_uuid,
                agent_id=agent_id,
                items=items,
                file_path=str(todo_file),
                mtime=mtime,
            )

            # Store raw todo
            storage.write_todo(session_uuid, todo_list.to_dict())

            # Chunk and store
            chunks = chunk_todo_list(todo_list)
            if chunks:
                storage.write_artifact_chunks(chunks)

            artifacts["todos"][session_uuid] = True
            storage.save_artifact_import_state(artifact_state)
            self.todos_imported += 1

        # --- Plans ---
        # Discover slugs from session JSONL
        cached_slugs = set(artifact_state.get("plan_slugs", []))
        discovered_slugs = discover_plan_slugs(cwd)
        all_slugs = cached_slugs | discovered_slugs
        artifact_state["plan_slugs"] = sorted(all_slugs)

        plan_files = discover_plans(all_slugs)
        for plan_file in plan_files:
            slug = plan_file.stem

            if slug in artifacts["plans"]:
                continue

            try:
                content_md = plan_file.read_text(encoding="utf-8")
            except OSError as exc:
                print(f"Warning: skipping unreadable plan {plan_file}: {exc}", file=sys.stderr)
                continue

            if not content_md.strip():
                continue

            if self._dry_run:
                self.plans_imported += 1
                continue

            mtime = plan_file.stat().st_mtime
            # Find which session referenced this slug (use first match)
            session_id_for_plan = ""
            for slug_session in _find_sessions_for_slug(cwd, slug):
                session_id_for_plan = slug_session
                break

            plan_meta = PlanMeta(
                slug=slug,
                session_id=session_id_for_plan,
                file_path=str(plan_file),
                mtime=mtime,
            )

            # Store raw plan
            storage.write_plan(slug, content_md, plan_meta.to_dict())

            # Chunk and store
            chunks = chunk_plan(slug, content_md, mtime)
            if chunks:
                storage.write_artifact_chunks(chunks)

            artifacts["plans"][slug] = True
            storage.save_artifact_import_state(artifact_state)
            self.plans_imported += 1

        # --- Tasks ---
        task_dirs = discover_task_dirs(cwd)
        for task_dir in task_dirs:
            session_uuid = task_dir.name

            if session_uuid in artifacts["tasks"]:
                continue

            task_files = [
                f for f in sorted(task_dir.iterdir())
                if f.suffix == ".json"
                and f.name not in (".lock", ".highwatermark")
                and not f.name.startswith(".")
            ]

            if not task_files:
                continue

            if self._dry_run:
                self.tasks_imported += 1
                self.dry_run_task_items += len(task_files)
                continue

            all_chunks = []
            for task_file in task_files:
                try:
                    raw = json.loads(task_file.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as exc:
                    print(
                        f"Warning: skipping malformed task file {task_file}: {exc}",
                        file=sys.stderr,
                    )
                    continue

                raw["session_id"] = session_uuid
                task_item = TaskItem.from_dict(raw)

                # Store individual task
                storage.write_task(session_uuid, task_item.id, task_item.to_dict())

                chunk = chunk_task(task_item)
                all_chunks.append(chunk)

            if all_chunks:
                storage.write_artifact_chunks(all_chunks)

            artifacts["tasks"][session_uuid] = True
            storage.save_artifact_import_state(artifact_state)
            self.tasks_imported += 1

        # Save final state
        storage.save_artifact_import_state(artifact_state)


def _find_sessions_for_slug(cwd: str, slug: str) -> list[str]:
    """Find session UUIDs that reference a given plan slug."""
    from mb.claude_adapter import encode_project_dir

    claude_projects = Path.home() / ".claude" / "projects"
    if not claude_projects.exists():
        return []

    project_dir = claude_projects / encode_project_dir(cwd)
    if not project_dir.exists():
        return []

    result: list[str] = []
    for f in project_dir.iterdir():
        if not f.suffix == ".jsonl" or f.name.startswith("agent-"):
            continue
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("slug") == slug:
                    result.append(f.stem)
                    break
    return result
