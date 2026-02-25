"""Storage abstraction layer — Protocol-based store with NdjsonStorage implementation."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Protocol, runtime_checkable

from mb.models import Chunk, Event, ProjectState, SessionMeta
from mb.redactor import Redactor

logger = logging.getLogger(__name__)

MEMORY_BANK_DIR = ".memory-bank"
CONFIG_VERSION = "1.0"

DEFAULT_CONFIG: dict[str, Any] = {
    "version": CONFIG_VERSION,
    "ollama": {
        "base_url": "http://localhost:11434",
        "embed_model": "nomic-embed-text",
        "chat_model": "gemma3:4b",
    },
    "chunking": {
        "max_tokens": 512,
        "overlap_tokens": 50,
    },
}


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class SessionStore(Protocol):
    def create_session(
        self,
        command: list[str],
        cwd: str | None = None,
        source: str | None = None,
        create_events: bool = True,
        started_at: float | None = None,
    ) -> SessionMeta: ...

    def finalize_session(self, session_id: str, exit_code: int | None = None) -> None: ...

    def delete_session(self, session_id: str) -> bool: ...

    def list_sessions(self) -> list[SessionMeta]: ...

    def read_meta(self, session_id: str) -> SessionMeta | None: ...


@runtime_checkable
class EventStore(Protocol):
    def write_event(
        self,
        session_id: str,
        stream: str,
        role: str,
        content: str,
        ts: float | None = None,
    ) -> None: ...

    def read_events(self, session_id: str) -> list[Event]: ...


@runtime_checkable
class ChunkStore(Protocol):
    def write_chunks(self, session_id: str, chunks: list[Chunk]) -> None: ...

    def read_chunks(self, session_id: str) -> list[Chunk]: ...

    def iter_all_chunks(self) -> Iterator[Chunk]: ...

    def has_chunks(self, session_id: str) -> bool: ...


@runtime_checkable
class StateStore(Protocol):
    def save_state(self, state: ProjectState) -> None: ...

    def load_state(self) -> ProjectState | None: ...

    def is_stale(self) -> bool: ...


# Type alias for a full storage backend.
# At runtime NdjsonStorage is the only implementation;
# the alias is used for type annotations in function signatures.
if TYPE_CHECKING:
    from typing import TypeAlias
    Storage: TypeAlias = "NdjsonStorage"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_session_id(started_at: float | None = None) -> str:
    if started_at and started_at > 0:
        now = datetime.fromtimestamp(started_at, tz=timezone.utc)
    else:
        now = datetime.now(timezone.utc)
    hex_suffix = os.urandom(2).hex()
    return now.strftime("%Y%m%d-%H%M%S") + f"-{hex_suffix}"


def _ensure_gitignore(storage: Path) -> None:
    project_root = storage.parent
    gitignore = project_root / ".gitignore"
    entry = MEMORY_BANK_DIR + "/"

    if gitignore.exists():
        content = gitignore.read_text(encoding="utf-8")
        if entry in content.splitlines():
            return
        if content and not content.endswith("\n"):
            content += "\n"
        content += entry + "\n"
        gitignore.write_text(content, encoding="utf-8")
    else:
        gitignore.write_text(entry + "\n", encoding="utf-8")


class MbStorageError(Exception):
    """Error raised for corrupt or unreadable storage files."""


# ---------------------------------------------------------------------------
# NdjsonStorage — single class implementing all four protocols
# ---------------------------------------------------------------------------


class NdjsonStorage:
    """NDJSON/JSON file-based storage implementing all Store protocols."""

    def __init__(self, root: Path, redactor: Redactor | None = None) -> None:
        self.root = root
        self._redactor = redactor or Redactor()

    # -- Factory methods ----------------------------------------------------

    @staticmethod
    def init(root: Path | None = None) -> tuple[bool, NdjsonStorage]:
        """Initialize .memory-bank/ directory structure.

        Returns (created, storage). created is False if already initialized.
        """
        storage_path = root or (Path.cwd() / MEMORY_BANK_DIR)

        if (storage_path / "config.json").exists():
            return False, NdjsonStorage(storage_path)

        storage_path.mkdir(exist_ok=True)
        (storage_path / "sessions").mkdir(exist_ok=True)
        (storage_path / "index").mkdir(exist_ok=True)
        (storage_path / "state").mkdir(exist_ok=True)

        config_path = storage_path / "config.json"
        config_path.write_text(
            json.dumps(DEFAULT_CONFIG, indent=2) + "\n", encoding="utf-8"
        )

        _ensure_gitignore(storage_path)

        # Auto-register in global project registry
        from mb.registry import register_project

        register_project(str(storage_path.parent.resolve()))

        return True, NdjsonStorage(storage_path)

    @staticmethod
    def open(root: Path | None = None) -> NdjsonStorage:
        """Open existing storage. Raises FileNotFoundError if not initialized."""
        storage_path = root or (Path.cwd() / MEMORY_BANK_DIR)
        if not (storage_path / "config.json").exists():
            raise FileNotFoundError(
                f"Memory Bank not initialized. Run `mb init` first. (looked in {storage_path})"
            )
        return NdjsonStorage(storage_path)

    # -- Config helpers -----------------------------------------------------

    def read_config(self) -> dict[str, Any]:
        config_path = self.root / "config.json"
        try:
            result: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise MbStorageError(f"Corrupt config.json: {e}") from e
        return result

    def write_config(self, config: dict[str, Any]) -> None:
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    # -- Hooks / Import state helpers ---------------------------------------

    def load_hooks_state(self) -> dict[str, Any]:
        path = self.root / "hooks_state.json"
        if not path.exists():
            return {"sessions": {}}
        result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return result

    def save_hooks_state(self, state: dict[str, Any]) -> None:
        path = self.root / "hooks_state.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        tmp.rename(path)

    def load_import_state(self) -> dict[str, Any]:
        path = self.root / "import_state.json"
        if not path.exists():
            return {"imported": {}}
        try:
            result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"imported": {}}
        return result

    def save_import_state(self, state: dict[str, Any]) -> None:
        path = self.root / "import_state.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        tmp.rename(path)

    # -- SessionStore -------------------------------------------------------

    def create_session(
        self,
        command: list[str],
        cwd: str | None = None,
        source: str | None = None,
        create_events: bool = True,
        started_at: float | None = None,
    ) -> SessionMeta:
        session_id = _generate_session_id(started_at)
        session_dir = self.root / "sessions" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        ts = started_at if started_at and started_at > 0 else time.time()
        meta_dict: dict[str, Any] = {
            "session_id": session_id,
            "command": command,
            "cwd": cwd or str(Path.cwd()),
            "started_at": ts,
            "ended_at": None,
            "exit_code": None,
        }
        if source is not None:
            meta_dict["source"] = source

        (session_dir / "meta.json").write_text(
            json.dumps(meta_dict, indent=2) + "\n", encoding="utf-8"
        )

        if create_events:
            (session_dir / "events.jsonl").touch()

        return SessionMeta.from_dict(meta_dict)

    def finalize_session(self, session_id: str, exit_code: int | None = None) -> None:
        meta_path = self.root / "sessions" / session_id / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["ended_at"] = time.time()
        if exit_code is not None:
            meta["exit_code"] = exit_code
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    def delete_session(self, session_id: str) -> bool:
        session_dir = self.root / "sessions" / session_id
        if not session_dir.exists():
            return False
        shutil.rmtree(session_dir)
        return True

    def list_sessions(self) -> list[SessionMeta]:
        sessions_dir = self.root / "sessions"
        if not sessions_dir.exists():
            return []

        sessions: list[SessionMeta] = []
        for entry in sessions_dir.iterdir():
            if not entry.is_dir():
                continue
            meta_path = entry / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta_dict = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("Skipping corrupt session %s", entry.name)
                continue
            sessions.append(SessionMeta.from_dict(meta_dict))

        sessions.sort(key=lambda s: s.started_at, reverse=True)
        return sessions

    def read_meta(self, session_id: str) -> SessionMeta | None:
        meta_path = self.root / "sessions" / session_id / "meta.json"
        if not meta_path.exists():
            return None
        try:
            meta_dict = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return SessionMeta.from_dict(meta_dict)

    # -- EventStore ---------------------------------------------------------

    def write_event(
        self,
        session_id: str,
        stream: str,
        role: str,
        content: str,
        ts: float | None = None,
    ) -> None:
        events_path = self.root / "sessions" / session_id / "events.jsonl"

        event = {
            "ts": ts if ts is not None else time.monotonic(),
            "session_id": session_id,
            "stream": stream,
            "role": role,
            "content": self._redactor.redact(content),
        }
        line = json.dumps(event, ensure_ascii=False) + "\n"

        with events_path.open("a", encoding="utf-8") as f:
            f.write(line)

    def read_events(self, session_id: str) -> list[Event]:
        events_path = self.root / "sessions" / session_id / "events.jsonl"
        if not events_path.exists():
            return []
        events: list[Event] = []
        with events_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                events.append(Event.from_dict(json.loads(line)))
        return events

    # -- ChunkStore ---------------------------------------------------------

    def write_chunks(self, session_id: str, chunks: list[Chunk]) -> None:
        session_dir = self.root / "sessions" / session_id
        chunks_path = session_dir / "chunks.jsonl"
        with chunks_path.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    def read_chunks(self, session_id: str) -> list[Chunk]:
        chunks_path = self.root / "sessions" / session_id / "chunks.jsonl"
        if not chunks_path.exists():
            return []
        chunks: list[Chunk] = []
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chunks.append(Chunk.from_dict(json.loads(line)))
        return chunks

    def iter_all_chunks(self) -> Iterator[Chunk]:
        sessions_dir = self.root / "sessions"
        if sessions_dir.exists():
            for session_dir in sorted(sessions_dir.iterdir()):
                if not session_dir.is_dir():
                    continue
                chunks_path = session_dir / "chunks.jsonl"
                if not chunks_path.exists():
                    continue
                with chunks_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        yield Chunk.from_dict(json.loads(line))

        # Also yield artifact chunks if they exist
        artifact_chunks_path = self.artifacts_dir / "chunks.jsonl"
        if artifact_chunks_path.exists():
            with artifact_chunks_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    yield Chunk.from_dict(json.loads(line))

    def has_chunks(self, session_id: str) -> bool:
        chunks_path = self.root / "sessions" / session_id / "chunks.jsonl"
        return chunks_path.exists() and chunks_path.stat().st_size > 0

    # -- StateStore ---------------------------------------------------------

    def save_state(self, state: ProjectState) -> None:
        state_dir = self.root / "state"
        state_dir.mkdir(exist_ok=True)
        state_path = state_dir / "state.json"
        state_path.write_text(
            json.dumps(state.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def load_state(self) -> ProjectState | None:
        state_path = self.root / "state" / "state.json"
        if not state_path.exists():
            return None
        data: dict[str, Any] = json.loads(state_path.read_text(encoding="utf-8"))
        return ProjectState.from_dict(data)

    def is_stale(self) -> bool:
        state_path = self.root / "state" / "state.json"
        if not state_path.exists():
            return False
        sessions_dir = self.root / "sessions"
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

    # -- Artifact Storage ---------------------------------------------------

    @property
    def artifacts_dir(self) -> Path:
        """Return the path to .memory-bank/artifacts/."""
        return self.root / "artifacts"

    def write_artifact_chunks(self, chunks: list[Chunk]) -> None:
        """Append artifact chunks to artifacts/chunks.jsonl."""
        artifacts = self.artifacts_dir
        artifacts.mkdir(exist_ok=True)
        chunks_path = artifacts / "chunks.jsonl"
        with chunks_path.open("a", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    def read_artifact_chunks(self) -> list[Chunk]:
        """Read all chunks from artifacts/chunks.jsonl."""
        chunks_path = self.artifacts_dir / "chunks.jsonl"
        if not chunks_path.exists():
            return []
        chunks: list[Chunk] = []
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chunks.append(Chunk.from_dict(json.loads(line)))
        return chunks

    def write_todo(self, session_id: str, data: dict[str, Any]) -> None:
        """Write a todo list JSON to artifacts/todos/{session_id}.json."""
        todos_dir = self.artifacts_dir / "todos"
        todos_dir.mkdir(parents=True, exist_ok=True)
        path = todos_dir / f"{session_id}.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def write_plan(self, slug: str, content_md: str, meta_dict: dict[str, Any]) -> None:
        """Write plan Markdown and metadata to artifacts/plans/."""
        plans_dir = self.artifacts_dir / "plans"
        plans_dir.mkdir(parents=True, exist_ok=True)
        (plans_dir / f"{slug}.md").write_text(content_md, encoding="utf-8")
        (plans_dir / f"{slug}.meta.json").write_text(
            json.dumps(meta_dict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def write_task(self, session_id: str, task_id: str, data: dict[str, Any]) -> None:
        """Write a task JSON to artifacts/tasks/{session_id}/{task_id}.json."""
        tasks_dir = self.artifacts_dir / "tasks" / session_id
        tasks_dir.mkdir(parents=True, exist_ok=True)
        path = tasks_dir / f"{task_id}.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def load_artifact_import_state(self) -> dict[str, Any]:
        """Load artifact sections from import_state.json."""
        state = self.load_import_state()
        return {
            "artifacts": state.get("artifacts", {"todos": {}, "plans": {}, "tasks": {}}),
            "plan_slugs": state.get("plan_slugs", []),
        }

    def count_artifacts(self) -> dict[str, Any]:
        """Count artifacts in .memory-bank/artifacts/ directories.

        Returns dict with keys: plans, todos, todo_active_items, tasks,
        task_pending. Returns empty dict if artifacts/ doesn't exist.
        """
        artifacts = self.artifacts_dir
        if not artifacts.exists():
            return {}

        plans = 0
        todos = 0
        todo_active_items = 0
        tasks = 0
        task_pending = 0

        # Count plans
        plans_dir = artifacts / "plans"
        if plans_dir.exists():
            plans = len(list(plans_dir.glob("*.md")))

        # Count todos and active items
        todos_dir = artifacts / "todos"
        if todos_dir.exists():
            for f in todos_dir.glob("*.json"):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                todos += 1
                raw_items = data if isinstance(data, list) else data.get("items", [])
                for item in raw_items:
                    status = item.get("status", "pending")
                    if status != "completed":
                        todo_active_items += 1

        # Count task sessions and pending tasks
        tasks_dir = artifacts / "tasks"
        if tasks_dir.exists():
            for session_dir in tasks_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                tasks += 1
                for tf in session_dir.glob("*.json"):
                    try:
                        task_data = json.loads(tf.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        continue
                    status = task_data.get("status", "pending")
                    if status in ("pending", "in_progress"):
                        task_pending += 1

        if plans == 0 and todos == 0 and tasks == 0:
            return {}

        return {
            "plans": plans,
            "todos": todos,
            "todo_active_items": todo_active_items,
            "tasks": tasks,
            "task_pending": task_pending,
        }

    def save_artifact_import_state(self, artifact_state: dict[str, Any]) -> None:
        """Save artifact sections back into import_state.json."""
        state = self.load_import_state()
        state["artifacts"] = artifact_state.get("artifacts", {"todos": {}, "plans": {}, "tasks": {}})
        state["plan_slugs"] = artifact_state.get("plan_slugs", [])
        self.save_import_state(state)
