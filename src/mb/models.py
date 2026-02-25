"""Canonical data models for Memory Bank domain entities."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventSource(Enum):
    """Ingestion source of a session."""

    PTY = "pty"
    HOOK = "hook"
    IMPORT = "import"


class PackFormat(Enum):
    """Output format for context packs."""

    XML = "xml"
    JSON = "json"
    MARKDOWN = "md"


def quality_score(text: str) -> float:
    """Score chunk quality: ratio of alphanumeric content to total length."""
    if not text or not text.strip():
        return 0.0
    stripped = text.strip()
    alnum_count = sum(1 for c in stripped if c.isalnum())
    return round(alnum_count / len(stripped), 3) if stripped else 0.0


@dataclass(frozen=True, slots=True)
class Event:
    """A single timestamped entry from a captured session."""

    event_id: str
    ts: float
    session_id: str
    stream: str
    role: str
    content: str
    source: EventSource | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        event_id = data.get("event_id") or _generate_event_id(
            data.get("session_id", ""), data.get("ts", 0)
        )
        source_val = data.get("source")
        source = None
        if source_val is not None:
            try:
                source = EventSource(source_val)
            except ValueError:
                source = None
        return cls(
            event_id=event_id,
            ts=data.get("ts", 0.0),
            session_id=data.get("session_id", ""),
            stream=data.get("stream", ""),
            role=data.get("role", ""),
            content=data.get("content", ""),
            source=source,
            meta=data.get("meta", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "event_id": self.event_id,
            "ts": self.ts,
            "session_id": self.session_id,
            "stream": self.stream,
            "role": self.role,
            "content": self.content,
        }
        if self.source is not None:
            d["source"] = self.source.value
        if self.meta:
            d["meta"] = self.meta
        return d


@dataclass(frozen=True, slots=True)
class SessionMeta:
    """Metadata about a captured session."""

    session_id: str
    command: list[str]
    cwd: str
    started_at: float
    ended_at: float | None = None
    exit_code: int | None = None
    source: EventSource | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionMeta:
        source_val = data.get("source")
        source = None
        if source_val is not None:
            try:
                source = EventSource(source_val)
            except ValueError:
                source = None
        command = data.get("command", [])
        if isinstance(command, str):
            command = [command]
        return cls(
            session_id=data.get("session_id", ""),
            command=list(command),
            cwd=data.get("cwd", ""),
            started_at=data.get("started_at", 0.0),
            ended_at=data.get("ended_at"),
            exit_code=data.get("exit_code"),
            source=source,
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "session_id": self.session_id,
            "command": self.command,
            "cwd": self.cwd,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "exit_code": self.exit_code,
        }
        if self.source is not None:
            d["source"] = self.source.value
        return d


@dataclass(frozen=True, slots=True)
class Chunk:
    """A semantically meaningful text segment extracted from session events."""

    chunk_id: str
    session_id: str
    index: int
    text: str
    ts_start: float
    ts_end: float
    token_estimate: int
    quality_score: float
    _extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        known_keys = {
            "chunk_id", "session_id", "index", "text",
            "ts_start", "ts_end", "token_estimate", "quality_score",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}
        return cls(
            chunk_id=data.get("chunk_id", ""),
            session_id=data.get("session_id", ""),
            index=data.get("index", 0),
            text=data.get("text", ""),
            ts_start=data.get("ts_start", 0.0),
            ts_end=data.get("ts_end", 0.0),
            token_estimate=data.get("token_estimate", 0),
            quality_score=data.get("quality_score", 0.0),
            _extra=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "session_id": self.session_id,
            "index": self.index,
            "text": self.text,
            "ts_start": self.ts_start,
            "ts_end": self.ts_end,
            "token_estimate": self.token_estimate,
            "quality_score": self.quality_score,
        }
        d.update(self._extra)
        return d


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A chunk matched by semantic search with a relevance score."""

    chunk_id: str
    session_id: str
    index: int
    text: str
    ts_start: float
    ts_end: float
    token_estimate: int
    quality_score: float
    score: float
    artifact_type: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchResult:
        return cls(
            chunk_id=data.get("chunk_id", ""),
            session_id=data.get("session_id", ""),
            index=data.get("index", 0),
            text=data.get("text", ""),
            ts_start=data.get("ts_start", 0.0),
            ts_end=data.get("ts_end", 0.0),
            token_estimate=data.get("token_estimate", 0),
            quality_score=data.get("quality_score", 0.0),
            score=data.get("score", 0.0),
            artifact_type=data.get("artifact_type"),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "session_id": self.session_id,
            "index": self.index,
            "text": self.text,
            "ts_start": self.ts_start,
            "ts_end": self.ts_end,
            "token_estimate": self.token_estimate,
            "quality_score": self.quality_score,
            "score": self.score,
        }
        if self.artifact_type is not None:
            d["artifact_type"] = self.artifact_type
        return d


@dataclass(frozen=True, slots=True)
class TodoItem:
    """A single todo item from a Claude Code todo list."""

    id: str
    content: str
    status: str = "pending"
    priority: str = "medium"
    active_form: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TodoItem:
        status = data.get("status", "pending")
        if status not in ("pending", "in_progress", "completed"):
            status = "pending"
        priority = data.get("priority", "medium")
        if priority not in ("high", "medium", "low"):
            priority = "medium"
        return cls(
            id=str(data.get("id", "")),
            content=data.get("content", ""),
            status=status,
            priority=priority,
            active_form=data.get("activeForm") or data.get("active_form"),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "content": self.content,
            "status": self.status,
            "priority": self.priority,
        }
        if self.active_form is not None:
            d["activeForm"] = self.active_form
        return d


@dataclass(frozen=True, slots=True)
class TodoList:
    """A complete todo list file associated with a session."""

    session_id: str
    agent_id: str | None
    items: tuple[TodoItem, ...]
    file_path: str
    mtime: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TodoList:
        raw_items = data.get("items", ())
        items = tuple(
            TodoItem.from_dict(i) if isinstance(i, dict) else i
            for i in raw_items
        )
        return cls(
            session_id=data.get("session_id", ""),
            agent_id=data.get("agent_id"),
            items=items,
            file_path=data.get("file_path", ""),
            mtime=data.get("mtime", 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "items": [i.to_dict() for i in self.items],
            "file_path": self.file_path,
            "mtime": self.mtime,
        }


@dataclass(frozen=True, slots=True)
class PlanMeta:
    """Metadata for an imported plan, stored alongside the plan Markdown."""

    slug: str
    session_id: str
    timestamp: str | None = None
    file_path: str = ""
    mtime: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanMeta:
        return cls(
            slug=data.get("slug", ""),
            session_id=data.get("session_id", ""),
            timestamp=data.get("timestamp"),
            file_path=data.get("file_path", ""),
            mtime=data.get("mtime", 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "file_path": self.file_path,
            "mtime": self.mtime,
        }


@dataclass(frozen=True, slots=True)
class TaskItem:
    """A single task from a Claude Code task tree."""

    id: str
    session_id: str
    subject: str = ""
    description: str = ""
    active_form: str | None = None
    status: str = "pending"
    blocks: tuple[str, ...] = ()
    blocked_by: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskItem:
        status = data.get("status", "pending")
        if status not in ("pending", "in_progress", "completed", "deleted"):
            status = "pending"
        blocks = tuple(str(x) for x in data.get("blocks", ()))
        blocked_by = tuple(str(x) for x in data.get("blockedBy", data.get("blocked_by", ())))
        return cls(
            id=str(data.get("id", "")),
            session_id=data.get("session_id", ""),
            subject=data.get("subject", ""),
            description=data.get("description", ""),
            active_form=data.get("activeForm") or data.get("active_form"),
            status=status,
            blocks=blocks,
            blocked_by=blocked_by,
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "session_id": self.session_id,
            "subject": self.subject,
            "description": self.description,
            "status": self.status,
            "blocks": list(self.blocks),
            "blockedBy": list(self.blocked_by),
        }
        if self.active_form is not None:
            d["activeForm"] = self.active_form
        return d


@dataclass(frozen=True, slots=True)
class ProjectState:
    """An LLM-generated summary of the project state."""

    summary: str
    decisions: list[dict[str, Any]]
    constraints: list[str]
    tasks: list[dict[str, Any]]
    updated_at: float
    source_sessions: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectState:
        return cls(
            summary=data.get("summary", ""),
            decisions=list(data.get("decisions", [])),
            constraints=list(data.get("constraints", [])),
            tasks=list(data.get("tasks", [])),
            updated_at=data.get("updated_at", 0.0),
            source_sessions=list(data.get("source_sessions", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "decisions": self.decisions,
            "constraints": self.constraints,
            "tasks": self.tasks,
            "updated_at": self.updated_at,
            "source_sessions": self.source_sessions,
        }


@dataclass(frozen=True, slots=True)
class ProjectEntry:
    """A registered Memory Bank project in the global registry."""

    path: str
    registered_at: float
    last_import: float = 0.0
    session_count: int = 0

    @classmethod
    def from_dict(cls, path: str, data: dict[str, Any]) -> ProjectEntry:
        return cls(
            path=path,
            registered_at=data.get("registered_at", 0.0),
            last_import=data.get("last_import", 0.0),
            session_count=data.get("session_count", 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "registered_at": self.registered_at,
            "last_import": self.last_import,
            "session_count": self.session_count,
        }


@dataclass(frozen=True, slots=True)
class GlobalSearchResult:
    """A search result with cross-project attribution."""

    project_path: str
    chunk_id: str
    session_id: str
    index: int
    text: str
    ts_start: float
    ts_end: float
    token_estimate: int
    quality_score: float
    score: float
    artifact_type: str | None = None

    @classmethod
    def from_search_result(cls, result: SearchResult, project_path: str) -> GlobalSearchResult:
        return cls(
            project_path=project_path,
            chunk_id=result.chunk_id,
            session_id=result.session_id,
            index=result.index,
            text=result.text,
            ts_start=result.ts_start,
            ts_end=result.ts_end,
            token_estimate=result.token_estimate,
            quality_score=result.quality_score,
            score=result.score,
            artifact_type=result.artifact_type,
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "project_path": self.project_path,
            "chunk_id": self.chunk_id,
            "session_id": self.session_id,
            "index": self.index,
            "text": self.text,
            "ts_start": self.ts_start,
            "ts_end": self.ts_end,
            "token_estimate": self.token_estimate,
            "quality_score": self.quality_score,
            "score": self.score,
        }
        if self.artifact_type is not None:
            d["artifact_type"] = self.artifact_type
        return d


def _generate_event_id(session_id: str, ts: float) -> str:
    """Generate a deterministic event_id from session_id and timestamp."""
    raw = f"{session_id}:{ts}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
