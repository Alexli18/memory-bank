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
        )

    def to_dict(self) -> dict[str, Any]:
        return {
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


def _generate_event_id(session_id: str, ts: float) -> str:
    """Generate a deterministic event_id from session_id and timestamp."""
    raw = f"{session_id}:{ts}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
