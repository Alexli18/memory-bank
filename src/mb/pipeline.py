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
    """Import source — imports historical Claude Code sessions."""

    def __init__(self, dry_run: bool = False) -> None:
        self._dry_run = dry_run
        self.imported: int = 0
        self.skipped: int = 0

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
            )
            session_id = session_meta.session_id

            # Update meta.json with original timestamps
            meta_path = storage.root / "sessions" / session_id / "meta.json"
            meta_dict = json.loads(meta_path.read_text(encoding="utf-8"))
            if started_at:
                meta_dict["started_at"] = started_at
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

        return session_ids
