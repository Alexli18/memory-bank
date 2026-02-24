"""Retriever protocol and implementations for context pack chunk selection.

Extracted from pack.py ``_collect_recent_excerpts`` — provides pluggable
retrieval strategies for selecting chunks to include in context packs.
"""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING, Protocol

from mb.models import Chunk, quality_score
from mb.store import NdjsonStorage

if TYPE_CHECKING:
    from mb.graph import EpisodeType


class Retriever(Protocol):
    """Protocol for chunk retrieval strategies."""

    def retrieve(self, storage: NdjsonStorage) -> list[Chunk]: ...


class RecencyRetriever:
    """Select most recent high-quality chunks.

    Uses a min-heap keyed by ``ts_end`` so that at most *max_excerpts*
    chunks are kept in memory at any time.  Chunks with quality_score
    below *min_quality* or stripped text shorter than *min_length*
    characters are filtered out.
    """

    def __init__(
        self,
        *,
        min_quality: float = 0.30,
        min_length: int = 30,
        max_excerpts: int = 200,
    ) -> None:
        self.min_quality = min_quality
        self.min_length = min_length
        self.max_excerpts = max_excerpts

    def retrieve(self, storage: NdjsonStorage) -> list[Chunk]:
        heap: list[tuple[float, int, Chunk]] = []
        counter = 0

        for chunk in storage.iter_all_chunks():
            text = chunk.text
            if len(text.strip()) < self.min_length:
                continue
            q = chunk.quality_score if chunk.quality_score > 0 else quality_score(text)
            if q < self.min_quality:
                continue

            ts_end = chunk.ts_end
            if len(heap) < self.max_excerpts:
                heapq.heappush(heap, (ts_end, counter, chunk))
            elif ts_end > heap[0][0]:
                heapq.heapreplace(heap, (ts_end, counter, chunk))
            counter += 1

        result = [entry[2] for entry in heap]
        result.sort(key=lambda c: c.ts_end, reverse=True)
        return result


class ContextualRetriever:
    """Episode-aware and failure-aware chunk retrieval.

    Provides two retrieval modes:
    - ``retrieve_around_failure``: returns chunks from a failed session and its
      temporal neighbors (useful for debugging context).
    - ``retrieve_by_episode``: returns chunks filtered by episode type.
    """

    def __init__(self, *, max_chunks: int = 200) -> None:
        self.max_chunks = max_chunks

    def retrieve_around_failure(
        self, storage: NdjsonStorage, session_id: str
    ) -> list[Chunk]:
        """Return chunks from *session_id* and its temporal neighbors."""
        from mb.graph import SessionGraph

        meta = storage.read_meta(session_id)
        if meta is None:
            return []

        all_metas = storage.list_sessions()
        graph = SessionGraph()
        related_ids = graph.find_related_sessions(session_id, all_metas)

        target_ids = {session_id, *related_ids}
        chunks: list[Chunk] = []
        for sid in target_ids:
            chunks.extend(storage.read_chunks(sid))

        chunks.sort(key=lambda c: c.ts_end, reverse=True)
        return chunks[: self.max_chunks]

    def retrieve_by_episode(
        self, storage: NdjsonStorage, episode_type: EpisodeType
    ) -> list[Chunk]:
        """Return chunks from sessions matching *episode_type*."""
        from mb.graph import SessionGraph

        graph = SessionGraph()
        all_metas = storage.list_sessions()

        matching_ids: set[str] = set()
        for meta in all_metas:
            session_chunks = storage.read_chunks(meta.session_id)
            if graph.classify_episode(meta, session_chunks) == episode_type:
                matching_ids.add(meta.session_id)

        chunks: list[Chunk] = []
        for sid in matching_ids:
            chunks.extend(storage.read_chunks(sid))

        chunks.sort(key=lambda c: c.ts_end, reverse=True)
        return chunks[: self.max_chunks]
