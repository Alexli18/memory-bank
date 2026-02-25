"""Retriever protocol and implementations for context pack chunk selection.

Extracted from pack.py ``_collect_recent_excerpts`` — provides pluggable
retrieval strategies for selecting chunks to include in context packs.
"""

from __future__ import annotations

import heapq
import re
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Protocol

from mb.models import Chunk, quality_score
from mb.store import NdjsonStorage

if TYPE_CHECKING:
    from mb.graph import EpisodeType


_WHITESPACE_RE = re.compile(r"\s+")

NEAR_DEDUP_THRESHOLD = 0.70


def _normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace for exact dedup hashing."""
    return _WHITESPACE_RE.sub(" ", text.lower()).strip()


def _deduplicate_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Remove exact and near-duplicate chunks.

    Two-phase dedup:
    1. **Exact**: hash normalized text (lowercase, whitespace-collapsed).
       Keep chunk with higher ``quality_score``; tie-break by ``ts_end``.
    2. **Near**: pairwise ``SequenceMatcher.ratio()`` on remaining chunks.
       If ratio > ``NEAR_DEDUP_THRESHOLD``, remove the lower-quality chunk.

    Order of the input list is preserved for the surviving chunks.
    """
    if len(chunks) <= 1:
        return chunks

    # --- Phase 1: exact dedup ---
    best_by_hash: dict[str, Chunk] = {}
    for chunk in chunks:
        key = _normalize_text(chunk.text)
        existing = best_by_hash.get(key)
        if existing is None:
            best_by_hash[key] = chunk
        elif (chunk.quality_score, chunk.ts_end) > (existing.quality_score, existing.ts_end):
            best_by_hash[key] = chunk

    survivors = {id(c) for c in best_by_hash.values()}
    deduped = [c for c in chunks if id(c) in survivors]

    # --- Phase 2: near dedup ---
    removed: set[int] = set()
    for i in range(len(deduped)):
        if id(deduped[i]) in removed:
            continue
        for j in range(i + 1, len(deduped)):
            if id(deduped[j]) in removed:
                continue
            ratio = SequenceMatcher(
                None, deduped[i].text, deduped[j].text
            ).ratio()
            if ratio > NEAR_DEDUP_THRESHOLD:
                # Keep the one with higher quality; tie-break by ts_end
                if (deduped[j].quality_score, deduped[j].ts_end) > (
                    deduped[i].quality_score, deduped[i].ts_end
                ):
                    removed.add(id(deduped[i]))
                    break  # i is removed, skip rest of j-loop
                else:
                    removed.add(id(deduped[j]))

    return [c for c in deduped if id(c) not in removed]


class Retriever(Protocol):
    """Protocol for chunk retrieval strategies."""

    def retrieve(self, storage: NdjsonStorage) -> list[Chunk]: ...


class RecencyRetriever:
    """Select most recent high-quality chunks.

    Uses a min-heap keyed by effective quality (with optional decay) so
    that at most *max_excerpts* chunks are kept in memory at any time.
    Chunks with quality_score below *min_quality* or stripped text shorter
    than *min_length* characters are filtered out.

    When *half_life_days* > 0, conversation chunks (non-artifact) have
    their quality decayed exponentially.  Artifact chunks always use
    original quality.
    """

    def __init__(
        self,
        *,
        min_quality: float = 0.30,
        min_length: int = 30,
        max_excerpts: int = 200,
        half_life_days: float = 0.0,
    ) -> None:
        self.min_quality = min_quality
        self.min_length = min_length
        self.max_excerpts = max_excerpts
        self.half_life_days = half_life_days

    def retrieve(self, storage: NdjsonStorage) -> list[Chunk]:
        from mb.decay import decayed_quality

        heap: list[tuple[float, float, int, Chunk]] = []
        counter = 0

        for chunk in storage.iter_all_chunks():
            text = chunk.text
            if len(text.strip()) < self.min_length:
                continue
            q = chunk.quality_score if chunk.quality_score > 0 else quality_score(text)

            # Apply decay to conversation chunks only
            if self.half_life_days > 0 and chunk._extra.get("artifact_type") is None:
                q_eff = decayed_quality(q, chunk.ts_end, self.half_life_days)
            else:
                q_eff = q

            if q_eff < self.min_quality:
                continue

            ts_end = chunk.ts_end
            if len(heap) < self.max_excerpts:
                heapq.heappush(heap, (q_eff, ts_end, counter, chunk))
            elif (q_eff, ts_end) > (heap[0][0], heap[0][1]):
                heapq.heapreplace(heap, (q_eff, ts_end, counter, chunk))
            counter += 1

        result = [entry[3] for entry in heap]
        result.sort(key=lambda c: c.ts_end, reverse=True)
        result = _deduplicate_chunks(result)
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
