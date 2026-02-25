"""Embedding index and cosine similarity search."""

from __future__ import annotations

import json
import logging
from itertools import groupby
from operator import attrgetter
from pathlib import Path
from typing import Any

import numpy as np

from mb.chunker import chunk_all_sessions
from mb.models import GlobalSearchResult, SearchResult
from mb.ollama_client import OllamaClient
from mb.store import NdjsonStorage

logger = logging.getLogger(__name__)


VECTOR_DIM = 768
VECTOR_BYTES = VECTOR_DIM * 4  # float32


class VectorIndex:
    """Append-only vector index backed by vectors.bin + metadata.jsonl."""

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.vectors_path = index_dir / "vectors.bin"
        self.metadata_path = index_dir / "metadata.jsonl"

    def add(self, vector: list[float], metadata: dict[str, Any]) -> None:
        """Normalize and append a vector with its metadata."""
        arr = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm

        # Append raw float32 bytes
        with self.vectors_path.open("ab") as f:
            f.write(arr.tobytes())

        # Append metadata line
        with self.metadata_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        artifact_type: str | None = None,
        half_life_days: float = 0.0,
        no_decay: bool = False,
    ) -> list[SearchResult]:
        """Search for top-K similar vectors by cosine similarity.

        Uses memory-mapped I/O for the vectors file so that only the pages
        touched by the matrix multiplication are read from disk.  Metadata
        is loaded only for the top-K indices.

        Args:
            query_vector: Query embedding (will be normalized).
            top_k: Number of results to return.
            artifact_type: If set, filter results by this artifact type.
                Use "session" for conversation chunks (artifact_type is None in metadata).

        Returns:
            List of SearchResult objects.
        """
        if not self.vectors_path.exists() or self.vectors_path.stat().st_size == 0:
            return []

        file_size = self.vectors_path.stat().st_size
        n_vectors = file_size // VECTOR_BYTES

        # Integrity check: verify metadata line count matches
        n_metadata = self._count_metadata_lines()
        if n_metadata != n_vectors:
            logger.warning(
                "Index integrity: %d vectors vs %d metadata; truncating",
                n_vectors, n_metadata,
            )
            n_vectors = min(n_vectors, n_metadata)

        if n_vectors == 0:
            return []

        # Memory-mapped vectors — OS manages paging
        matrix = np.memmap(
            self.vectors_path, dtype=np.float32, mode="r", shape=(n_vectors, VECTOR_DIM)
        )

        # Normalize query
        query = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        # Cosine similarity (vectors already normalized)
        scores = matrix @ query

        # Top-K via argpartition
        k = min(top_k, n_vectors)
        if k < n_vectors:
            top_indices = np.argpartition(scores, -k)[-k:]
        else:
            top_indices = np.arange(n_vectors)

        # Sort by score descending
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        apply_boost = half_life_days > 0 and not no_decay

        # For type filtering, we need to load more candidates
        if artifact_type is not None:
            # Load all metadata to filter by type, then pick top_k
            all_metadata = self._load_metadata()
            # Sort all indices by score descending
            sorted_indices = np.argsort(scores)[::-1]
            results: list[SearchResult] = []
            for idx in sorted_indices:
                if len(results) >= top_k:
                    break
                int_idx = int(idx)
                if int_idx >= len(all_metadata):
                    continue
                meta = all_metadata[int_idx]
                meta_type = meta.get("artifact_type")
                # "session" filter matches chunks with no artifact_type
                if artifact_type == "session":
                    if meta_type is not None:
                        continue
                elif meta_type != artifact_type:
                    continue
                cosine = float(scores[idx])
                meta["score"] = self._boosted_score(
                    cosine, meta, half_life_days,
                ) if apply_boost else cosine
                results.append(SearchResult.from_dict(meta))

            if apply_boost:
                results.sort(key=lambda r: r.score, reverse=True)
            return results

        # Load metadata only for the needed indices
        metadata_map = self._load_metadata_at_indices(set(top_indices))

        results = []
        for idx in top_indices:
            meta = metadata_map.get(int(idx), {})
            cosine = float(scores[idx])
            meta["score"] = self._boosted_score(
                cosine, meta, half_life_days,
            ) if apply_boost else cosine
            results.append(SearchResult.from_dict(meta))

        if apply_boost:
            results.sort(key=lambda r: r.score, reverse=True)
        return results

    @staticmethod
    def _boosted_score(
        cosine_score: float, meta: dict[str, Any], half_life_days: float,
    ) -> float:
        """Apply decay tiebreaker boost to a cosine score.

        Artifact chunks are not boosted (return raw cosine).
        """
        if meta.get("artifact_type") is not None:
            return cosine_score
        from mb.decay import decay_factor as _decay_factor

        ts_end = meta.get("ts_end", 0.0)
        return cosine_score * (1.0 + 0.1 * _decay_factor(ts_end, half_life_days))

    def _count_metadata_lines(self) -> int:
        """Count non-empty lines in metadata.jsonl without parsing JSON."""
        if not self.metadata_path.exists():
            return 0
        count = 0
        with self.metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def _load_metadata_at_indices(self, indices: set[int]) -> dict[int, dict[str, Any]]:
        """Load metadata only for the requested line indices (0-based).

        Iterates the file once and exits early when all requested indices
        have been collected.
        """
        if not self.metadata_path.exists() or not indices:
            return {}
        result: dict[int, dict[str, Any]] = {}
        idx = 0
        with self.metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                if idx in indices:
                    result[idx] = json.loads(line)
                    if len(result) == len(indices):
                        break
                idx += 1
        return result

    def _load_metadata(self) -> list[dict[str, Any]]:
        """Load all metadata lines."""
        if not self.metadata_path.exists():
            return []
        lines = []
        with self.metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(json.loads(line))
        return lines

    def indexed_sessions(self) -> set[str]:
        """Return set of session_ids already in the index."""
        metadata = self._load_metadata()
        return {m["session_id"] for m in metadata if "session_id" in m}

    def clear(self) -> None:
        """Remove all index data to force a full rebuild."""
        if self.vectors_path.exists():
            self.vectors_path.unlink()
        if self.metadata_path.exists():
            self.metadata_path.unlink()


def _index_is_stale(index: VectorIndex, sessions_dir: Path) -> bool:
    """Check if any session's chunks.jsonl is newer than the index."""
    if not index.metadata_path.exists():
        return False  # No index yet — nothing stale, will be built fresh
    index_mtime = index.metadata_path.stat().st_mtime
    for session_dir in sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue
        chunks_path = session_dir / "chunks.jsonl"
        if chunks_path.exists() and chunks_path.stat().st_mtime > index_mtime:
            return True
    return False


def build_index(storage: NdjsonStorage, ollama_client: OllamaClient) -> VectorIndex:
    """Build or incrementally update the embedding index.

    Iterates sessions, chunks them, embeds via Ollama, appends to index.
    Skips sessions already indexed.  Rebuilds if chunks are newer than index.
    """
    index_dir = storage.root / "index"
    index_dir.mkdir(exist_ok=True)
    index = VectorIndex(index_dir)

    sessions_dir = storage.root / "sessions"
    if not sessions_dir.exists():
        return index

    # Ensure all sessions are chunked
    chunk_all_sessions(storage)

    # If any chunks.jsonl is newer than the index, rebuild from scratch
    if _index_is_stale(index, sessions_dir):
        index.clear()

    already_indexed = index.indexed_sessions()

    for session_dir in sorted(sessions_dir.iterdir()):
        if not session_dir.is_dir():
            continue

        session_id = session_dir.name
        if session_id in already_indexed:
            continue

        chunks = storage.read_chunks(session_id)
        if not chunks:
            continue

        # Embed in batches
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            vectors = ollama_client.embed(texts)

            for chunk, vector in zip(batch, vectors):
                metadata: dict[str, Any] = {
                    "chunk_id": chunk.chunk_id,
                    "session_id": chunk.session_id,
                    "text": chunk.text[:500],  # Truncate for storage
                    "ts_start": chunk.ts_start,
                    "ts_end": chunk.ts_end,
                }
                # Propagate artifact_type from chunk._extra to index metadata
                art_type = chunk._extra.get("artifact_type")
                if art_type is not None:
                    metadata["artifact_type"] = art_type
                index.add(vector, metadata)

    # Index artifact chunks from artifacts/chunks.jsonl
    artifact_chunks = storage.read_artifact_chunks()
    # Group by session_id and skip already-indexed groups
    sorted_art = sorted(artifact_chunks, key=attrgetter("session_id"))
    for art_session_id, group in groupby(sorted_art, key=attrgetter("session_id")):
        if art_session_id in already_indexed:
            continue
        art_batch = list(group)
        batch_size = 10
        for i in range(0, len(art_batch), batch_size):
            batch = art_batch[i : i + batch_size]
            texts = [c.text for c in batch]
            vectors = ollama_client.embed(texts)

            for chunk, vector in zip(batch, vectors):
                metadata = {
                    "chunk_id": chunk.chunk_id,
                    "session_id": chunk.session_id,
                    "text": chunk.text[:500],
                    "ts_start": chunk.ts_start,
                    "ts_end": chunk.ts_end,
                }
                art_type = chunk._extra.get("artifact_type")
                if art_type is not None:
                    metadata["artifact_type"] = art_type
                index.add(vector, metadata)

    return index


def semantic_search(
    query: str,
    top_k: int,
    storage: NdjsonStorage,
    ollama_client: OllamaClient | None = None,
    artifact_type: str | None = None,
    rerank: bool = False,
    no_decay: bool = False,
) -> list[SearchResult]:
    """Orchestrate semantic search: ensure index, embed query, search.

    Args:
        query: Natural language search query.
        top_k: Number of results to return.
        storage: NdjsonStorage instance.
        ollama_client: OllamaClient instance (created from config if None).
        artifact_type: If set, filter results by artifact type ("session", "plan", "todo", "task").
        rerank: If True, use LLM reranker for better relevance.
        no_decay: If True, disable temporal decay boost.

    Returns:
        List of SearchResult objects.
    """
    config = storage.read_config()

    if ollama_client is None:
        from mb.ollama_client import client_from_config

        ollama_client = client_from_config(config)

    # Extract decay settings (read fresh each invocation)
    from mb.decay import get_decay_config

    half_life_days, enabled = get_decay_config(config)

    # Build/update index
    index = build_index(storage, ollama_client)

    # Embed query
    query_vectors = ollama_client.embed(query)
    query_vector = query_vectors[0]

    # Fetch more candidates when reranking
    fetch_k = 3 * top_k if rerank else top_k
    results = index.search(
        query_vector, top_k=fetch_k, artifact_type=artifact_type,
        half_life_days=half_life_days if enabled else 0.0,
        no_decay=no_decay,
    )

    if rerank and results:
        from mb.reranker import rerank as rerank_fn

        results = rerank_fn(query, results, ollama_client, top_k=top_k)

    # Filter out low-relevance results (cosine similarity noise floor)
    min_score = 0.35
    results = [r for r in results if r.score >= min_score]

    return results


def global_search(
    query: str,
    top_k: int,
    ollama_client: OllamaClient,
    artifact_type: str | None = None,
    no_decay: bool = False,
    rerank: bool = False,
) -> list[GlobalSearchResult]:
    """Search across all registered projects, merging results by score.

    Embeds the query once and reuses the vector across all projects.
    Skips unavailable projects with a stderr warning.
    """
    import click

    from mb.registry import list_projects

    projects = list_projects()
    if not projects:
        return []

    # Embed query once
    query_vectors = ollama_client.embed(query)
    query_vector = query_vectors[0]

    all_results: list[GlobalSearchResult] = []

    for project_path, _entry in projects.items():
        mb_dir = Path(project_path) / ".memory-bank"
        if not mb_dir.is_dir():
            click.echo(
                f"Warning: Skipping {project_path} (directory not found)",
                err=True,
            )
            continue

        try:
            storage = NdjsonStorage.open(mb_dir)
        except FileNotFoundError:
            click.echo(
                f"Warning: Skipping {project_path} (not initialized)",
                err=True,
            )
            continue

        try:
            config = storage.read_config()
        except Exception:
            click.echo(
                f"Warning: Skipping {project_path} (corrupt config)",
                err=True,
            )
            continue

        from mb.decay import get_decay_config

        half_life_days, enabled = get_decay_config(config)

        index = build_index(storage, ollama_client)
        fetch_k = 3 * top_k if rerank else top_k
        results = index.search(
            query_vector,
            top_k=fetch_k,
            artifact_type=artifact_type,
            half_life_days=half_life_days if enabled and not no_decay else 0.0,
            no_decay=no_decay,
        )

        if rerank and results:
            from mb.reranker import rerank as rerank_fn

            results = rerank_fn(query, results, ollama_client, top_k=fetch_k)

        for r in results:
            all_results.append(
                GlobalSearchResult.from_search_result(r, project_path)
            )

    # Merge by score, return top-K
    all_results.sort(key=lambda r: r.score, reverse=True)
    return all_results[:top_k]
