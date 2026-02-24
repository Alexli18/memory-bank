"""Embedding index and cosine similarity search."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from mb.chunker import chunk_all_sessions
from mb.ollama_client import OllamaClient

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

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        """Search for top-K similar vectors by cosine similarity.

        Uses memory-mapped I/O for the vectors file so that only the pages
        touched by the matrix multiplication are read from disk.  Metadata
        is loaded only for the top-K indices.

        Args:
            query_vector: Query embedding (will be normalized).
            top_k: Number of results to return.

        Returns:
            List of dicts with 'score' and metadata fields.
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

        # Load metadata only for the needed indices
        metadata_map = self._load_metadata_at_indices(set(top_indices))

        results = []
        for idx in top_indices:
            meta = metadata_map.get(int(idx), {})
            meta["score"] = float(scores[idx])
            results.append(meta)

        return results

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


def build_index(storage_root: Path, ollama_client: OllamaClient) -> VectorIndex:
    """Build or incrementally update the embedding index.

    Iterates sessions, chunks them, embeds via Ollama, appends to index.
    Skips sessions already indexed.  Rebuilds if chunks are newer than index.
    """
    index_dir = storage_root / "index"
    index_dir.mkdir(exist_ok=True)
    index = VectorIndex(index_dir)

    sessions_dir = storage_root / "sessions"
    if not sessions_dir.exists():
        return index

    # Ensure all sessions are chunked
    chunk_all_sessions(storage_root)

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

        chunks_path = session_dir / "chunks.jsonl"
        if not chunks_path.exists():
            continue

        # Read existing chunks
        chunks: list[dict[str, Any]] = []
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))

        if not chunks:
            continue

        # Embed in batches
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c["text"] for c in batch]
            vectors = ollama_client.embed(texts)

            for chunk, vector in zip(batch, vectors):
                metadata = {
                    "chunk_id": chunk["chunk_id"],
                    "session_id": chunk["session_id"],
                    "text": chunk["text"][:500],  # Truncate for storage
                    "ts_start": chunk["ts_start"],
                    "ts_end": chunk["ts_end"],
                }
                index.add(vector, metadata)

    return index


def semantic_search(
    query: str,
    top_k: int,
    storage_root: Path,
    ollama_client: OllamaClient | None = None,
) -> list[dict[str, Any]]:
    """Orchestrate semantic search: ensure index, embed query, search.

    Args:
        query: Natural language search query.
        top_k: Number of results to return.
        storage_root: Path to .memory-bank/.
        ollama_client: OllamaClient instance (created from config if None).

    Returns:
        List of result dicts with score, chunk_id, session_id, text, timestamps.
    """
    if ollama_client is None:
        from mb.ollama_client import client_from_config
        from mb.storage import read_config

        config = read_config(storage_root)
        ollama_client = client_from_config(config)

    # Build/update index
    index = build_index(storage_root, ollama_client)

    # Embed query
    query_vectors = ollama_client.embed(query)
    query_vector = query_vectors[0]

    # Search
    return index.search(query_vector, top_k=top_k)
