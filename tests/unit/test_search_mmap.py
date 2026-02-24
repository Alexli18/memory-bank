"""Tests for VectorIndex mmap-based search and lazy metadata loading."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mb.models import SearchResult
from mb.search import VECTOR_DIM, VectorIndex


def _make_index(tmp_path: Path, n: int = 10) -> VectorIndex:
    """Create an index with *n* random normalised vectors and matching metadata."""
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    idx = VectorIndex(index_dir)

    rng = np.random.default_rng(42)
    for i in range(n):
        vec = rng.standard_normal(VECTOR_DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)
        metadata = {"chunk_id": f"c-{i}", "session_id": "s1", "text": f"text-{i}"}
        idx.add(vec.tolist(), metadata)

    return idx


def test_search_returns_results_with_mmap(tmp_path: Path) -> None:
    """Basic search returns results using mmap code path."""
    idx = _make_index(tmp_path, n=20)

    rng = np.random.default_rng(99)
    query = rng.standard_normal(VECTOR_DIM).tolist()

    results = idx.search(query, top_k=5)

    assert len(results) == 5
    assert all(isinstance(r, SearchResult) for r in results)
    assert all(r.score is not None for r in results)
    assert all(r.chunk_id for r in results)
    # Scores should be descending
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_empty_index(tmp_path: Path) -> None:
    """Empty vectors.bin returns empty list."""
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True)
    idx = VectorIndex(index_dir)

    # Create empty files
    (index_dir / "vectors.bin").write_bytes(b"")
    (index_dir / "metadata.jsonl").write_text("")

    query = [0.0] * VECTOR_DIM
    assert idx.search(query) == []


def test_search_integrity_mismatch(tmp_path: Path) -> None:
    """More vectors than metadata lines — truncates to min without error."""
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True)
    idx = VectorIndex(index_dir)

    rng = np.random.default_rng(42)
    # Write 5 vectors
    vecs = rng.standard_normal((5, VECTOR_DIM)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms
    (index_dir / "vectors.bin").write_bytes(vecs.tobytes())

    # Write only 3 metadata lines
    with (index_dir / "metadata.jsonl").open("w") as f:
        for i in range(3):
            f.write(json.dumps({"chunk_id": f"c-{i}", "text": f"t-{i}"}) + "\n")

    query = rng.standard_normal(VECTOR_DIM).tolist()
    results = idx.search(query, top_k=10)

    # Should get at most 3 results (min of vectors=5, metadata=3)
    assert len(results) <= 3
    assert all(isinstance(r, SearchResult) for r in results)


def test_count_metadata_lines(tmp_path: Path) -> None:
    """_count_metadata_lines counts non-empty lines without parsing."""
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True)
    idx = VectorIndex(index_dir)

    lines = [
        json.dumps({"id": i}) + "\n" for i in range(7)
    ]
    # Add some blank lines
    lines.insert(3, "\n")
    lines.append("\n")
    (index_dir / "metadata.jsonl").write_text("".join(lines))

    assert idx._count_metadata_lines() == 7


def test_load_metadata_at_indices(tmp_path: Path) -> None:
    """_load_metadata_at_indices returns only requested indices."""
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True)
    idx = VectorIndex(index_dir)

    with (index_dir / "metadata.jsonl").open("w") as f:
        for i in range(10):
            f.write(json.dumps({"chunk_id": f"c-{i}"}) + "\n")

    result = idx._load_metadata_at_indices({2, 7})

    assert set(result.keys()) == {2, 7}
    assert result[2]["chunk_id"] == "c-2"
    assert result[7]["chunk_id"] == "c-7"


def test_load_metadata_at_indices_early_exit(tmp_path: Path) -> None:
    """_load_metadata_at_indices stops reading after all indices found."""
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True)
    idx = VectorIndex(index_dir)

    # Write 1000 lines — if early exit works, we don't parse all of them
    with (index_dir / "metadata.jsonl").open("w") as f:
        for i in range(1000):
            f.write(json.dumps({"chunk_id": f"c-{i}"}) + "\n")

    result = idx._load_metadata_at_indices({0, 1})

    assert len(result) == 2
    assert result[0]["chunk_id"] == "c-0"
    assert result[1]["chunk_id"] == "c-1"
