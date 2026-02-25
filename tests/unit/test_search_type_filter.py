"""Tests for search type filtering — artifact_type flows through index to SearchResult."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mb.search import VECTOR_DIM, VectorIndex


def _make_index_with_types(tmp_path: Path) -> VectorIndex:
    """Create an index with vectors of different artifact types."""
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    idx = VectorIndex(index_dir)

    rng = np.random.default_rng(42)

    # Add session chunks (no artifact_type)
    for i in range(3):
        vec = rng.standard_normal(VECTOR_DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)
        meta = {"chunk_id": f"session-{i}", "session_id": "s1", "text": f"session text {i}"}
        idx.add(vec.tolist(), meta)

    # Add plan chunks
    for i in range(2):
        vec = rng.standard_normal(VECTOR_DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)
        meta = {
            "chunk_id": f"plan-{i}", "session_id": "artifact-plan-test", "text": f"plan text {i}",
            "artifact_type": "plan",
        }
        idx.add(vec.tolist(), meta)

    # Add todo chunk
    vec = rng.standard_normal(VECTOR_DIM).astype(np.float32)
    vec /= np.linalg.norm(vec)
    meta = {
        "chunk_id": "todo-0", "session_id": "sess-todo", "text": "todo text",
        "artifact_type": "todo",
    }
    idx.add(vec.tolist(), meta)

    # Add task chunk
    vec = rng.standard_normal(VECTOR_DIM).astype(np.float32)
    vec /= np.linalg.norm(vec)
    meta = {
        "chunk_id": "task-0", "session_id": "sess-task", "text": "task text",
        "artifact_type": "task",
    }
    idx.add(vec.tolist(), meta)

    return idx


def test_search_no_filter_returns_all_types(tmp_path: Path) -> None:
    """Without artifact_type filter, results include all types."""
    idx = _make_index_with_types(tmp_path)

    rng = np.random.default_rng(99)
    query = rng.standard_normal(VECTOR_DIM).tolist()

    results = idx.search(query, top_k=7)
    assert len(results) == 7

    types = {r.artifact_type for r in results}
    assert None in types  # session chunks
    assert "plan" in types
    assert "todo" in types
    assert "task" in types


def test_search_filter_plan(tmp_path: Path) -> None:
    """artifact_type='plan' returns only plan results."""
    idx = _make_index_with_types(tmp_path)

    rng = np.random.default_rng(99)
    query = rng.standard_normal(VECTOR_DIM).tolist()

    results = idx.search(query, top_k=10, artifact_type="plan")
    assert len(results) == 2
    assert all(r.artifact_type == "plan" for r in results)


def test_search_filter_session(tmp_path: Path) -> None:
    """artifact_type='session' returns only chunks with no artifact_type."""
    idx = _make_index_with_types(tmp_path)

    rng = np.random.default_rng(99)
    query = rng.standard_normal(VECTOR_DIM).tolist()

    results = idx.search(query, top_k=10, artifact_type="session")
    assert len(results) == 3
    assert all(r.artifact_type is None for r in results)


def test_search_filter_todo(tmp_path: Path) -> None:
    """artifact_type='todo' returns only todo results."""
    idx = _make_index_with_types(tmp_path)

    rng = np.random.default_rng(99)
    query = rng.standard_normal(VECTOR_DIM).tolist()

    results = idx.search(query, top_k=10, artifact_type="todo")
    assert len(results) == 1
    assert results[0].artifact_type == "todo"


def test_search_filter_task(tmp_path: Path) -> None:
    """artifact_type='task' returns only task results."""
    idx = _make_index_with_types(tmp_path)

    rng = np.random.default_rng(99)
    query = rng.standard_normal(VECTOR_DIM).tolist()

    results = idx.search(query, top_k=10, artifact_type="task")
    assert len(results) == 1
    assert results[0].artifact_type == "task"


def test_artifact_type_flows_through_index(tmp_path: Path) -> None:
    """artifact_type is preserved from index metadata to SearchResult."""
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True)
    idx = VectorIndex(index_dir)

    rng = np.random.default_rng(42)
    vec = rng.standard_normal(VECTOR_DIM).astype(np.float32)
    vec /= np.linalg.norm(vec)

    meta = {
        "chunk_id": "c1", "session_id": "s1", "text": "plan content",
        "artifact_type": "plan", "ts_start": 100.0, "ts_end": 200.0,
    }
    idx.add(vec.tolist(), meta)

    results = idx.search(vec.tolist(), top_k=1)
    assert len(results) == 1
    assert results[0].artifact_type == "plan"
    assert results[0].chunk_id == "c1"


def test_backward_compat_session_chunks(tmp_path: Path) -> None:
    """Existing session chunks without artifact_type work correctly."""
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True)
    idx = VectorIndex(index_dir)

    rng = np.random.default_rng(42)
    vec = rng.standard_normal(VECTOR_DIM).astype(np.float32)
    vec /= np.linalg.norm(vec)

    # No artifact_type in metadata (old-style session chunk)
    meta = {"chunk_id": "c1", "session_id": "s1", "text": "session text"}
    idx.add(vec.tolist(), meta)

    results = idx.search(vec.tolist(), top_k=1)
    assert len(results) == 1
    assert results[0].artifact_type is None


def test_search_filter_no_matches(tmp_path: Path) -> None:
    """Filtering by type with no matching results returns empty list."""
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True)
    idx = VectorIndex(index_dir)

    rng = np.random.default_rng(42)
    vec = rng.standard_normal(VECTOR_DIM).astype(np.float32)
    vec /= np.linalg.norm(vec)

    # Only session chunks
    meta = {"chunk_id": "c1", "session_id": "s1", "text": "text"}
    idx.add(vec.tolist(), meta)

    results = idx.search(vec.tolist(), top_k=5, artifact_type="plan")
    assert results == []
