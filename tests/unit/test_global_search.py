"""Tests for mb.search.global_search — cross-project search."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mb.models import SearchResult


def _make_search_result(
    session_id: str = "s1",
    score: float = 0.9,
    text: str = "test text",
    index: int = 0,
) -> SearchResult:
    return SearchResult(
        chunk_id=f"chunk-{session_id}-{index}",
        session_id=session_id,
        index=index,
        text=text,
        ts_start=0.0,
        ts_end=60.0,
        token_estimate=50,
        quality_score=0.8,
        score=score,
    )


def _setup_project(path: Path) -> None:
    """Create a minimal .memory-bank/ with config.json."""
    mb_dir = path / ".memory-bank"
    mb_dir.mkdir(parents=True, exist_ok=True)
    (mb_dir / "config.json").write_text(
        json.dumps({"version": "1.0", "ollama": {}, "decay": {"enabled": False}}),
        encoding="utf-8",
    )
    (mb_dir / "sessions").mkdir(exist_ok=True)
    (mb_dir / "index").mkdir(exist_ok=True)


@pytest.fixture()
def two_projects(tmp_path: Path) -> tuple[Path, Path]:
    """Create two project directories with .memory-bank/."""
    proj_a = tmp_path / "proj-a"
    proj_b = tmp_path / "proj-b"
    proj_a.mkdir()
    proj_b.mkdir()
    _setup_project(proj_a)
    _setup_project(proj_b)
    return proj_a, proj_b


def test_merge_results_from_two_projects_sorted_by_score(
    two_projects: tuple[Path, Path],
) -> None:
    """Results from 2 projects are merged and sorted by score descending."""
    proj_a, proj_b = two_projects

    results_a = [_make_search_result("s1", score=0.85)]
    results_b = [_make_search_result("s2", score=0.92)]

    mock_client = MagicMock()
    mock_client.embed.return_value = [[0.1] * 768]

    mock_index = MagicMock()
    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return results_a if call_count == 1 else results_b

    mock_index.search.side_effect = side_effect

    from mb.models import ProjectEntry

    projects = {
        str(proj_a): ProjectEntry(path=str(proj_a), registered_at=1.0),
        str(proj_b): ProjectEntry(path=str(proj_b), registered_at=2.0),
    }

    with (
        patch("mb.registry.list_projects", return_value=projects),
        patch("mb.search.build_index", return_value=mock_index),
    ):
        from mb.search import global_search

        results = global_search("test query", top_k=10, ollama_client=mock_client)

    assert len(results) == 2
    assert results[0].score == 0.92
    assert results[0].project_path == str(proj_b)
    assert results[1].score == 0.85
    assert results[1].project_path == str(proj_a)


def test_top_k_limit_across_projects(
    two_projects: tuple[Path, Path],
) -> None:
    """top_k limits total results across all projects."""
    proj_a, proj_b = two_projects

    results_a = [
        _make_search_result("s1", score=0.95, index=0),
        _make_search_result("s1", score=0.90, index=1),
        _make_search_result("s1", score=0.85, index=2),
    ]
    results_b = [
        _make_search_result("s2", score=0.93, index=0),
        _make_search_result("s2", score=0.88, index=1),
    ]

    mock_client = MagicMock()
    mock_client.embed.return_value = [[0.1] * 768]

    mock_index = MagicMock()
    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return results_a if call_count == 1 else results_b

    mock_index.search.side_effect = side_effect

    from mb.models import ProjectEntry

    projects = {
        str(proj_a): ProjectEntry(path=str(proj_a), registered_at=1.0),
        str(proj_b): ProjectEntry(path=str(proj_b), registered_at=2.0),
    }

    with (
        patch("mb.registry.list_projects", return_value=projects),
        patch("mb.search.build_index", return_value=mock_index),
    ):
        from mb.search import global_search

        results = global_search("test query", top_k=3, ollama_client=mock_client)

    assert len(results) == 3
    # Top 3 by score: 0.95, 0.93, 0.90
    assert results[0].score == 0.95
    assert results[1].score == 0.93
    assert results[2].score == 0.90


def test_skip_missing_project_directory(tmp_path: Path) -> None:
    """Projects with missing .memory-bank/ are skipped with a warning."""
    mock_client = MagicMock()
    mock_client.embed.return_value = [[0.1] * 768]

    from mb.models import ProjectEntry

    missing = str(tmp_path / "missing-proj")
    projects = {
        missing: ProjectEntry(path=missing, registered_at=1.0),
    }

    with patch("mb.registry.list_projects", return_value=projects):
        from mb.search import global_search

        results = global_search("test query", top_k=5, ollama_client=mock_client)

    assert results == []


def test_empty_registry_returns_empty() -> None:
    """Empty registry returns empty results immediately."""
    mock_client = MagicMock()

    with patch("mb.registry.list_projects", return_value={}):
        from mb.search import global_search

        results = global_search("test query", top_k=5, ollama_client=mock_client)

    assert results == []
    mock_client.embed.assert_not_called()


def test_single_project_returns_same_as_local(tmp_path: Path) -> None:
    """Single project global search returns same results as local search."""
    proj = tmp_path / "solo"
    proj.mkdir()
    _setup_project(proj)

    expected = [_make_search_result("s1", score=0.88)]

    mock_client = MagicMock()
    mock_client.embed.return_value = [[0.1] * 768]

    mock_index = MagicMock()
    mock_index.search.return_value = expected

    from mb.models import ProjectEntry

    projects = {
        str(proj): ProjectEntry(path=str(proj), registered_at=1.0),
    }

    with (
        patch("mb.registry.list_projects", return_value=projects),
        patch("mb.search.build_index", return_value=mock_index),
    ):
        from mb.search import global_search

        results = global_search("test query", top_k=5, ollama_client=mock_client)

    assert len(results) == 1
    assert results[0].score == 0.88
    assert results[0].project_path == str(proj)
