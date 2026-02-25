"""Tests for mb.registry — global project registry CRUD."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect registry to a temp directory for every test."""
    reg_dir = tmp_path / ".memory-bank"
    reg_dir.mkdir()
    monkeypatch.setattr("mb.registry.REGISTRY_DIR", reg_dir)
    monkeypatch.setattr("mb.registry.REGISTRY_PATH", reg_dir / "projects.json")


def test_register_new_project(tmp_path: Path) -> None:
    """register_project creates a new entry with registered_at."""
    from mb.registry import register_project, REGISTRY_PATH

    entry = register_project(str(tmp_path / "proj-a"))
    assert entry.path == str((tmp_path / "proj-a").resolve())
    assert entry.registered_at > 0
    assert entry.session_count == 0
    assert entry.last_import == 0.0

    # Verify persisted
    data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    assert entry.path in data["projects"]


def test_register_idempotent_preserves_stats(tmp_path: Path) -> None:
    """Re-registering a project preserves existing stats."""
    from mb.registry import register_project, update_project_stats

    proj = str(tmp_path / "proj-b")
    entry1 = register_project(proj)
    update_project_stats(proj, session_count=10)
    entry2 = register_project(proj)

    assert entry2.registered_at == entry1.registered_at
    assert entry2.session_count == 10


def test_list_empty_registry() -> None:
    """list_projects returns empty dict when no projects registered."""
    from mb.registry import list_projects

    result = list_projects()
    assert result == {}


def test_list_multiple_projects(tmp_path: Path) -> None:
    """list_projects returns all registered projects."""
    from mb.registry import register_project, list_projects

    register_project(str(tmp_path / "proj-a"))
    register_project(str(tmp_path / "proj-b"))

    result = list_projects()
    assert len(result) == 2


def test_remove_existing_project(tmp_path: Path) -> None:
    """remove_project returns True and removes the entry."""
    from mb.registry import register_project, remove_project, list_projects

    proj = str(tmp_path / "proj-a")
    register_project(proj)
    assert remove_project(proj) is True
    assert len(list_projects()) == 0


def test_remove_nonexistent_project(tmp_path: Path) -> None:
    """remove_project returns False for unknown project."""
    from mb.registry import remove_project

    assert remove_project("/nonexistent/path") is False


def test_update_project_stats(tmp_path: Path) -> None:
    """update_project_stats updates last_import and session_count."""
    from mb.registry import register_project, update_project_stats, list_projects

    proj = str(tmp_path / "proj-a")
    register_project(proj)
    update_project_stats(proj, session_count=42)

    projects = list_projects()
    resolved = str((tmp_path / "proj-a").resolve())
    entry = projects[resolved]
    assert entry.session_count == 42
    assert entry.last_import > 0


def test_update_stats_auto_registers(tmp_path: Path) -> None:
    """update_project_stats registers the project if not already registered."""
    from mb.registry import update_project_stats, list_projects

    proj = str(tmp_path / "proj-new")
    update_project_stats(proj, session_count=5)

    projects = list_projects()
    assert len(projects) == 1


def test_atomic_write_cleanup(tmp_path: Path) -> None:
    """Atomic write cleans up temp file on os.replace failure."""
    from mb.registry import register_project

    with patch("os.replace", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            register_project(str(tmp_path / "proj-fail"))


def test_auto_create_registry_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Registry directory and file are auto-created on first register."""
    new_dir = tmp_path / "fresh" / ".memory-bank"
    monkeypatch.setattr("mb.registry.REGISTRY_DIR", new_dir)
    monkeypatch.setattr("mb.registry.REGISTRY_PATH", new_dir / "projects.json")

    from mb.registry import register_project

    register_project(str(tmp_path / "proj-x"))
    assert (new_dir / "projects.json").exists()
