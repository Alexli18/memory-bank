"""Tests for artifact discovery and import — discover_todos, discover_plans, discover_tasks, import pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mb.store import NdjsonStorage


def _make_storage(root: Path) -> NdjsonStorage:
    """Create a minimal NdjsonStorage for tests."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "sessions").mkdir(exist_ok=True)
    (root / "config.json").write_text('{"version": "1.0"}', encoding="utf-8")
    return NdjsonStorage(root)


def _claude_project_dir(tmp_path: Path, cwd: str) -> Path:
    """Create and return the Claude projects dir for a given cwd."""
    from mb.claude_adapter import encode_project_dir

    project_dir = tmp_path / ".claude" / "projects" / encode_project_dir(cwd)
    project_dir.mkdir(parents=True)
    return project_dir


def _make_claude_jsonl(path: Path, turns: list[dict] | None = None) -> None:
    """Write a minimal Claude Code JSONL session file."""
    if turns is None:
        turns = [
            {
                "type": "user",
                "message": {"content": "Hello"},
                "timestamp": "2026-01-15T10:00:00Z",
            },
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Hi there."}]},
                "timestamp": "2026-01-15T10:00:05Z",
            },
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for turn in turns:
            f.write(json.dumps(turn) + "\n")


# --- discover_todos ---


def test_discover_todos_finds_project_todos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = str(tmp_path / "my-project")

    # Create project session
    project_dir = _claude_project_dir(tmp_path, cwd)
    _make_claude_jsonl(project_dir / "session-aaa.jsonl")

    # Create todo file matching the session
    todos_dir = tmp_path / ".claude" / "todos"
    todos_dir.mkdir(parents=True)
    todo_data = [{"id": "1", "content": "Fix bug", "status": "pending", "priority": "high"}]
    (todos_dir / "session-aaa.json").write_text(json.dumps(todo_data), encoding="utf-8")

    from mb.claude_adapter import discover_todos

    result = discover_todos(cwd)
    assert len(result) == 1
    assert result[0].name == "session-aaa.json"


def test_discover_todos_skips_other_projects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = str(tmp_path / "my-project")

    # Create project session
    project_dir = _claude_project_dir(tmp_path, cwd)
    _make_claude_jsonl(project_dir / "session-aaa.jsonl")

    # Create todo file for a DIFFERENT session
    todos_dir = tmp_path / ".claude" / "todos"
    todos_dir.mkdir(parents=True)
    (todos_dir / "session-other.json").write_text('[{"id":"1","content":"x"}]', encoding="utf-8")

    from mb.claude_adapter import discover_todos

    result = discover_todos(cwd)
    assert result == []


def test_discover_todos_no_todos_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = str(tmp_path / "my-project")
    _claude_project_dir(tmp_path, cwd)

    from mb.claude_adapter import discover_todos

    result = discover_todos(cwd)
    assert result == []


def test_discover_todos_skips_empty_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = str(tmp_path / "my-project")

    project_dir = _claude_project_dir(tmp_path, cwd)
    _make_claude_jsonl(project_dir / "session-bbb.jsonl")

    todos_dir = tmp_path / ".claude" / "todos"
    todos_dir.mkdir(parents=True)
    (todos_dir / "session-bbb.json").write_text("", encoding="utf-8")

    from mb.claude_adapter import discover_todos

    result = discover_todos(cwd)
    assert result == []


# --- discover_task_dirs ---


def test_discover_task_dirs_finds_project_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = str(tmp_path / "my-project")

    project_dir = _claude_project_dir(tmp_path, cwd)
    _make_claude_jsonl(project_dir / "session-aaa.jsonl")

    tasks_dir = tmp_path / ".claude" / "tasks" / "session-aaa"
    tasks_dir.mkdir(parents=True)
    (tasks_dir / "1.json").write_text('{"id":"1","subject":"Task 1"}', encoding="utf-8")

    from mb.claude_adapter import discover_task_dirs

    result = discover_task_dirs(cwd)
    assert len(result) == 1
    assert result[0].name == "session-aaa"


def test_discover_task_dirs_filters_by_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = str(tmp_path / "my-project")

    project_dir = _claude_project_dir(tmp_path, cwd)
    _make_claude_jsonl(project_dir / "session-aaa.jsonl")

    # Task dir for different session
    tasks_dir = tmp_path / ".claude" / "tasks" / "session-other"
    tasks_dir.mkdir(parents=True)
    (tasks_dir / "1.json").write_text('{"id":"1"}', encoding="utf-8")

    from mb.claude_adapter import discover_task_dirs

    result = discover_task_dirs(cwd)
    assert result == []


# --- discover_plan_slugs ---


def test_discover_plan_slugs_from_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = str(tmp_path / "my-project")

    project_dir = _claude_project_dir(tmp_path, cwd)
    lines = [
        {"type": "user", "message": {"content": "Create a plan"}, "timestamp": "2026-01-15T10:00:00Z"},
        {"type": "assistant", "slug": "happy-dancing-fox", "message": {"content": [{"type": "text", "text": "Plan created"}]}},
        {"type": "user", "slug": "happy-dancing-fox", "message": {"content": "Update plan"}},
    ]
    jsonl_path = project_dir / "session-plan.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    from mb.claude_adapter import discover_plan_slugs

    slugs = discover_plan_slugs(cwd)
    assert slugs == {"happy-dancing-fox"}


def test_discover_plan_slugs_no_slugs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = str(tmp_path / "my-project")

    project_dir = _claude_project_dir(tmp_path, cwd)
    _make_claude_jsonl(project_dir / "session-no-plans.jsonl")

    from mb.claude_adapter import discover_plan_slugs

    slugs = discover_plan_slugs(cwd)
    assert slugs == set()


# --- discover_plans ---


def test_discover_plans_finds_matching_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    plans_dir = tmp_path / ".claude" / "plans"
    plans_dir.mkdir(parents=True)
    (plans_dir / "happy-dancing-fox.md").write_text("# Plan\nContent", encoding="utf-8")
    (plans_dir / "other-plan.md").write_text("# Other", encoding="utf-8")

    from mb.claude_adapter import discover_plans

    result = discover_plans({"happy-dancing-fox"})
    assert len(result) == 1
    assert result[0].name == "happy-dancing-fox.md"


def test_discover_plans_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    plans_dir = tmp_path / ".claude" / "plans"
    plans_dir.mkdir(parents=True)

    from mb.claude_adapter import discover_plans

    result = discover_plans({"nonexistent-slug"})
    assert result == []


# --- Artifact import integration ---


def test_import_with_todos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full import pipeline discovers and imports todo artifacts."""
    storage_root = tmp_path / ".memory-bank"
    storage = _make_storage(storage_root)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = str(storage_root.parent)

    # Create session JSONL
    project_dir = _claude_project_dir(tmp_path, cwd)
    _make_claude_jsonl(project_dir / "session-todo.jsonl")

    # Create todo file
    todos_dir = tmp_path / ".claude" / "todos"
    todos_dir.mkdir(parents=True)
    todo_data = [
        {"id": "1", "content": "Fix bug", "status": "pending", "priority": "high"},
        {"id": "2", "content": "Write docs", "status": "completed", "priority": "low"},
    ]
    (todos_dir / "session-todo.json").write_text(json.dumps(todo_data), encoding="utf-8")

    from mb.importer import import_claude_sessions_with_artifacts

    result = import_claude_sessions_with_artifacts(storage)

    assert result["imported"] == 1
    assert result["todos_imported"] == 1

    # Verify stored todo
    stored_todo = storage.artifacts_dir / "todos" / "session-todo.json"
    assert stored_todo.exists()

    # Verify chunks written
    chunks_path = storage.artifacts_dir / "chunks.jsonl"
    assert chunks_path.exists()
    chunk_lines = [line for line in chunks_path.read_text().splitlines() if line.strip()]
    assert len(chunk_lines) >= 1
    chunk = json.loads(chunk_lines[0])
    assert chunk["artifact_type"] == "todo"


def test_import_with_plans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Import discovers plan slugs from JSONL and imports plan files."""
    storage_root = tmp_path / ".memory-bank"
    storage = _make_storage(storage_root)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = str(storage_root.parent)

    # Create session JSONL with slug reference
    project_dir = _claude_project_dir(tmp_path, cwd)
    lines = [
        {"type": "user", "message": {"content": "Hello"}, "timestamp": "2026-01-15T10:00:00Z"},
        {"type": "assistant", "slug": "test-plan-slug", "message": {"content": [{"type": "text", "text": "OK"}]}},
    ]
    jsonl_path = project_dir / "session-plan.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    # Create plan file
    plans_dir = tmp_path / ".claude" / "plans"
    plans_dir.mkdir(parents=True)
    (plans_dir / "test-plan-slug.md").write_text("## Steps\n1. Step one\n\n## Notes\nSome notes", encoding="utf-8")

    from mb.importer import import_claude_sessions_with_artifacts

    result = import_claude_sessions_with_artifacts(storage)

    assert result["imported"] == 1
    assert result["plans_imported"] == 1

    # Verify stored plan
    assert (storage.artifacts_dir / "plans" / "test-plan-slug.md").exists()
    assert (storage.artifacts_dir / "plans" / "test-plan-slug.meta.json").exists()


def test_import_with_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Import discovers and imports task directories."""
    storage_root = tmp_path / ".memory-bank"
    storage = _make_storage(storage_root)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = str(storage_root.parent)

    # Create session JSONL
    project_dir = _claude_project_dir(tmp_path, cwd)
    _make_claude_jsonl(project_dir / "session-tasks.jsonl")

    # Create task directory
    tasks_dir = tmp_path / ".claude" / "tasks" / "session-tasks"
    tasks_dir.mkdir(parents=True)
    (tasks_dir / "1.json").write_text(
        json.dumps({"id": "1", "subject": "Create config", "status": "pending", "blocks": ["2"]}),
        encoding="utf-8",
    )
    (tasks_dir / "2.json").write_text(
        json.dumps({"id": "2", "subject": "Build feature", "status": "in_progress", "blockedBy": ["1"]}),
        encoding="utf-8",
    )
    # These should be skipped
    (tasks_dir / ".lock").write_text("", encoding="utf-8")
    (tasks_dir / ".highwatermark").write_text("2", encoding="utf-8")

    from mb.importer import import_claude_sessions_with_artifacts

    result = import_claude_sessions_with_artifacts(storage)

    assert result["imported"] == 1
    assert result["tasks_imported"] == 1

    # Verify stored tasks
    assert (storage.artifacts_dir / "tasks" / "session-tasks" / "1.json").exists()
    assert (storage.artifacts_dir / "tasks" / "session-tasks" / "2.json").exists()


def test_import_artifacts_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Second import skips already imported artifacts."""
    storage_root = tmp_path / ".memory-bank"
    storage = _make_storage(storage_root)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = str(storage_root.parent)

    project_dir = _claude_project_dir(tmp_path, cwd)
    _make_claude_jsonl(project_dir / "session-idem.jsonl")

    todos_dir = tmp_path / ".claude" / "todos"
    todos_dir.mkdir(parents=True)
    (todos_dir / "session-idem.json").write_text(
        json.dumps([{"id": "1", "content": "Task", "status": "pending"}]),
        encoding="utf-8",
    )

    from mb.importer import import_claude_sessions_with_artifacts

    result1 = import_claude_sessions_with_artifacts(storage)
    assert result1["todos_imported"] == 1

    result2 = import_claude_sessions_with_artifacts(storage)
    assert result2["todos_imported"] == 0  # Skipped on second run


def test_import_malformed_todo_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed todo JSON is skipped with warning, not crash."""
    storage_root = tmp_path / ".memory-bank"
    storage = _make_storage(storage_root)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = str(storage_root.parent)

    project_dir = _claude_project_dir(tmp_path, cwd)
    _make_claude_jsonl(project_dir / "session-bad.jsonl")

    todos_dir = tmp_path / ".claude" / "todos"
    todos_dir.mkdir(parents=True)
    (todos_dir / "session-bad.json").write_text("NOT VALID JSON{{{", encoding="utf-8")

    from mb.importer import import_claude_sessions_with_artifacts

    result = import_claude_sessions_with_artifacts(storage)
    assert result["todos_imported"] == 0  # Skipped, no crash


def test_import_dry_run_counts_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dry run counts artifacts without storing them."""
    storage_root = tmp_path / ".memory-bank"
    storage = _make_storage(storage_root)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = str(storage_root.parent)

    project_dir = _claude_project_dir(tmp_path, cwd)
    _make_claude_jsonl(project_dir / "session-dry.jsonl")

    todos_dir = tmp_path / ".claude" / "todos"
    todos_dir.mkdir(parents=True)
    (todos_dir / "session-dry.json").write_text(
        json.dumps([{"id": "1", "content": "T1"}, {"id": "2", "content": "T2"}]),
        encoding="utf-8",
    )

    tasks_dir = tmp_path / ".claude" / "tasks" / "session-dry"
    tasks_dir.mkdir(parents=True)
    (tasks_dir / "1.json").write_text(json.dumps({"id": "1", "subject": "Task"}), encoding="utf-8")

    from mb.importer import import_claude_sessions_with_artifacts

    result = import_claude_sessions_with_artifacts(storage, dry_run=True)
    assert result["todos_imported"] == 1
    assert result["dry_run_todo_items"] == 2
    assert result["tasks_imported"] == 1
    assert result["dry_run_task_items"] == 1

    # Nothing should be stored
    assert not (storage.artifacts_dir / "todos").exists()
    assert not (storage.artifacts_dir / "tasks").exists()
