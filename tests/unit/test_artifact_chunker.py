"""Tests for artifact_chunker — chunk_todo_list, chunk_plan, chunk_task."""

from __future__ import annotations

from mb.artifact_chunker import chunk_plan, chunk_task, chunk_todo_list
from mb.models import TaskItem, TodoItem, TodoList


# --- chunk_todo_list ---


def test_chunk_todo_list_basic() -> None:
    todo_list = TodoList(
        session_id="sess-1",
        agent_id=None,
        items=(
            TodoItem(id="1", content="Fix bug", status="pending", priority="high"),
            TodoItem(id="2", content="Write tests", status="in_progress", priority="medium"),
        ),
        file_path="/path/to/file.json",
        mtime=1700000000.0,
    )
    chunks = chunk_todo_list(todo_list)
    assert len(chunks) == 1

    chunk = chunks[0]
    assert "[TODO] pending (high): Fix bug" in chunk.text
    assert "[TODO] in_progress (medium): Write tests" in chunk.text
    assert chunk.session_id == "sess-1"
    assert chunk.ts_start == 1700000000.0
    assert chunk.ts_end == 1700000000.0
    assert chunk.token_estimate > 0
    assert chunk.quality_score > 0


def test_chunk_todo_list_extra_fields() -> None:
    todo_list = TodoList(
        session_id="sess-2",
        agent_id=None,
        items=(TodoItem(id="1", content="Task", status="pending", priority="low"),),
        file_path="/p",
        mtime=100.0,
    )
    chunks = chunk_todo_list(todo_list)
    chunk = chunks[0]
    assert chunk._extra["artifact_type"] == "todo"
    assert chunk._extra["source"] == "artifact"
    assert chunk._extra["artifact_id"] == "sess-2"


def test_chunk_todo_list_empty_items() -> None:
    todo_list = TodoList(
        session_id="s", agent_id=None, items=(), file_path="", mtime=0.0,
    )
    assert chunk_todo_list(todo_list) == []


# --- chunk_plan ---


def test_chunk_plan_splits_by_headings() -> None:
    content = "# Title\nIntro text\n\n## Section A\nContent A\n\n## Section B\nContent B"
    chunks = chunk_plan("my-plan", content, mtime=500.0)
    assert len(chunks) == 3  # Intro + Section A + Section B

    # First chunk is the intro (before any ##)
    assert "[PLAN: my-plan]" in chunks[0].text
    assert "Intro text" in chunks[0].text

    # Second chunk is Section A
    assert "## Section A" in chunks[1].text
    assert "Content A" in chunks[1].text

    # Third chunk is Section B
    assert "## Section B" in chunks[2].text
    assert "Content B" in chunks[2].text


def test_chunk_plan_extra_fields() -> None:
    content = "## Steps\n1. Do something"
    chunks = chunk_plan("slug-1", content, mtime=100.0)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk._extra["artifact_type"] == "plan"
    assert chunk._extra["source"] == "artifact"
    assert chunk._extra["artifact_id"] == "slug-1"
    assert chunk.session_id == "artifact-plan-slug-1"


def test_chunk_plan_empty_content() -> None:
    assert chunk_plan("s", "", 0.0) == []
    assert chunk_plan("s", "   \n  ", 0.0) == []


def test_chunk_plan_quality_and_tokens() -> None:
    content = "## Overview\nThis is a plan for database migration"
    chunks = chunk_plan("db-plan", content, mtime=200.0)
    assert len(chunks) == 1
    assert chunks[0].token_estimate > 0
    assert chunks[0].quality_score > 0


# --- chunk_task ---


def test_chunk_task_basic() -> None:
    task = TaskItem(
        id="3",
        session_id="sess-abc",
        subject="Create config",
        description="Set up project configuration",
        status="in_progress",
        blocks=("4", "5"),
        blocked_by=("1",),
    )
    chunk = chunk_task(task)
    assert "[TASK] Create config (in_progress)" in chunk.text
    assert "Set up project configuration" in chunk.text
    assert "Blocks: 4, 5" in chunk.text
    assert "Blocked by: 1" in chunk.text
    assert chunk.session_id == "sess-abc"


def test_chunk_task_extra_fields() -> None:
    task = TaskItem(id="1", session_id="sess-x", subject="Test", status="pending")
    chunk = chunk_task(task)
    assert chunk._extra["artifact_type"] == "task"
    assert chunk._extra["source"] == "artifact"
    assert chunk._extra["artifact_id"] == "sess-x"


def test_chunk_task_no_dependencies() -> None:
    task = TaskItem(id="1", session_id="s1", subject="Simple task", status="pending")
    chunk = chunk_task(task)
    assert "Blocks:" not in chunk.text
    assert "Blocked by:" not in chunk.text


def test_chunk_task_quality_and_tokens() -> None:
    task = TaskItem(
        id="2", session_id="s1", subject="Implement feature",
        description="Build the main feature", status="pending",
    )
    chunk = chunk_task(task)
    assert chunk.token_estimate > 0
    assert chunk.quality_score > 0


def test_chunk_task_empty_description() -> None:
    task = TaskItem(id="1", session_id="s1", subject="No desc", status="pending")
    chunk = chunk_task(task)
    assert chunk.text == "[TASK] No desc (pending)"
