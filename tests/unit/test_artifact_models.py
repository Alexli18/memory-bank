"""Tests for artifact model dataclasses — round-trip, lenient deserialization, validation."""

from __future__ import annotations

from mb.models import PlanMeta, SearchResult, TaskItem, TodoItem, TodoList


# --- TodoItem ---


def test_todo_item_round_trip() -> None:
    data = {
        "id": "1",
        "content": "Fix the bug",
        "status": "in_progress",
        "priority": "high",
        "activeForm": "Fixing the bug",
    }
    item = TodoItem.from_dict(data)
    assert item.id == "1"
    assert item.content == "Fix the bug"
    assert item.status == "in_progress"
    assert item.priority == "high"
    assert item.active_form == "Fixing the bug"
    roundtrip = TodoItem.from_dict(item.to_dict())
    assert roundtrip == item


def test_todo_item_defaults() -> None:
    item = TodoItem.from_dict({"id": "2", "content": "Task"})
    assert item.status == "pending"
    assert item.priority == "medium"
    assert item.active_form is None


def test_todo_item_invalid_status_defaults() -> None:
    item = TodoItem.from_dict({"id": "3", "content": "X", "status": "UNKNOWN"})
    assert item.status == "pending"


def test_todo_item_invalid_priority_defaults() -> None:
    item = TodoItem.from_dict({"id": "4", "content": "X", "priority": "critical"})
    assert item.priority == "medium"


def test_todo_item_missing_fields() -> None:
    item = TodoItem.from_dict({})
    assert item.id == ""
    assert item.content == ""
    assert item.status == "pending"
    assert item.priority == "medium"


# --- TodoList ---


def test_todo_list_round_trip() -> None:
    data = {
        "session_id": "abc-123",
        "agent_id": "agent-456",
        "items": [
            {"id": "1", "content": "Item 1", "status": "pending", "priority": "high"},
            {"id": "2", "content": "Item 2", "status": "completed", "priority": "low"},
        ],
        "file_path": "/path/to/file.json",
        "mtime": 1700000000.0,
    }
    tl = TodoList.from_dict(data)
    assert tl.session_id == "abc-123"
    assert tl.agent_id == "agent-456"
    assert len(tl.items) == 2
    assert isinstance(tl.items, tuple)
    assert tl.items[0].content == "Item 1"
    roundtrip = TodoList.from_dict(tl.to_dict())
    assert roundtrip == tl


def test_todo_list_defaults() -> None:
    tl = TodoList.from_dict({"session_id": "s1"})
    assert tl.agent_id is None
    assert tl.items == ()
    assert tl.file_path == ""
    assert tl.mtime == 0.0


# --- PlanMeta ---


def test_plan_meta_round_trip() -> None:
    data = {
        "slug": "abundant-jingling-snail",
        "session_id": "sess-1",
        "timestamp": "2026-02-23T20:22:36.241Z",
        "file_path": "/path/to/plan.md",
        "mtime": 1700000000.0,
    }
    pm = PlanMeta.from_dict(data)
    assert pm.slug == "abundant-jingling-snail"
    assert pm.timestamp == "2026-02-23T20:22:36.241Z"
    roundtrip = PlanMeta.from_dict(pm.to_dict())
    assert roundtrip == pm


def test_plan_meta_defaults() -> None:
    pm = PlanMeta.from_dict({"slug": "s", "session_id": "ss"})
    assert pm.timestamp is None
    assert pm.file_path == ""
    assert pm.mtime == 0.0


# --- TaskItem ---


def test_task_item_round_trip() -> None:
    data = {
        "id": "5",
        "session_id": "sess-abc",
        "subject": "Create config",
        "description": "Set up project configuration",
        "activeForm": "Creating config",
        "status": "in_progress",
        "blocks": ["6", "7"],
        "blockedBy": ["3"],
    }
    task = TaskItem.from_dict(data)
    assert task.id == "5"
    assert task.subject == "Create config"
    assert task.active_form == "Creating config"
    assert task.status == "in_progress"
    assert task.blocks == ("6", "7")
    assert task.blocked_by == ("3",)
    roundtrip = TaskItem.from_dict(task.to_dict())
    assert roundtrip == task


def test_task_item_defaults() -> None:
    task = TaskItem.from_dict({"id": "1", "session_id": "s1"})
    assert task.subject == ""
    assert task.description == ""
    assert task.active_form is None
    assert task.status == "pending"
    assert task.blocks == ()
    assert task.blocked_by == ()


def test_task_item_invalid_status_defaults() -> None:
    task = TaskItem.from_dict({"id": "1", "session_id": "s", "status": "GARBAGE"})
    assert task.status == "pending"


def test_task_item_deleted_status() -> None:
    task = TaskItem.from_dict({"id": "1", "session_id": "s", "status": "deleted"})
    assert task.status == "deleted"


def test_task_item_blocked_by_camel_case() -> None:
    """blockedBy in source JSON maps to blocked_by field."""
    task = TaskItem.from_dict({
        "id": "1",
        "session_id": "s",
        "blockedBy": ["2", "3"],
    })
    assert task.blocked_by == ("2", "3")


def test_task_item_blocked_by_snake_case() -> None:
    """blocked_by also accepted (round-trip from to_dict is camelCase though)."""
    task = TaskItem.from_dict({
        "id": "1",
        "session_id": "s",
        "blocked_by": ["4"],
    })
    assert task.blocked_by == ("4",)


# --- SearchResult with artifact_type ---


def test_search_result_artifact_type_round_trip() -> None:
    data = {
        "chunk_id": "s1-0",
        "session_id": "s1",
        "index": 0,
        "text": "plan text",
        "ts_start": 1.0,
        "ts_end": 2.0,
        "token_estimate": 10,
        "quality_score": 0.8,
        "score": 0.95,
        "artifact_type": "plan",
    }
    result = SearchResult.from_dict(data)
    assert result.artifact_type == "plan"
    d = result.to_dict()
    assert d["artifact_type"] == "plan"
    roundtrip = SearchResult.from_dict(d)
    assert roundtrip == result


def test_search_result_artifact_type_none_by_default() -> None:
    data = {
        "chunk_id": "s1-0",
        "session_id": "s1",
        "index": 0,
        "text": "text",
        "ts_start": 1.0,
        "ts_end": 2.0,
        "token_estimate": 1,
        "quality_score": 0.5,
        "score": 0.9,
    }
    result = SearchResult.from_dict(data)
    assert result.artifact_type is None
    d = result.to_dict()
    assert "artifact_type" not in d


def test_search_result_backward_compatible() -> None:
    """Existing SearchResult usage without artifact_type still works."""
    result = SearchResult.from_dict({
        "chunk_id": "c1",
        "session_id": "s1",
        "index": 0,
        "text": "t",
        "score": 0.5,
    })
    assert result.artifact_type is None
    roundtrip = SearchResult.from_dict(result.to_dict())
    assert roundtrip == result
