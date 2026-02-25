"""Tests for artifact sections in context packs (Phase 5: US3).

Tests:
- _load_active_items: filters pending/in_progress, respects max_sessions, sorts by priority
- _load_recent_plans: returns N most recent by mtime
- All three renderers: include artifact sections when data present, omit when absent
- Budget allocation: respects 15% limits
- Backward compatibility: no artifacts dir → pack works normally
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mb.models import Chunk, ProjectState
from mb.store import NdjsonStorage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def storage(tmp_path: Path) -> NdjsonStorage:
    """Create a minimal NdjsonStorage instance."""
    root = tmp_path / ".memory-bank"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"ollama": {}}), encoding="utf-8"
    )
    (root / "sessions").mkdir()
    (root / "index").mkdir()
    (root / "state").mkdir()
    return NdjsonStorage(root)


@pytest.fixture()
def state() -> ProjectState:
    """Minimal ProjectState for renderer tests."""
    return ProjectState(
        summary="Test summary",
        decisions=[],
        constraints=[],
        tasks=[],
        updated_at=time.time(),
        source_sessions=["s1"],
    )


def _make_chunk(text: str = "test chunk", **kwargs: Any) -> Chunk:
    return Chunk(
        chunk_id=kwargs.get("chunk_id", "c1"),
        session_id=kwargs.get("session_id", "s1"),
        index=kwargs.get("index", 0),
        text=text,
        ts_start=kwargs.get("ts_start", 0.0),
        ts_end=kwargs.get("ts_end", 10.0),
        token_estimate=kwargs.get("token_estimate", len(text) // 4),
        quality_score=kwargs.get("quality_score", 0.8),
    )


# ---------------------------------------------------------------------------
# _load_active_items tests
# ---------------------------------------------------------------------------


class TestLoadActiveItems:
    def test_no_artifacts_dir(self, storage: NdjsonStorage) -> None:
        from mb.pack import _load_active_items

        result = _load_active_items(storage)
        assert result == []

    def test_loads_pending_todos(self, storage: NdjsonStorage) -> None:
        from mb.pack import _load_active_items

        todos_dir = storage.artifacts_dir / "todos"
        todos_dir.mkdir(parents=True)
        (todos_dir / "session-abc.json").write_text(
            json.dumps([
                {"id": "1", "content": "Fix bug", "status": "pending", "priority": "high"},
                {"id": "2", "content": "Done task", "status": "completed", "priority": "low"},
                {"id": "3", "content": "In progress", "status": "in_progress", "priority": "medium"},
            ]),
            encoding="utf-8",
        )

        result = _load_active_items(storage)
        assert len(result) == 2
        assert all(r["type"] == "todo" for r in result)
        assert result[0]["priority"] == "high"  # sorted by priority
        assert result[1]["priority"] == "medium"

    def test_loads_pending_tasks(self, storage: NdjsonStorage) -> None:
        from mb.pack import _load_active_items

        tasks_dir = storage.artifacts_dir / "tasks" / "session-xyz"
        tasks_dir.mkdir(parents=True)
        (tasks_dir / "1.json").write_text(
            json.dumps({
                "id": "1", "subject": "Implement feature",
                "status": "in_progress", "priority": "high",
            }),
            encoding="utf-8",
        )
        (tasks_dir / "2.json").write_text(
            json.dumps({
                "id": "2", "subject": "Completed feature",
                "status": "completed",
            }),
            encoding="utf-8",
        )

        result = _load_active_items(storage)
        assert len(result) == 1
        assert result[0]["type"] == "task"
        assert result[0]["status"] == "in_progress"
        assert result[0]["text"] == "Implement feature"

    def test_max_sessions_limit(self, storage: NdjsonStorage) -> None:
        from mb.pack import _load_active_items

        todos_dir = storage.artifacts_dir / "todos"
        todos_dir.mkdir(parents=True)
        for i in range(10):
            (todos_dir / f"session-{i:03d}.json").write_text(
                json.dumps([{"id": "1", "content": f"Task {i}", "status": "pending"}]),
                encoding="utf-8",
            )
            # Ensure different mtimes
            time.sleep(0.01)

        result = _load_active_items(storage, max_sessions=3)
        assert len(result) == 3

    def test_sorts_by_priority(self, storage: NdjsonStorage) -> None:
        from mb.pack import _load_active_items

        todos_dir = storage.artifacts_dir / "todos"
        todos_dir.mkdir(parents=True)
        (todos_dir / "s1.json").write_text(
            json.dumps([
                {"id": "1", "content": "Low prio", "status": "pending", "priority": "low"},
                {"id": "2", "content": "High prio", "status": "pending", "priority": "high"},
                {"id": "3", "content": "Med prio", "status": "pending", "priority": "medium"},
            ]),
            encoding="utf-8",
        )

        result = _load_active_items(storage)
        priorities = [r["priority"] for r in result]
        assert priorities == ["high", "medium", "low"]

    def test_skips_malformed_files(self, storage: NdjsonStorage) -> None:
        from mb.pack import _load_active_items

        todos_dir = storage.artifacts_dir / "todos"
        todos_dir.mkdir(parents=True)
        (todos_dir / "bad.json").write_text("not valid json", encoding="utf-8")
        (todos_dir / "good.json").write_text(
            json.dumps([{"id": "1", "content": "OK", "status": "pending"}]),
            encoding="utf-8",
        )

        result = _load_active_items(storage)
        assert len(result) == 1
        assert result[0]["text"] == "OK"

    def test_handles_dict_format_todos(self, storage: NdjsonStorage) -> None:
        from mb.pack import _load_active_items

        todos_dir = storage.artifacts_dir / "todos"
        todos_dir.mkdir(parents=True)
        (todos_dir / "s1.json").write_text(
            json.dumps({
                "items": [{"id": "1", "content": "Task", "status": "pending"}],
            }),
            encoding="utf-8",
        )

        result = _load_active_items(storage)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _load_recent_plans tests
# ---------------------------------------------------------------------------


class TestLoadRecentPlans:
    def test_no_plans_dir(self, storage: NdjsonStorage) -> None:
        from mb.pack import _load_recent_plans

        result = _load_recent_plans(storage)
        assert result == []

    def test_loads_plans(self, storage: NdjsonStorage) -> None:
        from mb.pack import _load_recent_plans

        plans_dir = storage.artifacts_dir / "plans"
        plans_dir.mkdir(parents=True)
        (plans_dir / "my-plan.md").write_text("# My Plan\nSome content", encoding="utf-8")

        result = _load_recent_plans(storage)
        assert len(result) == 1
        assert result[0]["slug"] == "my-plan"
        assert "# My Plan" in result[0]["text"]

    def test_max_plans_limit(self, storage: NdjsonStorage) -> None:
        from mb.pack import _load_recent_plans

        plans_dir = storage.artifacts_dir / "plans"
        plans_dir.mkdir(parents=True)
        for i in range(5):
            (plans_dir / f"plan-{i}.md").write_text(f"Plan {i}", encoding="utf-8")
            time.sleep(0.01)

        result = _load_recent_plans(storage, max_plans=2)
        assert len(result) == 2

    def test_returns_most_recent_first(self, storage: NdjsonStorage) -> None:
        from mb.pack import _load_recent_plans

        plans_dir = storage.artifacts_dir / "plans"
        plans_dir.mkdir(parents=True)
        (plans_dir / "old-plan.md").write_text("Old", encoding="utf-8")
        time.sleep(0.02)
        (plans_dir / "new-plan.md").write_text("New", encoding="utf-8")

        result = _load_recent_plans(storage)
        assert result[0]["slug"] == "new-plan"

    def test_ignores_meta_json(self, storage: NdjsonStorage) -> None:
        from mb.pack import _load_recent_plans

        plans_dir = storage.artifacts_dir / "plans"
        plans_dir.mkdir(parents=True)
        (plans_dir / "my-plan.md").write_text("Content", encoding="utf-8")
        (plans_dir / "my-plan.meta.json").write_text("{}", encoding="utf-8")

        result = _load_recent_plans(storage)
        assert len(result) == 1
        assert result[0]["slug"] == "my-plan"


# ---------------------------------------------------------------------------
# XmlRenderer artifact section tests
# ---------------------------------------------------------------------------


class TestXmlRendererArtifacts:
    def test_active_tasks_with_artifact_items(self, state: ProjectState) -> None:
        from mb.renderers import XmlRenderer

        active_items = [
            {"type": "task", "session_id": "s1", "id": "3", "status": "in_progress", "text": "Do work"},
            {"type": "todo", "session_id": "s2", "status": "pending", "priority": "high", "text": "Review"},
        ]
        renderer = XmlRenderer()
        output = renderer.render(state, [], active_items=active_items)

        assert "<ACTIVE_TASKS>" in output
        assert '<task session="s1" id="3" status="in_progress">' in output
        assert "Do work" in output
        assert '<todo session="s2" status="pending" priority="high">' in output
        assert "Review" in output

    def test_plans_section(self, state: ProjectState) -> None:
        from mb.renderers import XmlRenderer

        plans = [{"slug": "my-plan", "text": "Plan content"}]
        renderer = XmlRenderer()
        output = renderer.render(state, [], plans=plans)

        assert "<PLANS>" in output
        assert '<plan slug="my-plan">' in output
        assert "Plan content" in output

    def test_no_plans_section_when_none(self, state: ProjectState) -> None:
        from mb.renderers import XmlRenderer

        renderer = XmlRenderer()
        output = renderer.render(state, [])

        assert "<PLANS>" not in output
        assert "</PLANS>" not in output

    def test_fallback_to_state_tasks(self, state: ProjectState) -> None:
        from mb.renderers import XmlRenderer

        state_with_tasks = ProjectState(
            summary="Summary",
            decisions=[], constraints=[],
            tasks=[{"id": "1", "status": "active"}],
            updated_at=time.time(),
            source_sessions=["s1"],
        )
        renderer = XmlRenderer()
        output = renderer.render(state_with_tasks, [])

        assert '<TASK id="1" status="active"/>' in output


# ---------------------------------------------------------------------------
# JsonRenderer artifact section tests
# ---------------------------------------------------------------------------


class TestJsonRendererArtifacts:
    def test_active_tasks_with_artifact_items(self, state: ProjectState) -> None:
        from mb.renderers import JsonRenderer

        active_items = [
            {"type": "task", "session_id": "s1", "id": "3", "status": "in_progress", "text": "Do work"},
        ]
        renderer = JsonRenderer()
        output = renderer.render(state, [], active_items=active_items)
        data = json.loads(output)

        assert data["active_tasks"] == active_items

    def test_plans_in_json(self, state: ProjectState) -> None:
        from mb.renderers import JsonRenderer

        plans = [{"slug": "my-plan", "text": "Content"}]
        renderer = JsonRenderer()
        output = renderer.render(state, [], plans=plans)
        data = json.loads(output)

        assert data["plans"] == plans

    def test_no_plans_key_when_none(self, state: ProjectState) -> None:
        from mb.renderers import JsonRenderer

        renderer = JsonRenderer()
        output = renderer.render(state, [])
        data = json.loads(output)

        assert "plans" not in data

    def test_fallback_to_state_tasks(self, state: ProjectState) -> None:
        from mb.renderers import JsonRenderer

        state_with_tasks = ProjectState(
            summary="Summary",
            decisions=[], constraints=[],
            tasks=[{"id": "1", "status": "active"}],
            updated_at=time.time(),
            source_sessions=["s1"],
        )
        renderer = JsonRenderer()
        output = renderer.render(state_with_tasks, [])
        data = json.loads(output)

        assert data["active_tasks"] == [{"id": "1", "status": "active"}]


# ---------------------------------------------------------------------------
# MarkdownRenderer artifact section tests
# ---------------------------------------------------------------------------


class TestMarkdownRendererArtifacts:
    def test_active_tasks_with_artifact_items(self, state: ProjectState) -> None:
        from mb.renderers import MarkdownRenderer

        active_items = [
            {"type": "task", "session_id": "s1", "id": "3", "status": "in_progress", "text": "Do work"},
            {"type": "todo", "session_id": "s2", "status": "pending", "priority": "high", "text": "Review"},
        ]
        renderer = MarkdownRenderer()
        output = renderer.render(state, [], active_items=active_items)

        assert "## Active Tasks" in output
        assert "**[in_progress]** Do work _(task #3, session s1)_" in output
        assert "**[pending/high]** Review _(todo, session s2)_" in output

    def test_plans_section(self, state: ProjectState) -> None:
        from mb.renderers import MarkdownRenderer

        plans = [{"slug": "my-plan", "text": "Plan content"}]
        renderer = MarkdownRenderer()
        output = renderer.render(state, [], plans=plans)

        assert "## Plans" in output
        assert "### my-plan" in output
        assert "> Plan content" in output

    def test_no_plans_section_when_none(self, state: ProjectState) -> None:
        from mb.renderers import MarkdownRenderer

        renderer = MarkdownRenderer()
        output = renderer.render(state, [])

        assert "## Plans" not in output


# ---------------------------------------------------------------------------
# Budget allocation tests
# ---------------------------------------------------------------------------


class TestBudgetAllocation:
    def test_artifact_sections_respect_15_percent_limit(self) -> None:
        from mb.budgeter import MAX_SHARE_ACTIVE_TASKS, MAX_SHARE_PLANS, Section, apply_budget

        # Large artifact section content
        large_content = "x" * 10000
        sections = [
            Section("PROJECT_STATE", "state", 0, True),
            Section("ACTIVE_TASKS", large_content, 2, False, max_tokens=int(6000 * 0.15)),
            Section("PLANS", large_content, 3, False, max_tokens=int(6000 * 0.15)),
            Section("RECENT_CONTEXT_EXCERPTS", "excerpts", 4, False),
            Section("INSTRUCTIONS", "instr", 0, True),
        ]
        result = apply_budget(sections, 6000)

        # ACTIVE_TASKS and PLANS should be capped
        at_section = next(s for s in result if s.name == "ACTIVE_TASKS")
        plans_section = next(s for s in result if s.name == "PLANS")
        assert at_section.token_count <= int(6000 * MAX_SHARE_ACTIVE_TASKS) + 1
        assert plans_section.token_count <= int(6000 * MAX_SHARE_PLANS) + 1

    def test_budget_constants_defined(self) -> None:
        from mb.budgeter import (
            MAX_SHARE_ACTIVE_TASKS,
            MAX_SHARE_PLANS,
            PRIORITY_ACTIVE_TASKS,
            PRIORITY_PLANS,
            PRIORITY_RECENT_CONTEXT,
        )

        assert MAX_SHARE_ACTIVE_TASKS == 0.15
        assert MAX_SHARE_PLANS == 0.15
        assert PRIORITY_ACTIVE_TASKS < PRIORITY_PLANS < PRIORITY_RECENT_CONTEXT


# ---------------------------------------------------------------------------
# Backward compatibility tests
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_no_artifacts_dir_pack_works(self, storage: NdjsonStorage) -> None:
        """When no artifacts/ exists, _load helpers return empty lists."""
        from mb.pack import _load_active_items, _load_recent_plans

        assert _load_active_items(storage) == []
        assert _load_recent_plans(storage) == []

    def test_renderers_omit_sections_without_artifacts(self, state: ProjectState) -> None:
        """When no artifact data, sections are not added."""
        from mb.renderers import JsonRenderer, MarkdownRenderer, XmlRenderer

        xml_out = XmlRenderer().render(state, [])
        assert "<PLANS>" not in xml_out

        json_out = JsonRenderer().render(state, [])
        data = json.loads(json_out)
        assert "plans" not in data

        md_out = MarkdownRenderer().render(state, [])
        assert "## Plans" not in md_out

    def test_xml_build_sections_backward_compat(self, state: ProjectState) -> None:
        """_build_sections without active_items uses state.tasks format."""
        from mb.renderers import XmlRenderer

        renderer = XmlRenderer()
        sections = renderer._build_sections(state, [])

        # Should have self-closing ACTIVE_TASKS (no artifact items)
        assert sections.get("ACTIVE_TASKS") == "  <ACTIVE_TASKS/>"

    def test_pack_apply_budget_with_plans(self) -> None:
        """_apply_budget handles PLANS section in section_order."""
        from mb.pack import _apply_budget

        sections = {
            "PROJECT_STATE": "  <PROJECT_STATE><SUMMARY>summary</SUMMARY></PROJECT_STATE>",
            "DECISIONS": "  <DECISIONS/>",
            "CONSTRAINTS": "  <CONSTRAINTS/>",
            "ACTIVE_TASKS": "  <ACTIVE_TASKS/>",
            "PLANS": '  <PLANS>\n    <plan slug="test">\n      content\n    </plan>\n  </PLANS>',
            "RECENT_CONTEXT_EXCERPTS": "  <RECENT_CONTEXT_EXCERPTS/>",
            "INSTRUCTIONS": "  <INSTRUCTIONS>Paste...</INSTRUCTIONS>",
        }
        result = _apply_budget(sections, 10000)
        assert "<PLANS>" in result
        assert "content" in result


# ---------------------------------------------------------------------------
# count_artifacts tests (Phase 6: US4 - co-located since they share fixtures)
# ---------------------------------------------------------------------------


class TestCountArtifacts:
    def test_no_artifacts_dir(self, storage: NdjsonStorage) -> None:
        result = storage.count_artifacts()
        assert result == {}

    def test_counts_plans(self, storage: NdjsonStorage) -> None:
        plans_dir = storage.artifacts_dir / "plans"
        plans_dir.mkdir(parents=True)
        (plans_dir / "plan-a.md").write_text("# Plan A", encoding="utf-8")
        (plans_dir / "plan-b.md").write_text("# Plan B", encoding="utf-8")

        result = storage.count_artifacts()
        assert result["plans"] == 2

    def test_counts_todos_and_active_items(self, storage: NdjsonStorage) -> None:
        todos_dir = storage.artifacts_dir / "todos"
        todos_dir.mkdir(parents=True)
        (todos_dir / "s1.json").write_text(
            json.dumps([
                {"id": "1", "content": "Active", "status": "pending"},
                {"id": "2", "content": "Done", "status": "completed"},
                {"id": "3", "content": "Working", "status": "in_progress"},
            ]),
            encoding="utf-8",
        )

        result = storage.count_artifacts()
        assert result["todos"] == 1
        assert result["todo_active_items"] == 2  # pending + in_progress

    def test_counts_tasks_and_pending(self, storage: NdjsonStorage) -> None:
        tasks_dir = storage.artifacts_dir / "tasks" / "session-1"
        tasks_dir.mkdir(parents=True)
        (tasks_dir / "1.json").write_text(
            json.dumps({"id": "1", "status": "pending"}), encoding="utf-8"
        )
        (tasks_dir / "2.json").write_text(
            json.dumps({"id": "2", "status": "completed"}), encoding="utf-8"
        )
        (tasks_dir / "3.json").write_text(
            json.dumps({"id": "3", "status": "in_progress"}), encoding="utf-8"
        )

        result = storage.count_artifacts()
        assert result["tasks"] == 1  # 1 session dir
        assert result["task_pending"] == 2  # pending + in_progress

    def test_returns_empty_when_all_zeros(self, storage: NdjsonStorage) -> None:
        storage.artifacts_dir.mkdir(parents=True)
        result = storage.count_artifacts()
        assert result == {}

    def test_sessions_cli_shows_artifact_summary(self, tmp_path: Path) -> None:
        """mb sessions displays artifact summary line when artifacts exist."""
        from click.testing import CliRunner

        from mb.cli import cli
        from mb.models import SessionMeta

        runner = CliRunner()
        sessions = [
            SessionMeta.from_dict({
                "session_id": "s1", "command": ["bash"],
                "cwd": str(tmp_path), "started_at": 1700000000.0, "exit_code": 0,
            }),
        ]
        mock_storage = MagicMock(spec=NdjsonStorage)
        mock_storage.list_sessions.return_value = sessions
        mock_storage.count_artifacts.return_value = {
            "plans": 2, "todos": 3, "todo_active_items": 5,
            "tasks": 1, "task_pending": 4,
        }
        with patch("mb.cli._require_storage", return_value=mock_storage):
            result = runner.invoke(cli, ["sessions"])

        assert result.exit_code == 0
        assert "Artifacts:" in result.output
        assert "2 plans" in result.output
        assert "3 todo lists (5 active items)" in result.output
        assert "1 task trees (4 pending tasks)" in result.output

    def test_sessions_cli_no_artifact_summary_when_empty(self, tmp_path: Path) -> None:
        """mb sessions omits artifact line when no artifacts exist."""
        from click.testing import CliRunner

        from mb.cli import cli
        from mb.models import SessionMeta

        runner = CliRunner()
        sessions = [
            SessionMeta.from_dict({
                "session_id": "s1", "command": ["bash"],
                "cwd": str(tmp_path), "started_at": 1700000000.0, "exit_code": 0,
            }),
        ]
        mock_storage = MagicMock(spec=NdjsonStorage)
        mock_storage.list_sessions.return_value = sessions
        mock_storage.count_artifacts.return_value = {}
        with patch("mb.cli._require_storage", return_value=mock_storage):
            result = runner.invoke(cli, ["sessions"])

        assert result.exit_code == 0
        assert "Artifacts:" not in result.output
