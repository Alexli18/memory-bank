"""Tests for mb.renderers — XML, JSON, Markdown renderers and get_renderer factory."""

from __future__ import annotations

import json

from mb.models import Chunk, PackFormat, ProjectState
from mb.renderers import (
    JsonRenderer,
    MarkdownRenderer,
    XmlRenderer,
    get_renderer,
)


def _sample_state() -> ProjectState:
    return ProjectState(
        summary="Test project summary",
        decisions=[{"id": "D1", "statement": "Use Python", "rationale": "Best fit"}],
        constraints=["No new dependencies"],
        tasks=[{"id": "T1", "status": "in_progress"}],
        updated_at=1700000000.0,
        source_sessions=["20260224-120000-abcd"],
    )


def _sample_excerpts() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="20260224-120000-abcd-0",
            session_id="20260224-120000-abcd",
            index=0,
            text="Hello, world! Process finished",
            ts_start=1.0,
            ts_end=2.0,
            token_estimate=7,
            quality_score=0.75,
        ),
    ]


class TestXmlRenderer:
    def test_output_has_envelope(self) -> None:
        renderer = XmlRenderer()
        output = renderer.render(_sample_state(), _sample_excerpts())
        assert output.startswith('<MEMORY_BANK_CONTEXT version="1.0">')
        assert output.endswith("</MEMORY_BANK_CONTEXT>")

    def test_contains_project_state(self) -> None:
        renderer = XmlRenderer()
        output = renderer.render(_sample_state(), _sample_excerpts())
        assert "<PROJECT_STATE>" in output
        assert "<SUMMARY>Test project summary</SUMMARY>" in output

    def test_contains_decisions(self) -> None:
        renderer = XmlRenderer()
        output = renderer.render(_sample_state(), _sample_excerpts())
        assert '<DECISION id="D1">' in output
        assert "<STATEMENT>Use Python</STATEMENT>" in output

    def test_contains_constraints(self) -> None:
        renderer = XmlRenderer()
        output = renderer.render(_sample_state(), _sample_excerpts())
        assert "<CONSTRAINT>No new dependencies</CONSTRAINT>" in output

    def test_contains_tasks(self) -> None:
        renderer = XmlRenderer()
        output = renderer.render(_sample_state(), _sample_excerpts())
        assert '<TASK id="T1" status="in_progress"/>' in output

    def test_contains_excerpts(self) -> None:
        renderer = XmlRenderer()
        output = renderer.render(_sample_state(), _sample_excerpts())
        assert "<RECENT_CONTEXT_EXCERPTS>" in output
        assert 'chunk_id="20260224-120000-abcd-0"' in output

    def test_contains_instructions(self) -> None:
        renderer = XmlRenderer()
        output = renderer.render(_sample_state(), _sample_excerpts())
        assert "<INSTRUCTIONS>" in output
        assert "Paste this into a fresh LLM session" in output

    def test_empty_state_produces_valid_xml(self) -> None:
        state = ProjectState(
            summary="", decisions=[], constraints=[], tasks=[],
            updated_at=0.0, source_sessions=[],
        )
        renderer = XmlRenderer()
        output = renderer.render(state, [])
        assert "<DECISIONS/>" in output
        assert "<CONSTRAINTS/>" in output
        assert "<ACTIVE_TASKS/>" in output
        assert "<RECENT_CONTEXT_EXCERPTS/>" in output

    def test_escapes_special_characters(self) -> None:
        state = ProjectState(
            summary="Use <b>bold</b> & 'quotes'",
            decisions=[], constraints=[], tasks=[],
            updated_at=0.0, source_sessions=[],
        )
        renderer = XmlRenderer()
        output = renderer.render(state, [])
        assert "&lt;b&gt;" in output
        assert "&amp;" in output


class TestJsonRenderer:
    def test_output_is_valid_json(self) -> None:
        renderer = JsonRenderer()
        output = renderer.render(_sample_state(), _sample_excerpts())
        data = json.loads(output)
        assert isinstance(data, dict)

    def test_contains_version(self) -> None:
        renderer = JsonRenderer()
        output = renderer.render(_sample_state(), _sample_excerpts())
        data = json.loads(output)
        assert data["version"] == "1.0"

    def test_contains_project_state(self) -> None:
        renderer = JsonRenderer()
        output = renderer.render(_sample_state(), _sample_excerpts())
        data = json.loads(output)
        assert data["project_state"]["summary"] == "Test project summary"
        assert data["project_state"]["source_sessions"] == ["20260224-120000-abcd"]

    def test_contains_decisions(self) -> None:
        renderer = JsonRenderer()
        output = renderer.render(_sample_state(), _sample_excerpts())
        data = json.loads(output)
        assert len(data["decisions"]) == 1
        assert data["decisions"][0]["id"] == "D1"

    def test_contains_constraints(self) -> None:
        renderer = JsonRenderer()
        output = renderer.render(_sample_state(), _sample_excerpts())
        data = json.loads(output)
        assert data["constraints"] == ["No new dependencies"]

    def test_contains_tasks(self) -> None:
        renderer = JsonRenderer()
        output = renderer.render(_sample_state(), _sample_excerpts())
        data = json.loads(output)
        assert len(data["active_tasks"]) == 1
        assert data["active_tasks"][0]["id"] == "T1"

    def test_contains_excerpts(self) -> None:
        renderer = JsonRenderer()
        output = renderer.render(_sample_state(), _sample_excerpts())
        data = json.loads(output)
        assert len(data["recent_excerpts"]) == 1
        assert data["recent_excerpts"][0]["chunk_id"] == "20260224-120000-abcd-0"

    def test_contains_instructions(self) -> None:
        renderer = JsonRenderer()
        output = renderer.render(_sample_state(), _sample_excerpts())
        data = json.loads(output)
        assert "Paste this into a fresh LLM session" in data["instructions"]

    def test_empty_state(self) -> None:
        state = ProjectState(
            summary="", decisions=[], constraints=[], tasks=[],
            updated_at=0.0, source_sessions=[],
        )
        renderer = JsonRenderer()
        output = renderer.render(state, [])
        data = json.loads(output)
        assert data["decisions"] == []
        assert data["recent_excerpts"] == []


class TestMarkdownRenderer:
    def test_has_heading(self) -> None:
        renderer = MarkdownRenderer()
        output = renderer.render(_sample_state(), _sample_excerpts())
        assert output.startswith("# Memory Bank Context")

    def test_contains_project_state(self) -> None:
        renderer = MarkdownRenderer()
        output = renderer.render(_sample_state(), _sample_excerpts())
        assert "## Project State" in output
        assert "Test project summary" in output

    def test_contains_decisions(self) -> None:
        renderer = MarkdownRenderer()
        output = renderer.render(_sample_state(), _sample_excerpts())
        assert "## Decisions" in output
        assert "**D1**" in output
        assert "Use Python" in output

    def test_contains_constraints(self) -> None:
        renderer = MarkdownRenderer()
        output = renderer.render(_sample_state(), _sample_excerpts())
        assert "## Constraints" in output
        assert "- No new dependencies" in output

    def test_contains_tasks(self) -> None:
        renderer = MarkdownRenderer()
        output = renderer.render(_sample_state(), _sample_excerpts())
        assert "## Active Tasks" in output
        assert "**T1**" in output

    def test_contains_excerpts(self) -> None:
        renderer = MarkdownRenderer()
        output = renderer.render(_sample_state(), _sample_excerpts())
        assert "## Recent Context" in output
        assert "Hello, world! Process finished" in output

    def test_ends_with_instructions(self) -> None:
        renderer = MarkdownRenderer()
        output = renderer.render(_sample_state(), _sample_excerpts())
        assert output.rstrip().endswith("*Paste this into a fresh LLM session to restore context.*")

    def test_empty_state(self) -> None:
        state = ProjectState(
            summary="", decisions=[], constraints=[], tasks=[],
            updated_at=0.0, source_sessions=[],
        )
        renderer = MarkdownRenderer()
        output = renderer.render(state, [])
        assert "No decisions recorded." in output
        assert "No active tasks." in output


class TestGetRenderer:
    def test_xml_returns_xml_renderer(self) -> None:
        renderer = get_renderer(PackFormat.XML)
        assert isinstance(renderer, XmlRenderer)

    def test_json_returns_json_renderer(self) -> None:
        renderer = get_renderer(PackFormat.JSON)
        assert isinstance(renderer, JsonRenderer)

    def test_md_returns_markdown_renderer(self) -> None:
        renderer = get_renderer(PackFormat.MARKDOWN)
        assert isinstance(renderer, MarkdownRenderer)


class TestCrossFormatComparison:
    """SC-003 validation: same logical content across all three formats."""

    def test_all_formats_contain_same_logical_sections(self) -> None:
        state = _sample_state()
        excerpts = _sample_excerpts()

        xml_out = XmlRenderer().render(state, excerpts)
        json_out = JsonRenderer().render(state, excerpts)
        md_out = MarkdownRenderer().render(state, excerpts)

        # All contain the summary
        assert "Test project summary" in xml_out
        assert "Test project summary" in json_out
        assert "Test project summary" in md_out

        # All contain the decision
        assert "Use Python" in xml_out
        assert "Use Python" in json_out
        assert "Use Python" in md_out

        # All contain the constraint
        assert "No new dependencies" in xml_out
        assert "No new dependencies" in json_out
        assert "No new dependencies" in md_out

        # All contain the excerpt text
        assert "Hello, world! Process finished" in xml_out
        assert "Hello, world! Process finished" in json_out
        assert "Hello, world! Process finished" in md_out

        # All contain instructions
        assert "Paste this into a fresh LLM session" in xml_out
        assert "Paste this into a fresh LLM session" in json_out
        assert "Paste this into a fresh LLM session" in md_out

    def test_json_output_is_parseable(self) -> None:
        state = _sample_state()
        excerpts = _sample_excerpts()
        json_out = JsonRenderer().render(state, excerpts)
        data = json.loads(json_out)
        assert data["version"] == "1.0"
        assert len(data["recent_excerpts"]) == 1
