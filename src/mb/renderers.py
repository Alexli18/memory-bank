"""Renderer protocol and implementations for context pack output formats.

Provides XML, JSON, and Markdown renderers for context packs.
Each renderer takes structured section data and produces formatted output.
"""

from __future__ import annotations

import json
import time
from typing import Any, Protocol

from xml.sax.saxutils import escape

from mb.models import Chunk, PackFormat, ProjectState


class Renderer(Protocol):
    """Protocol for context pack output renderers."""

    def render(
        self,
        state: ProjectState,
        excerpts: list[Chunk],
    ) -> str: ...


class XmlRenderer:
    """Render context pack as XML (backward-compatible with original pack.py output)."""

    def render(
        self,
        state: ProjectState,
        excerpts: list[Chunk],
    ) -> str:
        sections = self._build_sections(state, excerpts)
        section_order = [
            "PROJECT_STATE",
            "DECISIONS",
            "CONSTRAINTS",
            "ACTIVE_TASKS",
            "RECENT_CONTEXT_EXCERPTS",
            "INSTRUCTIONS",
        ]
        parts = ['<MEMORY_BANK_CONTEXT version="1.0">\n']
        for name in section_order:
            parts.append(sections.get(name, ""))
        parts.append("\n</MEMORY_BANK_CONTEXT>")
        return "\n".join(parts)

    def _build_sections(
        self, state: ProjectState, excerpts: list[Chunk]
    ) -> dict[str, str]:
        sections: dict[str, str] = {}

        # PROJECT_STATE
        summary = escape(state.summary)
        generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        source = ", ".join(state.source_sessions)
        sections["PROJECT_STATE"] = (
            f"  <PROJECT_STATE>\n"
            f"    <GENERATED_AT>{generated_at}</GENERATED_AT>\n"
            f"    <SOURCE_SESSIONS>{escape(source)}</SOURCE_SESSIONS>\n"
            f"    <SUMMARY>{summary}</SUMMARY>\n"
            f"  </PROJECT_STATE>"
        )

        # DECISIONS
        decisions = state.decisions
        if decisions:
            items = []
            for d in decisions:
                did = escape(str(d.get("id", "")))
                stmt = escape(str(d.get("statement", "")))
                rat = escape(str(d.get("rationale", "")))
                items.append(
                    f'    <DECISION id="{did}">\n'
                    f"      <STATEMENT>{stmt}</STATEMENT>\n"
                    f"      <RATIONALE>{rat}</RATIONALE>\n"
                    f"    </DECISION>"
                )
            sections["DECISIONS"] = (
                "  <DECISIONS>\n" + "\n".join(items) + "\n  </DECISIONS>"
            )
        else:
            sections["DECISIONS"] = "  <DECISIONS/>"

        # CONSTRAINTS
        constraints = state.constraints
        if constraints:
            items = [
                f"    <CONSTRAINT>{escape(str(c))}</CONSTRAINT>" for c in constraints
            ]
            sections["CONSTRAINTS"] = (
                "  <CONSTRAINTS>\n" + "\n".join(items) + "\n  </CONSTRAINTS>"
            )
        else:
            sections["CONSTRAINTS"] = "  <CONSTRAINTS/>"

        # ACTIVE_TASKS
        tasks = state.tasks
        if tasks:
            items = []
            for t in tasks:
                tid = escape(str(t.get("id", "")))
                status = escape(str(t.get("status", "")))
                items.append(f'    <TASK id="{tid}" status="{status}"/>')
            sections["ACTIVE_TASKS"] = (
                "  <ACTIVE_TASKS>\n" + "\n".join(items) + "\n  </ACTIVE_TASKS>"
            )
        else:
            sections["ACTIVE_TASKS"] = "  <ACTIVE_TASKS/>"

        # RECENT_CONTEXT_EXCERPTS
        if excerpts:
            items = []
            for ex in excerpts:
                cid = escape(ex.chunk_id)
                ts = str(ex.ts_end)
                text = escape(ex.text)
                items.append(
                    f'    <EXCERPT chunk_id="{cid}" ts_end="{ts}">\n'
                    f"      {text}\n"
                    f"    </EXCERPT>"
                )
            sections["RECENT_CONTEXT_EXCERPTS"] = (
                "  <RECENT_CONTEXT_EXCERPTS>\n"
                + "\n".join(items)
                + "\n  </RECENT_CONTEXT_EXCERPTS>"
            )
        else:
            sections["RECENT_CONTEXT_EXCERPTS"] = "  <RECENT_CONTEXT_EXCERPTS/>"

        # INSTRUCTIONS
        sections["INSTRUCTIONS"] = (
            "  <INSTRUCTIONS>Paste this into a fresh LLM session to restore context.</INSTRUCTIONS>"
        )

        return sections


class JsonRenderer:
    """Render context pack as JSON per CLI contract format."""

    def render(
        self,
        state: ProjectState,
        excerpts: list[Chunk],
    ) -> str:
        generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result: dict[str, Any] = {
            "version": "1.0",
            "project_state": {
                "generated_at": generated_at,
                "source_sessions": state.source_sessions,
                "summary": state.summary,
            },
            "decisions": state.decisions,
            "constraints": state.constraints,
            "active_tasks": state.tasks,
            "recent_excerpts": [
                {
                    "chunk_id": ex.chunk_id,
                    "ts_end": ex.ts_end,
                    "text": ex.text,
                }
                for ex in excerpts
            ],
            "instructions": "Paste this into a fresh LLM session to restore context.",
        }
        return json.dumps(result, indent=2, ensure_ascii=False)


class MarkdownRenderer:
    """Render context pack as Markdown per CLI contract format."""

    def render(
        self,
        state: ProjectState,
        excerpts: list[Chunk],
    ) -> str:
        generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        sources = ", ".join(state.source_sessions)

        lines: list[str] = []
        lines.append("# Memory Bank Context")
        lines.append("")

        # Project State
        lines.append("## Project State")
        lines.append(f"**Generated**: {generated_at}")
        lines.append(f"**Sources**: {sources}")
        lines.append("")
        lines.append(state.summary)
        lines.append("")

        # Decisions
        lines.append("## Decisions")
        if state.decisions:
            for d in state.decisions:
                did = d.get("id", "")
                stmt = d.get("statement", "")
                rat = d.get("rationale", "")
                lines.append(f"- **{did}**: {stmt} — *{rat}*")
        else:
            lines.append("No decisions recorded.")
        lines.append("")

        # Constraints
        lines.append("## Constraints")
        if state.constraints:
            for c in state.constraints:
                lines.append(f"- {c}")
        else:
            lines.append("No constraints recorded.")
        lines.append("")

        # Active Tasks
        lines.append("## Active Tasks")
        if state.tasks:
            for t in state.tasks:
                tid = t.get("id", "")
                status = t.get("status", "")
                lines.append(f"- **{tid}**: {status}")
        else:
            lines.append("No active tasks.")
        lines.append("")

        # Recent Context
        lines.append("## Recent Context")
        if excerpts:
            for ex in excerpts:
                duration_s = ex.ts_end - ex.ts_start if ex.ts_start else 0
                duration_min = int(duration_s // 60)
                duration_sec = int(duration_s % 60)
                lines.append(
                    f"### {ex.session_id} ({duration_min:02d}:{duration_sec:02d})"
                )
                lines.append(ex.text)
                lines.append("")
        else:
            lines.append("No recent context available.")
            lines.append("")

        lines.append("---")
        lines.append("*Paste this into a fresh LLM session to restore context.*")

        return "\n".join(lines)


def get_renderer(fmt: PackFormat) -> Renderer:
    """Factory: return the appropriate renderer for the given format."""
    renderers: dict[PackFormat, Renderer] = {
        PackFormat.XML: XmlRenderer(),
        PackFormat.JSON: JsonRenderer(),
        PackFormat.MARKDOWN: MarkdownRenderer(),
    }
    renderer = renderers.get(fmt)
    if renderer is None:
        raise ValueError(f"Unknown pack format: {fmt}")
    return renderer
