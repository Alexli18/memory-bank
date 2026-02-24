"""Context pack builder — thin orchestrator over retriever, budgeter, renderer.

Orchestration flow: retriever → budgeter → renderer.
"""

from __future__ import annotations

import sys
from typing import Any

from mb.budgeter import estimate_tokens
from mb.chunker import chunk_all_sessions
from mb.models import PackFormat
from mb.ollama_client import client_from_config
from mb.renderers import XmlRenderer, get_renderer
from mb.retriever import RecencyRetriever
from mb.state import _state_is_stale, generate_state, load_state
from mb.store import NdjsonStorage


def build_pack(
    budget: int,
    storage: NdjsonStorage,
    fmt: PackFormat = PackFormat.XML,
    retriever: Any | None = None,
) -> str:
    """Build a context pack within the given token budget.

    Steps:
    1. Create OllamaClient from config
    2. Generate/load ProjectState
    3. Retrieve recent excerpts via the given retriever (default: RecencyRetriever)
    4. Render via the chosen format renderer
    5. For XML: apply token budget with section truncation
       For JSON/MD: apply token budget by limiting excerpts

    Args:
        budget: Maximum token budget.
        storage: NdjsonStorage instance.
        fmt: Output format (default: XML).
        retriever: Retriever instance. If None, uses RecencyRetriever.

    Returns:
        Formatted context pack string.
    """
    config = storage.read_config()
    client = client_from_config(config)

    # Ensure all sessions are chunked before loading state
    chunk_all_sessions(storage)

    # Generate or load state (regenerate if stale)
    state = load_state(storage)
    if state is None or _state_is_stale(storage):
        state = generate_state(storage, client)

    # Retrieve excerpts
    if retriever is None:
        retriever = RecencyRetriever()
    excerpts = retriever.retrieve(storage)

    # Render with budget enforcement
    renderer = get_renderer(fmt)

    if fmt == PackFormat.XML:
        # XML uses the legacy budget truncation approach via _apply_budget
        xml_renderer = XmlRenderer()
        sections_dict = xml_renderer._build_sections(state, excerpts)
        return _apply_budget(sections_dict, budget)
    else:
        # For JSON/MD: render full, then truncate excerpts if over budget
        output = renderer.render(state, excerpts)
        while estimate_tokens(output) > budget and excerpts:
            excerpts = excerpts[:-1]
            output = renderer.render(state, excerpts)
        return output


# ---------------------------------------------------------------------------
# Backward-compatible internal helpers (used by XML path and existing tests)
# ---------------------------------------------------------------------------

def _collect_recent_excerpts(
    storage: NdjsonStorage,
    *,
    min_quality: float = 0.30,
    min_length: int = 30,
    max_excerpts: int = 200,
) -> list[Any]:
    """Collect most recent chunks — delegates to RecencyRetriever.

    Kept for backward compatibility with existing tests.
    """
    retriever = RecencyRetriever(
        min_quality=min_quality,
        min_length=min_length,
        max_excerpts=max_excerpts,
    )
    return retriever.retrieve(storage)


def _truncate_section(name: str, content: str, token_budget: int) -> str:
    """Truncate a section by removing whole XML elements from the end."""
    close_tags = {
        "RECENT_CONTEXT_EXCERPTS": "</EXCERPT>",
        "ACTIVE_TASKS": "/>",
        "DECISIONS": "</DECISION>",
    }
    close_tag = close_tags.get(name)
    if close_tag is None:
        return ""

    wrapper_close = {
        "RECENT_CONTEXT_EXCERPTS": "\n  </RECENT_CONTEXT_EXCERPTS>",
        "ACTIVE_TASKS": "\n  </ACTIVE_TASKS>",
        "DECISIONS": "\n  </DECISIONS>",
    }

    result = content
    while estimate_tokens(result) > token_budget:
        idx = result.rfind(close_tag)
        if idx < 0:
            return ""
        line_start = result.rfind("\n    <", 0, idx)
        if line_start < 0:
            return ""
        result = result[:line_start] + wrapper_close[name]

    return result


def _apply_budget(sections: dict[str, str], budget: int) -> str:
    """Apply token budget to XML sections, truncating in reverse priority order."""
    envelope_open = '<MEMORY_BANK_CONTEXT version="1.0">\n'
    envelope_close = "\n</MEMORY_BANK_CONTEXT>"

    section_order = [
        "PROJECT_STATE",
        "DECISIONS",
        "CONSTRAINTS",
        "ACTIVE_TASKS",
        "RECENT_CONTEXT_EXCERPTS",
        "INSTRUCTIONS",
    ]

    envelope_tokens = estimate_tokens(envelope_open + envelope_close)
    remaining_budget = budget - envelope_tokens

    section_tokens = {}
    for name in section_order:
        content = sections.get(name, "")
        section_tokens[name] = estimate_tokens(content)

    total_needed = sum(section_tokens.values())

    if total_needed <= remaining_budget:
        parts = [envelope_open]
        for name in section_order:
            parts.append(sections.get(name, ""))
        parts.append(envelope_close)
        return "\n".join(parts)

    truncatable = ["RECENT_CONTEXT_EXCERPTS", "ACTIVE_TASKS", "DECISIONS"]

    protected = sum(section_tokens[n] for n in section_order if n not in truncatable)
    available_for_truncatable = remaining_budget - protected

    if available_for_truncatable < 0:
        sys.stderr.write(
            f"Warning: Token budget ({budget}) too small for PROJECT_STATE. Output truncated.\n"
        )
        available_for_truncatable = 0

    allocated = {}
    budget_left = available_for_truncatable
    truncated = False
    for name in ["DECISIONS", "ACTIVE_TASKS", "RECENT_CONTEXT_EXCERPTS"]:
        needed = section_tokens.get(name, 0)
        if needed <= budget_left:
            allocated[name] = sections.get(name, "")
            budget_left -= needed
        elif budget_left > 0:
            allocated[name] = _truncate_section(name, sections.get(name, ""), budget_left)
            budget_left = 0
            truncated = True
        else:
            allocated[name] = ""
            truncated = True

    if truncated:
        sys.stderr.write(
            "Warning: Budget too small for full context. Some sections were truncated.\n"
        )

    parts = [envelope_open]
    for name in section_order:
        if name in truncatable:
            content = allocated.get(name, "")
            if content:
                parts.append(content)
        else:
            parts.append(sections.get(name, ""))
    parts.append(envelope_close)

    return "\n".join(parts)
