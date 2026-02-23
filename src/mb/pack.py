"""Context pack XML builder with token budget enforcement."""

from __future__ import annotations

import heapq
import json
import sys
import time
from pathlib import Path
from xml.sax.saxutils import escape

from mb.chunker import _quality_score, chunk_all_sessions
from mb.ollama_client import OllamaClient
from mb.state import _state_is_stale, generate_state, load_state
from mb.storage import read_config


def estimate_tokens(text: str) -> int:
    """Estimate token count: chars/4 with 10% safety margin (FR-013)."""
    return int(len(text) / 4 * 1.1)


def build_pack(budget: int, storage_root: Path) -> str:
    """Build an XML context pack within the given token budget.

    Steps:
    1. Create OllamaClient from config
    2. Generate/load ProjectState
    3. Build XML sections in fixed priority order
    4. Apply token budget with truncation in reverse priority
    5. Wrap in MEMORY_BANK_CONTEXT envelope

    Args:
        budget: Maximum token budget.
        storage_root: Path to .memory-bank/.

    Returns:
        Raw XML string.
    """
    config = read_config(storage_root)
    ollama_cfg = config.get("ollama", {})
    client = OllamaClient(
        base_url=ollama_cfg.get("base_url", "http://localhost:11434"),
        embed_model=ollama_cfg.get("embed_model", "nomic-embed-text"),
        chat_model=ollama_cfg.get("chat_model", "gemma3:4b"),
    )

    # Ensure all sessions are chunked before loading state
    chunk_all_sessions(storage_root)

    # Generate or load state (regenerate if stale)
    state = load_state(storage_root)
    if state is None or _state_is_stale(storage_root):
        state = generate_state(storage_root, client)

    # Build sections
    sections = _build_sections(state, storage_root)

    # Apply budget
    xml = _apply_budget(sections, budget)

    return xml


def _build_sections(state: dict, storage_root: Path) -> dict[str, str]:
    """Build XML content for each section from state data."""
    sections: dict[str, str] = {}

    # PROJECT_STATE — never truncated
    summary = escape(state.get("summary", ""))
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    source = ", ".join(state.get("source_sessions", []))
    sections["PROJECT_STATE"] = (
        f"  <PROJECT_STATE>\n"
        f"    <GENERATED_AT>{generated_at}</GENERATED_AT>\n"
        f"    <SOURCE_SESSIONS>{escape(source)}</SOURCE_SESSIONS>\n"
        f"    <SUMMARY>{summary}</SUMMARY>\n"
        f"  </PROJECT_STATE>"
    )

    # DECISIONS
    decisions = state.get("decisions", [])
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

    # CONSTRAINTS — never truncated
    constraints = state.get("constraints", [])
    if constraints:
        items = [f"    <CONSTRAINT>{escape(str(c))}</CONSTRAINT>" for c in constraints]
        sections["CONSTRAINTS"] = (
            "  <CONSTRAINTS>\n" + "\n".join(items) + "\n  </CONSTRAINTS>"
        )
    else:
        sections["CONSTRAINTS"] = "  <CONSTRAINTS/>"

    # ACTIVE_TASKS
    tasks = state.get("tasks", [])
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

    # RECENT_CONTEXT_EXCERPTS — from chunks.jsonl, most recent first
    excerpts = _collect_recent_excerpts(storage_root)
    if excerpts:
        items = []
        for ex in excerpts:
            cid = escape(str(ex.get("chunk_id", "")))
            ts = str(ex.get("ts_end", 0))
            text = escape(str(ex.get("text", "")))
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

    # INSTRUCTIONS — static text
    sections["INSTRUCTIONS"] = (
        "  <INSTRUCTIONS>Paste this into a fresh LLM session to restore context.</INSTRUCTIONS>"
    )

    return sections


def _truncate_section(name: str, content: str, token_budget: int) -> str:
    """Truncate a section by removing whole XML elements from the end.

    Instead of cutting mid-tag (which breaks XML), removes the last element
    repeatedly until the section fits within the token budget.
    """
    # Map section names to their child element closing tags
    close_tags = {
        "RECENT_CONTEXT_EXCERPTS": "</EXCERPT>",
        "ACTIVE_TASKS": "/>",
        "DECISIONS": "</DECISION>",
    }
    close_tag = close_tags.get(name)
    if close_tag is None:
        # Unknown section — return empty wrapper
        return ""

    # Wrapper tags for rebuilding
    wrapper_close = {
        "RECENT_CONTEXT_EXCERPTS": "\n  </RECENT_CONTEXT_EXCERPTS>",
        "ACTIVE_TASKS": "\n  </ACTIVE_TASKS>",
        "DECISIONS": "\n  </DECISIONS>",
    }

    # Progressively remove last element until it fits
    result = content
    while estimate_tokens(result) > token_budget:
        idx = result.rfind(close_tag)
        if idx < 0:
            # No more elements to remove
            return ""
        # Find the start of this element's opening tag
        # Look backwards from idx for the nearest "    <"
        line_start = result.rfind("\n    <", 0, idx)
        if line_start < 0:
            # First element — remove everything, return empty section
            return ""
        result = result[:line_start] + wrapper_close[name]

    return result


def _collect_recent_excerpts(
    storage_root: Path,
    *,
    min_quality: float = 0.30,
    min_length: int = 30,
    max_excerpts: int = 200,
) -> list[dict]:
    """Collect most recent chunks from all sessions, bounded by *max_excerpts*.

    Uses a min-heap keyed by ``ts_end`` so that at most *max_excerpts*
    chunks are kept in memory at any time.  Chunks with quality_score
    below *min_quality* or stripped text shorter than *min_length*
    characters are filtered out.
    For backward compatibility with old chunks.jsonl files that lack the
    quality_score field, the score is computed on the fly.
    """
    sessions_dir = storage_root / "sessions"
    if not sessions_dir.exists():
        return []

    # Min-heap of (ts_end, counter, chunk) — counter breaks ties to avoid
    # comparing dicts.
    heap: list[tuple[float, int, dict]] = []
    counter = 0

    for session_dir in sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue
        chunks_path = session_dir / "chunks.jsonl"
        if not chunks_path.exists():
            continue
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                text = chunk.get("text", "")
                if len(text.strip()) < min_length:
                    continue
                quality = chunk.get(
                    "quality_score", _quality_score(text)
                )
                if quality < min_quality:
                    continue

                ts_end = chunk.get("ts_end", 0)
                if len(heap) < max_excerpts:
                    heapq.heappush(heap, (ts_end, counter, chunk))
                elif ts_end > heap[0][0]:
                    heapq.heapreplace(heap, (ts_end, counter, chunk))
                counter += 1

    # Extract chunks and sort by ts_end descending (most recent first)
    result = [entry[2] for entry in heap]
    result.sort(key=lambda c: c.get("ts_end", 0), reverse=True)
    return result


def _apply_budget(sections: dict[str, str], budget: int) -> str:
    """Apply token budget, truncating sections in reverse priority order.

    Priority order (fill first):
    1. PROJECT_STATE (never truncated)
    2. DECISIONS
    3. CONSTRAINTS (never truncated)
    4. ACTIVE_TASKS
    5. RECENT_CONTEXT_EXCERPTS
    6. INSTRUCTIONS

    Truncation order (remove from end first):
    RECENT_CONTEXT_EXCERPTS → ACTIVE_TASKS → DECISIONS
    PROJECT_STATE and CONSTRAINTS are never truncated.
    """
    envelope_open = '<MEMORY_BANK_CONTEXT version="1.0">\n'
    envelope_close = "\n</MEMORY_BANK_CONTEXT>"

    # Priority order for inclusion
    section_order = [
        "PROJECT_STATE",
        "DECISIONS",
        "CONSTRAINTS",
        "ACTIVE_TASKS",
        "RECENT_CONTEXT_EXCERPTS",
        "INSTRUCTIONS",
    ]

    # Start with envelope overhead
    envelope_tokens = estimate_tokens(envelope_open + envelope_close)
    remaining_budget = budget - envelope_tokens

    # Calculate tokens for each section
    section_tokens = {}
    for name in section_order:
        content = sections.get(name, "")
        section_tokens[name] = estimate_tokens(content)

    total_needed = sum(section_tokens.values())

    if total_needed <= remaining_budget:
        # Everything fits
        parts = [envelope_open]
        for name in section_order:
            parts.append(sections.get(name, ""))
        parts.append(envelope_close)
        return "\n".join(parts)

    # Need to truncate — reverse priority order
    # Never truncate: PROJECT_STATE, CONSTRAINTS
    # Truncation order: RECENT_CONTEXT_EXCERPTS, ACTIVE_TASKS, DECISIONS
    truncatable = ["RECENT_CONTEXT_EXCERPTS", "ACTIVE_TASKS", "DECISIONS"]

    # Calculate non-truncatable budget
    protected = sum(section_tokens[n] for n in section_order if n not in truncatable)
    available_for_truncatable = remaining_budget - protected

    if available_for_truncatable < 0:
        # Even protected sections exceed budget
        sys.stderr.write(
            f"Warning: Token budget ({budget}) too small for PROJECT_STATE. Output truncated.\n"
        )
        available_for_truncatable = 0

    # Allocate budget to truncatable sections in priority order
    allocated = {}
    budget_left = available_for_truncatable
    truncated = False
    for name in ["DECISIONS", "ACTIVE_TASKS", "RECENT_CONTEXT_EXCERPTS"]:
        needed = section_tokens.get(name, 0)
        if needed <= budget_left:
            allocated[name] = sections.get(name, "")
            budget_left -= needed
        elif budget_left > 0:
            # Partially include — truncate by removing elements from the end
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

    # Build final XML
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
