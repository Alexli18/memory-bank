"""Token budget allocation for context packs.

Extracted from pack.py — provides format-independent budget logic.
Sections are abstract (name + content string + priority); the budgeter
allocates tokens and truncates without knowing the output format.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass


def estimate_tokens(text: str) -> int:
    """Estimate token count: chars/4 with 10% safety margin (FR-013)."""
    return int(len(text) / 4 * 1.1)


@dataclass
class Section:
    """A named content section with priority and budget metadata."""

    name: str
    content: str
    priority: int  # lower = higher priority (filled first)
    is_protected: bool  # never truncated if True

    @property
    def token_count(self) -> int:
        return estimate_tokens(self.content)


def truncate_elements(content: str, close_tag: str, budget: int) -> str:
    """Remove trailing XML/text elements until *content* fits within *budget* tokens.

    Elements are identified by their *close_tag* (e.g. ``</EXCERPT>``).
    Removes one element at a time from the end.  Returns empty string if
    nothing can fit.
    """
    result = content
    while estimate_tokens(result) > budget:
        idx = result.rfind(close_tag)
        if idx < 0:
            return ""
        # Find the start of this element's line
        line_start = result.rfind("\n", 0, idx)
        if line_start < 0:
            return ""
        # Find the section's closing wrapper tag (last line)
        after_close = idx + len(close_tag)
        rest = result[after_close:]
        result = result[:line_start] + rest
    return result


def apply_budget(sections: list[Section], budget: int) -> list[Section]:
    """Allocate token budget across *sections*, truncating as needed.

    Protected sections are never truncated.  Non-protected sections are
    truncated in reverse priority order (highest priority number first)
    when the total exceeds *budget*.

    Returns a new list of Sections with possibly shortened content.
    """
    total = sum(s.token_count for s in sections)
    if total <= budget:
        return list(sections)

    protected_cost = sum(s.token_count for s in sections if s.is_protected)
    available = budget - protected_cost

    if available < 0:
        sys.stderr.write(
            f"Warning: Token budget ({budget}) too small for protected sections. Output truncated.\n"
        )
        available = 0

    # Non-protected sections sorted by priority (lowest number = highest priority = filled first)
    truncatable = sorted(
        [s for s in sections if not s.is_protected],
        key=lambda s: s.priority,
    )

    allocated: dict[str, str] = {}
    budget_left = available
    truncated = False

    for s in truncatable:
        needed = s.token_count
        if needed <= budget_left:
            allocated[s.name] = s.content
            budget_left -= needed
        elif budget_left > 0:
            # Partially fit — keep what we can
            # Rough char limit from remaining token budget
            char_limit = int(budget_left * 4 / 1.1)
            allocated[s.name] = s.content[:char_limit]
            budget_left = 0
            truncated = True
        else:
            allocated[s.name] = ""
            truncated = True

    if truncated:
        sys.stderr.write(
            "Warning: Budget too small for full context. Some sections were truncated.\n"
        )

    result: list[Section] = []
    for s in sections:
        if s.is_protected:
            result.append(s)
        else:
            result.append(Section(
                name=s.name,
                content=allocated.get(s.name, ""),
                priority=s.priority,
                is_protected=False,
            ))
    return result
