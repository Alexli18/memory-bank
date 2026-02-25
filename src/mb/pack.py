"""Context pack builder — thin orchestrator over retriever, budgeter, renderer.

Orchestration flow: retriever → budgeter → renderer.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any

from mb.budgeter import estimate_tokens
from mb.chunker import chunk_all_sessions
from mb.models import Chunk, PackFormat, ProjectState
from mb.ollama_client import client_from_config
from mb.renderers import XmlRenderer, get_renderer
from mb.retriever import RecencyRetriever
from mb.state import _state_is_stale, generate_state, load_state
from mb.store import NdjsonStorage

if TYPE_CHECKING:
    from mb.retriever import ContextualRetriever


class _FailureRetrieverAdapter:
    """Adapts ContextualRetriever.retrieve_around_failure to Retriever protocol."""

    def __init__(self, ctx_retriever: ContextualRetriever, session_id: str) -> None:
        self._ctx_retriever = ctx_retriever
        self._session_id = session_id

    def retrieve(self, storage: NdjsonStorage) -> list[Chunk]:
        return self._ctx_retriever.retrieve_around_failure(storage, self._session_id)


def build_pack(
    budget: int,
    storage: NdjsonStorage,
    fmt: PackFormat = PackFormat.XML,
    retriever: Any | None = None,
    mode: str = "auto",
    lightweight: bool = False,
    refresh: bool = False,
) -> str:
    """Build a context pack within the given token budget.

    Steps:
    1. Resolve pack mode → budget profile
    2. Chunk all sessions (incremental, skips already processed)
    3. Load/generate ProjectState (Ollama only when refresh=True)
    4. Select retriever based on mode (debug uses ContextualRetriever)
    5. Retrieve recent excerpts
    6. Render via the chosen format renderer
    7. Apply token budget with mode-specific section allocation

    Args:
        budget: Maximum token budget.
        storage: NdjsonStorage instance.
        fmt: Output format (default: XML).
        retriever: Retriever instance. If None, selected by mode.
        mode: Pack mode (auto/debug/build/explore). Default: auto.
        lightweight: If True, skip Ollama-dependent steps (chunking,
            state regeneration). Uses cached state and pre-existing
            chunks only. Suitable for SessionStart hook.
        refresh: If True, force state regeneration via Ollama.
            Default (False) uses cached state without Ollama calls.

    Returns:
        Formatted context pack string.
    """
    from mb.pack_modes import PackMode, infer_mode, load_profile

    config = storage.read_config()

    if lightweight:
        # Lightweight path: no Ollama calls, no re-chunking, no state regeneration
        pack_mode = PackMode(mode)
        if pack_mode == PackMode.AUTO:
            pack_mode = infer_mode(storage)
        profile = load_profile(config, pack_mode)

        state = load_state(storage)
        if state is None:
            import time as _time
            state = ProjectState(
                summary="",
                decisions=[],
                constraints=[],
                tasks=[],
                updated_at=_time.time(),
                source_sessions=[],
            )

        if retriever is None:
            from mb.decay import get_decay_config
            half_life_days, enabled = get_decay_config(config)
            retriever = RecencyRetriever(
                half_life_days=half_life_days if enabled else 0.0,
            )
    else:
        # Resolve mode to profile (auto → infer from latest session)
        pack_mode = PackMode(mode)
        if pack_mode == PackMode.AUTO:
            pack_mode = infer_mode(storage)
        profile = load_profile(config, pack_mode)

        # Incremental chunking (skips already processed sessions)
        chunk_all_sessions(storage)

        state = load_state(storage)

        if refresh:
            # Explicit refresh: create Ollama client and regenerate state
            client = client_from_config(config)
            try:
                state = generate_state(storage, client)
            except Exception as exc:
                sys.stderr.write(
                    f"Warning: State generation failed ({exc}), using cached state.\n"
                )
        elif _state_is_stale(storage):
            sys.stderr.write(
                "Note: Project state is outdated. Run `mb pack --refresh` to update.\n"
            )

        if state is None:
            import time as _time

            state = ProjectState(
                summary="",
                decisions=[],
                constraints=[],
                tasks=[],
                updated_at=_time.time(),
                source_sessions=[],
            )

    # Retrieve excerpts — mode determines retriever when none provided
    if retriever is None:
        if pack_mode == PackMode.DEBUG:
            from mb.pack_modes import find_latest_error_session
            from mb.retriever import ContextualRetriever

            error_sid = find_latest_error_session(storage)
            if error_sid is not None:
                retriever = _FailureRetrieverAdapter(
                    ContextualRetriever(), error_sid,
                )

        if retriever is None:
            from mb.decay import get_decay_config

            half_life_days, enabled = get_decay_config(config)
            retriever = RecencyRetriever(
                half_life_days=half_life_days if enabled else 0.0,
            )
    excerpts = retriever.retrieve(storage)

    # Load artifact data (backward compatible: no artifacts → None)
    active_items = _load_active_items(storage) or None
    plans = _load_recent_plans(storage) or None

    # Render with budget enforcement
    renderer = get_renderer(fmt)

    if fmt == PackFormat.XML:
        # XML uses the legacy budget truncation approach via _apply_budget
        xml_renderer = XmlRenderer()
        sections_dict = xml_renderer._build_sections(
            state, excerpts, active_items=active_items, plans=plans,
        )
        return _apply_budget(sections_dict, budget, profile=profile)
    else:
        # For JSON/MD: render full, then truncate excerpts if over budget
        output = renderer.render(
            state, excerpts, active_items=active_items, plans=plans,
        )
        while estimate_tokens(output) > budget and excerpts:
            excerpts = excerpts[:-1]
            output = renderer.render(
                state, excerpts, active_items=active_items, plans=plans,
            )
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
    close_tags: dict[str, list[str]] = {
        "RECENT_CONTEXT_EXCERPTS": ["</EXCERPT>"],
        "ACTIVE_TASKS": ["/>", "</task>", "</todo>"],
        "PLANS": ["</plan>"],
        "DECISIONS": ["</DECISION>"],
    }
    tags = close_tags.get(name)
    if tags is None:
        return ""

    wrapper_close = {
        "RECENT_CONTEXT_EXCERPTS": "\n  </RECENT_CONTEXT_EXCERPTS>",
        "ACTIVE_TASKS": "\n  </ACTIVE_TASKS>",
        "PLANS": "\n  </PLANS>",
        "DECISIONS": "\n  </DECISIONS>",
    }

    result = content
    while estimate_tokens(result) > token_budget:
        # Find the last occurrence of any close tag
        best_idx = -1
        for tag in tags:
            idx = result.rfind(tag)
            if idx > best_idx:
                best_idx = idx
        if best_idx < 0:
            return ""
        line_start = result.rfind("\n    <", 0, best_idx)
        if line_start < 0:
            return ""
        result = result[:line_start] + wrapper_close[name]

    return result


def _apply_budget(
    sections: dict[str, str],
    budget: int,
    profile: Any | None = None,
) -> str:
    """Apply token budget to XML sections, truncating in reverse priority order.

    When *profile* (a ``BudgetProfile``) is provided, per-section caps are
    derived from profile percentages applied to the available budget
    (total minus protected INSTRUCTIONS+CONSTRAINTS).  Otherwise falls back
    to the legacy 15% caps for artifact sections.
    """
    from mb.budgeter import MAX_SHARE_ACTIVE_TASKS, MAX_SHARE_PLANS

    envelope_open = '<MEMORY_BANK_CONTEXT version="1.0">\n'
    envelope_close = "\n</MEMORY_BANK_CONTEXT>"

    section_order = [
        "PROJECT_STATE",
        "DECISIONS",
        "CONSTRAINTS",
        "ACTIVE_TASKS",
        "PLANS",
        "RECENT_CONTEXT_EXCERPTS",
        "INSTRUCTIONS",
    ]

    envelope_tokens = estimate_tokens(envelope_open + envelope_close)
    remaining_budget = budget - envelope_tokens

    section_tokens: dict[str, int] = {}
    for name in section_order:
        content = sections.get(name, "")
        section_tokens[name] = estimate_tokens(content)

    total_needed = sum(section_tokens.values())

    if total_needed <= remaining_budget:
        parts = [envelope_open]
        for name in section_order:
            content = sections.get(name, "")
            if content:
                parts.append(content)
        parts.append(envelope_close)
        return "\n".join(parts)

    # FR-013: INSTRUCTIONS, CONSTRAINTS, and PROJECT_STATE are always protected
    always_protected = {"INSTRUCTIONS", "CONSTRAINTS", "PROJECT_STATE"}

    if profile is not None:
        # Mode-specific allocation: PROJECT_STATE is protected (never truncated)
        truncatable = [
            "DECISIONS", "ACTIVE_TASKS",
            "PLANS", "RECENT_CONTEXT_EXCERPTS",
        ]
        protected = sum(
            section_tokens[n] for n in section_order if n in always_protected
        )
        available_for_truncatable = remaining_budget - protected

        if available_for_truncatable < 0:
            sys.stderr.write(
                f"Warning: Token budget ({budget}) too small for protected sections. Output truncated.\n"
            )
            available_for_truncatable = 0

        # Derive per-section caps from profile percentages (redistribute
        # project_state share among remaining sections)
        non_state_total = (
            profile.decisions + profile.active_tasks
            + profile.plans + profile.recent_context
        )
        scale = 1.0 / non_state_total if non_state_total > 0 else 1.0
        section_caps: dict[str, int] = {
            "DECISIONS": int(available_for_truncatable * profile.decisions * scale),
            "ACTIVE_TASKS": int(available_for_truncatable * profile.active_tasks * scale),
            "PLANS": int(available_for_truncatable * profile.plans * scale),
            "RECENT_CONTEXT_EXCERPTS": int(available_for_truncatable * profile.recent_context * scale),
        }
    else:
        # Legacy behavior: PROJECT_STATE is protected, hardcoded artifact caps
        truncatable = ["RECENT_CONTEXT_EXCERPTS", "PLANS", "ACTIVE_TASKS", "DECISIONS"]
        protected = sum(
            section_tokens[n] for n in section_order if n not in truncatable
        )
        available_for_truncatable = remaining_budget - protected

        if available_for_truncatable < 0:
            sys.stderr.write(
                f"Warning: Token budget ({budget}) too small for PROJECT_STATE. Output truncated.\n"
            )
            available_for_truncatable = 0

        section_caps = {
            "ACTIVE_TASKS": int(budget * MAX_SHARE_ACTIVE_TASKS),
            "PLANS": int(budget * MAX_SHARE_PLANS),
        }

    allocated: dict[str, str] = {}
    budget_left = available_for_truncatable
    truncated = False
    # Process truncatable sections in priority order
    for name in truncatable:
        needed = section_tokens.get(name, 0)
        if needed == 0:
            allocated[name] = ""
            continue
        cap = section_caps.get(name, budget_left)
        effective_limit = min(budget_left, cap)
        if needed <= effective_limit:
            allocated[name] = sections.get(name, "")
            budget_left -= needed
        elif effective_limit > 0:
            allocated[name] = _truncate_section(
                name, sections.get(name, ""), effective_limit,
            )
            budget_left -= estimate_tokens(allocated[name])
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
            content = sections.get(name, "")
            if content:
                parts.append(content)
    parts.append(envelope_close)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Artifact data loaders (used by build_pack for context pack artifact sections)
# ---------------------------------------------------------------------------


def _load_active_items(
    storage: NdjsonStorage, max_sessions: int = 5
) -> list[dict[str, Any]]:
    """Load active (pending/in_progress) todo and task items from artifacts.

    Reads raw todo/task JSON from ``.memory-bank/artifacts/``, filters to
    pending/in_progress only, sorts by priority (high first), limits to
    *max_sessions* most recent sessions by mtime.
    """
    artifacts_dir = storage.artifacts_dir
    if not artifacts_dir.exists():
        return []

    items: list[dict[str, Any]] = []

    # Load todos
    todos_dir = artifacts_dir / "todos"
    if todos_dir.exists():
        todo_files = sorted(
            todos_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        session_count = 0
        for f in todo_files:
            if session_count >= max_sessions:
                break
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            session_id = f.stem
            raw_items = data if isinstance(data, list) else data.get("items", [])
            has_active = False
            for item in raw_items:
                status = item.get("status", "pending")
                if status in ("pending", "in_progress"):
                    items.append({
                        "type": "todo",
                        "session_id": session_id,
                        "status": status,
                        "priority": item.get("priority", "medium"),
                        "text": item.get("content", ""),
                    })
                    has_active = True
            if has_active:
                session_count += 1

    # Load tasks
    tasks_dir = artifacts_dir / "tasks"
    if tasks_dir.exists():
        task_session_dirs = sorted(
            [d for d in tasks_dir.iterdir() if d.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        session_count = 0
        for session_dir in task_session_dirs:
            if session_count >= max_sessions:
                break
            session_id = session_dir.name
            has_active = False
            for tf in sorted(session_dir.glob("*.json")):
                try:
                    task_data = json.loads(tf.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                status = task_data.get("status", "pending")
                if status in ("pending", "in_progress"):
                    items.append({
                        "type": "task",
                        "session_id": session_id,
                        "id": task_data.get("id", ""),
                        "status": status,
                        "priority": task_data.get("priority", "medium"),
                        "text": task_data.get("subject", "") or task_data.get("description", ""),
                    })
                    has_active = True
            if has_active:
                session_count += 1

    # Sort by priority: high first
    priority_order = {"high": 0, "medium": 1, "low": 2}
    items.sort(key=lambda x: priority_order.get(x.get("priority", "medium"), 1))

    return items


def _load_recent_plans(
    storage: NdjsonStorage, max_plans: int = 3
) -> list[dict[str, Any]]:
    """Load most recent plan .md files from artifacts.

    Reads plan Markdown files from ``.memory-bank/artifacts/plans/``,
    sorts by mtime descending, returns top *max_plans* with slug and content.
    """
    plans_dir = storage.artifacts_dir / "plans"
    if not plans_dir.exists():
        return []

    plan_files = sorted(
        plans_dir.glob("*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:max_plans]

    plans: list[dict[str, Any]] = []
    for pf in plan_files:
        try:
            content = pf.read_text(encoding="utf-8")
        except OSError:
            continue
        plans.append({
            "slug": pf.stem,
            "text": content,
        })

    return plans
