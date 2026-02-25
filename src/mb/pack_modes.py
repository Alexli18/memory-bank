"""Pack mode definitions: budget profiles, mode inference, and config loading.

Provides mode-specific budget allocation for context packs.
Four modes (auto/debug/build/explore) determine how the token budget
is distributed across pack sections.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mb.store import NdjsonStorage


class PackMode(Enum):
    """Context pack generation strategy."""

    AUTO = "auto"
    DEBUG = "debug"
    BUILD = "build"
    EXPLORE = "explore"


@dataclass(frozen=True, slots=True)
class BudgetProfile:
    """Percentage allocation of available budget across pack sections.

    All values are fractions in [0.0, 1.0].  Normalized to sum to 1.0
    on construction via ``normalized()``.
    """

    project_state: float
    decisions: float
    active_tasks: float
    plans: float
    recent_context: float

    def normalized(self) -> BudgetProfile:
        """Return a copy where all values sum to 1.0."""
        total = (
            self.project_state
            + self.decisions
            + self.active_tasks
            + self.plans
            + self.recent_context
        )
        if total <= 0:
            return DEFAULT_PROFILES[PackMode.AUTO]
        if abs(total - 1.0) < 1e-9:
            return self
        return BudgetProfile(
            project_state=self.project_state / total,
            decisions=self.decisions / total,
            active_tasks=self.active_tasks / total,
            plans=self.plans / total,
            recent_context=self.recent_context / total,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BudgetProfile:
        """Create from a flat dict (e.g. config.json section)."""
        return cls(
            project_state=float(data.get("project_state", 0.0)),
            decisions=float(data.get("decisions", 0.0)),
            active_tasks=float(data.get("active_tasks", 0.0)),
            plans=float(data.get("plans", 0.0)),
            recent_context=float(data.get("recent_context", 0.0)),
        )

    def to_dict(self) -> dict[str, float]:
        """Serialize to a flat dict."""
        return {
            "project_state": self.project_state,
            "decisions": self.decisions,
            "active_tasks": self.active_tasks,
            "plans": self.plans,
            "recent_context": self.recent_context,
        }


# ---------------------------------------------------------------------------
# Built-in profiles (FR-002)
# ---------------------------------------------------------------------------

DEFAULT_PROFILES: dict[PackMode, BudgetProfile] = {
    PackMode.AUTO: BudgetProfile(
        project_state=0.15,
        decisions=0.15,
        active_tasks=0.15,
        plans=0.15,
        recent_context=0.40,
    ),
    PackMode.DEBUG: BudgetProfile(
        project_state=0.10,
        decisions=0.05,
        active_tasks=0.05,
        plans=0.05,
        recent_context=0.75,
    ),
    PackMode.BUILD: BudgetProfile(
        project_state=0.15,
        decisions=0.20,
        active_tasks=0.20,
        plans=0.20,
        recent_context=0.25,
    ),
    PackMode.EXPLORE: BudgetProfile(
        project_state=0.25,
        decisions=0.15,
        active_tasks=0.05,
        plans=0.15,
        recent_context=0.40,
    ),
}

# ---------------------------------------------------------------------------
# Episode-to-mode mapping (FR-004)
# ---------------------------------------------------------------------------

# Lazy import: EpisodeType is referenced by string key to avoid circular
# imports at module level.  The mapping uses string values matching
# ``EpisodeType.value`` and is resolved at call time.

EPISODE_TO_MODE: dict[str, PackMode] = {
    "debug": PackMode.DEBUG,
    "build": PackMode.BUILD,
    "refactor": PackMode.BUILD,
    "config": PackMode.BUILD,
    "test": PackMode.BUILD,
    "deploy": PackMode.BUILD,
    "explore": PackMode.EXPLORE,
    "docs": PackMode.EXPLORE,
    "review": PackMode.EXPLORE,
}


def resolve_profile(mode: PackMode) -> BudgetProfile:
    """Return the built-in BudgetProfile for *mode*."""
    return DEFAULT_PROFILES.get(mode, DEFAULT_PROFILES[PackMode.AUTO])


def load_profile(config: dict[str, Any], mode: PackMode) -> BudgetProfile:
    """Return a BudgetProfile for *mode*, merging config overrides over defaults.

    Reads ``config["pack_modes"][mode.value]`` (if present) and merges
    user-specified keys over the built-in default for *mode*.  Missing
    keys keep their default values.  The resulting profile is normalized.
    """
    default = DEFAULT_PROFILES.get(mode, DEFAULT_PROFILES[PackMode.AUTO])
    overrides = config.get("pack_modes", {}).get(mode.value)
    if not overrides:
        return default

    merged = default.to_dict()
    for key in merged:
        if key in overrides:
            merged[key] = float(overrides[key])

    return BudgetProfile.from_dict(merged).normalized()


def infer_mode(storage: NdjsonStorage) -> PackMode:
    """Infer pack mode from the most recent session's episode type.

    Loads the most recent ``SessionMeta`` (``list_sessions()`` returns
    newest-first), classifies its episode via ``SessionGraph``, and maps
    the result through ``EPISODE_TO_MODE``.

    Returns ``PackMode.AUTO`` when no sessions exist or the episode type
    is not in the mapping.
    """
    from mb.graph import SessionGraph

    sessions = storage.list_sessions()
    if not sessions:
        return PackMode.AUTO

    latest = sessions[0]  # already sorted by started_at descending
    chunks = storage.read_chunks(latest.session_id)
    graph = SessionGraph()
    episode = graph.classify_episode(latest, chunks)

    return EPISODE_TO_MODE.get(episode.value, PackMode.AUTO)


def find_latest_error_session(storage: NdjsonStorage) -> str | None:
    """Return the session_id of the most recent session with an error.

    Iterates ``storage.list_sessions()`` (already sorted by started_at
    descending) and returns the first session where
    ``SessionGraph.detect_error()`` is True.  Returns ``None`` when no
    error sessions exist.
    """
    from mb.graph import SessionGraph

    graph = SessionGraph()
    for meta in storage.list_sessions():
        chunks = storage.read_chunks(meta.session_id)
        if graph.detect_error(meta, chunks):
            return meta.session_id
    return None
