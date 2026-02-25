"""Tests for mb.pack_modes — budget profiles, mode enum, and mappings."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from mb.graph import EpisodeType
from mb.pack_modes import (
    DEFAULT_PROFILES,
    EPISODE_TO_MODE,
    BudgetProfile,
    PackMode,
    find_latest_error_session,
    infer_mode,
    load_profile,
    resolve_profile,
)
from mb.store import NdjsonStorage


# ---------------------------------------------------------------------------
# T004: Foundational tests
# ---------------------------------------------------------------------------


class TestPackModeEnum:
    """PackMode enum values match expected strings."""

    def test_values(self) -> None:
        assert PackMode.AUTO.value == "auto"
        assert PackMode.DEBUG.value == "debug"
        assert PackMode.BUILD.value == "build"
        assert PackMode.EXPLORE.value == "explore"

    def test_four_members(self) -> None:
        assert len(PackMode) == 4


class TestBudgetProfileNormalization:
    """BudgetProfile.normalized() ensures values sum to 1.0."""

    def test_already_normalized(self) -> None:
        p = BudgetProfile(0.2, 0.2, 0.2, 0.2, 0.2)
        n = p.normalized()
        assert n is p  # same object — no copy needed

    def test_not_summing_to_one(self) -> None:
        p = BudgetProfile(0.1, 0.1, 0.1, 0.1, 0.1)  # sum = 0.5
        n = p.normalized()
        total = (
            n.project_state + n.decisions + n.active_tasks + n.plans + n.recent_context
        )
        assert abs(total - 1.0) < 1e-9

    def test_all_zeros_returns_auto_default(self) -> None:
        p = BudgetProfile(0.0, 0.0, 0.0, 0.0, 0.0)
        n = p.normalized()
        assert n == DEFAULT_PROFILES[PackMode.AUTO]


class TestBudgetProfileSerialization:
    """from_dict / to_dict round-trip."""

    def test_round_trip(self) -> None:
        original = BudgetProfile(0.10, 0.20, 0.30, 0.15, 0.25)
        d = original.to_dict()
        restored = BudgetProfile.from_dict(d)
        assert restored == original

    def test_from_dict_missing_keys_default_zero(self) -> None:
        p = BudgetProfile.from_dict({"project_state": 0.5})
        assert p.project_state == 0.5
        assert p.decisions == 0.0

    def test_to_dict_keys(self) -> None:
        p = BudgetProfile(0.1, 0.2, 0.3, 0.15, 0.25)
        d = p.to_dict()
        assert set(d.keys()) == {
            "project_state",
            "decisions",
            "active_tasks",
            "plans",
            "recent_context",
        }


class TestDefaultProfiles:
    """DEFAULT_PROFILES covers all four modes with correct values."""

    def test_all_modes_present(self) -> None:
        for mode in PackMode:
            assert mode in DEFAULT_PROFILES

    def test_auto_profile(self) -> None:
        p = DEFAULT_PROFILES[PackMode.AUTO]
        assert p.project_state == 0.15
        assert p.decisions == 0.15
        assert p.active_tasks == 0.15
        assert p.plans == 0.15
        assert p.recent_context == 0.40

    def test_debug_profile(self) -> None:
        p = DEFAULT_PROFILES[PackMode.DEBUG]
        assert p.project_state == 0.10
        assert p.recent_context == 0.75

    def test_build_profile(self) -> None:
        p = DEFAULT_PROFILES[PackMode.BUILD]
        assert p.project_state == 0.15
        assert p.decisions == 0.20
        assert p.active_tasks == 0.20
        assert p.plans == 0.20
        assert p.recent_context == 0.25

    def test_explore_profile(self) -> None:
        p = DEFAULT_PROFILES[PackMode.EXPLORE]
        assert p.project_state == 0.25
        assert p.decisions == 0.15
        assert p.active_tasks == 0.05
        assert p.plans == 0.15
        assert p.recent_context == 0.40

    def test_profiles_sum_to_one(self) -> None:
        for mode, profile in DEFAULT_PROFILES.items():
            total = (
                profile.project_state
                + profile.decisions
                + profile.active_tasks
                + profile.plans
                + profile.recent_context
            )
            assert abs(total - 1.0) < 1e-9, f"{mode} profile sums to {total}"


class TestEpisodeToMode:
    """EPISODE_TO_MODE covers all 9 EpisodeType values."""

    def test_all_episode_types_covered(self) -> None:
        for et in EpisodeType:
            assert et.value in EPISODE_TO_MODE, f"EpisodeType.{et.name} not mapped"

    def test_debug_maps_to_debug(self) -> None:
        assert EPISODE_TO_MODE["debug"] == PackMode.DEBUG

    def test_build_types_map_to_build(self) -> None:
        for key in ("build", "refactor", "config", "test", "deploy"):
            assert EPISODE_TO_MODE[key] == PackMode.BUILD

    def test_explore_types_map_to_explore(self) -> None:
        for key in ("explore", "docs", "review"):
            assert EPISODE_TO_MODE[key] == PackMode.EXPLORE


class TestResolveProfile:
    """resolve_profile returns the correct profile for each mode."""

    def test_returns_matching_profile(self) -> None:
        for mode in PackMode:
            assert resolve_profile(mode) == DEFAULT_PROFILES[mode]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_storage(tmp_path: Path) -> NdjsonStorage:
    """Create a minimal NdjsonStorage for tests."""
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "sessions").mkdir(exist_ok=True)
    (tmp_path / "config.json").write_text('{"version": "1.0"}', encoding="utf-8")
    return NdjsonStorage(tmp_path)


def _write_session(
    root: Path,
    session_id: str,
    *,
    command: list[str] | None = None,
    exit_code: int = 0,
    started_at: float = 1.0,
    chunks: list[dict[str, object]] | None = None,
) -> None:
    """Write a minimal session (meta + optional chunks) to storage."""
    session_dir = root / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "session_id": session_id,
        "command": command or ["bash"],
        "cwd": str(root),
        "started_at": started_at,
        "exit_code": exit_code,
    }
    (session_dir / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    if chunks:
        with (session_dir / "chunks.jsonl").open("w") as f:
            for c in chunks:
                f.write(json.dumps(c) + "\n")


# ---------------------------------------------------------------------------
# T007: Debug mode — find_latest_error_session + retriever selection
# ---------------------------------------------------------------------------


class TestFindLatestErrorSession:
    """find_latest_error_session() returns the right session or None."""

    def test_returns_error_session(self, tmp_path: Path) -> None:
        root = tmp_path / ".mb"
        storage = _make_storage(root)
        _write_session(root, "s1", exit_code=0, started_at=1.0)
        _write_session(root, "s2", exit_code=1, started_at=2.0)

        result = find_latest_error_session(storage)
        assert result == "s2"

    def test_returns_none_when_no_errors(self, tmp_path: Path) -> None:
        root = tmp_path / ".mb"
        storage = _make_storage(root)
        _write_session(root, "s1", exit_code=0, started_at=1.0)
        _write_session(root, "s2", exit_code=0, started_at=2.0)

        result = find_latest_error_session(storage)
        assert result is None

    def test_returns_none_when_no_sessions(self, tmp_path: Path) -> None:
        root = tmp_path / ".mb"
        storage = _make_storage(root)

        result = find_latest_error_session(storage)
        assert result is None

    def test_returns_most_recent_error(self, tmp_path: Path) -> None:
        root = tmp_path / ".mb"
        storage = _make_storage(root)
        _write_session(root, "s1", exit_code=1, started_at=1.0)
        _write_session(root, "s2", exit_code=0, started_at=2.0)
        _write_session(root, "s3", exit_code=1, started_at=3.0)

        result = find_latest_error_session(storage)
        # list_sessions sorts by started_at descending → s3 is first
        assert result == "s3"

    def test_detects_error_from_chunk_content(self, tmp_path: Path) -> None:
        root = tmp_path / ".mb"
        storage = _make_storage(root)
        _write_session(
            root,
            "s1",
            exit_code=0,
            started_at=1.0,
            chunks=[{
                "chunk_id": "s1-0",
                "session_id": "s1",
                "text": "Traceback (most recent call last):\n  File ...",
                "ts_start": 0.0,
                "ts_end": 1.0,
                "quality_score": 0.8,
            }],
        )

        result = find_latest_error_session(storage)
        assert result == "s1"


class TestDebugModeRetrieverSelection:
    """Debug mode selects ContextualRetriever or falls back to RecencyRetriever."""

    def test_debug_uses_contextual_retriever_when_error_found(self) -> None:
        mock_storage = MagicMock(spec=NdjsonStorage)
        mock_storage.read_config.return_value = {"version": "1.0"}
        mock_storage.is_stale.return_value = False

        with (
            patch("mb.pack_modes.find_latest_error_session", return_value="err-session") as mock_find,
            patch("mb.pack.chunk_all_sessions"),
            patch("mb.pack.load_state", return_value=MagicMock()),
            patch("mb.pack._load_active_items", return_value=None),
            patch("mb.pack._load_recent_plans", return_value=None),
            patch("mb.pack.get_renderer") as mock_renderer,
            patch("mb.retriever.ContextualRetriever.retrieve_around_failure", return_value=[]) as mock_failure,
        ):
            mock_renderer.return_value.render.return_value = "pack"
            from mb.pack import build_pack

            build_pack(6000, mock_storage, mode="debug")
            mock_find.assert_called_once_with(mock_storage)
            mock_failure.assert_called_once_with(mock_storage, "err-session")

    def test_debug_falls_back_to_recency_when_no_errors(self) -> None:
        mock_storage = MagicMock(spec=NdjsonStorage)
        mock_storage.read_config.return_value = {"version": "1.0"}
        mock_storage.is_stale.return_value = False

        with (
            patch("mb.pack_modes.find_latest_error_session", return_value=None),
            patch("mb.pack.chunk_all_sessions"),
            patch("mb.pack.load_state", return_value=MagicMock()),
            patch("mb.pack._load_active_items", return_value=None),
            patch("mb.pack._load_recent_plans", return_value=None),
            patch("mb.pack.get_renderer") as mock_renderer,
            patch("mb.pack.RecencyRetriever") as mock_recency_cls,
        ):
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = []
            mock_recency_cls.return_value = mock_retriever
            mock_renderer.return_value.render.return_value = "pack"

            from mb.pack import build_pack

            build_pack(6000, mock_storage, mode="debug")
            mock_recency_cls.assert_called_once()


# ---------------------------------------------------------------------------
# T008: Build profile verification
# ---------------------------------------------------------------------------


class TestBuildProfileAllocations:
    """Verify build profile matches spec exactly."""

    def test_build_profile_values(self) -> None:
        p = DEFAULT_PROFILES[PackMode.BUILD]
        assert p.project_state == 0.15
        assert p.decisions == 0.20
        assert p.active_tasks == 0.20
        assert p.plans == 0.20
        assert p.recent_context == 0.25

    def test_build_profile_sums_to_one(self) -> None:
        p = DEFAULT_PROFILES[PackMode.BUILD]
        total = p.project_state + p.decisions + p.active_tasks + p.plans + p.recent_context
        assert abs(total - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# T009: Explore profile verification
# ---------------------------------------------------------------------------


class TestExploreProfileAllocations:
    """Verify explore profile matches spec exactly."""

    def test_explore_profile_values(self) -> None:
        p = DEFAULT_PROFILES[PackMode.EXPLORE]
        assert p.project_state == 0.25
        assert p.decisions == 0.15
        assert p.active_tasks == 0.05
        assert p.plans == 0.15
        assert p.recent_context == 0.40

    def test_explore_profile_sums_to_one(self) -> None:
        p = DEFAULT_PROFILES[PackMode.EXPLORE]
        total = p.project_state + p.decisions + p.active_tasks + p.plans + p.recent_context
        assert abs(total - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# T012: Auto mode inference tests
# ---------------------------------------------------------------------------


class TestInferMode:
    """infer_mode() returns correct mode based on latest session's episode type."""

    def test_returns_debug_when_latest_is_debug(self, tmp_path: Path) -> None:
        root = tmp_path / ".mb"
        storage = _make_storage(root)
        _write_session(root, "s1", command=["gdb"], started_at=1.0)

        result = infer_mode(storage)
        assert result == PackMode.DEBUG

    def test_returns_build_when_latest_is_build(self, tmp_path: Path) -> None:
        root = tmp_path / ".mb"
        storage = _make_storage(root)
        _write_session(root, "s1", command=["make"], started_at=1.0)

        result = infer_mode(storage)
        assert result == PackMode.BUILD

    def test_returns_build_when_latest_is_test(self, tmp_path: Path) -> None:
        root = tmp_path / ".mb"
        storage = _make_storage(root)
        _write_session(root, "s1", command=["pytest"], started_at=1.0)

        result = infer_mode(storage)
        assert result == PackMode.BUILD

    def test_returns_build_when_latest_is_refactor(self, tmp_path: Path) -> None:
        root = tmp_path / ".mb"
        storage = _make_storage(root)
        # "claude" with refactor-like content
        _write_session(
            root, "s1", command=["claude"], started_at=1.0,
            chunks=[{
                "chunk_id": "s1-0", "session_id": "s1",
                "text": "refactor this module to simplify the code",
                "ts_start": 0.0, "ts_end": 1.0, "quality_score": 0.8,
            }],
        )

        result = infer_mode(storage)
        assert result == PackMode.BUILD

    def test_returns_explore_when_latest_is_explore(self, tmp_path: Path) -> None:
        root = tmp_path / ".mb"
        storage = _make_storage(root)
        _write_session(
            root, "s1", command=["claude"], started_at=1.0,
            chunks=[{
                "chunk_id": "s1-0", "session_id": "s1",
                "text": "how does the architecture of this module work? explain the design",
                "ts_start": 0.0, "ts_end": 1.0, "quality_score": 0.8,
            }],
        )

        result = infer_mode(storage)
        assert result == PackMode.EXPLORE

    def test_returns_explore_when_latest_is_docs(self, tmp_path: Path) -> None:
        root = tmp_path / ".mb"
        storage = _make_storage(root)
        _write_session(
            root, "s1", command=["claude"], started_at=1.0,
            chunks=[{
                "chunk_id": "s1-0", "session_id": "s1",
                "text": "update the README documentation and CHANGELOG",
                "ts_start": 0.0, "ts_end": 1.0, "quality_score": 0.8,
            }],
        )

        result = infer_mode(storage)
        assert result == PackMode.EXPLORE

    def test_returns_auto_when_no_sessions(self, tmp_path: Path) -> None:
        root = tmp_path / ".mb"
        storage = _make_storage(root)

        result = infer_mode(storage)
        assert result == PackMode.AUTO

    def test_uses_most_recent_session(self, tmp_path: Path) -> None:
        root = tmp_path / ".mb"
        storage = _make_storage(root)
        _write_session(root, "s1", command=["make"], started_at=1.0)
        _write_session(root, "s2", command=["gdb"], started_at=2.0)

        result = infer_mode(storage)
        # s2 (gdb → debug) is most recent
        assert result == PackMode.DEBUG


# ---------------------------------------------------------------------------
# T015: Config-based profile loading tests
# ---------------------------------------------------------------------------


class TestLoadProfile:
    """load_profile() merges config overrides over defaults."""

    def test_returns_default_when_no_config(self) -> None:
        config: dict[str, object] = {"version": "1.0"}
        profile = load_profile(config, PackMode.DEBUG)
        assert profile == DEFAULT_PROFILES[PackMode.DEBUG]

    def test_returns_default_when_empty_pack_modes(self) -> None:
        config: dict[str, object] = {"version": "1.0", "pack_modes": {}}
        profile = load_profile(config, PackMode.BUILD)
        assert profile == DEFAULT_PROFILES[PackMode.BUILD]

    def test_partial_override_merges_correctly(self) -> None:
        config: dict[str, object] = {
            "version": "1.0",
            "pack_modes": {
                "debug": {
                    "recent_context": 0.80,
                    "project_state": 0.10,
                },
            },
        }
        profile = load_profile(config, PackMode.DEBUG)
        # Overridden keys
        assert profile.recent_context > 0.75  # was 0.75, now 0.80 (after normalization)
        assert profile.project_state > 0.0
        # All values should still sum to 1.0
        total = (
            profile.project_state + profile.decisions + profile.active_tasks
            + profile.plans + profile.recent_context
        )
        assert abs(total - 1.0) < 1e-9

    def test_full_override_replaces_all(self) -> None:
        config: dict[str, object] = {
            "version": "1.0",
            "pack_modes": {
                "build": {
                    "project_state": 0.10,
                    "decisions": 0.10,
                    "active_tasks": 0.30,
                    "plans": 0.30,
                    "recent_context": 0.20,
                },
            },
        }
        profile = load_profile(config, PackMode.BUILD)
        assert profile.project_state == 0.10
        assert profile.decisions == 0.10
        assert profile.active_tasks == 0.30
        assert profile.plans == 0.30
        assert profile.recent_context == 0.20

    def test_normalization_when_overrides_dont_sum_to_one(self) -> None:
        config: dict[str, object] = {
            "version": "1.0",
            "pack_modes": {
                "explore": {
                    "project_state": 0.50,
                    "decisions": 0.50,
                    "active_tasks": 0.50,
                    "plans": 0.50,
                    "recent_context": 0.50,
                },
            },
        }
        profile = load_profile(config, PackMode.EXPLORE)
        total = (
            profile.project_state + profile.decisions + profile.active_tasks
            + profile.plans + profile.recent_context
        )
        assert abs(total - 1.0) < 1e-9
        # All equal → each should be 0.20
        assert abs(profile.project_state - 0.20) < 1e-9

    def test_missing_pack_modes_key_returns_defaults(self) -> None:
        config: dict[str, object] = {"version": "1.0"}
        for mode in PackMode:
            profile = load_profile(config, mode)
            assert profile == DEFAULT_PROFILES[mode]
