"""Tests for mb.cli — Click CLI commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from mb.cli import cli
from mb.models import SearchResult, SessionMeta
from mb.store import NdjsonStorage


@pytest.fixture()
def runner() -> CliRunner:
    """Create a CliRunner for invoking CLI commands."""
    return CliRunner()


# --- mb --version ---


def test_version_outputs_version_string(runner: CliRunner) -> None:
    """mb --version prints the current version."""
    from mb import __version__

    result = runner.invoke(cli, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.output


# --- mb init ---


def test_init_creates_storage(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb init initializes .memory-bank/ in the current directory."""
    monkeypatch.chdir(tmp_path)

    mock_storage = MagicMock(spec=NdjsonStorage)
    with patch("mb.store.NdjsonStorage.init", return_value=(True, mock_storage)):
        result = runner.invoke(cli, ["init"])

    assert result.exit_code == 0
    assert "Initialized Memory Bank" in result.output


def test_init_idempotent(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb init on an already-initialized dir reports 'already initialized'."""
    monkeypatch.chdir(tmp_path)

    mock_storage = MagicMock(spec=NdjsonStorage)
    with patch("mb.store.NdjsonStorage.init", return_value=(False, mock_storage)):
        result = runner.invoke(cli, ["init"])

    assert result.exit_code == 0
    assert "already initialized" in result.output


# --- mb sessions ---


def test_sessions_empty(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb sessions with no recorded sessions prints 'No sessions found.'."""
    monkeypatch.chdir(tmp_path)

    mock_storage = MagicMock(spec=NdjsonStorage)
    mock_storage.list_sessions.return_value = []
    with patch("mb.cli._require_storage", return_value=mock_storage):
        result = runner.invoke(cli, ["sessions"])

    assert result.exit_code == 0
    assert "No sessions found" in result.output


def test_sessions_lists_entries(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb sessions lists session rows when sessions exist."""
    monkeypatch.chdir(tmp_path)

    sessions = [
        SessionMeta.from_dict({
            "session_id": "20260224-120000-abcd",
            "command": ["python", "hello.py"],
            "cwd": str(tmp_path),
            "started_at": 1700000000.0,
            "exit_code": 0,
        }),
    ]
    mock_storage = MagicMock(spec=NdjsonStorage)
    mock_storage.list_sessions.return_value = sessions
    with patch("mb.cli._require_storage", return_value=mock_storage):
        result = runner.invoke(cli, ["sessions"])

    assert result.exit_code == 0
    assert "20260224-120000-abcd" in result.output
    assert "python hello.py" in result.output


def test_sessions_not_initialized(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb sessions without init raises MbError (exit code 1)."""
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(cli, ["sessions"])

    assert result.exit_code == 1
    assert "not initialized" in result.output.lower()


# --- mb delete ---


def test_delete_existing_session(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb delete removes a session by ID."""
    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".memory-bank"
    root.mkdir(exist_ok=True)
    (root / "index").mkdir(exist_ok=True)

    mock_storage = MagicMock(spec=NdjsonStorage)
    mock_storage.delete_session.return_value = True
    mock_storage.root = root
    with patch("mb.cli._require_storage", return_value=mock_storage):
        result = runner.invoke(cli, ["delete", "20260224-120000-abcd"])

    assert result.exit_code == 0
    assert "Deleted session" in result.output


def test_delete_not_found(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb delete with unknown session ID exits with code 1."""
    monkeypatch.chdir(tmp_path)

    mock_storage = MagicMock(spec=NdjsonStorage)
    mock_storage.delete_session.return_value = False
    with patch("mb.cli._require_storage", return_value=mock_storage):
        result = runner.invoke(cli, ["delete", "nonexistent"])

    assert result.exit_code == 1
    assert "not found" in result.output.lower()


# --- mb search ---


def test_search_no_sessions(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb search with no sessions prints a hint and exits 0."""
    monkeypatch.chdir(tmp_path)

    mock_storage = MagicMock(spec=NdjsonStorage)
    mock_storage.list_sessions.return_value = []
    with patch("mb.cli._require_storage", return_value=mock_storage):
        result = runner.invoke(cli, ["search", "hello"])

    assert result.exit_code == 0
    assert "No sessions found" in result.output


def test_search_ollama_not_running(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb search when Ollama is down exits with code 2."""
    monkeypatch.chdir(tmp_path)

    from mb.ollama_client import OllamaNotRunningError

    sessions = [SessionMeta.from_dict({"session_id": "s1", "command": ["bash"], "cwd": "/", "started_at": 1.0, "exit_code": 0})]

    mock_storage = MagicMock(spec=NdjsonStorage)
    mock_storage.list_sessions.return_value = sessions
    mock_storage.read_config.return_value = {"ollama": {}}
    with (
        patch("mb.cli._require_storage", return_value=mock_storage),
        patch("mb.ollama_client.client_from_config", return_value=MagicMock()),
        patch("mb.search.semantic_search", side_effect=OllamaNotRunningError("down")),
    ):
        result = runner.invoke(cli, ["search", "hello"])

    assert result.exit_code == 2
    assert "Cannot connect to Ollama" in result.output


def test_search_returns_results(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb search displays results from semantic_search."""
    monkeypatch.chdir(tmp_path)

    sessions = [SessionMeta.from_dict({"session_id": "s1", "command": ["bash"], "cwd": "/", "started_at": 1.0, "exit_code": 0})]
    results = [
        SearchResult.from_dict({
            "score": 0.95,
            "session_id": "s1",
            "ts_start": 65.0,
            "ts_end": 130.0,
            "text": "some matching text",
        }),
    ]

    mock_storage = MagicMock(spec=NdjsonStorage)
    mock_storage.list_sessions.return_value = sessions
    mock_storage.read_config.return_value = {"ollama": {}}
    with (
        patch("mb.cli._require_storage", return_value=mock_storage),
        patch("mb.ollama_client.client_from_config", return_value=MagicMock()),
        patch("mb.search.semantic_search", return_value=results),
    ):
        result = runner.invoke(cli, ["search", "matching"])

    assert result.exit_code == 0
    assert "0.95" in result.output
    assert "01:05" in result.output  # 65 seconds = 01:05
    assert "some matching text" in result.output


def test_search_rerank_flag_passthrough(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb search --rerank passes rerank=True to semantic_search."""
    monkeypatch.chdir(tmp_path)

    sessions = [SessionMeta.from_dict({"session_id": "s1", "command": ["bash"], "cwd": "/", "started_at": 1.0, "exit_code": 0})]
    results = [
        SearchResult.from_dict({
            "score": 0.90,
            "session_id": "s1",
            "ts_start": 0.0,
            "ts_end": 60.0,
            "text": "reranked result",
        }),
    ]

    mock_storage = MagicMock(spec=NdjsonStorage)
    mock_storage.list_sessions.return_value = sessions
    mock_storage.read_config.return_value = {"ollama": {}}
    with (
        patch("mb.cli._require_storage", return_value=mock_storage),
        patch("mb.ollama_client.client_from_config", return_value=MagicMock()),
        patch("mb.search.semantic_search", return_value=results) as mock_search,
    ):
        result = runner.invoke(cli, ["search", "query", "--rerank"])

    assert result.exit_code == 0
    assert "reranked result" in result.output
    # Verify rerank=True was passed
    call_kwargs = mock_search.call_args[1]
    assert call_kwargs["rerank"] is True


# --- mb pack ---


def test_pack_ollama_not_running(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb pack when Ollama is unavailable exits with code 2."""
    monkeypatch.chdir(tmp_path)

    from mb.ollama_client import OllamaNotRunningError

    mock_storage = MagicMock(spec=NdjsonStorage)
    mock_storage.read_config.return_value = {"ollama": {}}
    with (
        patch("mb.cli._require_storage", return_value=mock_storage),
        patch("mb.pack.build_pack", side_effect=OllamaNotRunningError("down")),
    ):
        result = runner.invoke(cli, ["pack"])

    assert result.exit_code == 2
    assert "Cannot connect to Ollama" in result.output


def test_pack_writes_to_file(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb pack --out writes XML output to the specified file."""
    monkeypatch.chdir(tmp_path)
    out_file = tmp_path / "context.xml"

    mock_storage = MagicMock(spec=NdjsonStorage)
    mock_storage.read_config.return_value = {"ollama": {}}
    with (
        patch("mb.cli._require_storage", return_value=mock_storage),
        patch("mb.pack.build_pack", return_value="<memory-bank>pack content</memory-bank>"),
    ):
        result = runner.invoke(cli, ["pack", "--out", str(out_file)])

    assert result.exit_code == 0
    assert out_file.read_text(encoding="utf-8") == "<memory-bank>pack content</memory-bank>"


# --- mb hooks ---


def test_hooks_install_calls_function(
    runner: CliRunner,
) -> None:
    """mb hooks install delegates to install_hooks."""
    with patch("mb.hooks.install_hooks", return_value=(True, "Hook installed.")) as mock_fn:
        result = runner.invoke(cli, ["hooks", "install"])

    assert result.exit_code == 0
    assert "Hook installed." in result.output
    mock_fn.assert_called_once()


def test_hooks_uninstall_calls_function(
    runner: CliRunner,
) -> None:
    """mb hooks uninstall delegates to uninstall_hooks."""
    with patch("mb.hooks.uninstall_hooks", return_value=(True, "Hook uninstalled.")) as mock_fn:
        result = runner.invoke(cli, ["hooks", "uninstall"])

    assert result.exit_code == 0
    assert "Hook uninstalled." in result.output
    mock_fn.assert_called_once()


def test_hooks_status_both_installed(
    runner: CliRunner,
) -> None:
    """mb hooks status shows both hooks installed."""
    with patch(
        "mb.hooks.hooks_status",
        return_value={
            "stop": {"installed": True, "command": "python -m mb.hook_handler"},
            "session_start": {"installed": True, "command": "python -m mb.session_start_hook"},
        },
    ):
        result = runner.invoke(cli, ["hooks", "status"])

    assert result.exit_code == 0
    assert "Stop hook: Installed" in result.output
    assert "mb.hook_handler" in result.output
    assert "SessionStart hook: Installed" in result.output
    assert "mb.session_start_hook" in result.output


def test_hooks_status_not_installed(
    runner: CliRunner,
) -> None:
    """mb hooks status shows 'Not installed' when no hooks."""
    with patch(
        "mb.hooks.hooks_status",
        return_value={
            "stop": {"installed": False, "command": None},
            "session_start": {"installed": False, "command": None},
        },
    ):
        result = runner.invoke(cli, ["hooks", "status"])

    assert result.exit_code == 0
    assert "Stop hook: Not installed" in result.output
    assert "SessionStart hook: Not installed" in result.output


def test_hooks_status_stop_only(
    runner: CliRunner,
) -> None:
    """mb hooks status shows Stop installed, SessionStart not."""
    with patch(
        "mb.hooks.hooks_status",
        return_value={
            "stop": {"installed": True, "command": "python -m mb.hook_handler"},
            "session_start": {"installed": False, "command": None},
        },
    ):
        result = runner.invoke(cli, ["hooks", "status"])

    assert result.exit_code == 0
    assert "Stop hook: Installed" in result.output
    assert "SessionStart hook: Not installed" in result.output


# --- mb import ---


def test_import_command(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb import calls import_claude_sessions_with_artifacts and shows results."""
    monkeypatch.chdir(tmp_path)
    _create_config(tmp_path)

    mock_result = {
        "imported": 3, "skipped": 1,
        "plans_imported": 0, "todos_imported": 0, "tasks_imported": 0,
        "dry_run_todo_items": 0, "dry_run_task_items": 0,
    }
    with patch("mb.importer.import_claude_sessions_with_artifacts", return_value=mock_result) as mock_fn:
        result = runner.invoke(cli, ["import"])

    assert result.exit_code == 0
    assert "Imported 3 sessions" in result.output
    assert "1 skipped" in result.output
    mock_fn.assert_called_once()


def test_import_dry_run(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb import --dry-run shows what would be imported."""
    monkeypatch.chdir(tmp_path)
    _create_config(tmp_path)

    mock_result = {
        "imported": 2, "skipped": 0,
        "plans_imported": 1, "todos_imported": 0, "tasks_imported": 0,
        "dry_run_todo_items": 0, "dry_run_task_items": 0,
    }
    with patch("mb.importer.import_claude_sessions_with_artifacts", return_value=mock_result) as mock_fn:
        result = runner.invoke(cli, ["import", "--dry-run"])

    assert result.exit_code == 0
    assert "Would import 2 sessions" in result.output
    assert "1 plans" in result.output
    # Verify it was called with an NdjsonStorage instance and dry_run=True
    mock_fn.assert_called_once()
    call_args = mock_fn.call_args
    assert isinstance(call_args[0][0], NdjsonStorage)
    assert call_args[1]["dry_run"] is True


def test_import_no_sessions_found(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb import with no Claude sessions shows message."""
    monkeypatch.chdir(tmp_path)
    _create_config(tmp_path)

    mock_result = {
        "imported": 0, "skipped": 0,
        "plans_imported": 0, "todos_imported": 0, "tasks_imported": 0,
        "dry_run_todo_items": 0, "dry_run_task_items": 0,
    }
    with patch("mb.importer.import_claude_sessions_with_artifacts", return_value=mock_result):
        result = runner.invoke(cli, ["import"])

    assert result.exit_code == 0
    assert "No Claude Code sessions found" in result.output


# --- mb graph ---


def test_graph_empty(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb graph with no sessions prints 'No sessions found.'."""
    monkeypatch.chdir(tmp_path)

    mock_storage = MagicMock(spec=NdjsonStorage)
    mock_storage.list_sessions.return_value = []
    with (
        patch("mb.cli._require_storage", return_value=mock_storage),
        patch("mb.graph.SessionGraph.build_graph", return_value=[]),
    ):
        result = runner.invoke(cli, ["graph"])

    assert result.exit_code == 0
    assert "No sessions found" in result.output


def test_graph_with_sessions(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb graph displays table with episode classification."""
    monkeypatch.chdir(tmp_path)

    from mb.graph import EpisodeType, SessionNode

    meta = SessionMeta.from_dict({
        "session_id": "20260224-161618-0325",
        "command": ["pytest"],
        "cwd": str(tmp_path),
        "started_at": 1700000000.0,
        "exit_code": 1,
    })
    node = SessionNode(
        meta=meta,
        episode_type=EpisodeType.TEST,
        has_error=True,
        error_summary="Exit code 1",
        related_sessions=[],
    )

    mock_storage = MagicMock(spec=NdjsonStorage)
    with (
        patch("mb.cli._require_storage", return_value=mock_storage),
        patch("mb.graph.SessionGraph.build_graph", return_value=[node]),
    ):
        result = runner.invoke(cli, ["graph"])

    assert result.exit_code == 0
    assert "20260224-161618-0325" in result.output
    assert "test" in result.output
    assert "YES" in result.output
    assert "pytest" in result.output


def test_graph_json(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb graph --json outputs JSON list."""
    import json

    monkeypatch.chdir(tmp_path)

    from mb.graph import EpisodeType, SessionNode

    meta = SessionMeta.from_dict({
        "session_id": "20260224-161613-dcb6",
        "command": ["python", "-c", "print(42)"],
        "cwd": str(tmp_path),
        "started_at": 1700000000.0,
        "exit_code": 0,
    })
    node = SessionNode(
        meta=meta,
        episode_type=EpisodeType.BUILD,
        has_error=False,
        error_summary=None,
        related_sessions=[],
    )

    mock_storage = MagicMock(spec=NdjsonStorage)
    with (
        patch("mb.cli._require_storage", return_value=mock_storage),
        patch("mb.graph.SessionGraph.build_graph", return_value=[node]),
    ):
        result = runner.invoke(cli, ["graph", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data) == 1
    assert data[0]["session_id"] == "20260224-161613-dcb6"
    assert data[0]["episode_type"] == "build"
    assert data[0]["has_error"] is False


# --- mb pack --retriever ---


def test_pack_retriever_episode(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb pack --retriever episode --episode test passes episode retriever to build_pack."""
    monkeypatch.chdir(tmp_path)

    mock_storage = MagicMock(spec=NdjsonStorage)
    mock_storage.read_config.return_value = {"ollama": {}}
    with (
        patch("mb.cli._require_storage", return_value=mock_storage),
        patch("mb.pack.build_pack", return_value="<pack>test</pack>") as mock_build,
    ):
        result = runner.invoke(cli, ["pack", "--retriever", "episode", "--episode", "test"])

    assert result.exit_code == 0
    assert "<pack>test</pack>" in result.output
    # Verify retriever was passed
    call_kwargs = mock_build.call_args
    retriever_arg = call_kwargs[1].get("retriever") or call_kwargs[0][3] if len(call_kwargs[0]) > 3 else call_kwargs[1].get("retriever")
    assert retriever_arg is not None


@pytest.mark.parametrize("episode_type", ["explore", "config", "docs", "review"])
def test_pack_accepts_new_episode_types(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, episode_type: str
) -> None:
    """mb pack --retriever episode accepts the new episode types."""
    monkeypatch.chdir(tmp_path)

    mock_storage = MagicMock(spec=NdjsonStorage)
    mock_storage.read_config.return_value = {"ollama": {}}
    with (
        patch("mb.cli._require_storage", return_value=mock_storage),
        patch("mb.pack.build_pack", return_value="<pack>ok</pack>"),
    ):
        result = runner.invoke(cli, ["pack", "--retriever", "episode", "--episode", episode_type])

    assert result.exit_code == 0
    assert "<pack>ok</pack>" in result.output


def test_pack_retriever_episode_missing_episode(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb pack --retriever episode without --episode exits with error."""
    monkeypatch.chdir(tmp_path)

    mock_storage = MagicMock(spec=NdjsonStorage)
    mock_storage.read_config.return_value = {"ollama": {}}
    with patch("mb.cli._require_storage", return_value=mock_storage):
        result = runner.invoke(cli, ["pack", "--retriever", "episode"])

    assert result.exit_code == 1
    assert "--episode is required" in result.output


# --- mb run ---


def test_run_no_command_exits_with_error(runner: CliRunner) -> None:
    """mb run without a command exits with code 1."""
    result = runner.invoke(cli, ["run"])

    assert result.exit_code == 1
    assert "No command specified" in result.output


# --- mb projects ---


def test_projects_list_multiple(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb projects lists registered projects in table format."""
    from mb.models import ProjectEntry

    proj_a = tmp_path / "proj-a"
    proj_b = tmp_path / "proj-b"
    proj_a.mkdir()
    proj_b.mkdir()
    (proj_a / ".memory-bank").mkdir()
    (proj_b / ".memory-bank").mkdir()

    projects = {
        str(proj_a): ProjectEntry(path=str(proj_a), registered_at=1.0, session_count=10, last_import=1740000000.0),
        str(proj_b): ProjectEntry(path=str(proj_b), registered_at=2.0, session_count=5, last_import=0.0),
    }

    with patch("mb.registry.list_projects", return_value=projects):
        result = runner.invoke(cli, ["projects"])

    assert result.exit_code == 0
    assert str(proj_a) in result.output
    assert str(proj_b) in result.output
    assert "10" in result.output
    assert "never" in result.output


def test_projects_list_unreachable(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb projects marks unreachable projects."""
    from mb.models import ProjectEntry

    missing = str(tmp_path / "deleted-proj")
    projects = {
        missing: ProjectEntry(path=missing, registered_at=1.0, session_count=3, last_import=1740000000.0),
    }

    with patch("mb.registry.list_projects", return_value=projects):
        result = runner.invoke(cli, ["projects"])

    assert result.exit_code == 0
    assert "unreachable" in result.output


def test_projects_list_empty(
    runner: CliRunner,
) -> None:
    """mb projects with no registered projects shows message."""
    with patch("mb.registry.list_projects", return_value={}):
        result = runner.invoke(cli, ["projects"])

    assert result.exit_code == 0
    assert "No projects registered" in result.output


def test_projects_remove_existing(
    runner: CliRunner, tmp_path: Path,
) -> None:
    """mb projects remove removes a project from registry."""
    with patch("mb.registry.remove_project", return_value=True):
        result = runner.invoke(cli, ["projects", "remove", str(tmp_path)])

    assert result.exit_code == 0
    assert "Removed" in result.output


def test_projects_remove_nonexistent(
    runner: CliRunner,
) -> None:
    """mb projects remove for unknown project shows not found."""
    with patch("mb.registry.remove_project", return_value=False):
        result = runner.invoke(cli, ["projects", "remove", "/nonexistent"])

    assert result.exit_code == 0
    assert "not found in registry" in result.output


def test_projects_list_json(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb projects --json outputs JSON."""
    import json

    from mb.models import ProjectEntry

    proj = tmp_path / "proj-json"
    proj.mkdir()
    (proj / ".memory-bank").mkdir()

    projects = {
        str(proj): ProjectEntry(path=str(proj), registered_at=1.0, session_count=7, last_import=1740000000.0),
    }

    with patch("mb.registry.list_projects", return_value=projects):
        result = runner.invoke(cli, ["projects", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data["projects"]) == 1
    assert data["projects"][0]["session_count"] == 7
    assert data["projects"][0]["reachable"] is True


# --- mb init auto-registration ---


def test_init_registers_project_in_registry(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb init calls register_project to add project to global registry."""
    monkeypatch.chdir(tmp_path)

    with patch("mb.registry.register_project") as mock_reg:
        result = runner.invoke(cli, ["init"])

    assert result.exit_code == 0
    mock_reg.assert_called_once()


def test_import_updates_registry_stats(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb import calls update_project_stats after successful import."""
    monkeypatch.chdir(tmp_path)
    _create_config(tmp_path)

    mock_result = {
        "imported": 3, "skipped": 1,
        "plans_imported": 0, "todos_imported": 0, "tasks_imported": 0,
        "dry_run_todo_items": 0, "dry_run_task_items": 0,
    }
    mock_storage = MagicMock(spec=NdjsonStorage)
    mock_storage.list_sessions.return_value = [MagicMock()] * 3
    with (
        patch("mb.importer.import_claude_sessions_with_artifacts", return_value=mock_result),
        patch("mb.cli.NdjsonStorage", return_value=mock_storage),
        patch("mb.registry.update_project_stats") as mock_update,
    ):
        result = runner.invoke(cli, ["import"])

    assert result.exit_code == 0
    mock_update.assert_called_once()


# --- mb import --autostart tip ---


def test_import_first_import_shows_tip(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First import with sessions imported shows autostart tip."""
    monkeypatch.chdir(tmp_path)
    _create_config(tmp_path)

    mock_result = {
        "imported": 5, "skipped": 0,
        "plans_imported": 0, "todos_imported": 0, "tasks_imported": 0,
        "dry_run_todo_items": 0, "dry_run_task_items": 0,
    }
    mock_storage = MagicMock(spec=NdjsonStorage)
    mock_storage.list_sessions.return_value = [MagicMock()] * 5
    mock_storage.load_import_state.return_value = {"imported": {}}  # empty = first import
    with (
        patch("mb.importer.import_claude_sessions_with_artifacts", return_value=mock_result),
        patch("mb.cli.NdjsonStorage", return_value=mock_storage),
        patch("mb.registry.update_project_stats"),
    ):
        result = runner.invoke(cli, ["import"])

    assert result.exit_code == 0
    assert "mb hooks install --autostart" in result.output


def test_import_subsequent_import_no_tip(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Subsequent import does not show autostart tip."""
    monkeypatch.chdir(tmp_path)
    _create_config(tmp_path)

    mock_result = {
        "imported": 2, "skipped": 3,
        "plans_imported": 0, "todos_imported": 0, "tasks_imported": 0,
        "dry_run_todo_items": 0, "dry_run_task_items": 0,
    }
    mock_storage = MagicMock(spec=NdjsonStorage)
    mock_storage.list_sessions.return_value = [MagicMock()] * 5
    mock_storage.load_import_state.return_value = {"imported": {"session-1": "done"}}  # non-empty
    with (
        patch("mb.importer.import_claude_sessions_with_artifacts", return_value=mock_result),
        patch("mb.cli.NdjsonStorage", return_value=mock_storage),
        patch("mb.registry.update_project_stats"),
    ):
        result = runner.invoke(cli, ["import"])

    assert result.exit_code == 0
    assert "mb hooks install --autostart" not in result.output


def test_import_first_import_zero_sessions_no_tip(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First import with zero sessions imported does not show tip."""
    monkeypatch.chdir(tmp_path)
    _create_config(tmp_path)

    mock_result = {
        "imported": 0, "skipped": 0,
        "plans_imported": 0, "todos_imported": 0, "tasks_imported": 0,
        "dry_run_todo_items": 0, "dry_run_task_items": 0,
    }
    mock_storage = MagicMock(spec=NdjsonStorage)
    mock_storage.load_import_state.return_value = {"imported": {}}
    with (
        patch("mb.importer.import_claude_sessions_with_artifacts", return_value=mock_result),
        patch("mb.cli.NdjsonStorage", return_value=mock_storage),
    ):
        result = runner.invoke(cli, ["import"])

    assert result.exit_code == 0
    assert "mb hooks install --autostart" not in result.output


def test_reinit_preserves_existing_registration(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-running mb init preserves existing registry entry (idempotent)."""
    monkeypatch.chdir(tmp_path)

    with patch("mb.registry.register_project") as mock_reg:
        # First init
        runner.invoke(cli, ["init"])
        # Second init — should still call register_project (idempotent)
        result = runner.invoke(cli, ["init"])

    assert result.exit_code == 0
    # register_project is called on first init; second init doesn't create new storage
    # so it returns (False, storage) and doesn't call register_project again
    assert mock_reg.call_count >= 1


# --- helpers ---


def _create_config(tmp_path: Path) -> Path:
    """Create a minimal .memory-bank/config.json so _require_storage passes.

    Returns the storage root path.
    """
    import json

    root = tmp_path / ".memory-bank"
    root.mkdir(exist_ok=True)
    (root / "config.json").write_text(
        json.dumps({"ollama": {}}), encoding="utf-8"
    )
    return root
