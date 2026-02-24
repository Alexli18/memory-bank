"""Tests for mb.cli — Click CLI commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from mb.cli import cli


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

    with patch("mb.storage.init_storage", return_value=(True, tmp_path / ".memory-bank")) as mock_init:
        result = runner.invoke(cli, ["init"])

    assert result.exit_code == 0
    assert "Initialized Memory Bank" in result.output
    mock_init.assert_called_once()


def test_init_idempotent(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb init on an already-initialized dir reports 'already initialized'."""
    monkeypatch.chdir(tmp_path)

    with patch("mb.storage.init_storage", return_value=(False, tmp_path / ".memory-bank")):
        result = runner.invoke(cli, ["init"])

    assert result.exit_code == 0
    assert "already initialized" in result.output


# --- mb sessions ---


def test_sessions_empty(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb sessions with no recorded sessions prints 'No sessions found.'."""
    monkeypatch.chdir(tmp_path)
    _create_config(tmp_path)

    with patch("mb.storage.list_sessions", return_value=[]):
        result = runner.invoke(cli, ["sessions"])

    assert result.exit_code == 0
    assert "No sessions found" in result.output


def test_sessions_lists_entries(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb sessions lists session rows when sessions exist."""
    monkeypatch.chdir(tmp_path)
    _create_config(tmp_path)

    sessions = [
        {
            "session_id": "20260224-120000-abcd",
            "command": ["python", "hello.py"],
            "started_at": 1700000000.0,
            "exit_code": 0,
        },
    ]
    with patch("mb.storage.list_sessions", return_value=sessions):
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
    root = _create_config(tmp_path)
    # Create empty index dir so the cleanup loop works
    (root / "index").mkdir(exist_ok=True)

    with patch("mb.storage.delete_session", return_value=True):
        result = runner.invoke(cli, ["delete", "20260224-120000-abcd"])

    assert result.exit_code == 0
    assert "Deleted session" in result.output


def test_delete_not_found(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb delete with unknown session ID exits with code 1."""
    monkeypatch.chdir(tmp_path)
    _create_config(tmp_path)

    with patch("mb.storage.delete_session", return_value=False):
        result = runner.invoke(cli, ["delete", "nonexistent"])

    assert result.exit_code == 1
    assert "not found" in result.output.lower()


# --- mb search ---


def test_search_no_sessions(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb search with no sessions prints a hint and exits 0."""
    monkeypatch.chdir(tmp_path)
    _create_config(tmp_path)

    with patch("mb.storage.list_sessions", return_value=[]):
        result = runner.invoke(cli, ["search", "hello"])

    assert result.exit_code == 0
    assert "No sessions found" in result.output


def test_search_ollama_not_running(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb search when Ollama is down exits with code 2."""
    monkeypatch.chdir(tmp_path)
    _create_config(tmp_path)

    from mb.ollama_client import OllamaNotRunningError

    sessions = [{"session_id": "s1", "command": ["bash"], "started_at": 1.0, "exit_code": 0}]

    with (
        patch("mb.storage.list_sessions", return_value=sessions),
        patch("mb.storage.read_config", return_value={"ollama": {}}),
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
    _create_config(tmp_path)

    sessions = [{"session_id": "s1", "command": ["bash"], "started_at": 1.0, "exit_code": 0}]
    results = [
        {
            "score": 0.95,
            "session_id": "s1",
            "ts_start": 65.0,
            "ts_end": 130.0,
            "text": "some matching text",
        },
    ]

    with (
        patch("mb.storage.list_sessions", return_value=sessions),
        patch("mb.storage.read_config", return_value={"ollama": {}}),
        patch("mb.ollama_client.client_from_config", return_value=MagicMock()),
        patch("mb.search.semantic_search", return_value=results),
    ):
        result = runner.invoke(cli, ["search", "matching"])

    assert result.exit_code == 0
    assert "0.95" in result.output
    assert "01:05" in result.output  # 65 seconds = 01:05
    assert "some matching text" in result.output


# --- mb pack ---


def test_pack_ollama_not_running(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb pack when Ollama is unavailable exits with code 2."""
    monkeypatch.chdir(tmp_path)
    _create_config(tmp_path)

    from mb.ollama_client import OllamaNotRunningError

    with (
        patch("mb.storage.read_config", return_value={"ollama": {}}),
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
    _create_config(tmp_path)
    out_file = tmp_path / "context.xml"

    with (
        patch("mb.storage.read_config", return_value={"ollama": {}}),
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


def test_hooks_status_installed(
    runner: CliRunner,
) -> None:
    """mb hooks status shows installed state with command."""
    with patch(
        "mb.hooks.hooks_status",
        return_value={"installed": True, "command": "python -m mb.hook_handler"},
    ):
        result = runner.invoke(cli, ["hooks", "status"])

    assert result.exit_code == 0
    assert "Installed" in result.output
    assert "mb.hook_handler" in result.output


def test_hooks_status_not_installed(
    runner: CliRunner,
) -> None:
    """mb hooks status shows 'Not installed.' when hook is absent."""
    with patch("mb.hooks.hooks_status", return_value={"installed": False}):
        result = runner.invoke(cli, ["hooks", "status"])

    assert result.exit_code == 0
    assert "Not installed" in result.output


# --- mb import ---


def test_import_command(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb import calls import_claude_sessions and shows results."""
    monkeypatch.chdir(tmp_path)
    _create_config(tmp_path)

    with patch("mb.importer.import_claude_sessions", return_value=(3, 1)) as mock_fn:
        result = runner.invoke(cli, ["import"])

    assert result.exit_code == 0
    assert "Imported 3 session(s)" in result.output
    assert "1 skipped" in result.output
    mock_fn.assert_called_once()


def test_import_dry_run(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb import --dry-run shows what would be imported."""
    monkeypatch.chdir(tmp_path)
    _create_config(tmp_path)

    with patch("mb.importer.import_claude_sessions", return_value=(2, 0)) as mock_fn:
        result = runner.invoke(cli, ["import", "--dry-run"])

    assert result.exit_code == 0
    assert "Dry run" in result.output
    assert "2 session(s) would be imported" in result.output
    mock_fn.assert_called_once_with(tmp_path / ".memory-bank", dry_run=True)


def test_import_no_sessions_found(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb import with no Claude sessions shows message."""
    monkeypatch.chdir(tmp_path)
    _create_config(tmp_path)

    with patch("mb.importer.import_claude_sessions", return_value=(0, 0)):
        result = runner.invoke(cli, ["import"])

    assert result.exit_code == 0
    assert "No Claude Code sessions found" in result.output


# --- mb run ---


def test_run_no_command_exits_with_error(runner: CliRunner) -> None:
    """mb run without a command exits with code 1."""
    result = runner.invoke(cli, ["run"])

    assert result.exit_code == 1
    assert "No command specified" in result.output


# --- helpers ---


def _create_config(tmp_path: Path) -> Path:
    """Create a minimal .memory-bank/config.json so _require_initialized passes.

    Returns the storage root path.
    """
    import json

    root = tmp_path / ".memory-bank"
    root.mkdir(exist_ok=True)
    (root / "config.json").write_text(
        json.dumps({"ollama": {}}), encoding="utf-8"
    )
    return root
