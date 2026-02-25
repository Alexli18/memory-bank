"""Tests for mb.hooks — Claude Code hook management."""

from __future__ import annotations

import json
from pathlib import Path

from mb.hooks import install_hooks, uninstall_hooks, hooks_status


def test_install_creates_hook(tmp_path: Path) -> None:
    """install_hooks writes Stop hook to settings.json."""
    settings_path = tmp_path / "settings.json"
    ok, msg = install_hooks(settings_path)

    assert ok is True
    assert "installed" in msg.lower()

    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    stop_hooks = settings["hooks"]["Stop"]
    assert len(stop_hooks) == 1
    assert "mb.hook_handler" in stop_hooks[0]["hooks"][0]["command"]


def test_install_is_idempotent(tmp_path: Path) -> None:
    """Second install returns False without duplicating the hook."""
    settings_path = tmp_path / "settings.json"
    install_hooks(settings_path)
    ok, msg = install_hooks(settings_path)

    assert ok is False
    assert "already" in msg.lower()

    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    stop_hooks = settings["hooks"]["Stop"]
    assert len(stop_hooks) == 1


def test_install_preserves_existing_settings(tmp_path: Path) -> None:
    """Other keys in settings.json are preserved."""
    settings_path = tmp_path / "settings.json"
    existing = {"permissions": {"allow": ["Read"]}, "theme": "dark"}
    settings_path.write_text(json.dumps(existing), encoding="utf-8")

    install_hooks(settings_path)

    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    assert settings["permissions"] == {"allow": ["Read"]}
    assert settings["theme"] == "dark"
    assert "hooks" in settings


def test_install_preserves_existing_stop_hooks(tmp_path: Path) -> None:
    """Existing Stop hooks from other tools are preserved."""
    settings_path = tmp_path / "settings.json"
    existing = {
        "hooks": {
            "Stop": [
                {"matcher": "", "hooks": [{"type": "command", "command": "other-tool"}]}
            ]
        }
    }
    settings_path.write_text(json.dumps(existing), encoding="utf-8")

    install_hooks(settings_path)

    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    stop_hooks = settings["hooks"]["Stop"]
    assert len(stop_hooks) == 2
    assert stop_hooks[0]["hooks"][0]["command"] == "other-tool"
    assert "mb.hook_handler" in stop_hooks[1]["hooks"][0]["command"]


def test_uninstall_removes_hook(tmp_path: Path) -> None:
    """uninstall_hooks removes only the mb hook."""
    settings_path = tmp_path / "settings.json"
    install_hooks(settings_path)

    ok, msg = uninstall_hooks(settings_path)
    assert ok is True
    assert "uninstalled" in msg.lower()


def test_uninstall_cleans_empty_structures(tmp_path: Path) -> None:
    """Empty hooks/Stop arrays are removed after uninstall."""
    settings_path = tmp_path / "settings.json"
    install_hooks(settings_path)
    uninstall_hooks(settings_path)

    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    assert "hooks" not in settings


def test_uninstall_preserves_other_hooks(tmp_path: Path) -> None:
    """Uninstall keeps other Stop hooks in place."""
    settings_path = tmp_path / "settings.json"
    existing = {
        "hooks": {
            "Stop": [
                {"matcher": "", "hooks": [{"type": "command", "command": "other-tool"}]}
            ]
        }
    }
    settings_path.write_text(json.dumps(existing), encoding="utf-8")
    install_hooks(settings_path)
    uninstall_hooks(settings_path)

    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    stop_hooks = settings["hooks"]["Stop"]
    assert len(stop_hooks) == 1
    assert stop_hooks[0]["hooks"][0]["command"] == "other-tool"


def test_uninstall_not_found(tmp_path: Path) -> None:
    """uninstall_hooks returns False when hook not present."""
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{}", encoding="utf-8")

    ok, msg = uninstall_hooks(settings_path)
    assert ok is False


def test_uninstall_no_file(tmp_path: Path) -> None:
    """uninstall_hooks returns False when settings.json missing."""
    settings_path = tmp_path / "settings.json"
    ok, msg = uninstall_hooks(settings_path)
    assert ok is False


def test_status_installed(tmp_path: Path) -> None:
    """hooks_status reports Stop installed after install."""
    settings_path = tmp_path / "settings.json"
    install_hooks(settings_path)

    status = hooks_status(settings_path)
    assert status["stop"]["installed"] is True
    assert "mb.hook_handler" in status["stop"]["command"]
    assert status["session_start"]["installed"] is False


def test_status_not_installed(tmp_path: Path) -> None:
    """hooks_status reports not installed when no hook present."""
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{}", encoding="utf-8")

    status = hooks_status(settings_path)
    assert status["stop"]["installed"] is False
    assert status["session_start"]["installed"] is False


def test_status_no_file(tmp_path: Path) -> None:
    """hooks_status reports not installed when settings.json missing."""
    settings_path = tmp_path / "settings.json"
    status = hooks_status(settings_path)
    assert status["stop"]["installed"] is False
    assert status["session_start"]["installed"] is False


# --- Dual-hook (autostart) tests ---


def test_install_without_autostart_stop_only(tmp_path: Path) -> None:
    """install without autostart installs only Stop hook."""
    settings_path = tmp_path / "settings.json"
    ok, msg = install_hooks(settings_path, autostart=False)

    assert ok is True
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    assert "Stop" in settings["hooks"]
    assert "SessionStart" not in settings["hooks"]


def test_install_with_autostart_both_hooks(tmp_path: Path) -> None:
    """install with autostart installs both Stop and SessionStart hooks."""
    settings_path = tmp_path / "settings.json"
    ok, msg = install_hooks(settings_path, autostart=True)

    assert ok is True
    assert "Stop + SessionStart" in msg

    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    assert "Stop" in settings["hooks"]
    assert "SessionStart" in settings["hooks"]
    assert "mb.hook_handler" in settings["hooks"]["Stop"][0]["hooks"][0]["command"]
    assert "mb.session_start_hook" in settings["hooks"]["SessionStart"][0]["hooks"][0]["command"]


def test_install_autostart_when_stop_already_present(tmp_path: Path) -> None:
    """install autostart when Stop already present adds only SessionStart."""
    settings_path = tmp_path / "settings.json"
    install_hooks(settings_path, autostart=False)
    ok, msg = install_hooks(settings_path, autostart=True)

    assert ok is True
    assert "SessionStart hook installed" in msg

    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    assert len(settings["hooks"]["Stop"]) == 1
    assert len(settings["hooks"]["SessionStart"]) == 1


def test_install_autostart_both_already_present(tmp_path: Path) -> None:
    """install autostart when both already present is idempotent."""
    settings_path = tmp_path / "settings.json"
    install_hooks(settings_path, autostart=True)
    ok, msg = install_hooks(settings_path, autostart=True)

    assert ok is False
    assert "already installed" in msg.lower()


# --- Dual-hook status tests (T012) ---


def test_status_both_hooks_installed(tmp_path: Path) -> None:
    """hooks_status shows both hooks installed after autostart install."""
    settings_path = tmp_path / "settings.json"
    install_hooks(settings_path, autostart=True)

    status = hooks_status(settings_path)
    assert status["stop"]["installed"] is True
    assert "mb.hook_handler" in status["stop"]["command"]
    assert status["session_start"]["installed"] is True
    assert "mb.session_start_hook" in status["session_start"]["command"]


def test_status_only_stop_installed(tmp_path: Path) -> None:
    """hooks_status shows Stop installed, SessionStart not."""
    settings_path = tmp_path / "settings.json"
    install_hooks(settings_path, autostart=False)

    status = hooks_status(settings_path)
    assert status["stop"]["installed"] is True
    assert status["session_start"]["installed"] is False


def test_status_neither_installed(tmp_path: Path) -> None:
    """hooks_status shows neither hook installed."""
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{}", encoding="utf-8")

    status = hooks_status(settings_path)
    assert status["stop"]["installed"] is False
    assert status["stop"]["command"] is None
    assert status["session_start"]["installed"] is False
    assert status["session_start"]["command"] is None


def test_status_only_session_start_installed(tmp_path: Path) -> None:
    """hooks_status shows SessionStart installed, Stop not (edge case)."""
    settings_path = tmp_path / "settings.json"
    # Manually write only SessionStart hook
    settings = {
        "hooks": {
            "SessionStart": [
                {"matcher": "", "hooks": [{"type": "command", "command": "python -m mb.session_start_hook"}]}
            ]
        }
    }
    settings_path.write_text(json.dumps(settings), encoding="utf-8")

    status = hooks_status(settings_path)
    assert status["stop"]["installed"] is False
    assert status["session_start"]["installed"] is True
    assert "mb.session_start_hook" in status["session_start"]["command"]


# --- Dual-hook uninstall tests (T013) ---


def test_uninstall_removes_both_hooks(tmp_path: Path) -> None:
    """uninstall_hooks removes both Stop and SessionStart hooks."""
    settings_path = tmp_path / "settings.json"
    install_hooks(settings_path, autostart=True)

    ok, msg = uninstall_hooks(settings_path)
    assert ok is True
    assert "uninstalled" in msg.lower()

    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    assert "hooks" not in settings


def test_uninstall_only_stop_installed(tmp_path: Path) -> None:
    """uninstall_hooks removes Stop when only Stop present."""
    settings_path = tmp_path / "settings.json"
    install_hooks(settings_path, autostart=False)

    ok, msg = uninstall_hooks(settings_path)
    assert ok is True
    assert "uninstalled" in msg.lower()


def test_uninstall_neither_installed(tmp_path: Path) -> None:
    """uninstall_hooks returns 'not found' when neither hook present."""
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{}", encoding="utf-8")

    ok, msg = uninstall_hooks(settings_path)
    assert ok is False
    assert "not found" in msg.lower()
