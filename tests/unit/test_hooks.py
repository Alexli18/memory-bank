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
    """hooks_status reports installed after install."""
    settings_path = tmp_path / "settings.json"
    install_hooks(settings_path)

    status = hooks_status(settings_path)
    assert status["installed"] is True
    assert "mb.hook_handler" in status["command"]


def test_status_not_installed(tmp_path: Path) -> None:
    """hooks_status reports not installed when no hook present."""
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{}", encoding="utf-8")

    status = hooks_status(settings_path)
    assert status["installed"] is False


def test_status_no_file(tmp_path: Path) -> None:
    """hooks_status reports not installed when settings.json missing."""
    settings_path = tmp_path / "settings.json"
    status = hooks_status(settings_path)
    assert status["installed"] is False
