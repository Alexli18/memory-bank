"""Claude Code hooks management — install/uninstall/status for ~/.claude/settings.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HOOK_MARKER = "mb.hook_handler"
SESSION_START_MARKER = "mb.session_start_hook"
SETTINGS_PATH = Path.home() / ".claude" / "settings.json"


def _settings_path() -> Path:
    return SETTINGS_PATH


def _find_mb_hook(stop_hooks: list[dict[str, Any]]) -> int | None:
    """Find the index of the mb hook entry by matching HOOK_MARKER in command."""
    for i, entry in enumerate(stop_hooks):
        for hook in entry.get("hooks", []):
            if HOOK_MARKER in hook.get("command", ""):
                return i
    return None


def _build_hook_command() -> str:
    return f"{sys.executable} -m {HOOK_MARKER}"


def _find_session_start_hook(session_start_hooks: list[dict[str, Any]]) -> int | None:
    """Find the index of the mb SessionStart hook entry by matching SESSION_START_MARKER."""
    for i, entry in enumerate(session_start_hooks):
        for hook in entry.get("hooks", []):
            if SESSION_START_MARKER in hook.get("command", ""):
                return i
    return None


def _build_session_start_command() -> str:
    return f"{sys.executable} -m {SESSION_START_MARKER}"


def install_hooks(
    settings_path: Path | None = None,
    autostart: bool = False,
) -> tuple[bool, str]:
    """Install hook(s) into settings.json.

    Without autostart: installs Stop hook only (existing behavior).
    With autostart=True: installs both Stop and SessionStart hooks.

    Returns (True/False, message) describing what was installed.
    """
    path = settings_path or _settings_path()

    settings: dict = {}
    if path.exists():
        settings = json.loads(path.read_text(encoding="utf-8"))

    hooks = settings.setdefault("hooks", {})
    stop_hooks = hooks.setdefault("Stop", [])

    stop_installed = _find_mb_hook(stop_hooks) is not None
    stop_added = False

    if not stop_installed:
        stop_hooks.append({
            "matcher": "",
            "hooks": [{"type": "command", "command": _build_hook_command()}],
        })
        stop_added = True

    session_start_added = False
    if autostart:
        ss_hooks = hooks.setdefault("SessionStart", [])
        if _find_session_start_hook(ss_hooks) is None:
            ss_hooks.append({
                "matcher": "",
                "hooks": [{"type": "command", "command": _build_session_start_command()}],
            })
            session_start_added = True

    if not stop_added and not session_start_added:
        return False, "Memory Bank hooks already installed." if autostart else "Memory Bank hook already installed."

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

    if autostart:
        if stop_added and session_start_added:
            return True, "Memory Bank hooks installed (Stop + SessionStart)."
        if session_start_added:
            return True, "Memory Bank SessionStart hook installed."
        return True, "Memory Bank hook installed."
    return True, "Memory Bank hook installed."


def uninstall_hooks(settings_path: Path | None = None) -> tuple[bool, str]:
    """Remove both Stop and SessionStart mb hooks from settings.json.

    Returns (True, message) if at least one was removed,
    (False, message) if neither was found.
    """
    path = settings_path or _settings_path()

    if not path.exists():
        return False, "Memory Bank hook not found."

    settings = json.loads(path.read_text(encoding="utf-8"))
    hooks = settings.get("hooks", {})
    removed_any = False

    # Remove Stop hook
    stop_hooks = hooks.get("Stop", [])
    stop_idx = _find_mb_hook(stop_hooks)
    if stop_idx is not None:
        stop_hooks.pop(stop_idx)
        removed_any = True
        if not stop_hooks:
            del hooks["Stop"]

    # Remove SessionStart hook
    ss_hooks = hooks.get("SessionStart", [])
    ss_idx = _find_session_start_hook(ss_hooks)
    if ss_idx is not None:
        ss_hooks.pop(ss_idx)
        removed_any = True
        if not ss_hooks:
            del hooks["SessionStart"]

    if not removed_any:
        return False, "Memory Bank hook not found."

    # Clean up empty hooks dict
    if not hooks:
        del settings["hooks"]

    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return True, "Memory Bank hooks uninstalled."


def hooks_status(settings_path: Path | None = None) -> dict[str, Any]:
    """Check installation status of both Stop and SessionStart hooks.

    Returns dict with 'stop' and 'session_start' sub-dicts,
    each containing 'installed' (bool) and 'command' (str | None).
    """
    path = settings_path or _settings_path()

    not_installed: dict[str, Any] = {"installed": False, "command": None}

    if not path.exists():
        return {"stop": {**not_installed}, "session_start": {**not_installed}}

    settings = json.loads(path.read_text(encoding="utf-8"))
    hooks = settings.get("hooks", {})

    # Check Stop hook
    stop_hooks = hooks.get("Stop", [])
    stop_idx = _find_mb_hook(stop_hooks)
    if stop_idx is not None:
        cmd = None
        for hook in stop_hooks[stop_idx].get("hooks", []):
            if HOOK_MARKER in hook.get("command", ""):
                cmd = hook["command"]
                break
        stop_info: dict[str, Any] = {"installed": True, "command": cmd}
    else:
        stop_info = {**not_installed}

    # Check SessionStart hook
    ss_hooks = hooks.get("SessionStart", [])
    ss_idx = _find_session_start_hook(ss_hooks)
    if ss_idx is not None:
        cmd = None
        for hook in ss_hooks[ss_idx].get("hooks", []):
            if SESSION_START_MARKER in hook.get("command", ""):
                cmd = hook["command"]
                break
        ss_info: dict[str, Any] = {"installed": True, "command": cmd}
    else:
        ss_info = {**not_installed}

    return {"stop": stop_info, "session_start": ss_info}
