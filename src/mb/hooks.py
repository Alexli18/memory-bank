"""Claude Code hooks management — install/uninstall/status for ~/.claude/settings.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HOOK_MARKER = "mb.hook_handler"
SETTINGS_PATH = Path.home() / ".claude" / "settings.json"


def _settings_path() -> Path:
    return SETTINGS_PATH


def _find_mb_hook(stop_hooks: list[dict]) -> int | None:
    """Find the index of the mb hook entry by matching HOOK_MARKER in command."""
    for i, entry in enumerate(stop_hooks):
        for hook in entry.get("hooks", []):
            if HOOK_MARKER in hook.get("command", ""):
                return i
    return None


def _build_hook_command() -> str:
    return f"{sys.executable} -m {HOOK_MARKER}"


def install_hooks(settings_path: Path | None = None) -> tuple[bool, str]:
    """Append Stop hook to settings.json.

    Returns (True, message) if installed, (False, message) if already present.
    """
    path = settings_path or _settings_path()

    settings: dict = {}
    if path.exists():
        settings = json.loads(path.read_text(encoding="utf-8"))

    hooks = settings.setdefault("hooks", {})
    stop_hooks = hooks.setdefault("Stop", [])

    if _find_mb_hook(stop_hooks) is not None:
        return False, "Memory Bank hook already installed."

    stop_hooks.append({
        "matcher": "",
        "hooks": [{"type": "command", "command": _build_hook_command()}],
    })

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return True, "Memory Bank hook installed."


def uninstall_hooks(settings_path: Path | None = None) -> tuple[bool, str]:
    """Remove mb hook from settings.json.

    Returns (True, message) if removed, (False, message) if not found.
    """
    path = settings_path or _settings_path()

    if not path.exists():
        return False, "No settings.json found."

    settings = json.loads(path.read_text(encoding="utf-8"))
    hooks = settings.get("hooks", {})
    stop_hooks = hooks.get("Stop", [])

    idx = _find_mb_hook(stop_hooks)
    if idx is None:
        return False, "Memory Bank hook not found."

    stop_hooks.pop(idx)

    # Clean up empty structures
    if not stop_hooks:
        del hooks["Stop"]
    if not hooks:
        del settings["hooks"]

    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return True, "Memory Bank hook uninstalled."


def hooks_status(settings_path: Path | None = None) -> dict:
    """Check if mb hook is installed.

    Returns dict with 'installed' bool and 'command' str if installed.
    """
    path = settings_path or _settings_path()

    if not path.exists():
        return {"installed": False}

    settings = json.loads(path.read_text(encoding="utf-8"))
    stop_hooks = settings.get("hooks", {}).get("Stop", [])

    idx = _find_mb_hook(stop_hooks)
    if idx is None:
        return {"installed": False}

    entry = stop_hooks[idx]
    for hook in entry.get("hooks", []):
        if HOOK_MARKER in hook.get("command", ""):
            return {"installed": True, "command": hook["command"]}

    return {"installed": False}
