"""Global project registry — tracks Memory Bank projects across directories."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from mb.models import ProjectEntry

REGISTRY_DIR = Path.home() / ".memory-bank"
REGISTRY_PATH = REGISTRY_DIR / "projects.json"
REGISTRY_VERSION = 1


def _read_registry() -> dict[str, Any]:
    """Read the registry file, returning empty structure if missing/corrupt."""
    if not REGISTRY_PATH.exists():
        return {"version": REGISTRY_VERSION, "projects": {}}
    try:
        data: dict[str, Any] = json.loads(
            REGISTRY_PATH.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError):
        return {"version": REGISTRY_VERSION, "projects": {}}
    return data


def _write_registry(data: dict[str, Any]) -> None:
    """Atomically write the registry file via tempfile + os.replace()."""
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(REGISTRY_DIR), suffix=".tmp", prefix="projects_"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp_path, str(REGISTRY_PATH))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def register_project(path: str) -> ProjectEntry:
    """Register a project path. Idempotent — preserves existing stats."""
    resolved = str(Path(path).resolve())
    data = _read_registry()
    projects = data.get("projects", {})

    if resolved in projects:
        entry = ProjectEntry.from_dict(resolved, projects[resolved])
    else:
        entry = ProjectEntry(
            path=resolved,
            registered_at=time.time(),
        )
        projects[resolved] = entry.to_dict()
        data["projects"] = projects
        _write_registry(data)

    return entry


def list_projects() -> dict[str, ProjectEntry]:
    """Return all registered projects as {path: ProjectEntry}."""
    data = _read_registry()
    projects = data.get("projects", {})
    return {
        path: ProjectEntry.from_dict(path, info)
        for path, info in projects.items()
    }


def remove_project(path: str) -> bool:
    """Remove a project from the registry. Returns False if not found."""
    resolved = str(Path(path).resolve())
    data = _read_registry()
    projects = data.get("projects", {})

    if resolved not in projects:
        return False

    del projects[resolved]
    data["projects"] = projects
    _write_registry(data)
    return True


def update_project_stats(path: str, session_count: int) -> None:
    """Update last_import timestamp and session_count for a project."""
    resolved = str(Path(path).resolve())
    data = _read_registry()
    projects = data.get("projects", {})

    if resolved not in projects:
        register_project(resolved)
        data = _read_registry()
        projects = data.get("projects", {})

    projects[resolved]["last_import"] = time.time()
    projects[resolved]["session_count"] = session_count
    data["projects"] = projects
    _write_registry(data)
