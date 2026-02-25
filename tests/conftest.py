"""Shared test fixtures for Memory Bank tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mb.storage import DEFAULT_CONFIG
from mb.store import NdjsonStorage


@pytest.fixture(autouse=True)
def _isolate_registry(tmp_path: Path) -> Any:
    """Prevent tests from polluting the real global registry.

    Patches REGISTRY_DIR and REGISTRY_PATH in mb.registry so that
    every test writes to a temp directory instead of ~/.memory-bank/.
    """
    fake_dir = tmp_path / ".memory-bank-registry"
    fake_dir.mkdir()
    fake_path = fake_dir / "projects.json"
    with (
        patch("mb.registry.REGISTRY_DIR", fake_dir),
        patch("mb.registry.REGISTRY_PATH", fake_path),
    ):
        yield


@pytest.fixture()
def storage_root(tmp_path: Path) -> Path:
    """Initialized .memory-bank/ directory for tests."""
    root = tmp_path / ".memory-bank"
    root.mkdir()
    (root / "sessions").mkdir()
    (root / "index").mkdir()
    (root / "state").mkdir()
    (root / "config.json").write_text(
        json.dumps(DEFAULT_CONFIG, indent=2) + "\n", encoding="utf-8"
    )
    return root


@pytest.fixture()
def storage(storage_root: Path) -> NdjsonStorage:
    """NdjsonStorage backed by the same storage_root fixture."""
    return NdjsonStorage(storage_root)


@pytest.fixture()
def sample_session(storage_root: Path) -> str:
    """Create a sample session with meta.json, events.jsonl, and chunks.jsonl.

    Returns the session_id.
    """
    session_id = "20260224-120000-abcd"
    session_dir = storage_root / "sessions" / session_id
    session_dir.mkdir(parents=True)

    meta: dict[str, Any] = {
        "session_id": session_id,
        "command": ["python", "hello.py"],
        "cwd": str(storage_root.parent),
        "started_at": 1700000000.0,
        "ended_at": 1700000060.0,
        "exit_code": 0,
    }
    (session_dir / "meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    events = [
        {"ts": 1.0, "session_id": session_id, "stream": "stdout", "role": "terminal", "content": "Hello, world!"},
        {"ts": 2.0, "session_id": session_id, "stream": "stdout", "role": "terminal", "content": "Process finished"},
    ]
    with (session_dir / "events.jsonl").open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")

    chunks = [
        {
            "chunk_id": f"{session_id}-0",
            "session_id": session_id,
            "index": 0,
            "text": "Hello, world! Process finished",
            "ts_start": 1.0,
            "ts_end": 2.0,
            "token_estimate": 7,
            "quality_score": 0.75,
        },
    ]
    with (session_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch) + "\n")

    return session_id


@pytest.fixture()
def mock_ollama_client() -> MagicMock:
    """Mock OllamaClient with .embed(), .chat(), .is_running()."""
    client = MagicMock()
    client.is_running.return_value = True
    client.embed.return_value = [[0.1] * 768]
    client.chat.return_value = "Mock response"
    client.base_url = "http://localhost:11434"
    client.embed_model = "nomic-embed-text"
    client.chat_model = "gemma3:4b"
    return client
