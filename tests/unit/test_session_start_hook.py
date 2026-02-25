"""Tests for mb.session_start_hook — SessionStart hook handler."""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest


def _make_storage_with_data(tmp_path: Path) -> Path:
    """Create a .memory-bank directory with a session and chunks."""
    mb = tmp_path / ".memory-bank"
    mb.mkdir()
    (mb / "sessions").mkdir()
    (mb / "config.json").write_text('{"version": "1.0"}', encoding="utf-8")

    session_dir = mb / "sessions" / "s1"
    session_dir.mkdir()
    meta = {
        "session_id": "s1",
        "command": ["claude"],
        "cwd": str(tmp_path),
        "started_at": 1000.0,
    }
    (session_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    chunks = [
        {
            "chunk_id": "s1-0",
            "session_id": "s1",
            "index": 0,
            "text": "Implemented authentication module with JWT token validation",
            "ts_start": 1.0,
            "ts_end": 2.0,
            "token_estimate": 50,
            "quality_score": 0.8,
        },
    ]
    with (session_dir / "chunks.jsonl").open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    return tmp_path


def _run_hook(stdin_text: str, cwd: Path) -> str:
    """Run session_start_hook.main() capturing stdout. Returns output."""
    from mb.session_start_hook import main

    captured = io.StringIO()
    with (
        patch("sys.stdin", io.StringIO(stdin_text)),
        patch("sys.stdout", captured),
        patch.object(Path, "cwd", return_value=cwd),
    ):
        main()

    return captured.getvalue()


class TestSessionStartHookStartup:
    """Tests for startup source with populated storage."""

    def test_startup_with_data_produces_output(self, tmp_path: Path) -> None:
        """startup source with populated storage produces non-empty output."""
        project = _make_storage_with_data(tmp_path)
        payload = json.dumps({"session_id": "test", "source": "startup"})

        with patch("mb.pack.build_pack", return_value="<PACK>xml</PACK>") as mock_bp:
            output = _run_hook(payload, project)

        assert output == "<PACK>xml</PACK>"
        mock_bp.assert_called_once()
        _, kwargs = mock_bp.call_args
        assert kwargs["lightweight"] is True
        assert kwargs["mode"] == "auto"


class TestSessionStartHookNonStartup:
    """Tests for non-startup sources — should produce no output."""

    @pytest.mark.parametrize("source", ["resume", "clear", "compact"])
    def test_non_startup_source_no_output(self, tmp_path: Path, source: str) -> None:
        """Non-startup sources produce no output."""
        project = _make_storage_with_data(tmp_path)
        payload = json.dumps({"session_id": "test", "source": source})

        output = _run_hook(payload, project)

        assert output == ""


class TestSessionStartHookEdgeCases:
    """Tests for edge cases — missing data, malformed input."""

    def test_missing_memory_bank_no_output(self, tmp_path: Path) -> None:
        """Missing .memory-bank/ directory produces no output."""
        payload = json.dumps({"session_id": "test", "source": "startup"})

        output = _run_hook(payload, tmp_path)

        assert output == ""

    def test_zero_sessions_no_output(self, tmp_path: Path) -> None:
        """Storage exists but zero sessions produces no output."""
        mb = tmp_path / ".memory-bank"
        mb.mkdir()
        (mb / "sessions").mkdir()
        (mb / "config.json").write_text('{"version": "1.0"}', encoding="utf-8")

        payload = json.dumps({"session_id": "test", "source": "startup"})
        output = _run_hook(payload, tmp_path)

        assert output == ""

    def test_malformed_json_no_output(self, tmp_path: Path) -> None:
        """Malformed JSON on stdin produces no output."""
        output = _run_hook("not json at all {{{", tmp_path)

        assert output == ""

    def test_empty_stdin_no_output(self, tmp_path: Path) -> None:
        """Empty stdin produces no output."""
        output = _run_hook("", tmp_path)

        assert output == ""

    def test_exception_during_build_pack_no_output(self, tmp_path: Path) -> None:
        """Exception during build_pack produces no output, exit 0 via __main__ handler."""
        project = _make_storage_with_data(tmp_path)
        payload = json.dumps({"session_id": "test", "source": "startup"})

        captured = io.StringIO()
        with (
            patch("sys.stdin", io.StringIO(payload)),
            patch("sys.stdout", captured),
            patch.object(Path, "cwd", return_value=project),
            patch("mb.pack.build_pack", side_effect=RuntimeError("boom")),
        ):
            # main() raises — the __main__ block would catch it
            from mb.session_start_hook import main
            with pytest.raises(RuntimeError, match="boom"):
                main()

        assert captured.getvalue() == ""

    def test_sessions_exist_but_no_chunks_no_output(self, tmp_path: Path) -> None:
        """Sessions exist but with no chunks produces no output."""
        mb = tmp_path / ".memory-bank"
        mb.mkdir()
        (mb / "sessions").mkdir()
        (mb / "config.json").write_text('{"version": "1.0"}', encoding="utf-8")

        session_dir = mb / "sessions" / "s1"
        session_dir.mkdir()
        meta = {
            "session_id": "s1",
            "command": ["claude"],
            "cwd": str(tmp_path),
            "started_at": 1000.0,
        }
        (session_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

        payload = json.dumps({"session_id": "test", "source": "startup"})
        output = _run_hook(payload, tmp_path)

        assert output == ""

    def test_startup_no_cached_state_still_produces_output(self, tmp_path: Path) -> None:
        """startup source with sessions/chunks but no state.json still produces output (SC-005)."""
        project = _make_storage_with_data(tmp_path)
        state_dir = project / ".memory-bank" / "state"
        if state_dir.exists():
            import shutil
            shutil.rmtree(state_dir)

        payload = json.dumps({"session_id": "test", "source": "startup"})

        with patch("mb.pack.build_pack", return_value="<PACK>minimal</PACK>") as mock_bp:
            output = _run_hook(payload, project)

        assert output == "<PACK>minimal</PACK>"
        mock_bp.assert_called_once()
