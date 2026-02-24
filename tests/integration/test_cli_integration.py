"""Integration test — mb init → mb sessions round-trip via CliRunner."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from mb.cli import cli


@pytest.mark.integration
def test_init_then_sessions_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mb init creates storage, then mb sessions shows empty list."""
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    # Initialize
    result = runner.invoke(cli, ["init"])
    assert result.exit_code == 0
    assert "Initialized Memory Bank" in result.output

    # Sessions should be empty
    result = runner.invoke(cli, ["sessions"])
    assert result.exit_code == 0
    assert "No sessions found" in result.output

    # Init again is idempotent
    result = runner.invoke(cli, ["init"])
    assert result.exit_code == 0
    assert "already initialized" in result.output
