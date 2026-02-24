"""Memory Bank CLI — click command group and subcommand stubs."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from mb import __version__

logging.basicConfig(level=logging.WARNING)


class MbError(click.ClickException):
    """General Memory Bank error (exit code 1)."""

    exit_code = 1

    def __init__(self, message: str) -> None:
        super().__init__(message)


class OllamaUnavailableError(click.ClickException):
    """Ollama connection error (exit code 2)."""

    exit_code = 2

    def __init__(self, message: str) -> None:
        super().__init__(message)


def _storage_root() -> Path:
    """Return path to .memory-bank/ in the current directory."""
    return Path.cwd() / ".memory-bank"


def _require_initialized() -> Path:
    """Raise MbError if storage is not initialized. Return storage path."""
    storage = _storage_root()
    if not (storage / "config.json").exists():
        raise MbError("Memory Bank not initialized. Run `mb init` first.")
    return storage


@click.group()
@click.version_option(version=__version__, prog_name="mb")
def cli() -> None:
    """Memory Bank — capture, search, and restore LLM session context."""


@cli.command()
def init() -> None:
    """Initialize Memory Bank storage in the current project."""
    from mb import storage

    created, storage_path = storage.init_storage()
    if created:
        click.echo("Initialized Memory Bank in .memory-bank/")
        click.echo(
            "Warning: Captured sessions may contain sensitive data (API keys, passwords).\n"
            "         .memory-bank/ has been added to .gitignore."
        )
    else:
        click.echo("Memory Bank already initialized in .memory-bank/")


@cli.command(context_settings={"ignore_unknown_options": True})
@click.argument("child_cmd", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def run(ctx: click.Context, child_cmd: tuple[str, ...]) -> None:
    """Launch a command inside the PTY wrapper with session capture."""
    if not child_cmd:
        raise MbError("No command specified. Usage: mb run -- <command>")

    from mb import storage
    from mb.pty_runner import run_session

    # Auto-initialize if not initialized (FR-002)
    storage_root = _storage_root()
    if not (storage_root / "config.json").exists():
        storage.init_storage(storage_root)

    exit_code = run_session(list(child_cmd), storage_root)
    ctx.exit(exit_code)


@cli.command()
def sessions() -> None:
    """List all recorded sessions."""
    from datetime import datetime, timezone

    from mb import storage

    root = _require_initialized()
    all_sessions = storage.list_sessions(root)

    if not all_sessions:
        click.echo("No sessions found.")
        return

    # Header
    click.echo(f"{'SESSION':<25}{'COMMAND':<12}{'STARTED':<22}{'EXIT'}")
    for s in all_sessions:
        session_id = s.get("session_id", "?")
        command = " ".join(s.get("command", []))
        started_at = s.get("started_at")
        if started_at is not None:
            started = datetime.fromtimestamp(started_at, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        else:
            started = "?"
        exit_code = s.get("exit_code")
        exit_str = str(exit_code) if exit_code is not None else "-"
        click.echo(f"{session_id:<25}{command:<12}{started:<22}{exit_str}")


@cli.command()
@click.argument("session_id")
def delete(session_id: str) -> None:
    """Delete a session by ID."""
    from mb import storage

    root = _require_initialized()
    if storage.delete_session(session_id, root):
        # Clear stale index entries
        index_dir = root / "index"
        if index_dir.exists():
            for f in index_dir.iterdir():
                f.unlink()
        click.echo(f"Deleted session {session_id}. Index cleared.")
    else:
        raise MbError(f"Session {session_id} not found.")


@cli.command()
@click.argument("query")
@click.option("--top", default=5, type=int, help="Number of results to return.")
def search(query: str, top: int) -> None:
    """Semantic search across captured sessions."""
    from mb import storage
    from mb.ollama_client import (
        OllamaNotRunningError,
        OllamaModelNotFoundError,
        client_from_config,
    )

    root = _require_initialized()

    # Check for sessions
    all_sessions = storage.list_sessions(root)
    if not all_sessions:
        click.echo(
            "No sessions found. Run `mb run -- <command>` to capture a session first."
        )
        return

    config = storage.read_config(root)
    ollama_cfg = config.get("ollama", {})
    client = client_from_config(config)

    try:
        from mb.search import semantic_search

        results = semantic_search(
            query, top_k=top, storage_root=root, ollama_client=client
        )
    except OllamaNotRunningError:
        raise OllamaUnavailableError(
            f"Cannot connect to Ollama at {ollama_cfg.get('base_url', 'http://localhost:11434')}.\n"
            "Search requires a running Ollama instance.\n"
            "  1. Install Ollama: https://ollama.com/download\n"
            "  2. Start the server: ollama serve\n"
            f"  3. Pull the model: ollama pull {ollama_cfg.get('embed_model', 'nomic-embed-text')}"
        )
    except OllamaModelNotFoundError as e:
        raise OllamaUnavailableError(str(e))

    if not results:
        click.echo("No more results.")
        return

    for r in results:
        score = r.get("score", 0.0)
        session_id = r.get("session_id", "?")
        ts_start = r.get("ts_start", 0)
        ts_end = r.get("ts_end", 0)

        # Format timestamps as MM:SS
        start_str = f"{int(ts_start // 60):02d}:{int(ts_start % 60):02d}"
        end_str = f"{int(ts_end // 60):02d}:{int(ts_end % 60):02d}"

        click.echo(f"[{score:.2f}] Session {session_id} ({start_str} - {end_str})")

        text = r.get("text", "")
        snippet = text[:200].replace("\n", " ").strip()
        if len(text) > 200:
            snippet += "..."
        click.echo(f"  {snippet}")
        click.echo()

    click.echo("No more results.")


@cli.command("import")
@click.option("--dry-run", is_flag=True, help="Show what would be imported.")
def import_sessions(dry_run: bool) -> None:
    """Import historical Claude Code sessions into Memory Bank."""
    from mb import storage
    from mb.importer import import_claude_sessions

    # Auto-initialize if not initialized
    storage_root = _storage_root()
    if not (storage_root / "config.json").exists():
        storage.init_storage(storage_root)
        click.echo("Initialized Memory Bank in .memory-bank/")

    imported, skipped = import_claude_sessions(storage_root, dry_run=dry_run)

    if imported == 0 and skipped == 0:
        click.echo("No Claude Code sessions found for this project.")
        return

    if dry_run:
        click.echo(f"Dry run: {imported} session(s) would be imported, {skipped} skipped.")
    else:
        click.echo(f"Imported {imported} session(s), {skipped} skipped.")


@cli.group()
def hooks() -> None:
    """Manage Claude Code hooks for automatic session capture."""


@hooks.command()
def install() -> None:
    """Install Memory Bank hook into Claude Code settings."""
    from mb.hooks import install_hooks

    ok, msg = install_hooks()
    click.echo(msg)


@hooks.command()
def uninstall() -> None:
    """Remove Memory Bank hook from Claude Code settings."""
    from mb.hooks import uninstall_hooks

    ok, msg = uninstall_hooks()
    click.echo(msg)


@hooks.command()
def status() -> None:
    """Check if Memory Bank hook is installed."""
    from mb.hooks import hooks_status

    info = hooks_status()
    if info["installed"]:
        click.echo(f"Installed: {info['command']}")
    else:
        click.echo("Not installed.")


@cli.command()
@click.option(
    "--budget", default=6000, type=int, help="Token budget for context pack (default: 6000)."
)
@click.option("--out", type=click.Path(), default=None, help="Write output to file.")
def pack(budget: int, out: str | None) -> None:
    """Generate a deterministic context pack within a token budget."""
    import sys

    from mb.ollama_client import OllamaNotRunningError, OllamaModelNotFoundError
    from mb.pack import build_pack
    from mb.storage import read_config

    root = _require_initialized()
    config = read_config(root)
    ollama_cfg = config.get("ollama", {})

    try:
        xml_output = build_pack(budget, root)
    except OllamaNotRunningError:
        raise OllamaUnavailableError(
            f"Cannot connect to Ollama at {ollama_cfg.get('base_url', 'http://localhost:11434')}.\n"
            "Context pack generation requires a running Ollama instance.\n"
            "  1. Install Ollama: https://ollama.com/download\n"
            "  2. Start the server: ollama serve\n"
            f"  3. Pull the models: ollama pull {ollama_cfg.get('embed_model', 'nomic-embed-text')} && "
            f"ollama pull {ollama_cfg.get('chat_model', 'gemma3:4b')}"
        )
    except OllamaModelNotFoundError as e:
        raise OllamaUnavailableError(str(e))

    if out:
        Path(out).write_text(xml_output, encoding="utf-8")
        sys.stderr.write(f"Context pack written to {out}\n")
    else:
        click.echo(xml_output, nl=False)
