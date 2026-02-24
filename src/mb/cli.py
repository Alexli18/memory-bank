"""Memory Bank CLI — click command group and subcommand stubs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import click

from mb import __version__
from mb.store import NdjsonStorage

if TYPE_CHECKING:
    from mb.graph import EpisodeType
    from mb.models import Chunk
    from mb.retriever import ContextualRetriever

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


def _require_storage() -> NdjsonStorage:
    """Open existing storage or raise MbError."""
    try:
        return NdjsonStorage.open()
    except FileNotFoundError:
        raise MbError("Memory Bank not initialized. Run `mb init` first.")


class _EpisodeRetrieverAdapter:
    """Adapts ContextualRetriever.retrieve_by_episode to Retriever protocol."""

    def __init__(self, ctx_retriever: ContextualRetriever, episode_type: EpisodeType) -> None:
        self._ctx_retriever = ctx_retriever
        self._episode_type = episode_type

    def retrieve(self, storage: NdjsonStorage) -> list[Chunk]:
        return self._ctx_retriever.retrieve_by_episode(storage, self._episode_type)


@click.group()
@click.version_option(version=__version__, prog_name="mb")
def cli() -> None:
    """Memory Bank — capture, search, and restore LLM session context."""


@cli.command()
def init() -> None:
    """Initialize Memory Bank storage in the current project."""
    created, storage = NdjsonStorage.init()
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

    from mb.pipeline import ChunkProcessor, ProcessorPipeline, PtySource

    # Auto-initialize if not initialized (FR-002)
    storage_root = _storage_root()
    if not (storage_root / "config.json").exists():
        NdjsonStorage.init(storage_root)

    storage = NdjsonStorage(storage_root)
    source = PtySource(list(child_cmd))
    session_ids = source.ingest(storage)

    pipeline = ProcessorPipeline([ChunkProcessor()])
    pipeline.run(storage, session_ids)

    ctx.exit(source.exit_code)


@cli.command()
def sessions() -> None:
    """List all recorded sessions."""
    from datetime import datetime, timezone

    storage = _require_storage()
    all_sessions = storage.list_sessions()

    if not all_sessions:
        click.echo("No sessions found.")
        return

    # Header
    click.echo(f"{'SESSION':<25}{'COMMAND':<12}{'STARTED':<22}{'EXIT'}")
    for s in all_sessions:
        session_id = s.session_id
        command = " ".join(s.command)
        if s.started_at:
            started = datetime.fromtimestamp(s.started_at, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        else:
            started = "?"
        exit_str = str(s.exit_code) if s.exit_code is not None else "-"
        click.echo(f"{session_id:<25}{command:<12}{started:<22}{exit_str}")


@cli.command()
@click.argument("session_id")
def delete(session_id: str) -> None:
    """Delete a session by ID."""
    storage = _require_storage()
    if storage.delete_session(session_id):
        # Clear stale index entries
        index_dir = storage.root / "index"
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
    from mb.ollama_client import (
        OllamaNotRunningError,
        OllamaModelNotFoundError,
        client_from_config,
    )
    from mb.search import semantic_search

    storage = _require_storage()

    # Check for sessions
    all_sessions = storage.list_sessions()
    if not all_sessions:
        click.echo(
            "No sessions found. Run `mb run -- <command>` to capture a session first."
        )
        return

    config = storage.read_config()
    ollama_cfg = config.get("ollama", {})
    client = client_from_config(config)

    try:
        results = semantic_search(
            query, top_k=top, storage=storage, ollama_client=client
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
        # Format timestamps as MM:SS
        start_str = f"{int(r.ts_start // 60):02d}:{int(r.ts_start % 60):02d}"
        end_str = f"{int(r.ts_end // 60):02d}:{int(r.ts_end % 60):02d}"

        click.echo(f"[{r.score:.2f}] Session {r.session_id} ({start_str} - {end_str})")

        snippet = r.text[:200].replace("\n", " ").strip()
        if len(r.text) > 200:
            snippet += "..."
        click.echo(f"  {snippet}")
        click.echo()

    click.echo("No more results.")


@cli.command("import")
@click.option("--dry-run", is_flag=True, help="Show what would be imported.")
def import_sessions(dry_run: bool) -> None:
    """Import historical Claude Code sessions into Memory Bank."""
    from mb.importer import import_claude_sessions

    # Auto-initialize if not initialized
    storage_root = _storage_root()
    if not (storage_root / "config.json").exists():
        NdjsonStorage.init(storage_root)
        click.echo("Initialized Memory Bank in .memory-bank/")

    storage = NdjsonStorage(storage_root)
    imported, skipped = import_claude_sessions(storage, dry_run=dry_run)

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
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["xml", "json", "md"], case_sensitive=False),
    default="xml",
    help="Output format (default: xml).",
)
@click.option("--out", type=click.Path(), default=None, help="Write output to file.")
@click.option(
    "--retriever",
    type=click.Choice(["recency", "episode"], case_sensitive=False),
    default="recency",
    help="Retrieval strategy (default: recency).",
)
@click.option(
    "--episode",
    "episode_type",
    type=click.Choice(["build", "test", "deploy", "debug", "refactor", "explore", "config", "docs", "review"], case_sensitive=False),
    default=None,
    help="Episode type filter (only with --retriever episode).",
)
def pack(budget: int, fmt: str, out: str | None, retriever: str, episode_type: str | None) -> None:
    """Generate a deterministic context pack within a token budget."""
    import sys

    from mb.models import PackFormat
    from mb.ollama_client import OllamaNotRunningError, OllamaModelNotFoundError
    from mb.pack import build_pack

    if retriever == "episode" and episode_type is None:
        raise MbError("--episode is required when using --retriever episode.")

    storage = _require_storage()
    config = storage.read_config()
    ollama_cfg = config.get("ollama", {})

    pack_format = PackFormat(fmt.lower())

    # Build the retriever
    retriever_obj = None
    if retriever == "episode":
        from mb.graph import EpisodeType
        from mb.retriever import ContextualRetriever

        ep = EpisodeType(episode_type)
        ctx_retriever = ContextualRetriever()
        # Wrap retrieve_by_episode into a Retriever-compatible object
        retriever_obj = _EpisodeRetrieverAdapter(ctx_retriever, ep)

    try:
        output = build_pack(budget, storage, fmt=pack_format, retriever=retriever_obj)
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
        Path(out).write_text(output, encoding="utf-8")
        sys.stderr.write(f"Context pack written to {out}\n")
    else:
        click.echo(output, nl=False)


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def graph(as_json: bool) -> None:
    """Display session graph with episode classification and error status."""
    import json as json_mod

    from mb.graph import SessionGraph

    storage = _require_storage()
    nodes = SessionGraph().build_graph(storage)

    if not nodes:
        click.echo("No sessions found.")
        return

    if as_json:
        items = []
        for n in nodes:
            items.append({
                "session_id": n.meta.session_id,
                "episode_type": n.episode_type.value,
                "has_error": n.has_error,
                "error_summary": n.error_summary,
                "command": " ".join(n.meta.command),
                "related_sessions": n.related_sessions,
            })
        click.echo(json_mod.dumps(items, indent=2))
        return

    # Table output
    click.echo(f"{'SESSION':<25}{'EPISODE':<12}{'ERROR':<8}{'COMMAND'}")
    for n in nodes:
        session_id = n.meta.session_id
        episode = n.episode_type.value
        error = "YES" if n.has_error else "-"
        command = " ".join(n.meta.command)
        click.echo(f"{session_id:<25}{episode:<12}{error:<8}{command}")


@cli.command()
def migrate() -> None:
    """Detect and apply storage schema migrations."""
    from mb.migrations import migrate as run_migrate

    storage = _require_storage()
    old_version, new_version = run_migrate(storage)

    if old_version == new_version:
        click.echo(f"Already up to date (v{new_version}).")
    else:
        click.echo(f"Migrated from v{old_version} to v{new_version}.")


@cli.command()
def reindex() -> None:
    """Rebuild embedding index from all chunks."""
    from mb.migrations import reindex as run_reindex
    from mb.ollama_client import (
        OllamaNotRunningError,
        OllamaModelNotFoundError,
        client_from_config,
    )

    storage = _require_storage()
    config = storage.read_config()
    ollama_cfg = config.get("ollama", {})
    client = client_from_config(config)

    try:
        stats = run_reindex(storage, client)
    except OllamaNotRunningError:
        raise OllamaUnavailableError(
            f"Cannot connect to Ollama at {ollama_cfg.get('base_url', 'http://localhost:11434')}.\n"
            "Reindex requires a running Ollama instance.\n"
            "  1. Install Ollama: https://ollama.com/download\n"
            "  2. Start the server: ollama serve\n"
            f"  3. Pull the model: ollama pull {ollama_cfg.get('embed_model', 'nomic-embed-text')}"
        )
    except OllamaModelNotFoundError as e:
        raise OllamaUnavailableError(str(e))

    click.echo(f"Reindexed {stats['chunks']} chunks from {stats['sessions']} sessions.")
