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
    from mb.models import Chunk, SearchResult
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

    # Artifact summary
    counts = storage.count_artifacts()
    if counts:
        parts = []
        if counts.get("plans"):
            parts.append(f"{counts['plans']} plans")
        if counts.get("todos"):
            parts.append(
                f"{counts['todos']} todo lists ({counts.get('todo_active_items', 0)} active items)"
            )
        if counts.get("tasks"):
            parts.append(
                f"{counts['tasks']} task trees ({counts.get('task_pending', 0)} pending tasks)"
            )
        if parts:
            click.echo("")
            click.echo(f"Artifacts: {', '.join(parts)}")


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
@click.option(
    "--type", "-t", "result_type",
    type=click.Choice(["session", "plan", "todo", "task"], case_sensitive=False),
    default=None,
    help="Filter results by source type.",
)
@click.option("--rerank", is_flag=True, default=False, help="Use LLM reranker for better relevance.")
@click.option("--no-decay", is_flag=True, default=False, help="Disable temporal decay boost for this search.")
@click.option("--global", "is_global", is_flag=True, default=False, help="Search across all registered projects.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output results as JSON.")
def search(
    query: str,
    top: int,
    result_type: str | None,
    rerank: bool,
    no_decay: bool,
    is_global: bool,
    as_json: bool,
) -> None:
    """Semantic search across captured sessions and artifacts."""
    from mb.ollama_client import (
        OllamaNotRunningError,
        OllamaModelNotFoundError,
        client_from_config,
    )

    if not query.strip():
        raise MbError("Search query cannot be empty.")

    if top < 1:
        raise MbError("--top must be at least 1.")

    if is_global:
        _search_global(
            query, top, result_type, rerank, no_decay, as_json,
        )
        return

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
            query, top_k=top, storage=storage, ollama_client=client,
            artifact_type=result_type, rerank=rerank, no_decay=no_decay,
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
        _render_search_result(r)

    click.echo("No more results.")


def _search_global(
    query: str,
    top: int,
    result_type: str | None,
    rerank: bool,
    no_decay: bool,
    as_json: bool,
) -> None:
    """Handle --global search across all registered projects."""
    import json as json_mod

    from mb.ollama_client import (
        OllamaNotRunningError,
        OllamaModelNotFoundError,
        client_from_config,
    )
    from mb.registry import list_projects
    from mb.search import global_search

    projects = list_projects()
    if not projects:
        click.echo(
            "No projects registered. Run 'mb init' in your project directories.",
            err=True,
        )
        return

    # Use config from first reachable project for Ollama settings
    config: dict = {}
    for project_path in projects:
        config_path = Path(project_path) / ".memory-bank" / "config.json"
        if config_path.exists():
            import json as _json
            try:
                config = _json.loads(config_path.read_text(encoding="utf-8"))
            except (_json.JSONDecodeError, OSError):
                continue
            break

    if not config:
        raise MbError("No reachable projects found with valid configuration.")

    ollama_cfg = config.get("ollama", {})
    client = client_from_config(config)

    try:
        results = global_search(
            query,
            top_k=top,
            ollama_client=client,
            artifact_type=result_type,
            no_decay=no_decay,
            rerank=rerank,
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

    if as_json:
        click.echo(json_mod.dumps([r.to_dict() for r in results], indent=2))
        return

    for r in results:
        source_type = r.artifact_type or "session"
        label = f"[{source_type}]"

        # Shorten project path for display
        project_display = r.project_path.replace(str(Path.home()), "~")

        ident = f"{project_display} > {r.session_id} §{r.index}"
        click.echo(f"{label:<10}{ident}  (score: {r.score:.2f})")

        snippet = r.text[:200].replace("\n", " ").strip()
        if len(r.text) > 200:
            snippet += "..."
        click.echo(f"  {snippet}")
        click.echo()

    click.echo("No more results.")


def _render_search_result(r: SearchResult) -> None:
    """Render a single SearchResult for local search output."""
    from datetime import datetime, timezone

    source_type = r.artifact_type or "session"
    label = f"[{source_type}]"

    if source_type == "session":
        # ts_start/ts_end are epoch floats — format as HH:MM
        if r.ts_start > 1_000_000_000:
            start_str = datetime.fromtimestamp(r.ts_start, tz=timezone.utc).strftime("%H:%M")
            end_str = datetime.fromtimestamp(r.ts_end, tz=timezone.utc).strftime("%H:%M")
        else:
            start_str = f"{int(r.ts_start // 60):02d}:{int(r.ts_start % 60):02d}"
            end_str = f"{int(r.ts_end // 60):02d}:{int(r.ts_end % 60):02d}"
        ident = f"{r.session_id} ({start_str} - {end_str})"
    elif source_type == "plan":
        slug = r.session_id.removeprefix("artifact-plan-")
        ident = f"{slug} §{r.index}"
    elif source_type == "todo":
        ident = f"{r.session_id[:8]} #todo"
    elif source_type == "task":
        ident = f"{r.session_id[:8]} #task-{r.index}"
    else:
        ident = r.session_id

    click.echo(f"{label:<10}{ident:<35}(score: {r.score:.2f})")

    snippet = r.text[:200].replace("\n", " ").strip()
    if len(r.text) > 200:
        snippet += "..."
    click.echo(f"  {snippet}")
    click.echo()


@cli.command("import")
@click.option("--dry-run", is_flag=True, help="Show what would be imported.")
def import_sessions(dry_run: bool) -> None:
    """Import historical Claude Code sessions and artifacts into Memory Bank."""
    from mb.importer import import_claude_sessions_with_artifacts

    # Auto-initialize if not initialized
    storage_root = _storage_root()
    if not (storage_root / "config.json").exists():
        NdjsonStorage.init(storage_root)
        click.echo("Initialized Memory Bank in .memory-bank/")

    storage = NdjsonStorage(storage_root)

    # Snapshot pre-import state for first-import detection
    pre_import_state = storage.load_import_state()
    is_first_import = not pre_import_state.get("imported", {})

    result = import_claude_sessions_with_artifacts(storage, dry_run=dry_run)

    imported = result["imported"]

    # Update global registry after successful import
    if not dry_run and imported > 0:
        from mb.registry import update_project_stats

        session_count = len(storage.list_sessions())
        update_project_stats(str(Path.cwd().resolve()), session_count)
    skipped = result["skipped"]
    plans = result["plans_imported"]
    todos = result["todos_imported"]
    tasks = result["tasks_imported"]

    if imported == 0 and skipped == 0:
        click.echo("No Claude Code sessions found for this project.")
        return

    if dry_run:
        click.echo(f"Would import {imported} sessions ({skipped} already imported)")
        if plans or todos or tasks:
            parts = []
            if plans:
                parts.append(f"{plans} plans")
            if todos:
                items = result["dry_run_todo_items"]
                parts.append(f"{todos} todo lists ({items} items)")
            if tasks:
                items = result["dry_run_task_items"]
                parts.append(f"{tasks} task trees ({items} tasks)")
            click.echo(f"Would import artifacts: {', '.join(parts)}")
        else:
            click.echo("No artifacts found for this project")
    else:
        click.echo(f"Imported {imported} sessions ({skipped} skipped)")
        if plans or todos or tasks:
            parts = []
            if plans:
                parts.append(f"{plans} plans")
            if todos:
                parts.append(f"{todos} todo lists")
            if tasks:
                parts.append(f"{tasks} task trees")
            click.echo(f"Imported artifacts: {', '.join(parts)}")
        else:
            click.echo("No artifacts found for this project")

        if is_first_import and imported > 0:
            click.echo(
                "Tip: Auto-inject context on session start? "
                "Run: mb hooks install --autostart"
            )


@cli.group()
def hooks() -> None:
    """Manage Claude Code hooks for automatic session capture."""


@hooks.command()
@click.option("--autostart", is_flag=True, default=False, help="Also install SessionStart hook for automatic context injection.")
def install(autostart: bool) -> None:
    """Install Memory Bank hook into Claude Code settings."""
    from mb.hooks import install_hooks

    ok, msg = install_hooks(autostart=autostart)
    click.echo(msg)


@hooks.command()
def uninstall() -> None:
    """Remove Memory Bank hook from Claude Code settings."""
    from mb.hooks import uninstall_hooks

    ok, msg = uninstall_hooks()
    click.echo(msg)


@hooks.command()
def status() -> None:
    """Check if Memory Bank hooks are installed."""
    from mb.hooks import hooks_status

    info = hooks_status()
    stop = info["stop"]
    ss = info["session_start"]

    if stop["installed"]:
        click.echo(f"Stop hook: Installed ({stop['command']})")
    else:
        click.echo("Stop hook: Not installed")

    if ss["installed"]:
        click.echo(f"SessionStart hook: Installed ({ss['command']})")
    else:
        click.echo("SessionStart hook: Not installed")


@cli.group(invoke_without_command=True)
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
@click.pass_context
def projects(ctx: click.Context, as_json: bool) -> None:
    """View and manage registered Memory Bank projects."""
    if ctx.invoked_subcommand is not None:
        return

    # Default: list projects
    import json as json_mod
    from datetime import datetime, timezone

    from mb.registry import list_projects

    all_projects = list_projects()
    if not all_projects:
        click.echo("No projects registered.")
        return

    if as_json:
        items = []
        for path, entry in all_projects.items():
            reachable = (Path(path) / ".memory-bank").is_dir()
            last_import_str = (
                datetime.fromtimestamp(entry.last_import, tz=timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%S")
                if entry.last_import > 0
                else None
            )
            items.append({
                "path": path,
                "session_count": entry.session_count,
                "last_import": last_import_str,
                "reachable": reachable,
            })
        click.echo(json_mod.dumps({"projects": items}, indent=2))
        return

    click.echo(f"{'PROJECT':<42}{'SESSIONS':>8}  {'LAST IMPORT'}")
    for path, entry in all_projects.items():
        reachable = (Path(path) / ".memory-bank").is_dir()
        if entry.last_import > 0:
            last_import_str = datetime.fromtimestamp(
                entry.last_import, tz=timezone.utc
            ).strftime("%Y-%m-%d")
        else:
            last_import_str = "never"

        suffix = "  (unreachable)" if not reachable else ""
        click.echo(f"{path:<42}{entry.session_count:>8}  {last_import_str}{suffix}")


@projects.command("remove")
@click.argument("path")
def projects_remove(path: str) -> None:
    """Remove a project from the global registry."""
    from mb.registry import remove_project

    if remove_project(path):
        click.echo(f"Removed {path} from registry.")
    else:
        click.echo(f"Project {path} not found in registry.")


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
    "--mode",
    type=click.Choice(["auto", "debug", "build", "explore"], case_sensitive=False),
    default="auto",
    help="Pack mode: auto (infer), debug, build, explore (default: auto).",
)
@click.option(
    "--retriever",
    type=click.Choice(["recency", "episode"], case_sensitive=False),
    default="recency",
    hidden=True,
    help="[Deprecated] Retrieval strategy. Use --mode instead.",
)
@click.option(
    "--episode",
    "episode_type",
    type=click.Choice(["build", "test", "deploy", "debug", "refactor", "explore", "config", "docs", "review"], case_sensitive=False),
    default=None,
    hidden=True,
    help="[Deprecated] Episode type filter. Use --mode instead.",
)
def pack(
    budget: int,
    fmt: str,
    out: str | None,
    mode: str,
    retriever: str,
    episode_type: str | None,
) -> None:
    """Generate a deterministic context pack within a token budget."""
    import sys

    from mb.models import PackFormat
    from mb.ollama_client import OllamaNotRunningError, OllamaModelNotFoundError
    from mb.pack import build_pack

    if budget < 100:
        raise MbError("--budget must be at least 100.")

    # FR-012: conflict detection — --mode with deprecated flags
    using_deprecated = retriever != "recency" or episode_type is not None
    mode_explicitly_set = mode != "auto"
    if mode_explicitly_set and using_deprecated:
        raise MbError(
            "--mode cannot be used with --retriever/--episode. "
            "The --mode flag supersedes these deprecated options."
        )

    # FR-011: deprecation warning for old flags
    if using_deprecated:
        sys.stderr.write(
            "Warning: --retriever/--episode are deprecated. Use --mode instead.\n"
        )

    if retriever == "episode" and episode_type is None:
        raise MbError("--episode is required when using --retriever episode.")

    storage = _require_storage()
    config = storage.read_config()
    ollama_cfg = config.get("ollama", {})

    pack_format = PackFormat(fmt.lower())

    # Build the retriever (deprecated path)
    retriever_obj = None
    if retriever == "episode":
        from mb.graph import EpisodeType
        from mb.retriever import ContextualRetriever

        ep = EpisodeType(episode_type)
        ctx_retriever = ContextualRetriever()
        retriever_obj = _EpisodeRetrieverAdapter(ctx_retriever, ep)

    try:
        output = build_pack(
            budget, storage, fmt=pack_format,
            retriever=retriever_obj, mode=mode,
        )
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
