<p align="center">
  <img src="logo.svg" alt="Memory Bank" width="480" />
</p>

<p align="center">
  <a href="https://github.com/Alexli18/memory-bank/actions/workflows/ci.yml"><img src="https://github.com/Alexli18/memory-bank/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License: GPL v3"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
</p>

<p align="center">
  <strong>Never lose context between LLM sessions again.</strong><br>
  Capture, search, and restore AI coding assistant conversations across sessions.
</p>

> **Note:** Memory Bank is in early alpha. APIs and storage formats may change between versions. Bug reports and feedback are welcome!

## Highlights

- **Zero-friction capture** -- a Claude Code hook records every session automatically, no wrappers needed
- **Auto-context on startup** -- a SessionStart hook injects a lightweight context pack into every new Claude session, so you never start from scratch
- **Retroactive import** -- `mb import` brings in all your historical Claude Code sessions, plans, todos, and tasks instantly
- **Semantic search** -- find past decisions, code discussions, and debugging sessions by meaning, not keywords
- **Cross-project search** -- `mb search --global` finds relevant context across all your Memory Bank projects at once
- **Context packs** -- generate a portable XML/JSON/Markdown summary and paste it into a fresh LLM session to restore full project knowledge
- **Episode classification** -- sessions are auto-tagged by type (build, test, debug, refactor, etc.) with error detection
- **Per-project isolation** -- each project gets its own `.memory-bank/` directory, the hook works globally

<!-- TODO: Replace with terminal GIF recorded via vhs or asciinema -->
<!-- Example: mb hooks install → claude session → mb search "auth" → mb pack -->
<p align="center">
  <em>Demo: a 30-second workflow from hook install to context pack (coming soon)</em>
</p>

## Table of Contents

- [How It Works](#how-it-works)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Commands](#commands)
- [Multi-Project Usage](#multi-project-usage)
- [Configuration](#configuration)
- [Data Storage](#data-storage)
- [Workflow Examples](#workflow-examples)
- [Troubleshooting](#troubleshooting)
- [Contributing & Community](#contributing--community)
- [License](#license)

## How It Works

```
You use Claude Code normally
         |
    Stop hook fires ───────────── session captured
         |
  Transcript extracted (clean turns, no TUI noise)
         |
  Chunks stored in .memory-bank/sessions/<id>/
         |
  Ready for search / pack
         |
    Next session starts
         |
  SessionStart hook fires ────── context pack injected automatically
```

Four interaction modes:
- **Hooks** (recommended) -- installs a Claude Code Stop hook that captures sessions automatically
- **Auto-context** -- a SessionStart hook injects a lightweight context pack when Claude starts, restoring project knowledge automatically (`mb hooks install --autostart`)
- **Import** -- `mb import` retroactively imports all historical Claude Code sessions for the current project
- **PTY wrapper** (fallback) -- `mb run -- <command>` wraps any CLI in a pseudo-terminal

## Installation

### 1. Install Memory Bank

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv pip install -e ".[dev]"
```

Expected output:

```
Resolved 15 packages in 1.2s
Installed 15 packages in 0.8s
```

Or with pip:

```bash
pip install -e ".[dev]"
```

### 2. Verify

```bash
mb --version
```

Expected output:

```
mb, version 0.1.0
```

### Requirements

- Python 3.10+
- macOS or Linux
- [Ollama](https://ollama.com/download) (only for `search`, `pack`, and `graph` commands -- see [Ollama dependency table](#ollama-dependency-table))

> **Windows users:** Memory Bank is developed and tested on macOS and Linux. On Windows, use [WSL (Windows Subsystem for Linux)](https://learn.microsoft.com/en-us/windows/wsl/install) for the best experience.

### Ollama Setup

Ollama is only needed for commands that use embeddings or LLM summarization. You can install hooks, import sessions, and list sessions without it.

```bash
ollama serve                   # Start server (separate terminal)
ollama pull nomic-embed-text   # Embedding model
ollama pull gemma3:4b          # Summarization model
```

## Quick Start

### Step 1. Install the hook (one time)

```bash
mb hooks install
```

Expected output:

```
Memory Bank hook installed.
```

This adds a Stop hook to `~/.claude/settings.json`. **Restart Claude Code** for it to take effect.

> **Tip:** Use `mb hooks install --autostart` to also install a SessionStart hook that automatically injects project context into every new Claude session.

### Step 2. Use Claude Code normally

```bash
claude
```

Every time Claude responds, the hook captures the session transcript automatically. No wrappers, no extra steps. You can verify capture is working after your first session:

```bash
mb sessions
```

Expected output:

```
SESSION                  COMMAND     STARTED               EXIT
20260223-194057-025c     claude      2026-02-23 19:40:57   -
```

### Step 3. Search past sessions

Requires Ollama running (see [Ollama Setup](#ollama-setup)).

```bash
mb search "authentication approach"
```

Expected output:

```
[session] 20260223-194057-025c (00:00 - 00:00)  (score: 0.76)
  User: how should we handle auth?  Assistant: I recommend JWT with refresh tokens...

[plan]    abundant-jingling-snail §2         (score: 0.68)
  [PLAN: abundant-jingling-snail] ## Auth — JWT with refresh tokens...

No more results.
```

Results are labeled by source type: `[session]`, `[plan]`, `[todo]`, `[task]`.

### Step 4. Generate a context pack

Requires Ollama running (see [Ollama Setup](#ollama-setup)).

```bash
mb pack --budget 6000
```

This outputs an XML document containing project state, decisions, constraints, tasks, and recent session excerpts -- ready to paste into a fresh Claude session.

```bash
# Save to file
mb pack --budget 6000 --out context.xml
```

## Commands

### Ollama Dependency Table

| Command | Requires Ollama | Description |
|---------|:-:|-------------|
| `mb hooks install` | No | Install Claude Code capture hook |
| `mb hooks uninstall` | No | Remove the capture hook |
| `mb hooks status` | No | Check if hook is installed |
| `mb import` | No | Import historical Claude Code sessions and artifacts |
| `mb init` | No | Initialize `.memory-bank/` storage |
| `mb sessions` | No | List recorded sessions |
| `mb delete` | No | Delete a session |
| `mb run` | No | Capture any CLI via PTY wrapper |
| `mb search` | **Yes** | Semantic search across sessions and artifacts |
| `mb graph` | **Yes** | Session graph with episode classification |
| `mb pack` | **Yes** | Generate context pack for session restore |
| `mb migrate` | No | Detect and apply storage schema migrations |
| `mb reindex` | **Yes** | Rebuild embedding index from all chunks |
| `mb projects` | No | View and manage the global project registry |
| `mb projects remove` | No | Remove a project from the registry |

### `mb hooks install`

Install the Memory Bank hook into Claude Code settings.

```bash
mb hooks install               # Stop hook only (capture sessions)
mb hooks install --autostart   # Stop + SessionStart hooks (capture + auto-context)
```

Options:
- `--autostart` -- also install a SessionStart hook that injects a lightweight context pack into every new Claude session

The **Stop hook** fires on every Claude Code Stop event and:
1. Reads the native transcript JSONL (clean structured data)
2. Extracts user/assistant turns (filters out tool calls, system messages, TUI noise)
3. Splits into searchable chunks
4. Stores in `.memory-bank/` of the current project

The **SessionStart hook** (installed with `--autostart`) fires when Claude Code starts and:
1. Checks if `.memory-bank/` exists with at least one session
2. Generates a lightweight context pack (uses cached state, no Ollama calls)
3. Injects it into the session via stdout so Claude has project context immediately

### `mb hooks uninstall`

Remove the hook.

```bash
mb hooks uninstall
# Memory Bank hook uninstalled.
```

### `mb hooks status`

Check if hooks are installed.

```bash
mb hooks status
```

```
Stop hook:         Installed (/path/to/python -m mb.hook_handler)
SessionStart hook: Installed (/path/to/python -m mb.session_start_hook)
```

If only the Stop hook is installed (no `--autostart`), the SessionStart line shows "Not installed".

### `mb import`

Retroactively import all historical Claude Code sessions and artifacts for the current project.

```bash
mb import              # Import sessions + artifacts
mb import --dry-run    # Preview what would be imported
```

```
Imported 12 sessions (3 skipped)
Imported artifacts: 2 plans, 4 todo lists, 3 task trees
```

This discovers JSONL session files from `~/.claude/projects/` and artifacts from `~/.claude/plans/`, `~/.claude/todos/`, `~/.claude/tasks/` that match the current project, and imports them into `.memory-bank/`. Useful when you start using Memory Bank in a project that already has Claude Code history.

- Imports **sessions** (conversation transcripts) and **artifacts** (plans, todo lists, task trees)
- Automatically deduplicates -- running `mb import` again skips already-imported items
- Preserves original timestamps from Claude Code sessions
- Auto-initializes `.memory-bank/` if needed
- Artifacts are chunked and indexed alongside sessions for semantic search

Options:
- `--dry-run` -- show what would be imported without making changes

### `mb init`

Initialize Memory Bank storage in the current project. Usually not needed -- the hook auto-initializes.

```bash
mb init
# Initialized Memory Bank in .memory-bank/
```

Creates:
```
.memory-bank/
  config.json      # Ollama settings, chunking params
  sessions/        # One directory per session
  index/           # Vector search index
  state/           # LLM-generated project state
```

### `mb sessions`

List all recorded sessions. If artifacts have been imported, a summary line is shown at the end.

```bash
mb sessions
```

```
SESSION                  COMMAND     STARTED               EXIT
20260223-194057-025c     claude      2026-02-23 19:40:57   -
20260223-184440-664c     claude      2026-02-23 18:44:40   0

Artifacts: 2 plans, 4 todo lists (7 active items), 3 task trees (12 pending tasks)
```

### `mb delete <session_id>`

Delete a session and clear the search index.

```bash
mb delete 20260223-184440-664c
# Deleted session 20260223-184440-664c. Index cleared.
```

### `mb search "<query>"`

Semantic search across all captured sessions and artifacts.

```bash
mb search "database schema design"
mb search "nginx config" --top 10
mb search "auth" --type plan       # Only search plans
mb search "auth" --global          # Search across all projects
mb search "auth" --rerank          # LLM-based reranking for better relevance
mb search "auth" --no-decay        # Disable temporal decay boost
mb search "auth" --global --json   # Structured JSON output
```

Results include source type labels: `[session]`, `[plan]`, `[todo]`, `[task]`.

Options:
- `--top N` -- number of results (default: 5)
- `--type session|plan|todo|task` -- filter by source type
- `--global` -- search across all registered projects (see [Cross-Project Search](#cross-project-search))
- `--rerank` -- use LLM reranking for better relevance (fetches 3x candidates, reranks via Ollama)
- `--no-decay` -- disable temporal decay boost for this search
- `--json` -- output results as JSON (supported with `--global`)

#### Temporal Decay

By default, search results are boosted by recency using an exponential decay function with a 14-day half-life. A session from yesterday scores slightly higher than an identical match from a month ago. This helps surface the most relevant recent context. Use `--no-decay` to disable this boost and rank purely by semantic similarity. The half-life is configurable in `config.json` (see [Configuration](#configuration)).

Requires Ollama running with `nomic-embed-text` model.

### `mb graph`

Display the session graph with episode classification, error detection, and related sessions.

```bash
mb graph              # Table output
mb graph --json       # JSON output
```

Table output:
```
SESSION                  EPISODE     ERROR   COMMAND
20260224-161618-0325     test        YES     pytest
20260224-161613-dcb6     build       -       python -c print(42)
```

Options:
- `--json` -- output as JSON array of session node objects

Each session is classified by episode type (build, test, deploy, debug, refactor, explore, config, docs, review). Non-Claude commands use command-based heuristics; Claude/hook/import sessions use content-based classification by analyzing chunk text. Error detection uses exit code and error keywords in session content.

### `mb pack`

Generate an XML context pack for restoring session context in a fresh LLM conversation.

```bash
mb pack                          # Default: auto mode, 6000 tokens
mb pack --budget 4000            # Custom budget
mb pack --mode debug             # Debug mode: 75% budget to recent context
mb pack --mode build             # Build mode: more decisions/tasks/plans
mb pack --mode explore           # Explore mode: more project state
mb pack --budget 8000 --out context.xml  # Save to file
```

Options:
- `--budget N` -- token budget (default: 6000)
- `--mode auto|debug|build|explore` -- pack mode that controls budget allocation (default: auto)
- `--format xml|json|md` -- output format (default: xml)
- `--out PATH` -- write to file instead of stdout

> **Note:** The `--retriever` and `--episode` flags are deprecated. Use `--mode` instead. They cannot be combined with `--mode`.

#### Pack Modes

| Mode | project_state | decisions | active_tasks | plans | recent_context | Best for |
|------|:---:|:---:|:---:|:---:|:---:|----------|
| **auto** | 15% | 15% | 15% | 15% | 40% | General use (infers from latest session) |
| **debug** | 10% | 5% | 5% | 5% | **75%** | Debugging -- maximizes recent context |
| **build** | 15% | 20% | **20%** | **20%** | 25% | Building -- more decisions, tasks, plans |
| **explore** | **25%** | 15% | 5% | 15% | 40% | Exploring -- more project state overview |

In `auto` mode, the pack mode is inferred from the latest session's episode type: debug sessions select `debug`, build/refactor/test/config/deploy select `build`, and explore/docs/review select `explore`.

#### Ollama Fallback

If Ollama returns an error during state generation (e.g., server overloaded), `mb pack` falls back to the cached `state.json` and continues. If no cached state exists, it generates an empty pack with excerpts only.

The pack contains these sections in priority order:
1. **PROJECT_STATE** -- LLM-generated summary of the project (never truncated)
2. **DECISIONS** -- architectural decisions with rationale
3. **CONSTRAINTS** -- known limitations
4. **ACTIVE_TASKS** -- pending/in-progress tasks and todos from imported artifacts (up to 15% of budget)
5. **PLANS** -- recent plans from imported artifacts (up to 15% of budget)
6. **RECENT_CONTEXT_EXCERPTS** -- recent conversation excerpts
7. **INSTRUCTIONS** -- "Paste this into a fresh LLM session"

When budget is exceeded, lower-priority sections are truncated first. PROJECT_STATE and INSTRUCTIONS are never truncated. If no artifacts have been imported, the ACTIVE_TASKS and PLANS sections are omitted.

#### Budget Guide

| Budget | What fits | Use case |
|--------|-----------|----------|
| 500 | Summary + decisions + tasks | Quick reminder |
| 2000 | + a few excerpts | Session restore after break |
| 4000 | ~23% of excerpts | Good balance |
| **6000** | ~65% of excerpts | **Default, recommended** |
| 8000 | ~90% of excerpts | Deep context |
| 10000+ | Everything | Full history |

Requires Ollama running with `nomic-embed-text` and `gemma3:4b` models.

### `mb projects`

View and manage the global project registry. Projects are automatically registered when you run `mb import`.

```bash
mb projects             # List all registered projects
mb projects --json      # JSON output
mb projects remove /path/to/project  # Remove a project from registry
```

```
PROJECT                    SESSIONS  LAST IMPORT
/Users/alex/my-project     12        2026-02-24
/Users/alex/other-project  5         2026-02-20
```

Options:
- `--json` -- output as JSON

### `mb migrate`

Detect and apply storage schema migrations.

```bash
mb migrate
```

Checks the current schema version and applies any pending migrations (e.g., v1 to v2). Safe to run multiple times -- skips if already up to date.

### `mb reindex`

Rebuild the embedding index from all chunks.

```bash
mb reindex
```

Useful after manual edits, imports, or if the index gets corrupted. Requires Ollama running.

### `mb run -- <command>`

Fallback: capture any CLI via PTY wrapper. Use this for non-Claude tools (bash, codex, etc.).

```bash
mb run -- bash
mb run -- codex --model gpt-4
```

The child process runs in a pseudo-terminal with full interactivity (colors, cursor, resize). Session is saved on exit.

## Multi-Project Usage

The hook is global -- it works in any directory where you run `claude`. Each project gets its own isolated `.memory-bank/`:

```bash
cd ~/project-a
claude          # .memory-bank/ created here automatically

cd ~/project-b
claude          # separate .memory-bank/ created here

# Search within each project
cd ~/project-a && mb search "auth"
cd ~/project-b && mb search "deployment"
```

### Cross-Project Search

Search across all registered projects at once with `--global`:

```bash
mb search "auth middleware" --global
mb search "deployment config" --global --top 10
mb search "database" --global --json
```

Projects are automatically registered when you run `mb import`. You can manage the registry with `mb projects`:

```bash
mb projects                          # See all registered projects
mb projects remove /old/project      # Remove a stale project
```

The global registry is stored at `~/.memory-bank/projects.json`. Each project's `.memory-bank/` must be reachable for its results to appear; unreachable projects are skipped with a warning.

## Configuration

After initialization, edit `.memory-bank/config.json`:

```json
{
  "version": "1.0",
  "ollama": {
    "base_url": "http://localhost:11434",
    "embed_model": "nomic-embed-text",
    "chat_model": "gemma3:4b"
  },
  "chunking": {
    "max_tokens": 512,
    "overlap_tokens": 50
  },
  "decay": {
    "enabled": true,
    "half_life_days": 14.0
  },
  "pack_modes": {
    "debug": { "recent_context": 0.75, "project_state": 0.10 },
    "build": { "active_tasks": 0.20, "plans": 0.20 }
  }
}
```

| Key | Default | Description |
|-----|---------|-------------|
| `decay.enabled` | `true` | Enable temporal decay boost on search scores |
| `decay.half_life_days` | `14.0` | Half-life in days for the decay function |
| `pack_modes.<mode>.<section>` | (see table) | Override default budget fractions per pack mode |

## Data Storage

```
.memory-bank/
  config.json
  hooks_state.json           # Claude session -> mb session mapping (hooks)
  import_state.json          # Imported session UUIDs + artifact tracking
  sessions/
    20260223-194057-025c/
      meta.json              # Session metadata (source, timestamps, command)
      chunks.jsonl           # Extracted conversation chunks
    20260223-184440-664c/
      meta.json
      events.jsonl           # Raw PTY events (only for mb run sessions)
      chunks.jsonl
  artifacts/                 # Imported Claude Code artifacts
    chunks.jsonl             # All artifact chunks (for search/pack)
    plans/
      {slug}.md              # Plan Markdown content
      {slug}.meta.json       # Plan metadata (session, timestamp)
    todos/
      {session_id}.json      # Todo list items
    tasks/
      {session_id}/
        {task_id}.json       # Task with dependencies
  index/
    vectors.bin              # Float32 embedding vectors
    metadata.jsonl           # Chunk metadata for search results
  state/
    state.json               # LLM-generated project state
```

Hook sessions have `meta.json` + `chunks.jsonl` (clean data, no events.jsonl).
Imported sessions have `meta.json` + `chunks.jsonl` (same as hooks, `source: "import"`).
PTY sessions have `meta.json` + `events.jsonl` + `chunks.jsonl`.
Artifacts are stored in `artifacts/` with type-specific subdirectories and a shared `chunks.jsonl` for search indexing.

Global registry (cross-project search):
```
~/.memory-bank/
  projects.json    # Registered project paths, session counts, timestamps
```

## Workflow Examples

### Import existing sessions into a new project

```bash
cd ~/my-project
mb import --dry-run    # See what's available
mb import              # Import sessions + artifacts (plans, todos, tasks)
mb search "auth"       # Search across sessions and artifacts
```

### Search by artifact type

```bash
mb search "database migration" --type plan   # Only plans
mb search "pending tasks" --type task        # Only tasks
mb search "login flow" --type session        # Only conversations
```

### Restore context after a break

```bash
# Generate context pack
mb pack --out context.xml

# Start fresh Claude session
claude

# Paste the XML content as your first message
# Claude now has full project context
```

### Find a past decision

```bash
mb search "why did we choose JWT"
```

### Review what happened today

```bash
mb sessions
# Pick a session ID
mb search "refactoring"
```

### Analyze session episodes

```bash
mb graph                       # See episode types and errors
mb graph --json                # Machine-readable output
mb pack --mode debug           # Focus on recent context (debug sessions)
mb pack --mode build           # Focus on decisions, tasks, and plans
```

### Auto-context: zero-effort session restore

```bash
# Install both hooks (one time)
mb hooks install --autostart

# Every time you start Claude, the SessionStart hook
# automatically injects a context pack -- no manual steps needed.
claude
# → Claude starts with project context already loaded
```

### Search across all projects

```bash
mb search "authentication" --global         # Find auth-related context everywhere
mb search "docker config" --global --top 10 # More results across projects
mb projects                                 # See which projects are registered
```

### Clean up test sessions

```bash
mb sessions
mb delete 20260223-192329-f3e9
```

## Troubleshooting

### Common Installation Issues

**`pip install` fails with "externally managed environment":**

This happens on newer Python installations that use PEP 668. Use `uv` instead:

```bash
uv pip install -e ".[dev]"
```

Or create a virtual environment first:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**`command not found: mb` after installation:**

The `mb` script may not be on your PATH. Try:

```bash
# Check where it was installed
pip show -f memory-bank | grep mb

# Or run via python module
python -m mb --version
```

**Python version mismatch:**

Memory Bank requires Python 3.10+. Check your version:

```bash
python --version
```

If you have multiple Python versions, use the correct one:

```bash
python3.12 -m pip install -e ".[dev]"
```

### SessionStart hook not injecting context

1. Check both hooks are installed: `mb hooks status` (should show both Stop and SessionStart as "Installed")
2. Ensure `.memory-bank/` exists with at least one session: `mb sessions`
3. The SessionStart hook uses cached state only -- it never calls Ollama, so Ollama does not need to be running
4. The hook exits silently if there are no sessions yet (first session requires at least one prior captured session)

### Hook not capturing sessions

1. Check hook is installed: `mb hooks status`
2. Restart Claude Code after installing the hook
3. Hooks load at Claude Code startup -- they don't apply to the session where you installed them

### Search returns no results

1. Check Ollama is running: `curl http://localhost:11434/api/tags`
2. Check models are pulled: `ollama list`
3. Check sessions exist: `mb sessions`

### Pack generates stale state

Delete the cached state to force regeneration:

```bash
rm .memory-bank/state/state.json
mb pack --budget 6000
```

## Contributing & Community

We welcome contributions of all kinds! See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding guidelines, and the PR workflow.

- [Code of Conduct](CODE_OF_CONDUCT.md) — our community standards
- [Security Policy](SECURITY.md) — how to report vulnerabilities
- [GitHub Discussions](https://github.com/Alexli18/memory-bank/discussions) — ask questions and share ideas

## License

Memory Bank is licensed under the [GNU General Public License v3.0](LICENSE).
