# Memory Bank

Capture, search, and restore LLM session context across sessions.

Memory Bank automatically records your conversations with AI coding assistants (Claude Code, Codex, etc.), indexes them for semantic search, and generates context packs you can paste into fresh sessions to restore project knowledge.

## How It Works

```
You use Claude Code normally
         |
    Stop hook fires
         |
  Transcript extracted (clean turns, no TUI noise)
         |
  Chunks stored in .memory-bank/sessions/<id>/
         |
  Ready for search / pack
```

Two capture modes:
- **Hooks** (recommended) -- installs a Claude Code Stop hook that captures sessions automatically
- **PTY wrapper** (fallback) -- `mb run -- <command>` wraps any CLI in a pseudo-terminal

## Installation

```bash
# With uv (recommended)
uv pip install -e ".[dev]"

# Or pip
pip install -e ".[dev]"

# Verify
mb --version
```

### Requirements

- Python 3.10+
- macOS or Linux
- [Ollama](https://ollama.com/download) (only for `search` and `pack` commands)

### Ollama Setup

```bash
ollama serve                   # Start server (separate terminal)
ollama pull nomic-embed-text   # Embedding model
ollama pull gemma3:4b          # Summarization model
```

## Quick Start

### 1. Install the hook (one time)

```bash
mb hooks install
```

This adds a Stop hook to `~/.claude/settings.json`. Restart Claude Code for it to take effect.

### 2. Use Claude Code normally

```bash
claude
```

Every time Claude responds, the hook captures the session transcript automatically. No wrappers, no extra steps.

### 3. Search past sessions

```bash
mb search "authentication approach"
```

Output:
```
[0.76] Session 20260223-194057-025c (00:00 - 00:00)
  User: how should we handle auth?  Assistant: I recommend JWT with refresh tokens...

[0.68] Session 20260223-184440-664c (00:00 - 00:00)
  User: implement login endpoint  Assistant: Here's the implementation...

No more results.
```

### 4. Generate a context pack

```bash
mb pack --budget 6000
```

Output: an XML document containing project state, decisions, constraints, tasks, and recent session excerpts -- ready to paste into a fresh Claude session.

```bash
# Save to file
mb pack --budget 6000 --out context.xml
```

## Commands

### `mb hooks install`

Install the Memory Bank hook into Claude Code settings.

```bash
mb hooks install
# Memory Bank hook installed.
```

The hook fires on every Claude Code Stop event and:
1. Reads the native transcript JSONL (clean structured data)
2. Extracts user/assistant turns (filters out tool calls, system messages, TUI noise)
3. Splits into searchable chunks
4. Stores in `.memory-bank/` of the current project

### `mb hooks uninstall`

Remove the hook.

```bash
mb hooks uninstall
# Memory Bank hook uninstalled.
```

### `mb hooks status`

Check if the hook is installed.

```bash
mb hooks status
# Installed: /path/to/python -m mb.hook_handler
```

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

List all recorded sessions.

```bash
mb sessions
```

```
SESSION                  COMMAND     STARTED               EXIT
20260223-194057-025c     claude      2026-02-23 19:40:57   -
20260223-184440-664c     claude      2026-02-23 18:44:40   0
```

### `mb delete <session_id>`

Delete a session and clear the search index.

```bash
mb delete 20260223-184440-664c
# Deleted session 20260223-184440-664c. Index cleared.
```

### `mb search "<query>"`

Semantic search across all captured sessions.

```bash
mb search "database schema design"
mb search "nginx config" --top 10
```

Options:
- `--top N` -- number of results (default: 5)

Requires Ollama running with `nomic-embed-text` model.

### `mb pack`

Generate an XML context pack for restoring session context in a fresh LLM conversation.

```bash
mb pack                        # Default budget: 6000 tokens
mb pack --budget 4000          # Custom budget
mb pack --budget 8000 --out context.xml  # Save to file
```

Options:
- `--budget N` -- token budget (default: 6000)
- `--out PATH` -- write to file instead of stdout

The pack contains these sections in priority order:
1. **PROJECT_STATE** -- LLM-generated summary of the project (never truncated)
2. **DECISIONS** -- architectural decisions with rationale
3. **CONSTRAINTS** -- known limitations
4. **ACTIVE_TASKS** -- current task status
5. **RECENT_CONTEXT_EXCERPTS** -- recent conversation excerpts
6. **INSTRUCTIONS** -- "Paste this into a fresh LLM session"

When budget is exceeded, sections are removed from the bottom (excerpts first, then tasks, then decisions). PROJECT_STATE and CONSTRAINTS are never truncated.

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
  }
}
```

## Data Storage

```
.memory-bank/
  config.json
  hooks_state.json           # Claude session -> mb session mapping
  sessions/
    20260223-194057-025c/
      meta.json              # Session metadata (source, timestamps, command)
      chunks.jsonl           # Extracted conversation chunks
    20260223-184440-664c/
      meta.json
      events.jsonl           # Raw PTY events (only for mb run sessions)
      chunks.jsonl
  index/
    vectors.bin              # Float32 embedding vectors
    metadata.jsonl           # Chunk metadata for search results
  state/
    state.json               # LLM-generated project state
```

Hook sessions have `meta.json` + `chunks.jsonl` (clean data, no events.jsonl).
PTY sessions have `meta.json` + `events.jsonl` + `chunks.jsonl`.

## Workflow Examples

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

### Clean up test sessions

```bash
mb sessions
mb delete 20260223-192329-f3e9
```

## Troubleshooting

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

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check .

# Run a specific test file
uv run pytest tests/unit/test_hooks.py -v
```
