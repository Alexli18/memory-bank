# Architecture Deep-Dive

## Module Responsibilities

### cli.py — Entry Point
Click command group. All heavy imports are lazy (inside command functions).
`logging.basicConfig(level=logging.WARNING)` is set here only.
Custom exceptions: `MbError` (exit 1), `OllamaUnavailableError` (exit 2).

### storage.py — Data Layer
Session lifecycle: `create_session()`, `write_event()`, `finalize_session()`, `delete_session()`.
Config: `init_storage()`, `read_config()`, `write_config()`, `ensure_initialized()`.
`MbStorageError` for corrupt files. All functions accept `root: Path | None`.

### pty_runner.py — PTY Capture
`run_session(child_cmd, storage_root)` — forks via `pty.fork()`, sets raw mode,
forwards I/O via `select.select()`. Captures sanitized stdin/stdout events.
Handles SIGWINCH (window resize) and SIGINT (forward to child).
All disk write errors are logged, never crash the session.

### sanitizer.py — ANSI Stripping
`AnsiStripper` — streaming state machine. States: NORMAL, ESC, CSI, OSC, ESC_TWO_CHAR.
Strips CSI sequences, OSC titles, C0 controls (except \n, \t, \r).
Normalizes \r\n -> \n, bare \r -> \n. Handles incremental UTF-8 decoding.
`NoiseFilter` — regex-based post-processing: removes box-drawing, braille, spinners,
Claude Code UI chrome, collapses blank lines and horizontal whitespace.

### chunker.py — Text Chunking
`chunk_all_sessions(storage_root)` — iterates sessions, skips already-chunked.
For Claude sessions: delegates to `claude_adapter.chunk_claude_session()`.
For PTY sessions: segments events by time gaps, applies noise filter, splits by token limit.
`_quality_score(text)` — ratio of alphanumeric chars, rounded to 3 decimals.

### claude_adapter.py — Claude Code Transcript Parser
`extract_turns(session_file)` — parses Claude JSONL, extracts user/assistant turns.
Skips: `tool_result`, `tool_use`, `thinking`, `isSidechain`, `isMeta`, system tags.
`chunks_from_turns(turns, session_id)` — splits turns into chunks with overlap.
`find_claude_session_file(cwd, started_at)` — locates Claude's JSONL in `~/.claude/projects/`.

### importer.py — Retroactive Import
`discover_claude_sessions(cwd)` — finds all `.jsonl` files in `~/.claude/projects/<encoded-cwd>/`, excluding `agent-*`.
`import_claude_sessions(storage_root, dry_run)` — main flow: discover → dedup via `import_state.json` → extract_turns → create_session(source="import") → chunks_from_turns → write chunks.jsonl → finalize.
Timestamps (`started_at`/`ended_at`) are extracted from original turn timestamps, not current time.
`import_state.json` tracks imported Claude session UUIDs to prevent re-import.

### hook_handler.py — Stop Hook
Entry point: `python -m mb.hook_handler`. Reads JSON from stdin (transcript_path, session_id, cwd).
Tracks processed transcripts in `hooks_state.json` to skip unchanged files.
Always exits 0 — never blocks Claude Code.

### search.py — Vector Index
`VectorIndex` — append-only index: `vectors.bin` (float32) + `metadata.jsonl`.
Search uses `np.memmap` for memory-efficient cosine similarity.
`build_index()` — chunks all sessions, embeds via Ollama, handles staleness.
`semantic_search()` — orchestrates: build index -> embed query -> search.

### ollama_client.py — HTTP Client
`OllamaClient` — httpx wrapper for `/api/embed` and `/api/chat`.
`client_from_config(config)` — factory from config dict.
Error types: `OllamaNotRunningError`, `OllamaModelNotFoundError`, `OllamaTimeoutError`.

### pack.py — Context Pack
`build_pack(budget, storage_root)` — generates XML with token budget.
Sections in priority order: PROJECT_STATE, DECISIONS, CONSTRAINTS, ACTIVE_TASKS, EXCERPTS.
Truncation removes whole XML elements from lowest-priority sections first.

### state.py — Project State
`generate_state(storage_root, client)` — samples chunks, asks LLM to summarize.
Produces `project_state.json` with summary, decisions, constraints, tasks.
`_state_is_stale()` checks if chunks are newer than state file.

## CI Pipeline

`.github/workflows/ci.yml`: ruff check -> mypy src/ -> pytest
Triggers on push/PR to master. Python 3.10.

## Adding a New Feature Checklist

1. Add code in `src/mb/` with type annotations (`from typing import Any`)
2. Add `logger = logging.getLogger(__name__)` after imports
3. Accept `root: Path | None = None` for testability
4. Write tests in `tests/unit/` using existing fixtures from `conftest.py`
5. Verify: `uv run ruff check . && uv run mypy src/ && uv run pytest -v`
