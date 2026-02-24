---
name: memory-bank-dev
description: >
  Development guide for contributing to the memory-bank project — a Python CLI tool
  for capturing, indexing, and restoring LLM session context. Use when working on
  memory-bank source code, adding features, fixing bugs, writing tests, or refactoring
  modules in the src/mb/ directory. Also use when the user asks about the memory-bank
  architecture, data flow, module responsibilities, testing patterns, or CI pipeline.
---

# Memory Bank — Development Guide

Python 3.10+ CLI tool: PTY session capture -> ANSI sanitization -> chunking -> Ollama embeddings -> cosine search -> context pack generation.

## Project Layout

```
src/mb/
├── cli.py              # Click CLI commands (entry point: mb.cli:cli)
├── pty_runner.py        # PTY wrapper — transparent I/O capture
├── sanitizer.py         # ANSI escape stripper + noise filter
├── chunker.py           # Events -> text chunks with quality scoring
├── claude_adapter.py    # Claude Code native JSONL transcript parser
├── hook_handler.py      # Claude Code Stop hook entry point
├── hooks.py             # Hook install/uninstall in ~/.claude/settings.json
├── importer.py          # Retroactive import of historical Claude Code sessions
├── search.py            # VectorIndex (vectors.bin + metadata.jsonl) + semantic search
├── ollama_client.py     # httpx wrapper for Ollama embed/chat API
├── pack.py              # XML context pack builder with token budget
├── state.py             # ProjectState generation via LLM summarization
├── storage.py           # Session lifecycle, config, directory management
└── __init__.py          # Version string
tests/
├── conftest.py          # Shared fixtures: storage_root, sample_session, mock_ollama_client
├── unit/                # Unit tests (no external deps)
└── integration/         # Tests marked @pytest.mark.integration
```

## Dev Commands

```bash
uv run ruff check .      # Lint
uv run mypy src/          # Type check (strict: disallow_untyped_defs)
uv run pytest -v          # Run tests (194 tests)
uv run pytest -m "not integration"  # Skip integration tests
```

## Data Flow

```
PTY path:   pty_runner -> events.jsonl -> chunker -> chunks.jsonl -> search (embed) -> index/
Hook path:  hook_handler -> claude_adapter (extract_turns) -> chunks_from_turns -> chunks.jsonl
Import:     importer (discover ~/.claude/projects/) -> extract_turns -> chunks_from_turns -> chunks.jsonl
Search:     query -> ollama embed -> VectorIndex.search (cosine sim via numpy memmap)
Pack:       state.py (LLM summarize) -> pack.py (XML sections + token budget)
```

## Key Conventions

- All public functions use `root: Path | None = None` for testability (defaults to cwd)
- `storage.create_session()` is the single entry for session creation (PTY and hooks)
- `storage.finalize_session()` is the single entry for session completion
- `ollama_client.client_from_config(config)` is the factory for OllamaClient
- `chunker._quality_score(text)` is the single quality scoring function (returns rounded float)
- Error hierarchy: `OllamaError` -> `OllamaNotRunningError | OllamaModelNotFoundError | OllamaTimeoutError`
- `MbStorageError` for corrupt config/storage files
- Logging: `logger = logging.getLogger(__name__)` in every module; `basicConfig` only in `cli.py`

## Testing Patterns

- **CLI tests**: `click.testing.CliRunner` + `monkeypatch.chdir(tmp_path)` + mock heavy deps
- **Storage tests**: Real filesystem via `tmp_path`, no mocking
- **Ollama tests**: `@patch("httpx.get")` / `@patch("httpx.post")` — no real network
- **Sanitizer tests**: Direct `AnsiStripper()` instance, byte input -> string output
- Shared fixtures in `tests/conftest.py`: `storage_root`, `sample_session`, `mock_ollama_client`

For module details and architecture deep-dive, see [references/architecture.md](references/architecture.md).
