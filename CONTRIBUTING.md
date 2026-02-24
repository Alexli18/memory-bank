# Contributing to Memory Bank

Welcome! We're glad you're interested in contributing to Memory Bank. Every contribution matters — whether it's fixing a typo, reporting a bug, or building a new feature.

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before participating.

## Ways to Contribute

- **Report bugs** — open a [bug report](https://github.com/Alexli18/memory-bank/issues/new?template=bug_report.yml)
- **Suggest features** — open a [feature request](https://github.com/Alexli18/memory-bank/issues/new?template=feature_request.yml)
- **Improve documentation** — fix typos, add examples, clarify instructions
- **Write code** — fix bugs, implement features, improve tests
- **Answer questions** — help others in [GitHub Discussions](https://github.com/Alexli18/memory-bank/discussions)

## Finding Issues

Look for issues labeled:

- [`good first issue`](https://github.com/Alexli18/memory-bank/labels/good%20first%20issue) — great for newcomers
- [`help wanted`](https://github.com/Alexli18/memory-bank/labels/help%20wanted) — we'd love your input

If you want to work on something, comment on the issue so others know it's being addressed.

## Development Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- [Ollama](https://ollama.com/download) (only needed for integration tests)

### Getting started

```bash
# Clone the repo
git clone https://github.com/Alexli18/memory-bank.git
cd memory-bank

# Create a virtual environment and install dependencies
uv venv
uv pip install -e ".[dev]"

# Verify
mb --version
# mb, version 0.1.0
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Project Structure

```text
src/mb/
  cli.py              # Click CLI entry point
  models.py           # Shared data models
  storage.py          # Session storage (JSONL/JSON files)
  hooks.py            # Claude Code hook management
  hook_handler.py     # Hook event handler
  importer.py         # Historical session import
  chunker.py          # Transcript → searchable chunks
  pipeline.py         # Capture pipeline orchestration
  ollama_client.py    # Ollama HTTP client (embeddings + chat)
  search.py           # Semantic search over chunks
  retriever.py        # Context retrieval strategies
  pack.py             # Context pack generation
  budgeter.py         # Token budget allocation
  renderers.py        # XML/JSON/Markdown output
  graph.py            # Session graph with episode classification
  state.py            # LLM-generated project state
  pty_runner.py       # PTY wrapper (fallback capture)
tests/
  unit/               # Fast tests, no external dependencies
  integration/        # Tests requiring Ollama or filesystem
  conftest.py         # Shared fixtures
```

## Running Tests & Linting

```bash
# Run all unit tests
uv run pytest tests/ -v

# Run a specific test file
uv run pytest tests/unit/test_hooks.py -v

# Run integration tests (requires Ollama)
uv run pytest tests/integration/ -v -m integration

# Lint
uv run ruff check .

# Type check
uv run mypy src/
```

All tests must pass before submitting a PR. CI runs `pytest` and `ruff check` on every push.

## Code Style

- We use [ruff](https://docs.astral.sh/ruff/) for linting (configured in `pyproject.toml`)
- Target: Python 3.10+ — use `X | Y` union syntax, not `Union[X, Y]`
- Type annotations are required for public functions (`mypy --strict` subset enabled)
- Keep it simple — avoid unnecessary abstractions

## Submitting Changes

### Branch naming

Use descriptive branch names:

```text
fix/hook-not-capturing
feat/export-markdown-format
docs/improve-troubleshooting
```

### Commits

- Write clear, concise commit messages
- Use imperative mood: "Add feature" not "Added feature"
- Reference issues when relevant: `Fix #42`

### Pull request workflow

1. Fork the repository
2. Create a feature branch from `master`
3. Make your changes
4. Ensure tests pass and linting is clean
5. Push to your fork and open a PR against `master`
6. Fill in the PR template — include a summary and link related issues with `Closes #`

A maintainer will review your PR. We may suggest changes — this is normal and collaborative.

## Getting Help

- **Questions?** Open a thread in [GitHub Discussions](https://github.com/Alexli18/memory-bank/discussions)
- **Bug?** File a [bug report](https://github.com/Alexli18/memory-bank/issues/new?template=bug_report.yml)
- **Stuck on a PR?** Comment on the PR and we'll help

Thank you for contributing!
