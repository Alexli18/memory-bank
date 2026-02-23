# Contributing to Memory Bank

Thanks for your interest in contributing!

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) (only needed for integration tests)

## Development Setup

```bash
# Clone the repo
git clone https://github.com/Alexli18/memory-bank.git
cd memory-bank

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify
mb --version
```

## Running Tests and Linting

```bash
# Run all tests
pytest

# Run a specific test file
pytest tests/unit/test_hooks.py -v

# Lint
ruff check .
```

## PR Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b my-feature`)
3. Make your changes
4. Ensure tests pass (`pytest`) and linting is clean (`ruff check .`)
5. Commit your changes
6. Open a pull request against `main`

## Code Style

- We use [ruff](https://docs.astral.sh/ruff/) for linting
- Python 3.10+ conventions (use `|` for union types, etc.)
- Keep it simple — avoid unnecessary abstractions
