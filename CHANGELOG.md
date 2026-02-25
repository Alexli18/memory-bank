# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Artifact import**: `mb import` now discovers and imports Claude Code plans (`~/.claude/plans/`), todo lists (`~/.claude/todos/`), and task trees (`~/.claude/tasks/`) alongside conversation sessions
- **Search type filter**: `mb search --type plan|todo|task|session` filters results by source type; results display `[plan]`, `[todo]`, `[task]`, `[session]` labels
- **Artifact sections in context packs**: `mb pack` includes ACTIVE_TASKS (pending/in-progress items) and PLANS sections populated from imported artifacts, with 15% budget allocation each
- **Artifact summary in sessions**: `mb sessions` shows an artifact count summary line when artifacts are present
- `mb graph` command: display session graph with episode classification, error status, and related sessions (table and `--json` output)
- `mb pack --retriever episode --episode <type>`: episode-aware chunk retrieval for context packs
- `mb pack --format json|md`: JSON and Markdown output formats for context packs
- `mb migrate` command: detect and apply storage schema migrations (v1→v2)
- `mb reindex` command: rebuild embedding index from all chunks
- Content-based episode classification for hook/import sessions: `classify_episode` analyzes chunk text when the command is `claude`, expanding beyond command-only heuristics
- Session graph analysis: episode classification (BUILD/TEST/DEPLOY/DEBUG/REFACTOR/EXPLORE/CONFIG/DOCS/REVIEW), error detection, related session linking
- Contextual retriever: episode-aware and failure-aware chunk retrieval
- Secret redaction on ingestion (AWS keys, JWT, Stripe keys, API keys, passwords)
- Schema versioning and migration support
- Unified ingestion pipeline (Source/Processor plugin system)
- Type-safe data models (frozen dataclasses) for all domain entities
- Unified storage abstraction (NdjsonStorage) replacing scattered filesystem access

### Fixed

- Claude adapter chunks now carry real timestamps from turn data instead of hardcoded `0.0`
- State invalidation: `mb pack` regenerates `state.json` when new sessions appear
- `chunk_all_sessions` now re-chunks hook-created sessions that lack `events.jsonl`

### Improved

- Search uses memory-mapped vectors and lazy metadata loading for lower memory usage
- State generation samples chunks by quality instead of head-truncating
- Pack excerpt collection uses a bounded heap instead of loading all chunks

## [0.1.0] - 2026-02-23

### Added

- Session capture via Claude Code Stop hook (automatic, no wrappers)
- PTY wrapper fallback (`mb run -- <command>`) for non-Claude CLIs
- Semantic search across captured sessions (`mb search`)
- Context pack generation as XML (`mb pack`) with configurable token budget
- Hook management commands (`mb hooks install/uninstall/status`)
- Session management (`mb sessions`, `mb delete`)
- Per-project isolated storage in `.memory-bank/`
