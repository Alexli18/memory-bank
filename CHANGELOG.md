# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
