# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-02-23

### Added

- Session capture via Claude Code Stop hook (automatic, no wrappers)
- PTY wrapper fallback (`mb run -- <command>`) for non-Claude CLIs
- Semantic search across captured sessions (`mb search`)
- Context pack generation as XML (`mb pack`) with configurable token budget
- Hook management commands (`mb hooks install/uninstall/status`)
- Session management (`mb sessions`, `mb delete`)
- Per-project isolated storage in `.memory-bank/`
